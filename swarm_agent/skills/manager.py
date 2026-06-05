"""Skill CRUD — create / edit / patch / delete / write_file / remove_file.

Lean, standalone reimplementation of HermesAgent tools/skill_manager_tool.py: same SKILL.md
layout + telemetry hooks, but no skills_guard / fuzzy_match / path_security / hermes_cli
imports. Patch uses difflib for a whitespace-tolerant fallback; writes are atomic.

All ops return a dict {"success": bool, "message"|"error": str, ...}. These are called by the
harness (synth / curator), NOT exposed as model-callable tools — the LLM proposes structured
plans and the harness applies them through here.
"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from . import _env, usage
from .format import parse_frontmatter, read_skill_name, is_excluded_skill_path

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_CONTENT = 256 * 1024              # 256 KiB SKILL.md cap
_ALLOWED_SUBDIRS = ("references", "templates", "scripts", "assets")


def _ok(msg: str, **kw):
    return {"success": True, "message": msg, **kw}


def _err(msg: str, **kw):
    return {"success": False, "error": msg, **kw}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _validate_name(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return "name is required"
    if len(name) > 64:
        return "name must be <= 64 chars"
    if not _NAME_RE.match(name):
        return "name must be lowercase letters/digits/hyphens (e.g. 'deploy-rollback')"
    return None


def _validate_content(content: str) -> Optional[str]:
    if not content or not content.strip():
        return "content is empty"
    if len(content) > _MAX_CONTENT:
        return f"content exceeds {_MAX_CONTENT} bytes"
    fm, body = parse_frontmatter(content)
    if not fm:
        return "SKILL.md must start with YAML frontmatter (--- ... ---)"
    if not str(fm.get("name") or "").strip():
        return "frontmatter is missing required 'name'"
    if not str(fm.get("description") or "").strip():
        return "frontmatter is missing required 'description'"
    if not body.strip():
        return "skill body (markdown after frontmatter) is empty"
    return None


def find_skill_dir(name: str) -> Optional[Path]:
    return usage._find_skill_dir(name)


def _safe_subpath(skill_dir: Path, file_path: str) -> Optional[Path]:
    """Resolve file_path under skill_dir, rejecting traversal/absolute paths."""
    if not file_path or os.path.isabs(file_path) or ".." in Path(file_path).parts:
        return None
    dest = (skill_dir / file_path).resolve()
    try:
        dest.relative_to(skill_dir.resolve())
    except ValueError:
        return None
    return dest


# ── operations ────────────────────────────────────────────────────────────────
def create(name: str, content: str, category: Optional[str] = None) -> dict:
    err = _validate_name(name)
    if err:
        return _err(err)
    err = _validate_content(content)
    if err:
        return _err(err)
    if find_skill_dir(name) is not None:
        return _err(f"skill '{name}' already exists (use edit/patch)")
    base = _env.skills_dir()
    cat = (category or "").strip().strip("/")
    if cat and (os.path.isabs(cat) or ".." in Path(cat).parts):
        return _err("invalid category")
    skill_dir = base / cat / name if cat else base / name
    skill_md = skill_dir / "SKILL.md"
    try:
        _atomic_write(skill_md, content)
    except Exception as e:
        return _err(f"write failed: {e}")
    usage.mark_agent_created(name)   # opt into curator management
    return _ok(f"created skill '{name}'", path=str(skill_md))


def edit(name: str, content: str) -> dict:
    err = _validate_content(content)
    if err:
        return _err(err)
    skill_dir = find_skill_dir(name)
    if skill_dir is None:
        return _err(f"skill '{name}' not found")
    try:
        _atomic_write(skill_dir / "SKILL.md", content)
    except Exception as e:
        return _err(f"write failed: {e}")
    usage.bump_patch(name)
    return _ok(f"edited skill '{name}'", path=str(skill_dir / "SKILL.md"))


def _fuzzy_replace(text: str, old: str, new: str, replace_all: bool) -> tuple[Optional[str], str]:
    """Exact replace first; fall back to a whitespace-normalized contiguous match."""
    n = text.count(old)
    if n == 1 or (n > 1 and replace_all):
        return text.replace(old, new), f"replaced {n if replace_all else 1} occurrence(s)"
    if n > 1 and not replace_all:
        return None, f"old_string matched {n} times; pass replace_all=True or add context"
    # exact miss → difflib: find the closest contiguous block by line
    old_lines = old.splitlines()
    if not old_lines:
        return None, "old_string not found"
    text_lines = text.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, [l.strip() for l in text.splitlines()],
                                 [l.strip() for l in old_lines])
    blocks = [b for b in sm.get_matching_blocks() if b.size == len(old_lines)]
    if not blocks:
        return None, "old_string not found (even with whitespace-tolerant match)"
    b = blocks[0]
    start, end = b.a, b.a + len(old_lines)
    repl = (new if new.endswith("\n") or end >= len(text_lines) else new + "\n")
    patched = "".join(text_lines[:start]) + repl + "".join(text_lines[end:])
    return patched, "replaced 1 occurrence (whitespace-tolerant)"


def patch(name: str, old_string: str, new_string: str, replace_all: bool = False,
          file_path: Optional[str] = None) -> dict:
    skill_dir = find_skill_dir(name)
    if skill_dir is None:
        return _err(f"skill '{name}' not found")
    target = skill_dir / "SKILL.md"
    if file_path:
        sp = _safe_subpath(skill_dir, file_path)
        if sp is None:
            return _err("invalid file_path")
        target = sp
    if not target.exists():
        return _err(f"target file not found: {target.name}")
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as e:
        return _err(f"read failed: {e}")
    patched, msg = _fuzzy_replace(text, old_string, new_string, replace_all)
    if patched is None:
        return _err(msg)
    if target.name == "SKILL.md":
        err = _validate_content(patched)
        if err:
            return _err(f"patch would break SKILL.md: {err}")
    try:
        _atomic_write(target, patched)
    except Exception as e:
        return _err(f"write failed: {e}")
    usage.bump_patch(name)
    return _ok(f"patched {target.name} in '{name}': {msg}", path=str(target))


def write_file(name: str, file_path: str, file_content: str) -> dict:
    skill_dir = find_skill_dir(name)
    if skill_dir is None:
        return _err(f"skill '{name}' not found")
    sp = _safe_subpath(skill_dir, file_path)
    if sp is None:
        return _err("invalid file_path (traversal/absolute not allowed)")
    top = Path(file_path).parts[0] if Path(file_path).parts else ""
    if top not in _ALLOWED_SUBDIRS:
        return _err(f"supporting files go under one of {_ALLOWED_SUBDIRS}")
    try:
        _atomic_write(sp, file_content)
    except Exception as e:
        return _err(f"write failed: {e}")
    usage.bump_patch(name)
    return _ok(f"wrote {file_path} in '{name}'", path=str(sp))


def remove_file(name: str, file_path: str) -> dict:
    skill_dir = find_skill_dir(name)
    if skill_dir is None:
        return _err(f"skill '{name}' not found")
    sp = _safe_subpath(skill_dir, file_path)
    if sp is None or sp.name == "SKILL.md":
        return _err("invalid file_path")
    if not sp.exists():
        return _err(f"file not found: {file_path}")
    try:
        sp.unlink()
        for parent in sp.parents:
            if parent == skill_dir:
                break
            try:
                parent.rmdir()      # clean now-empty subdirs
            except OSError:
                break
    except Exception as e:
        return _err(f"remove failed: {e}")
    return _ok(f"removed {file_path} from '{name}'")


def delete(name: str, absorbed_into: Optional[str] = None) -> dict:
    skill_dir = find_skill_dir(name)
    if skill_dir is None:
        return _err(f"skill '{name}' not found")
    try:
        shutil.rmtree(skill_dir)
    except Exception as e:
        return _err(f"delete failed: {e}")
    usage.forget(name)
    note = f" (absorbed into '{absorbed_into}')" if absorbed_into else ""
    return _ok(f"deleted skill '{name}'{note}")


# ── dispatch (parity with HermesAgent skill_manage) ──────────────────────────
def skill_manage(action: str, name: str = "", **kw) -> dict:
    a = (action or "").strip().lower()
    if a == "create":
        return create(name, kw.get("content", ""), kw.get("category"))
    if a == "edit":
        return edit(name, kw.get("content", ""))
    if a == "patch":
        return patch(name, kw.get("old_string", ""), kw.get("new_string", ""),
                     bool(kw.get("replace_all", False)), kw.get("file_path"))
    if a == "write_file":
        return write_file(name, kw.get("file_path", ""), kw.get("file_content", ""))
    if a == "remove_file":
        return remove_file(name, kw.get("file_path", ""))
    if a == "delete":
        return delete(name, kw.get("absorbed_into"))
    return _err(f"unknown action '{action}'")
