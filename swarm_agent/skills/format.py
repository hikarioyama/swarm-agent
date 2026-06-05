"""Skill format + discovery — VENDORED from HermesAgent agent/skill_utils.py, trimmed and
decoupled. Dropped the config.yaml-coupled parts (disabled/external_dirs/config-vars,
termux gating) since the swarm store is a single private directory. No HermesAgent imports.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

EXCLUDED_SKILL_DIRS = frozenset((
    ".git", ".github", ".hub", ".archive", ".venv", "venv", "node_modules",
    "site-packages", "__pycache__", ".tox", ".nox", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
))

PLATFORM_MAP = {"macos": "darwin", "linux": "linux", "windows": "win32"}


def is_excluded_skill_path(path) -> bool:
    """True if any path component is an excluded (VCS/venv/cache/archive) dir."""
    try:
        parts = path.parts
    except AttributeError:
        from pathlib import PurePath
        parts = PurePath(str(path)).parts
    return any(part in EXCLUDED_SKILL_DIRS for part in parts)


_yaml_load_fn = None


def yaml_load(content: str):
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml
        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
        _yaml_load_fn = lambda v: yaml.load(v, Loader=loader)  # noqa: E731
    return _yaml_load_fn(content)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown string → (frontmatter_dict, body)."""
    frontmatter: Dict[str, Any] = {}
    body = content
    if not content.startswith("---"):
        return frontmatter, body
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body
    yaml_content = content[3: end_match.start() + 3]
    body = content[end_match.end() + 3:]
    try:
        parsed = yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """True when the skill's `platforms:` list includes the current OS (empty = all)."""
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    for platform in platforms:
        mapped = PLATFORM_MAP.get(str(platform).lower().strip(), str(platform).lower().strip())
        if current.startswith(mapped):
            return True
    return False


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    raw = frontmatter.get("description", "")
    if not raw:
        return ""
    desc = str(raw).strip().strip("'\"")
    return desc[:57] + "..." if len(desc) > 60 else desc


def read_skill_name(skill_md: Path, fallback: str) -> str:
    """Parse the `name:` field from SKILL.md frontmatter (cheap, no full YAML)."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_fm = False
    for line in text.split("\n"):
        s = line.strip()
        if s == "---":
            if in_fm:
                break
            in_fm = True
            continue
        if in_fm and s.startswith("name:"):
            v = s.split(":", 1)[1].strip().strip("\"'")
            if v:
                return v
    return fallback


def iter_skill_index_files(skills_dir: Path, filename: str = "SKILL.md"):
    """Walk skills_dir yielding sorted SKILL.md paths, excluding VCS/venv/cache dirs."""
    if not skills_dir.is_dir():
        return
    matches = []
    for root, dirs, files in os.walk(skills_dir, followlinks=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_SKILL_DIRS]
        if filename in files:
            matches.append(Path(root) / filename)
    for path in sorted(matches, key=lambda p: str(p.relative_to(skills_dir))):
        yield path


def list_skills(skills_dir: Path) -> list[dict]:
    """Return [{name, description, category, path}] for every SKILL.md under skills_dir."""
    out = []
    for md in iter_skill_index_files(skills_dir):
        try:
            fm, _ = parse_frontmatter(md.read_text(encoding="utf-8"))
        except Exception:
            fm = {}
        if not skill_matches_platform(fm):
            continue
        try:
            rel = md.parent.relative_to(skills_dir)
            category = str(rel.parent) if str(rel.parent) != "." else ""
        except ValueError:
            category = ""
        out.append({
            "name": str(fm.get("name") or md.parent.name),
            "description": str(fm.get("description") or "").strip(),
            "category": category,
            "path": str(md),
        })
    return out
