"""Skill usage telemetry + lifecycle state — VENDORED from HermesAgent tools/skill_usage.py,
decoupled. Sidecar JSON (<skills_dir>/.usage.json) keyed by skill name; counters bumped by
the manager/synth; the curator reads the derived activity timestamp for lifecycle transitions.

Decoupling vs the original:
  - paths come from swarm_agent.skills._env (swarm-private dir), not hermes_constants.
  - the swarm store has no bundled/hub provenance, so is_agent_created() is always True;
    curator eligibility is still gated on the explicit ``created_by == "agent"`` marker.

Lifecycle: active -> stale (unused > stale_after_days) -> archived (> archive_after_days,
moved to .archive/); ``pinned`` opts out of auto-transitions.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _env
from .format import is_excluded_skill_path, read_skill_name

logger = logging.getLogger(__name__)

msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}


def _skills_dir() -> Path:
    return _env.skills_dir()


def _usage_file() -> Path:
    return _env.usage_file()


def _archive_dir() -> Path:
    return _env.archive_dir()


@contextmanager
def _usage_file_lock():
    lock_path = _usage_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        yield
        return
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")
    fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        elif msvcrt:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        fd.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def latest_activity_at(record: Dict[str, Any]) -> Optional[str]:
    latest_dt: Optional[datetime] = None
    latest_raw: Optional[str] = None
    for key in ("last_used_at", "last_viewed_at", "last_patched_at"):
        raw = record.get(key)
        dt = _parse_iso_timestamp(raw)
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt, latest_raw = dt, str(raw)
    return latest_raw


def activity_count(record: Dict[str, Any]) -> int:
    total = 0
    for key in ("use_count", "view_count", "patch_count"):
        try:
            total += int(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


# ── provenance: the swarm store is all ours (no bundled/hub) ─────────────────
def is_agent_created(skill_name: str) -> bool:  # noqa: ARG001 - kept for API parity
    return True


def _is_curator_managed_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return record.get("created_by") == "agent" or record.get("agent_created") is True


def list_agent_created_skill_names() -> List[str]:
    """Skills explicitly marked created_by=='agent' — the curator's eligible set."""
    base = _skills_dir()
    if not base.exists():
        return []
    usage = load_usage()
    names: List[str] = []
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        name = read_skill_name(skill_md, fallback=skill_md.parent.name)
        if _is_curator_managed_record(usage.get(name)):
            names.append(name)
    return sorted(set(names))


def list_archived_skill_names() -> List[str]:
    root = _archive_dir()
    if not root.exists():
        return []
    return sorted({p.name for p in root.iterdir() if p.is_dir()})


# ── sidecar I/O ───────────────────────────────────────────────────────────────
def _empty_record() -> Dict[str, Any]:
    return {
        "created_by": None, "use_count": 0, "view_count": 0,
        "last_used_at": None, "last_viewed_at": None, "patch_count": 0,
        "last_patched_at": None, "created_at": _now_iso(),
        "state": STATE_ACTIVE, "pinned": False, "archived_at": None,
    }


def load_usage() -> Dict[str, Dict[str, Any]]:
    path = _usage_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_usage(data: Dict[str, Dict[str, Any]]) -> None:
    path = _usage_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".usage_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)


def get_record(skill_name: str) -> Dict[str, Any]:
    data = load_usage()
    rec = data.get(skill_name)
    if not isinstance(rec, dict):
        return _empty_record()
    for k, v in _empty_record().items():
        rec.setdefault(k, v)
    return rec


def _mutate(skill_name: str, mutator) -> None:
    if not skill_name:
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            rec = data.get(skill_name)
            if not isinstance(rec, dict):
                rec = _empty_record()
            mutator(rec)
            data[skill_name] = rec
            save_usage(data)
    except Exception as e:
        logger.debug("usage._mutate(%s) failed: %s", skill_name, e, exc_info=True)


def bump_view(skill_name: str) -> None:
    def _a(r):
        r["view_count"] = int(r.get("view_count") or 0) + 1
        r["last_viewed_at"] = _now_iso()
    _mutate(skill_name, _a)


def bump_use(skill_name: str) -> None:
    def _a(r):
        r["use_count"] = int(r.get("use_count") or 0) + 1
        r["last_used_at"] = _now_iso()
    _mutate(skill_name, _a)


def bump_patch(skill_name: str) -> None:
    def _a(r):
        r["patch_count"] = int(r.get("patch_count") or 0) + 1
        r["last_patched_at"] = _now_iso()
    _mutate(skill_name, _a)


def mark_agent_created(skill_name: str) -> None:
    _mutate(skill_name, lambda r: r.__setitem__("created_by", "agent"))


def set_state(skill_name: str, state: str) -> None:
    if state not in _VALID_STATES:
        return
    def _a(r):
        r["state"] = state
        if state == STATE_ARCHIVED:
            r["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            r["archived_at"] = None
    _mutate(skill_name, _a)


def set_pinned(skill_name: str, pinned: bool) -> None:
    _mutate(skill_name, lambda r: r.__setitem__("pinned", bool(pinned)))


def forget(skill_name: str) -> None:
    if not skill_name:
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            if skill_name in data:
                del data[skill_name]
                save_usage(data)
    except Exception as e:
        logger.debug("usage.forget(%s) failed: %s", skill_name, e, exc_info=True)


# ── archive / restore ─────────────────────────────────────────────────────────
def _find_skill_dir(skill_name: str) -> Optional[Path]:
    base = _skills_dir()
    if not base.exists():
        return None
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        if read_skill_name(skill_md, fallback=skill_md.parent.name) == skill_name:
            return skill_md.parent
    return None


def archive_skill(skill_name: str) -> Tuple[bool, str]:
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    root = _archive_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create archive dir: {e}"
    dest = root / skill_dir.name
    if dest.exists():
        dest = root / f"{skill_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    try:
        skill_dir.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(skill_dir), str(dest))
        except Exception as e2:
            return False, f"failed to archive: {e2}"
    set_state(skill_name, STATE_ARCHIVED)
    return True, f"archived to {dest}"


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    root = _archive_dir()
    if not root.exists():
        return False, "no archive directory"
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name == skill_name]
    if not candidates:
        candidates = sorted(
            [p for p in root.rglob("*") if p.is_dir() and p.name.startswith(f"{skill_name}-")],
            reverse=True)
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"
    src = candidates[0]
    dest = _skills_dir() / skill_name
    if dest.exists():
        return False, f"destination already exists: {dest}"
    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to restore: {e}"
    set_state(skill_name, STATE_ACTIVE)
    return True, f"restored to {dest}"


def agent_created_report() -> List[Dict[str, Any]]:
    data = load_usage()
    rows: List[Dict[str, Any]] = []
    for name in list_agent_created_skill_names():
        rec = data.get(name)
        if not isinstance(rec, dict):
            rec = _empty_record()
        for k, v in _empty_record().items():
            rec.setdefault(k, v)
        row = {"name": name, **rec}
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return rows
