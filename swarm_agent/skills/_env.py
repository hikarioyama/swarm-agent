"""Swarm-private paths + curator config — replaces HermesAgent's hermes_constants /
hermes_cli.config. The skill store lives in a swarm-only directory so the swarm curator
NEVER touches HermesAgent's ~/.hermes/skills/."""
from __future__ import annotations

import os
from pathlib import Path


def skills_dir() -> Path:
    """Swarm-private skills directory (NOT ~/.hermes/skills)."""
    return Path(os.path.expanduser(
        os.environ.get("SWARM_SKILLS_DIR", "~/.cache/swarm-agent/skills")))


def archive_dir() -> Path:
    return skills_dir() / ".archive"


def usage_file() -> Path:
    return skills_dir() / ".usage.json"


def curator_state_file() -> Path:
    return skills_dir() / ".curator_state"


def curator_log_dir() -> Path:
    return Path(os.path.expanduser(
        os.environ.get("SWARM_SKILLS_LOG_DIR", "~/.cache/swarm-agent/logs/curator")))


# ── curator config (env-tunable; no config.yaml dependency) ──────────────────
def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def curator_enabled() -> bool:
    return os.environ.get("SWARM_CURATOR", "1") not in ("0", "false", "False")


def stale_after_days() -> int:
    return _envi("SWARM_CURATOR_STALE_DAYS", 30)


def archive_after_days() -> int:
    return _envi("SWARM_CURATOR_ARCHIVE_DAYS", 90)


def curator_interval_hours() -> float:
    return _envf("SWARM_CURATOR_INTERVAL_H", 168.0)  # weekly


def synth_enabled() -> bool:
    return os.environ.get("SWARM_SKILL_SYNTH", "1") not in ("0", "false", "False")
