"""Persistent structured event log for swarm-agent.

EVERY event the runner publishes (planning / planned / task dispatch-done-fail / final /
error+traceback / manager / queued / idle / btw / route / session lifecycle) is appended as
one timestamped JSON line. All swarm state already flows through SwarmRunner's event publish
path, so teeing it here gives a complete, greppable post-hoc record of "what happened and
where it broke" across a long session — without touching the clean event/UI contract.

Layout (under ``SWARM_LOG_DIR`` or ~/.cache/swarm-agent/logs/):
  events-YYYYMMDD-HHMMSS-<sid>.jsonl   one file per SwarmRunner/session
  latest.jsonl -> the current session file (symlink; best-effort)
Each line: {"ts": iso8601, "sid": <8hex>, "seq": N, "kind": ..., ...event fields...}.
Disable with SWARM_EVENT_LOG=0. Never raises into the runtime.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DIR = Path.home() / ".cache" / "swarm-agent" / "logs"
# Soft per-field cap so one huge deliverable can't make a line unwieldy. error/detail/
# traceback fields are NEVER capped (they are the whole point of the log). 0 = no cap.
_FIELD_CAP = int(os.environ.get("SWARM_LOG_FIELD_CAP", "12000") or "0")
_NEVER_CAP = {"error", "detail", "traceback"}


def _enabled() -> bool:
    return os.environ.get("SWARM_EVENT_LOG", "1") not in ("0", "false", "False")


class SwarmLogger:
    def __init__(self, *, path: str | None = None) -> None:
        self.enabled = _enabled()
        self.sid = uuid.uuid4().hex[:8]
        self._lock = threading.Lock()
        self._seq = 0
        self._fh = None
        if path:
            self.path = Path(path)
        else:
            base = Path(os.environ.get("SWARM_LOG_DIR") or _DEFAULT_DIR)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = base / f"events-{stamp}-{self.sid}.jsonl"

    # ── internal ──
    def _open(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)   # line-buffered
            # best-effort 'latest' pointer for easy tailing; never fatal.
            try:
                link = self.path.with_name("latest.jsonl")
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(self.path.name)
            except Exception:
                pass
        return self._fh

    def _cap(self, key, val):
        if _FIELD_CAP and key not in _NEVER_CAP and isinstance(val, str) and len(val) > _FIELD_CAP:
            return val[:_FIELD_CAP] + f"…[+{len(val) - _FIELD_CAP} chars]"
        return val

    # ── public ──
    def log(self, ev: dict) -> None:
        """Append one event dict as a JSON line. Never raises."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._seq += 1
                rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                       "sid": self.sid, "seq": self._seq}
                for k, v in ev.items():
                    if k == "tasks" and isinstance(v, list):
                        rec[k] = v            # task-plan summaries are already small
                    else:
                        rec[k] = self._cap(k, v)
                self._open().write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass   # logging must NEVER break the runtime

    def event(self, kind: str, **kw) -> None:
        self.log({"kind": kind, **kw})

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
