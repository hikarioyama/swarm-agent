"""Outbound rendering: mirror ALL session events to Telegram without firehose spam (§C.4).

Conversational events (user / reply / final / error / manager / btw / status) are sent as
discrete chat messages. Dashboard-style events (task done/fail bursts, counts, gate, plan
progress) are folded into ONE live-updating status message (edited in place) — same information
the TUI shows, legible on a phone. Nothing is filtered; this is a rendering choice only.

``LogTailer`` reads the logbook JSONL by byte offset so the bridge mirrors exactly what the TUI
sees (the log is the durable, multi-subscriber stream — §C.1) and an offset resume never
re-sends events it already mirrored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional


class Renderer:
    """Map session events to transport ops via injected ``send``/``edit`` callables.

    ``send(text) -> message_id`` and ``edit(message_id, text) -> bool`` keep this decoupled from
    any concrete transport (real Bot API or a test fake)."""

    CONVERSATIONAL = {"user", "reply", "final", "error", "manager", "btw", "status"}
    DASHBOARD = {"task", "planned", "planning", "ready", "queued", "route", "boot", "idle"}

    def __init__(self, send: Callable[[str], Any], edit: Callable[[Any, str], Any]) -> None:
        self._send = send
        self._edit = edit
        self._status_id: Optional[Any] = None

    def feed(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind in self.CONVERSATIONAL:
            text = self._conversational(ev)
            if text:
                self._send(text)
        elif kind in self.DASHBOARD:
            text = self._dashboard(ev)
            if not text:
                return
            if self._status_id is None:
                self._status_id = self._send(text)   # create the single live status message
            else:
                self._edit(self._status_id, text)     # update it in place (no new message)
        # unknown kinds (session_start/end, route internals) are intentionally not mirrored

    # ── formatting ──
    @staticmethod
    def _one_line(value, limit: int = 320) -> str:
        s = " ".join(str(value or "").split())
        return s if len(s) <= limit else s[:limit] + "…"

    def _conversational(self, ev: dict) -> str:
        kind = ev.get("kind")
        label = {"user": "🧑", "reply": "💬", "final": "✅", "error": "⚠️",
                 "manager": "🛠", "btw": "💡", "status": "·"}.get(kind, kind)
        for key in ("text", "error", "note", "goal"):
            if ev.get(key):
                return f"{label} {self._one_line(ev[key])}"
        return ""

    def _dashboard(self, ev: dict) -> str:
        kind = ev.get("kind")
        counts = ev.get("counts") or {}
        gid = ev.get("goal_id") or ""
        parts = [f"swarm {kind}"]
        if gid:
            parts.append(f"goal={gid}")
        if ev.get("event"):
            parts.append(str(ev["event"]))
        if counts:
            parts.append(" ".join(f"{k}={v}" for k, v in counts.items()))
        if ev.get("gate") is not None:
            parts.append(f"gate={ev['gate']}")
        if ev.get("n") is not None:
            parts.append(f"tasks={ev['n']}")
        return self._one_line(" · ".join(parts))


class LogTailer:
    """Incrementally read new JSON events from a logbook file, tracking a byte offset so a
    resume from a saved offset never double-reads (§4.4 offset-resume)."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._offset = 0
        self._buf = ""

    @property
    def offset(self) -> int:
        return self._offset

    def seek_end(self) -> None:
        """Skip whatever already exists so the bridge mirrors only events from now on (it does
        not replay the whole session when it starts mid-flight)."""
        try:
            self._offset = self.path.stat().st_size
        except OSError:
            self._offset = 0
        self._buf = ""

    def read_new(self) -> list[dict]:
        out: list[dict] = []
        try:
            size = self.path.stat().st_size
        except OSError:
            return out
        if size < self._offset:                 # rotated/truncated → restart
            self._offset = 0
            self._buf = ""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                data = fh.read()
                self._offset = fh.tell()
        except OSError:
            return out
        if not data:
            return out
        text = self._buf + data.decode("utf-8", "replace")
        lines = text.splitlines(keepends=True)
        self._buf = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._buf = lines.pop()             # hold a partial trailing line for next read
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(ev, dict):
                out.append(ev)
        return out
