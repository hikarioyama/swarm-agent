"""In-process Telegram mirror of the live swarm session (SWARM_V2 §C).

Not a separate bot session — a SECOND front-end onto the one live ``SwarmRunner``, mirroring it
both ways: inbound messages go through the SAME path as TUI typing, and every event the TUI
sees is mirrored outbound by tailing the logbook JSONL. It lives and dies with the runner (the
bridge holds the runner ref; it is started by ``start_manager`` and stopped by ``shutdown``), so
"as long as the TUI session is alive, so is Telegram" holds automatically.

Soft-degrade: with no token or no chat-id allowlist, the bridge is a complete no-op (same as a
down inference server). Security: a strict chat-id allowlist — the bot is publicly addressable,
so every chat id except the owner's is ignored (§C.5)."""
from __future__ import annotations

import os
import threading
from typing import Optional

from .inbound import route_inbound
from .render import LogTailer, Renderer


def _parse_ids(raw: Optional[str]) -> set:
    out: set = set()
    for tok in (raw or "").replace(" ", "").split(","):
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            out.add(tok)
    return out


class TelegramBridge:
    def __init__(self, runner, *, transport=None, token: Optional[str] = None,
                 allowed_chat_ids=None, log_path: Optional[str] = None,
                 poll_interval: float = 1.0) -> None:
        self.runner = runner
        self.token = token if token is not None else os.environ.get("SWARM_TELEGRAM_TOKEN")
        self.allowed = (set(allowed_chat_ids) if allowed_chat_ids is not None
                        else _parse_ids(os.environ.get("SWARM_TELEGRAM_CHAT_IDS")))
        self._transport = transport
        self.log_path = str(log_path) if log_path else None
        self.poll_interval = poll_interval
        self.primary_chat = (sorted(self.allowed, key=str)[0] if self.allowed else None)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._inbound_offset = 0
        self._renderer: Optional[Renderer] = None
        self._tailer: Optional[LogTailer] = None

    @property
    def configured(self) -> bool:
        """No-op unless we have a way to talk (an injected transport or a token) AND an
        allowlist (never run an open bot)."""
        return bool((self._transport is not None or self.token) and self.allowed)

    def _ensure_transport(self):
        if self._transport is None and self.token:
            from .transport import HttpTelegramTransport
            self._transport = HttpTelegramTransport(self.token)
        return self._transport

    # ── lifecycle ──
    def start(self) -> bool:
        if not self.configured or self._threads:
            return False
        tr = self._ensure_transport()
        if tr is None:
            return False
        self._renderer = Renderer(
            send=lambda text: tr.send_message(self.primary_chat, text),
            edit=lambda mid, text: tr.edit_message(self.primary_chat, mid, text))
        if self.log_path:
            self._tailer = LogTailer(self.log_path)
            self._tailer.seek_end()          # mirror from now on, don't replay the whole session
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._inbound_loop, name="tg-inbound", daemon=True),
            threading.Thread(target=self._outbound_loop, name="tg-outbound", daemon=True),
        ]
        for t in self._threads:
            t.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []

    # ── loops (each step is also directly callable for tests) ──
    def _inbound_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_inbound_once()
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def _outbound_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain_outbound_once()
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def _poll_inbound_once(self) -> None:
        tr = self._transport
        if tr is None:
            return
        for u in (tr.get_updates(self._inbound_offset, timeout=0) or []):
            uid = u.get("update_id")
            if uid is not None:
                self._inbound_offset = max(self._inbound_offset, int(uid) + 1)
            self.handle_inbound(u.get("chat_id"), u.get("text") or "")

    def handle_inbound(self, chat_id, text: str) -> None:
        # §4.5 SECURITY: strict allowlist. Ignore every chat id but the owner's — no routing,
        # no reply (the bot is publicly addressable).
        if chat_id not in self.allowed:
            return
        reply = route_inbound(self.runner, text)
        if reply and self._transport is not None:
            self._transport.send_message(self.primary_chat, reply)

    def _drain_outbound_once(self) -> None:
        if self._tailer is None or self._renderer is None:
            return
        for ev in self._tailer.read_new():
            self._renderer.feed(ev)
