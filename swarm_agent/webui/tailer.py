"""Live log tailing and replay control for the swarm web sidecar."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from swarm_agent.cli import _latest_log, _read_events

from .graph import GraphState


class LogSource:
    """Feeds log events into GraphState in live or replay mode."""

    def __init__(self, *, log_dir: Path, graph: GraphState,
                 broadcast: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
                 replay: str | None = None) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.graph = graph
        self._broadcast = broadcast
        self._snapshot_fn: Callable[[], dict[str, Any]] | None = None
        self._initial_replay = replay
        self._controls: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.mode = "live"
        self.session: str | None = None
        self.events: list[dict[str, Any]] = []
        self.pos = 0
        self.playing = False
        self.speed = 1.0
        self._live_path: Path | None = None
        self._live_offset = 0
        self._live_buf = ""

    def set_broadcast(self, broadcast: Callable[[dict[str, Any]], Awaitable[None]]
                      ) -> None:
        """Attach the hub broadcast coroutine."""
        self._broadcast = broadcast

    def set_snapshot_fn(self, snapshot_fn: Callable[[], dict[str, Any]]) -> None:
        """Attach the full protocol snapshot provider."""
        self._snapshot_fn = snapshot_fn

    async def control(self, msg: dict[str, Any]) -> None:
        """Queue a client control message for the source loop."""
        await self._controls.put(msg)

    def replay_status(self) -> dict[str, Any] | None:
        """Return replay status, or None in live mode."""
        if self.mode != "replay":
            return None
        return {"playing": self.playing, "pos": self.pos,
                "total": len(self.events), "speed": self.speed,
                "session": self.session}

    async def run(self) -> None:
        """Run the selected source mode until cancelled."""
        try:
            if self._initial_replay:
                await self._switch_replay(self._initial_replay)
            else:
                await self._switch_live()
            while True:
                if self.mode == "live":
                    await self._live_loop_once()
                else:
                    await self._replay_loop_once()
        except asyncio.CancelledError:
            raise

    async def _live_loop_once(self) -> None:
        try:
            msg = await asyncio.wait_for(self._controls.get(), timeout=0.2)
        except asyncio.TimeoutError:
            await self._poll_live()
            return
        await self._handle_control(msg)

    async def _replay_loop_once(self) -> None:
        if not self.playing or self.pos >= len(self.events):
            msg = await self._controls.get()
            await self._handle_control(msg)
            return
        delay = self._next_delay()
        if delay > 0:
            try:
                msg = await asyncio.wait_for(self._controls.get(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            else:
                await self._handle_control(msg)
                return
        await self._replay_step()

    async def _handle_control(self, msg: dict[str, Any]) -> None:
        typ = msg.get("type")
        if typ == "mode":
            value = msg.get("value")
            if value == "live":
                await self._switch_live()
            elif value == "replay":
                await self._switch_replay(str(msg.get("session") or ""))
            return
        if typ != "replay":
            return
        action = msg.get("action")
        if action == "play":
            self.playing = True
            await self._broadcast_replay()
        elif action == "pause":
            self.playing = False
            await self._broadcast_replay()
        elif action == "seek":
            await self._seek(self._clamp_pos(msg.get("pos")))
        elif action == "speed":
            try:
                self.speed = max(0.01, float(msg.get("value") or self.speed))
            except (TypeError, ValueError):
                pass
            await self._broadcast_replay()

    async def _switch_live(self) -> None:
        self.mode = "live"
        self.session = None
        self.playing = False
        self.graph.reset()
        self._live_path = self._resolve_latest()
        self._live_offset = 0
        self._live_buf = ""
        if self._live_path is not None:
            await self._read_live_from_start(self._live_path)
        await self._broadcast_snapshot()

    async def _switch_replay(self, session: str) -> None:
        self.mode = "replay"
        self.session = Path(session).name if session else self._latest_session_name()
        self.playing = False
        self.pos = 0
        self.events = []
        self.graph.reset()
        if self.session:
            path = self.log_dir / self.session
            self.events = _read_events([path]) if path.exists() else []
        await self._broadcast_snapshot()
        await self._broadcast_replay()

    async def _seek(self, pos: int) -> None:
        self.graph.reset()
        for ev in self.events[:pos]:
            await self._process(ev, emit=False)
        self.pos = pos
        await self._broadcast_snapshot()
        await self._broadcast_replay()

    async def _replay_step(self) -> None:
        if self.pos >= len(self.events):
            self.playing = False
            await self._broadcast_replay()
            return
        ev = self.events[self.pos]
        await self._process(ev, emit=True)
        self.pos += 1
        if self.pos >= len(self.events):
            self.playing = False
        await self._broadcast_replay()

    def _next_delay(self) -> float:
        if self.pos <= 0 or self.pos >= len(self.events):
            return 0.0
        prev = self._event_ts(self.events[self.pos - 1])
        cur = self._event_ts(self.events[self.pos])
        if prev is None or cur is None:
            return 0.0
        return max(0.0, min(2.0, cur - prev)) / self.speed

    async def _poll_live(self) -> None:
        latest = self._resolve_latest()
        if latest != self._live_path:
            self.graph.reset()
            self._live_path = latest
            self._live_offset = 0
            self._live_buf = ""
            if latest is not None:
                await self._read_live_from_start(latest)
            await self._broadcast_snapshot()
            return
        if latest is None:
            return
        try:
            size = latest.stat().st_size
        except OSError:
            return
        if size < self._live_offset:
            self._live_offset = 0
            self._live_buf = ""
        try:
            with latest.open("rb") as fh:
                fh.seek(self._live_offset)
                data = fh.read()
                self._live_offset = fh.tell()
        except OSError:
            return
        if data:
            await self._feed_text(data.decode(errors="replace"), emit=True)

    async def _read_live_from_start(self, path: Path) -> None:
        try:
            with path.open("rb") as fh:
                data = fh.read()
                self._live_offset = fh.tell()
        except OSError:
            return
        await self._feed_text(data.decode(errors="replace"), emit=False)

    async def _feed_text(self, text: str, *, emit: bool) -> None:
        text = self._live_buf + text
        lines = text.splitlines(keepends=True)
        self._live_buf = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._live_buf = lines.pop()
        for line in lines:
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(ev, dict):
                await self._process(ev, emit=emit)

    async def _process(self, ev: dict[str, Any], *, emit: bool) -> None:
        messages = self.graph.apply(ev)
        if emit and self._broadcast is not None:
            for msg in messages:
                await self._broadcast(msg)

    async def _broadcast_snapshot(self) -> None:
        if self._broadcast is not None and self._snapshot_fn is not None:
            await self._broadcast(self._snapshot_fn())

    async def _broadcast_replay(self) -> None:
        if self._broadcast is not None and self.mode == "replay":
            msg = self.replay_status()
            if msg is not None:
                msg = {"type": "replay", **msg}
                await self._broadcast(msg)

    def _resolve_latest(self) -> Path | None:
        latest = self.log_dir / "latest.jsonl"
        try:
            if latest.exists():
                return latest.resolve()
        except OSError:
            return None
        return _latest_log(self.log_dir)

    def _latest_session_name(self) -> str | None:
        latest = self._resolve_latest()
        return latest.name if latest is not None else None

    def _clamp_pos(self, value: Any) -> int:
        try:
            pos = int(value)
        except (TypeError, ValueError):
            pos = self.pos
        return max(0, min(len(self.events), pos))

    def _event_ts(self, ev: dict[str, Any]) -> float | None:
        raw = ev.get("ts")
        if not raw:
            return None
        try:
            text = str(raw)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except (TypeError, ValueError):
            return None
