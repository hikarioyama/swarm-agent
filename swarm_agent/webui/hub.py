"""WebSocket hub for the swarm web sidecar."""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import WebSocket


class Hub:
    """Tracks websocket clients and routes client control messages."""

    def __init__(self, *, snapshot_fn: Callable[[], dict[str, Any]],
                 controller: Callable[[dict[str, Any]], Awaitable[None]],
                 sessions_fn: Callable[[], list[dict[str, Any]]]) -> None:
        self._clients: set[WebSocket] = set()
        self._snapshot_fn = snapshot_fn
        self._controller = controller
        self._sessions_fn = sessions_fn

    async def register(self, ws: WebSocket) -> None:
        """Accept and register a websocket, then send the current snapshot."""
        await ws.accept()
        self._clients.add(ws)
        await ws.send_text(json.dumps(self._snapshot_fn()))

    def unregister(self, ws: WebSocket) -> None:
        """Remove a websocket from the active connection set."""
        self._clients.discard(ws)

    async def broadcast(self, msg: dict[str, Any]) -> None:
        """Send a JSON message to all active clients, dropping dead sockets."""
        if not self._clients:
            return
        raw = json.dumps(msg)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    async def handle_client_msg(self, ws: WebSocket, msg: str) -> None:
        """Decode and route one client websocket message."""
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        if data.get("type") == "sessions":
            await ws.send_text(json.dumps(
                {"type": "sessions", "sessions": self._sessions_fn()}))
            return
        if data.get("type") in {"mode", "replay"}:
            await self._controller(data)
