"""Async metrics polling for the swarm web sidecar."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fleet.metrics import ThroughputMeter, scrape

from .graph import GraphState


class MetricsPoller:
    """Polls vLLM metrics and broadcasts protocol metrics messages."""

    def __init__(self, *, url: str, graph: GraphState,
                 broadcast: Callable[[dict[str, Any]], Awaitable[None]] | None = None
                 ) -> None:
        self.url = url
        self.graph = graph
        self._broadcast = broadcast
        self._meter = ThroughputMeter()
        self.latest: dict[str, Any] | None = None

    def set_broadcast(self, broadcast: Callable[[dict[str, Any]], Awaitable[None]]
                      ) -> None:
        """Attach the hub broadcast coroutine."""
        self._broadcast = broadcast

    async def run(self) -> None:
        """Poll forever until cancelled."""
        try:
            while True:
                sc = await asyncio.to_thread(scrape, self.url, 0.3)
                tok = self._meter.update(sc)
                if sc is None:
                    msg = {"type": "metrics", "running": None, "waiting": None,
                           "kv": None, "tok_s": None, "gate": self.graph.latest_gate}
                else:
                    msg = {"type": "metrics", "running": int(sc["running"]),
                           "waiting": int(sc["waiting"]),
                           "kv": round(float(sc["kv"]) * 100, 1),
                           "tok_s": int(tok or 0),
                           "gate": self.graph.latest_gate}
                self.latest = {k: v for k, v in msg.items() if k != "type"}
                if self._broadcast is not None:
                    await self._broadcast(msg)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    def snapshot(self) -> dict[str, Any] | None:
        """Return the latest metrics payload for snapshots."""
        return self.latest
