"""FastAPI app factory for the read-only swarm web sidecar."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from swarm_agent.cli import _event_logs

from .graph import GraphState
from .hub import Hub
from .metrics import MetricsPoller
from .tailer import LogSource


def create_app(*, log_dir: Path, metrics_url: str, no_metrics: bool = False,
               replay: str | None = None) -> FastAPI:
    """Create the webui FastAPI application."""
    module_dir = Path(__file__).resolve().parent
    log_dir = Path(log_dir).expanduser()
    graph = GraphState()
    source = LogSource(log_dir=log_dir, graph=graph, replay=replay)
    poller = None if no_metrics else MetricsPoller(url=metrics_url, graph=graph)

    def sessions() -> list[dict[str, Any]]:
        return _sessions_payload(log_dir)

    def metrics_snapshot() -> dict[str, Any] | None:
        if poller is not None:
            return poller.snapshot()
        return {"running": graph.latest_running, "waiting": None, "kv": None,
                "tok_s": None, "gate": graph.latest_gate}

    def snapshot() -> dict[str, Any]:
        return {"type": "snapshot", "mode": source.mode, "session": source.session,
                "graph": graph.snapshot(), "metrics": metrics_snapshot(),
                "replay": source.replay_status()}

    hub = Hub(snapshot_fn=snapshot, controller=source.control, sessions_fn=sessions)
    source.set_broadcast(hub.broadcast)
    source.set_snapshot_fn(snapshot)
    if poller is not None:
        poller.set_broadcast(hub.broadcast)

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=module_dir / "static", check_dir=False),
              name="static")

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(module_dir / "static" / "index.html")

    @app.get("/api/sessions")
    async def api_sessions() -> dict[str, Any]:
        return {"type": "sessions", "sessions": sessions()}

    @app.get("/api/snapshot")
    async def api_snapshot() -> dict[str, Any]:
        return snapshot()

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await hub.register(ws)
        try:
            while True:
                msg = await ws.receive_text()
                await hub.handle_client_msg(ws, msg)
        except WebSocketDisconnect:
            hub.unregister(ws)

    @app.on_event("startup")
    async def startup() -> None:
        tasks = [asyncio.create_task(source.run())]
        if poller is not None:
            tasks.append(asyncio.create_task(poller.run()))
        app.state.webui_tasks = tasks

    @app.on_event("shutdown")
    async def shutdown() -> None:
        tasks = getattr(app.state, "webui_tasks", [])
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return app


def _sessions_payload(log_dir: Path) -> list[dict[str, Any]]:
    latest_target: Path | None = None
    latest = log_dir / "latest.jsonl"
    try:
        if latest.exists():
            latest_target = latest.resolve()
    except OSError:
        latest_target = None
    sessions: list[dict[str, Any]] = []
    for path in _event_logs(log_dir):
        try:
            st = path.stat()
            resolved = path.resolve()
        except OSError:
            continue
        sessions.append({"name": path.name, "mtime": st.st_mtime,
                         "size": st.st_size,
                         "is_latest": latest_target is not None
                         and resolved == latest_target})
    return sessions
