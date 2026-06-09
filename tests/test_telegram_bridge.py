"""Offline tests for the in-process Telegram session mirror (SWARM_V2 Phase 4 + §6.2)."""
from __future__ import annotations

import ast
import json
import types
from pathlib import Path

import swarm_agent.telegram as tg_pkg
from swarm_agent.runner import SwarmRunner
from swarm_agent.telegram import LogTailer, Renderer, TelegramBridge, route_inbound


class FakeTransport:
    """Records sends/edits and replays queued inbound updates (the Bot-API contract)."""

    def __init__(self) -> None:
        self.sent: list = []
        self.edited: list = []
        self._inbox: list = []
        self._mid = 0

    def queue(self, chat_id, text, update_id=None) -> None:
        self._inbox.append({"update_id": update_id if update_id is not None else len(self._inbox) + 1,
                            "chat_id": chat_id, "text": text})

    def send_message(self, chat_id, text):
        self._mid += 1
        self.sent.append((chat_id, text))
        return self._mid

    def edit_message(self, chat_id, message_id, text):
        self.edited.append((chat_id, message_id, text))
        return True

    def get_updates(self, offset, timeout=0):
        out = [u for u in self._inbox if int(u.get("update_id") or 0) >= offset]
        self._inbox = []
        return out


class SpyRunner:
    """Minimal runner stand-in that records which methods inbound routing calls."""

    def __init__(self, submit_busy: bool = False) -> None:
        self.calls: list = []
        self._submit_busy = submit_busy
        self.tasks = types.SimpleNamespace(counts=lambda: {"pending": 1, "running": 0})

    def enqueue_task(self, goal):
        self.calls.append(("enqueue_task", goal))
        return {"id": "task-1", "goal": goal}

    def interrupt(self, *a):
        self.calls.append(("interrupt",))
        return 2

    def submit(self, text):
        self.calls.append(("submit", text))
        return None if self._submit_busy else "thread"

    def steer(self, text):
        self.calls.append(("steer", text))
        return 1


# ── 4.2 inbound = identical to TUI typing ────────────────────────────────────

def test_inbound_routes_like_tui_typing() -> None:
    spy = SpyRunner()
    assert route_inbound(spy, "/task build the X endpoint")
    assert ("enqueue_task", "build the X endpoint") in spy.calls
    assert route_inbound(spy, "/stop")
    assert ("interrupt",) in spy.calls
    assert route_inbound(spy, "hello there")
    assert ("submit", "hello there") in spy.calls
    assert "queue" in route_inbound(spy, "/tasks")
    assert route_inbound(spy, "/task") == "usage: /task <goal>"


# ── 4.3 busy contention → steer, not drop ────────────────────────────────────

def test_inbound_busy_steers_instead_of_dropping() -> None:
    spy = SpyRunner(submit_busy=True)
    reply = route_inbound(spy, "tweak the running plan")
    assert ("submit", "tweak the running plan") in spy.calls
    assert ("steer", "tweak the running plan") in spy.calls    # steered, not dropped
    assert "still working" in reply


# ── 4.5 chat-id allowlist (security-critical) ────────────────────────────────

def test_allowlist_blocks_foreign_chats() -> None:
    spy = SpyRunner()
    tr = FakeTransport()
    bridge = TelegramBridge(spy, transport=tr, allowed_chat_ids={123})

    bridge.handle_inbound(999, "/task malicious")          # foreign chat → ignored entirely
    assert spy.calls == []
    assert tr.sent == []

    bridge.handle_inbound(123, "/task legit")              # owner → routed + replied
    assert ("enqueue_task", "legit") in spy.calls
    assert tr.sent and tr.sent[0][0] == 123


def test_poll_inbound_routes_queued_updates() -> None:
    spy = SpyRunner()
    tr = FakeTransport()
    tr.queue(123, "/task do the thing")
    tr.queue(999, "/task foreign")                         # not allowed → must be ignored
    bridge = TelegramBridge(spy, transport=tr, allowed_chat_ids={123})
    bridge._poll_inbound_once()
    assert spy.calls == [("enqueue_task", "do the thing")]


# ── 4.4 outbound rendering: collapse dashboard, discrete conversational ───────

def test_renderer_collapses_dashboard_and_sends_conversational() -> None:
    sent: list = []
    edited: list = []

    def send(text):
        sent.append(text)
        return len(sent)

    r = Renderer(send=send, edit=lambda mid, text: edited.append((mid, text)))
    r.feed({"kind": "reply", "text": "hi"})
    r.feed({"kind": "final", "text": "done"})
    r.feed({"kind": "error", "text": "boom"})
    assert len(sent) == 3 and edited == []                 # conversational → discrete sends

    for i in range(5):
        r.feed({"kind": "task", "event": "done", "id": f"t{i}", "counts": {"done": i}})
    assert len(sent) == 4                                   # one NEW status message for the burst
    assert len(edited) == 4                                 # subsequent task events edit it


def test_log_tailer_offset_resume(tmp_path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"kind": "reply", "text": "a"}) + "\n"
                 + json.dumps({"kind": "final", "text": "b"}) + "\n")
    tailer = LogTailer(str(p))
    assert [e["text"] for e in tailer.read_new()] == ["a", "b"]
    with open(p, "a") as fh:
        fh.write(json.dumps({"kind": "error", "text": "c"}) + "\n")
    assert [e["text"] for e in tailer.read_new()] == ["c"]  # only the new event
    assert tailer.read_new() == []                          # offset-resume → no double-send


# ── 4.1 / 4.6 lifecycle: no-op unconfigured; clean start/stop ────────────────

def test_bridge_is_noop_when_unconfigured() -> None:
    spy = SpyRunner()
    tr = FakeTransport()
    # transport present but NO allowlist → never run an open bot
    b = TelegramBridge(spy, transport=tr, allowed_chat_ids=set())
    assert not b.configured
    assert b.start() is False and b._threads == []
    # allowlist but no transport/token → nothing to talk through
    b2 = TelegramBridge(spy, allowed_chat_ids={1})
    assert not b2.configured
    assert b2.start() is False


def test_bridge_lifecycle_start_and_stop(tmp_path) -> None:
    spy = SpyRunner()
    tr = FakeTransport()
    logp = tmp_path / "latest.jsonl"
    logp.write_text("")
    b = TelegramBridge(spy, transport=tr, allowed_chat_ids={123},
                       log_path=str(logp), poll_interval=0.02)
    assert b.configured
    assert b.start() is True
    assert b._threads and all(t.is_alive() for t in b._threads)
    b.stop()
    assert b._threads == []                                 # joined, no lingering loop


def test_runner_constructs_unconfigured_bridge(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SWARM_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("SWARM_TELEGRAM_CHAT_IDS", raising=False)
    monkeypatch.setenv("SWARM_TASKS_PATH", str(tmp_path / "tasks.json"))
    runner = SwarmRunner(warm=False, admission="static")
    assert runner._telegram is not None and not runner._telegram.configured
    runner.shutdown()
    assert runner._telegram._threads == []


# ── §6.2 the telegram package is self-contained (no hermes / no external lib) ─

def test_telegram_package_is_self_contained() -> None:
    pkgdir = Path(tg_pkg.__file__).parent
    bad: list = []
    for f in sorted(pkgdir.glob("*.py")):
        tree = ast.parse(f.read_text(), filename=f.name)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                # relative imports (level>0) are within the package — fine
                mods = [node.module or ""] if not node.level else []
            for mod in mods:
                if (any(x in mod for x in ("hermes", "gateway", "run_agent"))
                        or mod == "telegram" or mod.startswith("telegram.")):
                    bad.append((f.name, mod))
    assert not bad, f"telegram package must not import {bad}"
