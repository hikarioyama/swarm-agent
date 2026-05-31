"""Offline tests for the in-process swarm-agent front door (TUI + dashboard + runner).

No curses, no network: these exercise the pure logic of the conversational front door —
slash-command parsing, CJK-aware cursor width, chat-log wrapping, the live swarm view's
state machine (incl. deadlock/stranded + active indicator), and the runner's router
parsing, bounded history, and the busy admission guard.
"""

from __future__ import annotations

from swarm_agent.tui import parse_command, _disp_width, ChatLog
from swarm_agent.dashboard import SwarmView, _GLYPH
from swarm_agent.runner import SwarmRunner, _parse_route, _HISTORY_CAP


# ── command parsing ───────────────────────────────────────────────────────────
def test_parse_command_plain_and_slash() -> None:
    assert parse_command("hello there") == ("message", "hello there")
    assert parse_command("/swarm audit the repo") == ("swarm", "audit the repo")
    assert parse_command("/help") == ("help", "")
    assert parse_command("  /Gate 32 ") == ("gate", "32")


# ── CJK-aware display width (T3) ──────────────────────────────────────────────
def test_disp_width_counts_wide_glyphs_as_two() -> None:
    assert _disp_width("abc") == 3
    assert _disp_width("あい") == 4            # 2 wide glyphs → 4 cells
    assert _disp_width("aテスト") == 1 + 6      # ascii + 3 wide


# ── chat log wrapping ─────────────────────────────────────────────────────────
def test_chatlog_wraps_and_keeps_blocks() -> None:
    log = ChatLog()
    log.add("user", "hello")
    log.add("assistant", "a " * 80)
    assert len(log.blocks) == 2
    rows = []
    log.render(lambda y, x, t, a=0: rows.append(t), (0, 0, 40, 30),
               {k: 0 for k in ("gold", "fg", "dim", "err")})
    assert any("hello" in r for r in rows)
    log.clear()
    assert log.blocks == []


# ── swarm view state machine + active indicator ───────────────────────────────
def _seed(view: SwarmView) -> None:
    view.ingest({"kind": "planning"})
    view.ingest({"kind": "planned", "tasks": [
        {"id": "a", "lane": "writer", "deps": []},
        {"id": "b", "lane": "coder", "deps": []},
        {"id": "r", "lane": "reducer", "deps": ["a", "b"]}]})


def test_swarmview_dispatch_done_and_active_circle() -> None:
    v = SwarmView()
    _seed(v)
    v.ingest({"kind": "task", "event": "dispatch", "id": "a", "counts": {"running": 1}})
    v.ingest({"kind": "task", "event": "done", "id": "b", "wall_s": 5.0,
              "counts": {"running": 1, "done": 1}})
    assert v.tasks["a"]["state"] == "running"        # active → green ●
    assert v.tasks["b"]["state"] == "done"           # inactive → white ○
    assert v.tasks["r"]["state"] == "pending"        # inactive → white ○
    active = [tid for tid, t in v.tasks.items() if t["state"] == "running"]
    assert active == ["a"]


def test_swarmview_deadlock_marks_stranded() -> None:
    v = SwarmView()
    _seed(v)
    v.ingest({"kind": "task", "event": "dispatch", "id": "a", "counts": {}})
    v.ingest({"kind": "task", "event": "done", "id": "b", "counts": {}})
    v.ingest({"kind": "task", "event": "deadlock", "id": None, "counts": {}, "stranded": 2})
    assert v.stranded == 2
    assert v.tasks["a"]["state"] == "stranded"       # was running → stranded
    assert v.tasks["r"]["state"] == "stranded"       # was pending → stranded
    assert v.tasks["b"]["state"] == "done"           # terminal stays done
    assert "stranded" in _GLYPH


def test_swarmview_render_short_pane_no_crash() -> None:
    v = SwarmView()
    _seed(v)
    pal = {k: 0 for k in ("gold", "ok", "err", "run", "dim")}
    for bottom in (4, 6, 9, 30):                      # incl. degenerate short panes (D1/D3)
        v.render(lambda y, x, t, a=0: None, (2, 0, bottom, 40),
                 palette=pal, gate_limit=40, running=1, kv_pct="10%", tok_s=100.0)


# ── runner: router parsing, bounded history, busy admission guard ─────────────
def test_parse_route_tolerates_prose_and_rejects_nondict() -> None:
    assert _parse_route('{"mode":"swarm"}') == {"mode": "swarm"}
    assert _parse_route('sure: {"mode":"chat","reply":"hi"}')["reply"] == "hi"
    assert _parse_route("[1,2,3]") is None           # non-dict JSON → None (R4 fallback)
    assert _parse_route("not json at all") is None


def test_history_is_bounded() -> None:
    r = SwarmRunner()
    for _ in range(50):
        r._append_history("user", "m")
    assert len(r.history) == _HISTORY_CAP


def test_busy_guard_rejects_reentry() -> None:
    r = SwarmRunner()
    r.busy = True                                    # simulate an in-flight turn
    assert r.submit("second message") is None        # R1: rejected, no thread spawned
