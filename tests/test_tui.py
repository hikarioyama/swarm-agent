"""Offline tests for the in-process swarm-agent front door (TUI + dashboard + runner).

No curses, no network: these exercise the pure logic of the conversational front door —
slash-command parsing, CJK-aware cursor width, chat-log wrapping, the live swarm view's
state machine (incl. deadlock/stranded + active indicator), and the runner's router
parsing, bounded history, and the busy admission guard.
"""

from __future__ import annotations

from fleet import compat, prompts
from swarm_agent.tui import (
    parse_command, _disp_width, _hard_break, _cursor_rowcol, _wrap_input,
    decode_escape, ChatLog,
)
from swarm_agent.dashboard import SwarmView, _GLYPH
from swarm_agent.runner import SwarmRunner, _parse_route, _HISTORY_CAP
from swarm_agent.taskstore import TaskStore


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


def test_composer_hard_break_keeps_ascii_and_cjk_inside_width() -> None:
    for text, width in (("abcdefghijklmnopqrstuvwxyz", 7), ("長い日本語入力欄", 6)):
        lines = _hard_break(text, width)
        assert len(lines) > 1
        assert "".join(lines) == text
        assert all(_disp_width(line) <= width for line in lines)


# ── caret placement under wrapping (arrow-key cursor movement) ────────────────
def test_cursor_rowcol_ascii_single_line() -> None:
    assert _cursor_rowcol("hello", 0, 40) == (0, 0)      # start
    assert _cursor_rowcol("hello", 3, 40) == (0, 3)      # mid
    assert _cursor_rowcol("hello", 5, 40) == (0, 5)      # end


def test_cursor_rowcol_cjk_counts_two_cells() -> None:
    # caret after 2 wide glyphs sits at column 4, not 2.
    assert _cursor_rowcol("あいうえ", 2, 40) == (0, 4)
    assert _cursor_rowcol("aあb", 2, 40) == (0, 3)       # ascii + wide before caret


def test_cursor_rowcol_wraps_to_next_row() -> None:
    # width 4 → "abcd" fills row 0; caret at index 4 lands on a fresh row.
    assert _cursor_rowcol("abcdef", 4, 4) == (1, 0)
    assert _cursor_rowcol("abcdef", 5, 4) == (1, 1)
    # caret at the very end of a full last line spills onto its own row (matches draw()).
    assert _cursor_rowcol("abcd", 4, 4) == (1, 0)


def test_cursor_rowcol_clamps_out_of_range() -> None:
    assert _cursor_rowcol("abc", -3, 40) == (0, 0)
    assert _cursor_rowcol("abc", 99, 40) == (0, 3)


# ── multi-line composer (Shift+Enter / paste newlines) ────────────────────────
def test_wrap_input_splits_on_explicit_newlines() -> None:
    # each logical line wraps independently; a trailing newline yields a fresh row.
    assert _wrap_input("abc\ndef", 40) == ["abc", "def"]
    assert _wrap_input("abc\n", 40) == ["abc", ""]
    assert _wrap_input("", 40) == [""]
    # a long logical line is still hard-broken inside its own segment.
    assert _wrap_input("abcdef\nx", 4) == ["abcd", "ef", "x"]


def test_cursor_rowcol_tracks_explicit_newlines() -> None:
    txt = "ab\ncd"
    assert _cursor_rowcol(txt, 0, 40) == (0, 0)
    assert _cursor_rowcol(txt, 2, 40) == (0, 2)      # caret on the newline → end of row 0
    assert _cursor_rowcol(txt, 3, 40) == (1, 0)      # first glyph of the next line
    assert _cursor_rowcol(txt, 5, 40) == (1, 2)      # end of buffer
    # trailing newline parks the caret on an empty fresh row.
    assert _cursor_rowcol("ab\n", 40 - 0, 40) == (1, 0)


# ── ESC-sequence decoding (paste + kitty/xterm key encodings) ─────────────────
def _feed(rest: str):
    """A read_next() over the bytes that follow the initial ESC."""
    it = iter(rest)
    return lambda: next(it, None)


def test_decode_escape_shift_enter_inserts_newline() -> None:
    # kitty CSI-u: ESC [ 13 ; 2 u  (13 = Enter keycode, 2 = Shift modifier)
    assert decode_escape(_feed("[13;2u")) == ("newline",)
    # any modifier on Enter breaks the line (Ctrl/Alt+Enter too).
    assert decode_escape(_feed("[13;5u")) == ("newline",)
    # xterm modifyOtherKeys form: ESC [ 27 ; 2 ; 13 ~
    assert decode_escape(_feed("[27;2;13~")) == ("newline",)
    # legacy Alt+Enter: ESC then CR.
    assert decode_escape(_feed("\r")) == ("newline",)


def test_decode_escape_plain_enter_and_escape() -> None:
    assert decode_escape(_feed("[13u")) == ("enter",)     # unmodified Enter (kitty)
    assert decode_escape(_feed("[27u")) == ("esc",)       # Escape key (kitty)
    assert decode_escape(_feed("")) == ("esc",)           # lone ESC (legacy)


def test_decode_escape_bracketed_paste_is_one_block() -> None:
    # CRLF is normalised and the 201~ terminator stripped — no per-line auto-submit.
    assert decode_escape(_feed("[200~hello\r\nworld\x1b[201~")) == ("paste", "hello\nworld")
    # a pasted slash-command stays a single buffer (submitted later, intact).
    assert decode_escape(_feed("[200~/task ship it\x1b[201~")) == ("paste", "/task ship it")


def test_decode_escape_ignores_unhandled_sequences() -> None:
    # a cursor key that leaked past curses keypad decoding is swallowed, not inserted.
    assert decode_escape(_feed("[A")) == ("ignore",)
    # SS3 function-key intro is consumed without effect.
    assert decode_escape(_feed("OP")) == ("ignore",)


def test_decode_escape_modified_printables_are_not_typed() -> None:
    # Under kitty keyboard mode a modifier-bearing printable must NOT land in the buffer
    # (otherwise Ctrl+D / Ctrl+A etc. would corrupt input). Ctrl+letter = mods "5".
    assert decode_escape(_feed("[100;5u")) == ("ignore",)   # Ctrl+d
    assert decode_escape(_feed("[97;3u")) == ("ignore",)    # Alt+a
    # an UNMODIFIED CSI-u printable is still accepted as a character.
    assert decode_escape(_feed("[97u")) == ("char", "a")
    # modifyOtherKeys: a modified printable is ignored, unmodified is typed.
    assert decode_escape(_feed("[27;5;100~")) == ("ignore",)  # Ctrl+d
    assert decode_escape(_feed("[27;1;100~")) == ("char", "d")


def test_decode_escape_uses_patient_reader_for_paste_body() -> None:
    # The control reader stops early (simulating a slow link), but the paste body is
    # drained with a separate patient reader, so a chunked paste is not truncated.
    control = _feed("[200~")                 # only the begin marker is "immediately" ready
    body = _feed("chunked\x1b[201~")         # the body trickles in via the patient reader
    assert decode_escape(control, body) == ("paste", "chunked")


def test_cursor_rowcol_matches_wrap_after_full_row_then_newline() -> None:
    # A logical line that exactly fills the width, followed by an explicit newline: the
    # caret right after the newline lands on the next visual row (no double-advance).
    assert _wrap_input("abcd\nx", 4) == ["abcd", "x"]
    assert _cursor_rowcol("abcd\nx", 5, 4) == (1, 0)   # caret on 'x', visual row 1
    assert _wrap_input("abcd\n", 4) == ["abcd", ""]
    assert _cursor_rowcol("abcd\n", 5, 4) == (1, 0)    # caret on the trailing empty row


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


_PAL = {k: 0 for k in ("gold", "ok", "err", "run", "dim", "head", "fg", "warn")}


def _render_view(v: SwarmView, *, busy: bool, phase: str = "") -> list[str]:
    rows: list[str] = []
    v.render(lambda y, x, t, a=0: rows.append(t), (2, 0, 24, 40),
             palette=_PAL, gate_limit=40, running=0, kv_pct="1%", tok_s=0.0,
             busy=busy, phase=phase)
    return rows


def test_workers_show_main_worker_while_busy_without_fleet() -> None:
    # Chat reply: the runner is busy but no swarm was planned. The WORKERS list must
    # still show the front-door "main" worker animating (the user's "メインWorker").
    rows = _render_view(SwarmView(), busy=True, phase="thinking")
    assert any("WORKERS · 1 active" in r for r in rows)
    assert any(r.startswith(" main") and "thinking" in r for r in rows)
    assert any("● live" in r for r in rows)            # badge tracks busy, not just fan-out


def test_workers_idle_still_shows_main_worker() -> None:
    # Even with nothing running, the roster must always show the main worker — now
    # standing by (idle), never an empty "○ idle" pane. (待機中Workerも常に表示)
    rows = _render_view(SwarmView(), busy=False)
    assert any("WORKERS · 0 active" in r for r in rows)
    assert any(r.startswith(" main") and "idle" in r for r in rows)


def test_workers_list_main_plus_running_and_waiting() -> None:
    v = SwarmView()
    _seed(v)                                              # a, b, r (reducer) planned
    v.ingest({"kind": "task", "event": "dispatch", "id": "a", "counts": {"running": 1}})
    # b stays pending (waiting), r (reducer) stays pending (waiting on a+b)
    rows = _render_view(v, busy=True, phase="running · 0/3 done")
    # 2 active (main + a), 2 waiting (b + r)
    assert any("WORKERS · 2 active · 2 waiting" in r for r in rows)
    assert any(r.startswith(" main") for r in rows)
    assert any("a" in r and "writer" in r for r in rows)   # running sub-agent
    assert any("b" in r and "coder" in r for r in rows)    # waiting sub-agent shown


# ── chat-log scroll (arrow keys / wheel) ──────────────────────────────────────
def test_chatlog_scroll_reveals_older_and_clamps() -> None:
    log = ChatLog()
    log.add("user", "q")
    log.add("assistant", "\n".join(f"L{i:02d}" for i in range(30)))
    pal = {k: 0 for k in ("gold", "fg", "dim", "err")}

    def view(scroll: int):
        rows: list[str] = []
        mx = log.render(lambda y, x, t, a=0: rows.append(t), (0, 0, 9, 40), pal, scroll)
        return mx, [r for r in rows if r.strip()]

    max_scroll, tail = view(0)
    assert max_scroll > 0
    assert any("L29" in r for r in tail)               # scroll=0 pins newest at bottom
    _, older = view(max_scroll)                        # fully scrolled up
    assert any("q" in r for r in older)                # reveals the oldest line (user msg)
    over, _ = view(10_000)                             # over-scroll clamps, never blanks
    assert over == max_scroll


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
    r.shutdown()


def test_btw_runs_independently_and_emits(monkeypatch) -> None:
    import time
    r = SwarmRunner()
    monkeypatch.setattr(r, "_run_agent", lambda *a, **k: "current status: 2 tasks running")
    r.ask_status("what's going on?", "Runner: BUSY")
    # poll the event queue briefly for the 'btw' event (runs on a daemon thread)
    got = None
    for _ in range(50):
        try:
            ev = r.events.get(timeout=0.1)
        except Exception:
            ev = None
        if ev and ev.get("kind") == "btw":
            got = ev; break
    r.shutdown()
    assert got is not None and "2 tasks running" in got["text"]


# ── persistent completion queue ───────────────────────────────────────────────
def test_taskstore_transitions_persistence_and_running_recovery(tmp_path) -> None:
    path = tmp_path / "tasks.json"
    store = TaskStore(str(path), max_attempts=2)
    first = store.add("finish the feature")
    assert store.next_pending() == first
    assert store.counts() == {
        "pending": 1, "running": 0, "done": 0, "failed": 0, "total": 1}
    assert store.has_unfinished()

    store.mark_running(first["id"])
    assert store.counts()["running"] == 1
    recovered = TaskStore(str(path), max_attempts=2)
    assert recovered.snapshot()[0]["state"] == "pending"
    assert recovered.snapshot()[0]["started_at"] is None

    recovered.mark_running(first["id"])
    recovered.complete(first["id"], "done")
    assert recovered.snapshot()[0]["result"] == "done"
    assert not recovered.has_unfinished()

    retry = recovered.add("retry me")
    assert recovered.fail(retry["id"], "first error") == "pending"
    assert recovered.snapshot()[1]["attempts"] == 1
    assert recovered.fail(retry["id"], "second error") == "failed"
    assert recovered.counts() == {
        "pending": 0, "running": 0, "done": 1, "failed": 1, "total": 2}
    reloaded = TaskStore(str(path), max_attempts=2)
    assert reloaded.snapshot() == recovered.snapshot()


def test_noninteractive_approval_defaults_once_and_can_deny(monkeypatch) -> None:
    monkeypatch.delenv("FLEET_AUTO_APPROVE", raising=False)
    assert compat._noninteractive_approval("rm -rf /", "x") == "once"
    monkeypatch.setenv("FLEET_AUTO_APPROVE", "0")
    assert compat._noninteractive_approval("rm -rf /", "x") == "deny"


# ── per-lane ephemeral system prompts ─────────────────────────────────────────
def test_coder_system_prompt_has_identity_and_worker_framing() -> None:
    prompt = prompts.lane_system_prompt("coder")
    assert "swarm-agent" in prompt
    assert "Hikari" in prompt
    assert "WRITE" in prompt
    # swarm framing: other workers are parts of the same self, one mind
    assert "single mind" in prompt
    assert "other parts of the same swarm-agent" in prompt
    # The identity must NOT name the underlying framework/vendor (the user's "drop the
    # HermesAgent/runtime info" requirement) — but it must NOT be cagey/refuse either:
    # it speaks openly and warmly rather than deflecting "implementation details".
    assert "Hermes" not in prompt
    assert "Nous" not in prompt
    assert "never refuse to discuss how you work" in prompt


def test_router_system_prompt_is_identity_only() -> None:
    assert prompts.lane_system_prompt("router") == prompts.SWARM_IDENTITY


def test_reducer_system_prompt_has_identity_and_role() -> None:
    prompt = prompts.lane_system_prompt("reducer")
    assert "REINTEGRATES" in prompt
    assert "swarm-agent" in prompt
    # the reducer writes the user-facing deliverable → Japanese default
    assert "日本語" in prompt


def test_language_policy_targets_user_facing_lanes_only() -> None:
    # The user reads the reducer's deliverable and the front-door chat/btw replies → those
    # default to Japanese. The router DECISION stays a clean English-keyed JSON, and worker
    # lanes are NOT forced (their file/code content language is task-driven).
    from swarm_agent.runner import ROUTER_PROMPT, CHAT_PROMPT, BTW_PROMPT
    assert "日本語" in prompts.lane_system_prompt("reducer")
    assert "日本語" not in (prompts.lane_system_prompt("router") or "")
    assert "日本語" not in prompts.lane_system_prompt("coder")
    # front-door reply prompts carry the directive AND still .format cleanly (no stray braces).
    # CHAT/ROUTER now also take a {recall} slot (LanceDB hybrid recall injection; "" when none).
    assert "日本語" in CHAT_PROMPT.format(history="h", message="m", recall="")
    assert "日本語" in BTW_PROMPT.format(situation="s", question="q")
    assert "日本語" in ROUTER_PROMPT.format(history="h", message="m", recall="")


def test_unknown_lane_system_prompt_uses_worker_framing() -> None:
    assert "single mind" in prompts.lane_system_prompt("nonsense")


def test_swarm_system_prompt_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("FLEET_SWARM_SYSTEM", "0")
    for lane in ("coder", "router", "reducer", "nonsense"):
        assert prompts.lane_system_prompt(lane) is None


def test_front_door_lanes_are_lean_by_default() -> None:
    # Persona/memory injection is OFF by default so swarm-agent's own injected identity
    # is not contaminated by the user's HermesAgent SOUL/memory — which made the live
    # chat confabulate a wrong "27B main + qwen36 workers via delegate_task" self-image.
    from fleet import config
    assert config.PERSONA_LANES == set()
    assert not config.is_persona_lane("router")
    assert not config.is_persona_lane("reducer")


# ── clipboard: chat is copyable, the status panel is not ──────────────────────
def test_chat_plaintext_copies_chat_only_not_status() -> None:
    from swarm_agent.tui import _chat_plaintext
    blocks = [("user", "hi"), ("assistant", "answer one"),
              ("status", "queued → task added"), ("assistant", "answer two"),
              ("error", "boom")]
    whole = _chat_plaintext(blocks)
    assert "> hi" in whole
    assert "answer one" in whole and "answer two" in whole and "boom" in whole
    assert "queued → task added" not in whole         # status chrome is never copied
    assert _chat_plaintext(blocks, last_only=True) == "answer two"
    assert _chat_plaintext([], last_only=True) == ""


# ── planner DAG: auto-repair the common "orphan leaf" slip ────────────────────
def test_validate_tasks_autowires_orphan_leaves_into_reducer() -> None:
    from swarm_agent.goal import tasks_from_json
    # 3 parallel leaves but the reducer only wired one of them (b, c orphaned) — the
    # planner's most common mistake. validate must REPAIR (wire b, c in), not fail.
    plan = [
        {"id": "a", "prompt": "x", "lane": "writer", "deps": []},
        {"id": "b", "prompt": "y", "lane": "writer", "deps": []},
        {"id": "c", "prompt": "z", "lane": "writer", "deps": []},
        {"id": "r", "prompt": "synthesise", "lane": "reducer", "deps": ["a"]},
    ]
    tasks = tasks_from_json(plan)                       # must NOT raise
    red = next(t for t in tasks if t.lane == "reducer")
    assert set(red.deps) == {"a", "b", "c"}            # every leaf now feeds the reducer


def test_validate_tasks_collapses_two_reducers_to_one_sink() -> None:
    from swarm_agent.goal import tasks_from_json
    # Planner emitted TWO reducers (its other common slip). Repair: keep the last as the
    # sole sink, demote the earlier one to a 'writer' leaf that feeds it.
    plan = [
        {"id": "a", "prompt": "x", "lane": "writer", "deps": []},
        {"id": "r1", "prompt": "partial", "lane": "reducer", "deps": ["a"]},
        {"id": "r2", "prompt": "final", "lane": "reducer", "deps": ["a"]},
    ]
    tasks = tasks_from_json(plan)                       # must NOT raise
    reducers = [t for t in tasks if t.lane == "reducer"]
    assert len(reducers) == 1 and reducers[0].id == "r2"   # last reducer is canonical
    assert next(t for t in tasks if t.id == "r1").lane == "writer"  # the other demoted
    sinks = [t for t in tasks if t.id not in {d for x in tasks for d in x.deps}]
    assert len(sinks) == 1 and sinks[0].id == "r2"     # exactly one sink, the reducer


def test_validate_tasks_promotes_mislabeled_sink_to_reducer() -> None:
    from swarm_agent.goal import tasks_from_json
    # The planner's MOST common slip on wide goals (measured ~75%): it emits a perfect
    # fan-in DAG but labels the sole integrating sink "writer" instead of "reducer". Repair
    # must PROMOTE that sink to reducer (it's what everything feeds into) rather than fail +
    # cost a full ~8s plan regeneration on retry.
    plan = [{"id": f"u{i}", "prompt": "p", "lane": "writer", "deps": []} for i in range(5)]
    plan.append({"id": "reduce", "prompt": "synthesise", "lane": "writer",
                 "deps": [f"u{i}" for i in range(5)]})
    tasks = tasks_from_json(plan)                       # must NOT raise
    reducers = [t for t in tasks if t.lane == "reducer"]
    assert len(reducers) == 1 and reducers[0].id == "reduce"   # sink promoted to reducer
    sinks = [t for t in tasks if t.id not in {d for x in tasks for d in x.deps}]
    assert len(sinks) == 1 and sinks[0].id == "reduce"


def test_validate_tasks_no_reducer_no_integrator_is_rejected() -> None:
    from swarm_agent.goal import tasks_from_json
    # No reducer + only INDEPENDENT leaves (no fan-in integrator): there is no task that
    # synthesises the others, so promoting an arbitrary leaf would report just that leaf and
    # silently drop the rest. validate must REJECT (raise) so _plan() retries for a real
    # reducer — NOT fabricate one.
    plan = [{"id": x, "prompt": "p", "lane": "writer", "deps": []} for x in ("a", "b", "c")]
    try:
        tasks_from_json(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("independent leaves with no integrator must not be auto-repaired")


def test_validate_tasks_does_not_promote_write_capable_sink() -> None:
    from swarm_agent.goal import tasks_from_json
    # No reducer + only independent CODER leaves: promoting one to reducer would strip its
    # file/shell tools and silently drop its write work (Codex review P2). Must NOT repair —
    # raise instead so the planner retry regenerates a real reducer.
    plan = [{"id": "f1", "prompt": "write file A", "lane": "coder", "deps": []},
            {"id": "f2", "prompt": "write file B", "lane": "coder", "deps": []}]
    try:
        tasks_from_json(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("write-capable sinks must not be auto-promoted to reducer")
    # But a NON-write integrator sink alongside coder leaves IS promoted (coders keep tools).
    plan2 = [{"id": "f1", "prompt": "write file A", "lane": "coder", "deps": []},
             {"id": "sum", "prompt": "summarise", "lane": "writer", "deps": ["f1"]}]
    tasks = tasks_from_json(plan2)
    red = [t for t in tasks if t.lane == "reducer"]
    assert len(red) == 1 and red[0].id == "sum"        # the writer integrator promoted
    assert next(t for t in tasks if t.id == "f1").lane == "coder"   # coder keeps its tools


def test_multiswarm_single_goal_renders_like_single_view() -> None:
    from swarm_agent.dashboard import MultiSwarmView
    mv = MultiSwarmView()
    mv.ingest({"kind": "user", "text": "only goal", "goal_id": "g1"})
    mv.ingest({"kind": "planning", "goal_id": "g1"})
    mv.ingest({"kind": "planned", "goal_id": "g1", "tasks": [
        {"id": "a", "lane": "writer", "deps": []},
        {"id": "r", "lane": "reducer", "deps": ["a"]}]})
    rows: list[str] = []
    pal = {k: 0 for k in ("gold", "ok", "err", "run", "dim", "head", "fg", "warn")}
    mv.render(lambda y, x, t, a=0: rows.append(t), (2, 0, 24, 40),
              palette=pal, gate_limit=40, running=1, kv_pct="1%", tok_s=0.0, busy=True)
    assert any("WORKERS" in r for r in rows)          # delegated to a single SwarmView layout
    assert any("TASKS" in r for r in rows)


def test_multiswarm_two_goals_render_compact_blocks() -> None:
    from swarm_agent.dashboard import MultiSwarmView
    mv = MultiSwarmView()
    for gid, text in (("g1", "audit repo"), ("g2", "summarise docs")):
        mv.ingest({"kind": "user", "text": text, "goal_id": gid})
        mv.ingest({"kind": "planning", "goal_id": gid})
        mv.ingest({"kind": "planned", "goal_id": gid, "tasks": [
            {"id": "a", "lane": "writer", "deps": []},
            {"id": "r", "lane": "reducer", "deps": ["a"]}]})
    rows: list[str] = []
    pal = {k: 0 for k in ("gold", "ok", "err", "run", "dim", "head", "fg", "warn")}
    mv.render(lambda y, x, t, a=0: rows.append(t), (2, 0, 24, 40),
              palette=pal, gate_limit=40, running=2, kv_pct="2%", tok_s=10.0, busy=True)
    assert any("2 goals" in r for r in rows)          # multi title
    assert any("audit repo" in r for r in rows)
    assert any("summarise docs" in r for r in rows)


def test_multiswarm_render_short_pane_no_crash() -> None:
    from swarm_agent.dashboard import MultiSwarmView
    mv = MultiSwarmView()
    for gid in ("g1", "g2", "g3"):
        mv.ingest({"kind": "user", "text": gid, "goal_id": gid})
        mv.ingest({"kind": "planning", "goal_id": gid})
    pal = {k: 0 for k in ("gold", "ok", "err", "run", "dim", "head", "fg", "warn")}
    for bottom in (4, 6, 9, 30):
        mv.render(lambda y, x, t, a=0: None, (2, 0, bottom, 40),
                  palette=pal, gate_limit=40, running=1, kv_pct="1%", tok_s=1.0)


# ── parallel goals: a finished goal must not blank the global status while peers run ──
def test_idle_keeps_global_status_while_another_goal_runs(tmp_path, monkeypatch) -> None:
    # When two queued goals run at once, the FIRST 'idle' must not reset the global UI
    # chrome to "ready" / drop elapsed tracking while the second goal is still in flight.
    monkeypatch.setenv("SWARM_TASKS_PATH", str(tmp_path / "tasks.json"))
    from swarm_agent.tui import App
    app = App(None)                                  # __init__ touches no curses / network
    try:
        with app.runner._busy_lock:                 # simulate two goal turns in flight
            app.runner._active["g1"] = None
            app.runner._active["g2"] = None
        app.started_at, app.phase, app.message = 123.0, "running", "working…"

        with app.runner._busy_lock:                 # g1's turn ends (pops itself) then emits idle
            app.runner._active.pop("g1")
        app.runner.events.put({"kind": "idle", "goal_id": "g1"})
        app.pump()
        assert app.runner.busy                       # g2 still running
        assert (app.message, app.started_at, app.phase) == ("working…", 123.0, "running")

        with app.runner._busy_lock:                 # g2 ends -> now truly idle
            app.runner._active.pop("g2")
        app.runner.events.put({"kind": "idle", "goal_id": "g2"})
        app.pump()
        assert not app.runner.busy
        assert (app.message, app.started_at, app.phase) == ("ready", None, "")
    finally:
        app.runner.shutdown()


# ── mid-flight interject (steer / interrupt) ──────────────────────────────────
class _FakeAgent:
    """Stand-in for an AIAgent: records steer/interrupt calls (steer returns accepted)."""
    def __init__(self) -> None:
        self.steers: list = []
        self.interrupts: list = []

    def steer(self, text: str) -> bool:
        self.steers.append(text)
        return True

    def interrupt(self, message=None) -> None:
        self.interrupts.append(message)


def test_compat_steer_fans_out_and_stashes_for_late_agents() -> None:
    a, b = _FakeAgent(), _FakeAgent()
    compat._register_agent(a)
    compat._register_agent(b)
    try:
        assert compat.steer_all("focus on tests") == 2
        assert a.steers == ["focus on tests"] and b.steers == ["focus on tests"]
        # an agent that STARTS after the steer (reducer / late worker) drains the stash
        late = _FakeAgent()
        compat._drain_pending_steer_into(late)
        assert late.steers == ["focus on tests"]
        assert compat.steer_all("   ") == 0          # empty steer ignored
    finally:
        compat._unregister_agent(a)
        compat._unregister_agent(b)
        compat.reset_interject()
    fresh = _FakeAgent()                              # after reset → nothing leaks forward
    compat._drain_pending_steer_into(fresh)
    assert fresh.steers == []


def test_compat_interrupt_fans_out_to_live_agents() -> None:
    a = _FakeAgent()
    compat._register_agent(a)
    try:
        assert compat.interrupt_all("stop now") == 1
        assert a.interrupts == ["stop now"]
    finally:
        compat._unregister_agent(a)


def test_runner_steer_emits_event_and_reaches_live_agent() -> None:
    a = _FakeAgent()
    compat._register_agent(a)
    r = SwarmRunner()
    try:
        assert r.steer("also handle errors") == 1
        assert a.steers == ["also handle errors"]
        evs = []
        while True:
            try:
                evs.append(r.events.get_nowait())
            except Exception:
                break
        steer_evs = [e for e in evs if e.get("kind") == "steer"]
        assert len(steer_evs) == 1
        assert steer_evs[0]["reached"] == 1 and steer_evs[0]["text"] == "also handle errors"
    finally:
        compat._unregister_agent(a)
        compat.reset_interject()
        r.shutdown()


def test_dispatch_busy_plain_interjects_and_forced_queues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARM_TASKS_PATH", str(tmp_path / "tasks.json"))
    from swarm_agent.tui import App
    app = App(None)
    try:
        app.runner.busy = True                        # simulate a turn in flight
        steered: list = []
        monkeypatch.setattr(app.runner, "steer", lambda t: steered.append(t) or 1)
        app._dispatch("tweak the plan", None)         # plain → interject, not queued
        assert steered == ["tweak the plan"] and app.pending == []
        app._dispatch("a fresh question", "chat")     # explicit /chat → queued as new turn
        assert app.pending == [("a fresh question", "chat")]
    finally:
        app.runner.shutdown()
