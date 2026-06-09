"""Goal-driven planner front door for swarm-agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from fleet import config
from fleet.board import Task
from fleet.config import WRITE_TOOLSETS, lane_writes  # re-exported; capability-based classifier
from fleet.worker import _final_text


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str):
    """Parse a planner response, tolerating a fenced JSON block."""
    candidates = [m.group(1) for m in _JSON_BLOCK.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("["), candidate.rfind("]")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError:
                    pass
    raise ValueError("planner did not return a JSON task array")


def _extract_obj(text: str):
    """Tolerant JSON-OBJECT parse (bare or embedded in prose), mirroring runner._parse_route.
    Returns the dict or None — never raises. (``_extract_json`` above hunts for an ARRAY.)"""
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


ANALYZE_DEPS_PROMPT = """A swarm runs queued goals concurrently. Decide whether a NEW goal must
WAIT for any already-queued goals because it CONSUMES their result (an ordering dependency).

Only mark a dependency when the new goal genuinely needs another goal's OUTPUT to exist first
(e.g. "write tests for endpoint X" needs "implement endpoint X"). Do NOT mark a dependency just
because two goals touch related areas — independent goals should run in parallel (file-write
conflicts are handled separately, not here). When unsure, prefer INDEPENDENT (empty list).

Already-queued goals (id + goal):
{existing}

NEW goal:
{new}

Reply with ONLY a JSON object, nothing else:
  {{"deps":["<id of a goal the NEW goal must wait for>", ...]}}
Use ONLY ids from the list above; an empty list means the new goal is independent."""


def analyze_deps(new_goal: str, existing, *, run_agent) -> list[str]:
    """Return the ids of already-queued goals the NEW goal must wait for (inter-goal DAG edges).

    ``run_agent`` is an injectable callable ``(lane, prompt, task_id, **kw) -> str`` (default in
    production: ``runner._run_agent``) so offline tests pass a fake returning canned JSON — no
    inference server is touched. FAIL OPEN: no candidates, an empty/garbled reply, or any error
    → ``[]`` (independent; a wrong "ordered" needlessly serialises, and real WRITE conflicts are
    caught at merge anyway — §6.1). Only ids that exist in ``existing`` survive."""
    candidates = [e for e in (existing or []) if e.get("id")]
    if not candidates:
        return []
    rows = [{"id": e["id"], "goal": str(e.get("goal") or "")[:160]} for e in candidates]
    prompt = ANALYZE_DEPS_PROMPT.format(
        existing=json.dumps(rows, ensure_ascii=False), new=str(new_goal)[:400])
    try:
        text = run_agent("manager", prompt, "analyze-deps", max_iterations=1, max_tokens=512)
    except Exception:
        return []
    data = _extract_obj(text or "")
    deps = data.get("deps") if isinstance(data, dict) else None
    if not isinstance(deps, list):
        return []
    valid = {e["id"] for e in candidates}
    seen, out = set(), []
    for dep in deps:
        d = str(dep)
        if d in valid and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def validate_tasks(tasks: Iterable[Task]) -> list[Task]:
    """Return a validated acyclic DAG with exactly one reducer sink.

    Auto-repair (turn the common planner slips into valid plans instead of a hard
    "plan parse failed" — which costs a full ~8s plan REGENERATION on retry):
      * NO reducer lane — the planner very often (~75% on wide goals, measured) emits the
        final integrating task with the WRONG lane (e.g. "writer") even though it is the
        sole sink that depends on every leaf. PROMOTE that integrating sink to "reducer"
        instead of failing: the task everything feeds into IS the reducer regardless of the
        label the planner stuck on it.
      * MULTIPLE reducers — keep the last reducer as the canonical sink and demote the
        others to "writer" (intermediate synthesis leaves that feed the final reducer).
      * ORPHAN leaves — any task nothing depends on is wired INTO the canonical reducer
        (which is meant to integrate every leaf anyway), so it becomes the sole sink.
    """
    out = list(tasks)
    if not out:
        raise ValueError("planner returned no tasks")
    by_id = {task.id: task for task in out}
    if len(by_id) != len(out):
        raise ValueError("planner returned duplicate task ids")
    for task in out:
        missing = sorted(set(task.deps) - by_id.keys())
        if missing:
            raise ValueError(f"task {task.id!r} has unknown deps: {missing}")

    # Collapse to ONE reducer sink + wire orphan leaves into it (see docstring). No-op if
    # the plan is already well-formed. Cycle check runs AFTER, so any repair edges are
    # validated too.
    reducers = [t for t in out if t.lane == "reducer"]
    if not reducers and len(out) > 1:
        # No task is labelled "reducer" — promote the integrating sink (the planner very
        # often mislabels it "writer"). Promote ONLY a genuine fan-in INTEGRATOR: a sink that
        # (a) DEPENDS on other tasks (so its result actually synthesises them) and (b) is
        # NON-write-capable (promoting a coder/analyst would strip its file/shell tools and
        # drop real work). NEVER promote an independent leaf — its prompt is its own unit of
        # work, not synthesis, so final_result() would report just that leaf and silently omit
        # the rest. If there is no such integrator (e.g. several independent leaves, or only
        # write-capable sinks), leave the plan unrepaired: the sink check below raises and
        # _plan() retries for a properly-wired reducer. Tie-break to the LAST (most deps).
        depended_on = {dep for task in out for dep in task.deps}
        sinks = [t for t in out if t.id not in depended_on]
        integrators = [t for t in sinks if t.deps and not lane_writes(t.lane)]
        if integrators:
            cand = max(reversed(integrators), key=lambda t: len(t.deps))
            cand.lane = "reducer"
            reducers = [cand]
    if reducers:
        red = reducers[-1]                       # canonical sink = the last reducer
        for extra in reducers[:-1]:              # demote any other reducers to leaves
            extra.lane = "writer"
        depended_on = {dep for task in out for dep in task.deps}
        for task in out:
            if (task is not red and task.id not in depended_on
                    and task.id not in red.deps):
                red.deps.append(task.id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"planner returned a dependency cycle at {task_id!r}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dep in by_id[task_id].deps:
            visit(dep)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)

    depended_on = {dep for task in out for dep in task.deps}
    sinks = [task for task in out if task.id not in depended_on]
    if len(sinks) != 1 or sinks[0].lane != "reducer":
        raise ValueError("planner must return exactly one reducer sink")
    return out


def tasks_from_json(data) -> list[Task]:
    if not isinstance(data, list):
        raise ValueError("planner response must be a JSON array")
    tasks = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each planner task must be an object")
        tasks.append(Task(
            id=str(item["id"]),
            prompt=str(item["prompt"]),
            deps=[str(dep) for dep in item.get("deps", [])],
            lane=str(item.get("lane", "worker")),
            meta=dict(item.get("meta", {})),
        ))
    return validate_tasks(tasks)


def classify_plan(tasks: Iterable[Task]) -> str:
    """Return "read-only" iff NO task can write, else "writing" (run exclusively).

    Classified by real tool capability (``lane_writes``): a goal is safe to run concurrently
    only if every one of its lanes is genuinely non-mutating (writer/researcher/reducer and
    the like). A lane that carries file/terminal/code tools — including analyst/reviewer
    (whose "file" toolset exposes write_file/patch) and any unrecognised lane (→ write-capable
    worker fallback) — forces the whole goal to "writing". Fail closed: an empty/degenerate
    plan is also "writing" (never run something unclassifiable alongside others)."""
    tasks = list(tasks)
    if not tasks:
        return "writing"
    return "writing" if any(lane_writes(t.lane) for t in tasks) else "read-only"


def namespace_tasks(tasks: Iterable[Task], prefix: str) -> list[Task]:
    """Return NEW Task objects with every id (and dep) prefixed by ``prefix`` so two
    concurrent goals can never share a task id — which would collide their per-worker
    sandbox cwd (compat.worker_sandbox keys on spec["id"]) (PARALLEL_GOALS_PLAN §4.3).

    Does not mutate the input tasks. Re-validates the result (still an acyclic DAG with one
    reducer sink). A prefix already present is not doubled.
    """
    def nid(i: str) -> str:
        return i if i.startswith(f"{prefix}.") else f"{prefix}.{i}"
    out = [Task(id=nid(t.id), prompt=t.prompt, deps=[nid(d) for d in t.deps],
                lane=t.lane, meta=dict(t.meta)) for t in tasks]
    return validate_tasks(out)


PLANNER_PROMPT = """You are the planner for a high-concurrency software swarm. Decompose
the user's goal into independent tasks that run in parallel, plus ONE final reducer task
that synthesizes them into the deliverable.

Design the MOST EFFICIENT workflow for THIS environment. The runtime runs MANY tasks at
once (its efficient operating point is dozens of agents generating concurrently — tens in
flight is cheap), every worker is a lean ~8K-context agent with a bounded turn budget, and
total wall-clock ≈ the LONGEST dependency chain, NOT the sum of the work. So shape the DAG
for that machine:

1. MAXIMISE PARALLEL BREADTH, MINIMISE DEPTH. Tasks that do not consume each other's output
   MUST be independent (empty "deps") so they run at the SAME time. Add a dependency ONLY
   when a task genuinely needs another's result. Aim for one WIDE layer of independent
   tasks feeding the single reducer; avoid long A→B→C chains (each link is serial wall-time).
2. ONE TASK PER INDEPENDENT UNIT. If the goal spans many units ("for each X", several
   files / modules / topics / sections / sources), emit a SEPARATE task per unit so they
   run in parallel — never serialise them, never lump them into one task.
3. RIGHT-SIZE EACH TASK. Each task is run by ONE lean worker with a bounded turn budget. A
   task that is too big makes that worker churn or run out of turns — split it into parallel
   pieces. A trivially tiny task wastes overhead — merge it. Target a scope a focused worker
   finishes in a few turns.
4. RESPECT THE SINGLE REDUCER. Exactly ONE reducer integrates ALL leaves and its context is
   finite, so keep the leaf count sensible and instruct each leaf to return a CONCISE,
   structured result the reducer can absorb without overflowing.
5. CHEAPEST CAPABLE LANE (see below). Breadth is cheap, but every tool a lane carries is
   re-paid on every turn — still pick the smallest lane per task.

Scale the task count to the goal — do not invent busywork; each task must earn its place:
trivial → 1-3 tasks; medium → 4-8; large / many-unit → fan out wide (up to ~16-20 concise
leaves). Each task is a specific, standalone assignment whose result is plain text.

Assign each task a "lane" by the CAPABILITY it truly needs. Tools and skills are not
free: each tool a lane carries is re-sent on every turn and can fail on a task that did
not need it, and a skill-enabled lane also pays for a skills index. So pick the SMALLEST
lane that can do the job, and DEFAULT to "writer".

Lane menu (cheapest first):
  - "writer"     : NO tools. Pure reasoning, explanation, analysis, design, or synthesis
                   from your own knowledge. PREFER THIS for explanatory / conceptual /
                   writing / planning work. This is the DEFAULT — use it unless a task
                   genuinely needs files, the web, or execution.
  - "analyst"    : read-only repo access (read files + search), NO terminal, NO writes.
                   Use to READ/INSPECT/SUMMARIZE existing files or find code, when the
                   task must NOT run commands or modify anything.
  - "researcher" : web access + skills. Use ONLY when the task needs external or current
                   information from the internet. It also gets research/data skills.
  - "coder"      : read+write files, terminal, and code execution. Use when the task must
                   EDIT files, RUN commands, or EXECUTE code in this project. Most
                   expensive lane — use only for real build/run/edit work.
  - "reviewer"   : read files + search + skills (code-review / security-review). Use to
                   audit or critique existing code WITHOUT modifying it.

Rules of thumb:
  - Explaining, designing, drafting, comparing ideas → writer.
  - "What does file X do / where is Y / summarize the repo" → analyst (read-only).
  - "Look up / find current info / cite sources" → researcher.
  - "Edit / build / run / fix / generate code" → coder.
  - "Review / audit / find bugs or vulns in existing code" → reviewer.
When two lanes could work, choose the cheaper one.

Deliverable type — decide this FIRST:
  - TEXT deliverable (explanation, analysis, report, comparison, plan): use
    writer/analyst/researcher tasks + a reducer that synthesizes the final text.
  - FILE-ON-DISK or RUNNABLE deliverable ("write X to a file", "create a script",
    "build a game/app/tool", "generate a config"): you MUST include a "coder" task that
    AUTHORS AND WRITES the file(s) to disk, and (when sensible) RUNS a shell check to
    verify (e.g. the file exists and contains the expected markers). The "reducer" has
    NO TOOLS — it cannot read, write, or run anything; it only synthesizes text from
    upstream results, so NEVER expect the reducer to create or save a file. For a single
    self-contained file, prefer ONE coder task that writes the whole file over many text
    tasks whose output would have to be re-assembled (re-assembly in the reducer is slow
    and often truncates large files).

Return ONLY a JSON array. Each item has:
  {{"id":"short-id","prompt":"specific standalone assignment","deps":[],"lane":"writer"}}

REDUCER WIRING — get this exactly right or the plan is invalid:
  - There is EXACTLY ONE task with lane "reducer". It is the LAST item and the ONLY sink.
  - Its "deps" MUST list the id of EVERY other task (every leaf), so it waits for all of
    them and integrates their results. Do not leave any task unconnected.
  - NOTHING may depend on the reducer (no task lists the reducer's id in its deps).
  - Use at most ONE reducer — never two. Independent leaves keep "deps":[] (they run in
    parallel); only the reducer (and any genuine A→B consumer) carries deps.
Example shape for a 3-unit goal: three leaves with "deps":[], then
  {{"id":"reduce","lane":"reducer","deps":["unit1","unit2","unit3"],"prompt":"…"}}

The reducer produces the complete integrated deliverable — for a FILE/RUNNABLE goal the
files are written by the coder task(s); the reducer then only reports concisely what was
built and the verification result (it must not try to write files itself).

User goal:
{goal}
"""


def parse_plan(text: str) -> list[Task]:
    """Parse a planner response (fenced or bare JSON array) into a validated DAG."""
    return tasks_from_json(_extract_json(text))


def plan_goal(goal: str) -> list[Task]:
    """Ask a planner agent to decompose one high-level goal into a task DAG.

    Standalone entry point: it applies the compat shim itself. In-process callers
    that already own a DecodeGate (e.g. ``swarm_agent.runner``) must NOT use this —
    its ``compat.apply(None)`` would clobber their gate. Drive the planner via
    ``PLANNER_PROMPT`` + ``parse_plan`` instead.
    """
    from fleet import compat, config

    compat.apply(None)
    compat.prewarm([config.toolsets_for("planner")])
    planner = compat.make_agent(
        "planner",
        task_id="goal-planner",
        max_iterations=2,
        max_tokens=4096,
    )
    result = planner.run_conversation(PLANNER_PROMPT.format(goal=goal), task_id="goal-planner")
    text = _final_text(result.get("messages"), result.get("final_response"))
    return parse_plan(text)


def write_plan(tasks: Iterable[Task], path: str) -> None:
    with Path(path).open("w") as f:
        for task in tasks:
            f.write(json.dumps({
                "id": task.id,
                "prompt": task.prompt,
                "deps": task.deps,
                "lane": task.lane,
                "meta": task.meta,
            }, ensure_ascii=False) + "\n")


def final_result(board_results) -> str:
    tasks = list(board_results.values())
    depended_on = {dep for task in tasks for dep in task.deps}
    sinks = [task for task in tasks if task.id not in depended_on]
    if len(sinks) != 1:
        raise ValueError(f"expected one sink task, found {len(sinks)}")
    return sinks[0].result or ""
