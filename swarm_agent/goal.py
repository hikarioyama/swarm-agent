"""Goal-driven planner front door for swarm-agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from fleet.board import Task
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


def validate_tasks(tasks: Iterable[Task]) -> list[Task]:
    """Return a validated acyclic DAG with at least one reducer sink."""
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


PLANNER_PROMPT = """You are the planner for a high-concurrency software swarm. Decompose
the user's goal into independent tasks that run in parallel, plus ONE final reducer task
that synthesizes them into the deliverable.

Maximize useful parallelism but do not invent busywork: use as few tasks as the goal
needs (2-12; fewer for simple goals). Each task is a specific, standalone assignment
whose result is plain text.

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
The FINAL item must have lane "reducer", depend on every required upstream task, and
instruct the reducer to produce the complete integrated deliverable — for a FILE/RUNNABLE
goal the files are written by the coder task(s); the reducer then only reports concisely
what was built and the verification result (it must not try to write files itself).

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
