"""Goal-completion skill synthesis (the auto-generation half).

After a swarm goal completes, review (goal, outcome) and — only if it produced genuinely
REUSABLE procedural knowledge — propose creating/updating a skill. The LLM proposes a JSON
plan; the harness applies it via manager.* (no model-callable tools). Non-invasive: this is
the swarm-layer alternative to HermesAgent's agent-driven `skill_manage` tool calls.

Conservative by design: most tasks warrant NO skill. Fire-and-forget from the runner so it
never blocks a turn; fail-soft.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from . import _env, curator
from . import format as skfmt

SYNTH_PROMPT = """You are the skill synthesizer for swarm-agent. A task just completed. \
If — and ONLY if — it produced REUSABLE procedural knowledge worth saving for FUTURE tasks \
(a repeatable how-to, a non-obvious gotcha + fix, a multi-step workflow, an environment \
quirk), propose creating or updating a skill. MOST tasks do NOT warrant a skill — when in \
doubt, return {{"actions": []}}. Never save one-off task results, secrets, or trivia.

Existing skills (do not duplicate; PATCH/EDIT an existing one if it's the right home):
{existing}

Completed task goal:
{goal}

Outcome / result:
{result}

Reply with ONLY a JSON object (no prose):
{{"actions": [
  {{"op": "create", "name": "<lowercase-hyphen-name>", "content": "---\\nname: <name>\\ndescription: <one line>\\n---\\n# Title\\n<concise reusable steps>"}},
  {{"op": "patch", "name": "<existing>", "old_string": "...", "new_string": "..."}}
]}}
SKILL.md content MUST have frontmatter (name, description) + a non-empty markdown body. \
If nothing is worth saving, return {{"actions": []}}."""


def synthesize(goal_text: str, result_text: str,
               propose_fn: Optional[Callable[[str], str]] = None,
               dry_run: bool = False) -> Dict[str, Any]:
    """Review one completed goal and maybe create/patch a skill. Returns a summary dict."""
    if not _env.synth_enabled():
        return {"skipped": "disabled"}
    if not (goal_text or "").strip():
        return {"skipped": "empty goal"}
    if propose_fn is None:
        from .llm import make_proposer
        propose_fn = make_proposer()
    existing = ", ".join(s["name"] for s in skfmt.list_skills(_env.skills_dir())) or "(none)"
    prompt = SYNTH_PROMPT.format(existing=existing,
                                 goal=(goal_text or "")[:2000],
                                 result=(result_text or "")[:4000])
    try:
        raw = propose_fn(prompt) or ""
    except Exception as e:
        return {"error": f"proposer failed: {e}", "applied": []}
    plan = curator._extract_json(raw) or {}
    actions = plan.get("actions") or []
    applied = []
    for act in actions:
        if not isinstance(act, dict):
            continue
        if dry_run:
            applied.append({"op": act.get("op"), "name": act.get("name"), "dry_run": True})
        else:
            applied.append({"op": act.get("op"), "name": act.get("name"),
                            "result": curator._apply_action(act)})
    return {"applied": applied, "n_actions": len(actions)}


def synthesize_async(goal_text: str, result_text: str) -> None:
    """Fire-and-forget: never blocks the turn, never raises into the caller."""
    if not _env.synth_enabled():
        return
    import threading

    def _w():
        try:
            synthesize(goal_text, result_text)
        except Exception:
            pass
    threading.Thread(target=_w, daemon=True).start()
