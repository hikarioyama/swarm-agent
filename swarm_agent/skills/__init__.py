"""Standalone skill system for swarm-agent — auto-generation + self-improving curator.

VENDORED from HermesAgent's skill system (tools/skill_usage.py, agent/skill_utils.py,
tools/skill_manager_tool.py, agent/curator.py) but **completely decoupled**: zero runtime
imports from ~/.hermes/hermes-agent. All HermesAgent coupling (hermes_constants,
hermes_cli.config, tools.skills_guard, fuzzy_match, path_security, gateway, run_agent) is
replaced by swarm-local implementations:
  - paths/config        -> swarm_agent.skills._env  (swarm-private skills dir)
  - frontmatter/discovery -> swarm_agent.skills.format
  - telemetry/state machine -> swarm_agent.skills.usage
  - CRUD (create/edit/patch/delete/write_file) -> swarm_agent.skills.manager (difflib patch)
  - curator (auto-transition + LLM consolidation) -> swarm_agent.skills.curator
  - goal-completion skill synthesis -> swarm_agent.skills.synth

The LLM passes (synthesis, curator consolidation) run on swarm-agent's OWN agent driver
(fleet.compat.make_agent) and use a PROPOSE-JSON → harness-APPLIES pattern — the model never
needs a model-callable skill tool, so nothing is added to HermesAgent.
"""
