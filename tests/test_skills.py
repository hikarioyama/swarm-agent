"""Standalone skill system — CRUD, telemetry, auto-transitions, consolidation (stub LLM).

No HermesAgent imports; uses a temp SWARM_SKILLS_DIR (read at call time by _env)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from swarm_agent.skills import manager, usage, curator, format as skfmt, _env


@pytest.fixture()
def skdir(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    monkeypatch.setenv("SWARM_SKILLS_DIR", str(d))
    return d


SKILL = ("---\nname: deploy-rollback\ndescription: Roll back a bad deploy safely\n---\n"
         "# Deploy rollback\n1. find the last good tag\n2. redeploy it\n")


def test_create_list_view(skdir):
    r = manager.create("deploy-rollback", SKILL)
    assert r["success"], r
    assert (skdir / "deploy-rollback" / "SKILL.md").exists()
    # telemetry marks it curator-eligible
    assert usage.get_record("deploy-rollback")["created_by"] == "agent"
    names = [s["name"] for s in skfmt.list_skills(skdir)]
    assert "deploy-rollback" in names


def test_create_validation(skdir):
    assert not manager.create("Bad Name", SKILL)["success"]           # invalid name
    assert not manager.create("ok-name", "no frontmatter here")["success"]
    assert not manager.create("ok-name", "---\nname: x\n---\n")["success"]  # missing desc/body


def test_patch_and_telemetry(skdir):
    manager.create("deploy-rollback", SKILL)
    r = manager.patch("deploy-rollback", "redeploy it", "redeploy it and verify health")
    assert r["success"], r
    body = (skdir / "deploy-rollback" / "SKILL.md").read_text()
    assert "verify health" in body
    assert usage.get_record("deploy-rollback")["patch_count"] == 1


def test_write_file_and_traversal(skdir):
    manager.create("deploy-rollback", SKILL)
    assert manager.write_file("deploy-rollback", "references/runbook.md", "# runbook")["success"]
    assert (skdir / "deploy-rollback" / "references" / "runbook.md").exists()
    assert not manager.write_file("deploy-rollback", "../../escape.md", "x")["success"]
    assert not manager.write_file("deploy-rollback", "/etc/passwd", "x")["success"]


def test_delete(skdir):
    manager.create("deploy-rollback", SKILL)
    assert manager.delete("deploy-rollback")["success"]
    assert manager.find_skill_dir("deploy-rollback") is None
    assert usage.load_usage().get("deploy-rollback") is None


def _backdate(name, days):
    data = usage.load_usage()
    iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    data[name]["last_used_at"] = iso
    usage.save_usage(data)


def test_auto_transitions(skdir, monkeypatch):
    monkeypatch.setenv("SWARM_CURATOR_STALE_DAYS", "30")
    monkeypatch.setenv("SWARM_CURATOR_ARCHIVE_DAYS", "90")
    for n in ("fresh-skill", "stale-skill", "old-skill"):
        c = SKILL.replace("deploy-rollback", n)
        assert manager.create(n, c)["success"]
    _backdate("fresh-skill", 1)
    _backdate("stale-skill", 45)
    _backdate("old-skill", 120)
    counts = curator.apply_automatic_transitions()
    assert counts["archived"] == 1 and counts["marked_stale"] == 1
    assert usage.get_record("stale-skill")["state"] == "stale"
    assert "old-skill" in usage.list_archived_skill_names()
    assert manager.find_skill_dir("old-skill") is None   # moved to .archive


def test_consolidation_with_stub_proposer(skdir):
    manager.create("narrow-a", SKILL.replace("deploy-rollback", "narrow-a"))
    manager.create("narrow-b", SKILL.replace("deploy-rollback", "narrow-b"))
    plan = {"actions": [{"op": "delete", "name": "narrow-b", "absorbed_into": "narrow-a"}]}
    out = curator.run_llm_consolidation(lambda prompt: json.dumps(plan))
    assert out["applied"] and out["applied"][0]["result"]["success"]
    assert manager.find_skill_dir("narrow-b") is None
    assert manager.find_skill_dir("narrow-a") is not None


def test_consolidation_handles_fenced_json(skdir):
    manager.create("narrow-a", SKILL.replace("deploy-rollback", "narrow-a"))
    raw = "```json\n" + json.dumps({"actions": []}) + "\n```"
    out = curator.run_llm_consolidation(lambda p: raw)
    assert out["applied"] == []


def test_synth_creates_skill_from_outcome(skdir):
    from swarm_agent.skills import synth
    new = ("---\nname: cuda-gguf-pin\ndescription: Pin CUDA 13.1 for GGUF inference\n---\n"
           "# CUDA GGUF\nUse CUDA 13.1; 13.2 corrupts GGUF output.\n")
    plan = {"actions": [{"op": "create", "name": "cuda-gguf-pin", "content": new}]}
    out = synth.synthesize("debug GGUF gibberish", "root cause: CUDA 13.2 bug; pinned 13.1",
                           propose_fn=lambda prompt: json.dumps(plan))
    assert out["applied"][0]["result"]["success"], out
    assert manager.find_skill_dir("cuda-gguf-pin") is not None
    assert usage.get_record("cuda-gguf-pin")["created_by"] == "agent"


def test_synth_noop_when_nothing_reusable(skdir):
    from swarm_agent.skills import synth
    out = synth.synthesize("what time is it", "12:00",
                           propose_fn=lambda prompt: '{"actions": []}')
    assert out["applied"] == []
    assert skfmt.list_skills(skdir) == []


def test_synth_disabled(skdir, monkeypatch):
    from swarm_agent.skills import synth
    monkeypatch.setenv("SWARM_SKILL_SYNTH", "0")
    out = synth.synthesize("x", "y", propose_fn=lambda p: '{"actions":[{"op":"create"}]}')
    assert out == {"skipped": "disabled"}
