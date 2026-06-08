# swarm-agent webui — contract (backend ↔ frontend single source of truth)

A read-only sidecar. The swarm core is left untouched. It tails `latest.jsonl` / replays past logs,
builds graph state, and pushes it to the browser over WebSocket. Both the backend (Python) and the frontend (JS)
**must strictly follow the message shapes in this document.**

## Node ID namespace

Task ids (such as `weak_arch`) can collide across goals/turns, so outward-facing node ids must always be
namespaced: `node_id = f"{goal_key}::{task_id}"`. `deps`/`edges`/`flow` use the same namespaced ids.
- planner virtual node: `f"{goal_key}::__planner__"` (lane=`planner`, state=`done`, fixed).
- goal cluster id = `goal_key`.

## How goal_key is determined (to distinguish each turn in replay)

- Event with non-null `goal_id` → `goal_key = goal_id`.
- Null `goal_id` (interactive turn) → the server keeps an interactive turn counter N. On `kind=="user"` with a null goal_id,
  it increments N by 1, sets `current_interactive_key = f"turn-{N}"`, and the label = the user's `text`. Subsequent
  `planning/planned/task/final/idle` events with a null goal_id are assigned to `current_interactive_key`.
  → This way, consecutive swarm turns within the same session become separate clusters and are all visible in replay.
- Turns with no planning, such as chat responses, become a cluster with 0 nodes. **Clusters with 0 nodes must not be
  included in snapshot/patch** (the rendering side must not emit empty clusters).

## State machine (reuse)

Each goal_key holds one `swarm_agent.dashboard.SwarmView`, and the relevant kinds
(`planning/planned/task/final/error/idle`) are delegated to `view.ingest(ev)`. SwarmView maintains
`view.tasks[tid] = {lane, deps, prompt, state, wall_s, turns}` along with `view.order`/`view.counts`/`view.stranded`.
state values: `pending|running|done|retry|failed|stranded`.
Do not use MultiSwarmView (because it discards clusters on idle). Clusters are never discarded; on `final`/`idle` the
goal is simply set to `state="complete"`.

After ingest, diff the view against the previous snapshot to produce patch ops:
- tid newly appearing in view.order → `add_node` (plus an `add_edge` for each dep).
- tid whose state changed → `set_state`.
Per key, keep an "already-emitted node set" and a "node→latest state" map to compute the diff.

## flow (particle trigger) generation

- After ingesting `task done` (id=X), for every Y in the same view whose deps include X, emit
  `{from: ns(X), to: ns(Y), lane: laneof(X)}`.
- After ingesting `planned`, for each root task R with empty deps, emit a planner fan-out
  `{from: ns(__planner__), to: ns(R), lane: "planner"}`. At the same time, add the planner node and the planner→R edge.

## gate / running backfill

Pick up and retain the `target` (gate limit) and `running` from the `task dispatch` event on the graph side.
When `--no-metrics`, carry these values in the metrics message / snapshot.metrics (running may instead come from counts.running).

---

# WebSocket messages

`GET /ws`. Every message is a JSON object with a `type`.

## server → client

### snapshot (full state on connect / on mode/session switch)
```json
{
  "type": "snapshot",
  "mode": "live" | "replay",
  "session": "events-....jsonl" | null,
  "graph": {
    "goals": [ {"id": "<goal_key>", "label": "<text>", "state": "active" | "complete"} ],
    "nodes": [ {"id": "<ns_id>", "goal": "<goal_key>", "lane": "writer",
                "state": "pending|running|done|retry|failed|stranded",
                "prompt": "...", "wall_s": null, "turns": null, "deps": ["<ns_id>", ...]} ],
    "edges": [ {"from": "<ns_id>", "to": "<ns_id>", "goal": "<goal_key>"} ]
  },
  "metrics": {"running": 0, "waiting": 0, "kv": 0.0, "tok_s": 0, "gate": null} | null,
  "replay": {"playing": false, "pos": 0, "total": 0, "speed": 1.0} | null
}
```

### patch (diff)
```json
{"type": "patch", "ops": [
  {"op": "add_goal", "id": "<goal_key>", "label": "...", "state": "active"},
  {"op": "add_node", "node": { <same shape as the snapshot node> }},
  {"op": "add_edge", "edge": {"from": "<ns_id>", "to": "<ns_id>", "goal": "<goal_key>"}},
  {"op": "set_state", "id": "<ns_id>", "state": "running", "wall_s": null, "turns": null},
  {"op": "goal_state", "id": "<goal_key>", "state": "complete", "summary": { } | null}
]}
```

### flow (particles)
```json
{"type": "flow", "flows": [ {"from": "<ns_id>", "to": "<ns_id>", "lane": "<lane>"} ]}
```

### metrics (roughly once per second; kv is a percentage in 0..100)
```json
{"type": "metrics", "running": 5, "waiting": 0, "kv": 42.0, "tok_s": 1200, "gate": 120}
```

### replay (playback state notification)
```json
{"type": "replay", "playing": true, "pos": 12, "total": 61, "speed": 2.0, "session": "events-...jsonl"}
```

### sessions (log listing. Response to a `{"type":"sessions"}` request, or on startup)
```json
{"type": "sessions", "sessions": [
  {"name": "events-....jsonl", "mtime": 1717400000.0, "size": 39336, "is_latest": true}
]}
```

## client → server

```json
{"type": "mode", "value": "live"}
{"type": "mode", "value": "replay", "session": "events-....jsonl"}
{"type": "replay", "action": "play"}
{"type": "replay", "action": "pause"}
{"type": "replay", "action": "seek", "pos": 30}
{"type": "replay", "action": "speed", "value": 2.0}
{"type": "sessions"}
```

---

# HTTP routes
- `GET /` → `static/index.html`
- `GET /static/*` → StaticFiles mount (the `static/` directory)
- `GET /api/sessions` → the sessions payload above (JSON)
- `GET /api/snapshot` → the current snapshot (JSON, for debugging)
- `WS /ws` → WebSocket

# lane → color (frontend-authoritative; the backend just passes the lane string through)
planner=#a78bfa writer=#38bdf8 reducer=#f472b6 coder=#34d399 researcher=#fbbf24
analyst=#22d3ee reviewer=#fb923c director=#f87171 worker=#60a5fa (unknown lane=#94a3b8)

# state → border color (frontend-authoritative; follows `dashboard._GLYPH`)
pending=#64748b running=#fbbf24 done=#22c55e retry=#fb923c failed=#ef4444 stranded=#991b1b
