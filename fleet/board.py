"""The blackboard — stigmergic coordination substrate.

Agents do NOT talk to each other or to a central conductor. They coordinate only
through this shared board: claim a ready task, write its result, which unlocks its
dependents. Coordination emerges from the queue + dependency state, so the system
scales to dozens of workers without a "main" agent that panics (no central context
accumulation, no serial supervision bottleneck).

Two interchangeable backends, identical public API:

  * ``Board``       — in-memory DAG queue (default; fastest, single process).
  * ``SqliteBoard`` — file-backed (WAL) DAG queue: restart-safe and multi-producer.
                      ``claim_ready`` is ATOMIC across threads *and* processes, so a
                      crashed run can be resumed by reopening the same path (tasks
                      left RUNNING by the dead process are reset to be re-claimed).

Pick a backend with ``open_board(path)`` — ``None`` ⇒ in-memory, else SQLite at
that path. ``cli``/``engine`` go through the factory so persistence is a flag.
"""
from __future__ import annotations
import json
import os
import random
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class State(str, Enum):
    PENDING = "pending"     # waiting on dependencies
    READY = "ready"         # deps satisfied, can be claimed
    RUNNING = "running"     # claimed by a worker
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    prompt: str
    deps: List[str] = field(default_factory=list)   # task ids that must be DONE first
    lane: str = "worker"                            # role/context-class (router|worker|reducer|planner)
    state: State = State.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def spec(self) -> Dict[str, Any]:
        """Picklable payload handed to a worker process."""
        return {"id": self.id, "prompt": self.prompt, "lane": self.lane, "meta": self.meta}


class Board:
    """Thread-safe DAG work queue. The scheduler is the only writer of state
    transitions; producers may add tasks at any time (even while running)."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()

    # ---- producer side --------------------------------------------------
    def add(self, task: Task) -> None:
        with self._lock:
            self._tasks[task.id] = task
            self._refresh(task)

    def add_many(self, tasks: List[Task]) -> None:
        for t in tasks:
            self.add(t)

    # ---- scheduler side -------------------------------------------------
    def _deps_done(self, t: Task) -> bool:
        return all(self._tasks.get(d) and self._tasks[d].state == State.DONE for d in t.deps)

    def _refresh(self, t: Task) -> None:
        if t.state == State.PENDING and self._deps_done(t):
            t.state = State.READY

    def claim_ready(self, n: int) -> List[Task]:
        """Atomically claim up to n READY tasks -> RUNNING."""
        with self._lock:
            for t in self._tasks.values():
                self._refresh(t)
            ready = [t for t in self._tasks.values() if t.state == State.READY][:n]
            for t in ready:
                t.state = State.RUNNING
                if t.deps:                       # inject upstream results so reducers can reduce
                    t.meta = dict(t.meta, dep_results={d: self._tasks[d].result
                                                       for d in t.deps if d in self._tasks})
            return ready

    def complete(self, tid: str, result: str) -> None:
        with self._lock:
            t = self._tasks[tid]
            t.state, t.result = State.DONE, result
            for o in self._tasks.values():       # unlock dependents
                self._refresh(o)

    def fail(self, tid: str, error: str, max_retries: int) -> bool:
        """Mark failed; requeue if retries remain. Returns True if requeued."""
        with self._lock:
            t = self._tasks[tid]
            if t.retries < max_retries:
                t.retries += 1
                t.state = State.PENDING
                self._refresh(t)
                return True
            t.state, t.error = State.FAILED, error
            return False

    # ---- introspection --------------------------------------------------
    def counts(self) -> Dict[str, int]:
        with self._lock:
            c = {s.value: 0 for s in State}
            for t in self._tasks.values():
                c[t.state.value] += 1
            return c

    def unfinished(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values()
                       if t.state not in (State.DONE, State.FAILED))

    def has_ready(self) -> bool:
        with self._lock:
            for t in self._tasks.values():
                self._refresh(t)
            return any(t.state == State.READY for t in self._tasks.values())

    def results(self) -> Dict[str, Task]:
        with self._lock:
            return dict(self._tasks)

    def close(self) -> None:
        """No-op: the in-memory board owns no OS resources. Present so callers can
        ``board.close()`` uniformly regardless of backend (parity with SqliteBoard)."""
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SQLite / file-backed board (restart-safe, multi-producer, cross-process-atomic)
# ─────────────────────────────────────────────────────────────────────────────
class SqliteBoard:
    """File-backed DAG queue with the SAME public API as :class:`Board`.

    WHY a second backend (vs. just the in-memory Board):
      * Restart-safety (BUILD_SPEC §G): the file IS the state, so a mid-run kill
        loses no completed work. On reopen of a *dead* board, tasks left RUNNING by
        the dead process are reset so the remaining work resumes — no caller
        bookkeeping needed. The durable unit on disk is the ``{.db,-wal,-shm}`` set;
        ``close()`` checkpoints the WAL back into the main file (fix #4).
      * Multi-producer / multi-process: several engines or producers can share one
        board file; ``claim_ready`` is atomic across threads *and* processes so two
        claimers never hand out the same task.

    Atomicity mechanics (the load-bearing part):
      * WAL journal + ``busy_timeout`` lets readers and one writer proceed without
        "database is locked" under contention. On top of that, every transactional
        body is wrapped in a bounded retry-on-locked helper (fix #3) so a burst that
        outlasts ``busy_timeout`` is retried with jittered backoff instead of crashing
        a worker thread.
      * ``claim_ready`` runs inside ``BEGIN IMMEDIATE`` (grabs the write lock up
        front, before any other writer) and flips ``ready -> running`` with a single
        ``UPDATE ... RETURNING`` (SQLite ≥3.35). One winner per row, period.
      * ``complete``/``fail`` guard their write with ``WHERE id=? AND state='running'``
        and check ``rowcount`` (fix #1): a stale/duplicate claimer (e.g. one produced
        by a crash-recovery reset on another opener) can no longer overwrite a row it
        no longer legitimately owns; the lost claim is signalled, not silently applied.
      * Each thread gets its OWN connection (sqlite3 connections are not safe to
        share across threads); all live connections are tracked so ``close()`` shuts
        every one down (fix #5), not just the caller's.

    Liveness / crash recovery (fix #2): the owner records its pid + a wall-clock
    heartbeat in ``board_meta`` and refreshes it from a daemon thread. A *new* opener
    resets RUNNING rows ONLY when the prior owner is provably dead (stale heartbeat /
    different-and-not-running pid) — so a second engine attaching to a LIVE board no
    longer clobbers the in-flight rows of the running fleet.

    State on disk mirrors :class:`Task` exactly so ``results()`` rebuilds identical
    Task objects (same fields, same dep_results-injection contract on claim).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS tasks (
        id       TEXT PRIMARY KEY,
        prompt   TEXT NOT NULL,
        deps     TEXT NOT NULL DEFAULT '[]',   -- json list of upstream ids
        lane     TEXT NOT NULL DEFAULT 'worker',
        state    TEXT NOT NULL DEFAULT 'pending',
        result   TEXT,
        error    TEXT,
        retries  INTEGER NOT NULL DEFAULT 0,
        meta     TEXT NOT NULL DEFAULT '{}',   -- json object
        seq      INTEGER                       -- insertion order (claim fairness/determinism)
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
    CREATE TABLE IF NOT EXISTS board_meta (k TEXT PRIMARY KEY, v TEXT);
    """

    # board_meta keys for the owner liveness record (fix #2).
    _OWNER_PID_KEY = "owner_pid"
    _OWNER_HB_KEY = "owner_heartbeat"   # wall-clock seconds (time.time) of last refresh

    def __init__(self, path: str, *, reset_running: Optional[bool] = None) -> None:
        self.path = os.fspath(path)
        # busy_timeout (ms): how long a connection waits on the write lock before
        # raising "database is locked"; covers cross-process/thread claim contention.
        self._busy_ms = int(os.environ.get("FLEET_BOARD_BUSY_MS", "10000"))
        # Bounded retry-on-locked (fix #3): if busy_timeout is exhausted under a
        # contention burst sqlite raises OperationalError("database is locked"); we
        # retry a few times with jittered backoff before giving up.
        self._lock_retries = int(os.environ.get("FLEET_BOARD_LOCK_RETRIES", "8"))
        self._lock_backoff = float(os.environ.get("FLEET_BOARD_LOCK_BACKOFF", "0.05"))
        # Heartbeat freshness (fix #2): a prior owner whose heartbeat is older than
        # this (seconds) is treated as dead, so its RUNNING rows are recovered.
        self._hb_interval = float(os.environ.get("FLEET_BOARD_HB_INTERVAL", "5.0"))
        self._hb_stale = float(os.environ.get("FLEET_BOARD_HB_STALE", "20.0"))
        # WAL checkpoint cadence (fix #4): keep the -wal sidecar from growing without
        # bound so "the file IS the state" stays close to true between explicit closes.
        self._ckpt_every = float(os.environ.get("FLEET_BOARD_CKPT_S", "30.0"))

        self._tls = threading.local()          # one sqlite3 connection per thread
        self._seq_lock = threading.Lock()      # guards in-process insertion counter
        # fix #5: track EVERY per-thread connection so close() can shut them all,
        # not just the calling thread's. Guarded by its own lock.
        self._conns_lock = threading.Lock()
        self._conns: List[sqlite3.Connection] = []
        self._closed = False
        self.started_at = time.time()

        # One-time DB init on this (ctor-thread) connection. _conn() registers it in
        # self._conns and stows it on self._tls, so the ctor thread REUSES this one
        # connection for later calls (fix #5: no second ctor-thread connection).
        conn = self._conn()
        self._run_locked(lambda c: c.executescript(self._SCHEMA), conn=conn)

        # Decide whether to recover RUNNING rows. Default (reset_running is None):
        # recover only a provably-DEAD prior owner; never clobber a LIVE one (fix #2).
        # Explicit True/False overrides for tests/special callers.
        do_reset = self._should_reset(conn) if reset_running is None else bool(reset_running)
        if do_reset:
            # Resume semantics: a previous owner died mid-flight leaving rows in
            # RUNNING. They were never completed, so reset them to PENDING and let the
            # normal ready-refresh re-promote the ones whose deps are satisfied. Done
            # in one IMMEDIATE txn so a concurrent live claimer can't race the reset.
            def _reset(c: sqlite3.Connection) -> None:
                c.execute("BEGIN IMMEDIATE")
                c.execute("UPDATE tasks SET state='pending' WHERE state='running'")
                self._refresh_all(c)
            self._run_locked(_reset, conn=conn)

        # Claim ownership: stamp our pid + a fresh heartbeat, then keep it warm from a
        # daemon thread (also periodically checkpoints the WAL — fix #4).
        self._claim_ownership(conn)
        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, name="board-heartbeat", daemon=True)
        self._hb_thread.start()

        # Continue the seq counter past whatever is already on disk (multi-run append).
        row = conn.execute("SELECT COALESCE(MAX(seq), -1) FROM tasks").fetchone()
        self._next_seq = int(row[0]) + 1

    # ---- connection management ------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None ⇒ autocommit; we drive transactions explicitly with
        # BEGIN IMMEDIATE so the claim path takes the write lock atomically.
        c = sqlite3.connect(self.path, timeout=self._busy_ms / 1000.0,
                            isolation_level=None, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")        # WAL-durable enough, much faster
        c.execute(f"PRAGMA busy_timeout={self._busy_ms}")
        c.execute("PRAGMA foreign_keys=ON")
        c.row_factory = sqlite3.Row
        return c

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = self._connect()
            self._tls.conn = c
            # fix #5: register in the shared list so close() reaches every thread's conn.
            with self._conns_lock:
                self._conns.append(c)
        return c

    # ---- retry-on-locked helper (fix #3) --------------------------------
    @staticmethod
    def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg

    def _run_locked(self, body, *, conn: Optional[sqlite3.Connection] = None):
        """Run ``body(conn)`` with bounded retry on "database is locked"/"busy".

        The body runs inside ``with conn:`` — the same commit-on-success /
        rollback-on-exception envelope the original code relied on — so a body that
        opens ``BEGIN IMMEDIATE`` is COMMITTED when it returns, and a single autocommit
        statement is fine too. On a lock error (busy_timeout exhausted under a
        contention burst, fix #3) we roll back any half-open txn and retry with
        jittered exponential backoff; after ``_lock_retries`` attempts we re-raise so a
        genuine deadlock/corruption is never swallowed.
        """
        c = conn if conn is not None else self._conn()
        attempt = 0
        while True:
            try:
                with c:                       # commit on success, rollback on exception
                    return body(c)
            except sqlite3.OperationalError as exc:
                if not self._is_locked_error(exc) or attempt >= self._lock_retries:
                    raise
                # `with c:` already rolled back on the way out; belt-and-suspenders in
                # case a partial BEGIN IMMEDIATE outlived it, so the retry starts clean.
                try:
                    if c.in_transaction:
                        c.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                # Jittered exponential backoff (cap the exponent so the wait stays sane).
                sleep = self._lock_backoff * (2 ** min(attempt, 5))
                time.sleep(sleep * (0.5 + random.random()))
                attempt += 1

    # ---- owner liveness / crash recovery (fix #2) -----------------------
    def _read_owner(self, conn: sqlite3.Connection):
        rows = {
            r["k"]: r["v"]
            for r in conn.execute(
                "SELECT k, v FROM board_meta WHERE k IN (?, ?)",
                (self._OWNER_PID_KEY, self._OWNER_HB_KEY),
            )
        }
        pid = rows.get(self._OWNER_PID_KEY)
        hb = rows.get(self._OWNER_HB_KEY)
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        try:
            hb = float(hb) if hb is not None else None
        except (TypeError, ValueError):
            hb = None
        return pid, hb

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Best-effort: is `pid` a live process on THIS host? Signal 0 probes without
        killing. EPERM means it exists but is owned by another user (alive). On any
        platform where this is unreliable we conservatively fall back to the heartbeat
        staleness check, so a stuck-but-stale owner is still recovered."""
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            # Unknown errno (e.g. non-POSIX) — don't trust pid alone; let heartbeat decide.
            return True

    def _should_reset(self, conn: sqlite3.Connection) -> bool:
        """Reset RUNNING rows ONLY if the prior owner is provably dead (fix #2).

        Dead ⇔ (no owner recorded) OR (heartbeat is stale) OR (the recorded pid is the
        SAME host and not alive). A live owner with a fresh heartbeat from a different
        process is left untouched, so a second engine never clobbers an in-flight fleet.
        """
        pid, hb = self._read_owner(conn)
        if pid is None and hb is None:
            return True                      # first-ever / clean open: nothing to protect
        now = time.time()
        if hb is not None and (now - hb) > self._hb_stale:
            return True                      # heartbeat went cold → prior owner crashed
        # Heartbeat is fresh-ish. If it is OUR own pid from a previous in-process open,
        # or a live foreign pid, treat the board as LIVE and do NOT reset.
        if pid is not None and pid != os.getpid() and not self._pid_alive(pid):
            return True                      # recorded pid is dead on this host
        return False

    def _claim_ownership(self, conn: sqlite3.Connection) -> None:
        def _body(c: sqlite3.Connection) -> None:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                "INSERT INTO board_meta (k, v) VALUES (?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (self._OWNER_PID_KEY, str(os.getpid())),
            )
            c.execute(
                "INSERT INTO board_meta (k, v) VALUES (?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (self._OWNER_HB_KEY, repr(time.time())),
            )
        self._run_locked(_body, conn=conn)

    def _touch_heartbeat(self, conn: sqlite3.Connection) -> None:
        self._run_locked(
            lambda c: c.execute(
                "INSERT INTO board_meta (k, v) VALUES (?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (self._OWNER_HB_KEY, repr(time.time())),
            ),
            conn=conn,
        )

    def _release_ownership(self, conn: sqlite3.Connection) -> None:
        """Clear the owner record on a graceful close so the NEXT opener sees "no live
        owner" and correctly recovers any rows still RUNNING at shutdown (fix #2). A
        graceful close that leaves RUNNING rows is, for those rows, indistinguishable
        from a crash — the work was claimed but its holder is gone — so the reopener
        must reset them. Without this, a same-process reopen would see our own (now
        dead) pid as 'alive' and wrongly protect orphaned RUNNING rows."""
        self._run_locked(
            lambda c: c.execute(
                "DELETE FROM board_meta WHERE k IN (?, ?)",
                (self._OWNER_PID_KEY, self._OWNER_HB_KEY),
            ),
            conn=conn,
        )

    def _heartbeat_loop(self) -> None:
        """Daemon: refresh the ownership heartbeat and periodically checkpoint the WAL
        (fix #2 keeps the board provably-live; fix #4 keeps the -wal sidecar bounded).
        Uses its OWN connection (registered for close()) — never touches another
        thread's conn. All errors are swallowed: a missed heartbeat just shortens our
        apparent liveness window, it must never crash the fleet."""
        try:
            conn = self._conn()
        except Exception:
            return
        last_ckpt = 0.0
        while not self._hb_stop.wait(self._hb_interval):
            try:
                self._touch_heartbeat(conn)
            except Exception:
                pass
            now = time.time()
            if self._ckpt_every > 0 and (now - last_ckpt) >= self._ckpt_every:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                last_ckpt = now

    def close(self) -> None:
        """Shut the board down: stop the heartbeat, checkpoint the WAL back into the
        main .db (fix #4), and close EVERY per-thread connection (fix #5) — not just
        the caller's. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Stop the heartbeat thread first so it can't reopen/touch a closing connection.
        stop = getattr(self, "_hb_stop", None)
        if stop is not None:
            stop.set()
        hb = getattr(self, "_hb_thread", None)
        if hb is not None and hb is not threading.current_thread():
            hb.join(timeout=max(0.1, self._hb_interval * 2))
        # Release ownership so a reopener recovers any rows still RUNNING (fix #2),
        # then final-checkpoint so the durable state lives in the .db, not only the
        # -wal (fix #4). Both best-effort: a close must never raise.
        try:
            self._release_ownership(self._conn())
        except Exception:
            pass
        try:
            self._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        # Close all tracked connections. sqlite3.Connection.close() is safe to call
        # from any thread once the conn is otherwise idle.
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        # Drop the calling thread's stale handle so a post-close _conn() (if any)
        # would build a fresh one rather than reuse a closed connection.
        try:
            self._tls.conn = None
        except Exception:
            pass

    def __del__(self):  # best-effort safety net; explicit close() is preferred.
        try:
            self.close()
        except Exception:
            pass

    # ---- (de)serialization ----------------------------------------------
    @staticmethod
    def _row_to_task(r: sqlite3.Row) -> Task:
        return Task(
            id=r["id"], prompt=r["prompt"],
            deps=json.loads(r["deps"]) if r["deps"] else [],
            lane=r["lane"], state=State(r["state"]),
            result=r["result"], error=r["error"], retries=r["retries"],
            meta=json.loads(r["meta"]) if r["meta"] else {},
        )

    def _alloc_seq(self) -> int:
        with self._seq_lock:
            s = self._next_seq
            self._next_seq += 1
            return s

    # ---- ready-state maintenance (mirrors Board._refresh) ---------------
    @staticmethod
    def _refresh_all(conn: sqlite3.Connection) -> None:
        """Promote every PENDING task whose deps are all DONE to READY.

        Set-based equivalent of Board._refresh run over the whole table: a pending
        task is ready iff it has zero deps still un-DONE. Called inside the caller's
        write transaction so the promotion is part of the same atomic step.
        """
        conn.execute(
            """
            UPDATE tasks SET state='ready'
            WHERE state='pending'
              AND NOT EXISTS (
                  SELECT 1 FROM json_each(tasks.deps) d
                  LEFT JOIN tasks p ON p.id = d.value
                  WHERE p.id IS NULL OR p.state <> 'done'
              )
            """
        )

    # ---- producer side --------------------------------------------------
    def add(self, task: Task) -> None:
        # fix #6: task DEFS are immutable. ON CONFLICT(id) DO NOTHING so a resubmit of
        # an already-known id never clobbers its state/result/error/retries (a duplicate
        # add must NOT wipe a completed task's result). Matches in-mem Board's practical
        # contract that ids are unique; a re-add is a no-op rather than a reset.
        def _body(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO tasks (id, prompt, deps, lane, state, result, error,
                                      retries, meta, seq)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO NOTHING""",
                (task.id, task.prompt, json.dumps(task.deps), task.lane,
                 task.state.value, task.result, task.error, task.retries,
                 json.dumps(task.meta), self._alloc_seq()),
            )
            # New task may itself be ready, or unblock nothing yet — refresh all.
            self._refresh_all(conn)
        self._run_locked(_body)

    def add_many(self, tasks: List[Task]) -> None:
        # One transaction for the whole batch: fewer fsyncs and a single refresh.
        def _body(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                """INSERT INTO tasks (id, prompt, deps, lane, state, result, error,
                                      retries, meta, seq)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO NOTHING""",   # fix #6: immutable task defs (see add())
                [(t.id, t.prompt, json.dumps(t.deps), t.lane, t.state.value,
                  t.result, t.error, t.retries, json.dumps(t.meta), self._alloc_seq())
                 for t in tasks],
            )
            self._refresh_all(conn)
        self._run_locked(_body)

    # ---- scheduler side -------------------------------------------------
    def claim_ready(self, n: int) -> List[Task]:
        """Atomically claim up to n READY tasks -> RUNNING (cross-thread + process).

        ``BEGIN IMMEDIATE`` takes the write lock before anyone else, so two claimers
        (threads or processes) serialize here and never hand out the same row. We
        first promote freshly-unblocked tasks, then flip exactly ``n`` ready ids to
        running with ``UPDATE ... RETURNING`` (single statement, one winner per row).
        Upstream ``dep_results`` are injected into each claimed task's meta exactly
        like the in-memory Board, so reducers receive real upstream output.
        """
        if n <= 0:
            return []

        def _body(conn: sqlite3.Connection) -> List[Task]:
            conn.execute("BEGIN IMMEDIATE")
            self._refresh_all(conn)
            # Pick the oldest-inserted ready ids (deterministic, FIFO-ish fairness).
            rows = conn.execute(
                "SELECT id FROM tasks WHERE state='ready' ORDER BY seq LIMIT ?",
                (n,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            claimed_rows = conn.execute(
                f"UPDATE tasks SET state='running' WHERE id IN ({placeholders}) "
                f"AND state='ready' RETURNING *",
                ids,
            ).fetchall()
            tasks = [self._row_to_task(r) for r in claimed_rows]
            # Inject upstream results so reducers can reduce (same contract as Board).
            for t in tasks:
                if t.deps:
                    dep_rows = conn.execute(
                        f"SELECT id, result FROM tasks WHERE id IN "
                        f"({','.join('?' * len(t.deps))})",
                        t.deps,
                    ).fetchall()
                    t.meta = dict(t.meta, dep_results={r["id"]: r["result"] for r in dep_rows})
                    conn.execute("UPDATE tasks SET meta=? WHERE id=?",
                                 (json.dumps(t.meta), t.id))
            return tasks

        return self._run_locked(_body)

    def complete(self, tid: str, result: str) -> bool:
        """Mark a RUNNING task DONE. Returns True if the write landed, False if it was
        a LOST/STALE claim (the row was not RUNNING — e.g. a crash-recovery reset on
        another opener already requeued it, or a duplicate claimer is finishing late).

        Fix #1: the UPDATE is guarded with ``WHERE id=? AND state='running'`` and we
        check ``rowcount``. A blind UPDATE would let a stale claimer overwrite a row
        that has since been requeued/reclaimed/completed by someone else, corrupting
        state. On a lost claim we make NO change and signal it so the engine can drop
        the stale result instead of trusting it. (In-mem Board.complete keeps its
        ``None`` return; engine callers ignore the value, so parity is preserved.)
        """
        def _body(conn: sqlite3.Connection) -> bool:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE tasks SET state='done', result=? WHERE id=? AND state='running'",
                (result, tid))
            if cur.rowcount == 0:
                # Lost/stale claim: do not corrupt. Roll back the empty txn explicitly.
                conn.execute("ROLLBACK")
                return False
            self._refresh_all(conn)          # unlock dependents
            return True

        ok = self._run_locked(_body)
        if not ok:
            # Visible but non-fatal: helps debugging duplicate/stale claimers in prod.
            print(f"[board] complete() ignored stale/lost claim for task {tid!r} "
                  f"(row was not RUNNING)", file=sys.stderr)
        return ok

    def fail(self, tid: str, error: str, max_retries: int) -> bool:
        """Mark failed; requeue if retries remain. Returns True if requeued, False if
        permanently failed OR if this was a lost/stale claim (no RUNNING row to fail).

        Fix #1: like complete(), all writes are guarded with
        ``WHERE id=? AND state='running'`` and checked via ``rowcount`` so a stale
        claimer cannot resurrect/over-retry a row that is no longer RUNNING. The
        retry-count read happens INSIDE the same IMMEDIATE txn as the guarded write,
        so the decision and the mutation are atomic.
        """
        def _body(conn: sqlite3.Connection) -> bool:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT retries FROM tasks WHERE id=? AND state='running'", (tid,)
            ).fetchone()
            if row is None:
                # Not RUNNING (missing, or already requeued/done by another owner):
                # lost/stale claim → make no change. Returning False (not requeued)
                # matches the "permanent / nothing-to-do" branch for the caller.
                conn.execute("ROLLBACK")
                return False
            if row["retries"] < max_retries:
                conn.execute(
                    "UPDATE tasks SET retries=retries+1, state='pending', error=NULL "
                    "WHERE id=? AND state='running'", (tid,))
                self._refresh_all(conn)      # may be immediately re-ready
                return True
            conn.execute("UPDATE tasks SET state='failed', error=? WHERE id=? "
                         "AND state='running'", (error, tid))
            return False

        return self._run_locked(_body)

    # ---- introspection --------------------------------------------------
    def counts(self) -> Dict[str, int]:
        conn = self._conn()
        c = {s.value: 0 for s in State}
        for r in conn.execute("SELECT state, COUNT(*) AS n FROM tasks GROUP BY state"):
            c[r["state"]] = r["n"]
        return c

    def unfinished(self) -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE state NOT IN ('done','failed')"
        ).fetchone()
        return int(row["n"])

    def has_ready(self) -> bool:
        def _body(conn: sqlite3.Connection) -> bool:
            conn.execute("BEGIN IMMEDIATE")
            self._refresh_all(conn)
            row = conn.execute(
                "SELECT 1 FROM tasks WHERE state='ready' LIMIT 1").fetchone()
            return row is not None
        return self._run_locked(_body)

    def results(self) -> Dict[str, Task]:
        conn = self._conn()
        out: Dict[str, Task] = {}
        for r in conn.execute("SELECT * FROM tasks ORDER BY seq"):
            out[r["id"]] = self._row_to_task(r)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory — cli/engine call this so persistence is a single flag.
# ─────────────────────────────────────────────────────────────────────────────
BoardLike = Union[Board, SqliteBoard]


def open_board(path: Optional[str]) -> BoardLike:
    """Return a Board-like backend.

    ``path is None`` ⇒ fast in-memory :class:`Board` (default, no persistence).
    Otherwise a :class:`SqliteBoard` at ``path`` — created if absent, RESUMED if it
    already exists. Resume is now LIVENESS-GATED (fix #2): RUNNING rows are reset only
    when the prior owner is provably dead (stale heartbeat / dead pid). A second engine
    attaching to a *live* board does NOT clobber the running fleet's in-flight rows.
    Both backends expose the identical public API, so callers never branch.
    """
    if path is None:
        return Board()
    # reset_running=None ⇒ the liveness-gated default (recover dead, protect live).
    return SqliteBoard(path)
