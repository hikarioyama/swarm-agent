"""SqliteBoard self-test — run with the HermesAgent venv python:

    cd ~/projects/step37-harness
    PYTHONPATH=. /home/hikari/.hermes/hermes-agent/venv/bin/python tests/test_board_sqlite.py

Validates the file-backed board independently of any server:

  1. DAG flow + dep_results: add (a; b deps=[a]); only a is ready; claim+complete a;
     b becomes ready and, when claimed, receives a's result via meta['dep_results'].
  2. claim atomicity: claiming the same task twice never double-hands-out a row.
  3. fail/requeue + permanent-fail honour max_retries.
  4. Restart-safety: a task left RUNNING by a "crashed" board object is reset on
     reopen of the same file so the remaining work resumes (and prior DONE persists).
  5. open_board() factory: None -> in-memory Board, path -> SqliteBoard.
  6. Concurrent multi-thread claim: N threads claiming never overlap (cross-thread
     atomicity), proving the BEGIN IMMEDIATE + UPDATE...RETURNING claim is exclusive.

Review-fix regression cases (board.py hardening):
  7. (fix #1) complete()/fail() on a NON-running row is a no-op lost-claim (rowcount 0):
     a stale/duplicate claimer can no longer overwrite a row it no longer owns.
  8. (fix #2) liveness-gated recovery: a LIVE-owner board is NOT reset by a second
     opener, while a DEAD (stale-heartbeat) board IS recovered.
  9. (fix #3) concurrent writers retry past a transient "database is locked" instead of
     crashing — many threads hammering complete()/add() all succeed.
 10. (fix #4) close() checkpoints the WAL (TRUNCATE) so the -wal sidecar shrinks and the
     durable state lives in the .db.
"""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, ".")
from fleet.board import Board, SqliteBoard, Task, State, open_board  # noqa: E402


def test_dag_and_dep_results(path):
    board = SqliteBoard(path)
    board.add_many([
        Task(id="a", prompt="produce A"),
        Task(id="b", prompt="reduce using A", deps=["a"]),
    ])
    # only a is ready (b waits on a)
    assert board.has_ready()
    assert board.counts()["ready"] == 1, board.counts()
    assert board.counts()["pending"] == 1, board.counts()

    claimed = board.claim_ready(10)
    assert [t.id for t in claimed] == ["a"], claimed
    assert claimed[0].state == State.RUNNING
    # b must NOT be claimable yet
    assert board.claim_ready(10) == [], "b leaked before a completed"

    board.complete("a", "RESULT_OF_A")
    assert board.counts()["done"] == 1, board.counts()
    assert board.has_ready(), "b did not become ready after a completed"

    bclaim = board.claim_ready(10)
    assert [t.id for t in bclaim] == ["b"], bclaim
    # dep_results injected exactly like the in-memory Board
    dep = (bclaim[0].meta or {}).get("dep_results") or {}
    assert dep.get("a") == "RESULT_OF_A", f"dep_results not injected: {bclaim[0].meta}"
    board.complete("b", "RESULT_OF_B")
    assert board.unfinished() == 0, board.counts()

    # results() rebuilds identical Task objects with persisted fields
    res = board.results()
    assert res["a"].result == "RESULT_OF_A" and res["b"].result == "RESULT_OF_B"
    assert res["b"].deps == ["a"]
    board.close()
    print("  [1] DAG flow + dep_results injection OK:", board.counts())


def test_no_double_claim(path):
    board = SqliteBoard(path)
    board.add_many([Task(id=f"t{i}", prompt="x") for i in range(5)])
    first = {t.id for t in board.claim_ready(3)}
    second = {t.id for t in board.claim_ready(10)}
    assert len(first) == 3 and len(second) == 2, (first, second)
    assert first.isdisjoint(second), f"a task was claimed twice: {first & second}"
    assert board.claim_ready(10) == [], "claimed more than exist"
    board.close()
    print("  [2] no-double-claim OK:", sorted(first), sorted(second))


def test_fail_and_requeue(path):
    board = SqliteBoard(path)
    board.add(Task(id="f", prompt="will fail"))
    board.claim_ready(1)
    assert board.fail("f", "boom", max_retries=1) is True, "should requeue (retry left)"
    assert board.counts()["pending"] == 0 and board.counts()["ready"] == 1, board.counts()
    board.claim_ready(1)
    assert board.fail("f", "boom again", max_retries=1) is False, "should permanently fail"
    assert board.counts()["failed"] == 1, board.counts()
    assert board.results()["f"].error == "boom again"
    board.close()
    print("  [3] fail/requeue honours max_retries OK:", board.counts())


def test_restart_safety(path):
    # First board: complete a, leave b RUNNING (simulate a crash mid-task).
    b1 = SqliteBoard(path)
    b1.add_many([
        Task(id="a", prompt="A"),
        Task(id="b", prompt="B deps a", deps=["a"]),
        Task(id="c", prompt="independent"),
    ])
    b1.complete(*("a", "RESULT_A")) if False else None  # (a not claimed yet; complete via claim path)
    claimed_a = b1.claim_ready(1)
    assert [t.id for t in claimed_a] == ["a"]
    b1.complete("a", "RESULT_A")
    # now b is ready; claim it and DON'T complete -> it is left RUNNING
    left = b1.claim_ready(10)
    running_ids = {t.id for t in left}
    assert "b" in running_ids and "c" in running_ids, running_ids
    counts_before = b1.counts()
    assert counts_before["running"] == 2, counts_before  # b, c left in flight
    assert counts_before["done"] == 1, counts_before
    # SIMULATE CRASH: drop the object WITHOUT completing b/c. (Files remain on disk.)
    b1.close()
    del b1

    # Reopen the SAME path: a new process/board resumes. RUNNING -> reset; DONE persists.
    b2 = SqliteBoard(path)
    counts_after = b2.counts()
    assert counts_after["done"] == 1, f"completed work lost on reopen: {counts_after}"
    assert counts_after["running"] == 0, f"crashed RUNNING not reset: {counts_after}"
    assert counts_after["ready"] == 2, f"b,c not re-readied for resume: {counts_after}"
    # The remaining work is claimable again, and b STILL gets a's persisted result.
    resumed = {t.id: t for t in b2.claim_ready(10)}
    assert set(resumed) == {"b", "c"}, resumed
    assert (resumed["b"].meta.get("dep_results") or {}).get("a") == "RESULT_A", \
        f"dep_results lost across restart: {resumed['b'].meta}"
    b2.complete("b", "RESULT_B")
    b2.complete("c", "RESULT_C")
    assert b2.unfinished() == 0, b2.counts()
    b2.close()
    print("  [4] restart-safety OK: resumed", sorted(resumed), "| reopen counts", counts_after)


def test_factory(path):
    assert isinstance(open_board(None), Board), "None must give in-memory Board"
    sb = open_board(path)
    assert isinstance(sb, SqliteBoard), "path must give SqliteBoard"
    assert os.path.exists(path), "SqliteBoard did not create its file"
    sb.close()
    print("  [5] open_board factory OK (None->Board, path->SqliteBoard)")


def test_concurrent_claim(path):
    """Many threads claim simultaneously; cross-thread atomicity => no overlap."""
    board = SqliteBoard(path)
    board.add_many([Task(id=f"j{i}", prompt="x") for i in range(200)])
    seen = []
    seen_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def claimer():
        barrier.wait()                       # all fire at once for max contention
        mine = []
        while True:
            got = board.claim_ready(7)
            if not got:
                break
            mine.extend(t.id for t in got)
        with seen_lock:
            seen.extend(mine)

    threads = [threading.Thread(target=claimer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    assert len(seen) == 200, f"claimed {len(seen)} != 200 (lost or duplicated)"
    assert len(set(seen)) == 200, f"duplicate claims under contention: {len(seen) - len(set(seen))} dupes"
    board.close()
    print(f"  [6] concurrent claim OK: 8 threads claimed all 200 tasks, zero overlap")


def test_lost_claim_noop(path):
    """(fix #1) complete()/fail() guard on state='running'; a stale claimer that calls
    them on a row that is NOT running makes NO change and signals the lost claim."""
    board = SqliteBoard(path)
    board.add(Task(id="x", prompt="x"))
    board.claim_ready(1)                       # x -> running
    assert board.complete("x", "GOOD") is True, "first complete should land"
    assert board.results()["x"].result == "GOOD"
    assert board.counts()["done"] == 1, board.counts()

    # A duplicate/stale claimer finishing late: x is DONE, not running -> must NOT corrupt.
    assert board.complete("x", "STALE_OVERWRITE") is False, "stale complete must be a no-op"
    assert board.results()["x"].result == "GOOD", "stale complete corrupted a done row!"
    assert board.counts()["done"] == 1, board.counts()

    # fail() on the same non-running row is likewise a no-op (no resurrection / over-retry).
    assert board.fail("x", "stale", max_retries=5) is False, "stale fail must be a no-op"
    assert board.results()["x"].state == State.DONE and board.results()["x"].error is None
    # And fail() on a totally unknown id is a clean False, not a crash.
    assert board.fail("nonexistent", "boom", max_retries=5) is False
    board.close()
    print("  [7] lost/stale claim no-op (rowcount guard) OK:", board.counts())


def test_liveness_gated_recovery(path):
    """(fix #2) A SECOND opener must NOT reset a LIVE board's RUNNING rows, but MUST
    recover a DEAD (stale-heartbeat) one. We drive the heartbeat staleness window to a
    tiny value so the 'dead' case is testable without really killing a process."""
    # Tight heartbeat window so a paused owner looks dead fast.
    os.environ["FLEET_BOARD_HB_STALE"] = "0.6"
    os.environ["FLEET_BOARD_HB_INTERVAL"] = "0.2"
    try:
        # --- LIVE case: owner b1 is alive & heartbeating; b2 attaches, must NOT reset.
        b1 = SqliteBoard(path)
        b1.add_many([Task(id="r1", prompt="x"), Task(id="r2", prompt="x")])
        running = {t.id for t in b1.claim_ready(2)}
        assert running == {"r1", "r2"}, running
        assert b1.counts()["running"] == 2, b1.counts()
        # b1 stays open (heartbeat thread keeps the record fresh). Second opener:
        b2 = SqliteBoard(path)   # reset_running=None -> liveness-gated default
        assert b2.counts()["running"] == 2, \
            f"LIVE board clobbered by 2nd opener (in-flight rows lost!): {b2.counts()}"
        b2.close()
        # b1 can still legitimately complete its in-flight work.
        assert b1.complete("r1", "R1") is True
        b1.close()
        print("  [8a] live board NOT reset by 2nd opener OK")

        # --- DEAD case: a board left RUNNING whose heartbeat goes stale IS recovered.
        path2 = path + ".dead"
        d1 = SqliteBoard(path2)
        d1.add_many([Task(id="d1", prompt="x"), Task(id="d2", prompt="x")])
        assert {t.id for t in d1.claim_ready(2)} == {"d1", "d2"}
        # Simulate a CRASH: stop the heartbeat WITHOUT clearing ownership (no graceful
        # close/release), so the record stays but goes stale -> looks dead.
        d1._hb_stop.set()
        d1._hb_thread.join(timeout=2.0)
        time.sleep(0.8)                        # exceed HB_STALE=0.6s
        d2 = SqliteBoard(path2)                # opener sees a stale heartbeat -> recover
        assert d2.counts()["running"] == 0, f"dead board NOT recovered: {d2.counts()}"
        assert d2.counts()["ready"] == 2, f"dead board's work not re-readied: {d2.counts()}"
        d2.close()
        # Best-effort close the crashed handle's connections (no release needed).
        d1.close()
        print("  [8b] dead board IS recovered by opener OK")
    finally:
        os.environ.pop("FLEET_BOARD_HB_STALE", None)
        os.environ.pop("FLEET_BOARD_HB_INTERVAL", None)


def test_concurrent_writers_retry(path):
    """(fix #3) Many threads issue write transactions at once. busy_timeout + the
    retry-on-locked helper mean every write lands; none raises OperationalError."""
    board = SqliteBoard(path)
    n = 60
    board.add_many([Task(id=f"w{i}", prompt="x") for i in range(n)])
    # Claim everything so each id is RUNNING and completable by exactly one writer.
    claimed = []
    while True:
        got = board.claim_ready(100)
        if not got:
            break
        claimed.extend(t.id for t in got)
    assert len(claimed) == n, len(claimed)

    errors = []
    err_lock = threading.Lock()
    barrier = threading.Barrier(12)

    def writer(ids):
        barrier.wait()                          # fire together for max lock contention
        for tid in ids:
            try:
                # each thread also adds a fresh task to stir extra write contention
                board.add(Task(id=f"extra-{tid}", prompt="x"))
                board.complete(tid, f"R-{tid}")
            except Exception as e:              # OperationalError would land here
                with err_lock:
                    errors.append(repr(e))

    # Partition the ids across 12 threads (disjoint, so each complete() is a real owner).
    chunks = [claimed[i::12] for i in range(12)]
    threads = [threading.Thread(target=writer, args=(c,)) for c in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30.0)

    assert not errors, f"writers hit uncaught lock errors: {errors[:3]}"
    done = board.counts()["done"]
    assert done == n, f"not all completes landed under contention: done={done} != {n}"
    board.close()
    print(f"  [9] concurrent writers retry past transient lock OK: {done}/{n} done, 0 errors")


def test_close_checkpoints_wal(path):
    """(fix #4) close() runs wal_checkpoint(TRUNCATE), shrinking the -wal sidecar so the
    durable unit collapses back into the .db. We write enough to grow the WAL, then
    assert close() truncates it near-zero."""
    board = SqliteBoard(path)
    # Many separate write txns inflate the -wal file.
    board.add_many([Task(id=f"c{i}", prompt="x" * 200) for i in range(300)])
    for i in range(0, 60):
        board.claim_ready(1)
        board.complete(f"c{i}", "y" * 200)
    wal = path + "-wal"
    wal_before = os.path.getsize(wal) if os.path.exists(wal) else 0
    board.close()
    wal_after = os.path.getsize(wal) if os.path.exists(wal) else 0
    # TRUNCATE drives the -wal to 0 bytes (or removes it); accept a tiny residual.
    assert wal_after <= 4096, f"close() did not checkpoint/truncate WAL: {wal_after} bytes"
    # Durable state survived the checkpoint: reopen sees the completed work.
    b2 = SqliteBoard(path)
    assert b2.counts()["done"] == 60, b2.counts()
    b2.close()
    print(f"  [10] close() checkpoints WAL OK: -wal {wal_before} -> {wal_after} bytes")


if __name__ == "__main__":
    print("sqlite board self-test:")
    with tempfile.TemporaryDirectory() as d:
        test_dag_and_dep_results(os.path.join(d, "dag.db"))
        test_no_double_claim(os.path.join(d, "claim.db"))
        test_fail_and_requeue(os.path.join(d, "fail.db"))
        test_restart_safety(os.path.join(d, "restart.db"))
        test_factory(os.path.join(d, "factory.db"))
        test_concurrent_claim(os.path.join(d, "concurrent.db"))
        test_lost_claim_noop(os.path.join(d, "lost_claim.db"))
        test_liveness_gated_recovery(os.path.join(d, "liveness.db"))
        test_concurrent_writers_retry(os.path.join(d, "writers.db"))
        test_close_checkpoints_wal(os.path.join(d, "wal.db"))
    print("ALL SQLITE BOARD TESTS PASSED")
