import os
import pathlib
import tempfile

import pytest

# Isolate recall (LanceDB) state BEFORE swarm_agent.recall is imported (it reads these at
# module load). Default OFF so the unit suite never loads the CPU embedder or touches the
# real ~/.cache store; a recall test can override SWARM_RECALL=1 + a temp SWARM_RECALL_PATH.
os.environ.setdefault("SWARM_RECALL", "0")
os.environ.setdefault("SWARM_RECALL_PATH",
                      str(pathlib.Path(tempfile.gettempdir()) / "swarm-test-recall.lance"))


@pytest.fixture(autouse=True, scope="session")
def _isolate_swarm_state():
    d = tempfile.mkdtemp(prefix="swarm-test-")
    old = {k: os.environ.get(k) for k in ("SWARM_TASKS_PATH", "SWARM_LOG_DIR", "SWARM_EVENT_LOG")}
    os.environ["SWARM_TASKS_PATH"] = str(pathlib.Path(d) / "tasks.json")
    os.environ["SWARM_LOG_DIR"] = str(pathlib.Path(d) / "logs")
    os.environ.setdefault("SWARM_EVENT_LOG", "1")
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def path(tmp_path):
    """SQLite board tests expect a string DB path, matching their script-mode calls."""
    return str(tmp_path / "board.db")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: end-to-end test that needs a running Step-3.7 endpoint")


@pytest.fixture
def live_endpoint():
    """Skip a live test unless the Step-3.7 endpoint answers /metrics."""
    from fleet import config as fcfg
    from fleet import metrics
    if metrics.scrape(fcfg.METRICS_URL, timeout=1.0) is None:
        pytest.skip(f"Step-3.7 endpoint unreachable at {fcfg.METRICS_URL}")
    return fcfg.METRICS_URL
