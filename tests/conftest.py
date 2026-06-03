import os
import pathlib
import tempfile

import pytest


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
