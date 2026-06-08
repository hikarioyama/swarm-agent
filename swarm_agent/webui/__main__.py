"""Command-line entrypoint for the swarm web sidecar."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from fleet import config
from swarm_agent.cli import _log_dir

from .app import create_app


def main(argv=None) -> int:
    """Run the webui server."""
    parser = argparse.ArgumentParser(prog="swarm webui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--log-dir", type=Path, default=_log_dir())
    parser.add_argument("--metrics-url",
                        default=os.environ.get("FLEET_METRICS_URL")
                        or config.METRICS_URL)
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--replay")
    args = parser.parse_args(argv)

    app = create_app(log_dir=args.log_dir, metrics_url=args.metrics_url,
                     no_metrics=args.no_metrics, replay=args.replay)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
