"""Standalone ``swarm`` command.

This intentionally delegates to the measured fleet CLI while the goal-driven
planner front door is built. Keeping the compatibility layer means existing
benchmarks remain directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fleet.cli import main as _fleet_main


def _log_dir() -> Path:
    return Path(os.environ.get("SWARM_LOG_DIR") or
                Path.home() / ".cache" / "swarm-agent" / "logs").expanduser()


def _latest_log(log_dir: Path) -> Path | None:
    latest = log_dir / "latest.jsonl"
    if latest.exists():
        return latest.resolve()
    files = _event_logs(log_dir)
    return files[-1].resolve() if files else None


def _event_logs(log_dir: Path) -> list[Path]:
    try:
        return sorted(log_dir.glob("events-*.jsonl"),
                      key=lambda path: (path.stat().st_mtime_ns, path.name))
    except OSError:
        return []


def _read_events(paths: list[Path]) -> list[dict]:
    events = []
    for path in paths:
        try:
            with path.open() as fh:
                for line in fh:
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(event, dict):
                        events.append(event)
        except OSError:
            continue
    return events


def _is_error(event: dict) -> bool:
    return (event.get("kind") in {"error", "fail", "deadlock", "manager_error"}
            or event.get("event") in {"fail", "deadlock"})


def _one_line(value, limit: int = 100) -> str:
    text = " ".join(str(value or "").splitlines())
    return text if len(text) <= limit else text[:limit] + "…"


def _render_event(event: dict, *, details: bool = False) -> None:
    kind = str(event.get("kind") or "?")
    if event.get("event"):
        kind += f"/{event['event']}"
    goal_id = str(event.get("goal_id") or "")
    summary = ""
    for key in ("text", "error", "note", "goal"):
        if event.get(key):
            summary = _one_line(event[key])
            break
    print(f"{event.get('ts', '?')}  {kind}  {goal_id}  {summary}".rstrip())
    if details:
        for key in ("detail", "traceback"):
            if event.get(key):
                print(str(event[key]).rstrip())


def _logs_main(argv) -> int:
    parser = argparse.ArgumentParser(prog="swarm logs")
    parser.add_argument("--errors", action="store_true",
                        help="show only failures, with detail and traceback")
    parser.add_argument("--tail", type=int, default=40,
                        help="number of recent events to print (default: 40)")
    parser.add_argument("--path", action="store_true",
                        help="print only the latest event log path")
    parser.add_argument("--all", action="store_true",
                        help="read every session log instead of only the latest")
    args = parser.parse_args(argv)

    log_dir = _log_dir().resolve()
    latest = _latest_log(log_dir)
    if args.path:
        if latest is not None:
            print(latest)
            return 0
        return 1

    print(f"log dir: {log_dir}")
    print(f"latest: {latest or '(none)'}")
    if args.all:
        paths = _event_logs(log_dir)
    else:
        paths = [latest] if latest is not None else []
    events = _read_events(paths)
    if args.errors:
        events = [event for event in events if _is_error(event)]
    if args.tail >= 0:
        events = events[-args.tail:] if args.tail else []
    for event in events:
        _render_event(event, details=args.errors)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "logs":
        return _logs_main(argv[1:])
    if argv and argv[0] == "webui":
        from .webui.__main__ import main as webui_main
        return webui_main(argv[1:])
    if not argv or argv == ["tui"]:
        from .tui import main as tui_main
        return tui_main()
    return _fleet_main(argv, prog="swarm")


if __name__ == "__main__":
    raise SystemExit(main())
