"""Standalone ``swarm`` command.

This intentionally delegates to the measured fleet CLI while the goal-driven
planner front door is built. Keeping the compatibility layer means existing
benchmarks remain directly comparable.
"""

from __future__ import annotations

import sys

from fleet.cli import main as _fleet_main


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["tui"]:
        from .tui import main as tui_main
        return tui_main()
    return _fleet_main(argv, prog="swarm")


if __name__ == "__main__":
    raise SystemExit(main())
