"""Offline checks for the standalone swarm-agent command surface."""

from __future__ import annotations

import contextlib
import io

from swarm_agent import cli


def test_swarm_help_uses_public_command_name() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            cli.main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
    text = out.getvalue()
    assert text.startswith("usage: swarm ")
    assert "Swarm-agent" in text


if __name__ == "__main__":
    test_swarm_help_uses_public_command_name()
    print("swarm CLI offline smoke passed")
