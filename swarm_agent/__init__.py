"""Public package for the standalone swarm-agent harness.

The measured runtime remains in ``fleet`` for compatibility with the step37
prototype. New callers should use ``swarm_agent`` or the ``swarm`` command.
HermesAgent is a runtime dependency, not the application shell.
"""

from fleet import AIMDController, Board, State, Task, ThreadFleet, config

__all__ = ["AIMDController", "Board", "State", "Task", "ThreadFleet", "config"]
__version__ = "0.1.0"
