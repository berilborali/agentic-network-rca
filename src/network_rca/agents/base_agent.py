"""Abstract base class for all RCA agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for all agents in the agentic RCA pipeline.

    Each agent encapsulates a single responsibility and exposes a ``run``
    method that accepts an *input context* dict and returns an *output
    context* dict.  The orchestrator threads these contexts together to
    form the complete analysis pipeline.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config: dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"network_rca.agent.{name}")

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's task.

        Parameters
        ----------
        context:
            Shared pipeline context produced by preceding agents.

        Returns
        -------
        dict[str, Any]
            Updated context containing this agent's outputs.
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
