from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

REQUIRED_MEMORY_FIELDS = ("content", "timestamp", "source_domain", "memory_id")


def validate_memory(memory: dict[str, Any]) -> None:
    for field in REQUIRED_MEMORY_FIELDS:
        if field not in memory:
            raise ValueError(f"memory missing required field: {field}")


class BaseMemoryProvider(ABC):
    def __init__(self, top_k: int):
        self.top_k = top_k

    @abstractmethod
    def insert_memory(self, user_id: str, memory: dict[str, Any]) -> None:
        pass

    def insert_memories(self, user_id: str, memories: list[dict[str, Any]]) -> None:
        for memory in memories:
            self.insert_memory(user_id, memory)

    @abstractmethod
    def retrieve_memories(
        self, user_id: str, agent_id: str, query: str
    ) -> tuple[str, str | None]:
        """Return (agent context, raw retrieval).

        The first string is shown to the agent. The second must expose dataset
        memory_id values so retrieval analysis can score hits independently of
        task accuracy.
        """
        pass

    @abstractmethod
    def reset(self, user_id: str) -> None:
        pass
