from __future__ import annotations

from typing import Any

from .base import BaseMemoryProvider, validate_memory


class NoMemProvider(BaseMemoryProvider):
    def insert_memory(self, user_id: str, memory: dict[str, Any]) -> None:
        validate_memory(memory)

    def retrieve_memories(
        self, user_id: str, agent_id: str, query: str
    ) -> tuple[str, str | None]:
        # print("No memories retrieved, No mem active")
        return "", ""

    def reset(self, user_id: str) -> None:
        pass