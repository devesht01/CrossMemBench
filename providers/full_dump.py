from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paths import PROFILES_PATH

from .base import BaseMemoryProvider, validate_memory

PROVIDER_FILES_DIR = Path(__file__).resolve().parent / "provider_files"


def _user_db_path(user_id: str) -> Path:
    return PROVIDER_FILES_DIR / f"full_dump_db_{user_id}.json"

_user_first_names: dict[str, str] | None = None


def _get_user_first_name(user_id: str) -> str | None:
    global _user_first_names
    if _user_first_names is None:
        with PROFILES_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _user_first_names = {}
        for profile in data.get("profiles", []):
            uid = profile.get("user_id")
            name = str(profile.get("name", "")).strip()
            if not uid or not name:
                continue
            first_name = name.split()[0]
            if not first_name:
                continue
            _user_first_names[uid] = first_name[0].upper() + first_name[1:]
    return _user_first_names.get(user_id)


def _format_noise_inferred_memory(inferred_memory: str, user_id: str) -> str:
    if "The user" not in inferred_memory:
        return inferred_memory
    first_name = _get_user_first_name(user_id)
    if not first_name:
        return inferred_memory
    return inferred_memory.replace("The user", first_name, 1)


def _format_retrieved_memory(
    memory: dict[str, Any],
    *,
    user_id: str,
    insert_inferred: bool,
) -> str:
    timestamp = memory.get("timestamp", "")
    source_domain = memory["source_domain"]
    content = memory["content"]
    if insert_inferred and "inferred_memory" in memory:
        inferred = memory["inferred_memory"]
        if memory.get("type") == "NOISE":
            inferred = _format_noise_inferred_memory(inferred, user_id)
        content = f"{content} ({inferred})"
    prefix_parts = [part for part in (timestamp, source_domain) if part]
    if prefix_parts:
        return f"[{' | '.join(prefix_parts)}] {content}"
    return content


class FullDumpMemoryProvider(BaseMemoryProvider):
    def __init__(self, top_k: int, *, insert_inferred: bool):
        super().__init__(top_k=top_k)
        self.insert_inferred = insert_inferred

    def insert_memory(self, user_id: str, memory: dict[str, Any]) -> None:
        validate_memory(memory)
        memories = self._load_user_memories(user_id)
        memories.append(memory)
        self._save_user_memories(user_id, memories)

    def retrieve_memories(
        self, user_id: str, agent_id: str, query: str
    ) -> tuple[str, str | None]:
        memories = self._load_user_memories(user_id)
        return (
            "\n".join(
                _format_retrieved_memory(
                    memory,
                    user_id=user_id,
                    insert_inferred=self.insert_inferred,
                )
                for memory in memories
            ),
            json.dumps(memories),
        )

    def reset(self, user_id: str) -> None:
        path = _user_db_path(user_id)
        if path.exists():
            path.unlink()

    @staticmethod
    def _load_user_memories(user_id: str) -> list[dict[str, Any]]:
        path = _user_db_path(user_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_user_memories(user_id: str, memories: list[dict[str, Any]]) -> None:
        path = _user_db_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2)
