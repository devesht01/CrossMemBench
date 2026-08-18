from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseProvider(ABC):
    def __init__(
        self,
        *,
        config_path: Path | None = None,
    ):
        pass

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...
