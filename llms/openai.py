from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from .base import BaseProvider
from .utils import split_system_and_user


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
        config_path: Path | None = None,
        client: OpenAI | None = None,
    ):
        super().__init__(
            config_path=config_path,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = client or OpenAI()

    def complete(self, messages: list[dict[str, str]]) -> str:
        system, user = split_system_and_user(messages)
        kwargs: dict = {
            "model": self.model,
            "instructions": system or None,
            "input": user,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        resp = self.client.responses.create(**kwargs)
        if resp.status == "incomplete":
            reason = (
                resp.incomplete_details.reason
                if resp.incomplete_details is not None
                else "unknown"
            )
            raise ValueError(f"Incomplete model response: {reason}")
        content = resp.output_text
        if not content:
            raise ValueError("Empty model response")
        return content
