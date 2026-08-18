from __future__ import annotations

from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .base import BaseProvider
from .utils import split_system_and_user


def gemini_response_text(response: types.GenerateContentResponse) -> str:
    if response.text:
        return response.text.strip()
    parts: list[str] = []
    if response.candidates:
        for part in response.candidates[0].content.parts or []:
            if part.text:
                parts.append(part.text)
    content = "".join(parts).strip()
    if not content:
        raise ValueError("Empty model response")
    return content


class GeminiProvider(BaseProvider):
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        json_output: bool = False,
        config_path: Path | None = None,
        client: genai.Client | None = None,
    ):
        super().__init__(
            config_path=config_path,
        )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.json_output = json_output
        self.client = client or genai.Client()

    def complete(self, messages: list[dict[str, str]]) -> str:
        system, user = split_system_and_user(messages)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system or None,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.json_output:
            config_kwargs["response_mime_type"] = "application/json"

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return gemini_response_text(resp)
