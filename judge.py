from __future__ import annotations

import json
from typing import Any

from llms import OpenAIProvider

JUDGE_SYSTEM_PROMPT = "You are a strict benchmark judge."


class Judge:
    def __init__(self, llm_provider: OpenAIProvider):
        self.llm_provider = llm_provider

    def run(self, prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_provider.complete(messages)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        result = json.loads(text)
        if "score" not in result:
            raise ValueError(f"Judge response missing score: {result}")
        if "explanation" not in result:
            raise ValueError(f"Judge response missing explanation: {result}")
        parsed: dict[str, Any] = {
            "score": float(result["score"]),
            "explanation": result["explanation"],
        }
        if "constraint_a_referenced" in result:
            parsed["constraint_a_referenced"] = result["constraint_a_referenced"]
        if "constraint_b_referenced" in result:
            parsed["constraint_b_referenced"] = result["constraint_b_referenced"]
        return parsed
