from __future__ import annotations

from llms import BaseProvider


class Agent:
    def __init__(self, llm_provider: BaseProvider):
        self.llm_provider = llm_provider

    def build_messages(self, user_input: str) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": user_input},
        ]

    def run(self, prompt: str) -> str:
        messages = self.build_messages(prompt)
        return self.llm_provider.complete(messages)
