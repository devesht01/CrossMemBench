from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

from agent import Agent
from judge import Judge
from llms import BaseProvider, GeminiProvider, OpenAIProvider
from paths import CONFIG_PATH, CONFIGS_DIR, DATA_DIR, HARNESS_DIR, NOISE_FILE, PROMPTS_PATH, RUBRICS_PATH
from providers import MEMORY_PROVIDERS
from providers.base import BaseMemoryProvider

PROVIDERS = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_results_root() -> Path:
    return HARNESS_DIR / load_config()["benchmark"]["results_dir"]


def get_run_dir(run_name: str) -> Path:
    return get_results_root() / run_name


def create_judge() -> Judge:
    judge_cfg = load_config()["judge"]
    return Judge(
        OpenAIProvider(
            model=judge_cfg["model"],
            reasoning_effort=judge_cfg["reasoning_effort"],
            max_output_tokens=judge_cfg["max_output_tokens"],
        )
    )


def create_llm(provider: str, model: str) -> BaseProvider:
    agent_cfg = load_config()["agent"]
    return PROVIDERS[provider](
        model=model,
        temperature=agent_cfg["temperature"],
        max_output_tokens=agent_cfg["max_output_tokens"],
    )


def create_agent(provider: str, model: str) -> Agent:
    return Agent(create_llm(provider, model))


def create_memory_provider(name: str) -> BaseMemoryProvider:
    if name not in MEMORY_PROVIDERS:
        raise ValueError(f"Unknown memory provider: {name}")
    overlay_path = CONFIGS_DIR / f"{name}.yaml"
    with overlay_path.open(encoding="utf-8") as f:
        overlay = yaml.safe_load(f)
    memory_cfg = _deep_merge(load_config()["memory"], overlay["memory"])
    return MEMORY_PROVIDERS[name](memory_cfg)


def get_benchmark_users() -> list[str]:
    users = load_config()["benchmark"]["users"]
    if not isinstance(users, list) or not users:
        raise ValueError("benchmark.users must be a non-empty list")
    if not all(isinstance(user_id, str) and user_id for user_id in users):
        raise ValueError("benchmark.users must contain non-empty strings")
    return users



def load_memories(user_id: str) -> list[dict[str, Any]]:
    user_path = DATA_DIR / user_id / f"{user_id}.json"
    with user_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["memories"]




def get_noise_seed() -> int:
    return load_config()["memory"]["noise_seed"]


_shuffled_noise_cache: tuple[int, list[dict[str, Any]]] | None = None


def _get_shuffled_noise_memories() -> list[dict[str, Any]]:
    """Return noise memories shuffled once with memory.noise_seed for reproducible diversity."""
    global _shuffled_noise_cache
    seed = get_noise_seed()
    if _shuffled_noise_cache is not None and _shuffled_noise_cache[0] == seed:
        return _shuffled_noise_cache[1]

    memories = list(_load_noise_memories())
    rng = random.Random(seed)
    rng.shuffle(memories)
    _shuffled_noise_cache = (seed, memories)
    return memories



def load_noise_slice(start: int, end: int) -> list[dict[str, Any]]:
    """Return shuffled noise memories[start:end] (half-open interval)."""
    if start < 0 or end < start:
        raise ValueError(f"invalid noise slice: start={start}, end={end}")
    memories = _get_shuffled_noise_memories()
    return memories[start:end]



def _load_noise_memories() -> list[dict[str, Any]]:
    with NOISE_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    memories = data.get("memories")
    if not isinstance(memories, list):
        raise ValueError(f"noise file missing memories list: {NOISE_FILE}")
    return memories


def load_questions(user_id: str) -> list[dict[str, Any]]:
    """Load benchmark questions for a user."""
    user_path = DATA_DIR / user_id / f"{user_id}.json"
    with user_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def assemble_agent_prompt(question: dict[str, Any]) -> str:
    """Build the agent prompt for a benchmark question."""
    template = _load_agent_prompt()
    return _apply_template(
        template,
        question=question["question"],
        options=_format_options(question["options"]),
    )


def _load_agent_prompt() -> str:
    return PROMPTS_PATH.read_text(encoding="utf-8")


def _apply_template(template: str, **replacements: str) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))



#3 new stuff
def _load_rubrics() -> dict[str, Any]:
    with RUBRICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _format_correct_answer(question: dict[str, Any]) -> str:
    letter = question["correct_answer"]
    return f"{letter}. {question['options'][letter]}"


def _format_judge_memory(memory: dict[str, Any]) -> str:
    return f"{memory['memory_id']}: {memory['content']} ({memory['inferred_memory']})"


def assemble_judge_prompt(
    question: dict[str, Any],
    agent_response: dict[str, Any],
) -> str:
    """Build the judge prompt for a benchmark question and agent response."""
    task_type = question["task_type"]
    rubric_template = _load_rubrics()[task_type]["prompt"]
    user_id = _user_id_from_question(question)
    eval_ids = question["eval_memory_ids"]
    mem_index = {memory_id: load_memory(user_id, memory_id) for memory_id in eval_ids}

    selection = agent_response["selection"]
    reasoning = agent_response["reasoning"]
    correct_answer = _format_correct_answer(question)
    options = _format_options(question["options"])
    prompt_final = ""
    if task_type == "CMRT":
        memory_text = _format_judge_memory(mem_index[eval_ids[0]])
        prompt_final = _apply_template(
            rubric_template,
            memory=memory_text,
            question=question["question"],
            options=options,
            correct_answer=correct_answer,
            agent_selection=selection,
            agent_reasoning=reasoning,
        )

    if task_type == "DCA":
        constraint_a, constraint_b = [mem_index[memory_id] for memory_id in eval_ids]
        prompt_final = _apply_template(
            rubric_template,
            constraint_a=_format_judge_memory(constraint_a),
            constraint_b=_format_judge_memory(constraint_b),
            question=question["question"],
            options=options,
            correct_answer=correct_answer,
            agent_selection=selection,
            agent_reasoning=reasoning,
        )

    if prompt_final == "":
        raise ValueError(f"Unsupported task type for judge prompt: {task_type}")
    return prompt_final


def _parse_json_response(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError(f"Response is not a JSON object: {result}")
    return result


def validate_agent_response(agent_response: str) -> bool:
    try:
        result = _parse_json_response(agent_response)
    except (json.JSONDecodeError, ValueError):
        return False
    if "selection" not in result or "reasoning" not in result:
        return False
    if not isinstance(result["selection"], str):
        return False
    if not isinstance(result["reasoning"], str):
        return False
    return result["selection"].strip().upper() in {"A", "B", "C", "D"}


def format_agent_response(agent_response: str) -> dict[str, Any]:
    """Parse and normalize the raw agent LLM response."""
    result = _parse_json_response(agent_response)
    selection = result["selection"].strip().upper()
    if selection not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid selection: {result['selection']!r}")
    return {
        "selection": selection,
        "reasoning": result["reasoning"],
    }


def compute_final_score(
    agent_selection: str,
    correct_answer: str,
    judge_score: float,
) -> float:
    if agent_selection != correct_answer:
        return 0.0
    return judge_score


def validate_judge_response(judge_response: dict[str, Any], task_type: str) -> bool:
    if task_type not in {"CMRT", "DCA"}:
        return False
    if not isinstance(judge_response, dict):
        return False
    if "score" not in judge_response or "explanation" not in judge_response:
        return False
    if not isinstance(judge_response["explanation"], str):
        return False
    try:
        score = float(judge_response["score"])
    except (TypeError, ValueError):
        return False
    if score not in (0.0, 1.0):
        return False
    if task_type == "DCA":
        if "constraint_a_referenced" not in judge_response:
            return False
        if "constraint_b_referenced" not in judge_response:
            return False
        if not isinstance(judge_response["constraint_a_referenced"], bool):
            return False
        if not isinstance(judge_response["constraint_b_referenced"], bool):
            return False
    return True


def _user_id_from_question(question: dict[str, Any]) -> str:
    user_id, _ = question["question_id"].rsplit("_", 1)
    return user_id





def load_memory(user_id: str, memory_id: str) -> dict[str, Any]:
    """Load a single benchmark memory by ID for a user."""
    for memory in load_memories(user_id):
        if memory["memory_id"] == memory_id:
            return memory
    raise ValueError(f"memory not found: {memory_id}")

