from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from paths import CONFIGS_DIR
from utils import (
    compute_final_score,
    format_agent_response,
    get_run_dir,
    load_config,
    load_memory,
    validate_agent_response,
)


def _provider_overlay_path(memory_system: str) -> Path:
    return CONFIGS_DIR / f"{memory_system}.yaml"


_run_dir: Path | None = None


def get_run_directory() -> Path:
    return _run_directory()


def init_run_directory(
    run_name: str, memory_provider: str, noise_levels: list[int]
) -> Path:
    global _run_dir
    if _run_dir is not None:
        raise RuntimeError("run directory already initialized")
    path = get_run_dir(run_name)
    if path.exists():
        raise ValueError(f"invalid run name: {run_name} already exists")
    path.mkdir(parents=True, exist_ok=True)

    snapshots_dir = path / "config_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config = load_config()
    snapshot_config["memory"]["provider"] = memory_provider
    snapshot_config["memory"]["noise_levels"] = noise_levels
    with (snapshots_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(snapshot_config, f, default_flow_style=False, sort_keys=False)
    shutil.copy2(
        _provider_overlay_path(memory_provider),
        snapshots_dir / f"{memory_provider}.yaml",
    )

    _run_dir = path
    return _run_dir


def _run_directory() -> Path:
    if _run_dir is None:
        raise RuntimeError("run directory not initialized; pass --run-name")
    return _run_dir


def _user_results_path(
    user_id: str, noise_level: int, agent_model: str, memory_provider: str
) -> Path:
    noise_dir = _run_directory() / agent_model / user_id / f"noise_{noise_level}"
    noise_dir.mkdir(parents=True, exist_ok=True)
    return noise_dir / f"{user_id}_results_{memory_provider}.json"


def _load_user_results(
    user_id: str, noise_level: int, agent_model: str, memory_provider: str
) -> dict[str, Any]:
    path = _user_results_path(user_id, noise_level, agent_model, memory_provider)
    if not path.exists():
        return {
            "user_id": user_id,
            "memory_system": memory_provider,
            "agent_model": agent_model,
            "noise_level": noise_level,
            "questions": [],
        }
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_agent_response(agent_response: str) -> dict[str, str]:
    if validate_agent_response(agent_response):
        return format_agent_response(agent_response)
    return {"selection": "", "reasoning": agent_response}


def _build_question_result(
    user_id: str,
    question: dict[str, Any],
    agent_response: str,
    judge_response: dict[str, Any] | str | None,
    retrieved_context: str,
    raw_retrieval: str | None = None,
) -> dict[str, Any]:
    agent_response_dict = _build_agent_response(agent_response)
    result: dict[str, Any] = {
        "question_id": question["question_id"],
        "user_id": user_id,
        "task_type": question["task_type"],
        "target_domain": question["target_domain"],
        "source_domain": [
            load_memory(user_id, memory_id)["source_domain"]
            for memory_id in question["eval_memory_ids"]
        ],
        "correct_answer": question["correct_answer"],
        "eval_memory_ids": question["eval_memory_ids"],
        "retrieved_context": retrieved_context,
        "raw_retrieval": raw_retrieval,
        "agent_response": agent_response_dict,
        "judge_response": judge_response,
    }

    if not validate_agent_response(agent_response):
        result["final_score"] = 0.0
    elif isinstance(judge_response, dict):
        result["final_score"] = compute_final_score(
            agent_response_dict["selection"],
            question["correct_answer"],
            judge_response["score"],
        )

    return result


def write_results(
    user_id: str,
    question: dict[str, Any],
    agent_response: str,
    judge_response: dict[str, Any] | str | None,
    *,
    noise_level: int,
    agent_model: str,
    memory_provider: str,
    retrieved_context: str,
    raw_retrieval: str | None = None,
) -> None:
    """Write agent and judge results for a single question."""
    user_results = _load_user_results(
        user_id, noise_level, agent_model, memory_provider
    )
    user_results["questions"].append(
        _build_question_result(
            user_id,
            question,
            agent_response,
            judge_response,
            retrieved_context,
            raw_retrieval=raw_retrieval,
        )
    )

    path = _user_results_path(user_id, noise_level, agent_model, memory_provider)
    with path.open("w", encoding="utf-8") as f:
        json.dump(user_results, f, indent=2)


def write_run_failure(
    user_id: str,
    noise_level: int,
    *,
    agent_model: str,
    memory_provider: str,
    exception_type: str,
    error_message: str,
    failure_stage: str,
) -> None:
    """Write a provider-abort marker when insert or retrieval fails."""
    payload = {
        "user_id": user_id,
        "memory_system": memory_provider,
        "agent_model": agent_model,
        "noise_level": noise_level,
        "run_status": "failed",
        "failure_stage": failure_stage,
        "exception_type": exception_type,
        "error_message": error_message,
        "questions": [],
    }
    path = _user_results_path(user_id, noise_level, agent_model, memory_provider)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
