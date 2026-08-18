from argparse import ArgumentParser, Namespace
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import yaml

from utils import _deep_merge, get_run_dir

ANALYSIS_FILENAME = "analysis.json"

_STAT_COUNTER_KEYS = (
    "cmrt_n",
    "cmrt_retrieval_hits",
    "cmrt_accuracy_hits",
    "dca_n",
    "dca_both",
    "dca_partial",
    "dca_partial_in_domain",
    "dca_partial_other_domain",
    "dca_none",
    "dca_accuracy_hits",
)

_AGENT_MODEL_KEYS = (
    "agent_model",
    "judge_model",
    "judge_reasoning_effort",
    "judge_max_output_tokens",
    "agent_temperature",
    "agent_max_output_tokens",
    "top_k",
    "noise_levels",
    "noise_seed",
    "noise",
)

_NOISE_KEYS = (
    "noise_level",
    *_STAT_COUNTER_KEYS,
    "cmrt_accuracy",
    "dca_accuracy",
    "average_accuracy",
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--run-name",
        required=True,
        help="Results folder name under benchmark.results_dir",
    )
    return parser.parse_args()


def _load_snapshot_config(run_dir: Path) -> dict[str, Any]:
    snapshots = run_dir / "config_snapshots"
    with (snapshots / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    provider_name = config["memory"]["provider"]
    overlay_path = snapshots / f"{provider_name}.yaml"
    with overlay_path.open(encoding="utf-8") as f:
        overlay = yaml.safe_load(f)
    return _deep_merge(config, overlay)


def _id_in_raw(memory_id: str, raw_retrieval: str) -> bool:
    compact = "".join(raw_retrieval.split())
    return memory_id in compact


def _empty_stats() -> dict[str, int]:
    return {key: 0 for key in _STAT_COUNTER_KEYS}


def _score_question(question: dict[str, Any], stats: dict[str, int]) -> None:
    task_type = question["task_type"]
    eval_memory_ids = question["eval_memory_ids"]
    raw_retrieval = question["raw_retrieval"]
    accurate = question["final_score"] == 1.0
    if task_type == "CMRT":
        stats["cmrt_n"] += 1
        if _id_in_raw(eval_memory_ids[0], raw_retrieval):
            stats["cmrt_retrieval_hits"] += 1
        if accurate:
            stats["cmrt_accuracy_hits"] += 1
        return
    if task_type == "DCA":
        stats["dca_n"] += 1
        hits = int(_id_in_raw(eval_memory_ids[0], raw_retrieval)) + int(
            _id_in_raw(eval_memory_ids[1], raw_retrieval)
        )
        if hits == 2:
            stats["dca_both"] += 1
        elif hits == 1:
            stats["dca_partial"] += 1
            i = 0 if _id_in_raw(eval_memory_ids[0], raw_retrieval) else 1
            if question["source_domain"][i] == question["target_domain"]:
                stats["dca_partial_in_domain"] += 1
            else:
                stats["dca_partial_other_domain"] += 1
        else:
            stats["dca_none"] += 1
        if accurate:
            stats["dca_accuracy_hits"] += 1
        return
    raise ValueError(f"Unknown task_type: {task_type}")


def _iter_result_files(run_dir: Path):
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "config_snapshots":
            continue
        agent_model = model_dir.name
        for user_dir in sorted(model_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            for noise_dir in sorted(user_dir.iterdir()):
                if not noise_dir.is_dir() or not noise_dir.name.startswith("noise_"):
                    continue
                noise_level = int(noise_dir.name.removeprefix("noise_"))
                for path in sorted(noise_dir.glob("*.json")):
                    yield agent_model, noise_level, path


def _collect_stats(
    run_dir: Path,
) -> dict[str, dict[int, dict[str, int]]]:
    stats: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_empty_stats)
    )
    for agent_model, noise_level, path in _iter_result_files(run_dir):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for question in data["questions"]:
            _score_question(question, stats[agent_model][noise_level])
    return stats


def _accuracy(hits: int, n: int) -> float:
    return round(hits / n, 2)


def _noise_block(noise_level: int, stats: dict[str, int]) -> dict[str, Any]:
    total_n = stats["cmrt_n"] + stats["dca_n"]
    total_hits = stats["cmrt_accuracy_hits"] + stats["dca_accuracy_hits"]
    return {
        "noise_level": noise_level,
        **{key: stats[key] for key in _STAT_COUNTER_KEYS},
        "cmrt_accuracy": _accuracy(stats["cmrt_accuracy_hits"], stats["cmrt_n"]),
        "dca_accuracy": _accuracy(stats["dca_accuracy_hits"], stats["dca_n"]),
        "average_accuracy": _accuracy(total_hits, total_n),
    }


def _build_payload(
    config: dict[str, Any],
    grouped: dict[str, dict[int, dict[str, int]]],
) -> dict[str, Any]:
    judge = config["judge"]
    agent = config["agent"]
    memory = config["memory"]
    agent_models = []
    for agent_model in sorted(grouped):
        agent_models.append(
            {
                "agent_model": agent_model,
                "judge_model": judge["model"],
                "judge_reasoning_effort": judge["reasoning_effort"],
                "judge_max_output_tokens": judge["max_output_tokens"],
                "agent_temperature": agent["temperature"],
                "agent_max_output_tokens": agent["max_output_tokens"],
                "top_k": memory["top_k"],
                "noise_levels": memory["noise_levels"],
                "noise_seed": memory["noise_seed"],
                "noise": [
                    _noise_block(noise_level, grouped[agent_model][noise_level])
                    for noise_level in sorted(grouped[agent_model])
                ],
            }
        )
    return {"agent_models": agent_models}


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(f"analysis.json missing keys {missing} in {where}")


def _load_analysis(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if "agent_models" not in payload:
        raise ValueError("analysis.json missing key: agent_models")
    if not isinstance(payload["agent_models"], list):
        raise ValueError("analysis.json agent_models must be a list")
    for i, model in enumerate(payload["agent_models"]):
        if not isinstance(model, dict):
            raise ValueError(f"analysis.json agent_models[{i}] must be an object")
        _require_keys(model, _AGENT_MODEL_KEYS, f"agent_models[{i}]")
        if not isinstance(model["noise"], list):
            raise ValueError(f"analysis.json agent_models[{i}].noise must be a list")
        for j, noise in enumerate(model["noise"]):
            if not isinstance(noise, dict):
                raise ValueError(
                    f"analysis.json agent_models[{i}].noise[{j}] must be an object"
                )
            _require_keys(noise, _NOISE_KEYS, f"agent_models[{i}].noise[{j}]")
    return payload


def _fmt_rate(hits: int, n: int) -> str:
    return f"{hits / n:.2f} ({hits}/{n})"


def _print_metadata(model: dict[str, Any]) -> None:
    print(f"agent_model: {model['agent_model']}")
    print(f"judge_model: {model['judge_model']}")
    print(f"judge_reasoning_effort: {model['judge_reasoning_effort']}")
    print(f"judge_max_output_tokens: {model['judge_max_output_tokens']}")
    print(f"agent_temperature: {model['agent_temperature']}")
    print(f"agent_max_output_tokens: {model['agent_max_output_tokens']}")
    print(f"top_k: {model['top_k']}")
    print(f"noise_levels: {model['noise_levels']}")
    print(f"noise_seed: {model['noise_seed']}")


def _print_noise_stats(noise_level: int, stats: dict[str, Any]) -> None:
    print(f"noise_level {noise_level}")
    print(f"  CMRT n={stats['cmrt_n']}")
    print(f"    retrieval: {_fmt_rate(stats['cmrt_retrieval_hits'], stats['cmrt_n'])}")
    print(f"    accuracy: {_fmt_rate(stats['cmrt_accuracy_hits'], stats['cmrt_n'])}")
    print(f"  DCA n={stats['dca_n']}")
    print(f"    both_retrieved: {_fmt_rate(stats['dca_both'], stats['dca_n'])}")
    print(f"    partial_retrieved: {_fmt_rate(stats['dca_partial'], stats['dca_n'])}")
    if stats["dca_partial"] > 0:
        print(
            f"      in_domain: {_fmt_rate(stats['dca_partial_in_domain'], stats['dca_partial'])}"
        )
        print(
            f"      other_domain: {_fmt_rate(stats['dca_partial_other_domain'], stats['dca_partial'])}"
        )
    print(f"    non_retrieved: {_fmt_rate(stats['dca_none'], stats['dca_n'])}")
    print(f"    accuracy: {_fmt_rate(stats['dca_accuracy_hits'], stats['dca_n'])}")
    total_n = stats["cmrt_n"] + stats["dca_n"]
    total_hits = stats["cmrt_accuracy_hits"] + stats["dca_accuracy_hits"]
    print(f"  average accuracy: {_fmt_rate(total_hits, total_n)}")


def _print_payload(payload: dict[str, Any]) -> None:
    for model in payload["agent_models"]:
        _print_metadata(model)
        for noise in model["noise"]:
            _print_noise_stats(noise["noise_level"], noise)


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir(args.run_name)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    analysis_path = run_dir / ANALYSIS_FILENAME
    if analysis_path.exists():
        payload = _load_analysis(analysis_path)
    else:
        config = _load_snapshot_config(run_dir)
        grouped = _collect_stats(run_dir)
        payload = _build_payload(config, grouped)
        with analysis_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    _print_payload(payload)


if __name__ == "__main__":
    main()
