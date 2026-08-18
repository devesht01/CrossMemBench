from argparse import ArgumentParser, Namespace
import time

from tqdm import tqdm

from providers import MEMORY_PROVIDERS
from utils import (
    PROVIDERS,
    create_agent,
    create_judge,
    create_memory_provider,
    get_benchmark_users,
    load_memories,
    load_questions,
    load_noise_slice,
    assemble_agent_prompt,
    format_agent_response,
    assemble_judge_prompt,
    validate_agent_response,
    validate_judge_response,
)

from results_writer import init_run_directory, write_results, write_run_failure

_RETRY_SLEEPS = (30, 90, 120)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--provider",
        required=True,
        choices=list(PROVIDERS),
        help="LLM provider: gemini or openai",
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--memory-provider",
        required=True,
        choices=list(MEMORY_PROVIDERS),
        help="Memory system: no_mem, full_dump, or dense_retrieval",
    )
    parser.add_argument(
        "--noise-levels",
        required=True,
        help="Comma-separated noise levels, e.g. 0,100",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Results folder name under benchmark.results_dir",
    )
    return parser.parse_args()


def parse_noise_levels(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("noise-levels is empty")
    levels = [int(p) for p in parts]
    if any(level < 0 for level in levels):
        raise ValueError("noise-levels must be non-negative integers")
    for i in range(1, len(levels)):
        if levels[i] <= levels[i - 1]:
            raise ValueError("noise-levels must be strictly increasing")
    return levels


def call_with_retry(fn):
    last_exc = None
    attempts = 1 + len(_RETRY_SLEEPS)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == attempts:
                break
            sleep_s = _RETRY_SLEEPS[attempt - 1]
            print(f"attempt {attempt} failed ({e!r}); retrying in {sleep_s}s")
            time.sleep(sleep_s)
    raise last_exc


def abort_run(
    memory, user, noise_level, *, agent_model, memory_provider, error, failure_stage
):
    print(f"{failure_stage} failed after retries, aborting run")
    print(error)
    memory.reset(user)
    write_run_failure(
        user,
        noise_level,
        agent_model=agent_model,
        memory_provider=memory_provider,
        exception_type=type(error).__name__,
        error_message=str(error),
        failure_stage=failure_stage,
    )


def run_experiment(
    agent, judge, memory, users, noise_levels, run_name, memory_provider
):
    print(f"Running CrossMemBench with {agent.llm_provider.model} and {memory_provider}")
    init_run_directory(run_name, memory_provider, noise_levels)
    agent_model = agent.llm_provider.model

    for user in tqdm(users, desc="users", unit="user"):
        memory.reset(user)
        prev_noise = 0

        for noise_level in noise_levels:
            if noise_level == 0:
                to_insert = load_memories(user)
            else:
                to_insert = load_noise_slice(prev_noise, noise_level)

            if to_insert:
                try:
                    call_with_retry(lambda: memory.insert_memories(user, to_insert))
                    print(f"Inserted {len(to_insert)} memories at noise_level={noise_level}")
                except Exception as e:
                    abort_run(
                        memory,
                        user,
                        noise_level,
                        agent_model=agent_model,
                        memory_provider=memory_provider,
                        error=e,
                        failure_stage="insert",
                    )
                    return

            questions = load_questions(user)
            for question in questions:
                print(f"Asking question id {question['question_id']}")
                query = question["question"]

                agent_prompt = assemble_agent_prompt(question)

                try:
                    retrieved_context, raw_retrieval = call_with_retry(
                        lambda: memory.retrieve_memories(
                            user, question["target_domain"], query
                        )
                    )
                except Exception as e:
                    abort_run(
                        memory,
                        user,
                        noise_level,
                        agent_model=agent_model,
                        memory_provider=memory_provider,
                        error=e,
                        failure_stage="retrieve",
                    )
                    return

                final_prompt = agent_prompt.replace(
                    "{memories_section}", retrieved_context
                )

                def _agent():
                    resp = agent.run(final_prompt)
                    if not validate_agent_response(resp):
                        raise ValueError("invalid agent json")
                    return resp

                try:
                    agent_response = call_with_retry(_agent)
                except Exception as e:
                    print(
                        f"agent failed after retries for {question['question_id']}, skipping ({e!r})"
                    )
                    continue

                formatted_agent_response = format_agent_response(agent_response)
                if (formatted_agent_response["selection"] == question["correct_answer"]):
                    # agent selected correct answer, lets ask judge.
                    judge_prompt = assemble_judge_prompt(
                        question, formatted_agent_response
                    )

                    def _judge():
                        resp = judge.run(judge_prompt)
                        if not validate_judge_response(resp, question["task_type"]):
                            raise ValueError("invalid judge json")
                        return resp

                    try:
                        judge_response = call_with_retry(_judge)
                    except Exception as e:
                        abort_run(
                            memory,
                            user,
                            noise_level,
                            agent_model=agent_model,
                            memory_provider=memory_provider,
                            error=e,
                            failure_stage="judge",
                        )
                        return
                else:
                    judge_response = {
                        "score": 0.0,
                        "explanation": (
                            "Agent selected wrong answer -- auto generated judge response"
                        ),
                    }
                    if question["task_type"] == "DCA":
                        judge_response["constraint_a_referenced"] = False
                        judge_response["constraint_b_referenced"] = False

                write_results(
                    user,
                    question,
                    agent_response,
                    judge_response,
                    noise_level=noise_level,
                    agent_model=agent_model,
                    memory_provider=memory_provider,
                    retrieved_context=retrieved_context,
                    raw_retrieval=raw_retrieval,
                )

                prev_noise = noise_level

def main() -> None:
    args = parse_args()
    noise_levels = parse_noise_levels(args.noise_levels)
    agent = create_agent(args.provider, args.model)
    judge = create_judge()
    memory = create_memory_provider(args.memory_provider)
    users = get_benchmark_users()
    run_experiment(
        agent,
        judge,
        memory,
        users,
        noise_levels,
        args.run_name,
        args.memory_provider,
    )

if __name__ == "__main__":
    main()
