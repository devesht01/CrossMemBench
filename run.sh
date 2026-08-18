#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PROVIDER=""
MODEL=""
MEMORY_PROVIDER=""
NOISE_LEVELS=""

usage() {
  echo "usage: $0 --provider gemini|openai --model NAME --memory-provider no_mem|full_dump|dense_retrieval --noise-levels 0,100" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --memory-provider)
      MEMORY_PROVIDER="$2"
      shift 2
      ;;
    --noise-levels)
      NOISE_LEVELS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROVIDER" || -z "$MODEL" || -z "$MEMORY_PROVIDER" || -z "$NOISE_LEVELS" ]]; then
  usage
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${MEMORY_PROVIDER}_${TIMESTAMP}"
RUN_DIR="$(python -c "from utils import get_run_dir; print(get_run_dir('$RUN_NAME'))")"

if [[ -e "$RUN_DIR" ]]; then
  echo "invalid run name: ${RUN_NAME} already exists" >&2
  exit 1
fi

python run_memorysystem.py --provider "$PROVIDER" --model "$MODEL" --memory-provider "$MEMORY_PROVIDER" --noise-levels "$NOISE_LEVELS" --run-name "$RUN_NAME"
python analyze_results.py --run-name "$RUN_NAME"
