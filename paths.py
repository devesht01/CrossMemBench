from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
CONFIG_PATH = CONFIGS_DIR / "config.yaml"
PROFILES_PATH = PROJECT_ROOT / "data" / "profiles.json"
HARNESS_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
NOISE_FILE = PROJECT_ROOT / "data" / "noise.json"
PROMPTS_PATH = PROJECT_ROOT / "prompts" / "agent_prompt.md"
RUBRICS_PATH = PROJECT_ROOT / "prompts" / "rubrics.json"
