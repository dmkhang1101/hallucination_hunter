"""
Central configuration — reads from .env and exposes typed constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

QUESTIONS_CSV   = DATA_DIR / "sampled_questions.csv"
ANSWERS_CSV     = DATA_DIR / "primary_answers.csv"
CLAIMS_CSV      = DATA_DIR / "atomic_claims.csv"
CLAIMS_JSON     = DATA_DIR / "atomic_claims.json"

# Dry-run outputs go to a separate subfolder so they never overwrite real data
DRYRUN_DIR      = DATA_DIR / "dryrun"
DRYRUN_DIR.mkdir(exist_ok=True)

DRYRUN_QUESTIONS_CSV = DRYRUN_DIR / "sampled_questions.csv"
DRYRUN_ANSWERS_CSV   = DRYRUN_DIR / "primary_answers.csv"
DRYRUN_CLAIMS_CSV    = DRYRUN_DIR / "atomic_claims.csv"
DRYRUN_CLAIMS_JSON   = DRYRUN_DIR / "atomic_claims.json"

# TruthfulQA train/test split outputs
SPLIT_DIR = DATA_DIR / "split"
SPLIT_DIR.mkdir(exist_ok=True)
TRAIN_CSV = SPLIT_DIR / "truthfulqa_train.csv"
TEST_CSV  = SPLIT_DIR / "truthfulqa_test.csv"


def get_paths(dry_run: bool) -> tuple[object, object, object, object]:
    """Return (questions, answers, claims_csv, claims_json) paths for the given mode."""
    if dry_run:
        return DRYRUN_QUESTIONS_CSV, DRYRUN_ANSWERS_CSV, DRYRUN_CLAIMS_CSV, DRYRUN_CLAIMS_JSON
    return QUESTIONS_CSV, ANSWERS_CSV, CLAIMS_CSV, CLAIMS_JSON

# ── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL: str = "gpt-4o"

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME: str = "truthful_qa"
DATASET_SPLIT: str = "validation"   # 817 questions, has 'category' field

# ── Sampling ─────────────────────────────────────────────────────────────────
DEFAULT_N_QUESTIONS: int = 150
RANDOM_SEED: int = 42               # reproducible sampling

# ── spaCy ────────────────────────────────────────────────────────────────────
SPACY_MODEL: str = "en_core_web_lg"

# ── GPT generation ───────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are a knowledgeable assistant. "
    "Answer the following question concisely and factually in 2–4 sentences. "
    "Do not hedge excessively — state what you believe to be true."
)
