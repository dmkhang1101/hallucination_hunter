"""
Load TruthfulQA, stratified-sample N questions, call GPT-4o.

Output: data/sampled_questions.csv  (question metadata)
        data/primary_answers.csv    (question + GPT-4o answer)
"""

import hashlib
import time
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

import config


# Helpers

def _stable_id(question: str) -> str:
    """Short deterministic ID from question text."""
    return hashlib.md5(question.encode()).hexdigest()[:10]


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows from df with proportional representation per category.
    Any remainder rows go to the largest categories.
    """
    categories = df["category"].unique()
    n_cats = len(categories)
    base = n // n_cats
    remainder = n % n_cats

    # Sort categories by size desc so bigger buckets absorb the remainder
    cat_sizes = df["category"].value_counts()
    ordered_cats = cat_sizes.index.tolist()

    parts: list[pd.DataFrame] = []
    for i, cat in enumerate(ordered_cats):
        quota = base + (1 if i < remainder else 0)
        pool = df[df["category"] == cat]
        # If a category has fewer rows than quota, take all of them
        quota = min(quota, len(pool))
        parts.append(pool.sample(n=quota, random_state=seed))

    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def _dry_run_answer(question: str) -> str:
    return (
        f"[DRY RUN] This is a placeholder answer for: '{question}'. "
        "No API call was made."
    )


def _call_gpt(client: OpenAI, question: str, retries: int = 3) -> str:
    """Call GPT-4o with simple retry logic for transient errors."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                temperature=0.2,   # low temp for factual answers
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [warn] API error ({exc}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


# Public entry point

def run(n_questions: int = config.DEFAULT_N_QUESTIONS, dry_run: bool = False) -> pd.DataFrame:
    """
    Load TruthfulQA, sample n_questions, generate GPT-4o answers.
    Returns the answers DataFrame and writes both CSVs.
    """
    print(f"[generate] Loading TruthfulQA ({config.DATASET_SPLIT} split)…")
    dataset = load_dataset(config.DATASET_NAME, "generation", split=config.DATASET_SPLIT)
    df_full = dataset.to_pandas()

    # Keep only the columns we care about
    df_full = df_full[["question", "category", "correct_answers"]].copy()
    print(f"[generate] Full dataset: {len(df_full)} questions across "
          f"{df_full['category'].nunique()} categories")

    # Stratified sample
    df_sample = _stratified_sample(df_full, n=n_questions, seed=config.RANDOM_SEED)
    df_sample["question_id"] = df_sample["question"].apply(_stable_id)
    df_sample = df_sample[["question_id", "question", "category", "correct_answers"]]

    print(f"[generate] Sampled {len(df_sample)} questions")
    print(f"           Category breakdown:\n{df_sample['category'].value_counts().to_string()}")

    questions_path, answers_path, _, _ = config.get_paths(dry_run)

    # Save sampled questions
    df_sample.to_csv(questions_path, index=False)
    print(f"[generate] Saved questions → {questions_path}")

    # Generate GPT-4o answers
    if dry_run:
        print("[generate] DRY RUN — skipping real API calls")
        client = None
    else:
        if not config.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key, or use --dry-run."
            )
        client = OpenAI(api_key=config.OPENAI_API_KEY)

    answers: list[str] = []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample),
                       desc="Generating answers"):
        if dry_run:
            answer = _dry_run_answer(row["question"])
        else:
            answer = _call_gpt(client, row["question"])
        answers.append(answer)

    df_answers = df_sample.copy()
    df_answers["primary_answer"] = answers

    # Save answers
    df_answers.to_csv(answers_path, index=False)
    print(f"[generate] Saved answers → {answers_path}")

    return df_answers
