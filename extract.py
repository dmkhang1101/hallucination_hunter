"""
Step 3 — Split each GPT-4o answer into atomic facts using spaCy sentences.

Input:  data/primary_answers.csv
Output: data/atomic_claims.csv

Each row in the output is one sentence/claim, linked back to its
question_id, question, and category.
"""

import pandas as pd
import spacy
from tqdm import tqdm

import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_model() -> spacy.Language:
    print(f"[extract] Loading spaCy model '{config.SPACY_MODEL}'…")
    try:
        return spacy.load(config.SPACY_MODEL)
    except OSError:
        raise OSError(
            f"spaCy model '{config.SPACY_MODEL}' not found. "
            f"Run:  python -m spacy download {config.SPACY_MODEL}"
        )


def _extract_claims(nlp: spacy.Language, text: str) -> list[str]:
    """
    Split text into sentences using spaCy, strip whitespace,
    and filter out empty/very short fragments.
    """
    doc = nlp(text)
    claims = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    return claims


# ── Public entry point ───────────────────────────────────────────────────────

def run(answers_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Extract atomic claims from primary answers.

    Args:
        answers_df: If provided, use this DataFrame directly.
                    Otherwise load from config.ANSWERS_CSV.

    Returns the claims DataFrame and writes config.CLAIMS_CSV.
    """
    if answers_df is None:
        if not config.ANSWERS_CSV.exists():
            raise FileNotFoundError(
                f"Answers file not found at {config.ANSWERS_CSV}. "
                "Run the generate step first."
            )
        answers_df = pd.read_csv(config.ANSWERS_CSV)
        print(f"[extract] Loaded {len(answers_df)} answers from {config.ANSWERS_CSV}")

    nlp = _load_model()

    rows: list[dict] = []
    for _, row in tqdm(answers_df.iterrows(), total=len(answers_df),
                       desc="Extracting claims"):
        claims = _extract_claims(nlp, str(row["primary_answer"]))
        for idx, claim in enumerate(claims):
            rows.append({
                "question_id":    row["question_id"],
                "question":       row["question"],
                "category":       row["category"],
                "primary_answer": row["primary_answer"],
                "claim_index":    idx,          # 0-based position within the answer
                "claim":          claim,
            })

    df_claims = pd.DataFrame(rows)

    # Summary stats
    total_claims = len(df_claims)
    avg_per_answer = total_claims / len(answers_df) if len(answers_df) > 0 else 0
    print(f"[extract] Extracted {total_claims} claims "
          f"({avg_per_answer:.1f} avg per answer)")

    df_claims.to_csv(config.CLAIMS_CSV, index=False)
    print(f"[extract] Saved claims → {config.CLAIMS_CSV}")

    return df_claims
