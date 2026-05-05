"""
Split each GPT-4o answer into atomic facts using spaCy sentences.

Input:  data/primary_answers.csv
Output: data/atomic_claims.csv

Each row in the output is one sentence/claim, linked back to its
question_id, question, and category.
<<<<<<< HEAD
=======

This module hardens the default spaCy sentencizer against common
"mid-thought" splits observed in the 50-sample pilot annotation:
    - abbreviations like U.S., e.g., i.e., Dr., Mr., Ph.D., etc.
    - decimals (3.14) and version-like numbers
    - sentence-final periods inside quotes/parentheses
    - orphan fragments that start with a lowercase letter, a closing
      quote/paren, or a coordinating conjunction
The strategy combines:
    1. A custom token-boundary rule that prevents spaCy from ending a
       sentence on a known abbreviation or between digits in a decimal.
    2. A post-pass that merges short fragments lacking a verb back into
       the previous sentence so each emitted claim is contextually
       complete.
>>>>>>> main
"""

import ast
import json
<<<<<<< HEAD

import pandas as pd
import spacy
=======
import re

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc
>>>>>>> main
from tqdm import tqdm

import config


<<<<<<< HEAD
#  Helpers 

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
=======
# Constants

# Abbreviations whose trailing period must NOT end a sentence.
# Stored without the trailing dot — we match the token's lowercase form.
_ABBREVIATIONS: frozenset[str] = frozenset({
    "u.s", "u.k", "u.s.a", "e.g", "i.e", "etc", "vs",
    "dr", "mr", "mrs", "ms", "prof", "ph.d", "jr", "sr",
    "st", "no", "inc", "ltd", "co", "corp", "fig", "vol",
    "approx", "cf", "al",  # "et al."
})

# Tokens that, if they START a sentence, signal it is actually a
# continuation of the previous sentence and should be merged back.
_CONTINUATION_STARTERS: frozenset[str] = frozenset({
    "and", "but", "or", "so", "yet", "nor", "because",
})

# Closing punctuation that should never start a new sentence.
_CLOSING_PUNCT: frozenset[str] = frozenset({
    ")", "]", "}", '"', "'", "”", "’", "»",
})

# Minimum claim length kept from the original splitter.
_MIN_CLAIM_LEN: int = 10

# Fragments shorter than this without a verb are merged into the prior
# sentence — they are almost always orphan tails (e.g. trailing ", etc.").
_SHORT_FRAGMENT_LEN: int = 30


# Custom sentence-boundary component

@Language.component("protect_abbreviations")
def _protect_abbreviations(doc: Doc) -> Doc:
    """
    Override spaCy's sentence boundaries to prevent splits inside
    known abbreviations and decimals.

    Runs before the parser so the parser sees the corrected boundaries.
    """
    for i, token in enumerate(doc[:-1]):
        next_tok = doc[i + 1]

        # Case 1: abbreviation like "U.S." or "Dr." — token text without
        # a final period gives the canonical form. spaCy may tokenize
        # "U.S." as a single token or as "U.S" + ".".
        text = token.text.lower().rstrip(".")
        if text in _ABBREVIATIONS:
            next_tok.is_sent_start = False

        # Case 2: decimal numbers — "3.14" can be split as "3", ".", "14".
        # If a period sits between two digits, the next token cannot
        # start a sentence.
        if (
            token.text == "."
            and i > 0
            and doc[i - 1].text.isdigit()
            and next_tok.text.isdigit()
        ):
            next_tok.is_sent_start = False

    return doc


# Helpers

def _load_model() -> Language:
    """Load spaCy and inject the abbreviation-protection component."""
    print(f"[extract] Loading spaCy model '{config.SPACY_MODEL}'…")
    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{config.SPACY_MODEL}' not found. "
            f"Run:  python -m spacy download {config.SPACY_MODEL}"
        ) from exc

    # The custom component must run before the parser so its boundary
    # decisions are respected. Re-adding is a no-op if pipeline reload.
    if "protect_abbreviations" not in nlp.pipe_names:
        nlp.add_pipe("protect_abbreviations", before="parser")
    return nlp


def _has_verb(span_text: str, nlp: Language) -> bool:
    """Cheap check: does the fragment contain a finite/aux verb?"""
    # Re-parse only the fragment — short by construction, so cost is low.
    sub = nlp(span_text)
    return any(t.pos_ in {"VERB", "AUX"} for t in sub)


def _should_merge_into_previous(claim: str, nlp: Language) -> bool:
    """
    True if `claim` looks like a continuation of the prior sentence
    rather than a standalone atomic claim.
    """
    stripped = claim.strip()
    if not stripped:
        return True

    first_char = stripped[0]
    # Lowercase start (excluding "i" pronoun edge case — "i" is rare at
    # start of a real sentence anyway, so still merge).
    if first_char.islower():
        return True
    if first_char in _CLOSING_PUNCT:
        return True

    first_word = re.split(r"\W+", stripped, maxsplit=1)[0].lower()
    if first_word in _CONTINUATION_STARTERS:
        return True

    # Short fragment with no verb → orphan tail.
    if len(stripped) < _SHORT_FRAGMENT_LEN and not _has_verb(stripped, nlp):
        return True

    return False


def _extract_claims(nlp: Language, text: str) -> list[str]:
    """
    Split text into contextually-complete claims:
      1. spaCy sentence segmentation with abbreviation protection.
      2. Strip + length filter (`> _MIN_CLAIM_LEN`).
      3. Merge fragments that start with lowercase / closing punctuation /
         coordinator, or that are very short and verbless.
    """
    doc = nlp(text)

    # Initial pass: collect non-empty stripped sentences.
    raw: list[str] = [s.text.strip() for s in doc.sents if s.text.strip()]

    # Merge continuations into the prior sentence.
    merged: list[str] = []
    for sent in raw:
        if merged and _should_merge_into_previous(sent, nlp):
            merged[-1] = f"{merged[-1]} {sent}".strip()
        else:
            merged.append(sent)

    # Final length filter — keep behaviour parity with prior version.
    return [c for c in merged if len(c) > _MIN_CLAIM_LEN]
>>>>>>> main


# Public entry point

def run(answers_df: pd.DataFrame | None = None, dry_run: bool = False) -> pd.DataFrame:
    """
    Extract atomic claims from primary answers.

    Args:
        answers_df: If provided, use this DataFrame directly.
                    Otherwise load from the appropriate answers CSV.
        dry_run:    If True, read/write from the dryrun subfolder.

    Returns the claims DataFrame and writes both CSV and JSON.
    """
    _, answers_path, claims_csv, claims_json = config.get_paths(dry_run)

    if answers_df is None:
        if not answers_path.exists():
            raise FileNotFoundError(
                f"Answers file not found at {answers_path}. "
                "Run the generate step first."
            )
        answers_df = pd.read_csv(answers_path)
        print(f"[extract] Loaded {len(answers_df)} answers from {answers_path}")

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
                "correct_answers": row["correct_answers"],
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

    df_claims.to_csv(claims_csv, index=False)
    print(f"[extract] Saved claims → {claims_csv}")

    # JSON version: group claims by question for easier downstream consumption
    grouped: list[dict] = []
    for qid, g in df_claims.groupby("question_id", sort=False):
        # correct_answers may be: list/ndarray (from generate) or string (loaded from CSV)
        raw = g["correct_answers"].iloc[0]
        if isinstance(raw, str):
            correct = ast.literal_eval(raw)
        else:
            correct = list(raw)  # handles numpy ndarray and plain lists
        grouped.append({
            "question_id":    qid,
            "question":       g["question"].iloc[0],
            "category":       g["category"].iloc[0],
            "correct_answers": correct,
            "primary_answer": g["primary_answer"].iloc[0],
            "claims":         g["claim"].tolist(),
        })
    with open(claims_json, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)
    print(f"[extract] Saved claims → {claims_json}")

    return df_claims
