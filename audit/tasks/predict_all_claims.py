"""
Task H — Run NLI auditor on all atomic claims (full dataset, ~400 claims).

Premise = first correct answer from TruthfulQA; hypothesis = atomic claim.
Output: results/predictions/all_claims_predictions.csv
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from transformers import pipeline

import config
import utils


def _parse_first_correct_answer(raw: str) -> str:
    """Extract the first answer string from the numpy-array-repr correct_answers field."""
    matches = re.findall(r"'([^']+)'", str(raw))
    if matches:
        return matches[0]
    return str(raw).strip()


def run() -> pd.DataFrame:
    utils.set_all_seeds()
    print("\n[Task H] Predicting NLI labels for all atomic claims...")

    df = pd.read_csv(config.ATOMIC_CLAIMS)
    print(f"  Loaded {len(df)} claims from {config.ATOMIC_CLAIMS.name}")

    df["ground_truth_reference"] = df["correct_answers"].apply(_parse_first_correct_answer)

    premises = df["ground_truth_reference"].tolist()
    hypotheses = df["claim"].tolist()

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=config.AUDITOR_MODEL,
        device=device,
        truncation=True,
    )

    inputs = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
    predicted_labels: list[str] = []
    confidence_scores: list[float] = []

    batch_size = 16
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        out = pipe(batch, truncation=True, max_length=512)
        for item in out:
            predicted_labels.append(item["label"].lower())
            confidence_scores.append(round(float(item["score"]), 4))
        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i + batch_size, len(inputs))}/{len(inputs)} claims...")

    out_df = pd.DataFrame({
        "question_id": df["question_id"],
        "question": df["question"],
        "category": df["category"],
        "claim_index": df["claim_index"],
        "atomic_claim": df["claim"],
        "ground_truth_reference": df["ground_truth_reference"],
        "predicted_label": predicted_labels,
        "confidence": confidence_scores,
    })

    out_df.to_csv(config.ALL_CLAIMS_PREDICTIONS_CSV, index=False)

    label_dist = out_df["predicted_label"].value_counts().to_dict()
    print(f"  Saved {len(out_df)} predictions → {config.ALL_CLAIMS_PREDICTIONS_CSV}")
    print(f"  Label distribution: {label_dist}")

    return out_df


if __name__ == "__main__":
    run()
