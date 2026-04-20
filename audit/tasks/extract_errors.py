"""
Task G — Error analysis template.

Collects false negatives on contradiction class from TruthfulQA grounded predictions,
fills to 30 with false positives, saves CSV with blank failure_mode column for human review.

# Failure-mode taxonomy:
# claim_decontextualization, reference_incompleteness, subtle_entailment,
# numerical_reasoning, world_knowledge_gap, other

Output: results/error_samples.csv
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from transformers import pipeline

import config
import utils


_NLI_MAP = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}


def _predict(pipe: any, premises: list[str], hypotheses: list[str]) -> list[str]:
    inputs = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
    preds = []
    for i in range(0, len(inputs), 16):
        out = pipe(inputs[i : i + 16], truncation=True, max_length=512)
        for item in out:
            preds.append(_NLI_MAP.get(item["label"].lower(), item["label"].lower()))
    return preds


def run() -> pd.DataFrame:
    utils.set_all_seeds()
    print("\n[Task G] Building error analysis template...")

    if not config.PILOT_GOLD_JSON.exists():
        raise FileNotFoundError(f"Run Task A first — {config.PILOT_GOLD_JSON} missing")

    gold_data = json.loads(config.PILOT_GOLD_JSON.read_text())
    gold_records = [r for r in gold_data["records"] if r["gold_label"] is not None]

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=config.AUDITOR_MODEL, device=device, truncation=True)
    assert len(pipe.model.config.id2label) == 3

    premises = [r["ground_truth_reference"] for r in gold_records]
    hypotheses = [r["atomic_claim"] for r in gold_records]
    gold_labels = [r["gold_label"] for r in gold_records]

    preds = _predict(pipe, premises, hypotheses)

    rows = []
    for rec, gold, pred, premise, hyp in zip(gold_records, gold_labels, preds, premises, hypotheses):
        rows.append({
            "claim_id": rec["claim_id"],
            "question_id": rec["question_id"],
            "premise": premise,
            "hypothesis": hyp,
            "gold_label": gold,
            "predicted_label": pred,
            "is_fn": gold == "contradiction" and pred != "contradiction",
            "is_fp": gold != "contradiction" and pred == "contradiction",
        })

    # Priority: false negatives first, then false positives to fill to 30
    fns = [r for r in rows if r["is_fn"]]
    fps = [r for r in rows if r["is_fp"]]

    TARGET = 30
    selected = fns[:TARGET]
    remaining = TARGET - len(selected)
    if remaining > 0:
        selected += fps[:remaining]

    print(f"  False negatives: {len(fns)}, False positives used: {min(len(fps), remaining)}")
    print(f"  Total error samples: {len(selected)}")

    df = pd.DataFrame([{
        "claim_id": r["claim_id"],
        "question_id": r["question_id"],
        "premise": r["premise"],
        "hypothesis": r["hypothesis"],
        "gold_label": r["gold_label"],
        "predicted_label": r["predicted_label"],
        # Failure-mode taxonomy: claim_decontextualization, reference_incompleteness,
        # subtle_entailment, numerical_reasoning, world_knowledge_gap, other
        "failure_mode": "",
    } for r in selected])

    df.to_csv(config.ERROR_SAMPLES_CSV, index=False)
    print(f"  Saved → {config.ERROR_SAMPLES_CSV}")
    return df


if __name__ == "__main__":
    run()
