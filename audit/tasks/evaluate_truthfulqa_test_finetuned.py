"""
Task K — Evaluate DeBERTa on the annotated TruthfulQA test set
(truthfulqa_test_annotated.csv, 673 claims, 245 questions).

Premise = first correct answer from ground_truth_reference (grounded mode).
Hypothesis = atomic_claim.

Usage:
  python evaluate_truthfulqa_test_finetuned.py           # SciFact-finetuned model
  python evaluate_truthfulqa_test_finetuned.py --base    # zero-shot base model

Output (finetuned):
  results/eval/truthfulqa_test_finetuned.json
  results/figures/truthfulqa_test_finetuned_confusion.png

Output (base):
  results/eval/truthfulqa_test_base.json
  results/figures/truthfulqa_test_base_confusion.png
"""

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

import config
import utils


def _parse_first_reference(raw: str) -> str:
    """Extract the first string from a numpy-style list repr or plain string."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0]).strip()
    except (ValueError, SyntaxError):
        pass
    return raw.strip()


def _predict(pipe: object, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> list[str]:
    paired = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    preds: list[str] = []
    for i in range(0, len(paired), batch_size):
        batch = paired[i : i + batch_size]
        out = pipe(batch, truncation=True, max_length=512)
        for item in out:
            preds.append(item["label"].lower())
    return preds


def _plot_confusion(cm: list[list[int]], labels: list[str], path: Path, n: int, title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Greens")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix} — TruthfulQA Test (n={n})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(use_base: bool = False) -> dict:
    utils.set_all_seeds()

    if use_base:
        model_id: str = config.AUDITOR_MODEL
        eval_json = config.TQA_TEST_BASE_EVAL_JSON
        confusion_png = config.TQA_TEST_BASE_CONFUSION_PNG
        predictions_csv = config.TQA_TEST_BASE_PREDICTIONS_CSV
        model_label = f"base:{config.AUDITOR_MODEL}"
        title_prefix = "Zero-Shot DeBERTa"
        print("\n[Task K] Evaluating base (zero-shot) model on TruthfulQA test set...")
    else:
        if not config.FINETUNED_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Finetuned model not found at {config.FINETUNED_MODEL_DIR}. Run Task I first."
            )
        model_id = str(config.FINETUNED_MODEL_DIR)
        eval_json = config.TQA_TEST_FINETUNED_EVAL_JSON
        confusion_png = config.TQA_TEST_FINETUNED_CONFUSION_PNG
        predictions_csv = config.TQA_TEST_FINETUNED_PREDICTIONS_CSV
        model_label = f"finetuned:{config.AUDITOR_MODEL}"
        title_prefix = "SciFact-Finetuned DeBERTa"
        print("\n[Task K] Evaluating finetuned model on TruthfulQA test set...")

    test_df = pd.read_csv(config.TQA_TEST_ANNOTATED_CSV)

    premises: list[str] = []
    hypotheses: list[str] = []
    gold_labels: list[str] = []

    for _, row in test_df.iterrows():
        ref = _parse_first_reference(str(row["ground_truth_reference"]))
        premises.append(ref)
        hypotheses.append(str(row["atomic_claim"]))
        gold_labels.append(str(row["Gold_Label"]).strip().lower())

    print(f"  Evaluating {len(hypotheses)} claims across {test_df['question_id'].nunique()} questions...")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=model_id,
        device=device,
        truncation=True,
    )

    preds = _predict(pipe, premises, hypotheses)
    metrics = utils.compute_metrics(gold_labels, preds)

    contra_recall = metrics["per_class"].get("contradiction", {}).get("recall", 0.0)
    metrics["contradiction_recall"] = float(contra_recall)

    pred_df = test_df[["claim_id", "question_id", "category", "atomic_claim", "Gold_Label"]].copy()
    pred_df["predicted_label"] = preds
    pred_df["correct"] = pred_df["Gold_Label"].str.lower() == pred_df["predicted_label"]
    pred_df.to_csv(predictions_csv, index=False)
    print(f"  Saved → {predictions_csv.name}")

    n = len(hypotheses)
    utils.save_json(
        eval_json,
        {"model": model_label, "n_claims": n, **metrics},
        n_examples=n,
        model_name=model_label,
    )

    _plot_confusion(
        metrics["confusion_matrix"]["matrix"],
        metrics["confusion_matrix"]["labels"],
        confusion_png,
        n,
        title_prefix,
    )

    print(f"  Saved → {confusion_png.name}")
    print(f"  Macro-F1:             {metrics['macro_f1']:.4f}")
    print(f"  Accuracy:             {metrics['accuracy']:.4f}")
    print(f"  Contradiction recall: {contra_recall:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="Use zero-shot base model instead of finetuned")
    args = parser.parse_args()
    run(use_base=args.base)
