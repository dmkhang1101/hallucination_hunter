"""
Task J — TruthfulQA grounded vs ungrounded evaluation on the full
all_claims_annotation_complete.csv dataset (402 claims, 150 questions).

Mirrors Task D but uses the extended annotation set instead of the 50-claim
pilot, giving more statistical power for the McNemar comparison.

Grounded:   premise = ground_truth_reference, hypothesis = atomic_claim
Ungrounded: premise = primary_answer (GPT-4o),  hypothesis = atomic_claim

Output:
  results/eval/truthfulqa_grounded_all_claims.json
  results/eval/truthfulqa_ungrounded_all_claims.json
  results/eval/grounded_vs_ungrounded_all_claims.json
  results/figures/pilot_confusion_all_claims.png
  results/figures/pilot_f1_accuracy_all_claims.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from statsmodels.stats.contingency_tables import mcnemar
from transformers import pipeline

import config
import utils

# Normalize subtype labels to 3-class schema
_LABEL_MAP: dict[str, str] = {
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradiction (factual error)": "contradiction",
    "contradiction (imitative falsehood)": "contradiction",
}

_NLI_MAP = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}


def _normalize_gold(label: str) -> str:
    return _LABEL_MAP.get(label.strip().lower(), label.strip().lower())


def _predict(pipe: object, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> list[str]:
    paired = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    preds: list[str] = []
    for i in range(0, len(paired), batch_size):
        batch = paired[i : i + batch_size]
        out = pipe(batch, truncation=True, max_length=512)
        for item in out:
            preds.append(_NLI_MAP.get(item["label"].lower(), item["label"].lower()))
    return preds


def _mcnemar_contingency(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> dict:
    cc = cw = wc = ww = 0
    for t, a, b in zip(y_true, pred_a, pred_b):
        a_ok = a == t
        b_ok = b == t
        if a_ok and b_ok:
            cc += 1
        elif a_ok and not b_ok:
            cw += 1
        elif not a_ok and b_ok:
            wc += 1
        else:
            ww += 1
    table = [[cc, cw], [wc, ww]]
    result = mcnemar(table, exact=True)
    return {
        "contingency_table": table,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def _plot_confusion(cm: list[list[int]], labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = max(max(row) for row in cm) / 2
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > thresh else "black")
    ax.set_ylabel("Gold Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_f1_accuracy(metrics_g: dict, metrics_u: dict, path: Path, n: int) -> None:
    labels = list(metrics_g["per_class"].keys())
    x = np.arange(len(labels))
    width = 0.35

    f1_g = [metrics_g["per_class"][l]["f1"] for l in labels]
    f1_u = [metrics_u["per_class"][l]["f1"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x - width / 2, f1_g, width, label="Grounded", color="#4C72B0")
    axes[0].bar(x + width / 2, f1_u, width, label="Ungrounded", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title(f"Per-Class F1: Grounded vs Ungrounded (n={n})")
    axes[0].legend()
    for i, (vg, vu) in enumerate(zip(f1_g, f1_u)):
        axes[0].text(i - width / 2, vg + 0.02, f"{vg:.2f}", ha="center", fontsize=8)
        axes[0].text(i + width / 2, vu + 0.02, f"{vu:.2f}", ha="center", fontsize=8)

    summary_labels = ["Macro-F1", "Accuracy"]
    grounded_vals = [metrics_g["macro_f1"], metrics_g["accuracy"]]
    ungrounded_vals = [metrics_u["macro_f1"], metrics_u["accuracy"]]
    x2 = np.arange(len(summary_labels))
    axes[1].bar(x2 - width / 2, grounded_vals, width, label="Grounded", color="#4C72B0")
    axes[1].bar(x2 + width / 2, ungrounded_vals, width, label="Ungrounded", color="#DD8452")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(summary_labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Overall Accuracy & Macro-F1 (n={n})")
    axes[1].legend()
    for i, (vg, vu) in enumerate(zip(grounded_vals, ungrounded_vals)):
        axes[1].text(i - width / 2, vg + 0.02, f"{vg:.2f}", ha="center", fontsize=9)
        axes[1].text(i + width / 2, vu + 0.02, f"{vu:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run() -> tuple[dict, dict, dict]:
    utils.set_all_seeds()
    print("\n[Task J] TruthfulQA grounded vs ungrounded on all_claims (n=402)...")

    claims_df = pd.read_csv(config.ALL_CLAIMS_CSV)
    answers_df = pd.read_csv(config.PRIMARY_ANSWERS_PILOT)
    answers_map: dict[str, str] = dict(zip(answers_df["question_id"], answers_df["primary_answer"]))

    premises_grounded: list[str] = []
    premises_ungrounded: list[str] = []
    hypotheses: list[str] = []
    gold_labels: list[str] = []

    for _, row in claims_df.iterrows():
        norm = _normalize_gold(str(row["Gold_Label"]))
        qid: str = str(row["question_id"])
        full_answer = answers_map.get(qid, "")
        if not full_answer:
            continue
        premises_grounded.append(str(row["ground_truth_reference"]))
        premises_ungrounded.append(full_answer)
        hypotheses.append(str(row["atomic_claim"]))
        gold_labels.append(norm)

    print(f"  Evaluating {len(hypotheses)} claims across {claims_df['question_id'].nunique()} questions...")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=config.AUDITOR_MODEL,
        device=device,
        truncation=True,
    )

    preds_grounded = _predict(pipe, premises_grounded, hypotheses)
    preds_ungrounded = _predict(pipe, premises_ungrounded, hypotheses)

    metrics_g = utils.compute_metrics(gold_labels, preds_grounded)
    metrics_u = utils.compute_metrics(gold_labels, preds_ungrounded)
    mcnemar_result = _mcnemar_contingency(gold_labels, preds_grounded, preds_ungrounded)

    n = len(hypotheses)
    utils.save_json(
        config.TQA_GROUNDED_ALL_CLAIMS_JSON,
        {"mode": "grounded", "n_claims": n, **metrics_g},
        n_examples=n,
    )
    utils.save_json(
        config.TQA_UNGROUNDED_ALL_CLAIMS_JSON,
        {"mode": "ungrounded", "n_claims": n, **metrics_u},
        n_examples=n,
    )
    utils.save_json(
        config.TQA_COMPARISON_ALL_CLAIMS_JSON,
        {
            "n_claims": n,
            "grounded_macro_f1": metrics_g["macro_f1"],
            "ungrounded_macro_f1": metrics_u["macro_f1"],
            "mcnemar": mcnemar_result,
        },
        n_examples=n,
    )

    _plot_confusion(
        metrics_g["confusion_matrix"]["matrix"],
        metrics_g["confusion_matrix"]["labels"],
        config.PILOT_CONFUSION_ALL_CLAIMS_PNG,
        f"DeBERTa Auditor — Grounded Confusion Matrix (n={n})",
    )
    _plot_f1_accuracy(metrics_g, metrics_u, config.PILOT_F1_ACCURACY_ALL_CLAIMS_PNG, n)

    print(f"  Saved → {config.PILOT_CONFUSION_ALL_CLAIMS_PNG.name}")
    print(f"  Saved → {config.PILOT_F1_ACCURACY_ALL_CLAIMS_PNG.name}")
    print(f"  Grounded Macro-F1:   {metrics_g['macro_f1']:.4f}")
    print(f"  Ungrounded Macro-F1: {metrics_u['macro_f1']:.4f}")
    print(f"  McNemar p-value:     {mcnemar_result['pvalue']:.4f}")
    return metrics_g, metrics_u, mcnemar_result


if __name__ == "__main__":
    run()
