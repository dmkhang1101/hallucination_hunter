"""
Task D — TruthfulQA evaluation in grounded and ungrounded modes (RQ2).

Grounded:   premise = ground_truth_reference, hypothesis = atomic_claim
Ungrounded: premise = full GPT-4o answer,     hypothesis = atomic_claim
McNemar's test compares the two modes on the same 50 claims.
Blocks on: results/gold/pilot_gold.json
Output: results/eval/truthfulqa_grounded.json, results/eval/truthfulqa_ungrounded.json,
        results/eval/grounded_vs_ungrounded.json, results/figures/pilot_confusion.png,
        results/figures/pilot_f1_accuracy.png
"""

import json
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


_NLI_MAP = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}


def _predict(pipe: any, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> list[str]:
    inputs = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
    preds = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        out = pipe(batch, truncation=True, max_length=512)
        for item in out:
            preds.append(_NLI_MAP.get(item["label"].lower(), item["label"].lower()))
    return preds


def _mcnemar_contingency(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> dict:
    """McNemar's test: compare correct/incorrect between two predictors on same items."""
    # contingency table: [[both correct, a correct b wrong], [a wrong b correct, both wrong]]
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


def _plot_f1_accuracy(metrics_g: dict, metrics_u: dict, path: Path) -> None:
    labels = list(metrics_g["per_class"].keys())
    x = np.arange(len(labels))
    width = 0.35

    f1_g = [metrics_g["per_class"][l]["f1"] for l in labels]
    f1_u = [metrics_u["per_class"][l]["f1"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Per-class F1 bar chart
    axes[0].bar(x - width / 2, f1_g, width, label="Grounded", color="#4C72B0")
    axes[0].bar(x + width / 2, f1_u, width, label="Ungrounded", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Per-Class F1: Grounded vs Ungrounded (Pilot Set, n=50)")
    axes[0].legend()
    for i, (vg, vu) in enumerate(zip(f1_g, f1_u)):
        axes[0].text(i - width / 2, vg + 0.02, f"{vg:.2f}", ha="center", fontsize=8)
        axes[0].text(i + width / 2, vu + 0.02, f"{vu:.2f}", ha="center", fontsize=8)

    # Macro-F1 and Accuracy summary
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
    axes[1].set_title("Overall Accuracy & Macro-F1 (Pilot Set, n=50)")
    axes[1].legend()
    for i, (vg, vu) in enumerate(zip(grounded_vals, ungrounded_vals)):
        axes[1].text(i - width / 2, vg + 0.02, f"{vg:.2f}", ha="center", fontsize=9)
        axes[1].text(i + width / 2, vu + 0.02, f"{vu:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run() -> tuple[dict, dict, dict]:
    utils.set_all_seeds()
    print("\n[Task D] TruthfulQA grounded vs ungrounded evaluation (RQ2)...")

    if not config.PILOT_GOLD_JSON.exists():
        raise FileNotFoundError(f"Run Task A first — {config.PILOT_GOLD_JSON} missing")

    gold_data = json.loads(config.PILOT_GOLD_JSON.read_text())
    gold_records = gold_data["records"]

    # Build lookup from atomic_claims.csv: claim_id (index) → question_id
    claims_df = pd.read_csv(config.ATOMIC_CLAIMS)
    answers_df = pd.read_csv(config.PRIMARY_ANSWERS)

    # primary_answers keyed by question_id
    answers_map = dict(zip(answers_df["question_id"], answers_df["primary_answer"]))

    # Build eval rows by matching on claim text + question_id
    premises_grounded, premises_ungrounded, hypotheses, gold_labels = [], [], [], []
    for rec in gold_records:
        if rec["gold_label"] is None:
            continue
        qid = rec["question_id"]
        claim_text = rec["atomic_claim"]
        ground_truth = rec["ground_truth_reference"]
        full_answer = answers_map.get(qid, "")

        premises_grounded.append(ground_truth)
        premises_ungrounded.append(full_answer)
        hypotheses.append(claim_text)
        gold_labels.append(rec["gold_label"])

    print(f"  Evaluating {len(hypotheses)} claims (n=50 pilot)...")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=config.AUDITOR_MODEL,
        device=device,
        truncation=True,
    )
    assert len(pipe.model.config.id2label) == 3, "id2label check failed"

    preds_grounded = _predict(pipe, premises_grounded, hypotheses)
    preds_ungrounded = _predict(pipe, premises_ungrounded, hypotheses)

    metrics_g = utils.compute_metrics(gold_labels, preds_grounded)
    metrics_u = utils.compute_metrics(gold_labels, preds_ungrounded)
    mcnemar_result = _mcnemar_contingency(gold_labels, preds_grounded, preds_ungrounded)

    power_flag = {"n_claims": len(hypotheses), "power_warning": True}

    utils.save_json(
        config.TQA_GROUNDED_JSON,
        {"mode": "grounded", **power_flag, **metrics_g},
        n_examples=len(hypotheses),
    )
    utils.save_json(
        config.TQA_UNGROUNDED_JSON,
        {"mode": "ungrounded", **power_flag, **metrics_u},
        n_examples=len(hypotheses),
    )
    utils.save_json(
        config.TQA_COMPARISON_JSON,
        {
            **power_flag,
            "grounded_macro_f1": metrics_g["macro_f1"],
            "ungrounded_macro_f1": metrics_u["macro_f1"],
            "mcnemar": mcnemar_result,
        },
        n_examples=len(hypotheses),
    )

    _plot_confusion(
        metrics_g["confusion_matrix"]["matrix"],
        metrics_g["confusion_matrix"]["labels"],
        config.RESULTS / "pilot_confusion.png",
        "DeBERTa Auditor — Pilot Set Confusion Matrix (Grounded, n=50)",
    )
    _plot_f1_accuracy(metrics_g, metrics_u, config.RESULTS / "pilot_f1_accuracy.png")
    print(f"  Saved confusion matrix → results/figures/pilot_confusion.png")
    print(f"  Saved F1/accuracy chart → results/figures/pilot_f1_accuracy.png")

    print(f"  Grounded Macro-F1: {metrics_g['macro_f1']:.4f}")
    print(f"  Ungrounded Macro-F1: {metrics_u['macro_f1']:.4f}")
    print(f"  McNemar p-value: {mcnemar_result['pvalue']:.4f}")
    return metrics_g, metrics_u, mcnemar_result


if __name__ == "__main__":
    run()
