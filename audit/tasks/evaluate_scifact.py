"""
Task C — SciFact evaluation (RQ1): NLI transfer to scientific claim verification.

Premise = abstract, hypothesis = claim.
Label map: SUPPORT → entailment, CONTRADICT → contradiction (NOT_ENOUGH_INFO dropped).
SciFact data loaded directly from official S3 release (no HF loading script required).
Output: results/eval/scifact_eval.json, results/figures/scifact_confusion.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import pipeline

import config
import utils


_NLI_MAP = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}


def _predict_batch(pipe: any, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> list[str]:
    inputs = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        out = pipe(batch, truncation=True, max_length=512)
        for item in out:
            results.append(_NLI_MAP.get(item["label"].lower(), item["label"].lower()))
    return results


def _plot_confusion(cm: list[list[int]], labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SciFact Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def run() -> dict:
    utils.set_all_seeds()
    print("\n[Task C] SciFact evaluation (RQ1)...")

    premises, hypotheses, gold_labels = utils.load_scifact_test()
    print(f"  Claims with evidence: {len(premises)}")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=config.AUDITOR_MODEL,
        device=device,
        truncation=True,
    )
    assert len(pipe.model.config.id2label) == 3, "id2label check failed"

    pred_labels = _predict_batch(pipe, premises, hypotheses)
    # Scifact only has entailment/contradiction — restrict metrics to those two classes
    active_labels = sorted(set(gold_labels))
    metrics = utils.compute_metrics(gold_labels, pred_labels, labels=active_labels)

    _plot_confusion(
        metrics["confusion_matrix"]["matrix"],
        metrics["confusion_matrix"]["labels"],
        config.SCIFACT_CONFUSION_PNG,
    )

    utils.save_json(config.SCIFACT_EVAL_JSON, metrics, n_examples=len(premises))
    print(f"  Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    run()
