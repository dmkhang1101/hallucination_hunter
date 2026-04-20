"""
Task F — Subtype analysis (RQ3): recall per hallucination subtype.

Uses TruthfulQA grounded predictions (from Task D) and pilot gold subtypes.
Flags subtypes with n < 3 as insufficient.
Output: results/subtype_analysis.json, results/subtype_recall.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def run() -> dict:
    utils.set_all_seeds()
    print("\n[Task F] Subtype analysis (RQ3)...")

    if not config.PILOT_GOLD_JSON.exists():
        raise FileNotFoundError(f"Run Task A first — {config.PILOT_GOLD_JSON} missing")
    if not config.TQA_GROUNDED_JSON.exists():
        raise FileNotFoundError(f"Run Task D first — {config.TQA_GROUNDED_JSON} missing")

    gold_data = json.loads(config.PILOT_GOLD_JSON.read_text())
    gold_records = gold_data["records"]

    # Filter to contradictions with non-null subtype
    contra_records = [
        r for r in gold_records
        if r["gold_label"] == "contradiction" and r.get("subtype") is not None
    ]
    print(f"  Contradiction records with subtype: {len(contra_records)}")

    if not contra_records:
        print("  No contradiction records with subtypes — skipping subtype analysis")
        result = {"subtypes": {}, "note": "No contradiction records with non-null subtypes found"}
        utils.save_json(config.SUBTYPE_ANALYSIS_JSON, result, n_examples=0)
        return result

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=config.AUDITOR_MODEL, device=device, truncation=True)
    assert len(pipe.model.config.id2label) == 3

    premises = [r["ground_truth_reference"] for r in contra_records]
    hypotheses = [r["atomic_claim"] for r in contra_records]
    gold_labels = [r["gold_label"] for r in contra_records]
    subtypes = [r["subtype"] for r in contra_records]

    preds = _predict(pipe, premises, hypotheses)

    # Group by subtype
    subtype_map: dict[str, dict] = {}
    for gold, pred, subtype in zip(gold_labels, preds, subtypes):
        if subtype not in subtype_map:
            subtype_map[subtype] = {"n": 0, "correct": 0, "recall": 0.0, "insufficient": False}
        subtype_map[subtype]["n"] += 1
        if pred == gold:
            subtype_map[subtype]["correct"] += 1

    for key, val in subtype_map.items():
        n = val["n"]
        val["recall"] = val["correct"] / n if n > 0 else 0.0
        val["insufficient"] = n < 3

    # Plot
    sorted_subtypes = sorted(subtype_map.items(), key=lambda x: x[1]["recall"], reverse=True)
    names = [s for s, _ in sorted_subtypes]
    recalls = [v["recall"] for _, v in sorted_subtypes]
    ns = [v["n"] for _, v in sorted_subtypes]
    colors = ["#ff7f7f" if v["insufficient"] else "#4c8cf5" for _, v in sorted_subtypes]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, recalls, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall")
    ax.set_title("Contradiction Recall by Hallucination Subtype\n(red = n<3, insufficient)")
    ax.set_xticklabels(names, rotation=30, ha="right")
    for bar, n_val in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n_val}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(config.SUBTYPE_RECALL_PNG, dpi=150)
    plt.close()
    print(f"  Saved → {config.SUBTYPE_RECALL_PNG}")

    result = {"subtypes": subtype_map}
    utils.save_json(config.SUBTYPE_ANALYSIS_JSON, result, n_examples=len(contra_records))
    for name, val in sorted_subtypes:
        flag = " [INSUFFICIENT n<3]" if val["insufficient"] else ""
        print(f"  {name}: recall={val['recall']:.2f} (n={val['n']}){flag}")
    return result


if __name__ == "__main__":
    run()
