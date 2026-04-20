"""
Task D — TruthfulQA evaluation in grounded and ungrounded modes (RQ2).

Grounded:   premise = ground_truth_reference, hypothesis = atomic_claim
Ungrounded: premise = full GPT-4o answer,     hypothesis = atomic_claim
McNemar's test compares the two modes on the same 50 claims.
Blocks on: results/pilot_gold.json
Output: results/truthfulqa_grounded.json, results/truthfulqa_ungrounded.json,
        results/grounded_vs_ungrounded.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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

    print(f"  Grounded Macro-F1: {metrics_g['macro_f1']:.4f}")
    print(f"  Ungrounded Macro-F1: {metrics_u['macro_f1']:.4f}")
    print(f"  McNemar p-value: {mcnemar_result['pvalue']:.4f}")
    return metrics_g, metrics_u, mcnemar_result


if __name__ == "__main__":
    run()
