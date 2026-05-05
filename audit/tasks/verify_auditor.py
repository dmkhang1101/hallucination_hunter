"""
Task B — Pipeline sanity check for DeBERTa-v3 MNLI model.

1. Save id2label → results/eval/model_config.json
2. Assert 10 hand-written pairs classify correctly
3. Assert MNLI-mismatched 500-sample accuracy ∈ [0.85, 0.93]
Output: results/eval/sanity_check.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from transformers import pipeline

import config
import utils


# Maps HF MNLI label strings to our schema
_MNLI_LABEL_MAP = {
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
}

_HAND_PAIRS = [
    # entailment
    ("All cats are mammals.", "Cats are animals.", "entailment"),
    ("The sky is blue on a clear day.", "The sky has color on clear days.", "entailment"),
    ("Water boils at 100°C at sea level.", "Water can become a gas at 100°C.", "entailment"),
    ("Paris is the capital of France.", "France has a capital city.", "entailment"),
    # neutral
    ("The movie won three awards.", "The director was nervous at the ceremony.", "neutral"),
    ("She runs every morning.", "Her shoes are expensive.", "neutral"),
    ("The restaurant opened last year.", "The chef trained in Italy.", "neutral"),
    # contradiction
    ("The Earth orbits the Sun.", "The Sun orbits the Earth.", "contradiction"),
    ("He passed the exam.", "He failed the exam.", "contradiction"),
    ("The store is open on Sundays.", "The store is closed every Sunday.", "contradiction"),
]


def _run_nli(pipe: any, premise: str, hypothesis: str) -> str:
    result = pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
    label_raw = result[0]["label"].lower()
    return _MNLI_LABEL_MAP.get(label_raw, label_raw)


def run() -> dict:
    utils.set_all_seeds()
    print("\n[Task B] Running pipeline sanity check...")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=config.AUDITOR_MODEL,
        device=device,
        truncation=True,
    )

    # Verify id2label
    id2label: dict = pipe.model.config.id2label
    print(f"  id2label: {id2label}")
    assert len(id2label) == 3, f"Expected 3 labels, got {id2label}"

    utils.save_json(
        config.MODEL_CONFIG_JSON,
        {"id2label": {str(k): v for k, v in id2label.items()}},
        n_examples=0,
    )

    # Hand-written pairs
    hand_results = []
    failures = []
    for premise, hypothesis, expected in _HAND_PAIRS:
        predicted = _run_nli(pipe, premise, hypothesis)
        correct = predicted == expected
        hand_results.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        })
        if not correct:
            failures.append(f"  FAIL: '{hypothesis}' | expected={expected} got={predicted}")

    if failures:
        for f in failures:
            print(f)
        raise AssertionError(f"Hand-written sanity check failed on {len(failures)}/10 pairs")
    print(f"  Hand-written pairs: 10/10 correct")

    # MNLI-mismatched 500-sample
    mnli = load_dataset("nyu-mll/multi_nli", split="validation_mismatched")
    rng = utils.np.random.default_rng(config.SEED)
    indices = rng.choice(len(mnli), size=500, replace=False).tolist()
    subset = mnli.select(indices)

    label_str = {0: "entailment", 1: "neutral", 2: "contradiction"}
    y_true, y_pred = [], []
    for ex in subset:
        pred = _run_nli(pipe, ex["premise"], ex["hypothesis"])
        y_true.append(label_str[ex["label"]])
        y_pred.append(pred)

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"  MNLI-mismatched 500-sample accuracy: {acc:.4f}")
    assert 0.85 <= acc <= 0.93, f"MNLI accuracy {acc:.4f} outside expected range [0.85, 0.93]"

    metrics = utils.compute_metrics(y_true, y_pred)
    result = {
        "hand_written_pairs": hand_results,
        "hand_written_correct": sum(r["correct"] for r in hand_results),
        "mnli_mismatched_accuracy": acc,
        **metrics,
    }
    utils.save_json(config.SANITY_CHECK_JSON, result, n_examples=500)
    print(f"  Sanity check passed. Macro-F1: {metrics['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    run()
