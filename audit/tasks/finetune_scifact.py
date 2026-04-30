"""
Task I — Fine-tune DeBERTa on SciFact training data, evaluate on dev set,
and compare against zero-shot results from Task C.

Targeted improvement: boost recall on the `contradiction` class (technical
hallucinations) via per-epoch evaluation, best-checkpoint selection on
macro-F1, and stability-tuned hyperparameters (very low LR + tight gradient
clipping) chosen to avoid the classifier-head collapse that DeBERTa-v3-large
exhibits when fine-tuned on small, in-distribution datasets.

Output:
  results/finetuned_model/          — saved HuggingFace model + tokenizer
  results/eval/scifact_finetuned_eval.json
  results/eval/scifact_comparison.json
  results/figures/scifact_finetuned_confusion.png
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

import config
import utils


class _NLIDataset(Dataset):
    def __init__(self, encodings: dict, label_ids: list[int]) -> None:
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[idx])
        return item


def _build_compute_metrics(id2label: dict[int, str]):
    """Return a HF Trainer-compatible compute_metrics callback.

    Surfaces accuracy, macro-F1, and contradiction_recall.
    """
    contra_id = next((i for i, name in id2label.items() if name == "contradiction"), None)

    def _compute(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if contra_id is None:
            contra_recall = 0.0
        else:
            contra_recall = recall_score(
                labels, preds, labels=[contra_id], average="macro", zero_division=0
            )
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "contradiction_recall": float(contra_recall),
        }

    return _compute


def _plot_confusion(cm: list[list[int]], labels: list[str], path: Path) -> None:
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
    ax.set_title("SciFact Confusion Matrix (Fine-tuned)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def _get_pipeline_device() -> int:
    """ROCm-enabled PyTorch exposes AMD GPUs through the torch.cuda API."""
    return 0 if torch.cuda.is_available() else -1


def _get_precision_flags() -> dict[str, bool]:
    """Force fp32 during training.

    Mixed precision (bf16/fp16) was destabilizing the classifier head when
    fine-tuning DeBERTa-v3-large on SciFact — runs collapsed to predicting
    a single class. fp32 is slower but stable; on Lightning A100/H100 a
    DeBERTa-large run on 894 examples × 2 epochs is still <10min.
    """
    return {"bf16": False, "fp16": False}


def _predict_with_pipeline(
    model_path: str, premises: list[str], hypotheses: list[str]
) -> list[str]:
    # IMPORTANT: must mirror the training-time tokenization. Training uses
    # `tokenizer(premises, hypotheses)` which produces proper
    # `[CLS] premise [SEP] hypothesis [SEP]` with token_type_ids. Earlier
    # versions concatenated `f"{p} [SEP] {h}"` into a single string — the
    # literal "[SEP]" is NOT recognized as the special token by SentencePiece
    # tokenizers, so the model received malformed inputs and collapsed to the
    # majority class. Passing dict-style {"text", "text_pair"} routes through
    # the pipeline's pair-aware tokenization path.
    pipe = pipeline(
        "text-classification",
        model=model_path,
        device=_get_pipeline_device(),
        truncation=True,
    )
    paired_inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    predicted: list[str] = []
    for i in range(0, len(paired_inputs), 16):
        batch = paired_inputs[i : i + 16]
        out = pipe(batch, truncation=True, max_length=512)
        for item in out:
            predicted.append(item["label"].lower())
    return predicted


def run() -> dict:
    utils.set_all_seeds()
    print("\n[Task I] Fine-tuning DeBERTa on SciFact training data...")

    if torch.cuda.is_available():
        print(f"  Using accelerator: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU detected — fine-tuning will be very slow on CPU.")

    # Back up an existing finetuned eval (likely a prior run) so we don't lose it.
    if config.SCIFACT_FINETUNED_EVAL_JSON.exists():
        backup = config.SCIFACT_FINETUNED_EVAL_JSON.with_suffix(
            ".zeroshot_run_backup.json"
        )
        shutil.copy2(config.SCIFACT_FINETUNED_EVAL_JSON, backup)
        print(f"  Backed up existing eval → {backup}")

    # --- Load SciFact train + dev ---
    train_premises, train_hypotheses, train_labels = utils.load_scifact_train()
    dev_premises, dev_hypotheses, dev_labels = utils.load_scifact_test()
    print(f"  Train examples: {len(train_premises)} | Dev examples: {len(dev_premises)}")

    tokenizer = AutoTokenizer.from_pretrained(config.AUDITOR_MODEL)

    # Use the base model's existing label2id so the classification head stays aligned
    base_model = AutoModelForSequenceClassification.from_pretrained(config.AUDITOR_MODEL)
    label2id: dict[str, int] = {k.lower(): v for k, v in base_model.config.label2id.items()}
    id2label: dict[int, str] = {v: k for k, v in label2id.items()}

    train_label_ids = [label2id[lbl] for lbl in train_labels]
    dev_label_ids = [label2id[lbl] for lbl in dev_labels]

    # Tokenize train and dev separately — different padding lengths are fine
    # because HF data collator handles per-batch padding.
    train_encodings = tokenizer(
        train_premises,
        train_hypotheses,
        truncation=True,
        max_length=512,
        padding=True,
    )
    dev_encodings = tokenizer(
        dev_premises,
        dev_hypotheses,
        truncation=True,
        max_length=512,
        padding=True,
    )
    train_dataset = _NLIDataset(train_encodings, train_label_ids)
    dev_dataset = _NLIDataset(dev_encodings, dev_label_ids)

    precision_flags = _get_precision_flags()

    # Hyperparameters chosen to defuse a specific failure mode observed on
    # DeBERTa-v3-large + SciFact: the very first training batches produce
    # absurd gradient norms (observed: 99 on step 1, 515 on step 9). Default
    # `max_grad_norm=1.0` clips the *update* but AdamW's variance estimator
    # absorbs the raw magnitudes, polluting subsequent steps and collapsing
    # the classifier head to near-uniform output (training loss settles at
    # ln(3) ≈ 1.10, eval macro-F1 ≈ 0.20).
    #   - learning_rate=5e-7: 6× lower — tiny nudges only, the base is
    #     already at its NLI optimum
    #   - warmup_steps=100: ~45% of total steps, gives Adam's variance time
    #     to stabilize on small gradients before LR ramps up
    #   - max_grad_norm=0.3: aggressive clipping; keeps Adam's v_t bounded
    #     even on outlier batches
    training_args = TrainingArguments(
        output_dir=str(config.FINETUNED_MODEL_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-7,
        weight_decay=0.0,
        warmup_steps=100,
        max_grad_norm=0.3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        seed=config.SEED,
        fp16=precision_flags["fp16"],
        bf16=precision_flags["bf16"],
        report_to="none",
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=_build_compute_metrics(id2label),
    )

    # Pre-training eval — confirms the base model is healthy on this dev
    # set before any gradient steps. If these numbers don't roughly match
    # zero-shot eval, model loading is broken (not training).
    pre_metrics = trainer.evaluate()
    print(
        "  Pre-training eval: "
        f"acc={pre_metrics.get('eval_accuracy', 0):.4f} | "
        f"macro_f1={pre_metrics.get('eval_macro_f1', 0):.4f} | "
        f"contra_recall={pre_metrics.get('eval_contradiction_recall', 0):.4f}"
    )

    trainer.train()
    trainer.save_model(str(config.FINETUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(config.FINETUNED_MODEL_DIR))
    print(f"  Fine-tuned model saved → {config.FINETUNED_MODEL_DIR}")

    # --- Evaluate fine-tuned model on SciFact dev ---
    print("  Evaluating fine-tuned model on SciFact dev set...")
    ft_preds = _predict_with_pipeline(
        str(config.FINETUNED_MODEL_DIR), dev_premises, dev_hypotheses
    )
    active_labels = sorted(set(dev_labels))
    ft_metrics = utils.compute_metrics(dev_labels, ft_preds, labels=active_labels)

    # Surface contradiction_recall as a top-level field for easy comparison.
    contra_recall = (
        ft_metrics["per_class"].get("contradiction", {}).get("recall", 0.0)
    )
    ft_metrics["contradiction_recall"] = float(contra_recall)

    _plot_confusion(
        ft_metrics["confusion_matrix"]["matrix"],
        ft_metrics["confusion_matrix"]["labels"],
        config.SCIFACT_FINETUNED_CONFUSION_PNG,
    )
    utils.save_json(
        config.SCIFACT_FINETUNED_EVAL_JSON,
        ft_metrics,
        n_examples=len(dev_premises),
        model_name=f"finetuned:{config.AUDITOR_MODEL}",
    )
    print(
        f"  Fine-tuned Macro-F1: {ft_metrics['macro_f1']:.4f} | "
        f"Accuracy: {ft_metrics['accuracy']:.4f} | "
        f"Contradiction recall: {contra_recall:.4f}"
    )

    # --- Build comparison against zero-shot results ---
    comparison: dict = {
        "zero_shot": None,
        "finetuned": {
            "macro_f1": ft_metrics["macro_f1"],
            "accuracy": ft_metrics["accuracy"],
            "contradiction_recall": float(contra_recall),
            "per_class": ft_metrics["per_class"],
        },
    }
    if config.SCIFACT_EVAL_JSON.exists():
        with open(config.SCIFACT_EVAL_JSON) as fh:
            zs = json.load(fh)
        zs_contra_recall = (
            zs.get("per_class", {}).get("contradiction", {}).get("recall", 0.0)
        )
        comparison["zero_shot"] = {
            "macro_f1": zs.get("macro_f1"),
            "accuracy": zs.get("accuracy"),
            "contradiction_recall": float(zs_contra_recall),
            "per_class": zs.get("per_class"),
        }
        delta_f1 = ft_metrics["macro_f1"] - float(zs.get("macro_f1", 0))
        delta_acc = ft_metrics["accuracy"] - float(zs.get("accuracy", 0))
        delta_contra = float(contra_recall) - float(zs_contra_recall)
        comparison["delta_macro_f1"] = float(delta_f1)
        comparison["delta_accuracy"] = float(delta_acc)
        comparison["delta_contradiction_recall"] = float(delta_contra)
        print(
            f"  Delta vs zero-shot — Macro-F1: {delta_f1:+.4f} | "
            f"Accuracy: {delta_acc:+.4f} | "
            f"Contradiction recall: {delta_contra:+.4f}"
        )

    utils.save_json(
        config.SCIFACT_COMPARISON_JSON,
        comparison,
        n_examples=len(dev_premises),
        model_name=f"comparison:{config.AUDITOR_MODEL}",
    )
    print(f"  Comparison saved → {config.SCIFACT_COMPARISON_JSON}")

    return ft_metrics


if __name__ == "__main__":
    run()
