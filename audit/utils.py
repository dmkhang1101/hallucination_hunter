"""
Shared helpers: label normalization, metrics, seed setting, JSON saving, SciFact loader.
"""

import io
import json
import random
import re
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config


def set_all_seeds(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(s: str) -> str | None:
    """Strip parenthetical subtype, lowercase — e.g. 'Contradiction (Factual Error)' → 'contradiction'."""
    if not isinstance(s, str) or not s.strip():
        return None
    base = re.sub(r"\s*\(.*?\)", "", s).strip().lower()
    if base in {"entailment", "neutral", "contradiction"}:
        return base
    return None


def extract_subtype(s: str) -> str | None:
    """Return parenthetical content or None — e.g. 'Contradiction (Factual Error)' → 'Factual Error'."""
    if not isinstance(s, str):
        return None
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        val = m.group(1).strip()
        return None if val.lower() == "none" else val
    return None


def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> dict[str, Any]:
    """Per-class P/R/F1, macro-F1, accuracy, confusion matrix."""
    if labels is None:
        labels = config.LABEL_CLASSES
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = {
        lab: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lab in enumerate(labels)
    }
    return {
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


_SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
_SCIFACT_LABEL_MAP = {"SUPPORT": "entailment", "CONTRADICT": "contradiction"}

_scifact_cache: dict | None = None


def load_scifact_test() -> tuple[list[str], list[str], list[str]]:
    """Download SciFact directly from official S3 release; return (premises, hypotheses, labels)."""
    global _scifact_cache
    if _scifact_cache is not None:
        return _scifact_cache["premises"], _scifact_cache["hypotheses"], _scifact_cache["labels"]

    print("  Downloading SciFact from official release...")
    with urllib.request.urlopen(_SCIFACT_URL, timeout=60) as resp:
        raw = resp.read()

    corpus: dict[int, str] = {}
    claims_test: list[dict] = []

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            f = tar.extractfile(member)
            if f is None:
                continue
            lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
            if name.endswith("corpus.jsonl"):
                for line in lines:
                    obj = json.loads(line)
                    abstract = obj.get("abstract", [])
                    text = " ".join(abstract) if isinstance(abstract, list) else str(abstract)
                    corpus[int(obj["doc_id"])] = text
            elif name.endswith("claims_dev.jsonl") and "cross_validation" not in name:
                for line in lines:
                    claims_test.append(json.loads(line))

    premises, hypotheses, labels = [], [], []
    for claim in claims_test:
        evidence: dict = claim.get("evidence", {})
        if not evidence:
            continue
        # Take first doc with a label
        for doc_id_str, ev_list in evidence.items():
            doc_id = int(doc_id_str)
            if not ev_list:
                continue
            raw_label = ev_list[0].get("label", "")
            nli_label = _SCIFACT_LABEL_MAP.get(raw_label)
            if nli_label is None:
                continue
            abstract = corpus.get(doc_id, "")
            if not abstract:
                continue
            premises.append(abstract)
            hypotheses.append(str(claim["claim"]))
            labels.append(nli_label)
            break

    _scifact_cache = {"premises": premises, "hypotheses": hypotheses, "labels": labels}
    print(f"  SciFact test: {len(premises)} labelled claims loaded")
    return premises, hypotheses, labels


def save_json(
    path: Path,
    data: dict[str, Any],
    n_examples: int,
    model_name: str = config.AUDITOR_MODEL,
) -> None:
    """Write data to JSON, auto-injecting provenance fields."""
    payload = {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config.SEED,
        "n_examples": n_examples,
        **data,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  Saved → {path}")
