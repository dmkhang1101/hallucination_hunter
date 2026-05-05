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

# Whether to include NEI (Not Enough Info) claims as `neutral` examples.
# SciFact NEI = "the cited paper neither supports nor contradicts the claim",
# which aligns with NLI's neutral definition. Including them:
#   - nearly doubles the training set (505 -> ~840 train pairs)
#   - revives the neutral logit during fine-tuning (otherwise it's dead and
#     the model loses its "uncertain" safety net at inference)
INCLUDE_NEI_AS_NEUTRAL: bool = True

# Caches corpus + both splits in a single download
_scifact_raw: dict | None = None


def _ensure_scifact_loaded() -> None:
    """Download and parse SciFact corpus + train + dev (runs once per process)."""
    global _scifact_raw
    if _scifact_raw is not None:
        return

    print("  Downloading SciFact from official release...")
    with urllib.request.urlopen(_SCIFACT_URL, timeout=60) as resp:
        raw = resp.read()

    corpus: dict[int, str] = {}
    splits: dict[str, list[dict]] = {"train": [], "dev": []}

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
            elif name.endswith("claims_train.jsonl") and "cross_validation" not in name:
                for line in lines:
                    splits["train"].append(json.loads(line))
            elif name.endswith("claims_dev.jsonl") and "cross_validation" not in name:
                for line in lines:
                    splits["dev"].append(json.loads(line))

    _scifact_raw = {"corpus": corpus, "splits": splits}
    print(f"  SciFact loaded: {len(splits['train'])} train / {len(splits['dev'])} dev claims")


def _extract_nli_pairs(
    claims: list[dict], corpus: dict[int, str]
) -> tuple[list[str], list[str], list[str]]:
    """Convert raw SciFact claim dicts to (premises, hypotheses, labels).

    Fans out multi-document evidence (one pair per claim x cited abstract)
    instead of stopping at the first doc — earlier behavior silently dropped
    ~10% of the SUPPORT/CONTRADICT pairs.

    When INCLUDE_NEI_AS_NEUTRAL is True, claims with no evidence but with
    cited_doc_ids are emitted as `neutral` examples (paired with each cited
    abstract). NEI claims are ~38% of SciFact and provide essential gradient
    signal to the neutral logit during fine-tuning.
    """
    premises, hypotheses, labels = [], [], []
    for claim in claims:
        claim_text = str(claim["claim"])
        evidence: dict = claim.get("evidence", {})

        if evidence:
            # SUPPORT / CONTRADICT — fan out across all cited docs.
            for doc_id_str, ev_list in evidence.items():
                if not ev_list:
                    continue
                nli_label = _SCIFACT_LABEL_MAP.get(ev_list[0].get("label", ""))
                if nli_label is None:
                    continue
                abstract = corpus.get(int(doc_id_str), "")
                if not abstract:
                    continue
                premises.append(abstract)
                hypotheses.append(claim_text)
                labels.append(nli_label)
        elif INCLUDE_NEI_AS_NEUTRAL:
            # NEI — claim cites a paper but no clear support/contradict.
            cited = claim.get("cited_doc_ids") or []
            for doc_id in cited:
                abstract = corpus.get(int(doc_id), "")
                if not abstract:
                    continue
                premises.append(abstract)
                hypotheses.append(claim_text)
                labels.append("neutral")
    return premises, hypotheses, labels


def load_scifact_train() -> tuple[list[str], list[str], list[str]]:
    """Return SciFact training split as (premises, hypotheses, labels)."""
    _ensure_scifact_loaded()
    assert _scifact_raw is not None
    pairs = _extract_nli_pairs(_scifact_raw["splits"]["train"], _scifact_raw["corpus"])
    print(f"  SciFact train: {len(pairs[0])} labelled pairs")
    return pairs


def load_scifact_test() -> tuple[list[str], list[str], list[str]]:
    """Return SciFact dev split as (premises, hypotheses, labels)."""
    _ensure_scifact_loaded()
    assert _scifact_raw is not None
    pairs = _extract_nli_pairs(_scifact_raw["splits"]["dev"], _scifact_raw["corpus"])
    print(f"  SciFact test: {len(pairs[0])} labelled claims loaded")
    return pairs


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
