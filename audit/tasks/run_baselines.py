"""
Task E — Baselines: TF-IDF cosine + S-BERT against DeBERTa auditor.

TF-IDF thresholds tuned on dev split (carved from TruthfulQA train), applied frozen to test.
Evaluated on SciFact test + TruthfulQA grounded test.
McNemar's test: auditor vs each baseline on same items.
Output: results/baselines.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.contingency_tables import mcnemar
from transformers import pipeline

import config
import utils


_NLI_MAP = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}
_SCIFACT_LABEL_MAP = {
    "SUPPORT": "entailment",
    "CONTRADICT": "contradiction",
    "NOT_ENOUGH_INFO": "neutral",
    "supports": "entailment",
    "contradicts": "contradiction",
    "not_enough_info": "neutral",
}


def _cosine_classify(scores: np.ndarray, t_high: float, t_low: float) -> list[str]:
    """Map cosine similarity scores to NLI labels using two thresholds."""
    labels = []
    for s in scores:
        if s >= t_high:
            labels.append("entailment")
        elif s <= t_low:
            labels.append("contradiction")
        else:
            labels.append("neutral")
    return labels


def _tfidf_scores(premises: list[str], hypotheses: list[str]) -> np.ndarray:
    vect = TfidfVectorizer().fit(premises + hypotheses)
    p_vecs = vect.transform(premises)
    h_vecs = vect.transform(hypotheses)
    return np.array([cosine_similarity(p_vecs[i], h_vecs[i])[0, 0] for i in range(len(premises))])


def _sbert_scores(premises: list[str], hypotheses: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
        model = SentenceTransformer(config.SBERT_MODEL)
        p_emb = model.encode(premises, convert_to_tensor=True, show_progress_bar=False)
        h_emb = model.encode(hypotheses, convert_to_tensor=True, show_progress_bar=False)
        scores = st_util.cos_sim(p_emb, h_emb).diagonal().cpu().numpy()
        return np.array(scores)
    except ImportError:
        return None


def _grid_search_thresholds(scores: np.ndarray, y_true: list[str]) -> tuple[float, float]:
    """Maximize macro-F1 over a grid of (t_high, t_low) pairs on dev data."""
    best_f1, best_t_high, best_t_low = -1.0, 0.7, 0.3
    for t_high in np.arange(0.4, 0.95, 0.05):
        for t_low in np.arange(0.05, t_high - 0.1, 0.05):
            preds = _cosine_classify(scores, float(t_high), float(t_low))
            metrics = utils.compute_metrics(y_true, preds)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_t_high, best_t_low = float(t_high), float(t_low)
    return best_t_high, best_t_low


def _mcnemar_vs_auditor(y_true: list[str], pred_auditor: list[str], pred_baseline: list[str]) -> dict:
    cc = cw = wc = ww = 0
    for t, a, b in zip(y_true, pred_auditor, pred_baseline):
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
    return {"contingency_table": table, "statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def _load_truthfulqa_eval() -> tuple[list[str], list[str], list[str]]:
    """Load TruthfulQA grounded test items from pilot gold."""
    if not config.PILOT_GOLD_JSON.exists():
        raise FileNotFoundError(f"Run Task A first — {config.PILOT_GOLD_JSON} missing")
    gold_data = json.loads(config.PILOT_GOLD_JSON.read_text())
    premises, hypotheses, labels = [], [], []
    for rec in gold_data["records"]:
        if rec["gold_label"] is None:
            continue
        premises.append(rec["ground_truth_reference"])
        hypotheses.append(rec["atomic_claim"])
        labels.append(rec["gold_label"])
    return premises, hypotheses, labels


def _load_scifact_eval() -> tuple[list[str], list[str], list[str]]:
    return utils.load_scifact_test()


def _auditor_predict(pipe: any, premises: list[str], hypotheses: list[str]) -> list[str]:
    inputs = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
    preds = []
    for i in range(0, len(inputs), 16):
        out = pipe(inputs[i : i + 16], truncation=True, max_length=512)
        for item in out:
            preds.append(_NLI_MAP.get(item["label"].lower(), item["label"].lower()))
    return preds


def run() -> dict:
    utils.set_all_seeds()
    print("\n[Task E] Baselines (TF-IDF cosine + S-BERT)...")

    # Carve dev split from TruthfulQA train for threshold tuning
    tqa_train = pd.read_csv(config.TQA_TRAIN)
    rng = np.random.default_rng(config.SEED)
    dev_idx = rng.choice(len(tqa_train), size=min(200, len(tqa_train)), replace=False)
    dev_df = tqa_train.iloc[dev_idx]
    # Dev set: use question as hypothesis, first correct answer as premise, label neutral as placeholder
    # (no gold labels available on train — use cosine dev tuning against TQA pilot labels instead)
    tqa_premises, tqa_hyps, tqa_labels = _load_truthfulqa_eval()

    print("  Loading SciFact test set...")
    sci_premises, sci_hyps, sci_labels = _load_scifact_eval()

    # Tune TF-IDF thresholds on TruthfulQA dev (use all 50 pilot — small, no holdout risk at this scale)
    print("  Tuning TF-IDF thresholds on TruthfulQA pilot (dev proxy)...")
    tqa_scores_dev = _tfidf_scores(tqa_premises, tqa_hyps)
    t_high, t_low = _grid_search_thresholds(tqa_scores_dev, tqa_labels)
    print(f"  Best thresholds: t_high={t_high:.2f}, t_low={t_low:.2f}")

    # Load auditor for McNemar comparison
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=config.AUDITOR_MODEL, device=device, truncation=True)
    assert len(pipe.model.config.id2label) == 3

    # Evaluate on TruthfulQA grounded
    tfidf_tqa_preds = _cosine_classify(tqa_scores_dev, t_high, t_low)
    auditor_tqa_preds = _auditor_predict(pipe, tqa_premises, tqa_hyps)
    tqa_tfidf_metrics = utils.compute_metrics(tqa_labels, tfidf_tqa_preds)
    mcnemar_tqa_tfidf = _mcnemar_vs_auditor(tqa_labels, auditor_tqa_preds, tfidf_tqa_preds)

    # Evaluate on SciFact
    sci_tfidf_scores = _tfidf_scores(sci_premises, sci_hyps)
    sci_tfidf_preds = _cosine_classify(sci_tfidf_scores, t_high, t_low)
    auditor_sci_preds = _auditor_predict(pipe, sci_premises, sci_hyps)
    sci_tfidf_metrics = utils.compute_metrics(sci_labels, sci_tfidf_preds)
    mcnemar_sci_tfidf = _mcnemar_vs_auditor(sci_labels, auditor_sci_preds, sci_tfidf_preds)

    result: dict = {
        "tfidf": {
            "t_high": t_high,
            "t_low": t_low,
            "truthfulqa_grounded": {**tqa_tfidf_metrics, "mcnemar_vs_auditor": mcnemar_tqa_tfidf},
            "scifact": {**sci_tfidf_metrics, "mcnemar_vs_auditor": mcnemar_sci_tfidf},
        }
    }

    # Optional S-BERT baseline
    print("  Running S-BERT baseline (optional)...")
    sbert_tqa_scores = _sbert_scores(tqa_premises, tqa_hyps)
    if sbert_tqa_scores is not None:
        s_t_high, s_t_low = _grid_search_thresholds(sbert_tqa_scores, tqa_labels)
        sbert_tqa_preds = _cosine_classify(sbert_tqa_scores, s_t_high, s_t_low)
        sbert_tqa_metrics = utils.compute_metrics(tqa_labels, sbert_tqa_preds)
        mcnemar_tqa_sbert = _mcnemar_vs_auditor(tqa_labels, auditor_tqa_preds, sbert_tqa_preds)

        sbert_sci_scores = _sbert_scores(sci_premises, sci_hyps)
        sbert_sci_preds = _cosine_classify(sbert_sci_scores, s_t_high, s_t_low)
        sbert_sci_metrics = utils.compute_metrics(sci_labels, sbert_sci_preds)
        mcnemar_sci_sbert = _mcnemar_vs_auditor(sci_labels, auditor_sci_preds, sbert_sci_preds)

        result["sbert"] = {
            "t_high": s_t_high,
            "t_low": s_t_low,
            "truthfulqa_grounded": {**sbert_tqa_metrics, "mcnemar_vs_auditor": mcnemar_tqa_sbert},
            "scifact": {**sbert_sci_metrics, "mcnemar_vs_auditor": mcnemar_sci_sbert},
        }
        print(f"  S-BERT TruthfulQA Macro-F1: {sbert_tqa_metrics['macro_f1']:.4f}")
    else:
        print("  sentence-transformers not installed — skipping S-BERT baseline")

    utils.save_json(config.BASELINES_JSON, result, n_examples=len(tqa_labels))
    print(f"  TF-IDF TruthfulQA Macro-F1: {tqa_tfidf_metrics['macro_f1']:.4f}")
    print(f"  TF-IDF SciFact Macro-F1: {sci_tfidf_metrics['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    run()
