# Hallucination Hunter
**Dual-Model Auditing: Using NLI to Detect Hallucinations in LLM Outputs**
CS 485 — University of Massachusetts Amherst

---

## Overview

This project audits GPT-4o answers for hallucinations using a zero-shot NLI model (DeBERTa-v3-large). It has two stages:

1. **Data Generation** (`generate_data/`) — sample TruthfulQA questions, generate GPT-4o answers, extract atomic claims
2. **Audit** (`audit/`) — run NLI auditor against gold references, evaluate on SciFact and TruthfulQA, run baselines

---

## Repository Structure

```
hallucination_hunter/
├── generate_data/          # Stage 1: data generation pipeline
│   ├── main.py             # Entry point
│   ├── generate.py         # GPT-4o answer generation
│   ├── extract.py          # Atomic claim extraction
│   ├── config.py
│   ├── requirements.txt
│   └── data/
│       ├── sampled_questions.csv
│       ├── primary_answers.csv
│       ├── atomic_claims.csv
│       └── split/          # train/test splits
├── audit/                  # Stage 2: NLI audit pipeline
│   ├── main.py             # Entry point — runs Tasks A–H
│   ├── config.py           # All paths and model names
│   ├── utils.py            # Shared utilities
│   ├── requirements.txt
│   └── tasks/
│       ├── build_gold_set.py       # A: build pilot_gold.json from annotations
│       ├── verify_auditor.py       # B: sanity-check NLI pipeline
│       ├── evaluate_scifact.py     # C: RQ1 — SciFact transfer eval
│       ├── evaluate_truthfulqa.py  # D: RQ2 — TruthfulQA grounded eval
│       ├── run_baselines.py        # E: TF-IDF + S-BERT baselines
│       ├── analyze_subtypes.py     # F: RQ3 — hallucination subtype analysis
│       ├── extract_errors.py       # G: error sample extraction
│       └── predict_all_claims.py   # H: NLI predictions for all 400+ claims
├── results/                # All generated outputs
│   ├── pilot_gold.json
│   ├── scifact_eval.json
│   ├── truthfulqa_grounded.json
│   ├── baselines.json
│   ├── subtype_analysis.json
│   ├── error_samples.csv
│   └── predictions/
│       └── all_claims_predictions.csv
└── 50_sample_annotation.csv   # Gold-labeled 50-claim pilot set
```

---

## Setup

### Stage 1 — Data Generation

```bash
cd generate_data
pip install -r requirements.txt
python -m spacy download en_core_web_lg

cp .env.example .env
# Add your OpenAI API key to .env

python main.py              # full run (150 questions, real GPT-4o calls)
python main.py --dry-run    # no API calls, placeholder answers
```

Outputs saved to `generate_data/data/`.

### Stage 2 — Audit

```bash
cd audit
pip install -r requirements.txt

python main.py
```

Tasks run in order A → H. Tasks B, C, E, H are skipped automatically if their output already exists (model inference is expensive).

---

## Key Results

| Task | Dataset | Macro-F1 |
|------|---------|----------|
| C — SciFact transfer (RQ1) | SciFact dev (188 claims) | 0.706 |
| D — TruthfulQA grounded (RQ2) | Pilot gold (50 claims) | 0.499 |

**Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (zero-shot, no fine-tuning)

Full per-class metrics, confusion matrices, and subtype analysis are in `results/`.

---

## Notes on Member Contributions

Member B (`src/member_b/`) contributed exploratory scripts and visualizations that were merged into `main` via PR #2. There is partial overlap with the audit pipeline:

- **Lexical baseline** — `src/member_b/lexical_baseline.py` runs TF-IDF + Jaccard on all 402 claims but produces only a prediction distribution (no gold labels). **`audit/tasks/run_baselines.py` (Task E) supersedes this** with proper Macro-F1 evaluation, S-BERT comparison, and McNemar significance testing against the DeBERTa auditor. Use Task E results (`results/baselines.json`) as the canonical baseline.
- **All-claims predictions** — `results/lexical_baseline_results.csv` (Member B, lexical scores) and `results/predictions/all_claims_predictions.csv` (Member A, DeBERTa NLI) are complementary and cover the same 402 claims with different models.
- **Annotation data** — `results/annotation_sample_50.csv` (Member B) and `50_sample_annotation.csv` (root, Member A) refer to the same pilot set. The root-level file is the authoritative version with adjudicated `Gold_Label` and `Gold_Source` columns.

---

## Outputs Reference

| File | Description |
|------|-------------|
| `results/pilot_gold.json` | 50-claim annotated gold set with adjudications |
| `results/scifact_eval.json` | SciFact NLI transfer metrics + confusion matrix |
| `results/truthfulqa_grounded.json` | TruthfulQA grounded evaluation metrics |
| `results/truthfulqa_ungrounded.json` | TruthfulQA without gold references |
| `results/baselines.json` | TF-IDF and S-BERT baseline comparisons + McNemar test |
| `results/subtype_analysis.json` | Per-subtype recall for hallucination types |
| `results/error_samples.csv` | Sampled false positives / false negatives |
| `results/predictions/all_claims_predictions.csv` | NLI predictions for all 400+ atomic claims |
