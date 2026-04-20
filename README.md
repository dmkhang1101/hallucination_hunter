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
│       ├── build_gold_set.py       # A: build gold set from annotations
│       ├── verify_auditor.py       # B: sanity-check NLI pipeline
│       ├── evaluate_scifact.py     # C: RQ1 — SciFact transfer eval
│       ├── evaluate_truthfulqa.py  # D: RQ2 — TruthfulQA grounded eval
│       ├── run_baselines.py        # E: TF-IDF + S-BERT baselines
│       ├── analyze_subtypes.py     # F: RQ3 — hallucination subtype analysis
│       ├── extract_errors.py       # G: error analysis + FP explanation
│       └── predict_all_claims.py   # H: NLI predictions for all 400+ claims
├── src/member_b/           # Member B exploratory scripts (see notes below)
├── results/
│   ├── gold/
│   │   └── pilot_gold.json                     # 50-claim annotated gold set
│   ├── eval/
│   │   ├── scifact_eval.json                   # SciFact transfer metrics
│   │   ├── truthfulqa_grounded.json            # TruthfulQA grounded metrics
│   │   ├── truthfulqa_ungrounded.json          # TruthfulQA ungrounded metrics
│   │   ├── grounded_vs_ungrounded.json         # McNemar comparison
│   │   ├── subtype_analysis.json               # Per-subtype hallucination recall
│   │   ├── sanity_check.json
│   │   └── model_config.json
│   ├── baselines/
│   │   ├── baselines.json                      # TF-IDF + S-BERT Macro-F1 + McNemar
│   │   ├── baseline_summary.json               # Member B lexical baseline summary
│   │   └── lexical_baseline_results.csv        # Member B lexical scores (all 402 claims)
│   ├── error_analysis/
│   │   ├── error_samples.csv                   # FN + FP error samples
│   │   └── fp_analysis.json                    # Explained false positives
│   ├── figures/
│   │   ├── pilot_confusion.png                 # Confusion matrix on pilot set
│   │   ├── pilot_f1_accuracy.png               # F1 + accuracy bar chart
│   │   ├── scifact_confusion.png               # SciFact confusion matrix
│   │   ├── subtype_recall.png                  # Recall by hallucination subtype
│   │   └── ...                                 # Member B visualizations
│   └── predictions/
│       └── all_claims_predictions.csv          # DeBERTa NLI labels for all 402 claims
└── 50_sample_annotation.csv   # Authoritative gold-labeled 50-claim pilot set
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

| Task | Dataset | Macro-F1 | Accuracy |
|------|---------|----------|----------|
| C — SciFact transfer (RQ1) | SciFact dev (188 claims) | 0.706 | 0.660 |
| D — TruthfulQA grounded (RQ2) | Pilot gold (50 claims) | 0.499 | 0.600 |

**Baseline comparison (pilot set, 50 claims):**

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| DeBERTa auditor | **0.499** | **0.60** |
| S-BERT cosine | 0.444 | 0.48 |
| TF-IDF cosine | 0.332 | 0.34 |

**Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (zero-shot, no fine-tuning)

Confusion matrices and per-class F1 charts are in `results/figures/`.

---

## Notes on Member Contributions

Member B (`src/member_b/`) contributed exploratory scripts and visualizations merged via PR #2. Overlaps with the audit pipeline:

- **Lexical baseline** — `src/member_b/lexical_baseline.py` runs TF-IDF + Jaccard on all 402 claims but without gold labels (no accuracy/F1). **`audit/tasks/run_baselines.py` (Task E) supersedes this** with proper Macro-F1 evaluation, S-BERT comparison, and McNemar significance testing. Use `results/baselines/baselines.json` as the canonical baseline.
- **All-claims predictions** — `results/baselines/lexical_baseline_results.csv` (Member B, lexical scores) and `results/predictions/all_claims_predictions.csv` (Member A, DeBERTa NLI) are complementary outputs covering the same 402 claims with different models.
- **Annotation data** — `50_sample_annotation.csv` at the repo root is the single authoritative pilot annotation file with adjudicated `Gold_Label` and `Gold_Source` columns.
