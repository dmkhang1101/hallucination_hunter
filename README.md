# Hallucination Hunter
**Dual-Model Auditing: Using NLI to Detect Hallucinations in LLM Outputs**
CS 485 — University of Massachusetts Amherst

---

## Overview

This project audits GPT-4o answers for hallucinations using a DeBERTa-v3-large NLI auditor (zero-shot, plus an optional SciFact-fine-tuned variant). It has two stages:

1. **Data Generation** (`generate_data/`) — sample TruthfulQA questions, generate GPT-4o answers, extract atomic claims (with abbreviation/decimal-aware sentence splitting that prevents mid-claim breaks)
2. **Audit** (`audit/`) — run NLI auditor against gold references, evaluate on SciFact and TruthfulQA, run baselines, and (Task I) fine-tune DeBERTa on SciFact to boost contradiction recall

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
│   ├── main.py             # Entry point — runs Tasks A–I
│   ├── config.py           # All paths and model names
│   ├── utils.py            # Shared utilities (incl. SciFact loader w/ NEI)
│   ├── requirements.txt
│   └── tasks/
│       ├── build_gold_set.py       # A: build gold set from annotations
│       ├── verify_auditor.py       # B: sanity-check NLI pipeline
│       ├── evaluate_scifact.py     # C: RQ1 — SciFact transfer eval
│       ├── evaluate_truthfulqa.py  # D: RQ2 — TruthfulQA grounded eval
│       ├── run_baselines.py        # E: TF-IDF + S-BERT baselines
│       ├── analyze_subtypes.py     # F: RQ3 — hallucination subtype analysis
│       ├── extract_errors.py       # G: error analysis + FP explanation
│       ├── predict_all_claims.py   # H: NLI predictions for all 400+ claims
│       └── finetune_scifact.py     # I: fine-tune DeBERTa on SciFact (Lightning GPU)
├── tests/
│   └── test_finetune_scifact.py    # unit tests for Task I device/precision shims
├── src/member_b/           # Member B exploratory scripts (see notes below)
├── results/
│   ├── gold/
│   │   └── pilot_gold.json                     # 50-claim annotated gold set
│   ├── eval/
│   │   ├── scifact_eval.json                   # SciFact transfer metrics (zero-shot, 3-class)
│   │   ├── scifact_finetuned_eval.json         # Task I — fine-tuned DeBERTa on SciFact dev
│   │   ├── scifact_comparison.json             # zero-shot vs fine-tuned per-class deltas
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
│   │   ├── scifact_confusion.png               # SciFact zero-shot confusion matrix
│   │   ├── scifact_finetuned_confusion.png     # SciFact fine-tuned (Task I) confusion matrix
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

Tasks run in order A → I. Tasks B, C, E, H, I are skipped automatically if their output already exists (model inference is expensive).

**Task I (SciFact fine-tune)** is GPU-heavy and was run on Lightning.ai (A100/H100). It writes a ~5.7 GB fine-tuned checkpoint to `results/finetuned_model/` — that directory is `.gitignore`-d and regenerable; only the eval artifacts (`scifact_finetuned_eval.json`, `scifact_comparison.json`, `scifact_finetuned_confusion.png`) are committed.

---

## Key Results

| Task | Dataset | Macro-F1 | Accuracy | Contradiction recall |
|------|---------|----------|----------|----------------------|
| C — SciFact transfer, zero-shot (RQ1) | SciFact dev (323 pairs, 3-class) | 0.753 | 0.765 | 0.648 |
| I — SciFact fine-tuned                | SciFact dev (323 pairs, 3-class) | **0.870** | **0.876** | **0.859** |
| D — TruthfulQA grounded (RQ2)         | Pilot gold (50 claims)           | 0.499 | 0.600 | — |

**Baseline comparison (pilot set, 50 claims):**

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| DeBERTa auditor | **0.499** | **0.60** |
| S-BERT cosine | 0.444 | 0.48 |
| TF-IDF cosine | 0.332 | 0.34 |

**Auditor model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (zero-shot for Tasks C/D, fine-tuned on SciFact train for Task I).

Confusion matrices and per-class F1 charts are in `results/figures/`. Side-by-side zero-shot vs fine-tuned per-class deltas are in `results/eval/scifact_comparison.json`.

### Notes on the SciFact pipeline

The pre-Task-I SciFact eval reported 188 dev pairs and macro-F1 ≈ 0.706. The current numbers reflect three corrections that landed together:

1. **NEI claims included as `neutral`.** The earlier loader silently dropped ~38% of SciFact (claims that cite a paper but have no evidence label). They're now emitted as 3-class neutral examples, taking dev from 188 → 323 pairs and reviving the neutral logit during fine-tuning.
2. **Multi-document fan-out.** A `break` after the first cited doc dropped ~10% of SUPPORT/CONTRADICT pairs whose claims cite multiple abstracts.
3. **Paired-input tokenization.** Both inference paths previously concatenated `f"{p} [SEP] {h}"`; SentencePiece does not recognize the literal characters `[SEP]` as a special token, so the model received malformed inputs and collapsed toward the majority class. Fixed by passing dict-style `{"text": p, "text_pair": h}` to the HF pipeline.

### Notes on Task I fine-tune stability

DeBERTa-v3-large + SciFact reliably collapsed under default training settings. The recipe that worked is intentionally conservative: fp32 only, `lr=5e-7`, `warmup_steps=100`, `max_grad_norm=0.3`, vanilla `Trainer` (no class-weighted loss), best-checkpoint by macro-F1. The aggressive grad clip is the key knob — early batches produced grad norms of 99 and 515, which the default `max_grad_norm=1.0` does not contain inside AdamW's variance estimator.

---

## Notes on Member Contributions

Member B (`src/member_b/`) contributed exploratory scripts and visualizations merged via PR #2. Overlaps with the audit pipeline:

- **Lexical baseline** — `src/member_b/lexical_baseline.py` runs TF-IDF + Jaccard on all 402 claims but without gold labels (no accuracy/F1). **`audit/tasks/run_baselines.py` (Task E) supersedes this** with proper Macro-F1 evaluation, S-BERT comparison, and McNemar significance testing. Use `results/baselines/baselines.json` as the canonical baseline.
- **All-claims predictions** — `results/baselines/lexical_baseline_results.csv` (Member B, lexical scores) and `results/predictions/all_claims_predictions.csv` (Member A, DeBERTa NLI) are complementary outputs covering the same 402 claims with different models.
- **Annotation data** — `50_sample_annotation.csv` at the repo root is the single authoritative pilot annotation file with adjudicated `Gold_Label` and `Gold_Source` columns.
