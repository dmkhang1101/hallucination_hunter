# Hallucination Auditor

DeBERTa-v3-MNLI pipeline that classifies atomic claims from GPT-4o outputs as entailment / neutral / contradiction.

## Setup

```bash
cd audit/
pip install -r requirements.txt
```

## Run all tasks end-to-end

```bash
python main.py
```

## Run individual tasks

```bash
python task_a_gold.py      # Build pilot gold set
python task_b_sanity.py    # Pipeline sanity check
python task_c_scifact.py   # SciFact eval (RQ1)
python task_d_truthfulqa.py # TruthfulQA grounded vs ungrounded (RQ2)
python task_e_baselines.py  # TF-IDF + S-BERT baselines
python task_f_subtypes.py   # Subtype recall analysis (RQ3)
python task_g_errors.py     # Error analysis template (human fills failure_mode)
```

## Outputs

All results land in `results/`. Key files:

| File | Description |
|------|-------------|
| `pilot_gold.json` | 50-claim gold set with consensus labels |
| `sanity_check.json` | Model identity + MNLI accuracy check |
| `scifact_eval.json` | RQ1: NLI transfer metrics on SciFact |
| `scifact_confusion.png` | Confusion matrix heatmap |
| `truthfulqa_grounded.json` | RQ2: grounded mode metrics |
| `truthfulqa_ungrounded.json` | RQ2: ungrounded mode metrics |
| `grounded_vs_ungrounded.json` | McNemar's test result |
| `baselines.json` | TF-IDF and S-BERT comparison |
| `subtype_analysis.json` | RQ3: per-subtype recall |
| `subtype_recall.png` | Bar chart |
| `error_samples.csv` | 30 errors for manual annotation |

## Critical invariants

- Seed 42 everywhere.
- TruthfulQA splits in `generate_data/data/split/` are frozen — never re-split.
- Headline metric: macro-F1. Accuracy is secondary.
