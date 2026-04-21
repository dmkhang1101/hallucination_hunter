"""
Central configuration — paths, model names, seed.
"""

from pathlib import Path

ROOT = Path(__file__).parent
REPO = ROOT.parent
SEED = 42
AUDITOR_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Inputs
PILOT_CSV = REPO / "50_sample_annotation.csv"
GEN_DATA = REPO / "generate_data" / "data"
PRIMARY_ANSWERS = GEN_DATA / "primary_answers.csv"
ATOMIC_CLAIMS = GEN_DATA / "atomic_claims.csv"
SAMPLED_Q = GEN_DATA / "sampled_questions.csv"
TQA_TRAIN = GEN_DATA / "split" / "truthfulqa_train.csv"
TQA_TEST = GEN_DATA / "split" / "truthfulqa_test.csv"

# Output root
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# Subfolders
GOLD_DIR = RESULTS / "gold"
EVAL_DIR = RESULTS / "eval"
BASELINES_DIR = RESULTS / "baselines"
ERROR_DIR = RESULTS / "error_analysis"
FIGURES_DIR = RESULTS / "figures"
PREDICTIONS_DIR = RESULTS / "predictions"

for _d in (GOLD_DIR, EVAL_DIR, BASELINES_DIR, ERROR_DIR, FIGURES_DIR, PREDICTIONS_DIR):
    _d.mkdir(exist_ok=True)

# gold/
PILOT_GOLD_JSON = GOLD_DIR / "pilot_gold.json"

# eval/
MODEL_CONFIG_JSON = EVAL_DIR / "model_config.json"
SANITY_CHECK_JSON = EVAL_DIR / "sanity_check.json"
SCIFACT_EVAL_JSON = EVAL_DIR / "scifact_eval.json"
TQA_GROUNDED_JSON = EVAL_DIR / "truthfulqa_grounded.json"
TQA_UNGROUNDED_JSON = EVAL_DIR / "truthfulqa_ungrounded.json"
TQA_COMPARISON_JSON = EVAL_DIR / "grounded_vs_ungrounded.json"
SUBTYPE_ANALYSIS_JSON = EVAL_DIR / "subtype_analysis.json"

# baselines/
BASELINES_JSON = BASELINES_DIR / "baselines.json"

# error_analysis/
ERROR_SAMPLES_CSV = ERROR_DIR / "error_samples.csv"
FP_ANALYSIS_JSON = ERROR_DIR / "fp_analysis.json"

# figures/
SCIFACT_CONFUSION_PNG = FIGURES_DIR / "scifact_confusion.png"
SUBTYPE_RECALL_PNG = FIGURES_DIR / "subtype_recall.png"
PILOT_CONFUSION_PNG = FIGURES_DIR / "pilot_confusion.png"
PILOT_F1_ACCURACY_PNG = FIGURES_DIR / "pilot_f1_accuracy.png"

# predictions/
ALL_CLAIMS_PREDICTIONS_CSV = PREDICTIONS_DIR / "all_claims_predictions.csv"

LABEL_CLASSES = ["entailment", "neutral", "contradiction"]

# Keep PREDICTIONS as alias for backwards compatibility with main.py skip logic
PREDICTIONS = PREDICTIONS_DIR
