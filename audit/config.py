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

# Outputs
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)
PREDICTIONS = RESULTS / "predictions"
PREDICTIONS.mkdir(exist_ok=True)

PILOT_GOLD_JSON = RESULTS / "pilot_gold.json"
MODEL_CONFIG_JSON = RESULTS / "model_config.json"
SANITY_CHECK_JSON = RESULTS / "sanity_check.json"
SCIFACT_EVAL_JSON = RESULTS / "scifact_eval.json"
SCIFACT_CONFUSION_PNG = RESULTS / "scifact_confusion.png"
TQA_GROUNDED_JSON = RESULTS / "truthfulqa_grounded.json"
TQA_UNGROUNDED_JSON = RESULTS / "truthfulqa_ungrounded.json"
TQA_COMPARISON_JSON = RESULTS / "grounded_vs_ungrounded.json"
BASELINES_JSON = RESULTS / "baselines.json"
SUBTYPE_ANALYSIS_JSON = RESULTS / "subtype_analysis.json"
SUBTYPE_RECALL_PNG = RESULTS / "subtype_recall.png"
ERROR_SAMPLES_CSV = RESULTS / "error_samples.csv"

ALL_CLAIMS_PREDICTIONS_CSV = PREDICTIONS / "all_claims_predictions.csv"

LABEL_CLASSES = ["entailment", "neutral", "contradiction"]
