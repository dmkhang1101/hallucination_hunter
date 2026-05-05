import pandas as pd
import json
import os
from scipy.stats import chi2

os.makedirs('results/eval', exist_ok=True)

print("TASK 2 - SIGNIFICANCE TESTING")

baseline_df = pd.read_csv('results/baselines/scaled_baseline_results.csv')
print(f"Baseline: {len(baseline_df)} rows")

auditor_df = pd.read_csv('results/predictions/truthfulqa_test_finetuned_predictions.csv')
print(f"Auditor: {len(auditor_df)} rows")
print(f"Auditor columns: {auditor_df.columns.tolist()}")

def normalize_label(label):
    """Convert various label formats to standard 'Entailment', 'Contradiction', 'Neutral'"""
    if pd.isna(label):
        return 'Neutral'
    label_str = str(label).lower().strip()
    
    if 'entail' in label_str:
        return 'Entailment'
    elif 'contradict' in label_str:
        return 'Contradiction'
    elif 'neutral' in label_str:
        return 'Neutral'
    else:
        return 'Neutral'

merged = baseline_df.merge(auditor_df[['claim_id', 'predicted_label']], on='claim_id', how='inner')
print(f"Merged: {len(merged)} rows")

merged['gold_norm'] = merged['gold_label'].apply(normalize_label)
merged['baseline_norm'] = merged['prediction'].apply(normalize_label)
merged['auditor_norm'] = merged['predicted_label'].apply(normalize_label)

label_map = {'Entailment': 0, 'Contradiction': 1, 'Neutral': 2}
y_true = [label_map[l] for l in merged['gold_norm']]
y_baseline = [label_map[l] for l in merged['baseline_norm']]
y_auditor = [label_map[l] for l in merged['auditor_norm']]

baseline_correct = [1 if y_true[i] == y_baseline[i] else 0 for i in range(len(y_true))]
auditor_correct = [1 if y_true[i] == y_auditor[i] else 0 for i in range(len(y_true))]

baseline_errors = [1 - c for c in baseline_correct]
auditor_errors = [1 - c for c in auditor_correct]

both_wrong = sum(1 for i in range(len(baseline_errors)) if baseline_errors[i] == 1 and auditor_errors[i] == 1)
baseline_only_wrong = sum(1 for i in range(len(baseline_errors)) if baseline_errors[i] == 1 and auditor_errors[i] == 0)
auditor_only_wrong = sum(1 for i in range(len(baseline_errors)) if baseline_errors[i] == 0 and auditor_errors[i] == 1)
both_right = sum(1 for i in range(len(baseline_errors)) if baseline_errors[i] == 0 and auditor_errors[i] == 0)

if (baseline_only_wrong + auditor_only_wrong) > 0:
    chi2_stat = (abs(baseline_only_wrong - auditor_only_wrong) - 1)**2 / (baseline_only_wrong + auditor_only_wrong)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
else:
    p_value = 1.0

baseline_acc = sum(baseline_correct) / len(baseline_correct)
auditor_acc = sum(auditor_correct) / len(auditor_correct)

print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
print(f"Auditor Accuracy: {auditor_acc:.4f}")
print(f"Improvement: {auditor_acc - baseline_acc:+.4f}")
print(f"McNemar p-value: {p_value:.6f}")
print(f"Statistically significant (p<0.05): {p_value < 0.05}")

results = {
    'comparison_lexical_vs_finetuned_auditor': {
        'baseline_accuracy': float(baseline_acc),
        'auditor_accuracy': float(auditor_acc),
        'improvement': float(auditor_acc - baseline_acc),
        'mcnemar_p_value': float(p_value),
        'statistically_significant': str(p_value < 0.05),  # Convert to string
        'total_claims': len(merged),
        'contingency_table': {
            'both_correct': int(both_right),
            'baseline_only_correct': int(auditor_only_wrong),
            'auditor_only_correct': int(baseline_only_wrong),
            'both_wrong': int(both_wrong)
        }
    }
}

with open('results/eval/significance_testing.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nContingency Table:")
print(f"  Both correct: {both_right}")
print(f"  Baseline only correct: {baseline_only_wrong}")
print(f"  Auditor only correct: {auditor_only_wrong}")
print(f"  Both wrong: {both_wrong}")

print("\nSaved: results/eval/significance_testing.json")