#task 1: 
import pandas as pd
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.member_b.lexical_baseline import LexicalHallucinationDetector

pilot_df = pd.read_csv('all_claims_annotation_complete.csv')
test_df = pd.read_csv('truthfulqa_test_annotated.csv')

print(f"Pilot claims: {len(pilot_df)}")
print(f"Test claims: {len(test_df)}")

combined_df = pd.concat([pilot_df, test_df], ignore_index=True)
print(f"\nTotal claims: {len(combined_df)}")

detector = LexicalHallucinationDetector(threshold=0.3)
results = []

for idx, row in combined_df.iterrows():
    prediction, scores = detector.predict(row['atomic_claim'], row['ground_truth_reference'])
    results.append({
        'claim_id': row['claim_id'],
        'question_id': row['question_id'],
        'category': row['category'],
        'gold_label': row['Gold_Label'],
        'prediction': prediction,
        'token_overlap': scores['token_overlap'],
        'tfidf_similarity': scores['tfidf_similarity'],
        'combined_score': scores['combined']
    })

os.makedirs('results/baselines', exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv('results/baselines/scaled_baseline_results.csv', index=False)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

label_map = {'Entailment': 0, 'Contradiction': 1, 'Neutral': 2}
y_true = [label_map.get(l, 2) for l in results_df['gold_label']]
y_pred = [label_map.get(l, 2) for l in results_df['prediction']]

print("SCALED BASELINE RESULTS (1,075 claims)")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Entailment', 'Contradiction', 'Neutral']))

metrics = {
    'total_claims': len(combined_df),
    'accuracy': accuracy_score(y_true, y_pred),
    'pilot_claims': len(pilot_df),
    'test_claims': len(test_df)
}
with open('results/baselines/scaled_baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nResults saved to results/baselines/scaled_baseline_results.csv")