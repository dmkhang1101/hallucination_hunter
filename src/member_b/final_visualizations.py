#task 3: final visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import shutil
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs('results/figures/final', exist_ok=True)

print("TASK 3 - FINAL VISUALIZATIONS")

df = pd.read_csv('results/baselines/scaled_baseline_results.csv')
print(f"Loaded {len(df)} claims with gold labels")

def extract_label(label):
    """Extract 'Entailment', 'Contradiction', or 'Neutral' from labels like 'Contradiction (Factual Error)'"""
    if pd.isna(label):
        return 'Neutral'
    label_str = str(label)
    if 'Entailment' in label_str:
        return 'Entailment'
    elif 'Contradiction' in label_str:
        return 'Contradiction'
    elif 'Neutral' in label_str:
        return 'Neutral'
    else:
        return 'Neutral'

df['gold_label_simple'] = df['gold_label'].apply(extract_label)
df['prediction_simple'] = df['prediction'].apply(extract_label)

print("\nLabel distribution after fix:")
print(df['gold_label_simple'].value_counts())

# Confusion matrix for TruthfulQA
print("\n1. Creating TruthfulQA Confusion Matrix...")

label_map = {'Entailment': 0, 'Contradiction': 1, 'Neutral': 2}
y_true = [label_map[l] for l in df['gold_label_simple']]
y_pred = [label_map[l] for l in df['prediction_simple']]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Entailment', 'Contradiction', 'Neutral'],
            yticklabels=['Entailment', 'Contradiction', 'Neutral'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Lexical Baseline - TruthfulQA Confusion Matrix\n(1,075 claims)', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/final/truthfulqa_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: results/figures/final/truthfulqa_confusion_matrix.png")

cm_df = pd.DataFrame(cm, index=['Actual_E', 'Actual_C', 'Actual_N'], 
                     columns=['Pred_E', 'Pred_C', 'Pred_N'])
cm_df.to_csv('results/figures/final/truthfulqa_confusion_matrix.csv')
print("Saved: results/figures/final/truthfulqa_confusion_matrix.csv")

# RECALL BY CATEGORY (38 categories)
print("\n2. Creating Recall by Category Chart...")

category_recall = df.groupby('category').apply(
    lambda x: (x['gold_label_simple'] == x['prediction_simple']).sum() / len(x)
).sort_values(ascending=False)

print(f"Total categories found: {len(category_recall)}")

plt.figure(figsize=(12, 10))
colors = plt.cm.viridis(category_recall.values / category_recall.max() if category_recall.max() > 0 else category_recall.values)
category_recall.plot(kind='barh', color='steelblue')
plt.xlabel('Accuracy', fontsize=12)
plt.title(f'Lexical Baseline - Recall by Category\n({len(category_recall)} categories, {len(df)} claims)', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/final/recall_by_category_38.png', dpi=150)
plt.close()
print(f"Saved: results/figures/final/recall_by_category_38.png")

category_recall.to_csv('results/figures/final/category_recall.csv')
print("Saved: results/figures/final/category_recall.csv")

print("\nTop 5 categories (highest accuracy):")
for cat, acc in category_recall.head(5).items():
    print(f"  {cat}: {acc:.3f}")

print("\nBottom 5 categories (lowest accuracy):")
for cat, acc in category_recall.tail(5).items():
    print(f"  {cat}: {acc:.3f}")

# PILOT vs SCALED PERFORMANCE COMPARISON
print("\n3. Creating Pilot vs Scaled Comparison...")

pilot_df = df.head(402) 
test_df = df.tail(673)  

pilot_acc = (pilot_df['gold_label_simple'] == pilot_df['prediction_simple']).mean()
test_acc = (test_df['gold_label_simple'] == test_df['prediction_simple']).mean()
full_acc = (df['gold_label_simple'] == df['prediction_simple']).mean()

print(f"Pilot accuracy: {pilot_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Full accuracy: {full_acc:.4f}")

plt.figure(figsize=(8, 6))
bars = plt.bar(['Pilot (402 claims)', 'Test (673 claims)', 'Full (1,075 claims)'], 
               [pilot_acc, test_acc, full_acc],
               color=['#ff9999', '#66b3ff', '#99ff99'])
plt.ylim(0, 0.5)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Lexical Baseline Performance: Pilot vs Scaled', fontsize=14)
for bar, acc in zip(bars, [pilot_acc, test_acc, full_acc]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', fontweight='bold')
plt.axhline(y=full_acc, color='red', linestyle='--', alpha=0.5, label=f'Average: {full_acc:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('results/figures/final/pilot_vs_scaled_comparison.png', dpi=150)
plt.close()
print("Saved: results/figures/final/pilot_vs_scaled_comparison.png")

# SUMMARY STATISTICS TABLE
print("\n4. Generating Summary Table...")

from sklearn.metrics import recall_score

y_true_labels = [label_map[l] for l in df['gold_label_simple']]
y_pred_labels = [label_map[l] for l in df['prediction_simple']]

summary = {
    'total_claims': len(df),
    'accuracy': float(full_acc),
    'entailment_recall': float(recall_score(y_true_labels, y_pred_labels, labels=[0], average='micro')) if 0 in y_true_labels else 0,
    'contradiction_recall': float(recall_score(y_true_labels, y_pred_labels, labels=[1], average='micro')) if 1 in y_true_labels else 0,
    'neutral_recall': float(recall_score(y_true_labels, y_pred_labels, labels=[2], average='micro')) if 2 in y_true_labels else 0,
}

with open('results/figures/final/lexical_baseline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary Statistics:")
print(f"  Total claims: {summary['total_claims']}")
print(f"  Accuracy: {summary['accuracy']:.4f}")
print(f"  Entailment Recall: {summary['entailment_recall']:.4f}")
print(f"  Contradiction Recall: {summary['contradiction_recall']:.4f}")
print(f"  Neutral Recall: {summary['neutral_recall']:.4f}")

# 5. Grounded vs Ungrounded F1 Bar Chart
print("\n5. Creating Grounded vs Ungrounded F1 Bar Chart...")

try:
    with open('results/eval/grounded_vs_ungrounded_all_claims.json', 'r') as f:
        g_data = json.load(f)
    
    grounded_f1 = g_data.get('grounded_macro_f1', 0)
    ungrounded_f1 = g_data.get('ungrounded_macro_f1', 0)
    
    print(f"   Grounded F1: {grounded_f1:.4f}")
    print(f"   Ungrounded F1: {ungrounded_f1:.4f}")
    
    plt.figure(figsize=(6, 6))
    bars = plt.bar(['Grounded', 'Ungrounded'], [grounded_f1, ungrounded_f1],
                   color=['#66b3ff', '#ff9999'])
    plt.ylim(0, 0.5)
    plt.ylabel('F1 Score (Macro)')
    plt.title('DeBERTa Auditor: Grounded vs Ungrounded\n(402 claims)')
    for bar, f1 in zip(bars, [grounded_f1, ungrounded_f1]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{f1:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/final/grounded_vs_ungrounded_f1.png', dpi=150)
    plt.close()
    print("Saved: grounded_vs_ungrounded_f1.png")
except FileNotFoundError:
    print("File not found: results/eval/grounded_vs_ungrounded_all_claims.json")
except Exception as e:
    print(f"Error: {e}")

# 6. Copy SciFact confusion matrix
print("\n6. Copying SciFact confusion matrix...")
if os.path.exists('results/figures/scifact_finetuned_confusion.png'):
    shutil.copy('results/figures/scifact_finetuned_confusion.png',
                'results/figures/final/scifact_confusion_matrix.png')
    print("Copied: scifact_confusion_matrix.png")
else:
    print("File not found: results/figures/scifact_finetuned_confusion.png")

# 7. Copy TruthfulQA fine-tuned confusion matrix
print("\n7. Copying TruthfulQA fine-tuned confusion matrix...")
if os.path.exists('results/figures/truthfulqa_test_finetuned_confusion.png'):
    shutil.copy('results/figures/truthfulqa_test_finetuned_confusion.png',
                'results/figures/final/truthfulqa_finetuned_confusion.png')
    print("Copied: truthfulqa_finetuned_confusion.png")
else:
    print("File not found: results/figures/truthfulqa_test_finetuned_confusion.png")

print("\nAll visualizations saved to: results/figures/final/")