"""
Member B: Analyze baseline results and create visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def analyze_results():
    # Load baseline results
    baseline_df = pd.read_csv("results/lexical_baseline_results.csv")
    
    print("=" * 50)
    print("BASELINE PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Score distribution
    print("\nScore Distribution:")
    print(f"Token Overlap - Mean: {baseline_df['token_overlap'].mean():.3f}")
    print(f"Token Overlap - Median: {baseline_df['token_overlap'].median():.3f}")
    print(f"TF-IDF - Mean: {baseline_df['tfidf_similarity'].mean():.3f}")
    print(f"TF-IDF - Median: {baseline_df['tfidf_similarity'].median():.3f}")
    
    # Create visualizations
    os.makedirs("results/figures", exist_ok=True)
    
    # Figure 1: Score distribution histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(baseline_df['token_overlap'], bins=30, color='blue', alpha=0.7)
    axes[0].set_xlabel('Token Overlap Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Token Overlap Distribution')
    axes[0].axvline(x=baseline_df['token_overlap'].mean(), color='red', linestyle='--', label=f'Mean: {baseline_df["token_overlap"].mean():.3f}')
    axes[0].legend()
    
    axes[1].hist(baseline_df['tfidf_similarity'], bins=30, color='green', alpha=0.7)
    axes[1].set_xlabel('TF-IDF Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('TF-IDF Distribution')
    axes[1].axvline(x=baseline_df['tfidf_similarity'].mean(), color='red', linestyle='--', label=f'Mean: {baseline_df["tfidf_similarity"].mean():.3f}')
    axes[1].legend()
    
    axes[2].hist(baseline_df['combined_score'], bins=30, color='purple', alpha=0.7)
    axes[2].set_xlabel('Combined Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Combined Score Distribution')
    axes[2].axvline(x=baseline_df['combined_score'].mean(), color='red', linestyle='--', label=f'Mean: {baseline_df["combined_score"].mean():.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("results/figures/score_distributions.png", dpi=150)
    plt.close()
    print("\nScore distribution plot saved to results/figures/score_distributions.png")
    
    # Figure 2: Predictions pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    pred_counts = baseline_df['prediction'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Distribution of Predictions (Lexical Baseline)')
    plt.tight_layout()
    plt.savefig("results/figures/predictions_pie.png", dpi=150)
    plt.close()
    print("Predictions pie chart saved to results/figures/predictions_pie.png")
    
    # Figure 3: Predictions by category (top 15 categories)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get top 15 categories by count
    top_categories = baseline_df['category'].value_counts().head(15).index
    cat_df = baseline_df[baseline_df['category'].isin(top_categories)]
    
    category_predictions = pd.crosstab(cat_df['category'], cat_df['prediction'], normalize='index')
    category_predictions.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Category')
    ax.set_title('Prediction Distribution by Category (Top 15)')
    ax.legend(title='Prediction', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("results/figures/predictions_by_category.png", dpi=150)
    plt.close()
    print("Predictions by category plot saved to results/figures/predictions_by_category.png")
    
    # Figure 4: Score heatmap by category (top 15)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    category_stats = baseline_df[baseline_df['category'].isin(top_categories)].groupby('category')[['token_overlap', 'tfidf_similarity', 'combined_score']].mean()
    sns.heatmap(category_stats, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
    ax.set_title('Average Scores by Category (Top 15)')
    
    plt.tight_layout()
    plt.savefig("results/figures/scores_by_category.png", dpi=150)
    plt.close()
    print("Score heatmap saved to results/figures/scores_by_category.png")
    
    # Save summary statistics to JSON
    summary = {
        'total_claims': len(baseline_df),
        'predictions': baseline_df['prediction'].value_counts().to_dict(),
        'mean_scores': {
            'token_overlap': float(baseline_df['token_overlap'].mean()),
            'tfidf_similarity': float(baseline_df['tfidf_similarity'].mean()),
            'combined_score': float(baseline_df['combined_score'].mean())
        },
        'std_scores': {
            'token_overlap': float(baseline_df['token_overlap'].std()),
            'tfidf_similarity': float(baseline_df['tfidf_similarity'].std()),
            'combined_score': float(baseline_df['combined_score'].std())
        }
    }
    
    with open("results/baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary statistics saved to results/baseline_summary.json")
    
    return baseline_df

if __name__ == "__main__":
    import json
    df = analyze_results()