"""
Member B: Dataset Statistics Analysis
Run this to generate statistics for the progress report
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

# Paths
ATOMIC_CLAIMS_PATH = "generate_data/data/atomic_claims.csv"
SAMPLED_QUESTIONS_PATH = "generate_data/data/sampled_questions.csv"
PRIMARY_ANSWERS_PATH = "generate_data/data/primary_answers.csv"

def analyze_atomic_claims():
    """Analyze the atomic claims extracted from GPT-4o answers"""
    print("=" * 60)
    print("ATOMIC CLAIMS ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(ATOMIC_CLAIMS_PATH)
    
    print(f"\nTotal atomic claims: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Claim length statistics
    claim_lengths = df['claim'].str.split().str.len()
    print(f"\nClaim length (words):")
    print(f"  Mean: {claim_lengths.mean():.1f}")
    print(f"  Median: {claim_lengths.median():.1f}")
    print(f"  Min: {claim_lengths.min()}")
    print(f"  Max: {claim_lengths.max()}")
    
    # Questions distribution
    unique_questions = df['question_id'].nunique()
    print(f"\nUnique questions: {unique_questions}")
    print(f"Avg claims per question: {len(df)/unique_questions:.1f}")
    print(f"Claims per question - Min: {df.groupby('question_id').size().min()}")
    print(f"Claims per question - Max: {df.groupby('question_id').size().max()}")
    
    # Category distribution
    categories = df['category'].value_counts()
    print(f"\nCategory distribution:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} claims ({count/len(df)*100:.1f}%)")
    
    return df

def analyze_sampled_questions():
    """Analyze the sampled questions from TruthfulQA"""
    print("\n" + "=" * 60)
    print("SAMPLED QUESTIONS ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(SAMPLED_QUESTIONS_PATH)
    
    print(f"\nTotal questions: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Question length
    question_lengths = df['question'].str.split().str.len()
    print(f"\nQuestion length (words):")
    print(f"  Mean: {question_lengths.mean():.1f}")
    print(f"  Median: {question_lengths.median():.1f}")
    print(f"  Min: {question_lengths.min()}")
    print(f"  Max: {question_lengths.max()}")
    
    # Category distribution
    if 'category' in df.columns:
        categories = df['category'].value_counts()
        print(f"\nCategory distribution:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")
    
    return df

def analyze_primary_answers():
    """Analyze the GPT-4o generated answers"""
    print("\n" + "=" * 60)
    print("PRIMARY ANSWERS ANALYSIS (GPT-4o)")
    print("=" * 60)
    
    df = pd.read_csv(PRIMARY_ANSWERS_PATH)
    
    print(f"\nTotal answers: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Answer length - FIXED: using 'primary_answer' instead of 'answer'
    answer_lengths = df['primary_answer'].str.split().str.len()
    print(f"\nAnswer length (words):")
    print(f"  Mean: {answer_lengths.mean():.1f}")
    print(f"  Median: {answer_lengths.median():.1f}")
    print(f"  Min: {answer_lengths.min()}")
    print(f"  Max: {answer_lengths.max()}")
    
    return df

def save_statistics_report():
    """Generate markdown report for the project proposal"""
    
    atomic_df = pd.read_csv(ATOMIC_CLAIMS_PATH)
    questions_df = pd.read_csv(SAMPLED_QUESTIONS_PATH)
    answers_df = pd.read_csv(PRIMARY_ANSWERS_PATH)
    
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/dataset_statistics.md", "w") as f:
        f.write("# Dataset Statistics for Hallucination Hunter\n\n")
        f.write("## Overview\n")
        f.write(f"- **Total atomic claims extracted**: {len(atomic_df)}\n")
        f.write(f"- **Total questions sampled**: {len(questions_df)}\n")
        f.write(f"- **Total GPT-4o answers generated**: {len(answers_df)}\n\n")
        
        f.write("## Atomic Claims\n")
        claim_lengths = atomic_df['claim'].str.split().str.len()
        f.write(f"- Average claim length: {claim_lengths.mean():.1f} words\n")
        f.write(f"- Median claim length: {claim_lengths.median():.1f} words\n")
        f.write(f"- Min claim length: {claim_lengths.min()} words\n")
        f.write(f"- Max claim length: {claim_lengths.max()} words\n")
        
        unique_q = atomic_df['question_id'].nunique()
        f.write(f"- Unique questions represented: {unique_q}\n")
        f.write(f"- Average claims per question: {len(atomic_df)/unique_q:.1f}\n\n")
        
        f.write("## Category Distribution (Claims)\n")
        categories = atomic_df['category'].value_counts()
        for cat, count in categories.items():
            f.write(f"- {cat}: {count} claims ({count/len(atomic_df)*100:.1f}%)\n")
        
        f.write("\n## Questions (TruthfulQA Sample)\n")
        q_lengths = questions_df['question'].str.split().str.len()
        f.write(f"- Average question length: {q_lengths.mean():.1f} words\n")
        f.write(f"- Median question length: {q_lengths.median():.1f} words\n")
        
        if 'category' in questions_df.columns:
            f.write("\n### Question Category Distribution\n")
            q_categories = questions_df['category'].value_counts()
            for cat, count in q_categories.items():
                f.write(f"- {cat}: {count}\n")
        
        f.write("\n## GPT-4o Generated Answers\n")
        a_lengths = answers_df['primary_answer'].str.split().str.len()
        f.write(f"- Average answer length: {a_lengths.mean():.1f} words\n")
        f.write(f"- Median answer length: {a_lengths.median():.1f} words\n")
        f.write(f"- Min answer length: {a_lengths.min()} words\n")
        f.write(f"- Max answer length: {a_lengths.max()} words\n")
    
    print("\nStatistics report saved to reports/dataset_statistics.md")

if __name__ == "__main__":
    analyze_atomic_claims()
    analyze_sampled_questions()
    analyze_primary_answers()
    save_statistics_report()