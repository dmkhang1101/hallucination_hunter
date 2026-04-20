"""
Member B: Lexical-overlap baseline for hallucination detection
Compares atomic claims to ground truth references
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import os

class LexicalHallucinationDetector:
    """Simple baseline using lexical overlap to detect contradictions"""
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def preprocess(self, text):
        """Simple text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def token_overlap(self, claim, reference):
        """Calculate Jaccard similarity"""
        claim_tokens = set(self.preprocess(claim).split())
        ref_tokens = set(self.preprocess(reference).split())
        
        if not claim_tokens or not ref_tokens:
            return 0.0
        
        intersection = len(claim_tokens & ref_tokens)
        union = len(claim_tokens | ref_tokens)
        return intersection / union if union > 0 else 0.0
    
    def tfidf_similarity(self, claim, reference):
        """Calculate TF-IDF cosine similarity"""
        claim_clean = self.preprocess(claim)
        ref_clean = self.preprocess(reference)
        
        if not claim_clean or not ref_clean:
            return 0.0
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([claim_clean, ref_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def predict(self, claim, reference):
        """Predict if claim contradicts reference"""
        token_score = self.token_overlap(claim, reference)
        tfidf_score = self.tfidf_similarity(claim, reference)
        combined = (token_score + tfidf_score) / 2
        
        if combined > self.threshold:
            prediction = 'Entailment'
        elif combined < 0.15:
            prediction = 'Contradiction'
        else:
            prediction = 'Neutral'
        
        return prediction, {
            'token_overlap': token_score,
            'tfidf_similarity': tfidf_score,
            'combined': combined
        }

def run_baseline_evaluation():
    """Run the baseline on all atomic claims"""
    
    # Load data
    atomic_df = pd.read_csv("generate_data/data/atomic_claims.csv")
    grounding_df = pd.read_csv("data/grounding/truthfulqa_references.csv")
    
    print(f"Loaded {len(atomic_df)} atomic claims")
    print(f"Loaded {len(grounding_df)} grounding references")
    
    # Create a quick lookup for references by question_id
    ref_lookup = grounding_df.set_index('question_id')['primary_reference'].to_dict()
    
    # Initialize detector
    detector = LexicalHallucinationDetector(threshold=0.3)
    
    # Run evaluation
    results = []
    for idx, row in atomic_df.iterrows():
        claim = row['claim']
        question_id = row['question_id']
        
        # Get reference for this question
        reference = ref_lookup.get(question_id, '')
        
        if reference:
            prediction, scores = detector.predict(claim, reference)
            results.append({
                'claim_id': idx,
                'question_id': question_id,
                'category': row['category'],
                'claim': claim[:200] + '...' if len(claim) > 200 else claim,
                'reference': reference[:200] + '...' if len(reference) > 200 else reference,
                'prediction': prediction,
                'token_overlap': scores['token_overlap'],
                'tfidf_similarity': scores['tfidf_similarity'],
                'combined_score': scores['combined']
            })
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/lexical_baseline_results.csv", index=False)
    
    print(f"\nEvaluated {len(results)} claims")
    print(f"   Results saved to results/lexical_baseline_results.csv")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"\nPredictions distribution:")
    pred_counts = results_df['prediction'].value_counts()
    for pred, count in pred_counts.items():
        print(f"  {pred}: {count} ({count/len(results_df)*100:.1f}%)")
    
    print(f"\nScore statistics:")
    print(f"  Token Overlap - Mean: {results_df['token_overlap'].mean():.3f}")
    print(f"  Token Overlap - Std: {results_df['token_overlap'].std():.3f}")
    print(f"  TF-IDF - Mean: {results_df['tfidf_similarity'].mean():.3f}")
    print(f"  TF-IDF - Std: {results_df['tfidf_similarity'].std():.3f}")
    print(f"  Combined - Mean: {results_df['combined_score'].mean():.3f}")
    
    # By category
    print("\n" + "=" * 50)
    print("RESULTS BY CATEGORY")
    print("=" * 50)
    for category in results_df['category'].unique()[:10]:  # Show top 10 categories
        cat_df = results_df[results_df['category'] == category]
        print(f"\n{category}:")
        print(f"  Claims: {len(cat_df)}")
        print(f"  Contradictions detected: {len(cat_df[cat_df['prediction'] == 'Contradiction'])} ({len(cat_df[cat_df['prediction'] == 'Contradiction'])/len(cat_df)*100:.1f}%)")
        print(f"  Avg combined score: {cat_df['combined_score'].mean():.3f}")
    
    return results_df

if __name__ == "__main__":
    results = run_baseline_evaluation()