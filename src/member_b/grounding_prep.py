"""
Member B: Prepare ground truth references for each question
This matches the sampled questions with TruthfulQA references
"""

import pandas as pd
import json
import os
from datasets import load_dataset

def load_truthfulqa_references():
    """Load the full TruthfulQA dataset to get reference answers"""
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation")
    return dataset['validation']

def create_grounding_file():
    """Create a mapping from question to reference answer for sampled questions"""
    
    # Load sampled questions
    sampled_df = pd.read_csv("generate_data/data/sampled_questions.csv")
    print(f"Loaded {len(sampled_df)} sampled questions")
    
    # Load atomic claims to get the question_ids that were actually used
    atomic_df = pd.read_csv("generate_data/data/atomic_claims.csv")
    used_question_ids = atomic_df['question_id'].unique()
    print(f"Found {len(used_question_ids)} unique question IDs in atomic claims")
    
    # Load TruthfulQA references
    truthfulqa = load_truthfulqa_references()
    
    # Create a dictionary for fast lookup
    reference_map = {}
    for item in truthfulqa:
        question = item['question']
        reference_map[question] = {
            'correct_answers': item['correct_answers'],
            'incorrect_answers': item.get('incorrect_answers', []),
            'category': item.get('category', 'unknown')
        }
    
    # Create grounding entries for questions that have atomic claims
    grounding_data = []
    for idx, row in sampled_df.iterrows():
        question = row['question']
        question_id = row.get('question_id', idx)
        
        # Only include if this question was used in atomic claims
        if question_id not in used_question_ids:
            continue
            
        if question in reference_map:
            ref = reference_map[question]
            grounding_data.append({
                'question_id': question_id,
                'question': question,
                'primary_reference': ref['correct_answers'][0],
                'all_correct_answers': ref['correct_answers'],
                'incorrect_answers': ref['incorrect_answers'],
                'category': ref['category']
            })
        else:
            print(f"Warning: Question not found in TruthfulQA: {question[:50]}...")
    
    # Save to files
    os.makedirs("data/grounding", exist_ok=True)
    
    # Save as JSON
    with open("data/grounding/truthfulqa_references.json", "w") as f:
        json.dump(grounding_data, f, indent=2)
    
    # Save as CSV for easy viewing
    df = pd.DataFrame(grounding_data)
    df.to_csv("data/grounding/truthfulqa_references.csv", index=False)
    
    print(f"\nSaved {len(grounding_data)} question-reference pairs")
    print(f"   JSON: data/grounding/truthfulqa_references.json")
    print(f"   CSV: data/grounding/truthfulqa_references.csv")
    
    return grounding_data

if __name__ == "__main__":
    create_grounding_file()