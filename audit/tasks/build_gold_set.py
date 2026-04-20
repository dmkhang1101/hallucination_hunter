"""
Task A — Build pilot gold set from 50_sample_annotation.csv.

Reads Gold_Label and Gold_Source columns directly (pre-adjudicated by Member A).
No heuristic flipping or default-to-B logic applied.
Subtype from Hallucination_Subtype column, carried only for contradictions.
Output: results/pilot_gold.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
import utils


def run() -> list[dict]:
    utils.set_all_seeds()
    print("\n[Task A] Building pilot gold set...")

    df = pd.read_csv(config.PILOT_CSV)

    # Drop footer rows: keep only rows where claim_id is a valid integer
    df = df[pd.to_numeric(df["claim_id"], errors="coerce").notna()].copy()
    df["claim_id"] = df["claim_id"].astype(int)

    records = []
    for _, row in df.iterrows():
        gold_label = utils.normalize_label(str(row["Gold_Label"]))
        gold_source = str(row.get("Gold_Source", "")).strip()

        # Subtype from Hallucination_Subtype column; only relevant for contradictions
        raw_subtype = str(row.get("Hallucination_Subtype", ""))
        subtype = None
        if gold_label == "contradiction" and raw_subtype.lower() not in ("nan", "none", ""):
            subtype = raw_subtype.strip()

        records.append({
            "claim_id": int(row["claim_id"]),
            "question_id": str(row["question_id"]),
            "category": str(row["category"]),
            "atomic_claim": str(row["atomic_claim"]),
            "ground_truth_reference": str(row["ground_truth_reference"]),
            "gold_label": gold_label,
            "subtype": subtype,
            "adjudicated": gold_source == "adjudicated_by_A",
        })

    label_counts = pd.Series([r["gold_label"] for r in records]).value_counts().to_dict()
    adjudicated_count = sum(1 for r in records if r["adjudicated"])

    utils.save_json(
        config.PILOT_GOLD_JSON,
        {
            "records": records,
            "label_distribution": label_counts,
            "adjudicated_count": adjudicated_count,
        },
        n_examples=len(records),
        model_name="human_annotation",
    )

    print(f"  Gold set: {len(records)} claims | distribution: {label_counts} | adjudicated: {adjudicated_count}")
    return records


if __name__ == "__main__":
    run()
