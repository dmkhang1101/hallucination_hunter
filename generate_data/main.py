"""
Entry point — runs the full Member A pipeline:
  1. generate.py  →  sample questions + GPT-4o answers
  2. extract.py   →  atomic claim extraction via spaCy

Usage:
    python main.py                         # 150 questions, real API calls
    python main.py --n 100                 # custom question count
    python main.py --dry-run               # no API calls (placeholder answers)
    python main.py --skip-generate         # only run claim extraction (reuse existing answers)
"""

import argparse
import sys

import generate
import extract
import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hallucination Hunter — Member A pipeline"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=config.DEFAULT_N_QUESTIONS,
        help=f"Number of questions to sample (default: {config.DEFAULT_N_QUESTIONS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip real GPT-4o API calls; use placeholder answers instead",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation step and run extraction on existing answers CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Hallucination Hunter — Member A")
    print("=" * 60)
    print(f"  Questions : {args.n}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Skip gen  : {args.skip_generate}")
    print("=" * 60)

    # Step 1 & 2: generate answers
    if args.skip_generate:
        print("\n[main] Skipping generation step — using existing answers CSV")
        answers_df = None   # extract.run() will load from disk
    else:
        answers_df = generate.run(n_questions=args.n, dry_run=args.dry_run)

    # Step 3: extract atomic claims
    print()
    claims_df = extract.run(answers_df=answers_df, dry_run=args.dry_run)

    print()
    q_path, a_path, c_path, j_path = config.get_paths(args.dry_run)
    print("=" * 60)
    print("  Pipeline complete!")
    print(f"  Questions  → {q_path}")
    print(f"  Answers    → {a_path}")
    print(f"  Claims     → {c_path}")
    print(f"  Claims     → {j_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()