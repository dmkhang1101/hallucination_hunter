"""
Orchestrator — runs Tasks A → G sequentially, aborts on any failure.

Skip logic: tasks B, C, E skip if their output already exists (model inference is expensive).
Tasks A, D, F, G always re-run (A because gold labels changed; D/F/G depend on A's output).
Prints macro-F1 summary table at the end.
"""

import importlib
import traceback
from pathlib import Path

import config


def _macro_f1(result: object) -> str:
    if isinstance(result, dict):
        if "macro_f1" in result:
            return f"{result['macro_f1']:.4f}"
    if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], dict):
        return f"{result[0].get('macro_f1', 'N/A'):.4f}"
    return "N/A"


# (id, name, tasks.<module>, skip_if_exists: Path | None)
TASKS: list[tuple[str, str, str, Path | None]] = [
    ("A", "Pilot Gold Set",         "tasks.build_gold_set",       None),
    ("B", "Pipeline Sanity Check",  "tasks.verify_auditor",       config.SANITY_CHECK_JSON),
    ("C", "SciFact Eval (RQ1)",     "tasks.evaluate_scifact",     config.SCIFACT_EVAL_JSON),
    ("D", "TruthfulQA Eval (RQ2)",  "tasks.evaluate_truthfulqa",  None),
    ("E", "Baselines",              "tasks.run_baselines",        config.BASELINES_JSON),
    ("F", "Subtype Analysis (RQ3)", "tasks.analyze_subtypes",     None),
    ("G", "Error Analysis",         "tasks.extract_errors",       None),
    ("H", "Predict All Claims",     "tasks.predict_all_claims",   config.ALL_CLAIMS_PREDICTIONS_CSV),
]


def main() -> None:
    summary: list[tuple[str, str, str, str]] = []

    for task_id, task_name, module_path, skip_path in TASKS:
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_name}")
        print("=" * 60)

        if skip_path is not None and skip_path.exists():
            print(f"  [SKIP] Output already exists: {skip_path.name}")
            summary.append((task_id, task_name, "SKIP", "—"))
            continue

        try:
            module = importlib.import_module(module_path)
            result = module.run()
            f1 = _macro_f1(result)
            summary.append((task_id, task_name, "PASS", f1))
        except Exception as e:
            traceback.print_exc()
            summary.append((task_id, task_name, "FAIL", "—"))
            print(f"\n[ABORT] Task {task_id} failed: {e}")
            break

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<6} {'Name':<30} {'Status':<8} {'Macro-F1'}")
    print("-" * 60)
    for task_id, name, status, f1 in summary:
        print(f"{task_id:<6} {name:<30} {status:<8} {f1}")

    any_failed = any(s == "FAIL" for _, _, s, _ in summary)
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
