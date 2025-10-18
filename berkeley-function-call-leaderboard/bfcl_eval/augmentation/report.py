from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from bfcl_eval.constants.eval_config import PROJECT_ROOT
from bfcl_eval.eval_checker.eval_runner_helper import (
    update_leaderboard_table_with_local_score_file,
    generate_leaderboard_csv,
)


def _load_train_test_scores(run_dir: Path, model_name: str) -> Tuple[Dict, Dict]:
    table_train: Dict = {}
    table_test: Dict = {}
    train_scores = run_dir / "train" / "scores"
    test_scores = run_dir / "test" / "scores"
    if (train_scores / model_name).exists():
        update_leaderboard_table_with_local_score_file(table_train, train_scores)
    if (test_scores / model_name).exists():
        update_leaderboard_table_with_local_score_file(table_test, test_scores)
    return table_train, table_test


def run_report(tag: str) -> Path:
    run_dir = PROJECT_ROOT / "augment" / "runs" / tag
    output = run_dir / "report"
    output.mkdir(parents=True, exist_ok=True)

    # Aggregate whatever is present (all models under train/test)
    if (run_dir / "train" / "scores").exists():
        table_train: Dict = {}
        update_leaderboard_table_with_local_score_file(
            table_train, run_dir / "train" / "scores"
        )
        (output / "train").mkdir(parents=True, exist_ok=True)
        generate_leaderboard_csv(table_train, output / "train")

    if (run_dir / "test" / "scores").exists():
        table_test: Dict = {}
        update_leaderboard_table_with_local_score_file(
            table_test, run_dir / "test" / "scores"
        )
        (output / "test").mkdir(parents=True, exist_ok=True)
        generate_leaderboard_csv(table_test, output / "test")

    return output


def run_compare(
    tag: str, model_eval: str, source_model: str, retrieval_scope: str
) -> Dict:
    run_dir = PROJECT_ROOT / "augment" / "runs" / tag
    # Baseline lives in global SCORE_PATH; we will rely on bfcl scores CLI externally for display.
    # Here we just read augmented test overall file.
    test_overall = run_dir / "test" / "report" / "data_agentic.csv"
    result = {
        "message": "Comparison CSVs generated in report folder. Compare against baseline in global score/data_agentic.csv",
        "augmented_report_path": str(test_overall),
        "model_eval": model_eval,
        "source_model": source_model,
        "retrieval_scope": retrieval_scope,
    }
    return result
