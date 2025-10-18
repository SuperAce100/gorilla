from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH, PROJECT_ROOT
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.utils import extract_test_category_from_id
from .serialize import extract_final_answer_text


def _collect_success_ids(
    score_dir: Path, subcategories: List[str]
) -> Dict[str, List[str]]:
    success_by_subcat: Dict[str, List[str]] = {s: [] for s in subcategories}
    for subcat in subcategories:
        score_file = score_dir / f"BFCL_v4_{subcat}_score.json"
        if not score_file.exists():
            # No score -> assume no successes yet
            continue
        entries = load_file(score_file)
        # first entry is header; subsequent entries contain failures only in agentic runner
        # For agentic_runner, header has accuracy; we cannot get success IDs directly; infer from results vs failures
        # So we will compute successes by diff with result file
        pass
    return success_by_subcat


def build_success_pool(
    model_result_dir: Path,
    model_score_dir: Path,
    subcategories: List[str],
    output_pool_path: Path,
) -> Tuple[List[dict], List[str]]:
    """
    Extract successes from the model result files and score files.
    For agentic runner, the score file only lists failures; we take successes as result_ids - failure_ids.
    Returns (pool_records, success_ids_all)
    pool record schema: {id, subcategory, user_text, answer_text}
    """
    pool: List[dict] = []
    success_ids_all: List[str] = []

    for subcat in subcategories:
        # Load model results to know which IDs were run
        result_file = (
            model_result_dir
            / subcat_to_group_dir(subcat)
            / f"BFCL_v4_{subcat}_result.json"
        )
        if not result_file.exists():
            continue
        result_entries = load_file(result_file)
        result_by_id = {e["id"]: e for e in result_entries}

        # Load failures from score file
        score_file = (
            model_score_dir
            / subcat_to_group_dir(subcat)
            / f"BFCL_v4_{subcat}_score.json"
        )
        failure_ids = set()
        if score_file.exists():
            score_entries = load_file(score_file)
            # header first, then each failure entry has id
            for e in score_entries[1:]:
                if "id" in e:
                    failure_ids.add(e["id"])

        # successes are all run ids minus failure ids
        success_ids = [rid for rid in result_by_id.keys() if rid not in failure_ids]
        success_ids_all.extend(success_ids)

        # Create pool items
        for sid in success_ids:
            entry = result_by_id[sid]
            # We will load prompt entry text at query-time; here store basic fields and final answer
            answer_text = extract_final_answer_text(entry.get("result", []))
            pool.append(
                {
                    "id": sid,
                    "subcategory": subcat,
                    "answer_text": answer_text,
                }
            )

    output_pool_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pool_path, "w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return pool, success_ids_all


def subcat_to_group_dir(subcat: str) -> Path:
    # Mirrors get_directory_structure_by_category but only for agentic subcategories
    if subcat.startswith("web_search"):
        return Path("agentic")
    if subcat.startswith("memory_"):
        if "kv" in subcat:
            return Path("agentic") / "memory" / "kv"
        if "vector" in subcat:
            return Path("agentic") / "memory" / "vector"
        if "rec_sum" in subcat:
            return Path("agentic") / "memory" / "rec_sum"
    return Path("agentic")
