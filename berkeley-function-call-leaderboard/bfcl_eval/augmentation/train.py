from __future__ import annotations

import json
from dataclasses import dataclass
import os
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np

from bfcl_eval._llm_response_generation import (
    build_handler,
    generate_results,
    get_involved_test_entries,
    collect_test_cases,
)
from bfcl_eval.constants.eval_config import PROJECT_ROOT, RESULT_PATH, SCORE_PATH
from bfcl_eval.eval_checker.eval_runner import runner as evaluation_runner
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.utils import (
    parse_test_category_argument,
    load_dataset_entry,
    filter_entries_by_id,
)

from .index import SimpleFaissIndex, encode_texts
from .serialize import serialize_for_embedding
from .success_pool import build_success_pool, subcat_to_group_dir


AGENTIC_SUBCATEGORIES = [
    "web_search_base",
    "web_search_no_snippet",
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
]


@dataclass
class TrainConfig:
    tag: str
    model_train: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def _ensure_openai_fc(model: str) -> None:
    allowed = {"gpt-4o-mini-2024-07-18-FC", "gpt-5-2025-08-07-FC"}
    if model not in allowed:
        raise ValueError(f"model-train must be one of {allowed}")


def run_train(cfg: TrainConfig) -> None:
    _ensure_openai_fc(cfg.model_train)

    run_dir = PROJECT_ROOT / "augment" / "runs" / cfg.tag
    split_path = run_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        split_obj = json.load(f)
    subcategories: List[str] = split_obj["subcategories"]

    # Prepare custom result/score dirs for training
    train_result_dir = run_dir / "train" / "results" / cfg.model_train
    train_score_dir = run_dir / "train" / "scores" / cfg.model_train
    train_result_dir.mkdir(parents=True, exist_ok=True)
    train_score_dir.mkdir(parents=True, exist_ok=True)

    # Generate results only for train IDs per subcategory
    # We'll reuse collect_test_cases then filter by ids
    from types import SimpleNamespace

    # Stabilize threading/multiprocessing for tokenizers/BLAS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = SimpleNamespace(
        model=[cfg.model_train],
        test_category=["agentic"],
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=None,
        gpu_memory_utilization=0.9,
        backend="sglang",
        skip_server_setup=True,
        local_model_path=None,
        result_dir=train_result_dir,
        allow_overwrite=True,
        run_ids=False,
    )

    all_test_categories, all_entries = get_involved_test_entries(
        args.test_category, args.run_ids
    )
    # Restrict to our agentic subcategories only
    all_entries = [e for e in all_entries if e["id"].rsplit("_", 1)[0] in subcategories]
    # Filter to train IDs and include required memory prereqs via depends_on
    train_ids = set()
    for subcat in subcategories:
        train_ids.update(split_obj["ids_by_subcategory"][subcat]["train"])
    # include prereqs
    selected_ids = set(train_ids)
    id_to_entry = {e["id"]: e for e in all_entries}
    for tid in list(train_ids):
        entry = id_to_entry.get(tid)
        if entry and "depends_on" in entry:
            for dep in entry["depends_on"]:
                selected_ids.add(dep)
    all_entries = [e for e in all_entries if e["id"] in selected_ids]

    # Build test cases and run generation
    test_cases_total = collect_test_cases(
        args, cfg.model_train, subcategories, all_entries
    )
    if len(test_cases_total) > 0:
        generate_results(args, cfg.model_train, test_cases_total)

    # Evaluate training results
    evaluation_runner(
        model_names=[cfg.model_train],
        test_categories=subcategories,
        result_dir=run_dir / "train" / "results",
        score_dir=run_dir / "train" / "scores",
        allow_missing=True,
    )

    # Build success pool and index
    pool_path = run_dir / "train" / "success_pool.jsonl"
    pool, success_ids_all = build_success_pool(
        model_result_dir=train_result_dir,
        model_score_dir=train_score_dir,
        subcategories=subcategories,
        output_pool_path=pool_path,
    )

    # Serialize query texts for keys (load prompts to get user text)
    # We'll reconstruct text for each success id by reloading dataset entries and filtering
    texts: List[str] = []
    keys: List[dict] = []
    for subcat in subcategories:
        # load dataset without prereq for consistent index
        entries = load_dataset_entry(
            subcat, include_prereq=False, include_language_specific_hint=True
        )
        entries = [e for e in entries if e["id"] in success_ids_all]
        for e in entries:
            text, subc = serialize_for_embedding(e)
            texts.append(text)
            keys.append({"id": e["id"], "subcategory": subc})

    if texts:
        emb = encode_texts(texts, cfg.embedding_model)
        index_dir = run_dir / "train" / "index"
        index = SimpleFaissIndex(
            index_dir=index_dir, embedding_model=cfg.embedding_model
        )
        index.build(emb)
        index.save_keys(keys)

    # Log run metadata
    runs_path = run_dir / "train" / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "model_train": cfg.model_train,
                    "embedding_model": cfg.embedding_model,
                    "subcategories": subcategories,
                    "success_count": len(success_ids_all),
                }
            )
            + "\n"
        )
