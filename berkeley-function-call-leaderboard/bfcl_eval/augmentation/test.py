from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from bfcl_eval._llm_response_generation import (
    build_handler,
    generate_results,
    get_involved_test_entries,
    collect_test_cases,
)
from bfcl_eval.constants.eval_config import PROJECT_ROOT
from bfcl_eval.eval_checker.eval_runner import runner as evaluation_runner
from bfcl_eval.utils import load_dataset_entry

from .index import SimpleFaissIndex, encode_texts, scope_filter_keys
from .inject import inject_few_shots
from .serialize import (
    serialize_for_embedding,
    build_first_turn_observation,
    get_first_user_message,
)


AGENTIC_SUBCATEGORIES = [
    "web_search_base",
    "web_search_no_snippet",
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
]


@dataclass
class TestConfig:
    tag: str
    model_eval: str
    source_model: str
    retrieval_scope: str = "subcategory"  # or "agentic"
    k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def _ensure_openai_fc(model: str) -> None:
    allowed = {"gpt-4o-mini-2024-07-18-FC", "gpt-5-2025-08-07-FC"}
    if model not in allowed:
        raise ValueError(f"Model must be one of {allowed}")


def _load_index(
    run_dir: Path, embedding_model: str
) -> Tuple[SimpleFaissIndex, List[dict]]:
    index_dir = run_dir / "train" / "index"
    index = SimpleFaissIndex(index_dir=index_dir, embedding_model=embedding_model)
    index.load()
    return index, index.keys


def _gather_examples_for_entry(
    entry: dict,
    k: int,
    scope: str,
    index: SimpleFaissIndex | None,
    keys: List[dict],
    subcategories: List[str],
    run_dir: Path,
    model_result_dir: Path,
) -> List[Tuple[str, str]]:
    # If k <= 0, user requested no ICL; skip reading any pools or indices.
    if k <= 0:
        return []
    # Load pool map id -> answer_text
    pool_path = run_dir / "train" / "success_pool.jsonl"
    id_to_ans = {}
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            id_to_ans[rec["id"]] = rec.get("answer_text", "")

    # Prepare query using fully materialized first-turn observation
    _, query_text = build_first_turn_observation(entry, model_result_dir)
    subcat = entry["id"].rsplit("_", 1)[0]
    allowed_indices = scope_filter_keys(keys, subcat, scope, k)

    candidate_rows: List[int] = []
    if index is not None and keys:
        # Encode and search
        q = encode_texts([query_text], index.embedding_model)
        D, I = index.search(q, min(k, len(keys)))
        candidate_rows = [i for i in I[0] if i in allowed_indices]

    examples: List[Tuple[str, str]] = []
    if not candidate_rows:
        # Fallback: if no candidates in index, pick any successes from same subcategory
        fallback_ids = [
            rid
            for rid in id_to_ans.keys()
            if rid.rsplit("_", 1)[0] == subcat and rid != entry["id"]
        ]
        for rid in fallback_ids[:k]:
            user_text = get_user_text_by_id(rid)
            ans_text = id_to_ans.get(rid, "")
            if user_text and ans_text:
                examples.append((user_text, ans_text))
        return examples

    # Map from id -> user text for examples requires re-loading original entries
    # Build id->user_text lookup lazily per subcategory
    cache: dict = {}

    def get_user_text_by_id(target_id: str) -> str:
        subc = target_id.rsplit("_", 1)[0]
        if subc not in cache:
            entries = load_dataset_entry(
                subc, include_prereq=False, include_language_specific_hint=True
            )
            cache[subc] = {e["id"]: get_first_user_message(e) for e in entries}
        return cache[subc].get(target_id, "")

    seen = set()
    for row in candidate_rows:
        rec = keys[row]
        rid = rec["id"]
        if rid in seen or rid == entry["id"]:
            continue
        seen.add(rid)
        user_text = get_user_text_by_id(rid)
        ans_text = id_to_ans.get(rid, "")
        if user_text and ans_text:
            examples.append((user_text, ans_text))
        if len(examples) >= k:
            break

    return examples


def run_test(cfg: TestConfig) -> None:
    _ensure_openai_fc(cfg.model_eval)
    _ensure_openai_fc(cfg.source_model)

    run_dir = PROJECT_ROOT / "augment" / "runs" / cfg.tag
    split_path = run_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        split_obj = json.load(f)
    subcategories: List[str] = split_obj["subcategories"]

    # Try to load retrieval index; if not available, continue without augmentation
    index = None
    keys: List[dict] = []
    try:
        index, keys = _load_index(run_dir, cfg.embedding_model)
    except Exception:
        index = None
        keys = []

    # Prepare custom result/score dirs for testing
    test_result_dir = run_dir / "test" / "results" / cfg.model_eval
    test_score_dir = run_dir / "test" / "scores" / cfg.model_eval
    test_result_dir.mkdir(parents=True, exist_ok=True)
    test_score_dir.mkdir(parents=True, exist_ok=True)

    # Load test entries only, augment with examples, then run generation
    augmented_entries: List[dict] = []
    for subcat in subcategories:
        test_ids = set(split_obj["ids_by_subcategory"][subcat]["test"])  # type: ignore
        entries = load_dataset_entry(
            subcat, include_prereq=False, include_language_specific_hint=True
        )
        entries = [e for e in entries if e["id"] in test_ids]
        for e in entries:
            if index is not None and keys:
                exs = _gather_examples_for_entry(
                    entry=e,
                    k=cfg.k,
                    scope=cfg.retrieval_scope,
                    index=index,
                    keys=keys,
                    subcategories=subcategories,
                    run_dir=run_dir,
                    model_result_dir=test_result_dir / cfg.model_eval.replace("/", "_"),
                )
            else:
                exs = _gather_examples_for_entry(
                    entry=e,
                    k=cfg.k,
                    scope=cfg.retrieval_scope,
                    index=None,
                    keys=keys,
                    subcategories=subcategories,
                    run_dir=run_dir,
                    model_result_dir=test_result_dir / cfg.model_eval.replace("/", "_"),
                )
            e_aug = inject_few_shots(entry=e, examples=exs)
            augmented_entries.append(e_aug)

    # Run generation
    from types import SimpleNamespace

    args = SimpleNamespace(
        model=[cfg.model_eval],
        test_category=subcategories,
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=None,
        gpu_memory_utilization=0.9,
        backend="sglang",
        skip_server_setup=True,
        local_model_path=None,
        result_dir=test_result_dir,
        allow_overwrite=True,
        run_ids=False,
    )

    from bfcl_eval._llm_response_generation import collect_test_cases

    test_cases_total = collect_test_cases(
        args, cfg.model_eval, subcategories, augmented_entries
    )
    if len(test_cases_total) > 0:
        generate_results(args, cfg.model_eval, test_cases_total)

    # Evaluate
    evaluation_runner(
        model_names=[cfg.model_eval],
        test_categories=subcategories,
        result_dir=run_dir / "test" / "results",
        score_dir=run_dir / "test" / "scores",
        allow_missing=True,
    )

    # Log run
    runs_path = run_dir / "test" / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "model_eval": cfg.model_eval,
                    "source_model": cfg.source_model,
                    "retrieval_scope": cfg.retrieval_scope,
                    "k": cfg.k,
                }
            )
            + "\n"
        )
