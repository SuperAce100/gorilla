from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from bfcl_eval.constants.eval_config import PROJECT_ROOT
from bfcl_eval.utils import (
    load_dataset_entry,
    parse_test_category_argument,
)


AGENTIC_SUBCATEGORIES = [
    "web_search_base",
    "web_search_no_snippet",
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
]


@dataclass
class SplitConfig:
    tag: str
    seed: int = 42
    ratio: str = "2:1"  # train:test
    test_categories: List[str] | None = None


def _parse_ratio(ratio: str) -> Tuple[int, int]:
    try:
        a, b = ratio.split(":")
        return int(a), int(b)
    except Exception as e:
        raise ValueError(f"Invalid ratio '{ratio}', expected like '2:1'") from e


def _deterministic_split(
    ids: List[str], seed: int, train_weight: int, test_weight: int
) -> Tuple[List[str], List[str]]:
    ids_sorted = sorted(ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids_sorted)
    n = len(ids_sorted)
    train_n = math.floor(n * train_weight / (train_weight + test_weight))
    train_ids = ids_sorted[:train_n]
    test_ids = ids_sorted[train_n:]
    return sorted(train_ids), sorted(test_ids)


def run_split(cfg: SplitConfig) -> Path:
    tag_dir = PROJECT_ROOT / "augment" / "runs" / cfg.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    # Determine subcategories to split
    test_categories = cfg.test_categories or ["agentic"]
    expanded = parse_test_category_argument(test_categories)
    # Keep only agentic subcategories we support
    subcategories = [c for c in expanded if c in AGENTIC_SUBCATEGORIES]
    if not subcategories:
        subcategories = AGENTIC_SUBCATEGORIES

    train_w, test_w = _parse_ratio(cfg.ratio)

    ids_by_subcategory: Dict[str, Dict[str, List[str]]] = {}
    for subcat in subcategories:
        # Do NOT include memory prereq entries in split; generation will include them via dependencies
        entries = load_dataset_entry(
            subcat, include_prereq=False, include_language_specific_hint=True
        )
        ids = [e["id"] for e in entries]
        tr, te = _deterministic_split(ids, cfg.seed, train_w, test_w)
        ids_by_subcategory[subcat] = {"train": tr, "test": te}

    split_obj = {
        "seed": cfg.seed,
        "ratio": cfg.ratio,
        "subcategories": subcategories,
        "ids_by_subcategory": ids_by_subcategory,
    }

    out_path = tag_dir / "split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, ensure_ascii=False, indent=2)

    return out_path
