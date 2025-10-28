from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.utils import (
    extract_test_category_from_id,
    is_memory,
    populate_initial_settings_for_memory_test_cases,
    populate_initial_settings_for_web_search_test_cases,
)


def _first_user_message(entry: dict) -> str:
    # For agentic entries, first turn is entry["question"][0]
    for msg in entry["question"][0]:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def get_first_user_message(entry: dict) -> str:
    return _first_user_message(entry)


def _subcategory_from_id(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[0]


def _ensure_initial_config(entry: dict, model_result_dir: Path) -> dict:
    # Attach initial_config to the entry in-place if missing, mirroring utils population
    if "initial_config" not in entry:
        # Populate for memory if applicable
        entry_list: List[dict] = [entry]
        entry_list = populate_initial_settings_for_memory_test_cases(
            entry_list, model_result_dir
        )
        # Populate for web search if applicable (no harm if not web_search)
        entry_list = populate_initial_settings_for_web_search_test_cases(entry_list)
        entry = entry_list[0]
    return entry


def build_first_turn_observation(
    entry: dict, model_result_dir: Path
) -> Tuple[List[dict], str]:
    """
    Materialize the exact first-turn messages as seen at inference time
    (including memory system prompt if applicable), and return a flat
    observation text for embedding/retrieval (system(s) + first user).
    """
    entry = _ensure_initial_config(entry, model_result_dir)

    # Execute with empty tool calls to get instantiated instances (loads snapshots)
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]
    test_entry_id = entry["id"]
    test_category = extract_test_category_from_id(test_entry_id)

    _, involved_instances = execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        "observation_builder",
        test_entry_id,
    )

    # Inject memory system prompt exactly like inference
    if is_memory(test_category):
        assert len(involved_instances) >= 1
        memory_instance = list(involved_instances.values())[0]
        entry["question"] = add_memory_instruction_system_prompt(
            entry["question"], test_category, entry.get("scenario", ""), memory_instance
        )

    first_turn: List[dict] = entry["question"][0]

    # Build flat observation text: all system messages (in order) + first user message
    system_chunks: List[str] = [
        str(msg.get("content", "")) for msg in first_turn if msg.get("role") == "system"
    ]
    user_text: str = _first_user_message(entry)
    if len(system_chunks) > 0:
        observation_text = "\n\n".join(ch.strip() for ch in system_chunks if ch.strip())
        if user_text:
            observation_text = observation_text + "\n\n" + user_text
    else:
        observation_text = user_text

    return first_turn, observation_text


def serialize_for_embedding(
    entry: dict, model_result_dir: Optional[Path] = None
) -> Tuple[str, str]:
    entry_id = entry["id"]
    subcategory = _subcategory_from_id(entry_id)

    if model_result_dir is not None:
        # Use the fully materialized first-turn observation
        _, observation_text = build_first_turn_observation(entry, model_result_dir)
        return observation_text, subcategory

    # Fallback: previous behavior
    user = _first_user_message(entry)
    scenario = entry.get("scenario", "")
    if scenario:
        text = f"Scenario: {scenario}\nUser: {user}"
    else:
        text = f"User: {user}"
    return text, subcategory


def extract_final_answer_text(model_result_list) -> str:
    # Agentic runner expects last non-function-call message as the final answer.
    # For training successes, we will store the raw final assistant message text if available; otherwise, join strings.
    # model_result_list is a list with one element (list of steps); each step is either function call or text.
    if not isinstance(model_result_list, list) or len(model_result_list) == 0:
        return ""
    steps = model_result_list[0]
    last_text = ""
    for step in steps:
        if isinstance(step, str):
            last_text = step
        elif isinstance(step, list):
            # sometimes decode returns list of strings
            last_text = "\n".join([s for s in step if isinstance(s, str)])
        elif isinstance(step, dict):
            # function call; skip
            continue
    return str(last_text)
