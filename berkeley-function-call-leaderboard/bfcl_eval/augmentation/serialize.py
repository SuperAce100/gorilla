from __future__ import annotations

from typing import Tuple


def _first_user_message(entry: dict) -> str:
    # For agentic entries, first turn is entry["question"][0]
    for msg in entry["question"][0]:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _subcategory_from_id(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[0]


def serialize_for_embedding(entry: dict) -> Tuple[str, str]:
    entry_id = entry["id"]
    subcategory = _subcategory_from_id(entry_id)
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
