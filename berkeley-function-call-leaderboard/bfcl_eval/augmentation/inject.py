from __future__ import annotations

from typing import List, Tuple


def inject_few_shots(entry: dict, examples: List[Tuple[str, str]]) -> dict:
    """
    Inject few-shot examples as alternating user/assistant messages before the real user message
    in the first turn. Keep any existing system/developer messages intact at the front.

    Each example is (user_text, answer_text).
    """
    if not examples:
        return entry

    first_turn: List[dict] = entry["question"][0]
    # find first user message position
    user_pos = 0
    for i, msg in enumerate(first_turn):
        if msg.get("role") == "user":
            user_pos = i
            break
        user_pos = i + 1

    injected: List[dict] = []
    for u, a in examples:
        injected.append({"role": "user", "content": u})
        injected.append({"role": "assistant", "content": a})

    # splice in
    entry["question"][0] = first_turn[:user_pos] + injected + first_turn[user_pos:]
    return entry
