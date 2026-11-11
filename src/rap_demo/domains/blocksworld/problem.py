"""Problem definition for Blocksworld: default state, goal, parsing and validation.

The same helpers are used by the CLI (`--initial-state` / `--goal`) and by the
Web UI problem editor, so both entry points accept identical specifications.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from rap_demo.domains.blocksworld.models import BlockState

DEFAULT_INITIAL_STATE: BlockState = BlockState(
    clear={"red", "blue", "orange"},
    on_table={"yellow", "red", "blue"},
    on={"orange": "yellow"},
    holding=None,
)

DEFAULT_GOAL: list[str] = [
    "the red block is on top of the blue block",
    "the orange block is on top of the red block",
]


def _block_name_set(value: Any, field_name: str) -> set[str]:
    """Coerces a JSON list of block names into a normalized set of lowercase names."""
    if value is None:
        return set()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Invalid initial state '{field_name}': expected a list of block names.")
    return {str(block).strip().lower() for block in value if str(block).strip()}


def validate_initial_state(state: BlockState) -> None:
    """Checks physical consistency of a Blocksworld state.

    Raises:
        ValueError: If a block is placed twice, a held block is also placed or clear,
            a clear block supports another block, or the 'on' relation has a cycle.
    """
    on = state.on
    on_table = state.on_table
    holding = state.holding

    overlap = on_table & set(on)
    if overlap:
        raise ValueError(
            f"Invalid initial state: blocks {sorted(overlap)} are both on the table and on another block."
        )

    if holding is not None:
        if holding in on_table or holding in on:
            raise ValueError(
                f"Invalid initial state: block '{holding}' is held but also placed on the table or another block."
            )
        if holding in state.clear:
            raise ValueError(f"Invalid initial state: held block '{holding}' cannot be clear.")

    clear_supports = set(on.values()) & state.clear
    if clear_supports:
        raise ValueError(
            f"Invalid initial state: blocks {sorted(clear_supports)} are clear but support another block."
        )

    for block in on:
        seen = {block}
        current = on.get(block)
        while current is not None:
            if current in seen:
                raise ValueError(
                    f"Invalid initial state: cycle detected in 'on' relation at block '{current}'."
                )
            seen.add(current)
            current = on.get(current)


def parse_initial_state(spec: str | None) -> BlockState:
    """Parses a Blocksworld initial state specification into a BlockState.

    Args:
        spec: JSON object ({"clear": [...], "on_table": [...], "on": {...}, "holding": ...})
            or canonical natural language accepted by BlockState.from_text()
            (e.g. "the red block is on the table, the hand is empty"). None keeps the
            built-in demo state.

    Returns:
        Parsed and validated BlockState.

    Raises:
        ValueError: If the specification is empty, malformed, or physically inconsistent.
    """
    if spec is None:
        return DEFAULT_INITIAL_STATE

    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Invalid initial state: expected a non-empty JSON or text description.")

    text = spec.strip()
    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid initial state JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Invalid initial state: JSON root must be an object.")

        raw_on = raw.get("on", {})
        if raw_on is None:
            raw_on = {}
        if not isinstance(raw_on, dict):
            raise ValueError("Invalid initial state 'on': expected an object mapping block -> support.")

        raw_holding = raw.get("holding")
        state = BlockState(
            clear=_block_name_set(raw.get("clear"), "clear"),
            on_table=_block_name_set(raw.get("on_table"), "on_table"),
            on={str(top).strip().lower(): str(bottom).strip().lower() for top, bottom in raw_on.items()},
            holding=str(raw_holding).strip().lower() if raw_holding is not None else None,
        )
    else:
        try:
            state = BlockState.from_text(text)
        except ValueError as exc:
            raise ValueError(f"Invalid initial state description: {exc}") from exc

    validate_initial_state(state)
    return state


def parse_goal(specs: Sequence[str] | None) -> list[str]:
    """Parses goal specifications into a list of natural language goal conditions.

    Args:
        specs: Raw goal values (CLI flags or UI lines). Each value may hold several
            conditions joined by ", " / " and " (e.g. "the red block is on top of the
            blue block and the orange block is on top of the red block"). None keeps
            the built-in demo goal.

    Returns:
        List of non-empty goal conditions without trailing periods.

    Raises:
        ValueError: If a provided value contains no usable condition.
    """
    if specs is None:
        return list(DEFAULT_GOAL)

    conditions: list[str] = []
    for spec in specs:
        parts = re.split(r",\s*and\s+|\s+and\s+|,", spec)
        for part in parts:
            condition = part.strip().rstrip(".").strip()
            if condition:
                conditions.append(condition)

    if not conditions:
        raise ValueError("Invalid goal: provide at least one non-empty condition.")
    return conditions
