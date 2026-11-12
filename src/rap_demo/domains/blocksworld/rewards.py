"""Reward functions, goal heuristics, and self-evaluation scoring for Blocksworld."""

from __future__ import annotations

import re
from typing import Sequence
from rap_demo.domains.blocksworld.models import BlockAction, BlockState
from rap_demo.domains.blocksworld.prompts import build_self_eval_prompt
from rap_demo.llm.base import BaseLLM


def _is_condition_satisfied(state: BlockState, condition: str) -> bool:
    """Checks whether a single natural language condition holds in the given state."""
    cond = condition.strip().rstrip(".").lower()

    # Pattern: the X block is on top of the Y block
    m_on = re.search(r"the\s+(\w+)\s+block\s+is\s+on\s+top\s+of\s+the\s+(\w+)\s+block", cond)
    if m_on:
        top = m_on.group(1).lower()
        bottom = m_on.group(2).lower()
        return state.on.get(top) == bottom

    # Pattern: the X block is on the table
    m_table = re.search(r"the\s+(\w+)\s+block\s+is\s+on\s+the\s+table", cond)
    if m_table:
        block = m_table.group(1).lower()
        return block in state.on_table

    # Pattern: the X block is clear
    m_clear = re.search(r"the\s+(\w+)\s+block\s+is\s+clear", cond)
    if m_clear:
        block = m_clear.group(1).lower()
        return block in state.clear

    # Pattern: hand is empty
    if "hand is empty" in cond:
        return state.holding is None

    # Pattern: holding X
    m_holding = re.search(r"(?:hand\s+is\s+holding\s+the|holding\s+the)\s+(\w+)\s+block", cond)
    if m_holding:
        return state.holding == m_holding.group(1).lower()

    return False


def _split_goal_text(text: str) -> list[str]:
    """Split conjunctions and expand a compact chain such as A on B on C."""
    parts = re.split(r",\s*and\s+|\s+and\s+|,", text)
    conditions: list[str] = []
    for part in parts:
        cleaned = part.strip()
        blocks = re.findall(r"the\s+(\w+)\s+block", cleaned, flags=re.IGNORECASE)
        if len(blocks) > 2 and cleaned.lower().count("is on top of") == len(blocks) - 1:
            conditions.extend(
                f"the {top} block is on top of the {bottom} block"
                for top, bottom in zip(blocks, blocks[1:])
            )
        elif cleaned:
            conditions.append(cleaned)
    return conditions


def calculate_goal_reward(state: BlockState, goals: Sequence[str] | dict[str, str] | str) -> float:
    """Calculates goal heuristic reward: fraction of satisfied conditions (+100 bonus on full satisfaction)."""
    conditions: list[str] = []

    if isinstance(goals, str):
        conditions = _split_goal_text(goals)
    elif isinstance(goals, dict):
        for top, bottom in goals.items():
            conditions.append(f"the {top} block is on top of the {bottom} block")
    else:
        for g in goals:
            if isinstance(g, str):
                conditions.extend(_split_goal_text(g))

    if not conditions:
        return 100.0

    satisfied = sum(1 for c in conditions if _is_condition_satisfied(state, c))
    total = len(conditions)

    if satisfied == total:
        return 100.0
    return float(satisfied) / float(total)


def calculate_combined_reward(
    state: BlockState,
    action: BlockAction,
    goals: Sequence[str] | dict[str, str] | str,
    action_score: float = 1.0,
    alpha: float = 0.5,
    eval_score: float | None = None,
) -> float:
    """Combine R1/R2 and, when available, R3 for a non-terminal transition."""
    goal_score = calculate_goal_reward(state, goals)
    if eval_score is not None:
        return float((action_score + goal_score + eval_score) / 3.0)
    return float(alpha * action_score + (1.0 - alpha) * goal_score)


def self_evaluation_result(
    llm: BaseLLM,
    prompt: str,
) -> tuple[float, float, str | None]:
    """Return Yes/No probabilities and generated answer when the backend exposes it."""
    get_result = getattr(llm, "get_token_probabilities_and_response", None)
    if callable(get_result):
        probabilities, response = get_result(prompt, ["Yes", "No"])
        return (
            float(probabilities.get("Yes", 0.0)),
            float(probabilities.get("No", 0.0)),
            str(response) if response is not None else None,
        )

    get_probabilities = getattr(llm, "get_token_probabilities", None)
    if callable(get_probabilities):
        probabilities = get_probabilities(prompt, ["Yes", "No"])
        return float(probabilities.get("Yes", 0.0)), float(probabilities.get("No", 0.0)), None
    p_yes = llm.get_token_probability(prompt, "Yes")
    p_no = llm.get_token_probability(prompt, "No")
    return p_yes, p_no, None


def self_evaluation_probabilities(llm: BaseLLM, prompt: str) -> tuple[float, float]:
    """Return the model probabilities for Yes and No."""
    p_yes, p_no, _ = self_evaluation_result(llm, prompt)
    return p_yes, p_no


def score_self_evaluation(llm: BaseLLM, prompt: str) -> float:
    """Calculate the self-evaluation reward from the LLM.

    The RAP paper (Section 3.2) uses the next-word probability of the token
    " Yes" directly as R3, so this returns P(Yes) rather than a renormalized
    Yes/No ratio.
    """
    p_yes, p_no = self_evaluation_probabilities(llm, prompt)
    if p_yes + p_no > 0.0:
        return float(max(0.0, min(1.0, p_yes)))
    return 0.5


class BlocksworldRewardFunction:
    """Callable reward function combining goal heuristic, action scores, and LLM self-eval."""

    def __init__(
        self,
        goal: Sequence[str] | dict[str, str] | str,
        llm: BaseLLM | None = None,
        alpha: float = 0.5,
        use_self_eval: bool = True,
    ) -> None:
        self.goal = goal
        self.llm = llm
        self.alpha = alpha
        self.use_self_eval = use_self_eval

    def __call__(
        self,
        state: BlockState,
        action: BlockAction,
        action_score: float = 1.0,
    ) -> float:
        eval_score: float | None = None
        if self.use_self_eval and self.llm is not None:
            prompt = build_self_eval_prompt(state, action, self.goal)
            eval_score = score_self_evaluation(self.llm, prompt)

        return calculate_combined_reward(
            state=state,
            action=action,
            goals=self.goal,
            action_score=action_score,
            alpha=self.alpha,
            eval_score=eval_score,
        )
