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


def _condition_is_blocked(state: BlockState, condition: str) -> bool:
    """Whether a goal condition needs extra moves before it can be satisfied.

    To satisfy "the X block is on top of the Y block", X has to be held and Y
    has to be clear. When either of them is not available the condition is not
    simply unmet, it needs extra actions first, so the state is already moving
    away from the goal. Detecting that keeps the search out of the branches
    that block a block another condition needs.
    """
    match = re.search(
        r"the\s+(\w+)\s+block\s+is\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
        condition,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    top, bottom = match.group(1).lower(), match.group(2).lower()
    for block in (top, bottom):
        if block not in state.clear and state.holding != block:
            return True
    return False


def calculate_goal_reward(
    state: BlockState,
    goals: Sequence[str] | dict[str, str] | str,
    blocked_penalty: float = 0.5,
) -> float:
    """Calculates the task-specific goal heuristic R2 (RAP Section 4.1).

    R2 compares the predicted next state with the goal conditions and scales with
    the number of conditions met. When every condition holds, R2 becomes the
    "super large reward" that makes the plan win (paper Section 4.1).

    The reference implementation (rap/blocksworld_mcts.py, ``r1_fn``) returns
    ``sum(meetings) / len(meetings) + 0.5`` for partial progress and ``100`` for
    a complete goal. The ``+ 0.5`` offset keeps R2 strictly positive, which the
    weighted geometric mean combination requires.

    ``blocked_penalty`` is an extension: a condition that is not merely unmet
    but blocked (it needs extra moves before it can be satisfied) is penalized.
    With ``blocked_penalty = 0`` the reward is exactly the paper's one.
    """
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

    reward = float(satisfied) / float(total) + 0.5

    if blocked_penalty > 0.0:
        # A condition that is not merely unmet but blocked needs extra moves
        # before it can be satisfied. This is an extension of the paper's R2:
        # with blocked_penalty = 0 the reward is exactly the paper's one.
        blocked = sum(
            1
            for c in conditions
            if not _is_condition_satisfied(state, c) and _condition_is_blocked(state, c)
        )
        reward -= blocked_penalty * float(blocked) / float(total)

    return max(0.0, reward)


# A self-evaluation of "No" must not zero the whole reward: in a weighted
# geometric mean a single zero factor annihilates the product. The reference
# implementation keeps its goal reward strictly positive for the very same
# reason (the "+ 0.5" offset of ``r1_fn``).
_SELF_EVAL_FLOOR = 0.05


def calculate_combined_reward(
    state: BlockState,
    action: BlockAction,
    goals: Sequence[str] | dict[str, str] | str,
    action_score: float = 1.0,
    alpha: float = 0.5,
    eval_score: float | None = None,
    self_eval_weight: float = 0.25,
    blocked_penalty: float = 0.5,
) -> float:
    """Combine R1/R2 and, when available, R3 into the transition reward.

    The reference implementation combines rewards with a *weighted geometric
    mean* (rap/blocksworld_mcts.py: ``_r0 ** r_alpha * _r1 ** (1 - r_alpha)``),
    and the paper uses the same form for the math domain (Section 4.2:
    ``rt = r_t,1 ** alpha * r_t,2 ** (1 - alpha)``). ``alpha`` therefore
    weights R1 (action likelihood) against R2 (goal heuristic).

    R3 (self-evaluation) is the optional third reward of the ablation in
    Table 5. The paper reports that adding it helps (0.88 -> 0.91) but does
    not give a three-way formula, so its raw weight is ``self_eval_weight``
    and all weights are normalized to sum to one.

    ``eval_score`` is floored at ``_SELF_EVAL_FLOOR``: a model that answers
    "No" would otherwise zero the whole reward and hide the branch.
    """
    goal_score = calculate_goal_reward(state, goals, blocked_penalty=blocked_penalty)
    components: list[tuple[float, float]] = [
        (float(action_score), float(alpha)),
        (float(goal_score), float(1.0 - alpha)),
    ]
    if eval_score is not None and self_eval_weight > 0.0:
        components.append(
            (max(float(eval_score), _SELF_EVAL_FLOOR), float(self_eval_weight))
        )

    total_weight = sum(weight for _, weight in components)
    reward = 1.0
    for value, weight in components:
        reward *= value ** (weight / total_weight)
    return reward


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
            prompt = build_self_eval_prompt(state, state.apply_action(action), self.goal)
            eval_score = score_self_evaluation(self.llm, prompt)

        return calculate_combined_reward(
            state=state,
            action=action,
            goals=self.goal,
            action_score=action_score,
            alpha=self.alpha,
            eval_score=eval_score,
        )
