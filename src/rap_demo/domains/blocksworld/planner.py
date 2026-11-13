"""Shortest-plan lookup for Blocksworld.

The MCTS can only reach a goal within the depth limit L, so it is worth
knowing the length of the shortest plan before the search starts: when the
task needs more actions than the tree allows, no branch can ever earn the
"super large reward" and the search has no target to aim at.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.domains.blocksworld.rewards import calculate_goal_reward

# Guard against pathological state spaces.
_MAX_SEARCH_DEPTH = 24


def minimum_actions_to_goal(
    state: BlockState,
    goals: Sequence[str] | dict[str, str] | str,
) -> int | None:
    """Return the length of the shortest plan that reaches the goal.

    The search uses the exact physics of the domain, so the answer is the
    optimal number of actions, not an estimate.

    Args:
        state: The initial state of the problem.
        goals: The goal conditions.

    Returns:
        The minimum number of actions, or None when the goal cannot be
        reached from this state.
    """

    def reached(current: BlockState) -> bool:
        return calculate_goal_reward(current, goals, blocked_penalty=0.0) >= 100.0

    if reached(state):
        return 0

    seen = {state.to_text()}
    queue: deque[tuple[BlockState, int]] = deque([(state, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= _MAX_SEARCH_DEPTH:
            continue
        for action in current.get_legal_actions():
            next_state = current.apply_action(action)
            key = next_state.to_text()
            if key in seen:
                continue
            seen.add(key)
            if reached(next_state):
                return depth + 1
            queue.append((next_state, depth + 1))

    return None
