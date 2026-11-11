"""Tree and domain state serializers for visualizer clients (terminal & web)."""

from __future__ import annotations

import math
from typing import Any, Optional

from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.models import BlockState


def _sanitize_float(val: float, default_inf: Optional[float] = None) -> Optional[float]:
    """Sanitizes infinity or NaN floats to valid JSON values (or default/None)."""
    if math.isinf(val) or math.isnan(val):
        return default_inf
    return float(val)


def _serialize_state(state: Any) -> Any:
    """Serializes a state object into a JSON-compatible structure or string."""
    if state is None:
        return None
    if isinstance(state, BlockState):
        return serialize_blocksworld_state(state)
    if isinstance(state, (str, int, float, bool)):
        return state
    if isinstance(state, (list, tuple)):
        return [_serialize_state(item) for item in state]
    if isinstance(state, dict):
        return {str(k): _serialize_state(v) for k, v in state.items()}
    return str(state)


def _serialize_action(action: Any) -> Optional[str]:
    """Serializes an action into a string representation or None."""
    if action is None:
        return None
    if isinstance(action, str):
        return action
    return str(action)


def _compute_node_uct(node: MCTSNode) -> Optional[float]:
    """Calculates UCT score for serialization, returning None if unvisited or root."""
    if node.parent is None or node.N == 0 or node.parent.N == 0:
        return None

    exploitation = node.M if not math.isinf(node.M) else node.Q
    exploration = math.sqrt(math.log(max(node.parent.N, 1)) / node.N)
    score = exploitation + exploration
    return _sanitize_float(score)


def serialize_tree(root_node: MCTSNode) -> dict[str, Any]:
    """Converts an MCTS tree into a clean JSON-serializable hierarchy.

    Args:
        root_node: The root node of the MCTS hierarchy to serialize.

    Returns:
        A dictionary containing recursive node information:
        - id: Unique node ID
        - state: Serialized state representation
        - action: Serialized action string or None
        - visits: Visit count N
        - Q: Mean action value Q
        - M: Max action value M (or None if unvisited/-inf)
        - reward: Immediate transition reward
        - uct: Computed UCT score or None
        - is_terminal: Boolean flag indicating terminal state
        - depth: Depth in tree
        - children: Recursive list of serialized children
    """

    def _serialize_node(node: MCTSNode) -> dict[str, Any]:
        return {
            "id": f"node_{id(node)}",
            "state": _serialize_state(node.state),
            "action": _serialize_action(node.action),
            "thought": getattr(node, "thought", None) or (getattr(node.action, "thought", None) if node.action else None),
            "llm_trace": getattr(node, "llm_trace", None),
            "llm_calls": list(getattr(node, "llm_calls", None) or []),
            "visits": node.N,
            "Q": _sanitize_float(node.Q, default_inf=0.0) or 0.0,
            "M": _sanitize_float(node.M, default_inf=None),
            "reward": float(node.reward),
            "uct": _compute_node_uct(node),
            "is_terminal": bool(node.is_terminal),
            "depth": int(node.depth),
            "children": [_serialize_node(child) for child in node.children],
        }

    return _serialize_node(root_node)


def serialize_blocksworld_state(state: BlockState) -> dict[str, Any]:
    """Converts a BlockState into a tabletop visual representation.

    Args:
        state: Domain state containing clear, on_table, on, and holding relations.

    Returns:
        Dictionary with:
        - stacks: List of block stacks ordered from table (bottom) to top.
        - holding: Name of block held by gripper, or None.
        - clear: List of block names that are clear.
        - on_table: List of block names resting directly on the table.
    """
    # Inverse map: for block B, which block is ON top of B?
    # state.on has {top_block: under_block}
    top_map: dict[str, str] = {under: top for top, under in state.on.items()}

    stacks: list[list[str]] = []
    for bottom in sorted(state.on_table):
        stack = [bottom]
        curr = bottom
        while curr in top_map:
            curr = top_map[curr]
            stack.append(curr)
        stacks.append(stack)

    # Sort stacks deterministically by the base block
    stacks.sort(key=lambda s: s[0] if s else "")

    return {
        "stacks": stacks,
        "holding": state.holding,
        "clear": sorted(list(state.clear)),
        "on_table": sorted(list(state.on_table)),
    }
