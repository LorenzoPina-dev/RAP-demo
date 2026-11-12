"""Tree and domain state serializers for visualizer clients (terminal & web)."""

from __future__ import annotations

import math
import re
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
    """Serializes an action into canonical text or None.

    Domain actions expose ``to_text()``; using it keeps run files and the UI
    readable instead of storing a full ``BlockAction(...)`` dataclass repr.
    """
    if action is None:
        return None
    if isinstance(action, str):
        return _action_label(action)
    if hasattr(action, "to_text"):
        try:
            return str(action.to_text())
        except Exception:
            pass
    return str(action)


def _action_label(action: Any) -> Optional[str]:
    """Human-readable action text used to match a node with its parent's call log."""
    if action is None:
        return None
    if hasattr(action, "to_text"):
        try:
            return str(action.to_text())
        except Exception:
            pass

    text = str(action)
    # Older run files stored ``str(BlockAction(...))`` instead of canonical
    # text. Recover the fields so the self-evaluation entry can still be found.
    if "BlockAction(" in text:
        kind_match = re.search(
            r"action_type=(?:<ActionType\.\w+:\s*'([^']+)'>|'([^']+)'|ActionType\.(\w+))",
            text,
        )
        block_match = re.search(r"block='([^']+)'", text)
        target_match = re.search(r"target=(?:'([^']+)'|None)", text)
        if kind_match and block_match:
            kind = next((value for value in kind_match.groups() if value), "").lower()
            block = block_match.group(1)
            target = target_match.group(1) if target_match and target_match.group(1) else None
            if kind == "pick_up":
                return f"Pick up the {block} block."
            if kind == "put_down":
                return f"Put down the {block} block."
            if kind == "stack" and target:
                return f"Stack the {block} block on top of the {target} block."
            if kind == "unstack" and target:
                return f"Unstack the {block} block from on top of the {target} block."

    return text


def _node_reward_components(node: MCTSNode) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return r1/r2/r3 for a node, falling back to its trace and parent call log.

    Run files created before reward components were stored on the node still
    contain the self-evaluation entry in the parent's LLM call log. Reconstructing
    the breakdown from there lets replay mode show r1/r2/r3 for old runs too.
    """
    r1 = node.r1
    r2 = node.r2
    r3 = node.r3

    trace = getattr(node, "llm_trace", None)
    if isinstance(trace, dict) and r1 is None:
        raw_prob = trace.get("prob")
        if raw_prob is not None:
            try:
                r1 = float(raw_prob)
            except (TypeError, ValueError):
                r1 = None

    parent = getattr(node, "parent", None)
    if parent is not None and (r1 is None or r2 is None or r3 is None):
        node_action = _action_label(node.action)
        for call in reversed(getattr(parent, "llm_calls", None) or []):
            if call.get("phase") != "self-evaluation":
                continue
            if node_action is not None and call.get("action") != node_action:
                continue
            if r1 is None and call.get("r1") is not None:
                r1 = float(call["r1"])
            if r2 is None and call.get("r2") is not None:
                r2 = float(call["r2"])
            if r3 is None and call.get("r3") is not None:
                r3 = float(call["r3"])
            break

    return r1, r2, r3


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
        r1, r2, r3 = _node_reward_components(node)
        trace = getattr(node, "llm_trace", None)
        r1_source = None
        if isinstance(trace, dict):
            if trace.get("r1_source"):
                r1_source = str(trace["r1_source"])
            elif trace.get("prob") is not None:
                r1_source = "trace"
        return {
            "id": f"node_{id(node)}",
            "state": _serialize_state(node.state),
            "action": _serialize_action(node.action),
            "thought": getattr(node, "thought", None) or (getattr(node.action, "thought", None) if node.action else None),
            "llm_trace": trace,
            "llm_calls": list(getattr(node, "llm_calls", None) or []),
            "visits": node.N,
            "Q": _sanitize_float(node.Q, default_inf=0.0) or 0.0,
            "M": _sanitize_float(node.M, default_inf=None),
            "reward": float(node.reward),
            "r1": _sanitize_float(r1) if r1 is not None else None,
            "r2": _sanitize_float(r2) if r2 is not None else None,
            "r3": _sanitize_float(r3) if r3 is not None else None,
            "r1_source": r1_source,
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
