"""MCTS Tree Node contract for Reasoning via Planning (RAP)."""

from __future__ import annotations
import math
from typing import Any, Optional


class MCTSNode:
    """Represents a state node in the Monte Carlo Tree Search graph."""

    def __init__(
        self,
        state: Any,
        action: Optional[Any] = None,
        parent: Optional[MCTSNode] = None,
        depth: int = 0,
        reward: float = 0.0,
        is_terminal: bool = False,
        thought: Optional[str] = None,
        llm_trace: Optional[dict[str, Any]] = None,
        llm_calls: Optional[list[dict[str, Any]]] = None,
        r1: Optional[float] = None,
        r2: Optional[float] = None,
        r3: Optional[float] = None,
    ) -> None:
        self.state = state
        self.action = action
        self.parent = parent
        self.depth = depth
        self.reward = float(reward)
        # Components of the immediate transition reward (RAP Section 4.1):
        # r1 = action likelihood, r2 = goal progress, r3 = self-evaluation.
        # They stay optional because old run files and non-paper domains may not
        # have recorded them.
        self.r1: Optional[float] = None if r1 is None else float(r1)
        self.r2: Optional[float] = None if r2 is None else float(r2)
        self.r3: Optional[float] = None if r3 is None else float(r3)
        self.children: list[MCTSNode] = []
        self.visited: bool = False
        self.is_terminal: bool = is_terminal
        self.thought: Optional[str] = thought
        self.llm_trace: Optional[dict[str, Any]] = llm_trace
        # Every LLM interaction performed while expanding this node:
        # action scoring calls first, then one world-model call per action.
        self.llm_calls: list[dict[str, Any]] = list(llm_calls or [])

        # MCTS Statistics
        self.N: int = 0  # Visit count N(s)
        self.Q: float = 0.0  # Mean action value Q(s, a)
        self.M: float = -math.inf  # Max action value M(s, a)
        # Prior value used by UCT for not-yet-visited children (RAP Appendix A).
        self.prior: float = 1.0

    def set_reward_components(self, components: Optional[dict[str, Any]]) -> None:
        """Copies r1/r2/r3 from a world-model reward sink when present."""
        if not components:
            return
        for key in ("r1", "r2", "r3"):
            value = components.get(key)
            if value is not None:
                try:
                    setattr(self, key, float(value))
                except (TypeError, ValueError):
                    pass

    @property
    def is_leaf(self) -> bool:
        """Returns True if node has no children."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Returns True if node has no parent."""
        return self.parent is None

    @property
    def path_from_root(self) -> list[MCTSNode]:
        """Returns the sequence of nodes from the root node to this node."""
        curr: Optional[MCTSNode] = self
        path: list[MCTSNode] = []
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return list(reversed(path))

    def add_child(self, child_node: MCTSNode, action: Optional[Any] = None) -> MCTSNode:
        """Attaches a child node to this node and updates relationships."""
        child_node.parent = self
        child_node.depth = self.depth + 1
        if action is not None:
            child_node.action = action
            if getattr(action, "thought", None) and not child_node.thought:
                child_node.thought = getattr(action, "thought")
            if getattr(action, "llm_trace", None) and not child_node.llm_trace:
                child_node.llm_trace = getattr(action, "llm_trace")
        self.children.append(child_node)
        return child_node

    def __repr__(self) -> str:
        return (
            f"MCTSNode(state={self.state!r}, action={self.action!r}, depth={self.depth}, "
            f"N={self.N}, Q={self.Q:.3f}, M={self.M:.3f}, children={len(self.children)})"
        )
