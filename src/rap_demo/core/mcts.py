"""Monte Carlo Tree Search (MCTS) engine according to RAP (Reasoning via Planning)."""

from __future__ import annotations
import inspect
import math
from typing import Any, Optional, Protocol, runtime_checkable
from rap_demo.core.node import MCTSNode
from rap_demo.core.events import EventBus, EventType


@runtime_checkable
class WorldModel(Protocol):
    """Protocol for state transition and environment simulation."""

    def step(self, state: Any, action: Any) -> tuple[Any, float, bool]:
        """Given current state and action, returns (next_state, immediate_reward, is_terminal)."""
        ...


@runtime_checkable
class ReasoningAgent(Protocol):
    """Protocol for generating candidate reasoning steps/actions."""

    def get_actions(self, state: Any, remaining_steps: int | None = None) -> list[Any]:
        """Proposes candidate actions from the current state."""
        ...


_UNVISITED_BASE = 1e6


def _action_prior(action: Any) -> float:
    """Returns the action prior used to rank unvisited siblings.

    In paper mode this is r1 (softmax-normalised action likelihood); other modes
    fall back to a neutral 1.0.
    """
    trace = getattr(action, "llm_trace", None) or {}
    probability = trace.get("prob")
    if probability is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(probability)))
    except (TypeError, ValueError):
        return 1.0


def _action_text(action: Any) -> str | None:
    """Best-effort human-readable action label for the inspector."""
    if action is None:
        return None
    if hasattr(action, "to_text"):
        try:
            return str(action.to_text())
        except Exception:
            pass
    return str(action)


def _action_call_entry(action: Any) -> dict[str, Any] | None:
    """Builds one inspector entry from the trace that produced an action.

    Paper mode does not generate a plan line: it scores each legal action with
    ``get_logprobs``, so the recorded response is the scored continuation.
    """
    trace = getattr(action, "llm_trace", None)
    if not isinstance(trace, dict) or trace.get("prompt") is None:
        return None
    source = trace.get("source")
    response = trace.get("response")
    if response is None and source == "model-paper" and hasattr(action, "to_paper_text"):
        try:
            response = action.to_paper_text()
        except Exception:
            response = None
    return {
        "phase": "action-scoring" if source == "model-paper" else "action",
        "action": _action_text(action),
        "prompt": trace.get("prompt"),
        "response": response,
        "logprob": trace.get("logprob"),
        "prob": trace.get("prob"),
        "r1_source": trace.get("r1_source"),
        "source": source,
        "reason": trace.get("reason"),
        "error": None,
    }


def _world_step(
    world_model: WorldModel,
    state: Any,
    action: Any,
    lightweight: bool = False,
    call_log: list[dict[str, Any]] | None = None,
    reward_sink: dict[str, Any] | None = None,
) -> tuple[Any, float, bool]:
    """Calls ``step``, requesting the lightweight transition when supported.

    The paper (Section 3.3) discards computationally costly reward components
    during simulation. For Blocksworld the costly component is the LLM world
    model, so rollouts can use the exact physics simulator instead.

    ``reward_sink`` receives the r1/r2/r3 breakdown when the world model can
    expose it, so the inspector can show the components of the node reward.
    """
    try:
        parameters = inspect.signature(world_model.step).parameters
    except (TypeError, ValueError):
        return world_model.step(state, action)
    kwargs: dict[str, Any] = {}
    if "lightweight" in parameters:
        kwargs["lightweight"] = lightweight
    if "call_log" in parameters and call_log is not None:
        kwargs["call_log"] = call_log
    if "reward_sink" in parameters and reward_sink is not None:
        kwargs["reward_sink"] = reward_sink
    return world_model.step(state, action, **kwargs) if kwargs else world_model.step(state, action)


def _safe_step(
    world_model: WorldModel,
    state: Any,
    action: Any,
    lightweight: bool,
    call_log: list[dict[str, Any]] | None = None,
    reward_sink: dict[str, Any] | None = None,
) -> tuple[Any, float, bool]:
    """Predicts the next state, repairing an unusable LLM prediction with exact physics.

    The paper uses the LLM as the world model and is therefore exposed to
    hallucinated transitions. Keeping the deterministic simulator as a fallback
    preserves the paper's primary behaviour without letting one bad generation
    kill the whole search tree.
    """
    try:
        return _world_step(
            world_model,
            state,
            action,
            lightweight=lightweight,
            call_log=call_log,
            reward_sink=reward_sink,
        )
    except (ValueError, KeyError, AssertionError):
        if lightweight:
            raise
        return _world_step(
            world_model,
            state,
            action,
            lightweight=True,
            reward_sink=reward_sink,
        )


def _propose_actions(
    agent: ReasoningAgent,
    state: Any,
    remaining_steps: int,
    lightweight: bool = False,
) -> list[Any]:
    """Calls ``get_actions``, forwarding the remaining depth budget when supported.

    The paper's Appendix E truncates in-context demonstrations as the tree deepens,
    which requires the agent to know how many actions are still available.
    """
    try:
        parameters = inspect.signature(agent.get_actions).parameters
    except (TypeError, ValueError):
        return agent.get_actions(state)
    kwargs: dict[str, Any] = {}
    if "remaining_steps" in parameters:
        kwargs["remaining_steps"] = remaining_steps
    if "lightweight" in parameters:
        kwargs["lightweight"] = lightweight
    return agent.get_actions(state, **kwargs) if kwargs else agent.get_actions(state)


def extract_best_trajectory(root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
    """Extracts the best trajectory from root according to 'reward' or 'visits'.

    Following RAP Paper Section 3.3 and Appendix A:
    - For 'reward': evaluates complete paths to leaves/terminals, selecting the
      path with maximum cumulative reward (and terminal goal achievement).
    - For 'visits': selects the path with maximum exploration / visits.
    """
    if not root.children:
        return [root]

    all_leaves: list[MCTSNode] = []

    def collect_leaves(node: MCTSNode) -> None:
        if not node.children:
            all_leaves.append(node)
        else:
            for child in node.children:
                collect_leaves(child)

    collect_leaves(root)
    if not all_leaves:
        return [root]

    if criterion == "visits":
        best_leaf = max(
            all_leaves,
            key=lambda leaf: (
                sum(n.N for n in leaf.path_from_root),
                leaf.N,
                leaf.reward,
            ),
        )
    else:  # "reward"
        best_leaf = max(
            all_leaves,
            key=lambda leaf: (
                1 if leaf.is_terminal and leaf.reward >= 50.0 else 0,
                sum(n.reward for n in leaf.path_from_root),
                leaf.M if not math.isinf(leaf.M) else -1e9,
                leaf.reward,
                leaf.N,
            ),
        )

    return best_leaf.path_from_root


class MCTS:
    """Core MCTS planning engine implementing RAP Algorithm 1 & Section 3.3."""

    def __init__(
        self,
        w_exp: float = 1.0,
        discount: float = 1.0,
        aggr_reward: str = "sum",
        aggr_child: str = "max",
        max_depth: int = 10,
        world_model: Optional[WorldModel] = None,
        reasoning_agent: Optional[ReasoningAgent] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.w_exp = w_exp
        self.discount = discount
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child
        self.max_depth = max_depth
        self.world_model = world_model
        self.reasoning_agent = reasoning_agent
        self.event_bus = event_bus

    def search(self, root_state: Any, num_iterations: int) -> MCTSNode:
        """Runs MCTS search from the given root state for num_iterations cycles."""
        root = MCTSNode(state=root_state, depth=0)
        for iteration in range(num_iterations):
            self.step_iteration(root)
            if self.event_bus:
                self.event_bus.emit(
                    EventType.ON_ITERATION_COMPLETE,
                    {"iteration": iteration, "root": root},
                )
        return root

    def step_iteration(self, root: MCTSNode) -> list[MCTSNode]:
        """Executes a single MCTS iteration: Select -> Expand -> Simulate -> Backpropagate.

        Follows Algorithm 1 of the RAP paper: after expansion the simulation starts
        from the child with the largest local reward instead of re-querying the agent
        for the same leaf, which halves the LLM calls per iteration.
        """
        path = self.select(root)
        leaf = path[-1]

        rollout_reward = 0.0
        if not leaf.is_terminal:
            if leaf.is_leaf:
                self.expand(leaf)
            start = leaf
            if leaf.children:
                # Algorithm 1, line 16: a_{t+1} <- argmax_a r(s_t, a)
                start = max(leaf.children, key=lambda c: (c.reward, -c.depth))
                rollout_reward = start.reward
            _, rollout_reward = self.simulate(start, base_reward=rollout_reward)

        self.backpropagate(path, rollout_reward)
        return path

    def select(self, root: MCTSNode) -> list[MCTSNode]:
        """Selects a leaf node starting from root using UCT formula."""
        node = root
        path = [node]

        while not node.is_terminal and not node.is_leaf:
            node = max(node.children, key=lambda c: self.uct_value(node, c))
            path.append(node)
            if node.N == 0:
                break

        if self.event_bus:
            self.event_bus.emit(EventType.ON_SELECT, {"node": node, "path": path})

        return path

    def uct_value(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Computes the UCT score for a child node with respect to its parent.

        Unvisited children (N = 0) get infinite priority, exactly as Eq. 1 does
        when N(c(s,a)) = 0, so every action is tried once before exploitation.
        The prior (r1 in paper mode) ranks unvisited siblings, following
        RAP Appendix A: "if there are actions that haven't been visited before,
        we estimate the Q value with lightweight local rewards".
        """
        if child.N == 0:
            # Eq. 1 has N(c(s,a)) = 0 for unvisited children, i.e. infinite priority:
            # every action is tried once before exploitation starts. The prior
            # (r1 in paper mode) only ranks unvisited siblings against each other,
            # as described in RAP Appendix A.
            return _UNVISITED_BASE + child.prior

        if self.aggr_child == "mean":
            exploitation = child.Q
        else:  # "max"
            exploitation = child.M

        exploration = self.w_exp * math.sqrt(math.log(max(parent.N, 1)) / child.N)
        return exploitation + exploration

    def expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Expands the selected leaf node using ReasoningAgent and WorldModel."""
        if node.is_terminal or node.depth >= self.max_depth:
            return []

        if node.children:
            return node.children

        if self.reasoning_agent is None or self.world_model is None:
            return []

        remaining_steps = max(0, self.max_depth - node.depth)
        actions = _propose_actions(self.reasoning_agent, node.state, remaining_steps)

        # Record every model interaction that produced this node's children so the
        # inspector can replay the full prompt/response log for the step.
        node.llm_calls = []
        seen_traces: set[int] = set()
        for action in actions:
            trace = getattr(action, "llm_trace", None)
            if trace is None or id(trace) in seen_traces:
                continue
            seen_traces.add(id(trace))
            entry = _action_call_entry(action)
            if entry is not None:
                node.llm_calls.append(entry)

        for action in actions:
            reward_components: dict[str, Any] = {}
            try:
                next_state, step_reward, is_terminal = _safe_step(
                    self.world_model,
                    node.state,
                    action,
                    lightweight=False,
                    call_log=node.llm_calls,
                    reward_sink=reward_components,
                )
            except (ValueError, KeyError, AssertionError):
                # A hallucinated world-model transition must not kill the search:
                # the action is simply not added to the tree.
                continue
            # The paper terminates a node when the goal is met or the depth limit is hit.
            depth_terminal = node.depth + 1 >= self.max_depth
            child = MCTSNode(
                state=next_state,
                action=action,
                parent=node,
                depth=node.depth + 1,
                reward=step_reward,
                is_terminal=is_terminal or depth_terminal,
                thought=getattr(action, "thought", None),
                llm_trace=getattr(action, "llm_trace", None),
            )
            child.set_reward_components(reward_components)
            child.prior = _action_prior(action)
            node.children.append(child)

        if self.event_bus:
            self.event_bus.emit(EventType.ON_EXPAND, {"node": node, "children": node.children})

        return node.children

    def simulate(self, node: MCTSNode, base_reward: float = 0.0) -> tuple[list[MCTSNode], float]:
        """Performs rollout simulation until terminal state or max depth.

        Args:
            node: Rollout start node.
            base_reward: Reward already accumulated by the caller (the expansion
                child chosen by Algorithm 1, line 16).
        """
        curr = node
        sim_path: list[MCTSNode] = [curr]
        rollout_reward = base_reward
        discount_power = 1.0

        while not curr.is_terminal and curr.depth < self.max_depth:
            if self.reasoning_agent is None or self.world_model is None:
                break
            remaining_steps = max(0, self.max_depth - curr.depth)
            actions = _propose_actions(self.reasoning_agent, curr.state, remaining_steps, lightweight=True)
            if not actions:
                break

            best_action = None
            best_next_state = None
            best_reward = -math.inf
            best_is_terminal = False

            best_components: dict[str, Any] = {}
            for action in actions:
                reward_components: dict[str, Any] = {}
                try:
                    next_state, reward, is_terminal = _safe_step(
                        self.world_model,
                        curr.state,
                        action,
                        lightweight=True,
                        reward_sink=reward_components,
                    )
                except (ValueError, KeyError, AssertionError):
                    continue
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
                    best_next_state = next_state
                    best_is_terminal = is_terminal
                    best_components = reward_components

            if best_action is None:
                break

            next_node = MCTSNode(
                state=best_next_state,
                action=best_action,
                parent=curr,
                depth=curr.depth + 1,
                reward=best_reward,
                is_terminal=best_is_terminal or curr.depth + 1 >= self.max_depth,
                thought=getattr(best_action, "thought", None),
                llm_trace=getattr(best_action, "llm_trace", None),
            )
            next_node.set_reward_components(best_components)
            sim_path.append(next_node)
            rollout_reward += discount_power * best_reward
            discount_power *= self.discount
            curr = next_node
            if self.event_bus:
                self.event_bus.emit(
                    EventType.ON_SIMULATE_STEP,
                    {
                        "node": node,
                        "current_node": next_node,
                        "sim_path": list(sim_path),
                        "rollout_reward": rollout_reward,
                    },
                )

        if self.event_bus:
            self.event_bus.emit(
                EventType.ON_SIMULATE,
                {"node": node, "sim_path": sim_path, "rollout_reward": rollout_reward},
            )

        return sim_path, rollout_reward

    def backpropagate(self, path: list[MCTSNode], rollout_reward: float = 0.0) -> None:
        """Backpropagates rewards along the selected path back to the root.

        Q(s, a) is the mean return over simulations (Eq. 1). Eq. 2 and §4.3
        describe the update as the average reward of future steps, so Q is kept
        as a running average; ``M`` stores the maximum return for the
        alternative "max" selection criterion.

        The per-visit value is the mean of the rewards of the future steps of
        the node, as the paper describes it. The reference implementation
        divides the same sum by one more than the number of steps, which
        includes a phantom zero step in the average.
        """
        reward = rollout_reward
        coeff = 0

        for node in reversed(path):
            reward = node.reward + self.discount * reward
            coeff = coeff * self.discount + 1

            if self.aggr_reward == "mean":
                c_reward = reward / coeff
            else:
                c_reward = reward

            node.N += 1
            # Eq. 1 defines Q as the expected future reward and Eq. 2/§4.3
            # update it with the *average* reward of future steps. Keep Q as the
            # running mean over simulations (incremental average), not a sum:
            # a sum would make frequently visited nodes look better simply
            # because they were visited more often.
            node.Q += (c_reward - node.Q) / node.N
            node.M = max(node.M, c_reward)
            node.visited = True

        if self.event_bus:
            self.event_bus.emit(
                EventType.ON_BACKPROP,
                {"path": path, "rollout_reward": rollout_reward},
            )

    def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
        """Extracts the best trajectory from root according to 'reward' or 'visits'."""
        return extract_best_trajectory(root, criterion=criterion)
