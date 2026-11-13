"""Comprehensive unit and integration tests for Core MCTS Planning Engine.

Based on RAP (Reasoning via Planning) paper Section 3.3 and Algorithm 1.
Tests cover:
- MCTSNode attributes, invariants, tree operations, and path tracing.
- EventBus lifecycle hooks (ON_SELECT, ON_EXPAND, ON_SIMULATE, ON_BACKPROP, ON_ITERATION_COMPLETE).
- MCTS algorithm stages: selection (UCT), expansion, rollout simulation, backpropagation.
- Best trajectory extraction with reward and visit count criteria.
- Parameterized discount factors, exploration weights, and aggregation modes.
- Deterministic mock environments (GraphWorld and MathAccumulator).
"""

from __future__ import annotations
import math
from typing import Any, Optional
import pytest

from rap_demo.core.node import MCTSNode
from rap_demo.core.events import EventBus, EventType, Event
from rap_demo.core.mcts import MCTS, WorldModel, ReasoningAgent


# ============================================================================
# Deterministic Mock Environments
# ============================================================================


class DeterministicGraphWorld(WorldModel, ReasoningAgent):
    """A deterministic graph environment with known branching paths and rewards.

    Graph topology:
        START -> "left"  -> L1 (reward=1.0) -> "left_goal"  -> L_GOAL (reward=10.0, terminal)
              -> "right" -> R1 (reward=3.0) -> "right_goal" -> R_GOAL (reward=1.0, terminal)

    Total rewards:
        Left branch:  1.0 + 10.0 = 11.0 (Optimal)
        Right branch: 3.0 +  1.0 =  4.0 (Sub-optimal, but higher initial reward)
    """

    def __init__(self) -> None:
        self.transitions: dict[tuple[str, str], tuple[str, float, bool]] = {
            ("START", "left"): ("L1", 1.0, False),
            ("START", "right"): ("R1", 3.0, False),
            ("L1", "left_goal"): ("L_GOAL", 10.0, True),
            ("L1", "left_dead"): ("L_DEAD", -5.0, True),
            ("R1", "right_goal"): ("R_GOAL", 1.0, True),
        }
        self.actions: dict[str, list[str]] = {
            "START": ["left", "right"],
            "L1": ["left_goal", "left_dead"],
            "R1": ["right_goal"],
            "L_GOAL": [],
            "L_DEAD": [],
            "R_GOAL": [],
        }

    def get_actions(self, state: Any) -> list[Any]:
        return list(self.actions.get(state, []))

    def step(self, state: Any, action: Any) -> tuple[Any, float, bool]:
        key = (state, action)
        if key not in self.transitions:
            raise ValueError(f"Invalid transition from state {state!r} with action {action!r}")
        return self.transitions[key]


class MathAccumulatorWorld(WorldModel, ReasoningAgent):
    """A deterministic arithmetic planning world.

    State is an integer x, target is 10.
    Actions:
        "+1": adds 1 to x
        "+3": adds 3 to x
        "*2": multiplies x by 2

    Reward:
        +10.0 if reaching 10 exactly
        -1.0 if exceeding 10
        -0.1 step penalty otherwise
    Terminal:
        x >= 10
    """

    def __init__(self, target: int = 10, max_step_limit: int = 6) -> None:
        self.target = target
        self.max_step_limit = max_step_limit

    def get_actions(self, state: int) -> list[str]:
        if state >= self.target:
            return []
        return ["+1", "+3", "*2"]

    def step(self, state: int, action: str) -> tuple[int, float, bool]:
        if action == "+1":
            next_state = state + 1
        elif action == "+3":
            next_state = state + 3
        elif action == "*2":
            next_state = state * 2
        else:
            raise ValueError(f"Unknown action {action}")

        if next_state == self.target:
            return next_state, 10.0, True
        elif next_state > self.target:
            return next_state, -1.0, True
        else:
            return next_state, -0.1, False


# ============================================================================
# 1. MCTSNode Contract & Unit Tests
# ============================================================================


class TestMCTSNodeContract:
    """Verifies all structural, invariant, and behavioral contracts of MCTSNode."""

    def test_node_initialization_defaults(self) -> None:
        """Test default values on root node initialization."""
        node = MCTSNode(state="root_state")

        assert node.state == "root_state"
        assert node.action is None
        assert node.parent is None
        assert node.depth == 0
        assert node.reward == 0.0
        assert node.children == []
        assert node.visited is False
        assert node.is_terminal is False

        # Statistical counters
        assert node.N == 0
        assert node.Q == 0.0
        assert node.M == -math.inf

    def test_node_is_root_and_is_leaf(self) -> None:
        """Test is_root and is_leaf properties on root, intermediate, and leaf nodes."""
        root = MCTSNode(state="START")
        assert root.is_root is True
        assert root.is_leaf is True

        child = MCTSNode(state="CHILD", action="step1", parent=root, depth=1)
        root.children.append(child)

        assert root.is_root is True
        assert root.is_leaf is False
        assert child.is_root is False
        assert child.is_leaf is True

    def test_add_child_contract(self) -> None:
        """Test adding a child node establishes correct bidirectional relations and depth."""
        root = MCTSNode(state=0, depth=0)
        child = MCTSNode(state=1)

        result = root.add_child(child, action="increment")

        assert result is child
        assert len(root.children) == 1
        assert root.children[0] is child
        assert child.parent is root
        assert child.action == "increment"
        assert child.depth == 1
        assert child.is_root is False
        assert root.is_leaf is False

    def test_add_multiple_children(self) -> None:
        """Test adding multiple children maintains independent branches."""
        root = MCTSNode(state="hub")
        c1 = MCTSNode(state="c1", reward=1.0)
        c2 = MCTSNode(state="c2", reward=2.0)
        c3 = MCTSNode(state="c3", reward=3.0)

        root.add_child(c1, action="act1")
        root.add_child(c2, action="act2")
        root.add_child(c3, action="act3")

        assert len(root.children) == 3
        assert [c.action for c in root.children] == ["act1", "act2", "act3"]
        assert [c.depth for c in root.children] == [1, 1, 1]
        assert all(c.parent is root for c in root.children)

    def test_path_from_root_single_node(self) -> None:
        """Path from root for the root node is just [root]."""
        root = MCTSNode(state="only_root")
        assert root.path_from_root == [root]

    def test_path_from_root_deep_lineage(self) -> None:
        """Path from root traverses up the parent chain in correct genealogical order."""
        n0 = MCTSNode(state="s0", depth=0)
        n1 = MCTSNode(state="s1", action="a1")
        n2 = MCTSNode(state="s2", action="a2")
        n3 = MCTSNode(state="s3", action="a3")

        n0.add_child(n1, "a1")
        n1.add_child(n2, "a2")
        n2.add_child(n3, "a3")

        assert n3.depth == 3
        path = n3.path_from_root
        assert path == [n0, n1, n2, n3]
        assert [node.state for node in path] == ["s0", "s1", "s2", "s3"]

    def test_children_list_independence(self) -> None:
        """Ensures that children lists are not shared between instances."""
        node_a = MCTSNode(state="A")
        node_b = MCTSNode(state="B")

        node_a.add_child(MCTSNode(state="A1"), "actA")
        assert len(node_a.children) == 1
        assert len(node_b.children) == 0


# ============================================================================
# 2. EventBus Contract & Lifecycle Tests
# ============================================================================


class TestEventBusContract:
    """Verifies publish-subscribe mechanics and search lifecycle event emissions."""

    def test_event_bus_subscribe_and_emit(self) -> None:
        """Test that subscribers receive emitted events with the exact payload."""
        bus = EventBus()
        received_events: list[Event] = []

        def on_select(event: Event) -> None:
            received_events.append(event)

        bus.subscribe(EventType.ON_SELECT, on_select)
        bus.emit(EventType.ON_SELECT, {"node_state": "S1"})

        assert len(received_events) == 1
        assert received_events[0].type == EventType.ON_SELECT
        assert received_events[0].data["node_state"] == "S1"

    def test_event_bus_multiple_subscribers_order(self) -> None:
        """Test multiple subscribers on the same event type execute in subscription order."""
        bus = EventBus()
        call_log: list[str] = []

        bus.subscribe(EventType.ON_EXPAND, lambda e: call_log.append("first"))
        bus.subscribe(EventType.ON_EXPAND, lambda e: call_log.append("second"))

        bus.emit(EventType.ON_EXPAND, {})
        assert call_log == ["first", "second"]

    def test_event_bus_unsubscribe(self) -> None:
        """Test unsubscribing stops receiving events while leaving other subscribers intact."""
        bus = EventBus()
        tracker: list[str] = []

        def listener_a(event: Event) -> None:
            tracker.append("A")

        def listener_b(event: Event) -> None:
            tracker.append("B")

        bus.subscribe(EventType.ON_BACKPROP, listener_a)
        bus.subscribe(EventType.ON_BACKPROP, listener_b)

        bus.emit(EventType.ON_BACKPROP, {})
        assert tracker == ["A", "B"]

        bus.unsubscribe(EventType.ON_BACKPROP, listener_a)
        bus.emit(EventType.ON_BACKPROP, {})
        assert tracker == ["A", "B", "B"]

    def test_event_bus_event_type_isolation(self) -> None:
        """Emitting an event of type X does not invoke subscribers of type Y."""
        bus = EventBus()
        simulate_called = False

        bus.subscribe(EventType.ON_SIMULATE, lambda e: setattr(simulate_called, "__bool__", True))
        bus.emit(EventType.ON_SELECT, {"some": "data"})

        assert not simulate_called

    def test_event_bus_clear(self) -> None:
        """Clear removes all registered listeners across all event types."""
        bus = EventBus()
        calls = 0

        bus.subscribe(EventType.ON_SELECT, lambda e: calls.__add__(1))
        bus.subscribe(EventType.ON_EXPAND, lambda e: calls.__add__(1))
        bus.clear()

        bus.emit(EventType.ON_SELECT, {})
        bus.emit(EventType.ON_EXPAND, {})
        assert calls == 0


# ============================================================================
# 3. MCTS Core Unit Tests: UCT, Selection, Expansion, Simulation, Backprop
# ============================================================================


class TestMCTSSelectionAndUCT:
    """Verifies UCT mathematical formula and leaf selection contracts."""

    @pytest.fixture
    def mcts_default(self) -> MCTS:
        return MCTS(w_exp=1.0, discount=1.0, aggr_reward="sum", aggr_child="max")

    def test_uct_value_unvisited_child_priority(self, mcts_default: MCTS) -> None:
        """Unvisited child (N=0) must yield infinity or highest priority over visited children."""
        parent = MCTSNode(state="P")
        parent.N = 5

        visited_child = MCTSNode(state="C1", parent=parent)
        visited_child.N = 5
        visited_child.Q = 2.0
        visited_child.M = 2.0

        unvisited_child = MCTSNode(state="C2", parent=parent)
        unvisited_child.N = 0

        visited_uct = mcts_default.uct_value(parent, visited_child)
        unvisited_uct = mcts_default.uct_value(parent, unvisited_child)

        assert unvisited_uct > visited_uct
        assert math.isinf(unvisited_uct) or unvisited_uct >= 1e6

    def test_uct_value_exact_formula_calculation(self) -> None:
        """Verifies exact UCT calculation: Q + w * sqrt(ln(N_parent) / N_child)."""
        w_exp = 1.414
        mcts = MCTS(w_exp=w_exp, aggr_child="mean")

        parent = MCTSNode(state="root")
        parent.N = 10

        child = MCTSNode(state="child", parent=parent)
        child.N = 3
        child.Q = 4.5

        expected_bonus = w_exp * math.sqrt(math.log(10) / 3)
        expected_uct = 4.5 + expected_bonus

        actual_uct = mcts.uct_value(parent, child)
        assert pytest.approx(actual_uct, rel=1e-5) == expected_uct

    @pytest.mark.parametrize("aggr_child, expected_stat", [("max", 5.0), ("mean", 3.0)])
    def test_uct_value_aggr_child_modes(self, aggr_child: str, expected_stat: float) -> None:
        """UCT calculation correctly switches between M(s,a) for 'max' and Q(s,a) for 'mean'."""
        mcts = MCTS(w_exp=1.0, aggr_child=aggr_child)
        parent = MCTSNode(state="P")
        parent.N = 4

        child = MCTSNode(state="C", parent=parent)
        child.N = 2
        child.Q = 3.0
        child.M = 5.0

        expected = expected_stat + 1.0 * math.sqrt(math.log(4) / 2)
        assert pytest.approx(mcts.uct_value(parent, child), rel=1e-5) == expected

    def test_select_descends_to_unvisited_leaf(self) -> None:
        """Selection traverses down the tree until finding an unexpanded node."""
        env = DeterministicGraphWorld()
        mcts = MCTS(world_model=env, reasoning_agent=env)

        root = MCTSNode(state="START")
        root.N = 2

        child_left = MCTSNode(state="L1", action="left", parent=root)
        child_left.N = 1
        child_left.Q = 1.0

        child_right = MCTSNode(state="R1", action="right", parent=root)
        child_right.N = 0  # Unvisited

        root.children = [child_left, child_right]

        path = mcts.select(root)
        assert path[-1] is child_right
        assert path == [root, child_right]

    def test_select_stops_at_terminal_node(self) -> None:
        """Selection stops when reaching a terminal node even if visited."""
        env = DeterministicGraphWorld()
        mcts = MCTS(world_model=env, reasoning_agent=env)

        root = MCTSNode(state="START")
        root.N = 1
        terminal_child = MCTSNode(state="L_GOAL", action="goal", parent=root, is_terminal=True)
        terminal_child.N = 1
        root.children = [terminal_child]

        path = mcts.select(root)
        assert path[-1] is terminal_child


class TestMCTSExpansion:
    """Verifies node expansion with ReasoningAgent and WorldModel."""

    def test_expand_generates_children_from_environment(self) -> None:
        """Expansion queries ReasoningAgent for actions and WorldModel for state transitions."""
        env = DeterministicGraphWorld()
        mcts = MCTS(world_model=env, reasoning_agent=env)

        root = MCTSNode(state="START", depth=0)
        children = mcts.expand(root)

        assert len(children) == 2
        assert len(root.children) == 2
        states = {c.state for c in children}
        assert states == {"L1", "R1"}

        left_node = next(c for c in children if c.state == "L1")
        assert left_node.action == "left"
        assert left_node.reward == 1.0
        assert left_node.depth == 1
        assert left_node.parent is root
        assert left_node.is_terminal is False

    def test_expand_copies_reward_components_from_world_model_sink(self) -> None:
        """Expansion stores r1/r2/r3 on the child so the inspector can show them."""

        class ComponentWorld:
            def get_actions(self, state: Any, remaining_steps: int | None = None) -> list[str]:
                return ["go"]

            def step(
                self,
                state: Any,
                action: Any,
                reward_sink: dict[str, Any] | None = None,
            ) -> tuple[str, float, bool]:
                if reward_sink is not None:
                    reward_sink.update({"r1": 0.2, "r2": 0.3, "r3": 0.4, "reward": 0.3})
                return "next", 0.3, False

        mcts = MCTS(world_model=ComponentWorld(), reasoning_agent=ComponentWorld())
        root = MCTSNode(state="start", depth=0)

        children = mcts.expand(root)

        assert len(children) == 1
        assert children[0].r1 == 0.2
        assert children[0].r2 == 0.3
        assert children[0].r3 == 0.4

    def test_expand_terminal_node_produces_no_children(self) -> None:
        """Expanding a terminal state yields an empty child list and does not mutate tree."""
        env = DeterministicGraphWorld()
        mcts = MCTS(world_model=env, reasoning_agent=env)

        terminal_node = MCTSNode(state="L_GOAL", depth=2, is_terminal=True)
        children = mcts.expand(terminal_node)

        assert children == []
        assert len(terminal_node.children) == 0


class TestMCTSSimulation:
    """Verifies greedy rollout simulation up to terminal state or max depth."""

    def test_simulate_reaches_terminal_state(self) -> None:
        """Rollout from L1 reaches terminal L_GOAL and accumulates correct reward."""
        env = DeterministicGraphWorld()
        mcts = MCTS(world_model=env, reasoning_agent=env, max_depth=5)

        node = MCTSNode(state="L1", depth=1)
        sim_path, rollout_reward = mcts.simulate(node)

        # In DeterministicGraphWorld, L1 can transition to L_GOAL (reward=10) or L_DEAD (reward=-5)
        # Greedy simulation should pick the highest immediate reward action
        assert sim_path[-1].is_terminal is True
        assert sim_path[-1].state == "L_GOAL"
        assert rollout_reward >= 10.0

    def test_simulate_respects_max_depth_limit(self) -> None:
        """Rollout halts when reaching max_depth even if terminal state was not reached."""
        env = MathAccumulatorWorld(target=100)  # Far target
        mcts = MCTS(world_model=env, reasoning_agent=env, max_depth=3)

        start_node = MCTSNode(state=0, depth=0)
        sim_path, _ = mcts.simulate(start_node)

        assert len(sim_path) <= 4
        assert sim_path[-1].depth <= 3


    def test_simulate_stops_when_the_rollout_repeats_a_state(self) -> None:
        """A move that undoes the previous one must not loop the rollout."""

        class _LoopingWorld:
            """Every action returns to the previous state."""

            def get_actions(self, state, remaining_steps=None):
                return ["back"]

            def step(self, state, action):
                return state, 1.0, False

        mcts = MCTS(world_model=_LoopingWorld(), reasoning_agent=_LoopingWorld(), max_depth=8)

        node = MCTSNode(state="S0", depth=0)
        sim_path, _ = mcts.simulate(node)

        # The first step repeats the initial state, so the rollout stops there.
        assert len(sim_path) == 1


class TestMCTSBackpropagation:
    """Verifies backpropagation updates N(s), Q(s,a), and M(s,a) according to RAP Eq. 2."""

    def test_backpropagate_single_node(self) -> None:
        """Backpropagating a single node updates visit count and stats."""
        mcts = MCTS(discount=1.0, aggr_reward="sum")
        node = MCTSNode(state="root", reward=5.0)

        mcts.backpropagate([node], rollout_reward=0.0)

        assert node.N == 1
        assert node.Q == 5.0
        assert node.M == 5.0

        # Second backpropagation
        mcts.backpropagate([node], rollout_reward=0.0)
        assert node.N == 2
        assert node.Q == 5.0  # Mean of the two returns (5.0 and 5.0)
        assert node.M == 5.0

    def test_backpropagate_q_is_running_mean(self) -> None:
        """Q averages returns across visits (Eq. 1/2), it is not a running sum."""
        mcts = MCTS(discount=1.0, aggr_reward="sum")
        node = MCTSNode(state="root", reward=5.0)

        mcts.backpropagate([node], rollout_reward=0.0)  # return 5.0
        mcts.backpropagate([node], rollout_reward=5.0)  # return 10.0

        assert node.N == 2
        assert node.Q == pytest.approx(7.5)
        assert node.M == 10.0

    def test_backpropagate_two_step_path_discounted(self) -> None:
        """Backpropagation with discount factor gamma correctly discounts future returns.

        Path: Root (reward=1.0) -> Child (reward=4.0) with gamma=0.5.
        Return at Child: 4.0
        Return at Root:  1.0 + 0.5 * 4.0 = 3.0
        """
        mcts = MCTS(discount=0.5, aggr_reward="sum")
        root = MCTSNode(state="root", depth=0, reward=1.0)
        child = MCTSNode(state="child", depth=1, reward=4.0, parent=root)
        root.children.append(child)

        path = [root, child]
        mcts.backpropagate(path, rollout_reward=0.0)

        assert child.N == 1
        assert child.M == 4.0

        assert root.N == 1
        assert root.M == 3.0

    def test_backpropagate_aggr_reward_mean(self) -> None:
        """When aggr_reward is 'mean', future rewards are normalized by step count."""
        mcts = MCTS(discount=1.0, aggr_reward="mean")
        root = MCTSNode(state="root", reward=2.0)
        child = MCTSNode(state="child", reward=6.0, parent=root)

        path = [root, child]
        mcts.backpropagate(path, rollout_reward=0.0)

        assert child.N == 1
        assert child.M == 6.0

        # At root, sum = 2.0 + 6.0 = 8.0 over 2 nodes => mean = 4.0
        assert root.N == 1
        assert pytest.approx(root.M, rel=1e-5) == 4.0

    def test_backpropagate_negative_rewards(self) -> None:
        """Handles negative rewards without invalid math or breaking M update."""
        mcts = MCTS(discount=1.0, aggr_reward="sum")
        root = MCTSNode(state="root", reward=0.0)
        dead = MCTSNode(state="dead", reward=-10.0, parent=root)

        mcts.backpropagate([root, dead], rollout_reward=0.0)

        assert dead.N == 1
        assert dead.M == -10.0
        assert root.N == 1
        assert root.M == -10.0


# ============================================================================
# 4. Best Trajectory Extraction Tests
# ============================================================================


class TestBestTrajectoryExtraction:
    """Verifies optimal trajectory extraction using reward and visit count criteria."""

    @pytest.fixture
    def populated_tree(self) -> MCTSNode:
        """Constructs a search tree with two distinct terminal branches:

        Root (N=10)
          ├── Left (N=3, M=15.0) -> LeftGoal (N=3, reward=15.0)
          └── Right (N=7, M=5.0)  -> RightGoal (N=7, reward=5.0)
        """
        root = MCTSNode(state="root")
        root.N = 10

        left = MCTSNode(state="left", action="L", parent=root, depth=1)
        left.N = 3
        left.M = 15.0
        left.Q = 15.0
        left_goal = MCTSNode(state="left_goal", action="LG", parent=left, depth=2, is_terminal=True, reward=15.0)
        left_goal.N = 3
        left_goal.M = 15.0
        left.children = [left_goal]

        right = MCTSNode(state="right", action="R", parent=root, depth=1)
        right.N = 7
        right.M = 5.0
        right.Q = 5.0
        right_goal = MCTSNode(state="right_goal", action="RG", parent=right, depth=2, is_terminal=True, reward=5.0)
        right_goal.N = 7
        right_goal.M = 5.0
        right.children = [right_goal]

        root.children = [left, right]
        return root

    def test_best_trajectory_by_reward(self, populated_tree: MCTSNode) -> None:
        """Criterion 'reward' extracts highest cumulative reward path (Left branch)."""
        mcts = MCTS()
        traj = mcts.get_best_trajectory(populated_tree, criterion="reward")

        assert len(traj) == 3
        assert [n.state for n in traj] == ["root", "left", "left_goal"]

    def test_best_trajectory_by_visits(self, populated_tree: MCTSNode) -> None:
        """Criterion 'visits' extracts most explored path (Right branch)."""
        mcts = MCTS()
        traj = mcts.get_best_trajectory(populated_tree, criterion="visits")

        assert len(traj) == 3
        assert [n.state for n in traj] == ["root", "right", "right_goal"]

    def test_best_trajectory_single_root_node(self) -> None:
        """Root with no children returns [root] trajectory gracefully."""
        mcts = MCTS()
        root = MCTSNode(state="isolated")
        traj = mcts.get_best_trajectory(root, criterion="reward")

        assert traj == [root]


# ============================================================================
# 5. Full MCTS Integration Tests with Lifecycle Event Verification
# ============================================================================


class TestMCTSFullSearchIntegration:
    """End-to-end integration tests running full MCTS iterations on mock environments."""

    def test_search_on_deterministic_graph_world(self) -> None:
        """MCTS converges to the optimal 11.0 reward trajectory (left -> left_goal)."""
        env = DeterministicGraphWorld()
        bus = EventBus()
        mcts = MCTS(
            w_exp=1.0,
            discount=1.0,
            aggr_reward="sum",
            aggr_child="max",
            world_model=env,
            reasoning_agent=env,
            event_bus=bus,
        )

        lifecycle_events: list[EventType] = []
        bus.subscribe(EventType.ON_SELECT, lambda e: lifecycle_events.append(e.type))
        bus.subscribe(EventType.ON_EXPAND, lambda e: lifecycle_events.append(e.type))
        bus.subscribe(EventType.ON_SIMULATE, lambda e: lifecycle_events.append(e.type))
        bus.subscribe(EventType.ON_BACKPROP, lambda e: lifecycle_events.append(e.type))
        bus.subscribe(EventType.ON_ITERATION_COMPLETE, lambda e: lifecycle_events.append(e.type))

        root = mcts.search(root_state="START", num_iterations=10)

        # Invariants after 10 iterations
        assert root.N == 10
        assert len(root.children) == 2
        assert root.is_leaf is False

        # Verify all lifecycle events were triggered at least once
        assert EventType.ON_SELECT in lifecycle_events
        assert EventType.ON_EXPAND in lifecycle_events
        assert EventType.ON_SIMULATE in lifecycle_events
        assert EventType.ON_BACKPROP in lifecycle_events
        assert EventType.ON_ITERATION_COMPLETE in lifecycle_events

        # Check optimal trajectory selection
        best_traj = mcts.get_best_trajectory(root, criterion="reward")
        assert [n.state for n in best_traj] == ["START", "L1", "L_GOAL"]

    def test_search_on_math_accumulator_world(self) -> None:
        """MCTS solves integer target path: 1 -> (+3) -> 4 -> (*2) -> 8 -> (+1) -> 9 -> (+1) -> 10."""
        env = MathAccumulatorWorld(target=10)
        mcts = MCTS(
            w_exp=1.2,
            discount=0.95,
            world_model=env,
            reasoning_agent=env,
            max_depth=6,
        )

        root = mcts.search(root_state=1, num_iterations=15)
        best_traj = mcts.get_best_trajectory(root, criterion="reward")

        assert best_traj[0].state == 1
        assert best_traj[-1].state == 10
        assert best_traj[-1].is_terminal is True
