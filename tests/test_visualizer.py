"""Comprehensive unit and integration tests for Phase 5: Real-Time Visualizer.

Covers:
1. `src.rap_demo.visualizer.serializer`:
   - `serialize_tree`: Converts MCTS tree into JSON-serializable hierarchy
     (id, state, action, visits, Q, M, reward, uct, is_terminal, children).
   - `serialize_blocksworld_state`: Converts BlockState into visual tabletop
     representation (stacks of colored blocks on table, hand holding block, clear blocks).
2. `src.rap_demo.visualizer.console`:
   - `RichMCTSConsole`:
     - Renders tree with `rich.tree.Tree`.
     - Formats node labels with visits, Q, and action text.
     - Renders summary table of best trajectory and search metrics.
3. `src.rap_demo.visualizer.web.controller`:
   - `SearchController`:
     - Manages search state (`IDLE`, `RUNNING`, `PAUSED`, `FINISHED`).
     - Supports `step()`: runs exactly one MCTS iteration and emits events.
     - Supports `start()`, `pause()`, `resume()`, `reset()`.
     - Emits events to subscriber queues for SSE streaming.
4. `src.rap_demo.visualizer.web.app`:
   - FastAPI application testable with `httpx.AsyncClient`:
     - `GET /`: returns 200 with HTML content.
     - `GET /api/tree`: returns current tree JSON.
     - `GET /api/state`: returns domain state JSON.
     - `POST /api/search/step`: advances one iteration and returns updated status.
     - `POST /api/search/start`, `POST /api/search/pause`, `POST /api/search/reset`.
     - `GET /api/stream`: SSE endpoint emitting `text/event-stream`.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, AsyncGenerator, Optional
import httpx
from httpx import ASGITransport
import pytest
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from rap_demo.core.node import MCTSNode
from rap_demo.core.mcts import MCTS, WorldModel, ReasoningAgent
from rap_demo.core.events import EventBus
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState
from rap_demo.visualizer.serializer import (
    serialize_tree,
    serialize_blocksworld_state,
)
from rap_demo.visualizer.console import RichMCTSConsole
from rap_demo.visualizer.web.controller import (
    SearchController,
    SearchStatus,
    _build_tree_with_sim_preview,
)
from rap_demo.visualizer.web.app import create_app, app


# ============================================================================
# Test Fixtures & Mock Environments
# ============================================================================


class SimpleMockWorld(WorldModel, ReasoningAgent):
    """Deterministic 3-state chain environment for controller and MCTS tests."""

    def __init__(self) -> None:
        self.transitions = {
            ("S0", "go_to_s1"): ("S1", 1.0, False),
            ("S1", "go_to_goal"): ("GOAL", 10.0, True),
            ("S1", "go_to_trap"): ("TRAP", -5.0, True),
        }
        self.actions = {
            "S0": ["go_to_s1"],
            "S1": ["go_to_goal", "go_to_trap"],
            "GOAL": [],
            "TRAP": [],
        }

    def get_actions(self, state: Any) -> list[Any]:
        return list(self.actions.get(state, []))

    def step(self, state: Any, action: Any) -> tuple[Any, float, bool]:
        key = (state, action)
        if key not in self.transitions:
            return (state, -1.0, True)
        return self.transitions[key]


@pytest.fixture
def sample_mcts_tree() -> MCTSNode:
    """Creates a deterministic multi-level MCTS tree for serialization and rendering tests.

    Topology:
        Root (S0, N=10, Q=5.5, M=10.0)
        ├── Child 1: Action="pick_up red" -> S1 (N=6, Q=7.0, M=10.0, reward=1.0)
        │   ├── Grandchild 1A: Action="stack red on blue" -> S_GOAL (N=4, Q=10.0, M=10.0, reward=10.0, terminal)
        │   └── Grandchild 1B: Action="put_down red" -> S_FAIL (N=2, Q=1.0, M=1.0, reward=-2.0, terminal)
        └── Child 2: Action="pick_up blue" -> S2 (N=4, Q=3.0, M=4.0, reward=0.5)
    """
    root = MCTSNode(state="S0", depth=0)
    root.N = 10
    root.Q = 5.5
    root.M = 10.0

    # Child 1
    child1 = MCTSNode(
        state="S1",
        action=BlockAction(action_type=ActionType.PICK_UP, block="red"),
        depth=1,
        reward=1.0,
    )
    child1.N = 6
    child1.Q = 7.0
    child1.M = 10.0
    root.add_child(child1)

    # Grandchild 1A (terminal goal)
    gc1a = MCTSNode(
        state="S_GOAL",
        action=BlockAction(action_type=ActionType.STACK, block="red", target="blue"),
        depth=2,
        reward=10.0,
        is_terminal=True,
    )
    gc1a.N = 4
    gc1a.Q = 10.0
    gc1a.M = 10.0
    child1.add_child(gc1a)

    # Grandchild 1B (terminal failure)
    gc1b = MCTSNode(
        state="S_FAIL",
        action=BlockAction(action_type=ActionType.PUT_DOWN, block="red"),
        depth=2,
        reward=-2.0,
        is_terminal=True,
    )
    gc1b.N = 2
    gc1b.Q = 1.0
    gc1b.M = 1.0
    child1.add_child(gc1b)

    # Child 2
    child2 = MCTSNode(
        state="S2",
        action=BlockAction(action_type=ActionType.PICK_UP, block="blue"),
        depth=1,
        reward=0.5,
    )
    child2.N = 4
    child2.Q = 3.0
    child2.M = 4.0
    root.add_child(child2)

    return root


# ============================================================================
# 1. Tree & Blocksworld State Serialization Tests
# ============================================================================


class TestTreeSerializerContract:
    """Tests for `rap_demo.visualizer.serializer.serialize_tree`."""

    def test_serialize_single_root_node_structure(self) -> None:
        """A root node without children must contain all required schema fields."""
        root = MCTSNode(state="initial_state", depth=0)
        data = serialize_tree(root)

        assert isinstance(data, dict)
        assert "id" in data
        assert "state" in data
        assert "action" in data
        assert "visits" in data
        assert "Q" in data
        assert "M" in data
        assert "reward" in data
        assert "uct" in data
        assert "is_terminal" in data
        assert "children" in data

        assert data["action"] is None
        assert data["visits"] == 0
        assert data["reward"] == 0.0
        assert data["is_terminal"] is False
        assert data["children"] == []

    def test_serialize_tree_preserves_hierarchy_and_children(
        self, sample_mcts_tree: MCTSNode
    ) -> None:
        """Serializing a multi-level tree recursively serializes all children preserving hierarchy."""
        data = serialize_tree(sample_mcts_tree)

        assert data["state"] == "S0"
        assert data["visits"] == 10
        assert len(data["children"]) == 2

        # Verify Child 1
        c1 = data["children"][0]
        assert "red" in str(c1["action"])
        assert c1["visits"] == 6
        assert len(c1["children"]) == 2

        # Verify Grandchild 1A (terminal)
        gc1a = c1["children"][0]
        assert gc1a["is_terminal"] is True
        assert gc1a["reward"] == 10.0
        assert gc1a["visits"] == 4
        assert gc1a["children"] == []

    def test_serialize_tree_json_serializable(self, sample_mcts_tree: MCTSNode) -> None:
        """Serialized tree dictionary must be 100% JSON-serializable with json.dumps()."""
        data = serialize_tree(sample_mcts_tree)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        reloaded = json.loads(json_str)
        assert reloaded["visits"] == 10
        assert len(reloaded["children"]) == 2

    def test_serialize_tree_handles_infinite_m_and_unvisited_nodes(self) -> None:
        """Unvisited nodes have M = -math.inf which must be sanitized for valid standard JSON."""
        unvisited = MCTSNode(state="unvisited_state")
        assert unvisited.M == -math.inf
        assert unvisited.N == 0

        data = serialize_tree(unvisited)
        # Standard JSON does not permit -Infinity; serializer must convert or sanitize it
        # E.g., None, -999999.0, or json.dumps without error
        json_output = json.dumps(data)
        assert json_output is not None
        assert data["visits"] == 0

    def test_serialize_tree_unique_node_ids(self, sample_mcts_tree: MCTSNode) -> None:
        """Every node in the serialized tree must have a unique ID."""
        data = serialize_tree(sample_mcts_tree)

        node_ids: list[Any] = []

        def collect_ids(node_dict: dict[str, Any]) -> None:
            node_ids.append(node_dict["id"])
            for child in node_dict.get("children", []):
                collect_ids(child)

        collect_ids(data)
        assert len(node_ids) == 5  # root + 2 children + 2 grandchildren
        assert len(set(node_ids)) == 5, "All serialized node IDs must be distinct"

    def test_serialize_tree_with_blocksworld_state_and_action(self) -> None:
        """When state is BlockState and action is BlockAction, serializes them gracefully."""
        b_state = BlockState(
            clear={"red"},
            on_table={"blue"},
            on={"red": "blue"},
            holding=None,
        )
        b_action = BlockAction(action_type=ActionType.STACK, block="red", target="blue")
        node = MCTSNode(state=b_state, action=b_action, depth=1, reward=5.0)

        data = serialize_tree(node)
        assert data["reward"] == 5.0
        assert "stack" in str(data["action"]).lower()
        # Verify JSON dump works cleanly
        dumped = json.dumps(data)
        assert "red" in dumped

    @pytest.mark.parametrize("depth", [1, 5, 10])
    def test_serialize_tree_deep_linear_chain(self, depth: int) -> None:
        """Handles deep single-branch trajectories without recursion or stack overflow."""
        root = MCTSNode(state=0, depth=0)
        curr = root
        for d in range(1, depth + 1):
            next_node = MCTSNode(state=d, depth=d, reward=float(d))
            curr.add_child(next_node)
            curr = next_node

        serialized = serialize_tree(root)
        curr_dict = serialized
        for d in range(depth):
            assert len(curr_dict["children"]) == 1
            curr_dict = curr_dict["children"][0]
        assert curr_dict["reward"] == float(depth)

    def test_serialize_tree_includes_llm_trace(self) -> None:
        """Serialized nodes expose the model prompt/output trace for the inspector UI."""
        trace = {
            "source": "fallback",
            "prompt": "Goal: red on blue",
            "response": "not json",
            "reason": "Model output did not contain a JSON object.",
        }
        node = MCTSNode(state="S0", depth=0, llm_trace=trace)

        payload = serialize_tree(node)

        assert payload["llm_trace"] == trace

    def test_serialize_tree_includes_llm_calls(self) -> None:
        """Serialized nodes expose the full per-step call log for the inspector."""
        calls = [
            {
                "phase": "action-scoring",
                "action": "Pick up the red block.",
                "prompt": "P",
                "response": "pick up the red block",
                "logprob": -1.0,
                "prob": 0.5,
                "error": None,
            }
        ]
        node = MCTSNode(state="S0", depth=0, llm_calls=calls)

        payload = serialize_tree(node)

        assert payload["llm_calls"] == calls

    def test_sim_preview_nodes_expose_llm_trace(self) -> None:
        """Rollout preview nodes carry their trace so the inspector can explain them."""
        trace = {"source": "rollout", "prompt": None, "response": None}
        root = MCTSNode(state="S0", depth=0)
        sim_child = MCTSNode(
            state="S1",
            action="go_to_s1",
            parent=root,
            depth=1,
            reward=1.0,
            llm_trace=trace,
        )

        tree = _build_tree_with_sim_preview(root, root, [root, sim_child])

        preview = tree["children"][0]
        assert preview["is_preview"] is True
        assert preview["llm_trace"] == trace


class TestBlocksworldStateSerializer:
    """Tests for `rap_demo.visualizer.serializer.serialize_blocksworld_state`."""

    def test_serialize_all_blocks_on_table_flat(self) -> None:
        """When all blocks are on table, serializer produces 1-height stacks."""
        state = BlockState(
            clear={"red", "blue", "yellow"},
            on_table={"red", "blue", "yellow"},
            on={},
            holding=None,
        )
        result = serialize_blocksworld_state(state)

        assert isinstance(result, dict)
        assert "stacks" in result
        assert "holding" in result
        assert "clear" in result

        assert result["holding"] is None
        # Should have 3 stacks, each containing exactly one block
        assert len(result["stacks"]) == 3
        flattened = [b for stack in result["stacks"] for b in stack]
        assert sorted(flattened) == ["blue", "red", "yellow"]
        assert sorted(result["clear"]) == ["blue", "red", "yellow"]

    def test_serialize_single_vertical_tower(self) -> None:
        """Single tower on table: table -> red -> blue -> yellow -> clear."""
        state = BlockState(
            clear={"yellow"},
            on_table={"red"},
            on={"blue": "red", "yellow": "blue"},
            holding=None,
        )
        result = serialize_blocksworld_state(state)

        assert result["holding"] is None
        assert len(result["stacks"]) == 1
        # Stack ordered from bottom (on table) to top (clear)
        assert result["stacks"][0] == ["red", "blue", "yellow"]
        assert result["clear"] == ["yellow"]

    def test_serialize_multiple_disjoint_stacks(self) -> None:
        """Two stacks: [table -> A -> B] and [table -> C -> D]."""
        state = BlockState(
            clear={"B", "D"},
            on_table={"A", "C"},
            on={"B": "A", "D": "C"},
            holding=None,
        )
        result = serialize_blocksworld_state(state)

        assert len(result["stacks"]) == 2
        stacks_sorted = sorted(result["stacks"], key=lambda s: s[0])
        assert stacks_sorted == [["A", "B"], ["C", "D"]]
        assert sorted(result["clear"]) == ["B", "D"]

    def test_serialize_holding_block_with_empty_table(self) -> None:
        """Hand holding the only block in the universe."""
        state = BlockState(
            clear=set(),
            on_table=set(),
            on={},
            holding="orange",
        )
        result = serialize_blocksworld_state(state)

        assert result["holding"] == "orange"
        assert result["stacks"] == []
        assert result["clear"] == []

    def test_serialize_holding_block_with_other_stacks(self) -> None:
        """Hand holding one block while other blocks form a stack on the table."""
        state = BlockState(
            clear={"blue"},
            on_table={"red"},
            on={"blue": "red"},
            holding="green",
        )
        result = serialize_blocksworld_state(state)

        assert result["holding"] == "green"
        assert result["stacks"] == [["red", "blue"]]
        assert result["clear"] == ["blue"]

    def test_serialize_empty_state(self) -> None:
        """Empty state with no blocks."""
        state = BlockState()
        result = serialize_blocksworld_state(state)

        assert result["stacks"] == []
        assert result["holding"] is None
        assert result["clear"] == []

    def test_serialize_blocksworld_state_json_serializability(self) -> None:
        """Serialized blocksworld state must serialize with json.dumps without error."""
        state = BlockState(
            clear={"red"},
            on_table={"blue"},
            on={"red": "blue"},
            holding=None,
        )
        data = serialize_blocksworld_state(state)
        encoded = json.dumps(data)
        assert isinstance(encoded, str)
        decoded = json.loads(encoded)
        assert decoded["holding"] is None
        assert decoded["stacks"] == [["blue", "red"]]


# ============================================================================
# 2. Rich Terminal Console Visualizer Tests
# ============================================================================


class TestRichMCTSConsole:
    """Tests for `rap_demo.visualizer.console.RichMCTSConsole`."""

    def test_rich_console_initialization(self) -> None:
        """RichMCTSConsole initializes with default or custom rich Console."""
        custom_console = Console(record=True, width=100)
        visualizer = RichMCTSConsole(console=custom_console)
        assert visualizer is not None

    def test_render_tree_returns_rich_tree_instance(
        self, sample_mcts_tree: MCTSNode
    ) -> None:
        """`render_tree` converts MCTSNode hierarchy into a rich.tree.Tree."""
        visualizer = RichMCTSConsole()
        tree = visualizer.render_tree(sample_mcts_tree)

        assert isinstance(tree, Tree)
        # The tree root should have 2 children matching sample_mcts_tree
        assert len(tree.children) == 2

    def test_render_tree_formats_node_labels_with_metrics(
        self, sample_mcts_tree: MCTSNode
    ) -> None:
        """Node labels in the rendered Tree include visits (N), value (Q), and action info."""
        console = Console(record=True, width=120)
        visualizer = RichMCTSConsole(console=console)
        tree = visualizer.render_tree(sample_mcts_tree)
        console.print(tree)
        output = console.export_text()

        # Output should display visit counts and Q values
        assert "10" in output or "N=" in output
        assert "Q=" in output or "5.5" in output
        # Should display action texts
        assert "pick_up" in output.lower() or "pick up" in output.lower()

    def test_render_tree_max_depth_truncation(self, sample_mcts_tree: MCTSNode) -> None:
        """`render_tree` respects `max_depth` parameter and truncates deeper nodes."""
        visualizer = RichMCTSConsole()
        # Depth 1 allows root and child1/child2, but not grandchild 1A / 1B
        tree = visualizer.render_tree(sample_mcts_tree, max_depth=1)

        assert len(tree.children) == 2
        for child_tree in tree.children:
            assert len(child_tree.children) == 0

    def test_render_tree_terminal_node_styling(
        self, sample_mcts_tree: MCTSNode
    ) -> None:
        """Terminal nodes in the tree are visually distinct or labeled as terminal."""
        console = Console(record=True, width=120)
        visualizer = RichMCTSConsole(console=console)
        tree = visualizer.render_tree(sample_mcts_tree)
        console.print(tree)
        output = console.export_text()

        # Terminal nodes or rewards should be visible in text
        assert "terminal" in output.lower() or "10.0" in output

    def test_print_tree_writes_to_console(self, sample_mcts_tree: MCTSNode) -> None:
        """`print_tree` directly prints the rendered tree to the console."""
        console = Console(record=True, width=120)
        visualizer = RichMCTSConsole(console=console)
        visualizer.print_tree(sample_mcts_tree)
        text = console.export_text()
        assert len(text.strip()) > 0

    def test_render_summary_returns_rich_table(
        self, sample_mcts_tree: MCTSNode
    ) -> None:
        """`render_summary` produces a rich.table.Table displaying trajectory steps."""
        visualizer = RichMCTSConsole()
        best_path = sample_mcts_tree.children[0].children[0].path_from_root
        table = visualizer.render_summary(
            best_path, metrics={"iterations": 10, "best_reward": 10.0}
        )

        assert isinstance(table, Table)
        # Should have column headers for Depth/Step, Action, Reward, etc.
        col_names = [col.header.lower() for col in table.columns if isinstance(col.header, str)]
        assert any("step" in c or "depth" in c for c in col_names)
        assert any("action" in c for c in col_names)

    def test_print_summary_writes_to_console(
        self, sample_mcts_tree: MCTSNode
    ) -> None:
        """`print_summary` prints trajectory table and search metrics to console."""
        console = Console(record=True, width=120)
        visualizer = RichMCTSConsole(console=console)
        best_path = sample_mcts_tree.children[0].children[0].path_from_root
        visualizer.print_summary(best_path, metrics={"iterations": 10, "best_reward": 10.0})
        output = console.export_text()
        assert len(output.strip()) > 0
        assert "10" in output

    def test_rich_console_handles_single_root_tree(self) -> None:
        """Console cleanly renders a tree containing only the root node without crashing."""
        console = Console(record=True, width=120)
        visualizer = RichMCTSConsole(console=console)
        root = MCTSNode(state="lonely_root", depth=0)
        visualizer.print_tree(root)
        visualizer.print_summary([root], metrics={"iterations": 0})
        output = console.export_text()
        assert "lonely_root" in output or "N=" in output


# ============================================================================
# 3. Search Controller Tests (State Machine & Stepping)
# ============================================================================


class TestSearchController:
    """Tests for `rap_demo.visualizer.web.controller.SearchController`."""

    @pytest.fixture
    def mock_mcts(self) -> MCTS:
        world = SimpleMockWorld()
        return MCTS(
            world_model=world,
            reasoning_agent=world,
            max_depth=5,
        )

    @pytest.fixture
    def controller(self, mock_mcts: MCTS) -> SearchController:
        return SearchController(
            mcts=mock_mcts,
            initial_state="S0",
            max_iterations=10,
            step_delay=0.01,
        )

    def test_controller_initial_status_idle(self, controller: SearchController) -> None:
        """Controller starts in IDLE status at iteration 0."""
        assert controller.status == SearchStatus.IDLE
        assert controller.iteration == 0
        assert controller.root is not None
        assert controller.root.state == "S0"

    @pytest.mark.asyncio
    async def test_controller_step_advances_iteration(
        self, controller: SearchController
    ) -> None:
        """Invoking `step()` runs exactly one MCTS iteration and increments counter."""
        res = await controller.step()

        assert controller.iteration == 1
        assert res["iteration"] == 1
        assert res["status"] in (SearchStatus.IDLE, SearchStatus.PAUSED, "idle", "paused")
        assert controller.root.N >= 1

    @pytest.mark.asyncio
    async def test_controller_step_multiple_times(
        self, controller: SearchController
    ) -> None:
        """Calling `step()` sequentially builds out the MCTS tree iteration by iteration."""
        for i in range(1, 5):
            res = await controller.step()
            assert controller.iteration == i
            assert res["iteration"] == i

        assert controller.root.N == 4
        assert len(controller.root.children) > 0

    @pytest.mark.asyncio
    async def test_controller_start_pause_resume_lifecycle(
        self, controller: SearchController
    ) -> None:
        """Tests start -> running, pause -> paused, resume -> running state machine."""
        # 1. Start
        start_res = await controller.start()
        assert controller.status == SearchStatus.RUNNING
        assert start_res["status"] in (SearchStatus.RUNNING, "running")

        # Let it run briefly
        await asyncio.sleep(0.05)

        # 2. Pause
        pause_res = await controller.pause()
        assert controller.status == SearchStatus.PAUSED
        assert pause_res["status"] in (SearchStatus.PAUSED, "paused")
        paused_iter = controller.iteration
        assert paused_iter > 0

        # Wait to verify it has stopped advancing
        await asyncio.sleep(0.03)
        assert controller.iteration == paused_iter

        # 3. Resume
        resume_res = await controller.resume()
        assert controller.status == SearchStatus.RUNNING
        assert resume_res["status"] in (SearchStatus.RUNNING, "running")

        # Clean up by pausing
        await controller.pause()

    @pytest.mark.asyncio
    async def test_controller_reset_restores_initial_state(
        self, controller: SearchController
    ) -> None:
        """Reset resets iteration to 0, status to IDLE, and reinitializes tree root."""
        await controller.step()
        await controller.step()
        assert controller.iteration == 2

        reset_res = await controller.reset()
        assert controller.status == SearchStatus.IDLE
        assert controller.iteration == 0
        assert controller.root.N == 0
        assert len(controller.root.children) == 0
        assert reset_res["status"] in (SearchStatus.IDLE, "idle")

    @pytest.mark.asyncio
    async def test_controller_subscribers_receive_broadcast_events(
        self, controller: SearchController
    ) -> None:
        """Subscriber queues receive real-time event payloads when search steps."""
        queue1 = controller.subscribe()
        queue2 = controller.subscribe()

        await controller.step()

        # Both subscriber queues should receive phase events and the final step event
        events1: list[dict[str, Any]] = []
        events2: list[dict[str, Any]] = []
        while True:
            ev1 = await asyncio.wait_for(queue1.get(), timeout=1.0)
            events1.append(ev1)
            if ev1.get("type") in ("step", "iteration_complete", "tree_update"):
                break
        while True:
            ev2 = await asyncio.wait_for(queue2.get(), timeout=1.0)
            events2.append(ev2)
            if ev2.get("type") in ("step", "iteration_complete", "tree_update"):
                break

        assert events1[-1]["type"] in ("step", "iteration_complete", "tree_update")
        assert events1[-1]["iteration"] == 1
        assert events2[-1]["iteration"] == 1

        controller.unsubscribe(queue1)
        controller.unsubscribe(queue2)

    @pytest.mark.asyncio
    async def test_controller_unsubscribe_stops_events(
        self, controller: SearchController
    ) -> None:
        """Unsubscribed queues no longer receive subsequent event broadcasts."""
        queue = controller.subscribe()
        await controller.step()
        _ = await queue.get()
        while not queue.empty():
            queue.get_nowait()

        controller.unsubscribe(queue)
        await controller.step()

        assert queue.empty()

    @pytest.mark.asyncio
    async def test_controller_finishes_at_max_iterations(
        self, controller: SearchController
    ) -> None:
        """When reaching `max_iterations`, controller status transitions to FINISHED."""
        # max_iterations is 10
        for _ in range(10):
            await controller.step()

        assert controller.iteration == 10
        assert controller.status == SearchStatus.FINISHED


# ============================================================================
# 4. Web Dashboard & REST/SSE Endpoints Tests (FastAPI + httpx)
# ============================================================================


class TestVisualizerWebApp:
    """Integration tests for FastAPI dashboard and REST/SSE API endpoints."""

    @pytest.fixture
    def test_controller(self) -> SearchController:
        world = SimpleMockWorld()
        mcts = MCTS(world_model=world, reasoning_agent=world, max_depth=5)
        return SearchController(
            mcts=mcts,
            initial_state="S0",
            max_iterations=10,
            step_delay=0.01,
        )

    @pytest.fixture
    def app_instance(self, test_controller: SearchController) -> Any:
        return create_app(controller=test_controller)

    @pytest.fixture
    async def async_client(self, app_instance: Any) -> AsyncGenerator[httpx.AsyncClient, None]:
        transport = ASGITransport(app=app_instance)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_get_root_dashboard_html(self, async_client: httpx.AsyncClient) -> None:
        """GET / returns HTTP 200 with HTML dashboard content."""
        response = await async_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        body = response.text
        # Dashboard HTML must contain visualizer structural elements
        assert "<html" in body.lower()
        assert "tree" in body.lower() or "rap" in body.lower()

    @pytest.mark.asyncio
    async def test_get_api_tree_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """GET /api/tree returns JSON serialized MCTS tree hierarchy."""
        response = await async_client.get("/api/tree")
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")
        data = response.json()
        assert "id" in data or "root" in data or "state" in data

    @pytest.mark.asyncio
    async def test_get_api_state_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """GET /api/state returns current domain state representation."""
        response = await async_client.get("/api/state")
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")
        data = response.json()
        assert data is not None

    @pytest.mark.asyncio
    async def test_post_search_step_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """POST /api/search/step runs one iteration and returns updated status."""
        response = await async_client.post("/api/search/step")
        assert response.status_code == 200
        data = response.json()
        assert "iteration" in data
        assert data["iteration"] == 1
        assert "status" in data

    @pytest.mark.asyncio
    async def test_post_search_lifecycle_endpoints(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Tests POST /api/search/start, /pause, and /reset endpoints."""
        # 1. Start
        start_res = await async_client.post("/api/search/start")
        assert start_res.status_code == 200
        assert start_res.json()["status"] in ("running", SearchStatus.RUNNING)

        # 2. Pause
        pause_res = await async_client.post("/api/search/pause")
        assert pause_res.status_code == 200
        assert pause_res.json()["status"] in ("paused", SearchStatus.PAUSED)

        # 3. Reset
        reset_res = await async_client.post("/api/search/reset")
        assert reset_res.status_code == 200
        assert reset_res.json()["status"] in ("idle", SearchStatus.IDLE)
        assert reset_res.json()["iteration"] == 0

    @pytest.mark.asyncio
    async def test_get_api_stream_sse_endpoint(
        self, app_instance: Any, test_controller: SearchController
    ) -> None:
        """GET /api/stream streams Server-Sent Events (SSE) text/event-stream."""
        transport = ASGITransport(app=app_instance)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("GET", "/api/stream") as stream:
                assert stream.status_code == 200
                assert "text/event-stream" in stream.headers.get("content-type", "")

                # Trigger a step to generate an SSE broadcast event
                step_task = asyncio.create_task(test_controller.step())

                # Read the streamed event chunk
                chunk_found = False
                async for line in stream.aiter_lines():
                    if line.startswith("data:") or "step" in line or "iteration" in line:
                        chunk_found = True
                        break

                await step_task
                assert chunk_found, "SSE stream must emit event data for subscribers"

    @pytest.mark.asyncio
    async def test_default_app_instance_defined(self) -> None:
        """Module-level `app` FastAPI instance is importable and configured."""
        from fastapi import FastAPI

        assert app is not None
        assert isinstance(app, FastAPI)

    @pytest.mark.asyncio
    async def test_get_api_meta_endpoint(
        self, test_controller: SearchController
    ) -> None:
        """GET /api/meta returns max_iterations, step_delay_ms, goal, mode, and best_path_ids."""
        test_controller.goal = ["the red block is on top of the blue block"]
        test_controller.mode = "hybrid"
        app_with_meta = create_app(controller=test_controller)
        transport = ASGITransport(app=app_with_meta)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/meta")
            assert response.status_code == 200
            data = response.json()
            assert data["max_iterations"] == 10
            assert data["step_delay_ms"] == 10
            assert data["mode"] == "hybrid"
            assert data["goal"] == ["the red block is on top of the blue block"]
            assert isinstance(data["best_path_ids"], list)
            assert len(data["best_path_ids"]) >= 1

    @pytest.mark.asyncio
    async def test_post_search_config_updates_step_delay(
        self, async_client: httpx.AsyncClient, test_controller: SearchController
    ) -> None:
        """POST /api/search/config updates controller step_delay and validates bounds."""
        response = await async_client.post("/api/search/config", json={"step_delay_ms": 250})
        assert response.status_code == 200
        assert response.json()["step_delay_ms"] == 250
        assert math.isclose(test_controller.step_delay, 0.25, rel_tol=1e-6)

        invalid_res = await async_client.post("/api/search/config", json={"step_delay_ms": 5})
        assert invalid_res.status_code == 422

    @pytest.mark.asyncio
    async def test_step_includes_best_path_ids_and_max_iterations(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """POST /api/search/step includes best_path_ids and max_iterations for UI tracking."""
        response = await async_client.post("/api/search/step")
        assert response.status_code == 200
        data = response.json()
        assert data["max_iterations"] == 10
        assert isinstance(data["best_path_ids"], list)
        assert len(data["best_path_ids"]) >= 1

    @pytest.mark.asyncio
    async def test_step_maps_llm_error_to_502_json_instead_of_500(self) -> None:
        """POST /api/search/step returns HTTP 502 with detail when step raises LLMError."""
        from rap_demo.llm.base import LLMError

        class FailingController(SearchController):
            async def step(self) -> dict[str, Any]:
                raise LLMError("Malformed LLM response: missing message content")

        app = create_app(FailingController())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/search/step")
            assert response.status_code == 502
            assert "Malformed LLM response" in response.json()["detail"]


# ============================================================================
# 4. Problem Editor Tests (initial state / goal replacement)
# ============================================================================


class TestProblemEditor:
    """Tests for `POST /api/problem` and `SearchController.set_problem`."""

    @pytest.fixture
    def controller(self) -> SearchController:
        world = SimpleMockWorld()
        mcts = MCTS(world_model=world, reasoning_agent=world, max_depth=5)
        return SearchController(
            mcts=mcts,
            initial_state="S0",
            max_iterations=10,
            step_delay=0.01,
            goal=["the red block is on top of the blue block"],
        )

    @pytest.fixture
    async def async_client(self, controller: SearchController) -> AsyncGenerator[httpx.AsyncClient, None]:
        transport = ASGITransport(app=create_app(controller=controller))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_post_problem_updates_goal_state_and_resets_tree(
        self, async_client: httpx.AsyncClient, controller: SearchController
    ) -> None:
        """POST /api/problem replaces the problem, resets iteration, and returns the new state."""
        await controller.step()
        assert controller.iteration == 1

        payload = {
            "initial_state": json.dumps(
                {
                    "clear": ["red", "blue"],
                    "on_table": ["red", "blue"],
                    "on": {},
                    "holding": None,
                }
            ),
            "goal": [
                "the red block is on top of the blue block",
                "the blue block is on top of the red block",
            ],
        }
        response = await async_client.post("/api/problem", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["iteration"] == 0
        assert data["status"] in ("idle", SearchStatus.IDLE)
        assert data["goal"] == payload["goal"]
        assert data["state"]["stacks"] == [["blue"], ["red"]] or data["state"]["stacks"] == [
            ["red"],
            ["blue"],
        ]
        assert data["tree"]["state"] == data["state"]
        assert controller.iteration == 0
        assert isinstance(controller.initial_state, BlockState)
        assert controller.goal == payload["goal"]

    @pytest.mark.asyncio
    async def test_post_problem_accepts_canonical_text_state(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """POST /api/problem also accepts the canonical English state description."""
        response = await async_client.post(
            "/api/problem",
            json={
                "initial_state": "the red block is on the table, the hand is empty",
                "goal": ["the red block is on top of the blue block"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["state"]["stacks"] == [["red"]]
        assert data["state"]["holding"] is None

    @pytest.mark.asyncio
    async def test_post_problem_rejects_inconsistent_state(
        self, async_client: httpx.AsyncClient, controller: SearchController
    ) -> None:
        """POST /api/problem returns 400 when the state is physically impossible."""
        response = await async_client.post(
            "/api/problem",
            json={
                "initial_state": json.dumps(
                    {
                        "clear": ["red"],
                        "on_table": ["red", "blue"],
                        "on": {"red": "blue"},
                        "holding": None,
                    }
                ),
                "goal": ["the red block is on top of the blue block"],
            },
        )

        assert response.status_code == 400
        assert "Invalid initial state" in response.json()["detail"]
        assert controller.iteration == 0
        assert controller.initial_state == "S0"

    @pytest.mark.asyncio
    async def test_post_problem_rejects_empty_goal(self, async_client: httpx.AsyncClient) -> None:
        """POST /api/problem rejects a payload without goal conditions."""
        response = await async_client.post(
            "/api/problem",
            json={"initial_state": "the red block is on the table, the hand is empty", "goal": []},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_set_problem_resets_status_and_iteration(self, controller: SearchController) -> None:
        """`set_problem` stops a running search and rebuilds the root with the new state."""
        await controller.start()
        await asyncio.sleep(0.05)
        assert controller.iteration > 0

        new_state = BlockState(
            clear={"red"},
            on_table={"blue"},
            on={"red": "blue"},
            holding=None,
        )
        result = await controller.set_problem(new_state, ["the red block is on top of the blue block"])

        assert controller.status == SearchStatus.IDLE
        assert controller.iteration == 0
        assert result["iteration"] == 0
        assert controller.initial_state is new_state
        assert controller.root.state is new_state
        assert controller.goal == ["the red block is on top of the blue block"]

    @pytest.mark.asyncio
    async def test_set_problem_propagates_goal_to_agent_and_world_model(self) -> None:
        """`set_problem` must update the Blocksworld agent prompt goal and reward goal."""
        from rap_demo.domains.blocksworld.agent import BlocksworldAgent
        from rap_demo.domains.blocksworld.world_model import BlocksworldWorldModel
        from rap_demo.llm.mock import MockLLM

        llm = MockLLM(default_response="")
        agent = BlocksworldAgent(llm=llm, goal="the red block is on top of the blue block")
        world_model = BlocksworldWorldModel(goal=["the red block is on top of the blue block"], llm=llm)
        mcts = MCTS(world_model=world_model, reasoning_agent=agent, max_depth=3)
        controller = SearchController(mcts=mcts, initial_state="S0", max_iterations=5)

        goal = [
            "the orange block is on top of the red block",
            "the red block is on top of the blue block",
        ]
        await controller.set_problem(
            BlockState(clear={"orange"}, on_table={"red", "blue"}, on={}, holding=None),
            goal,
        )

        assert agent.goal == (
            "the orange block is on top of the red block "
            "and the red block is on top of the blue block"
        )
        assert world_model.goal == goal

