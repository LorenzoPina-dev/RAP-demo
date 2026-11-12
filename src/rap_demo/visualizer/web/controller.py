"""Search controller managing MCTS search lifecycle, stepping, and SSE broadcasting."""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

from rap_demo.core.events import Event, EventBus, EventType
from rap_demo.core.mcts import MCTS, extract_best_trajectory
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.prompts import format_goal
from rap_demo.visualizer.serializer import _serialize_state, serialize_tree


_PHASE_EVENT_MAP: dict[EventType, str] = {
    EventType.ON_SELECT: "select",
    EventType.ON_EXPAND: "expand",
    EventType.ON_SIMULATE_STEP: "simulate_step",
    EventType.ON_SIMULATE: "simulate",
    EventType.ON_BACKPROP: "backprop",
}


def _find_serialized_node(node_dict: dict[str, Any], target_id: str) -> dict[str, Any] | None:
    """Locate a node dictionary inside a serialized tree by its id."""
    if node_dict.get("id") == target_id:
        return node_dict
    for child in node_dict.get("children", []):
        found = _find_serialized_node(child, target_id)
        if found is not None:
            return found
    return None


def _build_tree_with_sim_preview(
    root: MCTSNode,
    anchor_node: MCTSNode | None,
    sim_path: list[MCTSNode],
) -> dict[str, Any]:
    """Serialize root and attach temporary rollout preview nodes without mutating MCTSNode tree."""
    tree_dict = serialize_tree(root)
    if anchor_node is None or len(sim_path) <= 1:
        return tree_dict

    target = _find_serialized_node(tree_dict, f"node_{id(anchor_node)}")
    if target is None:
        return tree_dict

    parent_dict = target
    for sim_node in sim_path[1:]:
        preview_dict: dict[str, Any] = {
            "id": f"preview_{id(sim_node)}",
            "state": _serialize_state(sim_node.state),
            "action": str(sim_node.action) if sim_node.action is not None else None,
            "thought": getattr(sim_node, "thought", None)
            or (getattr(sim_node.action, "thought", None) if sim_node.action else None),
            "llm_trace": getattr(sim_node, "llm_trace", None),
            "llm_calls": list(getattr(sim_node, "llm_calls", None) or []),
            "visits": sim_node.N,
            "Q": float(sim_node.Q),
            "M": None,
            "reward": float(sim_node.reward),
            "r1": getattr(sim_node, "r1", None),
            "r2": getattr(sim_node, "r2", None),
            "r3": getattr(sim_node, "r3", None),
            "r1_source": (sim_node.llm_trace or {}).get("r1_source")
            if isinstance(getattr(sim_node, "llm_trace", None), dict)
            else None,
            "uct": None,
            "is_terminal": bool(sim_node.is_terminal),
            "depth": int(sim_node.depth),
            "is_preview": True,
            "children": [],
        }
        parent_dict.setdefault("children", []).append(preview_dict)
        parent_dict = preview_dict

    return tree_dict


class SearchStatus(str, Enum):
    """Execution state of the MCTS search controller."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class SearchController:
    """Controls interactive MCTS search and broadcasts real-time events to subscribers."""

    def __init__(
        self,
        mcts: Optional[MCTS] = None,
        initial_state: Optional[Any] = None,
        max_iterations: int = 50,
        step_delay: float = 0.05,
        goal: Optional[Any] = None,
        mode: Optional[str] = None,
        root: Optional[MCTSNode] = None,
        initial_iteration: int = 0,
        initial_status: Optional[SearchStatus] = None,
        on_finish: Optional[Callable[[SearchController], Any]] = None,
        llm: Optional[Any] = None,
        replay_path: Optional[list[MCTSNode]] = None,
        replay_index: int = 0,
        replay_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.mcts: Optional[MCTS] = mcts
        if llm is None and mcts is not None:
            agent = getattr(mcts, "reasoning_agent", None)
            llm = getattr(agent, "llm", None) if agent is not None else None
        self.llm: Optional[Any] = llm
        self.initial_state: Any = initial_state if initial_state is not None else "initial"
        self.max_iterations: int = max_iterations
        self.step_delay: float = step_delay
        self.goal: Optional[Any] = goal
        self.mode: Optional[str] = mode

        self._status: SearchStatus = (
            initial_status if initial_status is not None else SearchStatus.IDLE
        )
        self._iteration: int = initial_iteration
        self._root: MCTSNode = (
            root if root is not None else MCTSNode(state=self.initial_state, depth=0)
        )
        self._on_finish: Optional[Callable[[SearchController], Any]] = on_finish
        self._finish_notified: bool = self._status == SearchStatus.FINISHED
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._running_task: Optional[asyncio.Task[None]] = None
        self._active_loop: Optional[asyncio.AbstractEventLoop] = None
        self._active_iteration: int = 0

        # Replay mode: walk a cached trajectory without invoking the LLM or MCTS.
        self._replay_path: Optional[list[MCTSNode]] = (
            list(replay_path) if replay_path else None
        )
        self._replay_meta: dict[str, Any] = dict(replay_meta or {})
        self._live_snapshot: Optional[dict[str, Any]] = None
        self._playing: bool = False

        if self._replay_path is not None:
            last_index = max(len(self._replay_path) - 1, 0)
            self._iteration = max(0, min(int(replay_index), last_index))
            self.max_iterations = last_index
            self._status = (
                SearchStatus.FINISHED
                if self._iteration >= self.max_iterations
                else (SearchStatus.IDLE if self._iteration == 0 else SearchStatus.PAUSED)
            )
            self._finish_notified = True

        if self.mcts is not None:
            if getattr(self.mcts, "event_bus", None) is None:
                self.mcts.event_bus = EventBus()
            self._attach_event_bus(self.mcts.event_bus)

    def _enqueue_nowait(self, payload: dict[str, Any]) -> None:
        """Enqueue an event payload onto all subscriber queues on the event loop thread."""
        for queue in list(self._subscribers):
            queue.put_nowait(payload)

    def _dispatch_event_threadsafe(self, payload: dict[str, Any]) -> None:
        """Broadcast an SSE payload safely from either the event loop or a worker thread."""
        loop = self._active_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._enqueue_nowait, payload)
        else:
            self._enqueue_nowait(payload)

    def _on_mcts_phase_event(self, event: Event) -> None:
        """Forward MCTS lifecycle events as real-time SSE 'phase' events."""
        phase = _PHASE_EVENT_MAP.get(event.type)
        if phase is None:
            return
        iteration = self._active_iteration or (self._iteration + 1)
        payload: dict[str, Any] = {
            "type": "phase",
            "phase": phase,
            "iteration": iteration,
        }
        if event.type in (EventType.ON_EXPAND, EventType.ON_BACKPROP):
            payload["tree"] = serialize_tree(self._root)
            payload["best_path_ids"] = self.get_best_path_ids()
        elif event.type == EventType.ON_SIMULATE_STEP:
            anchor = event.data.get("node")
            sim_path = event.data.get("sim_path") or []
            current_node = event.data.get("current_node")
            payload["tree"] = _build_tree_with_sim_preview(self._root, anchor, sim_path)
            payload["best_path_ids"] = self.get_best_path_ids()
            if current_node is not None:
                payload["preview_node_id"] = f"preview_{id(current_node)}"
                payload["preview_state"] = _serialize_state(current_node.state)
        self._dispatch_event_threadsafe(payload)

    def _attach_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe to MCTS lifecycle phase events on the given EventBus."""
        for ev_type in _PHASE_EVENT_MAP:
            event_bus.subscribe(ev_type, self._on_mcts_phase_event)

    def set_step_delay(self, delay_seconds: float) -> float:
        """Updates step delay (clamped to [0.01s, 2.0s]) for live playback control."""
        clamped = max(0.01, min(2.0, float(delay_seconds)))
        self.step_delay = clamped
        return clamped

    def _get_best_trajectory(self) -> list[MCTSNode]:
        """Return best trajectory via MCTS engine or standalone extractor."""
        if self.mcts is not None:
            traj = self.mcts.get_best_trajectory(self._root)
        else:
            traj = extract_best_trajectory(self._root)
        return traj if traj else [self._root]

    async def _notify_finish_once(self) -> None:
        """Invoke on_finish callback at most once per search completion."""
        if self._finish_notified:
            return
        self._finish_notified = True
        if self._on_finish is not None:
            result = self._on_finish(self)
            if inspect.isawaitable(result):
                await result

    def get_best_path_ids(self) -> list[str]:
        """Returns serialized node IDs along the current best trajectory."""
        if self._replay_path is not None:
            index = min(self._iteration, len(self._replay_path) - 1)
            return [f"node_{id(n)}" for n in self._replay_path[: index + 1]]
        traj = self._get_best_trajectory()
        return [f"node_{id(n)}" for n in traj]

    @property
    def status(self) -> SearchStatus:
        """Current status of the search execution."""
        return self._status

    @property
    def is_replay(self) -> bool:
        """True when the controller is walking a cached trajectory instead of searching."""
        return self._replay_path is not None

    @property
    def can_exit_replay(self) -> bool:
        """True when a live search state is available to restore after replay."""
        return self._live_snapshot is not None

    @property
    def replay_meta(self) -> dict[str, Any]:
        """Summary metadata of the run currently loaded in replay mode."""
        return dict(self._replay_meta)

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration

    @property
    def root(self) -> MCTSNode:
        """Root node of the MCTS tree."""
        return self._root

    @property
    def current_state(self) -> Any:
        """Current state at the most promising node of best trajectory."""
        if self._replay_path is not None:
            index = min(self._iteration, len(self._replay_path) - 1)
            return self._replay_path[index].state
        return self._get_best_trajectory()[-1].state

    async def step(self) -> dict[str, Any]:
        """Runs exactly one MCTS iteration and emits events."""
        if self._replay_path is not None:
            return await self._replay_step()

        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
            await self._notify_finish_once()
            return {
                "status": self._status,
                "iteration": self._iteration,
                "root": self._root,
            }

        if self.mcts is not None:
            next_iter = self._iteration + 1
            t0 = time.perf_counter()
            print(
                f"[RAP] Starting MCTS iteration {next_iter}/{self.max_iterations}...",
                file=sys.stderr,
                flush=True,
            )
            self._active_loop = asyncio.get_running_loop()
            self._active_iteration = next_iter
            has_on_token = self.llm is not None and hasattr(self.llm, "on_token")
            prev_on_token = getattr(self.llm, "on_token", None) if has_on_token else None
            if has_on_token:
                def _stream_token(kind: str, text: str) -> None:
                    self._dispatch_event_threadsafe(
                        {
                            "type": "token",
                            "kind": kind,
                            "text": text,
                            "call_id": getattr(self.llm, "call_count", 0),
                            "iteration": next_iter,
                        }
                    )

                self.llm.on_token = _stream_token

            try:
                await asyncio.to_thread(self.mcts.step_iteration, self._root)
                # Yield once so any call_soon_threadsafe phase/token callbacks queued by
                # the worker thread flush onto subscriber queues before emitting 'step'.
                await asyncio.sleep(0)
            finally:
                if has_on_token:
                    self.llm.on_token = prev_on_token
                self._active_loop = None
                self._active_iteration = 0

            elapsed = time.perf_counter() - t0
            print(
                f"[RAP] Completed MCTS iteration {next_iter}/{self.max_iterations} in {elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )

        self._iteration += 1

        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED

        best_traj = self._get_best_trajectory()
        best_node = best_traj[-1]

        payload = {
            "type": "step",
            "iteration": self._iteration,
            "max_iterations": self.max_iterations,
            "status": self._status.value,
            "tree": serialize_tree(self._root),
            "best_node_id": f"node_{id(best_node)}",
            "best_path_ids": [f"node_{id(n)}" for n in best_traj],
            "thought": getattr(best_node, "thought", None) or (getattr(best_node.action, "thought", None) if best_node.action else None),
        }
        await self.emit_event("step", payload)

        if self._status == SearchStatus.FINISHED:
            await self._notify_finish_once()

        return {
            "status": self._status,
            "iteration": self._iteration,
            "root": self._root,
        }

    async def _replay_step(self) -> dict[str, Any]:
        """Advances the replay cursor along the cached trajectory without any LLM call."""
        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
            return {
                "status": self._status,
                "iteration": self._iteration,
                "root": self._root,
            }

        self._iteration += 1
        node = self._replay_path[self._iteration]
        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
        elif self._playing:
            self._status = SearchStatus.RUNNING
        else:
            self._status = SearchStatus.PAUSED

        thought = getattr(node, "thought", None) or (
            getattr(node.action, "thought", None) if node.action else None
        )
        payload = {
            "type": "step",
            "iteration": self._iteration,
            "max_iterations": self.max_iterations,
            "status": self._status.value,
            "tree": serialize_tree(self._root),
            "state": _serialize_state(node.state),
            "best_node_id": f"node_{id(node)}",
            "best_path_ids": self.get_best_path_ids(),
            "thought": thought,
            "replay": True,
            "run": dict(self._replay_meta),
        }
        await self.emit_event("step", payload)

        if self._status == SearchStatus.FINISHED:
            await self._notify_finish_once()

        return {
            "status": self._status,
            "iteration": self._iteration,
            "root": self._root,
        }

    async def start(self) -> dict[str, Any]:
        """Starts continuous search execution in background."""
        if self._status == SearchStatus.RUNNING:
            return {"status": self._status, "iteration": self._iteration}

        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
            return {"status": self._status, "iteration": self._iteration}

        self._status = SearchStatus.RUNNING
        self._playing = True
        self._running_task = asyncio.create_task(self._run_loop())
        await self.emit_event(
            "start",
            {"type": "start", "status": self._status.value, "iteration": self._iteration},
        )
        return {"status": self._status, "iteration": self._iteration}

    async def _run_loop(self) -> None:
        """Internal asynchronous loop advancing iterations while RUNNING."""
        try:
            while self._status == SearchStatus.RUNNING and self._iteration < self.max_iterations:
                await self.step()
                if self._status != SearchStatus.RUNNING or self._iteration >= self.max_iterations:
                    break
                await asyncio.sleep(self.step_delay)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[RAP UI Error] {exc}", file=sys.stderr, flush=True)
            self._status = SearchStatus.PAUSED
            await self.emit_event(
                "error",
                {
                    "type": "error",
                    "status": self._status.value,
                    "iteration": self._iteration,
                    "message": str(exc),
                },
            )
        finally:
            self._playing = False
            if self._iteration >= self.max_iterations:
                self._status = SearchStatus.FINISHED

    async def pause(self) -> dict[str, Any]:
        """Pauses running search execution."""
        if self._status == SearchStatus.RUNNING:
            self._playing = False
            self._status = SearchStatus.PAUSED
            if self._running_task and not self._running_task.done():
                self._running_task.cancel()
                self._running_task = None
            await self.emit_event(
                "pause",
                {"type": "pause", "status": self._status.value, "iteration": self._iteration},
            )

        return {"status": self._status, "iteration": self._iteration}

    async def resume(self) -> dict[str, Any]:
        """Resumes paused search execution."""
        if self._status in (SearchStatus.PAUSED, SearchStatus.IDLE) and self._iteration < self.max_iterations:
            return await self.start()
        return {"status": self._status, "iteration": self._iteration}

    async def reset(self) -> dict[str, Any]:
        """Resets controller state, tree, and iteration counter to initial state."""
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            self._running_task = None

        self._playing = False
        self._status = SearchStatus.IDLE
        self._iteration = 0

        if self._replay_path is not None:
            # Replay reset rewinds the cursor and keeps the cached tree intact.
            await self.emit_event("reset", self._reset_payload())
            return {"status": self._status, "iteration": self._iteration}

        self._finish_notified = False
        self._root = MCTSNode(state=self.initial_state, depth=0)

        await self.emit_event("reset", self._reset_payload())

        return {"status": self._status, "iteration": self._iteration}

    def _reset_payload(self) -> dict[str, Any]:
        """Builds the SSE payload shared by live and replay resets."""
        return {
            "type": "reset",
            "iteration": self._iteration,
            "max_iterations": self.max_iterations,
            "status": self._status.value,
            "goal": self.goal,
            "mode": self.mode,
            "tree": serialize_tree(self._root),
            "state": _serialize_state(self.current_state),
            "best_path_ids": self.get_best_path_ids(),
            "replay": self._replay_path is not None,
            "run": dict(self._replay_meta),
        }

    async def enter_replay(
        self,
        root: MCTSNode,
        replay_path: list[MCTSNode],
        goal: Any = None,
        mode: Optional[str] = None,
        run_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Switches to replay mode using a cached tree; no LLM or MCTS calls are made."""
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            self._running_task = None
        self._playing = False

        if self._replay_path is None:
            self._live_snapshot = {
                "mcts": self.mcts,
                "llm": self.llm,
                "root": self._root,
                "initial_state": self.initial_state,
                "goal": self.goal,
                "mode": self.mode,
                "iteration": self._iteration,
                "max_iterations": self.max_iterations,
                "status": self._status,
            }

        self.mcts = None
        self.llm = None
        self._root = root
        self._replay_path = list(replay_path) if replay_path else [root]
        self._replay_meta = dict(run_meta or {})
        self.initial_state = root.state
        self.goal = goal
        self.mode = mode
        self._iteration = 0
        self.max_iterations = max(len(self._replay_path) - 1, 0)
        self._status = SearchStatus.IDLE
        self._finish_notified = True

        await self.emit_event("reset", self._reset_payload())
        return {"status": self._status, "iteration": self._iteration}

    async def exit_replay(self) -> dict[str, Any]:
        """Restores the live search state captured when replay mode was entered."""
        if self._replay_path is None:
            return {"status": self._status, "iteration": self._iteration}

        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            self._running_task = None
        self._playing = False

        snapshot = self._live_snapshot
        self._live_snapshot = None
        self._replay_path = None
        self._replay_meta = {}

        if snapshot is not None:
            self.mcts = snapshot["mcts"]
            self.llm = snapshot["llm"]
            self._root = snapshot["root"]
            self.initial_state = snapshot["initial_state"]
            self.goal = snapshot["goal"]
            self.mode = snapshot["mode"]
            self._iteration = snapshot["iteration"]
            self.max_iterations = snapshot["max_iterations"]
            self._status = (
                SearchStatus.PAUSED
                if snapshot["status"] == SearchStatus.RUNNING
                else snapshot["status"]
            )
            self._finish_notified = snapshot["status"] == SearchStatus.FINISHED
        else:
            self.mcts = None
            self.llm = None
            self._root = MCTSNode(state=self.initial_state, depth=0)
            self._iteration = 0
            self.max_iterations = 0
            self._status = SearchStatus.IDLE
            self._finish_notified = False

        await self.emit_event("reset", self._reset_payload())
        return {"status": self._status, "iteration": self._iteration}

    async def set_problem(self, initial_state: Any, goal: Any) -> dict[str, Any]:
        """Replaces the problem definition (initial state and goal) and resets the search."""
        if self._replay_path is not None:
            await self.exit_replay()
        self.initial_state = initial_state
        self.goal = goal
        self._apply_goal_to_components(goal)
        return await self.reset()

    def _apply_goal_to_components(self, goal: Any) -> None:
        """Propagates a new goal to the MCTS reasoning agent and world model, when supported."""
        if self.mcts is None:
            return

        goal_text = format_goal(goal)
        agent = getattr(self.mcts, "reasoning_agent", None)
        world_model = getattr(self.mcts, "world_model", None)

        if agent is not None and hasattr(agent, "goal"):
            agent.goal = goal_text
        if world_model is not None and hasattr(world_model, "goal"):
            world_model.goal = goal

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribes an event queue for SSE streaming."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribes an event queue."""
        self._subscribers.discard(queue)

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcasts event payload to all active subscriber queues."""
        data.setdefault("type", event_type)
        for queue in list(self._subscribers):
            await queue.put(data)
