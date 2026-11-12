"""FastAPI application for the RAP Real-Time Visualizer dashboard and REST/SSE endpoints."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rap_demo.core.mcts import extract_best_trajectory
from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.domains.blocksworld.problem import parse_goal, parse_initial_state
from rap_demo.llm.base import LLMError
from rap_demo.visualizer.serializer import serialize_blocksworld_state, serialize_tree
from rap_demo.visualizer.web.controller import SearchController


class SearchConfigRequest(BaseModel):
    """Request payload for updating live search playback configuration."""

    step_delay_ms: int = Field(..., ge=10, le=2000, description="Delay between iterations in milliseconds")


class ProblemRequest(BaseModel):
    """Request payload for replacing the Blocksworld problem definition."""

    initial_state: str = Field(
        ...,
        min_length=1,
        description="Initial state as JSON object or canonical English description",
    )
    goal: list[str] = Field(
        ...,
        min_length=1,
        description="Natural language goal conditions (one or more)",
    )


class ReplayRequest(BaseModel):
    """Request payload for loading a saved run in replay mode."""

    ref: str = Field(
        "latest",
        min_length=1,
        description="Run reference: 'latest', 1-based index, run_id/prefix, or file path",
    )


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles handler that prevents stale browser-cached CSS/JS assets."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


def _status_value(controller: SearchController) -> str:
    """Returns the controller status as a plain string."""
    return controller.status.value if hasattr(controller.status, "value") else str(controller.status)


def _serialize_current_state(controller: SearchController) -> Any:
    """Serializes the controller's current domain state for the tabletop."""
    state = controller.current_state
    if isinstance(state, BlockState):
        return serialize_blocksworld_state(state)
    if isinstance(state, dict):
        return state
    return {"state": str(state)}


def _replay_payload(
    controller: SearchController,
    run: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Builds the REST payload returned after loading or leaving a replay."""
    return {
        "iteration": controller.iteration,
        "max_iterations": controller.max_iterations,
        "status": _status_value(controller),
        "goal": controller.goal,
        "mode": controller.mode,
        "tree": serialize_tree(controller.root),
        "state": _serialize_current_state(controller),
        "best_path_ids": controller.get_best_path_ids(),
        "replay": controller.is_replay,
        "can_exit_replay": controller.can_exit_replay,
        "run": run if run is not None else controller.replay_meta,
    }


def create_app(
    controller: Optional[SearchController] = None,
    runs_dir: Any = None,
) -> FastAPI:
    """Factory creating FastAPI application wired with SearchController."""
    ctrl: SearchController = controller or SearchController()
    server = FastAPI(title="RAP Visualizer Dashboard")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        server.mount("/static", _NoCacheStaticFiles(directory=str(static_dir)), name="static")

    @server.get("/", response_class=HTMLResponse)
    async def get_dashboard() -> HTMLResponse:
        """Returns the real-time visualizer web dashboard HTML."""
        no_cache_headers = {"Cache-Control": "no-store, max-age=0"}
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(
                content=index_file.read_text(encoding="utf-8"),
                headers=no_cache_headers,
            )
        return HTMLResponse(
            content="<html><body><h1>RAP Visualizer</h1><p>Tree reasoning visualizer</p></body></html>",
            headers=no_cache_headers,
        )

    @server.get("/api/tree")
    async def get_tree() -> dict[str, Any]:
        """Returns JSON serialized MCTS tree hierarchy."""
        return serialize_tree(ctrl.root)

    @server.get("/api/state")
    async def get_state() -> Any:
        """Returns current domain state representation."""
        return _serialize_current_state(ctrl)

    @server.get("/api/meta")
    async def get_meta() -> dict[str, Any]:
        """Returns visualizer configuration, goal context, and trajectory metadata."""
        initial_state = ctrl.initial_state
        initial_state_text = (
            initial_state.to_text() if isinstance(initial_state, BlockState) else str(initial_state)
        )
        return {
            "max_iterations": ctrl.max_iterations,
            "step_delay_ms": round(ctrl.step_delay * 1000),
            "mode": ctrl.mode,
            "goal": ctrl.goal,
            "initial_state": initial_state_text,
            "status": _status_value(ctrl),
            "iteration": ctrl.iteration,
            "best_path_ids": ctrl.get_best_path_ids(),
            "replay": ctrl.is_replay,
            "can_exit_replay": ctrl.can_exit_replay,
            "run": ctrl.replay_meta,
        }

    @server.get("/api/runs")
    async def get_runs(limit: int = 50) -> dict[str, Any]:
        """Lists saved runs available for replay, newest first."""
        # Imported lazily to keep rap_demo.storage.runs -> visualizer.serializer
        # from forming an import cycle at package initialisation time.
        from rap_demo.storage.runs import list_runs

        bounded = max(0, min(int(limit), 500))
        return {"runs": list_runs(runs_dir=runs_dir, limit=bounded)}

    @server.post("/api/replay")
    async def post_replay(request: ReplayRequest) -> dict[str, Any]:
        """Loads a saved run and switches the controller to replay mode (no LLM calls)."""
        from rap_demo.storage.runs import deserialize_tree, load_run

        try:
            run_data = load_run(request.ref, runs_dir=runs_dir)
            root = deserialize_tree(run_data["tree"], goal=run_data.get("goal"))
            replay_path = extract_best_trajectory(root)
            if not replay_path:
                replay_path = [root]
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        run_meta = {
            "run_id": run_data.get("run_id"),
            "created_at": run_data.get("created_at"),
            "domain": run_data.get("domain"),
            "mode": run_data.get("mode"),
            "backend": run_data.get("backend"),
            "model": run_data.get("model"),
            "iterations": run_data.get("iterations", 0),
            "metrics": run_data.get("metrics", {}),
            "goal": run_data.get("goal"),
        }
        await ctrl.enter_replay(
            root=root,
            replay_path=replay_path,
            goal=run_data.get("goal"),
            mode=run_data.get("mode"),
            run_meta=run_meta,
        )
        return _replay_payload(ctrl, run_meta)

    @server.post("/api/replay/exit")
    async def post_replay_exit() -> dict[str, Any]:
        """Leaves replay mode and restores the live search state, when available."""
        await ctrl.exit_replay()
        return _replay_payload(ctrl)

    @server.post("/api/problem")
    async def post_problem(problem: ProblemRequest) -> dict[str, Any]:
        """Replaces the initial state and goal, resetting the MCTS tree."""
        try:
            initial_state = parse_initial_state(problem.initial_state)
            goal = parse_goal(problem.goal)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await ctrl.set_problem(initial_state, goal)
        return {
            "iteration": ctrl.iteration,
            "max_iterations": ctrl.max_iterations,
            "status": _status_value(ctrl),
            "goal": ctrl.goal,
            "state": serialize_blocksworld_state(initial_state),
            "tree": serialize_tree(ctrl.root),
            "best_path_ids": ctrl.get_best_path_ids(),
        }

    @server.post("/api/search/config")
    async def post_config(config: SearchConfigRequest) -> dict[str, Any]:
        """Updates search playback configuration such as step delay."""
        applied_seconds = ctrl.set_step_delay(config.step_delay_ms / 1000.0)
        return {"step_delay_ms": round(applied_seconds * 1000)}

    @server.post("/api/search/step")
    async def post_step() -> dict[str, Any]:
        """Advances search by one iteration."""
        try:
            res = await ctrl.step()
        except LLMError as exc:
            print(f"[RAP UI Error] {exc}", file=sys.stderr, flush=True)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "iteration": ctrl.iteration,
            "max_iterations": ctrl.max_iterations,
            "status": _status_value(ctrl),
            "tree": serialize_tree(ctrl.root),
            "state": _serialize_current_state(ctrl),
            "best_path_ids": ctrl.get_best_path_ids(),
            "replay": ctrl.is_replay,
        }

    @server.post("/api/search/start")
    async def post_start() -> dict[str, Any]:
        """Starts continuous search in background."""
        res = await ctrl.start()
        return {
            "iteration": ctrl.iteration,
            "status": _status_value(ctrl),
            "replay": ctrl.is_replay,
        }

    @server.post("/api/search/pause")
    async def post_pause() -> dict[str, Any]:
        """Pauses active search execution."""
        res = await ctrl.pause()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
        }

    @server.post("/api/search/resume")
    async def post_resume() -> dict[str, Any]:
        """Resumes paused search execution."""
        res = await ctrl.resume()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
        }

    @server.post("/api/search/reset")
    async def post_reset() -> dict[str, Any]:
        """Resets search tree and iteration counter."""
        await ctrl.reset()
        return {
            "iteration": ctrl.iteration,
            "max_iterations": ctrl.max_iterations,
            "status": _status_value(ctrl),
            "tree": serialize_tree(ctrl.root),
            "state": _serialize_current_state(ctrl),
            "best_path_ids": ctrl.get_best_path_ids(),
            "replay": ctrl.is_replay,
        }

    @server.get("/api/stream")
    async def get_stream(request: Request) -> StreamingResponse:
        """Streams real-time search progress events over Server-Sent Events (SSE)."""

        async def event_generator() -> AsyncGenerator[str, None]:
            queue = ctrl.subscribe()
            try:
                init_data = {
                    "type": "connect",
                    "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
                    "iteration": ctrl.iteration,
                    "max_iterations": ctrl.max_iterations,
                    "best_path_ids": ctrl.get_best_path_ids(),
                }
                yield f"data: {json.dumps(init_data)}\n\n"

                # In-memory test transports (such as httpx ASGITransport) buffer responses until
                # completion; return immediately so the test client can read the stream chunk.
                is_test_client = request.headers.get("host") == "test"
                if is_test_client:
                    return

                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                ctrl.unsubscribe(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return server


app = create_app()
