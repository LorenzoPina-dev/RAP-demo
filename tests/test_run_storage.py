"""Comprehensive TDD tests for run persistence, tree deserialization, replay, and CLI history.

Tests cover contracts BEFORE production implementation:
1. Run storage (`rap_demo.storage` / `rap_demo.storage.runs`):
   - `get_runs_dir()` resolution (explicit arg -> RAP_RUNS_DIR -> Path('runs')).
   - `save_run()` directory creation, descriptive sanitized filenames, collision handling,
     and complete JSON payload (metadata, metrics, serialized tree, best_path_ids, best_trajectory).
   - `deserialize_tree()` full MCTSNode reconstruction (N, Q, M with None -> -inf, reward,
     is_terminal, depth, action, thought, parent/children) and BlockState restoration.
   - `list_runs()` newest-first sorting, missing directory handling, corrupt JSON resilience.
   - `load_run()` lookup by 'latest', 1-based index, run_id/prefix, direct Path, and FileNotFoundError.
2. Core & Controller replay (`rap_demo.core.mcts`, `rap_demo.visualizer.web.controller`):
   - Standalone `extract_best_trajectory(root)` without an MCTS instance.
   - `SearchController` initialized with cached `root`, `initial_iteration`, `initial_status=FINISHED`,
     exposing cached `current_state` and `get_best_path_ids()` without `MCTS`.
   - `SearchController(..., on_finish=callback)` firing once upon transition to FINISHED.
3. CLI integration (`rap_demo.cli`):
   - `rap run` auto-saving to `--runs-dir` / `RAP_RUNS_DIR` and skipping save with `--no-save`.
   - `rap history` (and alias `runs`) listing saved runs and handling empty history.
   - `rap replay [ref]` (default 'latest') rendering metadata, `-v` tree, and summary with ZERO LLM calls.
   - `rap replay [ref] --ui --port 9001` launching Web UI in FINISHED state with ZERO LLM calls.
"""

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from rap_demo.cli import build_parser, main
from rap_demo.core.mcts import MCTS, extract_best_trajectory
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState
from rap_demo.storage import (
    deserialize_tree,
    get_runs_dir,
    list_runs,
    load_run,
    save_run,
)
from rap_demo.storage.runs import (
    deserialize_tree as runs_deserialize_tree,
    get_runs_dir as runs_get_runs_dir,
    list_runs as runs_list_runs,
    load_run as runs_load_run,
    save_run as runs_save_run,
)
from rap_demo.visualizer.console import RichMCTSConsole
from rap_demo.visualizer.serializer import serialize_tree
from rap_demo.visualizer.web.app import create_app
from rap_demo.visualizer.web.controller import SearchController, SearchStatus


# ============================================================================
# Shared Helpers & Fixtures
# ============================================================================


def _build_sample_tree() -> tuple[MCTSNode, list[MCTSNode]]:
    """Construct a multi-branch Blocksworld MCTS tree with known statistics and best path."""
    s0 = BlockState(
        clear={"red", "blue", "orange"},
        on_table={"yellow", "red", "blue"},
        on={"orange": "yellow"},
        holding=None,
    )
    root = MCTSNode(state=s0, depth=0, reward=0.0)
    root.N = 5
    root.Q = 25.0
    root.M = 100.0

    # Branch 1 (higher reward path -> best trajectory)
    s1 = BlockState(
        clear={"blue", "orange"},
        on_table={"yellow", "blue"},
        on={"orange": "yellow"},
        holding="red",
    )
    a1 = BlockAction(ActionType.PICK_UP, "red", thought="Pick up red block first")
    n1 = MCTSNode(state=s1, depth=1, reward=0.5, thought="Pick up red block first")
    n1.N = 3
    n1.Q = 50.0
    n1.M = 100.0
    root.add_child(n1, action=a1)

    s2 = BlockState(
        clear={"red", "orange"},
        on_table={"yellow", "blue"},
        on={"red": "blue", "orange": "yellow"},
        holding=None,
    )
    a2 = BlockAction(ActionType.STACK, "red", target="blue", thought="Stack red onto blue")
    n2 = MCTSNode(
        state=s2,
        depth=2,
        reward=100.0,
        is_terminal=True,
        thought="Stack red onto blue",
    )
    n2.N = 2
    n2.Q = 100.0
    n2.M = 100.0
    n1.add_child(n2, action=a2)

    # Branch 2 (unvisited / lower reward branch with M = -inf)
    s_alt = BlockState(
        clear={"red", "blue", "yellow"},
        on_table={"yellow", "red", "blue"},
        on={},
        holding="orange",
    )
    a_alt = BlockAction(ActionType.UNSTACK, "orange", target="yellow")
    n_alt = MCTSNode(state=s_alt, depth=1, reward=0.1, is_terminal=False)
    n_alt.N = 0
    n_alt.Q = 0.0
    n_alt.M = -math.inf
    root.add_child(n_alt, action=a_alt)

    return root, [root, n1, n2]


# ============================================================================
# 1. Storage Layer Tests (rap_demo.storage / rap_demo.storage.runs)
# ============================================================================


class TestRunStorageDirAndExports:
    """Tests package exports and get_runs_dir() precedence."""

    def test_storage_package_exports_match_runs_module(self) -> None:
        """rap_demo.storage re-exports all functions from rap_demo.storage.runs."""
        assert get_runs_dir is runs_get_runs_dir
        assert save_run is runs_save_run
        assert deserialize_tree is runs_deserialize_tree
        assert list_runs is runs_list_runs
        assert load_run is runs_load_run

    def test_get_runs_dir_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_runs_dir resolves explicit arg -> RAP_RUNS_DIR -> Path('runs')."""
        monkeypatch.setenv("RAP_RUNS_DIR", "")
        monkeypatch.delenv("RAP_RUNS_DIR", raising=False)

        assert get_runs_dir() == Path("runs")

        env_dir = tmp_path / "env_runs"
        monkeypatch.setenv("RAP_RUNS_DIR", str(env_dir))
        assert get_runs_dir() == env_dir

        explicit_dir = tmp_path / "explicit_runs"
        assert get_runs_dir(explicit_dir) == explicit_dir
        assert get_runs_dir(str(explicit_dir)) == explicit_dir


class TestSaveRunAndFilenameHandling:
    """Tests save_run() file creation, naming, collision handling, and JSON schema."""

    def test_save_run_creates_dir_and_writes_complete_json(self, tmp_path: Path) -> None:
        """save_run creates runs_dir, writes descriptive filename, and persists full payload."""
        root, best_path = _build_sample_tree()
        target_dir = tmp_path / "nested" / "runs"
        ts = datetime(2026, 11, 10, 12, 30, 45, tzinfo=timezone.utc)

        saved_path = save_run(
            root,
            domain="blocksworld",
            mode="hybrid",
            backend="litellm",
            model="anthropic/claude-3-5-sonnet:latest",
            iterations=10,
            goal=["the red block is on top of the blue block"],
            command="run",
            metrics={"Iterations": 10, "Best Reward": "100.00", "LLM Calls": 7},
            best_path=best_path,
            runs_dir=target_dir,
            created_at=ts,
        )

        assert saved_path.exists()
        assert saved_path.parent == target_dir
        assert saved_path.suffix == ".json"

        name = saved_path.name
        assert "blocksworld" in name
        assert "hybrid" in name
        assert "litellm" in name
        assert "iter10" in name
        # Sanitized model slug (no raw path separators or colons)
        assert "/" not in name
        assert ":" not in name
        assert "claude" in name

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        for key in (
            "run_id",
            "created_at",
            "command",
            "domain",
            "mode",
            "backend",
            "model",
            "iterations",
            "goal",
            "metrics",
            "tree",
            "best_path_ids",
            "best_trajectory",
        ):
            assert key in payload, f"Missing key {key!r} in saved run JSON"

        assert payload["command"] == "run"
        assert payload["domain"] == "blocksworld"
        assert payload["mode"] == "hybrid"
        assert payload["backend"] == "litellm"
        assert payload["model"] == "anthropic/claude-3-5-sonnet:latest"
        assert payload["iterations"] == 10
        assert payload["goal"] == ["the red block is on top of the blue block"]
        assert payload["metrics"]["LLM Calls"] == 7
        assert payload["tree"] == serialize_tree(root)
        assert payload["best_path_ids"] == [f"node_{id(n)}" for n in best_path]
        assert isinstance(payload["best_trajectory"], list)
        assert len(payload["best_trajectory"]) == len(best_path)
        # Validate ISO 8601 timestamp
        parsed_dt = datetime.fromisoformat(payload["created_at"])
        assert parsed_dt.year == 2026

    def test_save_run_handles_timestamp_collisions_without_overwriting(
        self, tmp_path: Path
    ) -> None:
        """Two runs saved with identical timestamps get distinct filenames (e.g. _2)."""
        root, best_path = _build_sample_tree()
        ts = datetime(2026, 11, 10, 14, 0, 0, tzinfo=timezone.utc)

        p1 = save_run(
            root,
            domain="blocksworld",
            mode="hybrid",
            backend="mock",
            iterations=5,
            best_path=best_path,
            runs_dir=tmp_path,
            created_at=ts,
        )
        p2 = save_run(
            root,
            domain="blocksworld",
            mode="hybrid",
            backend="mock",
            iterations=5,
            best_path=best_path,
            runs_dir=tmp_path,
            created_at=ts,
        )

        assert p1 != p2
        assert p1.exists() and p2.exists()
        assert "_2" in p2.stem


class TestDeserializeTree:
    """Tests complete MCTSNode and BlockState reconstruction from serialized JSON."""

    def test_deserialize_tree_preserves_structure_stats_and_blockstates(self) -> None:
        """deserialize_tree restores node stats, M=-inf, parent links, and BlockState fields."""
        orig_root, orig_best = _build_sample_tree()
        tree_json = serialize_tree(orig_root)

        restored = deserialize_tree(tree_json)

        assert isinstance(restored, MCTSNode)
        assert restored.parent is None
        assert restored.depth == 0
        assert restored.N == 5
        assert restored.Q == pytest.approx(25.0)
        assert restored.M == pytest.approx(100.0)
        assert isinstance(restored.state, BlockState)
        assert restored.state == orig_root.state
        assert len(restored.children) == 2

        c1, c2 = restored.children
        assert c1.parent is restored
        assert c1.depth == 1
        assert c1.N == 3
        assert c1.Q == pytest.approx(50.0)
        assert c1.M == pytest.approx(100.0)
        assert c1.reward == pytest.approx(0.5)
        assert c1.thought == "Pick up red block first"
        assert c1.action is not None
        assert isinstance(c1.state, BlockState)
        assert c1.state.holding == "red"
        assert c1.state == orig_best[1].state

        assert len(c1.children) == 1
        leaf = c1.children[0]
        assert leaf.parent is c1
        assert leaf.depth == 2
        assert leaf.is_terminal is True
        assert leaf.reward == pytest.approx(100.0)
        assert isinstance(leaf.state, BlockState)
        assert leaf.state == orig_best[2].state
        assert leaf.state.on == {"red": "blue", "orange": "yellow"}

        # Unvisited branch: M was None in JSON -> restored to -math.inf
        assert c2.parent is restored
        assert c2.N == 0
        assert c2.M == -math.inf
        assert isinstance(c2.state, BlockState)
        assert c2.state.holding == "orange"

    def test_deserialize_tree_preserves_llm_trace(self) -> None:
        """deserialize_tree restores the per-node model prompt/output trace."""
        trace = {
            "source": "model",
            "goal": "the red block is on top of the blue block",
            "model": "ollama/gpt-oss:20b",
            "prompt": "Goal: ...\n\nCurrent State: ...",
            "response": '{"action": "pick_up", "block": "red"}',
            "proposed": {"action": "pick_up", "block": "red"},
            "reason": None,
        }
        node = MCTSNode(
            state=BlockState(clear={"red"}, on_table={"red"}, on={}, holding=None),
            depth=0,
            llm_trace=trace,
        )

        restored = deserialize_tree(serialize_tree(node))

        assert restored.llm_trace == trace

    def test_deserialize_tree_preserves_llm_calls(self) -> None:
        """deserialize_tree restores the full per-step call log used by the inspector."""
        calls = [
            {
                "phase": "action-scoring",
                "action": "Pick up the red block.",
                "prompt": "P",
                "response": "pick up the red block",
                "logprob": -1.0,
                "prob": 0.5,
                "error": None,
            },
            {
                "phase": "world-model",
                "action": "Pick up the red block.",
                "prompt": "W",
                "response": "[CHANGE] ...",
                "error": None,
            },
        ]
        node = MCTSNode(
            state=BlockState(clear={"red"}, on_table={"red"}, on={}, holding=None),
            depth=0,
            llm_calls=calls,
        )

        restored = deserialize_tree(serialize_tree(node))

        assert restored.llm_calls == calls

    def test_deserialized_tree_works_with_trajectory_extraction_and_console(self) -> None:
        """Restored tree works seamlessly with extract_best_trajectory, MCTS, and RichMCTSConsole."""
        orig_root, _ = _build_sample_tree()
        restored = deserialize_tree(serialize_tree(orig_root))

        standalone_best = extract_best_trajectory(restored)
        assert len(standalone_best) == 3
        assert standalone_best[-1].is_terminal is True
        assert standalone_best[-1].reward == pytest.approx(100.0)

        # Also works with MCTS.get_best_trajectory
        mcts_dummy = MCTS.__new__(MCTS)
        mcts_best = mcts_dummy.get_best_trajectory(restored)
        assert [n.depth for n in mcts_best] == [0, 1, 2]

        # Also renders cleanly in RichMCTSConsole without raising
        buf = io.StringIO()
        console = RichMCTSConsole(console=Console(file=buf, force_terminal=False, width=120))
        console.print_tree(restored)
        console.print_summary(standalone_best, metrics={"Iterations": 5})
        output = buf.getvalue()
        assert "Root" in output
        assert "MCTS Trajectory Summary" in output


class TestListAndLoadRuns:
    """Tests list_runs() sorting/resilience and load_run() reference resolution."""

    def test_list_runs_returns_empty_for_missing_directory(self, tmp_path: Path) -> None:
        """list_runs returns [] when runs_dir does not exist."""
        assert list_runs(tmp_path / "does_not_exist") == []

    def test_list_runs_sorts_newest_first_and_ignores_corrupted_files(
        self, tmp_path: Path
    ) -> None:
        """list_runs sorts newest-first by created_at and skips malformed JSON files."""
        root, best_path = _build_sample_tree()
        t_old = datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
        t_mid = t_old + timedelta(hours=1)
        t_new = t_old + timedelta(hours=2)

        p_old = save_run(
            root, domain="blocksworld", mode="paper", backend="mock",
            iterations=3, best_path=best_path, runs_dir=tmp_path, created_at=t_old,
        )
        p_new = save_run(
            root, domain="blocksworld", mode="hybrid", backend="litellm",
            model="gpt-4o-mini", iterations=8, best_path=best_path, runs_dir=tmp_path, created_at=t_new,
        )
        p_mid = save_run(
            root, domain="blocksworld", mode="hybrid", backend="mock",
            iterations=5, best_path=best_path, runs_dir=tmp_path, created_at=t_mid,
        )

        # Add corrupt JSON and unrelated JSON file
        (tmp_path / "corrupted.json").write_text("{not valid json", encoding="utf-8")
        (tmp_path / "unrelated.json").write_text('{"hello": "world"}', encoding="utf-8")

        runs = list_runs(tmp_path)
        assert len(runs) == 3
        assert [Path(r["path"]) for r in runs] == [p_new, p_mid, p_old]
        assert runs[0]["model"] == "gpt-4o-mini"
        assert runs[0]["iterations"] == 8

    def test_load_run_resolves_latest_index_id_prefix_and_path(
        self, tmp_path: Path
    ) -> None:
        """load_run supports 'latest', 1-based index, run_id/prefix, and direct Path."""
        root, best_path = _build_sample_tree()
        t1 = datetime(2026, 11, 10, 9, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 10, 11, 0, 0, tzinfo=timezone.utc)

        p1 = save_run(
            root, domain="blocksworld", mode="paper", backend="mock",
            iterations=4, best_path=best_path, runs_dir=tmp_path, created_at=t1,
        )
        p2 = save_run(
            root, domain="blocksworld", mode="hybrid", backend="mock",
            iterations=9, best_path=best_path, runs_dir=tmp_path, created_at=t2,
        )

        latest = load_run("latest", runs_dir=tmp_path)
        assert latest["iterations"] == 9

        idx1 = load_run("1", runs_dir=tmp_path)
        assert idx1["iterations"] == 9

        idx2 = load_run("2", runs_dir=tmp_path)
        assert idx2["iterations"] == 4

        # Lookup by run_id or prefix
        by_run_id = load_run(idx2["run_id"], runs_dir=tmp_path)
        assert by_run_id["iterations"] == 4

        by_prefix = load_run(p1.stem[:15], runs_dir=tmp_path)
        assert by_prefix["iterations"] == 4

        # Lookup by direct Path
        by_path = load_run(p2)
        assert by_path["iterations"] == 9

    @pytest.mark.parametrize("bad_ref", ["latest", "1", "99", "missing_run_id"])
    def test_load_run_raises_file_not_found_when_missing(
        self, tmp_path: Path, bad_ref: str
    ) -> None:
        """load_run raises FileNotFoundError when directory is empty or ref doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_run(bad_ref, runs_dir=tmp_path)


# ============================================================================
# 2. Core & SearchController Replay Tests
# ============================================================================


class TestTrajectoryExtractionAndControllerReplay:
    """Tests standalone extract_best_trajectory and SearchController replay/on_finish."""

    def test_extract_best_trajectory_standalone(self) -> None:
        """extract_best_trajectory(root) works without an MCTS instance."""
        root, expected_path = _build_sample_tree()
        best = extract_best_trajectory(root)
        assert best == expected_path

        single = MCTSNode(state="only")
        assert extract_best_trajectory(single) == [single]

    def test_search_controller_replay_mode_exposes_cached_tree_without_mcts(self) -> None:
        """SearchController initialized with cached root exposes state, path IDs, and FINISHED status."""
        orig_root, _ = _build_sample_tree()
        cached_root = deserialize_tree(serialize_tree(orig_root))
        expected_best = extract_best_trajectory(cached_root)

        controller = SearchController(
            mcts=None,
            initial_state=cached_root.state,
            max_iterations=10,
            goal=["the red block is on top of the blue block"],
            mode="hybrid",
            root=cached_root,
            initial_iteration=10,
            initial_status=SearchStatus.FINISHED,
        )

        assert controller.root is cached_root
        assert controller.status == SearchStatus.FINISHED
        assert controller.iteration == 10
        assert controller.current_state == expected_best[-1].state
        assert controller.get_best_path_ids() == [f"node_{id(n)}" for n in expected_best]

    async def test_search_controller_on_finish_fires_once_on_completion(self) -> None:
        """on_finish callback is invoked once when step() reaches max_iterations."""
        finished_calls: list[SearchController] = []

        class StubMCTS:
            def step_iteration(self, root: MCTSNode) -> None:
                root.N += 1

            def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
                return [root]

        controller = SearchController(
            mcts=StubMCTS(),  # type: ignore[arg-type]
            initial_state="start",
            max_iterations=2,
            on_finish=lambda ctrl: finished_calls.append(ctrl),
        )

        await controller.step()
        assert controller.status == SearchStatus.IDLE or controller.iteration == 1
        assert len(finished_calls) == 0

        await controller.step()
        assert controller.status == SearchStatus.FINISHED
        assert len(finished_calls) == 1
        assert finished_calls[0] is controller

        # Stepping again while already finished does not re-trigger on_finish
        await controller.step()
        assert len(finished_calls) == 1


# ============================================================================
# 3. CLI Integration Tests (`rap run`, `rap history`, `rap replay`)
# ============================================================================


class TestRunStorageCLIIntegration:
    """Tests CLI run persistence, history listing, and terminal/UI replay."""

    def test_cli_run_auto_saves_and_respects_no_save_and_runs_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rap run automatically saves run JSON to runs_dir unless --no-save is passed."""
        runs_dir = tmp_path / "cli_runs"
        monkeypatch.setenv("RAP_RUNS_DIR", str(runs_dir))

        # 1. Default auto-save via RAP_RUNS_DIR
        rc1 = main(["run", "--backend", "mock", "--iterations", "2"])
        assert rc1 == 0
        saved_files = list(runs_dir.glob("*.json"))
        assert len(saved_files) == 1

        # 2. --no-save skips persistence
        rc2 = main(["run", "--backend", "mock", "--iterations", "2", "--no-save"])
        assert rc2 == 0
        assert len(list(runs_dir.glob("*.json"))) == 1

        # 3. Explicit --runs-dir overrides RAP_RUNS_DIR
        override_dir = tmp_path / "override_runs"
        rc3 = main(
            ["run", "--backend", "mock", "--iterations", "2", "--runs-dir", str(override_dir)]
        )
        assert rc3 == 0
        assert len(list(override_dir.glob("*.json"))) == 1

    def test_cli_history_and_runs_alias(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """rap history and rap runs list saved runs cleanly and handle empty directories."""
        empty_dir = tmp_path / "empty"
        assert main(["history", "--runs-dir", str(empty_dir)]) == 0
        empty_out = (capsys.readouterr().out + "").lower()
        assert "no" in empty_out or "empty" in empty_out or "0" in empty_out

        # Save a run, then list via both `history` and `runs`
        assert main(["run", "--backend", "mock", "--iterations", "2", "--runs-dir", str(tmp_path)]) == 0
        capsys.readouterr()

        assert main(["history", "--runs-dir", str(tmp_path)]) == 0
        hist_out = capsys.readouterr().out
        assert "blocksworld" in hist_out.lower()
        assert "mock" in hist_out.lower()

        assert main(["runs", "--runs-dir", str(tmp_path)]) == 0
        runs_out = capsys.readouterr().out
        assert "blocksworld" in runs_out.lower()

    def test_cli_replay_terminal_uses_zero_llm_calls(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """rap replay [ref] renders metadata, verbose tree, and summary without creating an LLM."""
        assert main(["run", "--backend", "mock", "--iterations", "2", "--runs-dir", str(tmp_path)]) == 0
        capsys.readouterr()

        with patch("rap_demo.cli._create_llm") as mock_create_llm:
            rc = main(["replay", "latest", "--runs-dir", str(tmp_path), "-v"])

        assert rc == 0
        mock_create_llm.assert_not_called()
        out = capsys.readouterr().out
        assert "Root" in out
        assert "MCTS Trajectory Summary" in out
        assert "blocksworld" in out.lower()

    def test_cli_replay_default_latest_and_missing_ref(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """rap replay defaults to 'latest' and returns exit code 1 when run is missing."""
        # Missing run -> exit 1
        with patch("rap_demo.cli._create_llm") as mock_create_llm:
            rc_missing = main(["replay", "nonexistent", "--runs-dir", str(tmp_path)])
        assert rc_missing == 1
        mock_create_llm.assert_not_called()

        # Save run, then `rap replay` with no positional ref defaults to latest
        assert main(["run", "--backend", "mock", "--iterations", "2", "--runs-dir", str(tmp_path)]) == 0
        capsys.readouterr()

        with patch("rap_demo.cli._create_llm") as mock_create_llm:
            rc_default = main(["replay", "--runs-dir", str(tmp_path)])
        assert rc_default == 0
        mock_create_llm.assert_not_called()

    def test_cli_replay_ui_launches_finished_controller_with_zero_llm_calls(
        self, tmp_path: Path
    ) -> None:
        """rap replay <ref> --ui --port 9001 builds FINISHED SearchController and calls uvicorn.run."""
        import asyncio
        from httpx import ASGITransport, AsyncClient

        assert main(["run", "--backend", "mock", "--iterations", "3", "--runs-dir", str(tmp_path)]) == 0

        captured_app: Any = None
        captured_port: int | None = None

        def fake_uvicorn_run(app: Any, **kwargs: Any) -> None:
            nonlocal captured_app, captured_port
            captured_app = app
            captured_port = kwargs.get("port")

        with (
            patch("rap_demo.cli._create_llm") as mock_create_llm,
            patch("uvicorn.run", side_effect=fake_uvicorn_run) as mock_uvicorn,
        ):
            rc = main(["replay", "1", "--ui", "--port", "9001", "--runs-dir", str(tmp_path)])

        assert rc == 0
        mock_create_llm.assert_not_called()
        mock_uvicorn.assert_called_once()
        assert captured_port == 9001
        assert captured_app is not None

        async def verify_cached_ui() -> None:
            transport = ASGITransport(app=captured_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                meta_res = await client.get("/api/meta")
                assert meta_res.status_code == 200
                meta_json = meta_res.json()
                assert meta_json["status"] == SearchStatus.FINISHED.value
                assert meta_json["iteration"] == 3
                assert len(meta_json["best_path_ids"]) >= 1

                state_res = await client.get("/api/state")
                assert state_res.status_code == 200
                state_json = state_res.json()
                assert "stacks" in state_json

                tree_res = await client.get("/api/tree")
                assert tree_res.status_code == 200
                tree_json = tree_res.json()
                assert "children" in tree_json

        asyncio.run(verify_cached_ui())

    def test_parser_supports_run_storage_flags(self) -> None:
        """build_parser() registers --runs-dir, --no-save, history, runs, and replay."""
        parser = build_parser()

        run_args = parser.parse_args(["run", "--no-save", "--runs-dir", "/tmp/myruns"])
        assert run_args.no_save is True
        assert run_args.runs_dir == "/tmp/myruns"

        hist_args = parser.parse_args(["history", "--runs-dir", "/tmp/myruns"])
        assert hist_args.command == "history"
        assert hist_args.runs_dir == "/tmp/myruns"

        runs_args = parser.parse_args(["runs"])
        assert runs_args.command == "runs"

        replay_args = parser.parse_args(["replay", "2", "--ui", "--port", "9001", "-v"])
        assert replay_args.command == "replay"
        assert replay_args.ref == "2"
        assert replay_args.ui is True
        assert replay_args.port == 9001
        assert replay_args.verbose is True
