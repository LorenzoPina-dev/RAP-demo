"""Comprehensive unit and integration tests for Phase 6: CLI & Full System Integration.

Contracts tested:
1. `rap_demo.cli.build_parser`:
   - Configures top-level ArgumentParser with subcommands: `run`, `ui`, `compare`.
   - Subcommand `run`:
     - `--domain`: domain identifier (default: "blocksworld").
     - `--mode`: reasoning mode ("hybrid" | "paper", default: "hybrid").
     - `--backend`: LLM backend identifier ("mock" | "llama-server", default: "mock").
     - `--iterations`: MCTS search cycles (default: 10, type: int).
     - `--verbose` / `-v`: detailed terminal output (flag, default: False).
     - `--initial-state`: initial Blocksworld state as JSON or canonical English (default: None).
     - `--goal`: repeatable goal condition (default: None).
     - `--max-depth`: MCTS depth limit L (default: 6, type: int).
   - Subcommand `ui`:
     - `--host`: visualizer bind address (default: "127.0.0.1").
     - `--port`: visualizer port (default: 8000, type: int).
     - `--backend`: LLM backend identifier (default: "mock").
     - `--mode`: reasoning mode (default: "hybrid").
     - `--iterations`: maximum MCTS iterations (default: 50, type: int).
     - `--domain`: domain identifier (default: "blocksworld").
   - Subcommand `compare`:
     - `--domain`: domain identifier (default: "blocksworld").
     - `--backend`: LLM backend identifier (default: "mock").
     - `--iterations`: MCTS search cycles per mode (default: 10, type: int).
     - `--verbose` / `-v`: detailed output (flag, default: False).

2. `rap_demo.cli.main`:
   - CLI entrypoint supporting both explicit `args` sequence and `sys.argv[1:]` fallback.
   - Subcommand dispatch to respective command handlers.
   - Proper exit codes:
     - 0 on success.
     - Non-zero (1 or 2) on argument parsing errors, validation failures, or runtime exceptions.
   - Clean handling of `--help` / `-h` without unhandled exceptions.

3. Execution contracts & Mocking:
   - `rap run`: runs MCTS search in terminal, formats output via RichMCTSConsole, respects verbose mode.
   - `rap ui`: creates SearchController and FastAPI app, invokes `uvicorn.run` with specified host and port.
     Uvicorn is mocked in tests to prevent blocking network servers.
   - `rap compare`: executes Hybrid and Paper modes side-by-side, extracts metrics, renders comparative summary table.
   - Comprehensive error recovery for backend connection errors, server bind errors, and domain validation failures.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any, Generator
from unittest.mock import MagicMock, call, patch
import pytest
from rich.console import Console

from rap_demo.cli import (
    DEFAULT_GOAL,
    DEFAULT_INITIAL_STATE,
    _parse_goal,
    _parse_initial_state,
    build_parser,
    main,
)
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.llm.base import BaseLLM
from rap_demo.llm.mock import MockLLM


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    """Fixture providing configured CLI argument parser."""
    return build_parser()


@pytest.fixture
def mock_uvicorn_run() -> Generator[MagicMock, None, None]:
    """Mocks uvicorn.run to prevent launching a real web server during tests."""
    with patch("uvicorn.run") as mock_run:
        yield mock_run


@pytest.fixture
def string_console() -> Console:
    """Provides a Rich console configured to write to a string buffer for inspection."""
    return Console(file=io.StringIO(), force_terminal=False, width=120)


# ============================================================================
# 1. CLI Argument Parser Tests
# ============================================================================


class TestCLIParser:
    """Unit tests for `rap_demo.cli.build_parser` flag definitions, defaults, and choices."""

    def test_parser_creation(self, parser: argparse.ArgumentParser) -> None:
        """Parser is created with subcommands for run, ui, and compare."""
        assert isinstance(parser, argparse.ArgumentParser)
        # Verify subparser actions exist
        subparsers_actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        assert len(subparsers_actions) == 1
        subparser_choices = subparsers_actions[0].choices
        assert "run" in subparser_choices
        assert "ui" in subparser_choices
        assert "compare" in subparser_choices

    # --- `run` subcommand arguments ---

    def test_run_parser_defaults(self, parser: argparse.ArgumentParser) -> None:
        """`rap run` has correct default arguments."""
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.domain == "blocksworld"
        assert args.mode == "paper"
        assert args.backend == "mock"
        assert args.iterations == 10
        assert args.max_depth == 6
        assert args.verbose is False

    def test_run_parser_custom_values(self, parser: argparse.ArgumentParser) -> None:
        """`rap run` correctly parses all custom flags and short options."""
        args = parser.parse_args(
            [
                "run",
                "--domain",
                "blocksworld",
                "--mode",
                "paper",
                "--backend",
                "llama-server",
                "--iterations",
                "25",
                "--max-depth",
                "9",
                "--verbose",
            ]
        )
        assert args.domain == "blocksworld"
        assert args.mode == "paper"
        assert args.backend == "llama-server"
        assert args.iterations == 25
        assert args.max_depth == 9
        assert args.verbose is True

    def test_run_parser_short_verbose_flag(self, parser: argparse.ArgumentParser) -> None:
        """`rap run -v` enables verbose mode."""
        args = parser.parse_args(["run", "-v"])
        assert args.verbose is True

    @pytest.mark.parametrize("invalid_mode", ["magic", "unknown", "tree", ""])
    def test_run_parser_rejects_invalid_mode(
        self, parser: argparse.ArgumentParser, invalid_mode: str
    ) -> None:
        """`rap run` rejects modes other than 'hybrid' and 'paper'."""
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--mode", invalid_mode])

    @pytest.mark.parametrize("invalid_backend", ["openai", "claude", "gpt4", "xyz"])
    def test_run_parser_rejects_invalid_backend(
        self, parser: argparse.ArgumentParser, invalid_backend: str
    ) -> None:
        """`rap run` rejects unsupported backend names."""
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--backend", invalid_backend])

    def test_run_parser_rejects_non_integer_iterations(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """`rap run` rejects non-integer iterations."""
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--iterations", "ten"])

    def test_run_parser_problem_flags(self, parser: argparse.ArgumentParser) -> None:
        """`rap run` accepts a custom initial state and repeatable goal conditions."""
        args = parser.parse_args(
            [
                "run",
                "--initial-state",
                '{"clear": ["red"], "on_table": ["blue"]}',
                "--goal",
                "the red block is on top of the blue block",
                "--goal",
                "the blue block is on the table",
            ]
        )
        assert args.initial_state == '{"clear": ["red"], "on_table": ["blue"]}'
        assert args.goal == [
            "the red block is on top of the blue block",
            "the blue block is on the table",
        ]

    def test_problem_flags_default_to_none(self, parser: argparse.ArgumentParser) -> None:
        """Problem flags default to None so the built-in demo problem is used."""
        for subcommand in ("run", "ui", "compare"):
            args = parser.parse_args([subcommand])
            assert args.initial_state is None
            assert args.goal is None
            assert args.max_depth == 6

    # --- `ui` subcommand arguments ---

    def test_ui_parser_defaults(self, parser: argparse.ArgumentParser) -> None:
        """`rap ui` has sensible default server host, port, mode, and backend."""
        args = parser.parse_args(["ui"])
        assert args.command == "ui"
        assert args.host in ("127.0.0.1", "0.0.0.0", "localhost")
        assert args.port == 8000
        assert args.backend == "mock"
        assert args.mode == "paper"
        assert args.iterations >= 10
        assert args.domain == "blocksworld"

    def test_ui_parser_custom_values(self, parser: argparse.ArgumentParser) -> None:
        """`rap ui` correctly parses custom host, port, mode, iterations, and backend."""
        args = parser.parse_args(
            [
                "ui",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--backend",
                "llama-server",
                "--mode",
                "paper",
                "--iterations",
                "30",
            ]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.backend == "llama-server"
        assert args.mode == "paper"
        assert args.iterations == 30

    def test_ui_parser_rejects_non_integer_port(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """`rap ui` rejects non-integer port values."""
        with pytest.raises(SystemExit):
            parser.parse_args(["ui", "--port", "http"])

    # --- `compare` subcommand arguments ---

    def test_compare_parser_defaults(self, parser: argparse.ArgumentParser) -> None:
        """`rap compare` has correct defaults."""
        args = parser.parse_args(["compare"])
        assert args.command == "compare"
        assert args.domain == "blocksworld"
        assert args.backend == "mock"
        assert args.iterations == 10
        assert args.verbose is False

    def test_compare_parser_custom_values(self, parser: argparse.ArgumentParser) -> None:
        """`rap compare` parses custom iterations, backend, and verbose flag."""
        args = parser.parse_args(
            [
                "compare",
                "--domain",
                "blocksworld",
                "--backend",
                "llama-server",
                "--iterations",
                "20",
                "-v",
            ]
        )
        assert args.domain == "blocksworld"
        assert args.backend == "llama-server"
        assert args.iterations == 20
        assert args.verbose is True


# ============================================================================
# 2. Main Entrypoint & Dispatch Tests
# ============================================================================


class TestMainEntrypoint:
    """Unit tests for `rap_demo.cli.main` dispatching, sys.argv handling, and exit codes."""

    def test_main_no_arguments_returns_non_zero_exit_code(self) -> None:
        """Invoking `main([])` without subcommands returns a non-zero exit code."""
        exit_code = main([])
        assert exit_code != 0

    def test_main_invalid_subcommand_returns_non_zero_exit_code(self) -> None:
        """Invoking `main(["invalid_cmd"])` returns a non-zero exit code."""
        exit_code = main(["invalid_cmd"])
        assert exit_code != 0

    def test_main_help_flag_returns_zero(self) -> None:
        """Invoking `main(["--help"])` or `main(["-h"])` exits cleanly with code 0."""
        # ArgumentParser typically raises SystemExit(0) on --help
        try:
            exit_code = main(["--help"])
            assert exit_code == 0
        except SystemExit as exc:
            assert exc.code == 0

    def test_main_subcommand_help_flag_returns_zero(self) -> None:
        """Invoking `main(["run", "--help"])` exits cleanly with code 0."""
        try:
            exit_code = main(["run", "--help"])
            assert exit_code == 0
        except SystemExit as exc:
            assert exc.code == 0

    def test_main_uses_sys_argv_when_args_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When `args=None`, `main()` falls back to `sys.argv[1:]`."""
        with patch("rap_demo.cli.cmd_run") as mock_cmd_run:
            mock_cmd_run.return_value = 0
            monkeypatch.setattr(sys, "argv", ["rap", "run", "--iterations", "5"])
            exit_code = main(None)
            assert exit_code == 0
            assert mock_cmd_run.called

    def test_main_dispatches_to_run_command(self) -> None:
        """Calling `main(["run", ...])` dispatches to `cmd_run` handler."""
        with patch("rap_demo.cli.cmd_run", return_value=0) as mock_cmd:
            exit_code = main(["run", "--iterations", "5"])
            assert exit_code == 0
            assert mock_cmd.called
            args_passed = mock_cmd.call_args[0][0]
            assert args_passed.command == "run"
            assert args_passed.iterations == 5

    def test_main_dispatches_to_ui_command(self) -> None:
        """Calling `main(["ui", ...])` dispatches to `cmd_ui` handler."""
        with patch("rap_demo.cli.cmd_ui", return_value=0) as mock_cmd:
            exit_code = main(["ui", "--port", "8080"])
            assert exit_code == 0
            assert mock_cmd.called
            args_passed = mock_cmd.call_args[0][0]
            assert args_passed.command == "ui"
            assert args_passed.port == 8080

    def test_main_dispatches_to_compare_command(self) -> None:
        """Calling `main(["compare", ...])` dispatches to `cmd_compare` handler."""
        with patch("rap_demo.cli.cmd_compare", return_value=0) as mock_cmd:
            exit_code = main(["compare", "--iterations", "7"])
            assert exit_code == 0
            assert mock_cmd.called
            args_passed = mock_cmd.call_args[0][0]
            assert args_passed.command == "compare"
            assert args_passed.iterations == 7

    def test_main_catches_unhandled_exceptions_and_returns_exit_code_1(self) -> None:
        """Unhandled exceptions in command execution return exit code 1 instead of unhandled crash."""
        with patch("rap_demo.cli.cmd_run", side_effect=RuntimeError("Unexpected crash")):
            exit_code = main(["run"])
            assert exit_code == 1


# ============================================================================
# 3. Subcommand `rap run` Tests
# ============================================================================


class TestRunCommand:
    """Unit and functional tests for terminal search execution `rap run`."""

    def test_run_executes_mcts_search_successfully(self) -> None:
        """`rap run` executes MCTS search on default Blocksworld and returns 0."""
        exit_code = main(["run", "--iterations", "3", "--backend", "mock", "--mode", "hybrid"])
        assert exit_code == 0

    def test_run_hybrid_mode_execution(self) -> None:
        """`rap run` in hybrid mode runs deterministic physics transitions."""
        exit_code = main(["run", "--mode", "hybrid", "--iterations", "2"])
        assert exit_code == 0

    def test_run_paper_mode_execution(self) -> None:
        """`rap run` in paper mode runs text-based state updates."""
        exit_code = main(["run", "--mode", "paper", "--iterations", "2"])
        assert exit_code == 0

    def test_run_renders_rich_summary_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """`rap run` displays a formatted trajectory summary table in the terminal."""
        exit_code = main(["run", "--iterations", "3"])
        assert exit_code == 0
        captured = capsys.readouterr()
        combined_output = captured.out + captured.err
        # Rich table should print headers or step info
        assert (
            "Step" in combined_output
            or "Action" in combined_output
            or "Reward" in combined_output
            or "Trajectory" in combined_output
        )

    def test_run_verbose_flag_prints_detailed_tree(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With `--verbose`, `rap run` prints the full reasoning tree."""
        exit_code = main(["run", "--iterations", "3", "--verbose"])
        assert exit_code == 0
        captured = capsys.readouterr()
        combined_output = captured.out + captured.err
        # Verbose output includes tree structure or node metrics
        assert (
            "Root" in combined_output
            or "N=" in combined_output
            or "Q=" in combined_output
        )

    def test_run_forwards_iterations_to_mcts(self) -> None:
        """`rap run` passes the specified iterations count to the MCTS engine."""
        with patch("rap_demo.core.mcts.MCTS.search") as mock_search:
            mock_search.return_value = MCTSNode(state="mock_state", depth=0)
            exit_code = main(["run", "--iterations", "17"])
            assert exit_code == 0
            assert mock_search.called
            # Second argument or keyword argument is num_iterations
            _, kwargs = mock_search.call_args
            args = mock_search.call_args[0]
            iterations_passed = kwargs.get("num_iterations") or (args[1] if len(args) > 1 else None)
            assert iterations_passed == 17

    def test_run_handles_backend_connection_error(self) -> None:
        """When LLM backend fails, `rap run` catches error, logs message, and returns non-zero."""
        with patch("rap_demo.llm.factory.get_llm", side_effect=ConnectionError("Server unreachable")):
            exit_code = main(["run", "--backend", "llama-server"])
            assert exit_code != 0

    def test_run_validates_iterations_positive(self) -> None:
        """`rap run` returns non-zero when iterations <= 0."""
        exit_code = main(["run", "--iterations", "0"])
        assert exit_code != 0

        exit_code = main(["run", "--iterations", "-5"])
        assert exit_code != 0

    def test_run_validates_max_depth_positive(self) -> None:
        """`rap run` returns non-zero when the depth limit is not positive."""
        assert main(["run", "--max-depth", "0"]) != 0
        assert main(["run", "--max-depth", "-1"]) != 0

    def test_run_forwards_max_depth_to_mcts(self) -> None:
        """`rap run` passes the depth limit L to the MCTS engine (Algorithm 1)."""
        with patch("rap_demo.cli.MCTS") as mock_mcts_cls:
            mock_mcts = mock_mcts_cls.return_value
            mock_mcts.search.return_value = MCTSNode(state="mock_state", depth=0)
            mock_mcts.get_best_trajectory.return_value = [
                MCTSNode(state="mock_state", reward=0.0)
            ]

            exit_code = main(["run", "--max-depth", "9", "--no-save"])

        assert exit_code == 0
        assert mock_mcts_cls.call_args.kwargs["max_depth"] == 9


# ============================================================================
# 4. Subcommand `rap ui` Tests
# ============================================================================


class TestUICommand:
    """Unit and functional tests for `rap ui` web visualizer server launcher."""

    def test_ui_launches_uvicorn_with_configured_host_and_port(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """`rap ui` calls uvicorn.run with specified host and port."""
        exit_code = main(["ui", "--host", "0.0.0.0", "--port", "8888"])
        assert exit_code == 0
        assert mock_uvicorn_run.called

        _, kwargs = mock_uvicorn_run.call_args
        assert kwargs.get("host") == "0.0.0.0"
        assert kwargs.get("port") == 8888

    def test_ui_passes_fastapi_app_to_uvicorn(self, mock_uvicorn_run: MagicMock) -> None:
        """`rap ui` passes an instantiated FastAPI application to uvicorn.run."""
        from fastapi import FastAPI

        exit_code = main(["ui", "--backend", "mock"])
        assert exit_code == 0
        assert mock_uvicorn_run.called

        app_arg = mock_uvicorn_run.call_args[0][0]
        assert isinstance(app_arg, FastAPI)

    def test_ui_wires_controller_with_requested_mode(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """`rap ui --mode paper` configures the backend domain in paper mode."""
        with patch("rap_demo.visualizer.web.app.create_app") as mock_create_app:
            from fastapi import FastAPI
            mock_create_app.return_value = FastAPI()

            exit_code = main(["ui", "--mode", "paper", "--iterations", "25"])
            assert exit_code == 0
            assert mock_create_app.called

            controller_passed = mock_create_app.call_args[0][0]
            assert controller_passed.max_iterations == 25

    def test_ui_exits_zero_on_clean_server_termination(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """When uvicorn.run exits cleanly, `rap ui` returns exit code 0."""
        mock_uvicorn_run.return_value = None
        exit_code = main(["ui"])
        assert exit_code == 0

    def test_ui_handles_port_already_in_use_error(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """When uvicorn.run raises OSError (port in use), `rap ui` returns non-zero code."""
        mock_uvicorn_run.side_effect = OSError("Address already in use")
        exit_code = main(["ui", "--port", "8000"])
        assert exit_code != 0

    def test_ui_validates_port_range(self) -> None:
        """`rap ui` rejects out-of-range port numbers (< 1 or > 65535)."""
        exit_code = main(["ui", "--port", "0"])
        assert exit_code != 0

        exit_code = main(["ui", "--port", "70000"])
        assert exit_code != 0


# ============================================================================
# 5. Subcommand `rap compare` Tests
# ============================================================================


class TestCompareCommand:
    """Unit and functional tests for `rap compare` side-by-side mode execution."""

    def test_compare_executes_both_modes_successfully(self) -> None:
        """`rap compare` executes both Hybrid and Paper modes and returns 0."""
        exit_code = main(["compare", "--iterations", "2"])
        assert exit_code == 0

    def test_compare_renders_side_by_side_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`rap compare` renders a Rich comparison table displaying Hybrid and Paper metrics."""
        exit_code = main(["compare", "--iterations", "2"])
        assert exit_code == 0
        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # Check table contents
        assert "Hybrid" in combined_output
        assert "Paper" in combined_output

    def test_compare_includes_key_metrics(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Comparison table includes metrics: reward, visits/iterations, LLM calls, and steps."""
        exit_code = main(["compare", "--iterations", "3"])
        assert exit_code == 0
        captured = capsys.readouterr()
        combined_output = captured.out + captured.err.lower()

        # Check for core comparison metrics
        assert "reward" in combined_output or "score" in combined_output
        assert "call" in combined_output or "token" in combined_output or "llm" in combined_output

    def test_compare_verbose_prints_both_search_trees(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With `--verbose`, `rap compare` prints detailed trees for both modes."""
        exit_code = main(["compare", "--iterations", "2", "--verbose"])
        assert exit_code == 0
        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # Should render tree indicators
        assert "Root" in combined_output

    def test_compare_handles_search_failure_gracefully(self) -> None:
        """If one mode fails during comparison, returns non-zero exit code."""
        with patch("rap_demo.core.mcts.MCTS.search", side_effect=RuntimeError("Tree search failed")):
            exit_code = main(["compare", "--iterations", "3"])
            assert exit_code != 0


# ============================================================================
# 6. Domain & Argument Validation Tests
# ============================================================================


class TestCLIValidationErrors:
    """Tests argument parsing and domain validation edge cases."""

    @pytest.mark.parametrize(
        "subcommand",
        ["run", "compare"],
    )
    def test_unsupported_domain_returns_non_zero_exit_code(self, subcommand: str) -> None:
        """Unsupported domains like 'chess' or 'sokoban' return a non-zero exit code."""
        exit_code = main([subcommand, "--domain", "sokoban"])
        assert exit_code != 0

    @pytest.mark.parametrize(
        "invalid_arg",
        ["--unknown-flag", "--foo=bar", "-z"],
    )
    def test_unknown_flags_return_non_zero_exit_code(self, invalid_arg: str) -> None:
        """Unknown command-line flags return non-zero exit code."""
        exit_code = main(["run", invalid_arg])
        assert exit_code != 0

    def test_validation_error_message_printed_to_stderr_or_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Validation errors output an explanatory message."""
        main(["run", "--domain", "nonexistent_domain"])
        captured = capsys.readouterr()
        combined_output = (captured.out + captured.err).lower()
        assert "nonexistent_domain" in combined_output or "unsupported" in combined_output or "error" in combined_output or "invalid" in combined_output


# ============================================================================
# 7. End-to-End System Integration Tests
# ============================================================================


class TestCLIEndToEndIntegration:
    """Full-system end-to-end integration tests using MockLLM across CLI commands."""

    def test_e2e_run_blocksworld_hybrid_search(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: executes `rap run` on Blocksworld problem in hybrid mode."""
        exit_code = main(
            [
                "run",
                "--domain",
                "blocksworld",
                "--mode",
                "hybrid",
                "--backend",
                "mock",
                "--iterations",
                "5",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_e2e_run_blocksworld_paper_search(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: executes `rap run` on Blocksworld problem in paper mode."""
        exit_code = main(
            [
                "run",
                "--domain",
                "blocksworld",
                "--mode",
                "paper",
                "--backend",
                "mock",
                "--iterations",
                "5",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_e2e_compare_hybrid_vs_paper(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: executes `rap compare` and verifies comparative output."""
        exit_code = main(
            [
                "compare",
                "--domain",
                "blocksworld",
                "--backend",
                "mock",
                "--iterations",
                "4",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "Hybrid" in output
        assert "Paper" in output

    def test_e2e_ui_initialization_with_mock_uvicorn(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """End-to-end: verifies `rap ui` builds working FastAPI app with valid endpoints."""
        from httpx import ASGITransport, AsyncClient
        import asyncio

        created_app = None

        def fake_uvicorn_run(app: Any, **kwargs: Any) -> None:
            nonlocal created_app
            created_app = app

        mock_uvicorn_run.side_effect = fake_uvicorn_run

        exit_code = main(["ui", "--host", "127.0.0.1", "--port", "8000", "--backend", "mock"])
        assert exit_code == 0
        assert created_app is not None

        # Verify the created FastAPI app responds to basic queries
        async def verify_app() -> None:
            transport = ASGITransport(app=created_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                res_root = await client.get("/")
                assert res_root.status_code == 200

                res_tree = await client.get("/api/tree")
                assert res_tree.status_code == 200
                tree_data = res_tree.json()
                assert "id" in tree_data

                res_state = await client.get("/api/state")
                assert res_state.status_code == 200

        asyncio.run(verify_app())


# ============================================================================
# 8. Problem Definition (initial state & goal) Tests
# ============================================================================


class TestProblemDefinition:
    """Unit tests for `--initial-state` and `--goal` parsing helpers."""

    def test_parse_initial_state_from_json(self) -> None:
        """JSON specifications are converted into an equivalent BlockState."""
        state = _parse_initial_state(
            '{"clear": ["red"], "on_table": ["blue"], "on": {"red": "blue"}, "holding": null}'
        )
        assert state.clear == {"red"}
        assert state.on_table == {"blue"}
        assert state.on == {"red": "blue"}
        assert state.holding is None

    def test_parse_initial_state_from_text(self) -> None:
        """Canonical English descriptions are parsed by BlockState.from_text()."""
        state = _parse_initial_state(
            "the red block is clear, the hand is empty, "
            "the red block is on top of the blue block and the blue block is on the table."
        )
        assert state.clear == {"red"}
        assert state.on_table == {"blue"}
        assert state.on == {"red": "blue"}
        assert state.holding is None

    def test_parse_initial_state_keeps_demo_state_when_omitted(self) -> None:
        """Omitting --initial-state keeps the built-in demo state."""
        assert _parse_initial_state(None) == DEFAULT_INITIAL_STATE

    @pytest.mark.parametrize(
        "invalid_spec",
        [
            "",
            "   ",
            '{"clear": ["red"',
            '{"clear": "red"}',
            '{"clear": ["a"], "on": {"b": "a"}}',
            '{"clear": ["a"], "on_table": ["a"], "on": {"a": "b"}}',
            '{"holding": "a", "clear": ["a"]}',
            "garbage",
        ],
    )
    def test_parse_initial_state_rejects_invalid_specs(self, invalid_spec: str) -> None:
        """Malformed or physically inconsistent specifications raise ValueError."""
        with pytest.raises(ValueError):
            _parse_initial_state(invalid_spec)

    def test_parse_initial_state_detects_on_cycle(self) -> None:
        """Cyclic 'on' relations are rejected."""
        with pytest.raises(ValueError):
            _parse_initial_state('{"on": {"a": "b", "b": "a"}}')

    def test_parse_goal_splits_joined_conditions(self) -> None:
        """Conditions joined by 'and' or commas are split into separate goals."""
        goals = _parse_goal(
            [
                "the red block is on top of the blue block "
                "and the orange block is on top of the red block."
            ]
        )
        assert goals == [
            "the red block is on top of the blue block",
            "the orange block is on top of the red block",
        ]

    def test_parse_goal_keeps_demo_goal_when_omitted(self) -> None:
        """Omitting --goal keeps the built-in demo goal."""
        assert _parse_goal(None) == DEFAULT_GOAL

    @pytest.mark.parametrize("invalid_goal", [[], [""], ["   "], [" . "]])
    def test_parse_goal_rejects_empty_conditions(self, invalid_goal: list[str]) -> None:
        """Goal specifications without a usable condition raise ValueError."""
        with pytest.raises(ValueError):
            _parse_goal(invalid_goal)

    def test_run_accepts_custom_problem_definition(self) -> None:
        """`rap run` executes with a custom initial state and goal."""
        exit_code = main(
            [
                "run",
                "--backend",
                "mock",
                "--iterations",
                "2",
                "--no-save",
                "--initial-state",
                '{"clear": ["red"], "on_table": ["blue"], "on": {"red": "blue"}}',
                "--goal",
                "the red block is on top of the blue block",
            ]
        )
        assert exit_code == 0

    def test_run_rejects_invalid_problem_definition(self) -> None:
        """`rap run` returns non-zero for invalid initial state or goal specifications."""
        exit_code = main(["run", "--iterations", "1", "--initial-state", '{"clear": "red"}'])
        assert exit_code != 0

        exit_code = main(["run", "--iterations", "1", "--goal", "  "])
        assert exit_code != 0
