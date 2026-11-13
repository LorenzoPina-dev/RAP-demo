"""Command Line Interface for RAP Demonstration.

Provides CLI entry points for:
- `rap run`: terminal-based MCTS search execution with Rich visualization.
- `rap ui`: real-time web visualizer powered by FastAPI and Uvicorn.
- `rap compare`: side-by-side comparison of Hybrid vs Paper reasoning modes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from rich.console import Console
from rich.table import Table

from rap_demo.config import ModelsConfig, load_models_config
from rap_demo.core.events import EventBus
from rap_demo.core.mcts import MCTS, extract_best_trajectory
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.domain import BlocksworldDomain
from rap_demo.domains.blocksworld.planner import minimum_actions_to_goal
from rap_demo.domains.blocksworld.problem import (
    DEFAULT_GOAL,
    DEFAULT_INITIAL_STATE,
    parse_goal as _parse_goal,
    parse_initial_state as _parse_initial_state,
)
from rap_demo.llm.base import BaseLLM, LLMError
from rap_demo.llm.factory import get_llm
from rap_demo.llm.litellm_client import _load_dotenv_file
from rap_demo.llm.mock import MockLLM
from rap_demo.storage import (
    deserialize_tree,
    get_runs_dir,
    list_runs,
    load_run,
    save_run,
)
from rap_demo.visualizer.console import RichMCTSConsole
import rap_demo.visualizer.web.app
from rap_demo.visualizer.web.controller import SearchController, SearchStatus

SUPPORTED_DOMAINS: set[str] = {"blocksworld"}


def _create_llm(
    backend: str,
    model: str | None = None,
    api_base: str | None = None,
    tpm: int | None = None,
    rpm: int | None = None,
    timeout: float | None = None,
    config_path: str | Path | None = None,
    reasoning_effort: str | None = None,
) -> BaseLLM:
    """Instantiates and configures LLM backend for planning.

    Args:
        backend: Backend identifier ('mock', 'llama-server', or 'litellm').
        model: Optional model identifier forwarded to non-mock backends.
        api_base: Optional custom API base URL forwarded when provided.
        tpm: Optional tokens-per-minute rate limit for LiteLLM backend.
        rpm: Optional requests-per-minute rate limit for LiteLLM backend.
        timeout: Optional request timeout in seconds.
        config_path: Optional models.json path forwarded to LiteLLM backend.

    Returns:
        Configured BaseLLM instance.
    """
    if backend == "mock":
        blocks = ["red", "blue", "orange", "yellow"]
        rules: list[tuple[str | re.Pattern[str], str]] = []

        # 1. World model transitions for paper mode
        for b in blocks:
            rules.append((
                f"Pick up the {b} block.",
                f"[CHANGE] the hand was empty and now is holding the {b} block, "
                f"the {b} block was on the table and now is in the hand, "
                f"and the {b} block is no longer clear.",
            ))
            rules.append((
                f"Put down the {b} block.",
                f"[CHANGE] hand is empty, the {b} block was in the hand and now is on the table, "
                f"and the {b} block is now clear.",
            ))
            for t in blocks:
                if b != t:
                    rules.append((
                        f"Stack the {b} block on top of the {t} block.",
                        f"[CHANGE] hand is empty, the {b} block was in the hand and now is on top of the {t} block, "
                        f"the {b} block is now clear, and the {t} block is no longer clear.",
                    ))
                    rules.append((
                        f"Unstack the {b} block from on top of the {t} block.",
                        f"[CHANGE] the hand was empty and now is holding the {b} block, "
                        f"the {b} block was on top of the {t} block and now is in the hand, "
                        f"the {b} block is no longer clear, and the {t} block is now clear.",
                    ))

        # 2. Hybrid action proposals
        rules.append((
            re.compile(r"holding the red block"),
            json.dumps({
                "thought": "Stack red onto blue",
                "action": "stack",
                "block": "red",
                "target": "blue",
            }),
        ))
        rules.append((
            re.compile(r"holding the orange block"),
            json.dumps({
                "thought": "Stack orange onto red",
                "action": "stack",
                "block": "orange",
                "target": "red",
            }),
        ))
        rules.append((
            re.compile(r"red block is on top of the blue block"),
            json.dumps({
                "thought": "Unstack orange from yellow",
                "action": "unstack",
                "block": "orange",
                "target": "yellow",
            }),
        ))
        rules.append((
            re.compile(r".*"),
            json.dumps({
                "thought": "Pick up red from table",
                "action": "pick_up",
                "block": "red",
                "target": None,
            }),
        ))

        default_resp = (
            "[CHANGE] the hand was empty and now is holding the red block, "
            "the red block was on the table and now is in the hand, "
            "and the red block is no longer clear."
        )
        return get_llm(backend, default_response=default_resp, response_rules=rules)

    _load_dotenv_file()

    normalized = backend.strip().lower().replace("-", "_")
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if api_base is not None:
        if normalized in ("llama", "llama_server"):
            kwargs["base_url"] = api_base
        else:
            kwargs["api_base"] = api_base
    if timeout is not None:
        kwargs["timeout"] = timeout
    if normalized in ("litellm", "lite_llm"):
        if tpm is not None:
            kwargs["tokens_per_minute"] = tpm
        if rpm is not None:
            kwargs["requests_per_minute"] = rpm
        if config_path is not None:
            kwargs["config_path"] = config_path
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
    return get_llm(backend, **kwargs)


class _TrackExplicitAction(argparse.Action):
    """Store CLI argument value and record destination in namespace._explicit_args."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        explicit: set[str] | None = getattr(namespace, "_explicit_args", None)
        if explicit is None:
            explicit = set()
            setattr(namespace, "_explicit_args", explicit)
        explicit.add(self.dest)


def _resolve_execution_llm_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve effective LLM configuration by merging CLI flags with models.json profiles."""
    config_path = getattr(args, "config", None)
    model = getattr(args, "model", None)
    backend = str(getattr(args, "backend", "mock"))
    api_base = getattr(args, "api_base", None)
    tpm = getattr(args, "tpm", None)
    rpm = getattr(args, "rpm", None)
    timeout: float | None = None
    reasoning_effort: str | None = None
    explicit_args: set[str] = getattr(args, "_explicit_args", set())

    cfg: ModelsConfig | None = None
    if config_path is not None or model is not None or backend != "mock":
        cfg = load_models_config(config_path)

    if cfg is not None and model is None and backend == "litellm" and cfg.default_model:
        model = cfg.default_model

    notices: list[str] = []
    if model is not None:
        # Windows shells pass backslashes; provider model IDs use forward slashes.
        model = model.replace("\\", "/")
        while "//" in model:
            model = model.replace("//", "/")

    if cfg is not None and model is not None:
        profile = cfg.resolve_model(model)
        if profile is not None:
            if "backend" not in explicit_args:
                backend = profile.backend
            model = profile.model or profile.alias
            if api_base is None:
                api_base = profile.api_base
            if tpm is None:
                tpm = profile.tpm
            if rpm is None:
                rpm = profile.rpm
            timeout = profile.timeout if profile.timeout is not None else cfg.defaults.get("timeout")
            reasoning_effort = profile.reasoning_effort
            if "mode" not in explicit_args and profile.mode is not None:
                args.mode = profile.mode
            if "iterations" not in explicit_args and profile.iterations is not None:
                args.iterations = profile.iterations
        else:
            # A model was requested but is not a models.json profile. Treat it as a
            # raw provider model ID instead of silently running MockLLM (which made
            # the UI report P(Yes)=P(No)=0.5 and uniform r1 values).
            if "backend" not in explicit_args and backend == "mock":
                backend = "litellm"
            notices.append(
                f"Model '{model}' is not defined in models.json; using backend "
                f"'{backend}' with default settings. Add a profile to models.json to "
                "configure api_base, timeout, and rate limits."
            )

    resolved: dict[str, Any] = {
        "backend": backend,
        "model": model,
        "api_base": api_base,
        "tpm": tpm,
        "rpm": rpm,
    }
    if timeout is not None:
        resolved["timeout"] = float(timeout)
    if reasoning_effort is not None:
        resolved["reasoning_effort"] = reasoning_effort
    if config_path is not None:
        resolved["config_path"] = config_path
    if notices:
        resolved["notices"] = notices
    return resolved


def _add_llm_arguments(subparser: argparse.ArgumentParser) -> None:
    """Register shared LLM backend selection arguments on a CLI subparser."""
    subparser.add_argument(
        "--backend",
        choices=["mock", "llama-server", "litellm"],
        default="mock",
        action=_TrackExplicitAction,
        help="LLM backend identifier (choices: mock, llama-server, litellm; default: mock)",
    )
    subparser.add_argument(
        "--model",
        default=None,
        help="Model identifier or models.json alias (e.g. gpt-4o-mini, gemma4-26b)",
    )
    subparser.add_argument(
        "--api-base",
        default=None,
        help="Optional custom API base URL for LLM backend requests",
    )
    subparser.add_argument(
        "--tpm",
        type=int,
        default=None,
        help="Max tokens per minute (TPM) rate limit",
    )
    subparser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Max requests per minute (RPM) rate limit",
    )
    subparser.add_argument(
        "--config",
        default=None,
        help="Path to models.json configuration file",
    )


def _print_backend_error_tip(err_console: Console, backend: str) -> None:
    """Print a backend-specific troubleshooting tip to stderr."""
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in ("litellm", "lite_llm"):
        err_console.print(
            "\n[yellow]Tip:[/yellow] Make sure the appropriate API key environment variable is set "
            "(or defined in your [bold].env[/bold] file, e.g., [bold]OPENAI_API_KEY[/bold], "
            "[bold]ANTHROPIC_API_KEY[/bold], [bold]GEMINI_API_KEY[/bold]) and verify "
            "[bold]--model[/bold] / [bold]--api-base[/bold].\n"
        )
    elif normalized in ("llama", "llama_server"):
        err_console.print(
            "\n[yellow]Tip:[/yellow] Make sure llama-server is running with your Hugging Face model:"
        )
        err_console.print(
            "[bold green]llama-server -hf saidutta69/LFM3-VL-2B-heretic --port 8080 -ngl 99[/bold green]\n"
        )


def _add_problem_arguments(subparser: argparse.ArgumentParser) -> None:
    """Register Blocksworld problem definition flags on run/ui/compare subparsers."""
    subparser.add_argument(
        "--initial-state",
        default=None,
        metavar="SPEC",
        help=(
            "Initial state as JSON ({'clear': [...], 'on_table': [...], 'on': {...}, 'holding': ...}) "
            "or canonical English (default: built-in demo state)"
        ),
    )
    subparser.add_argument(
        "--goal",
        action="append",
        default=None,
        metavar="CONDITION",
        help=(
            "Goal condition, repeatable or joined with 'and' "
            "(default: built-in demo goal)"
        ),
    )
    subparser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        metavar="L",
        help=(
            "Depth limit L of the MCTS tree (Algorithm 1; default: 6). "
            "Must be at least the minimum number of actions needed by the goal."
        ),
    )
    subparser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        metavar="A",
        help=(
            "Weight of r1 (action likelihood) against r2 (goal heuristic) in the "
            "weighted geometric mean (default: 0.5, the paper's value). "
            "Lower values let the rule-based r2 steer the search more."
        ),
    )
    subparser.add_argument(
        "--self-eval-weight",
        type=float,
        default=0.25,
        metavar="W",
        help=(
            "Weight of r3 (self-evaluation) in the weighted geometric mean "
            "(default: 0.25; the paper does not specify a three-way weight)."
        ),
    )
    subparser.add_argument(
        "--r2-blocked-penalty",
        type=float,
        default=0.5,
        metavar="P",
        help=(
            "Penalty for goal conditions that are blocked, i.e. that need extra "
            "moves before they can be satisfied (default: 0.5). "
            "Use 0 for the exact reward of the paper."
        ),
    )


def _warn_unreachable_goal(
    err_console: Any,
    domain: BlocksworldDomain,
    max_depth: int,
) -> None:
    """Warn when the depth limit is too small for the shortest plan.

    The MCTS can only reach a goal within the depth limit L. When the task
    needs more actions than that, no branch can earn the "super large reward"
    and the search has no target to aim at.
    """
    try:
        needed = minimum_actions_to_goal(domain.get_initial_state(), domain.goal)
    except Exception:  # pragma: no cover - the check must never break the run
        return
    if needed is None:
        err_console.print(
            "[bold red]Warning:[/bold red] the goal cannot be reached from the initial state."
        )
        return
    if needed > max_depth:
        err_console.print(
            f"[bold yellow]Warning:[/bold yellow] the shortest plan needs {needed} actions, "
            f"but --max-depth is {max_depth}. The goal is unreachable in this tree: "
            f"no branch can earn the final reward. Use --max-depth {needed}."
        )


def _reward_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the reward-shaping flags shared by the run and ui commands."""
    return {
        "alpha": float(getattr(args, "alpha", 0.5)),
        "self_eval_weight": float(getattr(args, "self_eval_weight", 0.25)),
        "blocked_penalty": float(getattr(args, "r2_blocked_penalty", 0.5)),
    }


def _add_storage_arguments(subparser: argparse.ArgumentParser) -> None:
    """Register shared run persistence flags on run/ui subparsers."""
    subparser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory to store run JSON snapshots (default: $RAP_RUNS_DIR or ./runs)",
    )
    subparser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not automatically save run results to disk",
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the top-level argument parser for the rap CLI.

    Configures subcommands:
    - `run`: Terminal-based MCTS search execution.
    - `ui`: Real-time web visualizer server.
    - `compare`: Side-by-side mode comparison.
    - `history` (`runs`): List saved runs.
    - `replay`: Replay a cached run without making LLM calls.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="rap",
        description="Reasoning via Planning (RAP) Reimplementation with Real-Time Visualization",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. `rap run`
    run_parser = subparsers.add_parser("run", help="Run MCTS search in terminal")
    run_parser.add_argument(
        "--domain",
        default="blocksworld",
        help="Domain identifier (default: blocksworld)",
    )
    run_parser.add_argument(
        "--mode",
        choices=["hybrid", "paper"],
        default="paper",
        action=_TrackExplicitAction,
        help="Reasoning mode (choices: paper, hybrid; default: paper)",
    )
    _add_llm_arguments(run_parser)
    _add_problem_arguments(run_parser)
    _add_storage_arguments(run_parser)
    run_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        action=_TrackExplicitAction,
        help="MCTS search iterations (default: 10)",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed tree structure in terminal",
    )

    # 2. `rap ui`
    ui_parser = subparsers.add_parser("ui", help="Launch real-time web visualizer server")
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind server (default: 127.0.0.1)",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server (default: 8000)",
    )
    ui_parser.add_argument(
        "--domain",
        default="blocksworld",
        help="Domain identifier (default: blocksworld)",
    )
    ui_parser.add_argument(
        "--mode",
        choices=["hybrid", "paper"],
        default="paper",
        action=_TrackExplicitAction,
        help="Reasoning mode (choices: paper, hybrid; default: paper)",
    )
    _add_llm_arguments(ui_parser)
    _add_problem_arguments(ui_parser)
    _add_storage_arguments(ui_parser)
    ui_parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        action=_TrackExplicitAction,
        help="Maximum MCTS iterations (default: 50)",
    )

    # 3. `rap compare`
    compare_parser = subparsers.add_parser("compare", help="Side-by-side mode comparison")
    compare_parser.add_argument(
        "--domain",
        default="blocksworld",
        help="Domain identifier (default: blocksworld)",
    )
    _add_llm_arguments(compare_parser)
    _add_problem_arguments(compare_parser)
    compare_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="MCTS search iterations per mode (default: 10)",
    )
    compare_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed search trees for both modes",
    )

    # 4. `rap history` (alias: `runs`)
    history_parser = subparsers.add_parser(
        "history",
        aliases=["runs"],
        help="List saved MCTS runs",
    )
    history_parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory containing saved runs (default: $RAP_RUNS_DIR or ./runs)",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent runs to show (default: 20)",
    )

    # 5. `rap replay`
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a saved MCTS run without LLM API calls",
    )
    replay_parser.add_argument(
        "ref",
        nargs="?",
        default="latest",
        help="Run reference ('latest', 1-based index, run_id/prefix, or path; default: latest)",
    )
    replay_parser.add_argument(
        "--ui",
        action="store_true",
        default=False,
        help="Launch the Web UI preloaded with the cached run",
    )
    replay_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for replay Web UI (default: 127.0.0.1)",
    )
    replay_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for replay Web UI (default: 8000)",
    )
    replay_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print full search tree in terminal replay mode",
    )
    replay_parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory containing saved runs (default: $RAP_RUNS_DIR or ./runs)",
    )

    # 6. `rap models`
    models_parser = subparsers.add_parser(
        "models",
        help="List configured model profiles from models.json",
    )
    models_parser.add_argument(
        "--config",
        default=None,
        help="Path to models.json configuration file",
    )

    return parser


def _report_llm_notices(err_console: Console, cfg: dict[str, Any]) -> None:
    """Prints non-fatal LLM configuration warnings and removes them from the kwargs."""
    for notice in cfg.pop("notices", []):
        err_console.print(f"[yellow]Warning:[/yellow] {notice}")


def cmd_run(args: argparse.Namespace) -> int:
    """Executes terminal MCTS search according to parsed CLI arguments.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    console = Console()
    err_console = Console(stderr=True)

    if args.domain.lower() not in SUPPORTED_DOMAINS:
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported domain '{args.domain}'. "
            f"Supported domains: {', '.join(sorted(SUPPORTED_DOMAINS))}."
        )
        return 1

    if args.iterations <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Iterations must be positive, got {args.iterations}."
        )
        return 1

    if args.max_depth <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Max depth must be positive, got {args.max_depth}."
        )
        return 1

    try:
        initial_state = _parse_initial_state(getattr(args, "initial_state", None))
        goal = _parse_goal(getattr(args, "goal", None))
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    effective_backend = str(args.backend)
    try:
        resolved_llm_cfg = _resolve_execution_llm_config(args)
        effective_backend = str(resolved_llm_cfg["backend"])
        _report_llm_notices(err_console, resolved_llm_cfg)
        llm = _create_llm(**resolved_llm_cfg)
        domain = BlocksworldDomain(
            initial_state=initial_state,
            goal=goal,
            llm=llm,
            mode=args.mode,
        )
        _warn_unreachable_goal(err_console, domain, args.max_depth)
        mcts = MCTS(
            world_model=domain.create_world_model(**_reward_options(args)),
            reasoning_agent=domain.create_agent(),
            max_depth=args.max_depth,
            aggr_child="mean",
            event_bus=EventBus(),
        )

        root = mcts.search(domain.get_initial_state(), num_iterations=args.iterations)

        visualizer = RichMCTSConsole(console=console)
        if args.verbose:
            visualizer.print_tree(root)

        best_path = mcts.get_best_trajectory(root)
        best_reward = best_path[-1].reward if best_path else 0.0
        metrics: dict[str, Any] = {
            "Iterations": args.iterations,
            "Best Reward": f"{best_reward:.2f}",
            "Trajectory Steps": max(0, len(best_path) - 1),
        }
        if hasattr(llm, "call_count"):
            metrics["LLM Calls"] = llm.call_count

        visualizer.print_summary(best_path, metrics=metrics)

        if not getattr(args, "no_save", False):
            saved_path = save_run(
                root,
                domain=args.domain,
                mode=args.mode,
                backend=effective_backend,
                model=resolved_llm_cfg.get("model") or getattr(llm, "model", None),
                iterations=args.iterations,
                goal=goal,
                command="run",
                metrics=metrics,
                best_path=best_path,
                runs_dir=getattr(args, "runs_dir", None),
            )
            console.print(f"[dim]Saved run to {saved_path}[/dim]")

        return 0
    except (ConnectionError, ValueError, FileNotFoundError, LLMError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        if not isinstance(exc, FileNotFoundError):
            _print_backend_error_tip(err_console, effective_backend)
        return 1


def cmd_ui(args: argparse.Namespace) -> int:
    """Prepares and launches the real-time web visualizer server.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    console = Console()
    err_console = Console(stderr=True)

    if args.domain.lower() not in SUPPORTED_DOMAINS:
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported domain '{args.domain}'. "
            f"Supported domains: {', '.join(sorted(SUPPORTED_DOMAINS))}."
        )
        return 1

    if args.port < 1 or args.port > 65535:
        err_console.print(
            f"[bold red]Error:[/bold red] Port must be between 1 and 65535, got {args.port}."
        )
        return 1

    if args.iterations <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Iterations must be positive, got {args.iterations}."
        )
        return 1

    if args.max_depth <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Max depth must be positive, got {args.max_depth}."
        )
        return 1

    try:
        initial_state = _parse_initial_state(getattr(args, "initial_state", None))
        goal = _parse_goal(getattr(args, "goal", None))
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    effective_backend = str(args.backend)
    try:
        resolved_llm_cfg = _resolve_execution_llm_config(args)
        effective_backend = str(resolved_llm_cfg["backend"])
        _report_llm_notices(err_console, resolved_llm_cfg)
        llm = _create_llm(**resolved_llm_cfg)
        domain = BlocksworldDomain(
            initial_state=initial_state,
            goal=goal,
            llm=llm,
            mode=args.mode,
        )
        _warn_unreachable_goal(err_console, domain, args.max_depth)
        mcts = MCTS(
            world_model=domain.create_world_model(**_reward_options(args)),
            reasoning_agent=domain.create_agent(),
            max_depth=args.max_depth,
            aggr_child="mean",
            event_bus=EventBus(),
        )
        def _save_ui_run(ctrl: SearchController) -> None:
            best_traj = extract_best_trajectory(ctrl.root)
            best_reward = best_traj[-1].reward if best_traj else 0.0
            ui_metrics: dict[str, Any] = {
                "Iterations": ctrl.iteration,
                "Best Reward": f"{best_reward:.2f}",
                "Trajectory Steps": max(0, len(best_traj) - 1),
            }
            if hasattr(llm, "call_count"):
                ui_metrics["LLM Calls"] = llm.call_count
            save_run(
                ctrl.root,
                domain=args.domain,
                mode=args.mode,
                backend=effective_backend,
                model=resolved_llm_cfg.get("model") or getattr(llm, "model", None),
                iterations=ctrl.iteration,
                goal=ctrl.goal,
                command="ui",
                metrics=ui_metrics,
                best_path=best_traj,
                runs_dir=getattr(args, "runs_dir", None),
            )

        on_finish = None if getattr(args, "no_save", False) else _save_ui_run

        controller = SearchController(
            mcts=mcts,
            initial_state=domain.get_initial_state(),
            max_iterations=args.iterations,
            goal=goal,
            mode=args.mode,
            on_finish=on_finish,
            llm=llm,
        )

        create_app_fn = sys.modules["rap_demo.visualizer.web.app"].create_app
        runs_dir = getattr(args, "runs_dir", None)
        app = create_app_fn(controller, runs_dir=runs_dir) if runs_dir else create_app_fn(controller)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except OSError as exc:
        err_console.print(f"[bold red]Server Bind Error:[/bold red] {exc}")
        return 1
    except (ConnectionError, ValueError, FileNotFoundError, LLMError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        if not isinstance(exc, FileNotFoundError):
            _print_backend_error_tip(err_console, effective_backend)
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Executes side-by-side comparison of Hybrid vs Paper reasoning modes.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    console = Console()
    err_console = Console(stderr=True)

    if args.domain.lower() not in SUPPORTED_DOMAINS:
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported domain '{args.domain}'. "
            f"Supported domains: {', '.join(sorted(SUPPORTED_DOMAINS))}."
        )
        return 1

    if args.iterations <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Iterations must be positive, got {args.iterations}."
        )
        return 1

    if args.max_depth <= 0:
        err_console.print(
            f"[bold red]Error:[/bold red] Max depth must be positive, got {args.max_depth}."
        )
        return 1

    try:
        initial_state = _parse_initial_state(getattr(args, "initial_state", None))
        goal = _parse_goal(getattr(args, "goal", None))
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    effective_backend = str(args.backend)
    try:
        resolved_llm_cfg = _resolve_execution_llm_config(args)
        effective_backend = str(resolved_llm_cfg["backend"])
        _report_llm_notices(err_console, resolved_llm_cfg)

        # 1. Execute Hybrid Mode
        llm_hybrid = _create_llm(**resolved_llm_cfg)
        domain_hybrid = BlocksworldDomain(
            initial_state=initial_state,
            goal=goal,
            llm=llm_hybrid,
            mode="hybrid",
        )
        mcts_hybrid = MCTS(
            world_model=domain_hybrid.create_world_model(),
            reasoning_agent=domain_hybrid.create_agent(),
            max_depth=args.max_depth,
            aggr_child="mean",
        )
        t0_hybrid = time.perf_counter()
        root_hybrid = mcts_hybrid.search(domain_hybrid.get_initial_state(), num_iterations=args.iterations)
        time_hybrid = time.perf_counter() - t0_hybrid
        best_path_hybrid = mcts_hybrid.get_best_trajectory(root_hybrid)
        reward_hybrid = best_path_hybrid[-1].reward if best_path_hybrid else 0.0
        calls_hybrid = getattr(llm_hybrid, "call_count", 0)

        # 2. Execute Paper Mode
        llm_paper = _create_llm(**resolved_llm_cfg)
        domain_paper = BlocksworldDomain(
            initial_state=initial_state,
            goal=goal,
            llm=llm_paper,
            mode="paper",
        )
        mcts_paper = MCTS(
            world_model=domain_paper.create_world_model(),
            reasoning_agent=domain_paper.create_agent(),
            max_depth=args.max_depth,
            aggr_child="mean",
        )
        t0_paper = time.perf_counter()
        root_paper = mcts_paper.search(domain_paper.get_initial_state(), num_iterations=args.iterations)
        time_paper = time.perf_counter() - t0_paper
        best_path_paper = mcts_paper.get_best_trajectory(root_paper)
        reward_paper = best_path_paper[-1].reward if best_path_paper else 0.0
        calls_paper = getattr(llm_paper, "call_count", 0)

        visualizer = RichMCTSConsole(console=console)
        if args.verbose:
            console.print("\n[bold cyan]=== Hybrid Mode Reasoning Tree ===[/bold cyan]")
            visualizer.print_tree(root_hybrid)
            console.print("\n[bold cyan]=== Paper Mode Reasoning Tree ===[/bold cyan]")
            visualizer.print_tree(root_paper)

        # 3. Format Comparative Summary Table
        table = Table(
            title="Reasoning Mode Comparison: Hybrid vs Paper",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Hybrid Mode", style="green", justify="right")
        table.add_column("Paper Mode", style="yellow", justify="right")

        table.add_row("Final reward / score", f"{reward_hybrid:.2f}", f"{reward_paper:.2f}")
        table.add_row("Trajectory Steps", str(max(0, len(best_path_hybrid) - 1)), str(max(0, len(best_path_paper) - 1)))
        table.add_row("MCTS Iterations / Visits", str(args.iterations), str(args.iterations))
        table.add_row("LLM calls count", str(calls_hybrid), str(calls_paper))
        table.add_row("Execution Time (s)", f"{time_hybrid:.4f}", f"{time_paper:.4f}")
        table.add_row("Thoughts Preserved", "Yes (Structured JSON)", "No (Raw Text)")

        console.print(table)
        return 0
    except (ConnectionError, ValueError, FileNotFoundError, LLMError, RuntimeError) as exc:
        err_console.print(f"[bold red]Comparison Error:[/bold red] {exc}")
        if isinstance(exc, (ConnectionError, LLMError)):
            _print_backend_error_tip(err_console, effective_backend)
        return 1


def cmd_history(args: argparse.Namespace) -> int:
    """Lists saved MCTS search runs from disk."""
    console = Console()
    err_console = Console(stderr=True)
    limit = getattr(args, "limit", 20)
    if limit is not None and limit <= 0:
        err_console.print(f"[bold red]Error:[/bold red] Limit must be positive, got {limit}.")
        return 1

    runs_dir_arg = getattr(args, "runs_dir", None)
    runs = list_runs(runs_dir=runs_dir_arg, limit=limit)
    if not runs:
        console.print(f"No saved runs found in '{get_runs_dir(runs_dir_arg)}'.")
        return 0

    table = Table(
        title="Saved RAP Runs",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="cyan", justify="right", no_wrap=True)
    table.add_column("Run ID", style="green")
    table.add_column("Domain", style="white", no_wrap=True)
    table.add_column("Mode", style="yellow", no_wrap=True)
    table.add_column("Backend", style="blue", no_wrap=True)
    table.add_column("Iter", justify="right", no_wrap=True)
    table.add_column("Reward", justify="right", no_wrap=True)

    for idx, run in enumerate(runs, start=1):
        metrics = run.get("metrics") or {}
        best_reward = str(metrics.get("Best Reward", "-"))
        backend_label = str(run.get("backend", ""))
        if run.get("model"):
            backend_label = f"{backend_label} ({run['model']})"
        table.add_row(
            str(idx),
            str(run.get("run_id", "")),
            str(run.get("domain", "")),
            str(run.get("mode", "")),
            backend_label,
            str(run.get("iterations", 0)),
            best_reward,
        )

    console.print(table)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replays a saved MCTS run in terminal or Web UI without invoking any LLM."""
    console = Console()
    err_console = Console(stderr=True)

    if getattr(args, "ui", False) and (args.port < 1 or args.port > 65535):
        err_console.print(
            f"[bold red]Error:[/bold red] Port must be between 1 and 65535, got {args.port}."
        )
        return 1

    try:
        run_data = load_run(
            getattr(args, "ref", "latest"),
            runs_dir=getattr(args, "runs_dir", None),
        )
        root = deserialize_tree(run_data["tree"])
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        err_console.print(f"[bold red]Replay Error:[/bold red] {exc}")
        return 1

    if getattr(args, "ui", False):
        try:
            iterations = int(run_data.get("iterations", 0))
            controller = SearchController(
                mcts=None,
                initial_state=root.state,
                max_iterations=iterations,
                goal=run_data.get("goal"),
                mode=run_data.get("mode"),
                root=root,
                initial_iteration=iterations,
                initial_status=SearchStatus.FINISHED,
            )
            create_app_fn = sys.modules["rap_demo.visualizer.web.app"].create_app
            runs_dir = getattr(args, "runs_dir", None)
            app = create_app_fn(controller, runs_dir=runs_dir) if runs_dir else create_app_fn(controller)
            import uvicorn

            uvicorn.run(app, host=args.host, port=args.port)
            return 0
        except OSError as exc:
            err_console.print(f"[bold red]Server Bind Error:[/bold red] {exc}")
            return 1

    console.print(
        f"[bold cyan]Replaying Run:[/bold cyan] {run_data.get('run_id', 'unknown')} "
        f"| domain={run_data.get('domain')} mode={run_data.get('mode')} "
        f"backend={run_data.get('backend')} model={run_data.get('model') or '-'} "
        f"iterations={run_data.get('iterations')} created_at={run_data.get('created_at')}"
    )
    visualizer = RichMCTSConsole(console=console)
    if getattr(args, "verbose", False):
        visualizer.print_tree(root)

    best_path = extract_best_trajectory(root)
    metrics = run_data.get("metrics") or {"Iterations": run_data.get("iterations", 0)}
    visualizer.print_summary(best_path, metrics=metrics)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Lists configured model profiles from models.json."""
    console = Console()
    err_console = Console(stderr=True)

    try:
        cfg = load_models_config(getattr(args, "config", None))
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[bold red]Config Error:[/bold red] {exc}")
        return 1

    if not cfg.models:
        console.print("No models configured in models.json.")
        return 0

    table = Table(
        title="Configured RAP Model Profiles",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Backend", style="green", no_wrap=True)
    table.add_column("Model", style="white", no_wrap=True)
    table.add_column("TPM", justify="right", no_wrap=True)
    table.add_column("RPM", justify="right", no_wrap=True)
    table.add_column("API Base", style="dim")
    table.add_column("Default", justify="center", no_wrap=True)

    for alias, profile in cfg.models.items():
        is_default = "Yes" if cfg.default_model == alias else ""
        table.add_row(
            alias,
            profile.backend,
            profile.model or "-",
            str(profile.tpm) if profile.tpm is not None else "-",
            str(profile.rpm) if profile.rpm is not None else "-",
            profile.api_base or "-",
            is_default,
        )

    console.print(table)
    return 0


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for the `rap` command.

    Args:
        args: Optional list of command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Integer exit status code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    argv = list(args) if args is not None else sys.argv[1:]

    if not argv:
        parser.print_help(file=sys.stderr)
        return 1

    try:
        parsed_args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        console = Console(stderr=True)
        console.print(f"[bold red]Parsing Error:[/bold red] {exc}")
        return 2

    command = getattr(parsed_args, "command", None)
    if command is None:
        parser.print_help(file=sys.stderr)
        return 1

    try:
        _load_dotenv_file()
        if command == "run":
            return cmd_run(parsed_args)
        elif command == "ui":
            return cmd_ui(parsed_args)
        elif command == "compare":
            return cmd_compare(parsed_args)
        elif command in ("history", "runs"):
            return cmd_history(parsed_args)
        elif command == "replay":
            return cmd_replay(parsed_args)
        elif command == "models":
            return cmd_models(parsed_args)
        else:
            parser.print_help(file=sys.stderr)
            return 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        console = Console(stderr=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
