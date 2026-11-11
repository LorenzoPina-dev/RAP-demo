"""Persistence, listing, loading, and tree deserialization for RAP MCTS runs."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rap_demo.core.mcts import extract_best_trajectory
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.visualizer.serializer import _serialize_state, serialize_tree


def get_runs_dir(runs_dir: Path | str | None = None) -> Path:
    """Resolve the run storage directory (explicit arg -> RAP_RUNS_DIR -> Path('runs'))."""
    if runs_dir is not None:
        if isinstance(runs_dir, Path) or runs_dir.strip():
            return Path(runs_dir)

    env_dir = os.environ.get("RAP_RUNS_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip())

    return Path("runs")


def _slugify(value: str | None, fallback: str = "unknown") -> str:
    """Sanitize a string for safe inclusion in a filename."""
    if not value or not value.strip():
        return fallback
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-_.")
    return slug or fallback


def _resolve_unique_path(directory: Path, stem: str) -> Path:
    """Return a non-colliding .json path inside directory, appending _2, _3, ... if needed."""
    candidate = directory / f"{stem}.json"
    if not candidate.exists():
        return candidate

    suffix_idx = 2
    while True:
        candidate = directory / f"{stem}_{suffix_idx}.json"
        if not candidate.exists():
            return candidate
        suffix_idx += 1


def _reconstruct_state(state_data: Any) -> Any:
    """Reconstruct a BlockState from serialized state JSON, or return raw value."""
    if isinstance(state_data, dict) and {"stacks", "clear", "on_table"}.issubset(state_data):
        on: dict[str, str] = {}
        for stack in state_data.get("stacks", []):
            if isinstance(stack, list):
                for idx in range(1, len(stack)):
                    on[str(stack[idx])] = str(stack[idx - 1])
        return BlockState(
            clear={str(b) for b in state_data.get("clear", [])},
            on_table={str(b) for b in state_data.get("on_table", [])},
            on=on,
            holding=state_data.get("holding"),
        )
    return state_data


def _serialize_trajectory_steps(best_path: list[MCTSNode]) -> list[dict[str, Any]]:
    """Serialize best-trajectory nodes into a concise JSON-friendly summary list."""
    steps: list[dict[str, Any]] = []
    for node in best_path:
        q_val = 0.0 if math.isinf(node.Q) or math.isnan(node.Q) else float(node.Q)
        thought = getattr(node, "thought", None) or (
            getattr(node.action, "thought", None) if node.action else None
        )
        steps.append(
            {
                "depth": int(node.depth),
                "action": str(node.action) if node.action is not None else None,
                "thought": thought,
                "llm_trace": getattr(node, "llm_trace", None),
                "reward": float(node.reward),
                "visits": int(node.N),
                "Q": q_val,
                "is_terminal": bool(node.is_terminal),
                "state": _serialize_state(node.state),
            }
        )
    return steps


def save_run(
    root: MCTSNode,
    domain: str = "blocksworld",
    mode: str = "hybrid",
    backend: str = "mock",
    model: str | None = None,
    iterations: int = 0,
    goal: Any = None,
    command: str = "run",
    metrics: dict[str, Any] | None = None,
    best_path: list[MCTSNode] | None = None,
    runs_dir: Path | str | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Persist an MCTS run's metadata, metrics, tree, and best trajectory to JSON."""
    timestamp = created_at or datetime.now(timezone.utc)
    target_dir = get_runs_dir(runs_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        timestamp.strftime("%Y%m%d_%H%M%S"),
        _slugify(domain),
        _slugify(mode),
        _slugify(backend),
    ]
    if model:
        parts.append(_slugify(model))
    parts.append(f"iter{int(iterations)}")

    file_path = _resolve_unique_path(target_dir, "_".join(parts))
    resolved_best = best_path if best_path is not None else extract_best_trajectory(root)

    payload: dict[str, Any] = {
        "run_id": file_path.stem,
        "created_at": timestamp.isoformat(),
        "command": command,
        "domain": domain,
        "mode": mode,
        "backend": backend,
        "model": model,
        "iterations": int(iterations),
        "goal": goal,
        "metrics": dict(metrics) if metrics is not None else {},
        "tree": serialize_tree(root),
        "best_path_ids": [f"node_{id(node)}" for node in resolved_best],
        "best_trajectory": _serialize_trajectory_steps(resolved_best),
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path


def deserialize_tree(tree_data: dict[str, Any], parent: MCTSNode | None = None) -> MCTSNode:
    """Reconstruct an MCTSNode tree (including stats and BlockState) from serialized JSON."""
    node = MCTSNode(
        state=_reconstruct_state(tree_data.get("state")),
        action=tree_data.get("action"),
        parent=parent,
        depth=int(tree_data.get("depth", 0)),
        reward=float(tree_data.get("reward", 0.0)),
        is_terminal=bool(tree_data.get("is_terminal", False)),
        thought=tree_data.get("thought"),
        llm_trace=tree_data.get("llm_trace"),
        llm_calls=tree_data.get("llm_calls"),
    )
    node.N = int(tree_data.get("visits", 0))
    node.Q = float(tree_data.get("Q", 0.0))
    raw_m = tree_data.get("M")
    node.M = -math.inf if raw_m is None else float(raw_m)
    node.visited = node.N > 0

    for child_data in tree_data.get("children", []) or []:
        node.children.append(deserialize_tree(child_data, parent=node))

    return node


def _parse_created_at_sort_key(created_at: str, path: Path) -> float:
    """Convert created_at ISO string to a numeric timestamp for newest-first sorting."""
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def _read_run_json(path: Path) -> dict[str, Any] | None:
    """Read and minimally validate a saved run JSON file, returning None if malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "tree" not in data or "created_at" not in data:
        return None
    return data


def list_runs(
    runs_dir: Path | str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List saved runs sorted newest-first, ignoring malformed or unrelated JSON files."""
    target_dir = get_runs_dir(runs_dir)
    if not target_dir.is_dir():
        return []

    records: list[tuple[float, dict[str, Any]]] = []
    for path in target_dir.glob("*.json"):
        data = _read_run_json(path)
        if data is None:
            continue
        created_at = str(data.get("created_at", ""))
        summary: dict[str, Any] = {
            "path": str(path),
            "run_id": data.get("run_id", path.stem),
            "created_at": created_at,
            "command": data.get("command", "run"),
            "domain": data.get("domain", "unknown"),
            "mode": data.get("mode", "unknown"),
            "backend": data.get("backend", "unknown"),
            "model": data.get("model"),
            "iterations": int(data.get("iterations", 0)),
            "metrics": data.get("metrics", {}),
            "goal": data.get("goal"),
        }
        records.append((_parse_created_at_sort_key(created_at, path), summary))

    records.sort(key=lambda item: (item[0], item[1]["run_id"]), reverse=True)
    summaries = [item[1] for item in records]
    if limit is not None and limit >= 0:
        return summaries[:limit]
    return summaries


def load_run(ref: str | Path = "latest", runs_dir: Path | str | None = None) -> dict[str, Any]:
    """Load a saved run by 'latest', 1-based index, run_id/prefix, or direct file Path."""
    if isinstance(ref, Path):
        if not ref.is_file():
            raise FileNotFoundError(f"Run file not found: {ref}")
        data = _read_run_json(ref)
        if data is None:
            raise ValueError(f"Malformed run file: {ref}")
        return data

    ref_str = str(ref).strip()
    direct_path = Path(ref_str)
    if ref_str and direct_path.is_file():
        data = _read_run_json(direct_path)
        if data is None:
            raise ValueError(f"Malformed run file: {direct_path}")
        return data

    runs = list_runs(runs_dir=runs_dir)
    if not runs:
        raise FileNotFoundError(
            f"No saved runs found in '{get_runs_dir(runs_dir)}' matching {ref!r}."
        )

    matched: dict[str, Any] | None = None
    if ref_str.lower() == "latest":
        matched = runs[0]
    elif ref_str.isdigit():
        idx = int(ref_str) - 1
        if 0 <= idx < len(runs):
            matched = runs[idx]
    else:
        for run in runs:
            run_id = str(run["run_id"])
            stem = Path(run["path"]).stem
            if run_id == ref_str or stem == ref_str:
                matched = run
                break
        if matched is None:
            for run in runs:
                run_id = str(run["run_id"])
                stem = Path(run["path"]).stem
                if run_id.startswith(ref_str) or stem.startswith(ref_str):
                    matched = run
                    break

    if matched is None:
        raise FileNotFoundError(
            f"Saved run {ref!r} not found in '{get_runs_dir(runs_dir)}'."
        )

    matched_path = Path(matched["path"])
    data = _read_run_json(matched_path)
    if data is None:
        raise ValueError(f"Malformed run file: {matched_path}")
    return data
