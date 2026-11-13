"""Prompt templates for Blocksworld planning.

The ``paper`` prompts reproduce the templates published in the RAP paper
(Hao et al., EMNLP 2023), Appendix C.1: the action-likelihood prompt and the
world-model state-update prompt. The ``hybrid`` prompt is an extra, structured
JSON interface that is not part of the paper.

Appendix E ("Adaptive Prompting") is implemented by ``remaining_steps``: the
in-context demonstrations are truncated from the beginning as the search tree
deepens, so the minimum number of actions in the demos tracks the number of
actions still needed from the current node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rap_demo.domains.blocksworld.models import BlockAction, BlockState


HYBRID_SYSTEM_PROMPT = """You are a reasoning and planning agent for the Blocksworld domain.
Analyze the current state and propose the next action in JSON format.

JSON format:
```json
{
  "thought": "<your reasoning here>",
  "action": "<pick_up | put_down | stack | unstack>",
  "block": "<block name>",
  "target": "<target block name or null>"
}
```"""


# ---------------------------------------------------------------------------
# Paper, Appendix C.1: shared instruction block
# ---------------------------------------------------------------------------

# The marker the action prompt ends with. The action is scored as the
# continuation that follows it.
PAPER_PLAN_MARKER = "[PLAN]\n"

BLOCKSWORLD_RULES = """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty."""


@dataclass(frozen=True)
class PlanDemo:
    """One in-context demonstration from the paper (Appendix C.1)."""

    initial_state: str
    goal: str
    plan: tuple[str, ...]


# The four demonstrations listed in Appendix C.1, verbatim.
PAPER_PLAN_DEMOS: tuple[PlanDemo, ...] = (
    PlanDemo(
        initial_state=(
            "the red block is clear, the yellow block is clear, the hand is empty, "
            "the red block is on top of the blue block, the yellow block is on top of the orange block, "
            "the blue block is on the table and the orange block is on the table"
        ),
        goal="the orange block is on top of the red block",
        plan=(
            "unstack the yellow block from on top of the orange block",
            "put down the yellow block",
            "pick up the orange block",
            "stack the orange block on top of the red block",
        ),
    ),
    PlanDemo(
        initial_state=(
            "the orange block is clear, the yellow block is clear, the hand is empty, "
            "the blue block is on top of the red block, the orange block is on top of the blue block, "
            "the red block is on the table and the yellow block is on the table"
        ),
        goal="the blue block is on top of the red block and the yellow block is on top of the orange block",
        plan=(
            "pick up the yellow block",
            "stack the yellow block on top of the orange block",
        ),
    ),
    PlanDemo(
        initial_state=(
            "the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, "
            "the blue block is on top of the yellow block, the red block is on the table, "
            "the orange block is on the table and the yellow block is on the table"
        ),
        goal="the blue block is on top of the orange block and the yellow block is on top of the red block",
        plan=(
            "unstack the blue block from on top of the yellow block",
            "stack the blue block on top of the orange block",
            "pick up the yellow block",
            "stack the yellow block on top of the red block",
        ),
    ),
    PlanDemo(
        initial_state=(
            "the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, "
            "the yellow block is on top of the orange block, the red block is on the table, "
            "the blue block is on the table and the orange block is on the table"
        ),
        goal="the orange block is on top of the blue block and the yellow block is on top of the red block",
        plan=(
            "unstack the yellow block from on top of the orange block",
            "stack the yellow block on top of the red block",
            "pick up the orange block",
            "stack the orange block on top of the blue block",
        ),
    ),
)


# The two world-model demonstrations listed in Appendix C.1, verbatim.
PAPER_WORLD_DEMOS: tuple[tuple[str, str, str, str], ...] = (
    (
        "the white block is clear, the cyan block is clear, the brown block is clear, the hand is empty, "
        "the white block is on top of the purple block, the purple block is on the table, "
        "the cyan block is on the table and the brown block is on the table",
        "Pick up the brown block.",
        "The hand was empty and is now holding the brown block, the brown block was on the table "
        "and is now in the hand, and the brown block is no longer clear.",
        "the white block is clear, the cyan block is clear, the brown block is in the hand, "
        "the hand is holding the brown block, the white block is on top of the purple block, "
        "the purple block is on the table and the cyan block is on the table",
    ),
    (
        "the purple block is clear, the cyan block is clear, the white block is clear, the hand is empty, "
        "the white block is on top of the brown block, the purple block is on the table, "
        "the cyan block is on the table and the brown block is on the table",
        "Pick up the cyan block.",
        "The hand was empty and is now holding the cyan block, the cyan block was on the table "
        "and is now in the hand, and the cyan block is no longer clear.",
        "the cyan block is in the hand, the white block is clear, the purple block is clear, "
        "the hand is holding the cyan block, the white block is on top of the brown block, "
        "the purple block is on the table and the brown block is on the table",
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_goal(goal: str | Sequence[str] | dict[str, str]) -> str:
    """Formats goal representation into a canonical natural language string."""
    if isinstance(goal, str):
        return goal
    elif isinstance(goal, dict):
        clauses = [f"the {k} block is on top of the {v} block" for k, v in sorted(goal.items())]
        if not clauses:
            return ""
        if len(clauses) == 1:
            return clauses[0]
        return ", ".join(clauses[:-1]) + " and " + clauses[-1]
    else:
        clauses = list(goal)
        if not clauses:
            return ""
        if len(clauses) == 1:
            return clauses[0]
        return ", ".join(clauses[:-1]) + " and " + clauses[-1]


def _strip_period(text: str) -> str:
    """Removes a single trailing period so clauses can be re-joined cleanly."""
    cleaned = text.strip()
    return cleaned[:-1] if cleaned.endswith(".") else cleaned


def _demo_state_after(demo: PlanDemo, skip: int) -> str:
    """Replays the first ``skip`` demo actions to obtain the truncated initial state.

    This mirrors Appendix E: demonstrations are pre-computed intermediate states,
    and the trace is truncated from the beginning as the tree deepens. If the
    replay is not possible the untouched demo state is returned.
    """
    if skip <= 0:
        return demo.initial_state
    try:
        state = BlockState.from_text(demo.initial_state)
        for action_text in demo.plan[:skip]:
            state = state.apply_action(BlockAction.from_text(action_text))
    except Exception:
        return demo.initial_state
    return _strip_period(state.to_text())


def _render_plan_demo(demo: PlanDemo, skip: int = 0) -> str:
    """Renders one ``[STATEMENT] ... [PLAN] ... [PLAN END]`` demonstration block."""
    actions = demo.plan[skip:] or demo.plan[-1:]
    lines = [
        "[STATEMENT]",
        f"As initial conditions I have that, {_demo_state_after(demo, skip)}.",
        f"My goal is to have that {_strip_period(demo.goal)}.",
        "My plan is as follows:",
        "[PLAN]",
        *actions,
        "[PLAN END]",
    ]
    return "\n".join(lines)


def _demo_skip(plan_length: int, remaining_steps: int | None) -> int:
    """Computes how many leading demo actions to drop for a given remaining budget."""
    if remaining_steps is None:
        return 0
    keep = max(1, min(plan_length, remaining_steps))
    return plan_length - keep


# ---------------------------------------------------------------------------
# Paper prompts (Appendix C.1)
# ---------------------------------------------------------------------------


def build_paper_action_prompt(
    state: BlockState,
    goal: str | Sequence[str] | dict[str, str] = "",
    remaining_steps: int | None = None,
) -> str:
    """Builds the RAP action-likelihood prompt (Appendix C.1).

    The returned prompt ends with ``[PLAN]`` so that an action can be scored as a
    continuation, exactly as the paper computes r1 = log P(action | state, goal).
    """
    goal_str = _strip_period(format_goal(goal))
    blocks = [BLOCKSWORLD_RULES, ""]
    for demo in PAPER_PLAN_DEMOS:
        blocks.append(_render_plan_demo(demo, _demo_skip(len(demo.plan), remaining_steps)))
        blocks.append("")
    blocks.extend(
        [
            "[STATEMENT]",
            f"As initial conditions I have that, {_strip_period(state.to_text())}.",
            f"My goal is to have that {goal_str}.",
            "My plan is as follows:",
            "[PLAN]",
            "",
        ]
    )
    return "\n".join(blocks)


def split_paper_plan_prompt(prompt: str) -> tuple[str, str]:
    """Split the action prompt into its context and the plan marker.

    The paper scores an action as the continuation of the prompt, which ends
    with the ``[PLAN]`` marker. A chat backend continues the action in the
    assistant turn instead, so the marker is handed over as the assistant-side
    context and removed from the prompt to avoid repeating it.

    Args:
        prompt: The action-likelihood prompt.

    Returns:
        The prompt without the marker and the marker itself (empty if absent).
    """
    head, marker, _ = prompt.rpartition(PAPER_PLAN_MARKER)
    if not marker:
        return prompt, ""
    return head, marker


def build_world_update_prompt(state: BlockState, action: BlockAction) -> str:
    """Builds the RAP world-model prompt (Appendix C.1).

    The model is expected to answer with ``[CHANGE] ...`` and then ``[STATE 1]``.
    Callers should stop generation at ``[STATE 1]``.
    """
    blocks = [
        BLOCKSWORLD_RULES,
        "",
        "After being given an initial state and an action, give the new state after performing the action.",
        "Answer with a single [CHANGE] description in the same style as the examples: state only the facts "
        "that changed, in one sentence, and do not explain your reasoning.",
        "",
    ]
    for index, (state_text, action_text, change_text, next_state_text) in enumerate(PAPER_WORLD_DEMOS, start=1):
        blocks.extend(
            [
                f"[SCENARIO {index}]",
                f"[STATE 0] I have that, {state_text}.",
                f"[ACTION] {action_text}",
                f"[CHANGE] {change_text}",
                f"[STATE 1] I have that, {next_state_text}.",
                "",
            ]
        )
    blocks.extend(
        [
            f"[SCENARIO {len(PAPER_WORLD_DEMOS) + 1}]",
            f"[STATE 0] I have that, {_strip_period(state.to_text())}.",
            f"[ACTION] {action.to_text()}",
            "[CHANGE]",
        ]
    )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Hybrid prompts (not part of the paper)
# ---------------------------------------------------------------------------


def build_hybrid_action_prompt(state: BlockState, goal: str | Sequence[str] | dict[str, str] = "") -> str:
    """Builds prompt requesting a structured JSON action proposal with internal reasoning thought."""
    goal_str = format_goal(goal)
    goal_line = f"Goal: {goal_str}\n\n" if goal_str else ""
    return (
        f"{HYBRID_SYSTEM_PROMPT}\n\n"
        f"{goal_line}"
        f"Current State: {state.to_text()}\n\n"
        f"Output your proposal in valid JSON:"
    )


def build_self_eval_prompt(
    state: BlockState,
    next_state: BlockState,
    goal: str | Sequence[str] | dict[str, str],
) -> str:
    """Builds prompt querying LLM for self-evaluation of a state transition.

    The question is about the *transition*, not about the action: the reference
    implementation asks "is STATE 1 closer to the goal than STATE 0?". Asking
    about the action instead makes the model answer "No" for the necessary
    preparatory moves that do not satisfy a condition right away, which zeroes
    the reward of the right first step.
    """
    goal_str = format_goal(goal)
    return (
        "After being given an initial state, a new state, and a goal, determine if the new state "
        'is closer to the goal. If the new state is closer, respond with "Yes". '
        'Otherwise, respond with "No". Do not explain your reasoning.\n\n'
        f"[STATE 0] I have that, {_strip_period(state.to_text())}.\n"
        f"[STATE 1] I have that, {_strip_period(next_state.to_text())}.\n"
        f"[GOAL] My goal is to have that {_strip_period(goal_str)}.\n"
        '[QUESTION] Is "STATE 1" closer to the goal than "STATE 0"? Answer Yes or No:'
    )
