"""World model simulation for Blocksworld."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence
from rap_demo.core.mcts import WorldModel
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState
from rap_demo.domains.blocksworld.prompts import build_self_eval_prompt, build_world_update_prompt
from rap_demo.domains.blocksworld.rewards import (
    calculate_combined_reward,
    calculate_goal_reward,
    self_evaluation_result,
)
from rap_demo.llm.base import BaseLLM

# The world-model update is a one-sentence task, so the reasoning models must not
# spend the whole budget reasoning about the prompt instead of answering.
_WORLD_MODEL_REASONING_EFFORT = "low"


def _apply_text_change(state: BlockState, change_text: str) -> BlockState:
    """Applies a natural language state transition description to a BlockState."""
    cleaned = change_text.strip()
    # Remove prefix like [CHANGE]
    cleaned = re.sub(r"^\[change\]:?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    # Normalize "and is now" / "and are now" to the canonical "and now is" form
    # used by the paper's demonstrations.
    cleaned = re.sub(r"\band\s+is\s+now\b", "and now is", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+are\s+now\b", "and now are", cleaned, flags=re.IGNORECASE)

    # Reject hallucinated / impossible change texts
    hallucination_indicators = ["disappear", "destroy", "vanish", "teleport", "unknown", "invalid", "broken"]
    if any(h in cleaned.lower() for h in hallucination_indicators):
        raise ValueError(f"State update rejected due to hallucinated physics in LLM output: {change_text!r}")

    new_clear = set(state.clear)
    new_on_table = set(state.on_table)
    new_on = dict(state.on)
    new_holding = state.holding

    raw_clauses = [c.strip().rstrip(".") for c in re.split(r"[,;]\s*(?:and\s+)?", cleaned) if c.strip()]
    if not raw_clauses:
        raise ValueError(f"No valid state change clauses found in: {change_text!r}")

    matched_any = False
    for raw_clause in raw_clauses:
        clause = re.sub(r"^and\s+", "", raw_clause.strip(), flags=re.IGNORECASE)
        c_lower = clause.lower()

        # 1. Hand holding
        m = re.search(r"(?:hand\s+was\s+empty\s+and\s+now\s+is\s+holding\s+the|holding\s+the)\s+(\w+)\s+block", c_lower)
        if m:
            new_holding = m.group(1)
            matched_any = True
            continue

        # 2. Hand now empty
        if (
            ("hand was holding" in c_lower and "now is empty" in c_lower)
            or re.search(r"\bhand\s+is\s+(?:now\s+)?empty\b", c_lower)
        ):
            new_holding = None
            matched_any = True
            continue

        # 3. Block was on table and now in hand
        m = re.search(r"the\s+(\w+)\s+block\s+was\s+on\s+the\s+table\s+and\s+now\s+is\s+in\s+the\s+hand", c_lower)
        if m:
            b = m.group(1)
            new_on_table.discard(b)
            new_holding = b
            matched_any = True
            continue

        # 3b. A concise equivalent often returned by models: "X is in the hand".
        m = re.search(r"the\s+(\w+)\s+block\s+is\s+(?:now\s+)?in\s+the\s+hand", c_lower)
        if m:
            b = m.group(1)
            new_on_table.discard(b)
            new_on.pop(b, None)
            new_holding = b
            matched_any = True
            continue

        # 4. Block was in hand and now on table
        m = re.search(r"the\s+(\w+)\s+block\s+was\s+in\s+the\s+hand\s+and\s+now\s+is\s+on\s+the\s+table", c_lower)
        if m:
            b = m.group(1)
            new_on_table.add(b)
            new_holding = None
            matched_any = True
            continue

        # 5. Block was in hand and now on top of target
        m = re.search(
            r"the\s+(\w+)\s+block\s+was\s+in\s+the\s+hand\s+and\s+now\s+is\s+on\s+top\s+of\s+the\s+(\w+)\s+block",
            c_lower,
        )
        if m:
            top = m.group(1)
            bottom = m.group(2)
            new_on[top] = bottom
            new_holding = None
            matched_any = True
            continue

        # 6. A concise equivalent often returned by models: "X is now on top of Y".
        m = re.search(
            r"the\s+(\w+)\s+block\s+is\s+(?:now\s+)?on\s+top\s+of\s+the\s+(\w+)\s+block",
            c_lower,
        )
        if m:
            top = m.group(1)
            bottom = m.group(2)
            new_on[top] = bottom
            new_on_table.discard(top)
            if new_holding == top:
                new_holding = None
            matched_any = True
            continue

        # 7. Block was on top of target and now in hand
        m = re.search(
            r"the\s+(\w+)\s+block\s+was\s+on\s+top\s+of\s+the\s+(\w+)\s+block\s+and\s+now\s+is\s+in\s+the\s+hand",
            c_lower,
        )
        if m:
            top = m.group(1)
            new_on.pop(top, None)
            new_holding = top
            matched_any = True
            continue

        # 8. Block is no longer covered by another block, which frees it up.
        m = re.search(
            r"the\s+(\w+)\s+block\s+is\s+no\s+longer\s+(?:covered\s+by|on\s+top\s+of)\s+the\s+(\w+)\s+block",
            c_lower,
        )
        if m:
            top, bottom = m.group(1), m.group(2)
            if new_on.get(top) == bottom:
                del new_on[top]
            # The same clause often states the resulting state as well.
            if re.search(r"and\s+now\s+is\s+clear", c_lower):
                new_clear.add(top)
            matched_any = True
            continue

        # 9. Block is no longer clear
        m = re.search(r"the\s+(\w+)\s+block\s+is\s+no\s+longer\s+clear", c_lower)
        if m:
            new_clear.discard(m.group(1))
            matched_any = True
            continue
        # 10. Block is now clear
        m = re.search(r"the\s+(\w+)\s+block\s+is\s+(?:now\s+)?clear", c_lower)
        if m:
            new_clear.add(m.group(1))
            matched_any = True
            continue

        # If a clause couldn't be matched, it's an unrecognized physical effect
        raise ValueError(f"Unrecognized transition clause in change text: {clause!r}")

    if not matched_any:
        raise ValueError(f"No recognized state change in text: {change_text!r}")

    # ``clear`` and ``on_table`` are logical consequences of the stack relations:
    # the prompt defines a block as clear iff nothing is on top of it and it is
    # not held. The LLM occasionally omits one of those clauses (for instance it
    # marks the target as no longer clear but forgets that the stacked block is
    # now clear), which used to leave an impossible state and collapse the legal
    # action space. Rebuild the derived sets from the predicted ``on``/``holding``
    # relations, as the paper's state definition requires.
    blocks = (
        set(state.clear)
        | set(state.on_table)
        | set(state.on)
        | set(state.on.values())
        | set(new_clear)
        | set(new_on_table)
        | set(new_on)
        | set(new_on.values())
    )
    if new_holding:
        blocks.add(new_holding)
        # A held block cannot also be on the table or on top of another block.
        new_on.pop(new_holding, None)
        new_on_table.discard(new_holding)
        if new_holding in new_on.values():
            raise ValueError(
                f"The held block {new_holding!r} cannot have blocks on top of it: {new_on!r}"
            )

    for top, bottom in new_on.items():
        if top == bottom:
            raise ValueError(f"A block cannot be stacked on itself: {top!r}")

    for start in new_on:
        chain: set[str] = set()
        current = start
        while current in new_on:
            if current in chain:
                raise ValueError(f"Cyclic block configuration predicted: {new_on!r}")
            chain.add(current)
            current = new_on[current]

    new_clear = {b for b in blocks if b not in new_on.values() and b != new_holding}
    new_on_table = {b for b in blocks if b not in new_on and b != new_holding}

    return BlockState(
        clear=new_clear,
        on_table=new_on_table,
        on=new_on,
        holding=new_holding,
    )


def _action_likelihood(action: BlockAction) -> float:
    """Returns r1 for an action: its normalized likelihood in [0, 1].

    Paper mode stores the softmax-normalized log-probability on the action trace.
    Modes that do not score actions fall back to 1.0, matching the original
    "action is always plausible" behaviour.
    """
    trace = getattr(action, "llm_trace", None) or {}
    probability = trace.get("prob")
    if probability is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(probability)))
    except (TypeError, ValueError):
        return 1.0


def _extract_change_section(text: str) -> str:
    """Return the part of a world-model response that follows the [CHANGE] marker.

    Models sometimes explain the prompt before answering. The marker section is
    the only part that describes the transition.
    """
    marker = text.rfind("[CHANGE")
    if marker < 0:
        return text
    return text[marker + len("[CHANGE") :].lstrip(": ]").strip()


class BlocksworldWorldModel(WorldModel):
    """Simulates transitions and rewards in Blocksworld in hybrid or paper mode."""

    def __init__(
        self,
        goal: Sequence[str] | dict[str, str] | str,
        llm: Optional[BaseLLM] = None,
        mode: str = "hybrid",
        alpha: float = 0.5,
    ) -> None:
        self.goal = goal
        self.llm = llm
        self.mode = mode
        self.alpha = alpha

    def _generate_change(self, prompt: str, temperature: float = 0.8) -> str:
        """Ask the world model for the state change and return the [CHANGE] section.

        The reasoning models spend the whole budget explaining the prompt
        themselves when they reason at high effort, and the answer never reaches
        the ``[CHANGE]`` marker. The update is a one-sentence task, so the
        effort is capped and the marker section is extracted from whatever the
        model wrapped around it.
        """
        change_text = self.llm.generate(
            prompt,
            max_tokens=2048,
            temperature=temperature,
            stop=["[STATE 1]"],
            reasoning_effort=_WORLD_MODEL_REASONING_EFFORT,
        )
        return _extract_change_section(change_text)

    def step(
        self,
        state: BlockState,
        action: BlockAction,
        lightweight: bool = False,
        call_log: list[dict[str, Any]] | None = None,
        reward_sink: dict[str, Any] | None = None,
    ) -> tuple[BlockState, float, bool]:
        """Given current state and action, returns (next_state, immediate_reward, is_terminal).

        The immediate reward follows the paper's Blocksworld reward (Section 4.1):
        R1 is the action likelihood, R2 is goal progress, and R3 is the
        model's self-evaluation of the proposed action. R3 is skipped in rollouts.

        Args:
            lightweight: When True, skip the LLM world model and use the exact
                physics simulator. MCTS rollouts request this, mirroring the
                paper's "lightweight reward" used during simulation.
            call_log: Optional sink receiving one dict per LLM call, so the
                inspector can show the exact world-model prompt and response.
            reward_sink: Optional dict populated with the r1/r2/r3 components
                and the combined reward for the inspector.
        """
        if self.mode == "hybrid" or lightweight:
            next_state = state.apply_action(action)
        elif self.mode == "paper":
            if self.llm is None:
                raise ValueError("LLM backend is required for paper mode world model simulation")
            prompt = build_world_update_prompt(state, action)
            change_text = self._generate_change(prompt)
            entry: dict[str, Any] | None = None
            if call_log is not None:
                entry = {
                    "phase": "world-model",
                    "action": action.to_text(),
                    "prompt": prompt,
                    "response": change_text,
                    "logprob": None,
                    "prob": None,
                    "source": "world-model",
                    "reason": None,
                    "error": None,
                }
                call_log.append(entry)
            try:
                next_state = _apply_text_change(state, change_text)
            except ValueError as exc:
                # The sampled answer sometimes misses the [CHANGE] description.
                # One greedy retry usually recovers it, and it keeps the
                # self-evaluation of this action from being skipped.
                retry_text = self._generate_change(prompt, temperature=0.0)
                if entry is not None:
                    entry["response"] = retry_text
                    entry["reason"] = "Retried with greedy decoding."
                try:
                    next_state = _apply_text_change(state, retry_text)
                except ValueError as retry_exc:
                    if entry is not None:
                        entry["error"] = str(retry_exc)
                    raise retry_exc from exc
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        goal_reward = calculate_goal_reward(next_state, self.goal)
        is_terminal = goal_reward >= 100.0
        action_score = _action_likelihood(action)
        eval_score: float | None = None

        if is_terminal:
            # The paper assigns a "super large reward" when every goal condition holds.
            step_reward = goal_reward
        else:
            eval_score = 0.5 if not lightweight else None
            if not lightweight and self.llm is not None:
                eval_prompt = build_self_eval_prompt(state, action, self.goal)
                p_yes, p_no, eval_response = self_evaluation_result(self.llm, eval_prompt)
                # Paper (Section 3.2): R3 is the next-word probability of " Yes",
                # not a renormalized Yes/No ratio.
                answered = p_yes + p_no > 0.0
                if answered:
                    eval_score = max(0.0, min(1.0, p_yes))
                else:
                    # The model never answered Yes/No, so there is no probability
                    # to read. Fall back to a neutral score instead of using a
                    # meaningless probability, and say so in the log.
                    eval_score = 0.5
                if call_log is not None:
                    action_trace = getattr(action, "llm_trace", None)
                    call_log.append(
                        {
                            "phase": "self-evaluation",
                            "action": action.to_text(),
                            "prompt": eval_prompt,
                            "response": (
                                f"Answer: {eval_response.strip() if eval_response else 'unavailable (no response returned)'}\n"
                                f"P(Yes)={p_yes:.6f}, P(No)={p_no:.6f}"
                                + (
                                    ""
                                    if answered
                                    else "\nNo Yes/No answer in the response: R3 falls back to the neutral 0.5."
                                )
                            ),
                            "logprob": None,
                            "prob": None,
                            "r1": action_score,
                            "r2": goal_reward,
                            "r3": eval_score,
                            "r1_source": action_trace.get("r1_source")
                            if isinstance(action_trace, dict)
                            else None,
                            "p_yes": p_yes,
                            "p_no": p_no,
                            "source": "self-evaluation",
                            "reason": None,
                            "error": None,
                        }
                    )
            step_reward = calculate_combined_reward(
                state=next_state,
                action=action,
                goals=self.goal,
                action_score=action_score,
                alpha=self.alpha,
                eval_score=eval_score,
            )

        if reward_sink is not None:
            reward_sink.update(
                {
                    "r1": float(action_score),
                    "r2": float(goal_reward),
                    "r3": None if eval_score is None else float(eval_score),
                    "reward": float(step_reward),
                }
            )

        return next_state, float(step_reward), is_terminal
