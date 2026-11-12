"""Reasoning agent implementation for Blocksworld.

``paper`` mode follows the RAP paper (Hao et al., EMNLP 2023, Section 4.1):

* the action space is every currently valid action, given the domain rules;
* r1, the "likelihood of the action" reward, is the log-probability of the
  action continuation under the Appendix C.1 prompt;
* the prompt demonstrations are truncated from the beginning as the tree
  deepens (Appendix E, "Adaptive Prompting").

``hybrid`` mode is an extra JSON interface that is not part of the paper.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Any, Sequence

from rap_demo.core.mcts import ReasoningAgent
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState, HybridActionProposal
from rap_demo.domains.blocksworld.prompts import (
    build_hybrid_action_prompt,
    build_paper_action_prompt,
    split_paper_plan_prompt,
)
from rap_demo.llm.base import BaseLLM, ContinuationScorer


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Extracts a JSON object from text, handling markdown code blocks and conversational commentary."""
    # 1. Match code blocks with or without json tag
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Match first balanced or outermost curly-bracket object
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Direct parse of stripped text
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax over action log-probabilities."""
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


class BlocksworldAgent(ReasoningAgent):
    """Generates candidate reasoning steps/actions for Blocksworld in hybrid or paper mode."""

    def __init__(
        self,
        llm: BaseLLM,
        goal: str,
        mode: str = "hybrid",
        num_candidates: int | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.llm = llm
        self.goal = goal
        self.mode = mode
        # ``None`` means "the whole legal action space", as in the paper.
        self.num_candidates = num_candidates
        # Paper Appendix B.1 samples with temperature 0.8; the hybrid JSON
        # interface stays deterministic unless the caller overrides it.
        self.temperature = (0.8 if mode == "paper" else 0.0) if temperature is None else temperature
        self.max_tokens = max_tokens
        self.last_proposals: list[HybridActionProposal] = []

    def get_actions(
        self,
        state: BlockState,
        remaining_steps: int | None = None,
        lightweight: bool = False,
    ) -> list[BlockAction]:
        """Proposes candidate actions from current state.

        Args:
            state: Current Blocksworld configuration.
            remaining_steps: Actions still available before the depth limit. Used by
                the paper prompts for adaptive demonstration truncation (Appendix E).
            lightweight: Skip action scoring (r1). MCTS rollouts request this, as the
                paper discards costly reward components during simulation.
        """
        if self.mode == "hybrid":
            return self._get_hybrid_actions(state)
        elif self.mode == "paper":
            return self._get_paper_actions(state, remaining_steps=remaining_steps, lightweight=lightweight)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Paper mode (RAP Section 4.1 / Appendix C.1 / Appendix E)
    # ------------------------------------------------------------------

    def _get_paper_actions(
        self,
        state: BlockState,
        remaining_steps: int | None = None,
        lightweight: bool = False,
    ) -> list[BlockAction]:
        """Enumerates the legal action space and scores each action with r1 = log P(a | s)."""
        self.last_proposals = []
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return []

        if lightweight:
            # Rollout mode: no LLM calls, r1 is neutral and only r2 guides the rollout.
            trace = {
                "source": "rollout",
                "goal": self.goal,
                "model": getattr(self.llm, "model", None),
                "prompt": None,
                "response": None,
                "proposed": None,
                "reason": "Lightweight rollout: action likelihood skipped.",
                "r1_source": "skipped",
            }
            return [replace(action, llm_trace=dict(trace)) for action in legal_actions]

        prompt = build_paper_action_prompt(state, self.goal, remaining_steps=remaining_steps)
        scored, r1_source = self._score_actions(prompt, legal_actions)

        probabilities = _softmax([logprob for logprob, _ in scored])
        ranked = sorted(
            zip(scored, probabilities),
            key=lambda item: item[0][0],
            reverse=True,
        )

        trace = {
            "source": "model-paper",
            "goal": self.goal,
            "model": getattr(self.llm, "model", None),
            "prompt": prompt,
            "response": None,
            "proposed": None,
            "reason": None,
            "r1_source": r1_source,
        }
        actions: list[BlockAction] = []
        for (logprob, action), probability in ranked:
            actions.append(
                replace(
                    action,
                    llm_trace={**trace, "logprob": logprob, "prob": probability},
                )
            )

        if self.num_candidates is not None:
            actions = actions[: self.num_candidates]
        return actions

    def _score_actions(
        self,
        prompt: str,
        actions: Sequence[BlockAction],
    ) -> tuple[list[tuple[float, BlockAction]], str]:
        """Return the r1 score of every action and the source of those scores.

        The paper's r1 is log P(action | state). Backends able to report
        token-level logprobs score the whole action space in one teacher-forcing
        pass, which is what the reference implementation does.

        Some providers (Ollama's OpenAI-compatible endpoints) cannot return
        continuation logprobs. Every action then scores 0.0 and softmax yields
        a uniform prior. Label that case so the UI can explain the value.
        """
        texts = [action.to_paper_text() for action in actions]
        if isinstance(self.llm, ContinuationScorer):
            context, marker = split_paper_plan_prompt(prompt)
            try:
                scores = self.llm.score_continuations(context, texts, assistant_prefix=marker)
            except Exception:
                scores = None
            if scores:
                values = [float(scores.get(text, 0.0)) for text in texts]
                source = "uniform-fallback" if all(value == 0.0 for value in values) else "logprob"
                return (
                    list(zip(values, actions)),
                    source,
                )

        logprobs = [self._action_logprob(prompt, action) for action in actions]
        source = (
            "uniform-fallback"
            if logprobs and all(logprob == 0.0 for logprob in logprobs)
            else "logprob"
        )
        return list(zip(logprobs, actions)), source

    def _action_logprob(self, prompt: str, action: BlockAction) -> float:
        """Returns r1 for one action, degrading to a uniform score if the backend cannot score."""
        try:
            return float(self.llm.get_logprobs(prompt, action.to_paper_text()))
        except (NotImplementedError, AttributeError):
            return 0.0

    # ------------------------------------------------------------------
    # Hybrid mode (extra, not part of the paper)
    # ------------------------------------------------------------------

    def _get_hybrid_actions(self, state: BlockState) -> list[BlockAction]:
        """Hybrid mode: requests structured JSON with thought, parses action, verifies legality."""
        prompt = build_hybrid_action_prompt(state, self.goal)
        response = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)

        payload = _extract_json_payload(response)
        reason: str | None = None
        if payload is not None:
            try:
                proposal = HybridActionProposal.model_validate(payload)
                self.last_proposals = [proposal]
                action, repaired = self._repair_proposal(proposal)
                if state.is_legal_action(action):
                    reason = repaired
                    trace = self._build_trace(prompt, response, payload, "model", reason)
                    trace.update({"logprob": None, "prob": 1.0})
                    return [replace(action, llm_trace=trace)]
                reason = f"Model proposed an illegal action: {action.to_text()}"
            except Exception as exc:
                reason = f"Model output could not be parsed as a valid proposal: {exc}"
        else:
            reason = "Model output did not contain a JSON object."

        # Fallback to legal actions if parsing fails or proposed action is illegal
        legal_actions = state.get_legal_actions()
        fallback = legal_actions[: self.num_candidates] if legal_actions and self.num_candidates else legal_actions
        trace = self._build_trace(prompt, response, payload, "fallback", reason)
        probability = 1.0 / len(fallback) if fallback else 0.0
        return [
            replace(action, llm_trace={**trace, "logprob": None, "prob": probability})
            for action in fallback
        ]

    @staticmethod
    def _repair_proposal(proposal: HybridActionProposal) -> tuple[BlockAction, str | None]:
        """Fixes the common ``put_down`` + target confusion by turning it into a ``stack``.

        ``put_down`` only releases a block on the table, so a model that says
        "put the orange block down onto the red block" means ``stack``. Accepting
        the raw ``put_down`` would silently drop the target and loop forever.
        """
        action = proposal.to_block_action()
        atype = action.action_type
        atype_value = atype.value if isinstance(atype, ActionType) else str(atype)
        if atype_value == ActionType.PUT_DOWN.value and action.target:
            return (
                BlockAction(
                    action_type=ActionType.STACK,
                    block=action.block,
                    target=action.target,
                    thought=action.thought,
                ),
                f"Repaired put_down with target '{action.target}' into stack (put_down ignores targets).",
            )
        return action, None

    def _build_trace(
        self,
        prompt: str,
        response: str | None,
        proposed: dict[str, Any] | None,
        source: str,
        reason: str | None,
    ) -> dict[str, Any]:
        """Builds a serializable trace of the model call that produced an action."""
        return {
            "source": source,
            "goal": self.goal,
            "model": getattr(self.llm, "model", None),
            "prompt": prompt,
            "response": response,
            "proposed": proposed,
            "reason": reason,
        }
