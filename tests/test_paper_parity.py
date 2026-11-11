"""Parity tests for the RAP paper reconstruction (prompts, rewards, MCTS)."""

from __future__ import annotations

import pytest

from rap_demo.core.mcts import MCTS
from rap_demo.core.node import MCTSNode
from rap_demo.domains.blocksworld.agent import BlocksworldAgent
from rap_demo.domains.blocksworld.models import ActionType, BlockAction, BlockState
from rap_demo.domains.blocksworld.prompts import (
    BLOCKSWORLD_RULES,
    PAPER_PLAN_DEMOS,
    build_paper_action_prompt,
    build_world_update_prompt,
)
from rap_demo.domains.blocksworld.world_model import BlocksworldWorldModel
from rap_demo.llm.mock import MockLLM


GOAL = ["the red block is on top of the blue block", "the orange block is on top of the red block"]

INITIAL = BlockState(
    clear={"red", "blue", "orange"},
    on_table={"yellow", "red", "blue"},
    on={"orange": "yellow"},
    holding=None,
)


class TestPaperActionPrompt:
    """Appendix C.1 action-likelihood prompt."""

    def test_contains_rules_demos_and_open_plan(self) -> None:
        prompt = build_paper_action_prompt(INITIAL, GOAL)

        assert BLOCKSWORLD_RULES in prompt
        assert prompt.count("[STATEMENT]") == len(PAPER_PLAN_DEMOS) + 1
        assert prompt.count("[PLAN END]") == len(PAPER_PLAN_DEMOS)
        assert "As initial conditions I have that," in prompt
        assert "My goal is to have that the red block is on top of the blue block and the orange block is on top of the red block." in prompt
        assert prompt.rstrip().endswith("[PLAN]")

    def test_adaptive_prompting_truncates_demos_with_remaining_steps(self) -> None:
        """Appendix E: demos lose leading actions as the tree deepens."""
        full = build_paper_action_prompt(INITIAL, GOAL, remaining_steps=6)
        shallow = build_paper_action_prompt(INITIAL, GOAL, remaining_steps=2)

        assert full.count("unstack the yellow block from on top of the orange block") > 1
        # The first demo (4 actions) is cut to its last 2 actions at remaining_steps=2.
        assert "put down the yellow block" not in shallow.split("[STATEMENT]")[1]
        assert "stack the orange block on top of the red block" in shallow

    def test_world_prompt_contains_scenarios_and_change_cue(self) -> None:
        prompt = build_world_update_prompt(INITIAL, BlockAction(ActionType.UNSTACK, "orange", "yellow"))

        assert prompt.count("[SCENARIO") == 3
        assert "[STATE 1]" in prompt
        assert prompt.rstrip().endswith("[CHANGE]")
        assert "[ACTION] Unstack the orange block from on top of the yellow block." in prompt


class TestPaperAgent:
    """Section 4.1: action space = all legal actions, scored by r1."""

    def test_returns_entire_legal_action_space_with_normalized_prior(self) -> None:
        llm = MockLLM(default_logprob=-1.0)
        agent = BlocksworldAgent(llm=llm, goal=" ".join(GOAL), mode="paper")

        actions = agent.get_actions(INITIAL)

        legal = INITIAL.get_legal_actions()
        assert len(actions) == len(legal)
        assert {a.to_text() for a in actions} == {a.to_text() for a in legal}
        assert sum(a.llm_trace["prob"] for a in actions) == pytest.approx(1.0)
        assert all(a.llm_trace["source"] == "model-paper" for a in actions)

    def test_lightweight_rollout_skips_logprob_calls(self) -> None:
        llm = MockLLM()
        agent = BlocksworldAgent(llm=llm, goal=" ".join(GOAL), mode="paper")

        actions = agent.get_actions(INITIAL, lightweight=True)

        assert llm.call_count == 0
        assert len(actions) == len(INITIAL.get_legal_actions())
        assert all(a.llm_trace["source"] == "rollout" for a in actions)

    def test_hybrid_repairs_put_down_with_target_into_stack(self) -> None:
        """The orange-loop bug: "put down onto red" must become a stack, not a table drop."""
        import json

        llm = MockLLM(
            default_response=json.dumps(
                {"thought": "put orange onto red", "action": "put_down", "block": "orange", "target": "red"}
            )
        )
        holding_orange = BlockState(
            clear={"red", "yellow"},
            on_table={"yellow", "blue"},
            on={"red": "blue"},
            holding="orange",
        )
        agent = BlocksworldAgent(llm=llm, goal=" ".join(GOAL), mode="hybrid")

        actions = agent.get_actions(holding_orange)

        assert len(actions) == 1
        assert actions[0].action_type == ActionType.STACK
        assert actions[0].target == "red"


class TestPutDownTargetContract:
    """put_down releases a block on the table and must not accept a stack target."""

    def test_put_down_with_target_is_illegal(self) -> None:
        state = BlockState(clear={"red"}, on_table={"blue"}, on={"red": "blue"}, holding="orange")

        assert state.is_legal_action(BlockAction(ActionType.PUT_DOWN, "orange")) is True
        assert state.is_legal_action(BlockAction(ActionType.PUT_DOWN, "orange", "red")) is False

    def test_targeted_put_down_is_rejected_by_physics(self) -> None:
        """The physics simulator enforces the contract that paper mode relies on.

        Paper mode delegates legality to the action enumeration, so the rejection
        lives in ``apply_action`` rather than in ``world_model.step``.
        """
        state = BlockState(clear={"red"}, on_table={"blue"}, on={"red": "blue"}, holding="orange")

        with pytest.raises(ValueError):
            state.apply_action(BlockAction(ActionType.PUT_DOWN, "orange", "red"))


class TestPaperWorldModel:
    """Section 3.1: the LLM predicts transitions, but rollouts use cheap physics."""

    def test_lightweight_step_skips_the_llm(self) -> None:
        llm = MockLLM()
        world_model = BlocksworldWorldModel(goal=GOAL, llm=llm, mode="paper")

        next_state, _, _ = world_model.step(
            INITIAL, BlockAction(ActionType.PICK_UP, "red"), lightweight=True
        )

        assert llm.call_count == 0
        assert next_state.holding == "red"


class TestPaperMCTS:
    """Algorithm 1 details: priors for unvisited children and depth-limit terminals."""

    def test_unvisited_children_are_ranked_by_prior(self) -> None:
        mcts = MCTS()
        parent = MCTSNode(state="P")
        parent.N = 3

        low = MCTSNode(state="low", parent=parent)
        low.prior = 0.1
        high = MCTSNode(state="high", parent=parent)
        high.prior = 0.9
        visited = MCTSNode(state="visited", parent=parent)
        visited.N = 4
        visited.M = 0.5
        visited.Q = 0.5

        assert mcts.uct_value(parent, low) > mcts.uct_value(parent, visited)
        assert mcts.uct_value(parent, high) > mcts.uct_value(parent, low)

    def test_expand_marks_depth_limit_as_terminal(self) -> None:
        class Env:
            def get_actions(self, state, remaining_steps=None):
                return [BlockAction(ActionType.PICK_UP, "red")]

            def step(self, state, action, lightweight=False):
                return state, 1.0, False

        mcts = MCTS(world_model=Env(), reasoning_agent=Env(), max_depth=1)
        root = MCTSNode(state=INITIAL, depth=0)

        children = mcts.expand(root)

        assert len(children) == 1
        assert children[0].is_terminal is True

    def test_expand_records_every_llm_call_for_the_step(self) -> None:
        """The inspector can replay all prompts/responses: N action scores + N world calls."""
        llm = MockLLM(
            default_response=(
                "[CHANGE] the hand was empty and now is holding the red block, "
                "the red block was on the table and now is in the hand, "
                "and the red block is no longer clear."
            ),
            default_logprob=-0.5,
        )
        agent = BlocksworldAgent(llm=llm, goal=" ".join(GOAL), mode="paper")
        world_model = BlocksworldWorldModel(goal=GOAL, llm=llm, mode="paper")
        mcts = MCTS(world_model=world_model, reasoning_agent=agent, max_depth=6)
        root = MCTSNode(state=INITIAL, depth=0)

        mcts.expand(root)

        n_actions = len(INITIAL.get_legal_actions())
        assert len(root.llm_calls) == 2 * n_actions
        phases = [call["phase"] for call in root.llm_calls]
        assert phases[:n_actions] == ["action-scoring"] * n_actions
        assert phases[n_actions:] == ["world-model"] * n_actions
        assert all(call["prompt"] for call in root.llm_calls)
        assert all(call["response"] for call in root.llm_calls)
        assert all(call["error"] is None for call in root.llm_calls)

    def test_expand_logs_world_model_hallucination_with_error(self) -> None:
        """A rejected world-model transition stays visible in the step log."""
        llm = MockLLM(default_response="the blocks vanish into thin air")
        agent = BlocksworldAgent(llm=llm, goal=" ".join(GOAL), mode="paper")
        world_model = BlocksworldWorldModel(goal=GOAL, llm=llm, mode="paper")
        mcts = MCTS(world_model=world_model, reasoning_agent=agent, max_depth=6)
        root = MCTSNode(state=INITIAL, depth=0)

        children = mcts.expand(root)

        # The physics fallback keeps the search alive and every action gets a child.
        assert children
        world_calls = [c for c in root.llm_calls if c["phase"] == "world-model"]
        assert world_calls
        assert all(call["error"] for call in world_calls)

    def test_simulation_reuses_expansion_child(self) -> None:
        """Algorithm 1 line 16: simulation starts from argmax local reward, not a fresh query."""
        calls: list[str] = []

        class Env:
            def get_actions(self, state, remaining_steps=None, lightweight=False):
                calls.append("agent")
                return [BlockAction(ActionType.PICK_UP, "red")]

            def step(self, state, action, lightweight=False):
                calls.append("world")
                return state, 0.5, False

        mcts = MCTS(world_model=Env(), reasoning_agent=Env(), max_depth=2)
        root = MCTSNode(state=INITIAL, depth=0)
        mcts.step_iteration(root)

        # 1 agent call + 1 world call for the expansion, 1 agent + 1 world for one rollout step.
        assert calls.count("agent") == 2
        assert calls.count("world") == 2
        assert root.children
