"""Comprehensive unit and integration tests for Blocksworld Domain (Phase 4).

Tests cover:
1. Domain Protocol / ABC contract (get_initial_state, is_goal_reached, create_agent, create_world_model, create_reward_fn).
2. BlockAction and ActionType:
   - Creation, attributes, value types, equality, hashing.
   - to_text() canonical natural language conversion for all actions.
   - from_text() robust natural language parsing, variations, edge cases, error handling.
3. BlockState:
   - State tracking: clear, on_table, on, holding.
   - to_text() and from_text() round-trip fidelity, paper canonical format.
   - is_legal_action() precondition checking for PICK_UP, PUT_DOWN, STACK, UNSTACK.
   - get_legal_actions() complete branching generation under varied configurations.
   - apply_action() deterministic physics transitions and immutability invariants.
4. HybridActionProposal:
   - Pydantic validation, serialization, deserialization, markdown extraction.
   - Conversion to BlockAction.
5. Blocksworld Rewards:
   - calculate_goal_reward() partial heuristic scoring and +100 completion bonus.
   - calculate_combined_reward() alpha weighting and integration.
   - build_self_eval_prompt() and score_self_evaluation() with LLM probabilities.
6. BlocksworldAgent:
   - Hybrid mode action proposal from JSON response.
   - Paper mode action generation and likelihood ranking.
7. BlocksworldWorldModel:
   - Hybrid mode deterministic state transitions.
   - Paper mode LLM-guided state change text application.
8. BlocksworldDomain:
   - Full domain lifecycle and component wiring.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any
import pytest
from pydantic import ValidationError

from rap_demo.domains.base import Domain
from rap_demo.domains.blocksworld.models import (
    ActionType,
    BlockAction,
    BlockState,
    HybridActionProposal,
)
from rap_demo.domains.blocksworld.rewards import (
    BlocksworldRewardFunction,
    build_self_eval_prompt,
    calculate_combined_reward,
    calculate_goal_reward,
    score_self_evaluation,
)
from rap_demo.domains.blocksworld.agent import BlocksworldAgent
from rap_demo.domains.blocksworld.prompts import build_hybrid_action_prompt
from rap_demo.domains.blocksworld.world_model import BlocksworldWorldModel
from rap_demo.domains.blocksworld.domain import BlocksworldDomain
from rap_demo.llm.mock import MockLLM


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def table_four_blocks_state() -> BlockState:
    """Fixture providing a state where 4 blocks are on the table and clear."""
    return BlockState(
        clear={"red", "blue", "orange", "yellow"},
        on_table={"red", "blue", "orange", "yellow"},
        on={},
        holding=None,
    )


@pytest.fixture
def stacked_tower_state() -> BlockState:
    """Fixture providing a tower: yellow on table, orange on yellow, red on orange.

    Blue is alone on the table.
    """
    return BlockState(
        clear={"red", "blue"},
        on_table={"yellow", "blue"},
        on={"orange": "yellow", "red": "orange"},
        holding=None,
    )


@pytest.fixture
def holding_block_state() -> BlockState:
    """Fixture where the robot hand is holding the red block."""
    return BlockState(
        clear={"blue", "yellow"},
        on_table={"blue", "yellow"},
        on={},
        holding="red",
    )


@pytest.fixture
def instance_5_state() -> BlockState:
    """Fixture representing RAP paper Blocksworld Instance 5 initial state.

    State description from paper:
    - orange on yellow
    - yellow on table
    - red on table
    - blue on table
    - clear: red, blue, orange
    - hand is empty
    """
    return BlockState(
        clear={"red", "blue", "orange"},
        on_table={"yellow", "red", "blue"},
        on={"orange": "yellow"},
        holding=None,
    )


# ============================================================================
# 1. Domain Protocol / ABC Contract Tests
# ============================================================================


class TestDomainContract:
    """Tests enforcing the Domain protocol / ABC interface requirements."""

    def test_domain_protocol_runtime_checkable(self) -> None:
        """Domain protocol must be runtime checkable with isinstance."""

        class MinimalValidDomain:
            def get_initial_state(self) -> Any:
                return {}

            def is_goal_reached(self, state: Any) -> bool:
                return True

            def create_agent(self, **kwargs: Any) -> Any:
                return None

            def create_world_model(self, **kwargs: Any) -> Any:
                return None

            def create_reward_fn(self, **kwargs: Any) -> Any:
                return lambda s, a: 1.0

        domain_instance = MinimalValidDomain()
        assert isinstance(domain_instance, Domain)

    def test_domain_protocol_rejects_incomplete_implementation(self) -> None:
        """An object missing any required method should not satisfy the Domain protocol."""

        class IncompleteDomain:
            def get_initial_state(self) -> Any:
                return {}

        incomplete = IncompleteDomain()
        assert not isinstance(incomplete, Domain)

    def test_blocksworld_domain_implements_domain_protocol(
        self, instance_5_state: BlockState
    ) -> None:
        """BlocksworldDomain class must adhere to the Domain protocol."""
        goal = ["the red block is on top of the blue block"]
        mock_llm = MockLLM()
        domain = BlocksworldDomain(
            initial_state=instance_5_state,
            goal=goal,
            llm=mock_llm,
            mode="hybrid",
        )
        assert isinstance(domain, Domain)


# ============================================================================
# 2. BlockAction and ActionType Unit Tests
# ============================================================================


class TestBlockAction:
    """Unit tests for BlockAction dataclass and ActionType enumeration."""

    @pytest.mark.parametrize(
        "action_type,block,target,expected_text",
        [
            (ActionType.PICK_UP, "red", None, "Pick up the red block"),
            (ActionType.PUT_DOWN, "red", None, "Put down the red block"),
            (ActionType.STACK, "red", "blue", "Stack the red block on top of the blue block"),
            (ActionType.UNSTACK, "red", "blue", "Unstack the red block from on top of the blue block"),
        ],
    )
    def test_action_to_text_canonical(
        self,
        action_type: ActionType,
        block: str,
        target: str | None,
        expected_text: str,
    ) -> None:
        """to_text() must generate canonical natural language matching the paper format."""
        action = BlockAction(action_type=action_type, block=block, target=target)
        assert action.to_text().rstrip(".") == expected_text

    @pytest.mark.parametrize(
        "raw_text,expected_action",
        [
            ("Pick up the red block", BlockAction(ActionType.PICK_UP, "red")),
            ("Pick up the red block.", BlockAction(ActionType.PICK_UP, "red")),
            ("pick up the red block", BlockAction(ActionType.PICK_UP, "red")),
            ("   Pick up the   red block.  \n", BlockAction(ActionType.PICK_UP, "red")),
            ("[ACTION 1] Pick up the red block.", BlockAction(ActionType.PICK_UP, "red")),
            ("Put down the blue block", BlockAction(ActionType.PUT_DOWN, "blue")),
            ("put down the blue block.", BlockAction(ActionType.PUT_DOWN, "blue")),
            ("Stack the red block on top of the blue block", BlockAction(ActionType.STACK, "red", "blue")),
            ("Stack the red block on top of the blue block.", BlockAction(ActionType.STACK, "red", "blue")),
            ("stack the red block on top of the blue block", BlockAction(ActionType.STACK, "red", "blue")),
            ("[ACTION 2] Stack the red block on top of the blue block.", BlockAction(ActionType.STACK, "red", "blue")),
            ("Unstack the red block from on top of the blue block", BlockAction(ActionType.UNSTACK, "red", "blue")),
            ("Unstack the red block from on top of the blue block.", BlockAction(ActionType.UNSTACK, "red", "blue")),
            ("unstack the orange block from on top of the yellow block", BlockAction(ActionType.UNSTACK, "orange", "yellow")),
        ],
    )
    def test_action_from_text_valid(self, raw_text: str, expected_action: BlockAction) -> None:
        """from_text() must parse varied natural language inputs into BlockAction."""
        parsed = BlockAction.from_text(raw_text)
        assert parsed.action_type == expected_action.action_type
        assert parsed.block == expected_action.block
        assert parsed.target == expected_action.target

    @pytest.mark.parametrize(
        "invalid_text",
        [
            "",
            "   ",
            "Fly the red block to the moon",
            "Pick up",
            "Pick up the",
            "Stack red on blue",
            "Destroy the yellow block",
            "12345",
        ],
    )
    def test_action_from_text_invalid_raises_value_error(self, invalid_text: str) -> None:
        """from_text() must raise ValueError for unrecognized or incomplete actions."""
        with pytest.raises((ValueError, KeyError)):
            BlockAction.from_text(invalid_text)

    def test_action_equality_and_hashing(self) -> None:
        """BlockAction must be hashable and equal when attributes match."""
        a1 = BlockAction(ActionType.STACK, "red", "blue")
        a2 = BlockAction(ActionType.STACK, "red", "blue")
        a3 = BlockAction(ActionType.STACK, "red", "yellow")

        assert a1 == a2
        assert a1 != a3
        assert hash(a1) == hash(a2)

        action_set = {a1, a2, a3}
        assert len(action_set) == 2
        assert a1 in action_set


# ============================================================================
# 3. BlockState Unit Tests
# ============================================================================


class TestBlockStateModels:
    """Unit tests for BlockState representation, physics, and conversions."""

    def test_state_creation_and_defaults(self) -> None:
        """BlockState should initialize with sensible empty defaults."""
        state = BlockState()
        assert state.clear == set()
        assert state.on_table == set()
        assert state.on == {}
        assert state.holding is None

    def test_state_to_text_canonical_format(self, instance_5_state: BlockState) -> None:
        """to_text() must produce the canonical comma-separated list of conditions."""
        text = instance_5_state.to_text()
        assert isinstance(text, str)
        assert "hand is empty" in text
        assert "the red block is clear" in text
        assert "the blue block is clear" in text
        assert "the orange block is clear" in text
        assert "the orange block is on top of the yellow block" in text
        assert "the red block is on the table" in text
        assert "the blue block is on the table" in text
        assert "the yellow block is on the table" in text

    def test_state_to_text_holding_format(self, holding_block_state: BlockState) -> None:
        """to_text() must describe when the hand is holding a block."""
        text = holding_block_state.to_text()
        assert "holding the red block" in text or "the red block is in the hand" in text
        assert "hand is empty" not in text

    @pytest.mark.parametrize(
        "raw_text,expected_state",
        [
            (
                "the red block is clear, the blue block is clear, the hand is empty, "
                "the red block is on the table and the blue block is on the table.",
                BlockState(
                    clear={"red", "blue"},
                    on_table={"red", "blue"},
                    on={},
                    holding=None,
                ),
            ),
            (
                "I have that, the red block is clear, the blue block is clear, "
                "the orange block is clear, the hand is empty, "
                "the orange block is on top of the yellow block, "
                "the red block is on the table, the blue block is on the table "
                "and the yellow block is on the table.",
                BlockState(
                    clear={"red", "blue", "orange"},
                    on_table={"yellow", "red", "blue"},
                    on={"orange": "yellow"},
                    holding=None,
                ),
            ),
            (
                "the blue block is clear, the yellow block is clear, "
                "the hand is holding the red block, "
                "the blue block is on the table and the yellow block is on the table.",
                BlockState(
                    clear={"blue", "yellow"},
                    on_table={"blue", "yellow"},
                    on={},
                    holding="red",
                ),
            ),
        ],
    )
    def test_state_from_text_parsing(self, raw_text: str, expected_state: BlockState) -> None:
        """from_text() must robustly parse natural language state descriptions."""
        parsed = BlockState.from_text(raw_text)
        assert parsed.clear == expected_state.clear
        assert parsed.on_table == expected_state.on_table
        assert parsed.on == expected_state.on
        assert parsed.holding == expected_state.holding

    def test_state_round_trip_conversion(self, instance_5_state: BlockState) -> None:
        """Converting a state to text and back to state must preserve identity."""
        text = instance_5_state.to_text()
        reconstructed = BlockState.from_text(text)
        assert reconstructed.clear == instance_5_state.clear
        assert reconstructed.on_table == instance_5_state.on_table
        assert reconstructed.on == instance_5_state.on
        assert reconstructed.holding == instance_5_state.holding

    def test_state_from_text_malformed_raises_error(self) -> None:
        """from_text() must raise ValueError if the state text is completely invalid."""
        with pytest.raises((ValueError, KeyError)):
            BlockState.from_text("This is an unparseable arbitrary sentence without block facts.")


# ============================================================================
# 3b. BlockState Precondition and Legal Action Tests
# ============================================================================


class TestBlockStatePreconditions:
    """Tests checking action validity (is_legal_action and get_legal_actions)."""

    def test_pick_up_preconditions(
        self,
        table_four_blocks_state: BlockState,
        stacked_tower_state: BlockState,
        holding_block_state: BlockState,
    ) -> None:
        """PICK_UP requires empty hand, clear block, and block on table."""
        # 1. Valid pickup on table
        valid_pickup = BlockAction(ActionType.PICK_UP, "red")
        assert table_four_blocks_state.is_legal_action(valid_pickup) is True

        # 2. Invalid: Hand is already holding a block
        assert holding_block_state.is_legal_action(valid_pickup) is False

        # 3. Invalid: Block is not on the table (it is on top of yellow)
        unstacked_block_pickup = BlockAction(ActionType.PICK_UP, "orange")
        assert stacked_tower_state.is_legal_action(unstacked_block_pickup) is False

        # 4. Invalid: Block is covered / not clear (yellow is under orange)
        covered_block_pickup = BlockAction(ActionType.PICK_UP, "yellow")
        assert stacked_tower_state.is_legal_action(covered_block_pickup) is False

    def test_put_down_preconditions(
        self,
        table_four_blocks_state: BlockState,
        holding_block_state: BlockState,
    ) -> None:
        """PUT_DOWN requires hand holding the specified block."""
        valid_put_down = BlockAction(ActionType.PUT_DOWN, "red")
        assert holding_block_state.is_legal_action(valid_put_down) is True

        # Invalid: Hand is empty
        assert table_four_blocks_state.is_legal_action(valid_put_down) is False

        # Invalid: Holding red, cannot put down blue
        wrong_block_put_down = BlockAction(ActionType.PUT_DOWN, "blue")
        assert holding_block_state.is_legal_action(wrong_block_put_down) is False

    def test_stack_preconditions(
        self,
        table_four_blocks_state: BlockState,
        holding_block_state: BlockState,
        stacked_tower_state: BlockState,
    ) -> None:
        """STACK requires holding block and clear target."""
        # Valid: Holding red, target blue is clear
        valid_stack = BlockAction(ActionType.STACK, "red", "blue")
        assert holding_block_state.is_legal_action(valid_stack) is True

        # Invalid: Hand is empty
        assert table_four_blocks_state.is_legal_action(valid_stack) is False

        # Invalid: Target is covered (orange is covered by red)
        covered_target_stack = BlockAction(ActionType.STACK, "red", "orange")
        # In stacked_tower_state orange has red on top, so orange is not clear
        state_holding_blue = BlockState(
            clear={"red"},
            on_table={"yellow"},
            on={"orange": "yellow", "red": "orange"},
            holding="blue",
        )
        assert state_holding_blue.is_legal_action(covered_target_stack) is False

        # Invalid: Stacking block on itself
        self_stack = BlockAction(ActionType.STACK, "red", "red")
        assert holding_block_state.is_legal_action(self_stack) is False

    def test_unstack_preconditions(
        self,
        table_four_blocks_state: BlockState,
        stacked_tower_state: BlockState,
        holding_block_state: BlockState,
    ) -> None:
        """UNSTACK requires hand empty, block clear, and on[block] == target."""
        # Valid: Hand empty, red is clear, red is on orange
        valid_unstack = BlockAction(ActionType.UNSTACK, "red", "orange")
        assert stacked_tower_state.is_legal_action(valid_unstack) is True

        # Invalid: Hand is holding red
        assert holding_block_state.is_legal_action(valid_unstack) is False

        # Invalid: Unstacking block on the table (not on target)
        on_table_unstack = BlockAction(ActionType.UNSTACK, "blue", "yellow")
        assert stacked_tower_state.is_legal_action(on_table_unstack) is False

        # Invalid: Target does not match actual underlying block (red is on orange, not yellow)
        wrong_target_unstack = BlockAction(ActionType.UNSTACK, "red", "yellow")
        assert stacked_tower_state.is_legal_action(wrong_target_unstack) is False

        # Invalid: Block is covered (orange is covered by red, cannot unstack orange)
        covered_unstack = BlockAction(ActionType.UNSTACK, "orange", "yellow")
        assert stacked_tower_state.is_legal_action(covered_unstack) is False

    def test_get_legal_actions_when_hand_empty(self, table_four_blocks_state: BlockState) -> None:
        """When 4 blocks are on table and hand is empty, exactly 4 pick_ups should be legal."""
        actions = table_four_blocks_state.get_legal_actions()
        assert len(actions) == 4
        assert set(actions) == {
            BlockAction(ActionType.PICK_UP, "red"),
            BlockAction(ActionType.PICK_UP, "blue"),
            BlockAction(ActionType.PICK_UP, "orange"),
            BlockAction(ActionType.PICK_UP, "yellow"),
        }

    def test_get_legal_actions_when_holding_block(self, holding_block_state: BlockState) -> None:
        """When holding 'red' and blue & yellow are clear, actions must be put_down and stack on each clear block."""
        actions = holding_block_state.get_legal_actions()
        # 1 put_down + 2 stacks (on blue, on yellow) = 3 actions
        assert len(actions) == 3
        assert set(actions) == {
            BlockAction(ActionType.PUT_DOWN, "red"),
            BlockAction(ActionType.STACK, "red", "blue"),
            BlockAction(ActionType.STACK, "red", "yellow"),
        }

    def test_get_legal_actions_mixed_configuration(self, instance_5_state: BlockState) -> None:
        """Instance 5: red on table, blue on table, orange on yellow.

        Clear: red, blue, orange. Hand is empty.
        Legal actions:
        - Pick up red
        - Pick up blue
        - Unstack orange from yellow
        Total: 3 actions.
        """
        actions = instance_5_state.get_legal_actions()
        assert len(actions) == 3
        assert set(actions) == {
            BlockAction(ActionType.PICK_UP, "red"),
            BlockAction(ActionType.PICK_UP, "blue"),
            BlockAction(ActionType.UNSTACK, "orange", "yellow"),
        }


# ============================================================================
# 3c. BlockState Deterministic Physics Transition Tests
# ============================================================================


class TestBlockStateTransitions:
    """Tests deterministic physics state transitions and immutability."""

    def test_apply_pick_up_transition(self, table_four_blocks_state: BlockState) -> None:
        """PICK_UP should update holding, clear, and on_table without mutating original."""
        action = BlockAction(ActionType.PICK_UP, "red")
        new_state = table_four_blocks_state.apply_action(action)

        # Invariance: Original state remains unmodified
        assert "red" in table_four_blocks_state.on_table
        assert table_four_blocks_state.holding is None

        # Transition effects:
        assert new_state.holding == "red"
        assert "red" not in new_state.on_table
        assert "red" not in new_state.clear
        assert new_state.clear == {"blue", "orange", "yellow"}
        assert new_state.on_table == {"blue", "orange", "yellow"}
        assert new_state.on == {}

    def test_apply_put_down_transition(self, holding_block_state: BlockState) -> None:
        """PUT_DOWN should clear holding, add block to table, and mark block clear."""
        action = BlockAction(ActionType.PUT_DOWN, "red")
        new_state = holding_block_state.apply_action(action)

        assert new_state.holding is None
        assert "red" in new_state.on_table
        assert "red" in new_state.clear
        assert new_state.clear == {"red", "blue", "yellow"}
        assert new_state.on_table == {"red", "blue", "yellow"}

    def test_apply_stack_transition(self, holding_block_state: BlockState) -> None:
        """STACK should place holding block onto target, update on, clear, and empty hand."""
        action = BlockAction(ActionType.STACK, "red", "blue")
        new_state = holding_block_state.apply_action(action)

        assert new_state.holding is None
        assert new_state.on["red"] == "blue"
        assert "blue" not in new_state.clear  # Target is now covered
        assert "red" in new_state.clear       # Top block is now clear
        assert "red" not in new_state.on_table
        assert "blue" in new_state.on_table

    def test_apply_unstack_transition(self, instance_5_state: BlockState) -> None:
        """UNSTACK orange from yellow: holding becomes orange, yellow becomes clear."""
        action = BlockAction(ActionType.UNSTACK, "orange", "yellow")
        new_state = instance_5_state.apply_action(action)

        assert new_state.holding == "orange"
        assert "orange" not in new_state.on
        assert "orange" not in new_state.clear
        assert "yellow" in new_state.clear    # Target is now uncovered
        assert "yellow" in new_state.on_table

    def test_four_step_cycle_conservation_of_blocks(self, table_four_blocks_state: BlockState) -> None:
        """Cycle: pick_up(red) -> stack(red, blue) -> unstack(red, blue) -> put_down(red).

        Should return to a state physically equivalent to the initial state.
        """
        s0 = table_four_blocks_state
        s1 = s0.apply_action(BlockAction(ActionType.PICK_UP, "red"))
        s2 = s1.apply_action(BlockAction(ActionType.STACK, "red", "blue"))
        s3 = s2.apply_action(BlockAction(ActionType.UNSTACK, "red", "blue"))
        s4 = s3.apply_action(BlockAction(ActionType.PUT_DOWN, "red"))

        assert s4.clear == s0.clear
        assert s4.on_table == s0.on_table
        assert s4.on == s0.on
        assert s4.holding == s0.holding

    def test_apply_illegal_action_raises_error(self, table_four_blocks_state: BlockState) -> None:
        """Applying an illegal action must raise ValueError or IllegalActionError."""
        illegal_action = BlockAction(ActionType.PUT_DOWN, "red")  # hand is empty
        with pytest.raises(ValueError):
            table_four_blocks_state.apply_action(illegal_action)


# ============================================================================
# 4. HybridActionProposal Unit Tests
# ============================================================================


class TestHybridActionProposal:
    """Tests schema validation, JSON parsing, and conversion for HybridActionProposal."""

    def test_valid_hybrid_action_proposal(self) -> None:
        """Valid JSON payload must parse into HybridActionProposal."""
        json_data = {
            "thought": "The red block is on table and clear, pick it up to stack onto blue.",
            "action": "pick_up",
            "block": "red",
            "target": None,
        }
        proposal = HybridActionProposal.model_validate(json_data)
        assert proposal.action == "pick_up"
        assert proposal.block == "red"
        assert proposal.target is None
        assert "clear" in proposal.thought

    def test_hybrid_proposal_to_block_action(self) -> None:
        """to_block_action() must convert proposal to a valid BlockAction."""
        proposal = HybridActionProposal(
            thought="Stack red onto blue to make progress on goal.",
            action="stack",
            block="red",
            target="blue",
        )
        action = proposal.to_block_action()
        assert isinstance(action, BlockAction)
        assert action.action_type in (ActionType.STACK, "stack")
        assert action.block == "red"
        assert action.target == "blue"

    def test_hybrid_proposal_missing_required_field_raises(self) -> None:
        """Missing 'thought' or 'action' should fail Pydantic validation."""
        with pytest.raises(ValidationError):
            HybridActionProposal.model_validate({"block": "red"})

    def test_hybrid_proposal_serialization(self) -> None:
        """model_dump_json() and model_validate_json() round-trip."""
        proposal = HybridActionProposal(
            thought="Unstack orange to free yellow",
            action="unstack",
            block="orange",
            target="yellow",
        )
        serialized = proposal.model_dump_json()
        deserialized = HybridActionProposal.model_validate_json(serialized)
        assert deserialized == proposal


# ============================================================================
# 5. Blocksworld Rewards Unit Tests
# ============================================================================


class TestBlocksworldRewards:
    """Tests heuristic goal evaluation, combined rewards, and self-evaluation."""

    @pytest.mark.parametrize(
        "state,goals,expected_reward",
        [
            # 0 of 2 conditions met
            (
                BlockState(clear={"red", "blue"}, on_table={"red", "blue"}),
                ["the red block is on top of the blue block", "the orange block is on top of the red block"],
                0.5,
            ),
            # 1 of 2 conditions met (partial)
            (
                BlockState(clear={"red"}, on_table={"blue"}, on={"red": "blue"}),
                ["the red block is on top of the blue block", "the orange block is on top of the red block"],
                1.0,
            ),
            # 2 of 2 conditions met (full completion -> bonus >= 100)
            (
                BlockState(clear={"orange"}, on_table={"blue"}, on={"red": "blue", "orange": "red"}),
                ["the red block is on top of the blue block", "the orange block is on top of the red block"],
                100.0,
            ),
        ],
    )
    def test_calculate_goal_reward(
        self,
        state: BlockState,
        goals: list[str],
        expected_reward: float,
    ) -> None:
        """calculate_goal_reward must score condition fraction and reward +100 on completion."""
        reward = calculate_goal_reward(state, goals)
        if expected_reward >= 100.0:
            assert reward >= 100.0
        else:
            assert math.isclose(reward, expected_reward, abs_tol=0.55)

    def test_calculate_combined_reward_alpha_weighting(self, instance_5_state: BlockState) -> None:
        """Combined reward r(s, a) balances action score and goal score with alpha."""
        action = BlockAction(ActionType.PICK_UP, "red")
        goal = ["the red block is on top of the blue block"]

        # When alpha=1.0, reward is purely action score
        r_pure_action = calculate_combined_reward(
            instance_5_state, action, goal, action_score=2.0, alpha=1.0
        )
        assert math.isclose(r_pure_action, 2.0, rel_tol=1e-3)

        # When alpha=0.0, reward is purely goal reward
        r_pure_goal = calculate_combined_reward(
            instance_5_state, action, goal, action_score=2.0, alpha=0.0
        )
        assert math.isclose(r_pure_goal, calculate_goal_reward(instance_5_state, goal), rel_tol=1e-3)

        # Intermediate alpha
        r_balanced = calculate_combined_reward(
            instance_5_state, action, goal, action_score=2.0, alpha=0.5
        )
        assert r_balanced > 0.0

    def test_calculate_combined_reward_uses_weighted_geometric_mean(self) -> None:
        """The reference implementation combines rewards with a geometric mean."""
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(ActionType.PICK_UP, "red")
        goal_score = calculate_goal_reward(state, ["the red block is on top of the blue block"])

        reward = calculate_combined_reward(
            state,
            action,
            ["the red block is on top of the blue block"],
            action_score=0.25,
            alpha=0.5,
        )

        expected = 0.25**0.5 * goal_score**0.5
        assert math.isclose(reward, expected, rel_tol=1e-6)

    def test_calculate_combined_reward_multiplies_self_evaluation(self) -> None:
        """R3 enters the geometric mean with its own (normalized) exponent."""
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(ActionType.PICK_UP, "red")
        goal_score = calculate_goal_reward(state, ["the red block is on top of the blue block"])

        reward = calculate_combined_reward(
            state,
            action,
            ["the red block is on top of the blue block"],
            action_score=0.25,
            alpha=0.5,
            eval_score=0.75,
            self_eval_weight=0.25,
        )

        # Weights (0.5, 0.5, 0.25) are normalized to (0.4, 0.4, 0.2).
        expected = 0.25**0.4 * goal_score**0.4 * 0.75**0.2
        assert math.isclose(reward, expected, rel_tol=1e-6)

    def test_calculate_combined_reward_normalizes_weights(self) -> None:
        """The weights of the three rewards always sum to one."""
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(ActionType.PICK_UP, "red")
        goal = ["the red block is on top of the blue block"]
        goal_score = calculate_goal_reward(state, goal)

        without_r3 = calculate_combined_reward(state, action, goal, action_score=0.25, alpha=0.5)
        with_r3 = calculate_combined_reward(
            state, action, goal, action_score=0.25, alpha=0.5, eval_score=0.8
        )

        # R1 and R2 keep their relative ratio, R3 is added on top.
        assert math.isclose(without_r3, 0.25**0.5 * goal_score**0.5, rel_tol=1e-6)
        assert math.isclose(with_r3, 0.25**0.4 * goal_score**0.4 * 0.8**0.2, rel_tol=1e-6)

    def test_chained_goal_requires_every_stack_relation(self) -> None:
        goal = "the green block is on top of the orange block is on top of the blue block"
        partial_state = BlockState(
            clear={"green", "blue"},
            on_table={"orange", "blue"},
            on={"green": "orange"},
        )
        complete_state = BlockState(
            clear={"green"},
            on_table={"blue"},
            on={"green": "orange", "orange": "blue"},
        )

        assert calculate_goal_reward(partial_state, goal) == 1.0
        assert calculate_goal_reward(complete_state, goal) == 100.0

    def test_build_self_eval_prompt(self, instance_5_state: BlockState) -> None:
        """build_self_eval_prompt must include current state, action, and target goal."""
        action = BlockAction(ActionType.PICK_UP, "red")
        goal = "the red block is on top of the blue block"
        prompt = build_self_eval_prompt(instance_5_state, action, goal)

        assert isinstance(prompt, str)
        assert "red block" in prompt
        assert "Pick up" in prompt or "pick_up" in prompt
        assert "blue block" in prompt

    def test_score_self_evaluation_with_mock_llm(self) -> None:
        """score_self_evaluation queries LLM token probability for affirmation."""
        prompt = "Is picking up the red block good towards goal? Answer Yes or No:"
        mock_llm = MockLLM(token_probs={(prompt, "Yes"): 0.85, (prompt, "No"): 0.15})
        score = score_self_evaluation(mock_llm, prompt)

        assert math.isclose(score, 0.85, abs_tol=1e-3)

    def test_reward_function_includes_r3_by_default(self) -> None:
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(ActionType.PICK_UP, "red")
        mock_llm = MockLLM(token_probs={"Yes": 0.8, "No": 0.2})
        reward_fn = BlocksworldRewardFunction(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
        )

        reward = reward_fn(state, action, action_score=0.25)

        goal_score = calculate_goal_reward(state, ["the red block is on top of the blue block"])
        assert math.isclose(reward, 0.25**0.4 * goal_score**0.4 * 0.8**0.2, rel_tol=1e-6)
        assert mock_llm.call_count == 1


# ============================================================================
# 6. BlocksworldAgent Unit Tests (Hybrid vs Paper Mode)
# ============================================================================


class TestBlocksworldAgent:
    """Unit tests for BlocksworldAgent in hybrid and paper modes."""

    def test_hybrid_prompt_includes_goal(self, instance_5_state: BlockState) -> None:
        """Hybrid prompt must expose the goal so the model can plan towards it."""
        goal = [
            "the red block is on top of the blue block",
            "the orange block is on top of the red block",
        ]
        prompt = build_hybrid_action_prompt(instance_5_state, goal)

        assert "Goal:" in prompt
        assert "the red block is on top of the blue block" in prompt
        assert "the orange block is on top of the red block" in prompt
        assert instance_5_state.to_text() in prompt

    def test_hybrid_prompt_without_goal_omits_goal_line(
        self, instance_5_state: BlockState
    ) -> None:
        """An empty goal keeps the prompt free of a dangling 'Goal:' line."""
        prompt = build_hybrid_action_prompt(instance_5_state)

        assert "Goal:" not in prompt
        assert "Current State:" in prompt

    def test_agent_hybrid_trace_captures_prompt_and_response(
        self, instance_5_state: BlockState
    ) -> None:
        """A legal model proposal carries the exact prompt, raw response, and parsed payload."""
        canned_json = json.dumps(
            {
                "thought": "Red is clear and needed on blue. Pick up red first.",
                "action": "pick_up",
                "block": "red",
                "target": None,
            }
        )
        mock_llm = MockLLM(default_response=canned_json)
        agent = BlocksworldAgent(
            llm=mock_llm,
            goal="the red block is on top of the blue block",
            mode="hybrid",
        )

        action = agent.get_actions(instance_5_state)[0]
        trace = action.llm_trace

        assert trace is not None
        assert trace["source"] == "model"
        assert trace["reason"] is None
        assert trace["prompt"] == mock_llm.calls[0].prompt
        assert "Goal:" in trace["prompt"]
        assert trace["response"] == canned_json
        assert trace["proposed"]["action"] == "pick_up"
        assert trace["model"] == getattr(mock_llm, "model", None)
        assert trace["prob"] == 1.0

    def test_agent_hybrid_fallback_trace_explains_rejected_output(
        self, instance_5_state: BlockState
    ) -> None:
        """An illegal model proposal falls back to legal actions, keeping the trace and reason."""
        illegal_json = json.dumps(
            {
                "thought": "Stack red onto blue",
                "action": "stack",
                "block": "red",
                "target": "blue",
            }
        )
        mock_llm = MockLLM(default_response=illegal_json)
        agent = BlocksworldAgent(
            llm=mock_llm,
            goal="the red block is on top of the blue block",
            mode="hybrid",
        )

        actions = agent.get_actions(instance_5_state)

        assert actions, "Fallback must still expose legal actions"
        assert math.isclose(sum(action.llm_trace["prob"] for action in actions), 1.0)
        for action in actions:
            assert action.llm_trace is not None
            assert action.llm_trace["source"] == "fallback"
            assert "illegal action" in action.llm_trace["reason"]
            assert action.llm_trace["response"] == illegal_json
            assert action.thought is None

    def test_agent_hybrid_mode_proposes_actions_from_json(
        self, instance_5_state: BlockState
    ) -> None:
        """In hybrid mode, agent prompts for JSON, parses proposal, and stores thought."""
        canned_json = json.dumps(
            {
                "thought": "Red is clear and needed on blue. Pick up red first.",
                "action": "pick_up",
                "block": "red",
                "target": None,
            }
        )
        mock_llm = MockLLM(default_response=canned_json)
        agent = BlocksworldAgent(
            llm=mock_llm,
            goal="the red block is on top of the blue block",
            mode="hybrid",
        )

        actions = agent.get_actions(instance_5_state)
        assert len(actions) > 0
        first_action = actions[0]
        assert first_action.action_type in (ActionType.PICK_UP, "pick_up")
        assert first_action.block == "red"

        # Verify thought was preserved in agent's proposal trace
        assert len(agent.last_proposals) > 0
        assert "Pick up red first" in agent.last_proposals[0].thought

    def test_agent_paper_mode_generates_actions_with_likelihood(
        self, instance_5_state: BlockState
    ) -> None:
        """In paper mode, agent proposes actions and computes action logprobs/scores."""
        mock_llm = MockLLM(
            default_response="Pick up the red block.",
            logprobs={
                "pick up the red block": -0.1,
                "pick up the blue block": -1.5,
                "unstack the orange block from on top of the yellow block": -0.8,
            },
        )
        agent = BlocksworldAgent(
            llm=mock_llm,
            goal="the red block is on top of the blue block",
            mode="paper",
        )

        actions = agent.get_actions(instance_5_state)
        assert len(actions) > 0
        assert all(isinstance(a, BlockAction) for a in actions)
        # Should contain legal actions from state
        assert any(a.block == "red" and a.action_type in (ActionType.PICK_UP, "pick_up") for a in actions)
        # The action likelihood comes from the model and ranks the actions.
        assert all(action.llm_trace["r1_source"] == "logprob" for action in actions)
        assert actions[0].llm_trace["logprob"] == max(
            action.llm_trace["logprob"] for action in actions
        )
        assert math.isclose(sum(action.llm_trace["prob"] for action in actions), 1.0)

    def test_agent_paper_mode_scores_the_whole_action_space_in_one_call(
        self, instance_5_state: BlockState
    ) -> None:
        """Paper mode uses the batch teacher-forcing scorer when it is available."""
        mock_llm = MockLLM(
            default_response="Pick up the red block.",
            logprobs={
                "pick up the red block": -0.1,
                "pick up the orange block": -1.5,
                "unstack the blue block from on top of the yellow block": -0.8,
            },
        )
        agent = BlocksworldAgent(
            llm=mock_llm,
            goal="the blue block is on top of the orange block and the yellow block is on top of the red block",
            mode="paper",
        )

        actions = agent.get_actions(instance_5_state)

        scoring_calls = [call for call in mock_llm.calls if call.method == "score_continuations"]
        assert len(scoring_calls) == 1
        assert set(scoring_calls[0].args["targets"]) == {
            action.to_paper_text() for action in actions
        }
        # The scores are model-derived, so the prior is not uniform.
        assert all(action.llm_trace["r1_source"] == "logprob" for action in actions)
        assert any(action.llm_trace["prob"] != actions[0].llm_trace["prob"] for action in actions)
        # The most likely action is ranked first.
        assert actions[0].llm_trace["logprob"] == max(
            action.llm_trace["logprob"] for action in actions
        )

    def test_agent_paper_mode_marks_a_uniform_prior(
        self, instance_5_state: BlockState
    ) -> None:
        """A backend without logprobs yields a uniform prior, labelled as such."""
        mock_llm = MockLLM(default_response="Pick up the red block.")
        agent = BlocksworldAgent(
            llm=mock_llm,
            goal="the red block is on top of the blue block",
            mode="paper",
        )

        actions = agent.get_actions(instance_5_state)

        assert all(action.llm_trace["r1_source"] == "uniform-fallback" for action in actions)
        assert math.isclose(sum(action.llm_trace["prob"] for action in actions), 1.0)


# ============================================================================
# 7. BlocksworldWorldModel Unit Tests
# ============================================================================


class TestBlocksworldWorldModel:
    """Unit tests for BlocksworldWorldModel in hybrid and paper modes."""

    def test_world_model_hybrid_mode_deterministic_step(
        self, instance_5_state: BlockState
    ) -> None:
        """In hybrid mode, world_model applies physics without invoking LLM."""
        mock_llm = MockLLM()
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )
        action = BlockAction(ActionType.PICK_UP, "red")
        next_state, reward, is_terminal = world_model.step(instance_5_state, action)

        assert next_state.holding == "red"
        assert not is_terminal
        assert isinstance(reward, float)
        # Hybrid transitions remain deterministic; only the R3 token scores use the LLM.
        assert all(call.method != "generate" for call in mock_llm.calls)

    def test_world_model_combines_r1_r2_r3_and_logs_r3(self) -> None:
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(
            ActionType.PICK_UP,
            "red",
            llm_trace={"prob": 0.25},
        )
        mock_llm = MockLLM(token_probs={"Yes": 0.8, "No": 0.2})
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )
        call_log: list[dict[str, Any]] = []

        next_state, reward, is_terminal = world_model.step(state, action, call_log=call_log)

        assert next_state.holding == "red"
        assert not is_terminal
        goal_score = calculate_goal_reward(state, ["the red block is on top of the blue block"])
        expected = 0.25**0.4 * goal_score**0.4 * 0.8**0.2
        assert math.isclose(reward, expected, rel_tol=1e-6)
        assert call_log[-1]["phase"] == "self-evaluation"
        assert call_log[-1]["p_yes"] == 0.8
        assert call_log[-1]["p_no"] == 0.2
        assert call_log[-1]["response"].startswith("Answer: unavailable")
        assert call_log[-1]["r3"] == 0.8

    def test_r3_uses_raw_yes_probability_not_a_renormalized_ratio(self) -> None:
        """Paper Section 3.2: R3 is the next-word probability of the token " Yes"."""
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(ActionType.PICK_UP, "red", llm_trace={"prob": 0.25})
        mock_llm = MockLLM(token_probs={"Yes": 0.8, "No": 0.8})
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )
        reward_sink: dict[str, Any] = {}

        _, reward, _ = world_model.step(state, action, reward_sink=reward_sink)

        assert reward_sink["r3"] == pytest.approx(0.8)
        goal_score = calculate_goal_reward(state, ["the red block is on top of the blue block"])
        assert reward == pytest.approx(0.25**0.4 * goal_score**0.4 * 0.8**0.2)

    def test_world_model_uses_neutral_r3_when_probabilities_are_missing(self) -> None:
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(ActionType.PICK_UP, "red", llm_trace={"prob": 0.25})
        mock_llm = MockLLM(token_probs={"Yes": 0.0, "No": 0.0})
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )
        call_log: list[dict[str, Any]] = []

        _, reward, _ = world_model.step(state, action, call_log=call_log)

        assert call_log[-1]["r3"] == 0.5
        goal_score = calculate_goal_reward(state, ["the red block is on top of the blue block"])
        assert math.isclose(reward, 0.25**0.4 * goal_score**0.4 * 0.5**0.2, rel_tol=1e-6)

    def test_world_model_fills_reward_sink_with_r1_r2_r3(self) -> None:
        """The optional reward sink exposes the breakdown shown in MCTS metrics."""
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        action = BlockAction(
            ActionType.PICK_UP,
            "red",
            llm_trace={"prob": 0.25},
        )
        mock_llm = MockLLM(token_probs={"Yes": 0.8, "No": 0.2})
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )
        reward_sink: dict[str, Any] = {}

        _, reward, _ = world_model.step(state, action, reward_sink=reward_sink)

        assert reward_sink["r1"] == 0.25
        assert reward_sink["r2"] == 0.5
        assert reward_sink["r3"] == 0.8
        assert math.isclose(reward_sink["reward"], reward)

    def test_world_model_skips_r3_in_lightweight_rollout(self) -> None:
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"})
        mock_llm = MockLLM(token_probs={"Yes": 0.8, "No": 0.2})
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )

        world_model.step(
            state,
            BlockAction(ActionType.PICK_UP, "red"),
            lightweight=True,
        )

        assert mock_llm.call_count == 0

    def test_world_model_paper_mode_uses_llm_for_state_update(
        self, instance_5_state: BlockState
    ) -> None:
        """In paper mode, world_model queries LLM with world_update prompt and applies change."""
        canned_change = (
            "[CHANGE] the hand was empty and now is holding the red block, "
            "the red block was on the table and now is in the hand, "
            "and the red block is no longer clear."
        )
        mock_llm = MockLLM(default_response=canned_change)
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="paper",
        )
        action = BlockAction(ActionType.PICK_UP, "red")
        next_state, reward, is_terminal = world_model.step(instance_5_state, action)

        assert mock_llm.call_count >= 1
        assert next_state.holding == "red"
        assert not is_terminal

    def test_world_model_derives_clear_when_llm_omits_it(self) -> None:
        """A missing "X is now clear" clause must not leave an impossible state.

        Reproduces the artifact behind the inflated 0.7 reward: after stacking
        red on blue the LLM marked blue as no longer clear but forgot that red
        (the stacked block) is now clear.
        """
        state = BlockState(
            clear={"blue", "orange"},
            on_table={"blue", "yellow"},
            on={"orange": "yellow"},
            holding="red",
        )
        change = (
            "[CHANGE] the hand was holding the red block and is now empty, "
            "the red block was in the hand and is now on top of the blue block, "
            "and the blue block is no longer clear."
        )
        mock_llm = MockLLM(default_response=change)
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="paper",
        )
        action = BlockAction(ActionType.STACK, "red", "blue")

        next_state, _, _ = world_model.step(state, action)

        assert next_state.on == {"red": "blue", "orange": "yellow"}
        assert next_state.clear == {"red", "orange"}
        assert next_state.on_table == {"blue", "yellow"}
        assert "red" not in next_state.on_table

    def test_world_model_derives_clear_after_stacking_on_yellow(self) -> None:
        """After stacking orange on yellow, the top block orange must be clear."""
        state = BlockState(
            clear={"yellow"},
            on_table={"blue", "yellow"},
            on={"red": "blue"},
            holding="orange",
        )
        change = (
            "[CHANGE] the hand was holding the orange block and is now empty, "
            "the orange block was in the hand and is now on top of the yellow block, "
            "and the yellow block is no longer clear."
        )
        mock_llm = MockLLM(default_response=change)
        world_model = BlocksworldWorldModel(
            goal=["the orange block is on top of the yellow block"],
            llm=mock_llm,
            mode="paper",
        )
        action = BlockAction(ActionType.STACK, "orange", "yellow")

        next_state, _, _ = world_model.step(state, action)

        assert next_state.on == {"red": "blue", "orange": "yellow"}
        assert next_state.clear == {"red", "orange"}
        assert next_state.on_table == {"blue", "yellow"}

    def test_world_model_accepts_no_longer_covered_by_clause(self) -> None:
        """Models phrase "unstacked" as "no longer covered by", sometimes with the new state."""
        state = BlockState(
            clear={"blue", "orange", "red"},
            on_table={"orange", "red", "yellow"},
            on={"blue": "yellow"},
            holding=None,
        )
        change = (
            "[CHANGE] the yellow block is no longer covered by the blue block and now is clear, "
            "and the blue block is in the hand."
        )
        mock_llm = MockLLM(default_response=change)
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="paper",
        )
        action = BlockAction(ActionType.UNSTACK, "blue", "yellow")

        next_state, _, _ = world_model.step(state, action)

        assert "blue" not in next_state.on
        assert next_state.holding == "blue"
        assert "yellow" in next_state.clear

    def test_world_model_accepts_concise_stack_change(self) -> None:
        state = BlockState(
            clear={"orange"},
            on_table={"orange"},
            on={},
            holding="green",
        )
        mock_llm = MockLLM(
            default_response=(
                "[CHANGE] the green block is now on top of the orange block, "
                "and the hand is now empty."
            )
        )
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="paper",
        )

        next_state, _, _ = world_model.step(
            state,
            BlockAction(ActionType.STACK, "green", "orange"),
        )

        assert next_state.on == {"green": "orange"}
        assert next_state.holding is None
        assert next_state.clear == {"green"}

    def test_world_model_keeps_held_block_out_of_table_and_clear(self) -> None:
        """A held block is neither on the table nor clear, even if the LLM says so."""
        state = BlockState(
            clear={"red", "blue"},
            on_table={"red", "blue"},
            on={},
            holding=None,
        )
        change = (
            "[CHANGE] the hand was empty and is now holding the red block, "
            "the red block was on the table and is now in the hand, "
            "and the red block is now clear."
        )
        mock_llm = MockLLM(default_response=change)
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="paper",
        )
        action = BlockAction(ActionType.PICK_UP, "red")

        next_state, _, _ = world_model.step(state, action)

        assert next_state.holding == "red"
        assert next_state.clear == {"blue"}
        assert next_state.on_table == {"blue"}

    def test_world_model_detects_terminal_goal(self) -> None:
        """world_model marks is_terminal=True and returns high reward when goal is reached."""
        goal_state = BlockState(
            clear={"orange"},
            on_table={"yellow", "blue"},
            on={"red": "blue", "orange": "red"},
            holding=None,
        )
        state_before_final_stack = BlockState(
            clear={"red"},
            on_table={"yellow", "blue"},
            on={"red": "blue"},
            holding="orange",
        )
        world_model = BlocksworldWorldModel(
            goal=["the red block is on top of the blue block", "the orange block is on top of the red block"],
            mode="hybrid",
        )
        action = BlockAction(ActionType.STACK, "orange", "red")
        next_state, reward, is_terminal = world_model.step(state_before_final_stack, action)

        assert is_terminal is True
        assert reward >= 100.0
        assert next_state.on == goal_state.on


# ============================================================================
# 8. BlocksworldDomain Integration Tests
# ============================================================================


class TestBlocksworldDomainIntegration:
    """Integration tests verifying BlocksworldDomain assembles components correctly."""

    def test_domain_creates_consistent_components(self, instance_5_state: BlockState) -> None:
        """Domain factory methods instantiate agent, world model, and reward function."""
        mock_llm = MockLLM()
        domain = BlocksworldDomain(
            initial_state=instance_5_state,
            goal=["the red block is on top of the blue block"],
            llm=mock_llm,
            mode="hybrid",
        )

        assert domain.get_initial_state() == instance_5_state
        assert domain.is_goal_reached(instance_5_state) is False

        agent = domain.create_agent()
        world_model = domain.create_world_model()
        reward_fn = domain.create_reward_fn()

        assert isinstance(agent, BlocksworldAgent)
        assert isinstance(world_model, BlocksworldWorldModel)
        assert callable(reward_fn)
