"""Domain models for Blocksworld: states, actions, and hybrid proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Supported atomic actions in Blocksworld."""

    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    STACK = "stack"
    UNSTACK = "unstack"


@dataclass(frozen=True)
class BlockAction:
    """An action manipulation of blocks in Blocksworld."""

    action_type: ActionType | str
    block: str
    target: Optional[str] = None
    thought: Optional[str] = None
    llm_trace: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Normalizes action_type to ActionType enum if string is supplied."""
        if isinstance(self.action_type, str):
            try:
                object.__setattr__(self, "action_type", ActionType(self.action_type.lower()))
            except ValueError:
                # Keep as-is if not directly convertible, so equality/testing handles it
                pass

    def to_text(self) -> str:
        """Converts action to paper canonical natural language representation."""
        atype = self.action_type.value if isinstance(self.action_type, ActionType) else str(self.action_type)
        if atype == ActionType.PICK_UP.value:
            return f"Pick up the {self.block} block."
        elif atype == ActionType.PUT_DOWN.value:
            return f"Put down the {self.block} block."
        elif atype == ActionType.STACK.value:
            return f"Stack the {self.block} block on top of the {self.target} block."
        elif atype == ActionType.UNSTACK.value:
            return f"Unstack the {self.block} block from on top of the {self.target} block."
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

    def to_paper_text(self) -> str:
        """Canonical continuation used by the RAP paper prompts (lowercase, no period)."""
        text = self.to_text().rstrip(".")
        return text[:1].lower() + text[1:]

    @classmethod
    def from_text(cls, text: str) -> BlockAction:
        """Parses natural language string into a structured BlockAction."""
        if not text or not text.strip():
            raise ValueError("Cannot parse empty action text")

        cleaned = text.strip()
        # Remove optional leading prefix like '[ACTION 1]', 'Action 1:', '1.'
        cleaned = re.sub(r"^(?:\[action\s*\d*\]|\d+[\.\)]|action\s*\d*:?)\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Match Pick up
        m = re.fullmatch(r"pick\s+up\s+the\s+([a-zA-Z0-9_-]+)\s+block\.?", cleaned, re.IGNORECASE)
        if m:
            return cls(action_type=ActionType.PICK_UP, block=m.group(1).lower())

        # Match Put down
        m = re.fullmatch(r"put\s+down\s+the\s+([a-zA-Z0-9_-]+)\s+block\.?", cleaned, re.IGNORECASE)
        if m:
            return cls(action_type=ActionType.PUT_DOWN, block=m.group(1).lower())

        # Match Stack
        m = re.fullmatch(
            r"stack\s+the\s+([a-zA-Z0-9_-]+)\s+block\s+on\s+top\s+of\s+the\s+([a-zA-Z0-9_-]+)\s+block\.?",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            return cls(action_type=ActionType.STACK, block=m.group(1).lower(), target=m.group(2).lower())

        # Match Unstack
        m = re.fullmatch(
            r"unstack\s+the\s+([a-zA-Z0-9_-]+)\s+block\s+from\s+on\s+top\s+of\s+the\s+([a-zA-Z0-9_-]+)\s+block\.?",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            return cls(action_type=ActionType.UNSTACK, block=m.group(1).lower(), target=m.group(2).lower())

        raise ValueError(f"Cannot parse action from text: {text!r}")


@dataclass
class BlockState:
    """Represents the physical configuration of blocks in Blocksworld."""

    clear: set[str] = field(default_factory=set)
    on_table: set[str] = field(default_factory=set)
    on: dict[str, str] = field(default_factory=dict)  # top_block -> bottom_block
    holding: Optional[str] = None

    def to_text(self) -> str:
        """Serializes current configuration into canonical paper English format."""
        clauses: list[str] = []

        # 1. Clear blocks
        for b in sorted(self.clear):
            clauses.append(f"the {b} block is clear")

        # 2. Hand status
        if self.holding is None:
            clauses.append("the hand is empty")
        else:
            clauses.append(f"the hand is holding the {self.holding} block")

        # 3. Stacked relations
        for top, bottom in sorted(self.on.items()):
            clauses.append(f"the {top} block is on top of the {bottom} block")

        # 4. Table blocks
        for b in sorted(self.on_table):
            clauses.append(f"the {b} block is on the table")

        if not clauses:
            return ""
        if len(clauses) == 1:
            return f"{clauses[0]}."
        return f"{', '.join(clauses[:-1])} and {clauses[-1]}."

    @classmethod
    def from_text(cls, text: str) -> BlockState:
        """Robustly parses natural language state description into BlockState."""
        if not text or not text.strip():
            raise ValueError("Cannot parse empty state text")

        clean_matches = set(re.findall(r"the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+clear", text, re.IGNORECASE))
        table_matches = set(re.findall(r"the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+on\s+the\s+table", text, re.IGNORECASE))
        on_matches = dict(
            re.findall(
                r"the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-zA-Z0-9_-]+)\s+block",
                text,
                re.IGNORECASE,
            )
        )

        # Holding detection
        holding: str | None = None
        holding_m = re.search(
            r"(?:hand\s+is\s+holding\s+the|holding\s+the)\s+([a-zA-Z0-9_-]+)\s+block",
            text,
            re.IGNORECASE,
        )
        if holding_m:
            holding = holding_m.group(1).lower()
        else:
            holding_in_hand_m = re.search(
                r"the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+in\s+the\s+hand",
                text,
                re.IGNORECASE,
            )
            if holding_in_hand_m:
                holding = holding_in_hand_m.group(1).lower()

        hand_empty = bool(re.search(r"hand\s+is\s+empty", text, re.IGNORECASE))

        # Check for malformed text where no block facts were parsed
        if not clean_matches and not table_matches and not on_matches and holding is None and not hand_empty:
            raise ValueError(f"Malformed state text: no block facts recognized in {text!r}")

        return cls(
            clear={c.lower() for c in clean_matches},
            on_table={t.lower() for t in table_matches},
            on={k.lower(): v.lower() for k, v in on_matches.items()},
            holding=holding,
        )

    def is_legal_action(self, action: BlockAction) -> bool:
        """Checks if preconditions for action are satisfied in current state."""
        atype = action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type)

        if atype == ActionType.PICK_UP.value:
            # Preconditions: hand empty, block clear, block on table
            return (
                self.holding is None
                and action.block in self.clear
                and action.block in self.on_table
            )

        elif atype == ActionType.PUT_DOWN.value:
            # Preconditions: hand is holding the block and no stack target is given.
            # ``put_down`` always releases the block on the table; a non-null target
            # means the caller wanted ``stack`` and must not be silently accepted.
            return self.holding == action.block and action.target is None

        elif atype == ActionType.STACK.value:
            # Preconditions: holding block, target is clear, target != block, target is specified
            return (
                self.holding == action.block
                and action.target is not None
                and action.target in self.clear
                and action.target != action.block
            )

        elif atype == ActionType.UNSTACK.value:
            # Preconditions: hand empty, block clear, block is on target
            return (
                self.holding is None
                and action.target is not None
                and action.block in self.clear
                and self.on.get(action.block) == action.target
            )

        return False

    def get_legal_actions(self) -> list[BlockAction]:
        """Generates all physically valid actions from current state."""
        legal_actions: list[BlockAction] = []

        if self.holding is not None:
            # 1. Put down currently held block
            legal_actions.append(BlockAction(ActionType.PUT_DOWN, self.holding))
            # 2. Stack onto any clear block
            for c in sorted(self.clear):
                if c != self.holding:
                    legal_actions.append(BlockAction(ActionType.STACK, self.holding, c))
        else:
            # Hand is empty
            for c in sorted(self.clear):
                if c in self.on_table:
                    legal_actions.append(BlockAction(ActionType.PICK_UP, c))
                elif c in self.on:
                    target = self.on[c]
                    legal_actions.append(BlockAction(ActionType.UNSTACK, c, target))

        return legal_actions

    def apply_action(self, action: BlockAction) -> BlockState:
        """Deterministically applies action according to physics rules, returning new state."""
        if not self.is_legal_action(action):
            raise ValueError(f"Action {action} is not legal in current state: {self.to_text()}")

        atype = action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type)
        new_clear = set(self.clear)
        new_on_table = set(self.on_table)
        new_on = dict(self.on)
        new_holding = self.holding

        if atype == ActionType.PICK_UP.value:
            new_holding = action.block
            new_clear.remove(action.block)
            new_on_table.remove(action.block)

        elif atype == ActionType.PUT_DOWN.value:
            new_holding = None
            new_on_table.add(action.block)
            new_clear.add(action.block)

        elif atype == ActionType.STACK.value:
            assert action.target is not None
            new_holding = None
            new_on[action.block] = action.target
            new_clear.remove(action.target)
            new_clear.add(action.block)

        elif atype == ActionType.UNSTACK.value:
            assert action.target is not None
            new_holding = action.block
            del new_on[action.block]
            new_clear.remove(action.block)
            new_clear.add(action.target)

        return BlockState(
            clear=new_clear,
            on_table=new_on_table,
            on=new_on,
            holding=new_holding,
        )


class HybridActionProposal(BaseModel):
    """Pydantic schema for structured hybrid action proposal with reasoning thought."""

    thought: str = Field(description="Internal reasoning thought explaining the action choice")
    action: str = Field(description="Action name (pick_up, put_down, stack, unstack)")
    block: str = Field(description="Primary block being manipulated")
    target: Optional[str] = Field(default=None, description="Target block for stack/unstack")

    def to_block_action(self) -> BlockAction:
        """Converts proposal to a BlockAction instance."""
        try:
            atype = ActionType(self.action.lower())
        except ValueError:
            atype = self.action.lower()  # type: ignore[assignment]

        return BlockAction(
            action_type=atype,
            block=self.block,
            target=self.target,
            thought=self.thought,
        )
