"""
This is the hand state management for tracking poker games.
It maintains the state and converts events to format for the calculator in src.
"""
from typing import List, Dict, Any
from schemas import *


class HandState:
    """
    Tracks the current state of a poker hand.
    Converts between iOS events and PokerCalculator format.
    """

    def __init__(self):
        self.reset()
        self.hand_number = 0

    def reset(self):
        self.hero_cards: List[str] = []
        self.board_cards: List[str] = []
        self.street: Street = Street.PREFLOP

        self.hero_stack: int = 0
        self.villain_stack: int = 0
        self.starting_hero_stack: int = 0
        self.starting_villain_stack: int = 0

        self.small_blind: int = 1
        self.big_blind: int = 2
        self.hero_is_button: bool = True

        self.pot: int = 0
        self.hero_invested: int = 0
        self.villain_invested: int = 0

        self.hero_to_act: bool = False
        self.hand_over: bool = False

        self.action_history: List[Dict[str, Any]] = []
        self.abstract_history: List[str] = []

    def start_hand(self, event: HandStartEvent):
        self.reset()
        self.hand_number += 1

        self.hero_stack = event.hero_stack
        self.villain_stack = event.villain_stack
        self.starting_hero_stack = event.hero_stack
        self.starting_villain_stack = event.villain_stack
        self.small_blind = event.small_blind
        self.big_blind = event.big_blind
        self.hero_is_button = event.hero_is_button

        # post blinds
        if self.hero_is_button:
            # Hero is the small blind and villian is the big blind
            sb_amount = min(self.small_blind, self.hero_stack)
            bb_amount = min(self.big_blind, self.villain_stack)
            self.hero_invested = sb_amount
            self.villain_invested = bb_amount
            self.hero_stack -= sb_amount
            self.villain_stack -= bb_amount
            self.hero_to_act = True
        else:
            # Villain is SB, hero is BB
            sb_amount = min(self.small_blind, self.villain_stack)
            bb_amount = min(self.big_blind, self.hero_stack)
            self.villain_invested = sb_amount
            self.hero_invested = bb_amount
            self.villain_stack -= sb_amount
            self.hero_stack -= bb_amount
            self.hero_to_act = False

        self.pot = self.hero_invested + self.villain_invested

    def set_hole_cards(self, event: HoleCardsEvent):
        self.hero_cards = event.cards

    def update_board(self, event: BoardUpdateEvent):
        self.board_cards = event.cards
        self.street = event.street

        # Reset betting for new street (postflop: BB acts first)
        if event.street != Street.PREFLOP:
            self.hero_to_act = not self.hero_is_button

    def process_action(self, event: ActionEvent):
        """Process a player action."""
        is_hero = event.player == Player.HERO

        # Record action
        self.action_history.append({
            "player": event.player.value,
            "action": event.action_type.value,
            "amount": event.amount,
            "street": event.street.value,
        })

        # Update abstract history for CFR
        self.abstract_history.append(event.action_type.value)

        # Calculate amounts
        if event.action_type == ActionType.FOLD:
            self.hand_over = True
            return

        if event.action_type == ActionType.CHECK:
            pass  # No chips move

        elif event.action_type == ActionType.CALL:
            # Match the other player's investment
            if is_hero:
                call_amount = self.villain_invested - self.hero_invested
                call_amount = min(call_amount, self.hero_stack)
                self.hero_invested += call_amount
                self.hero_stack -= call_amount
            else:
                call_amount = self.hero_invested - self.villain_invested
                call_amount = min(call_amount, self.villain_stack)
                self.villain_invested += call_amount
                self.villain_stack -= call_amount

        elif event.action_type in [ActionType.BET_MIN, ActionType.BET_MID, ActionType.BET_MAX]:
            amount = event.amount or 0
            if is_hero:
                amount = min(amount, self.hero_stack)
                self.hero_invested += amount
                self.hero_stack -= amount
            else:
                amount = min(amount, self.villain_stack)
                self.villain_invested += amount
                self.villain_stack -= amount

        # Update pot
        self.pot = self.hero_invested + self.villain_invested

        # Toggle turn
        self.hero_to_act = not is_hero

    def end_hand(self, event: HandEndEvent):
        self.hand_over = True

        if event.winner == Player.HERO:
            self.hero_stack += self.pot
        elif event.winner == Player.VILLAIN:
            self.villain_stack += self.pot
        else:
            # Split pot
            self.hero_stack += self.pot // 2
            self.villain_stack += self.pot - (self.pot // 2)

    def get_to_call(self) -> int:
        """Amount hero needs to call."""
        return max(0, self.villain_invested - self.hero_invested)

    def get_state_response(self) -> GameStateResponse:
        """Convert current state to response for iOS."""
        return GameStateResponse(
            hand_number=self.hand_number,
            street=self.street,
            hero_cards=self.hero_cards,
            board_cards=self.board_cards,
            pot=self.pot,
            hero_stack=self.hero_stack,
            villain_stack=self.villain_stack,
            hero_to_act=self.hero_to_act,
            to_call=self.get_to_call(),
            action_history=self.action_history,
        )

    def get_calculator_params(self) -> Dict[str, Any]:
        """Get parameters for PokerCalculator.get_ai_action()."""
        return {
            "hole_cards": self.hero_cards,
            "community_cards": self.board_cards if self.board_cards else None,
            "history": self.abstract_history,
            "pot_size": self.pot,
            "to_call": self.get_to_call(),
            "stack_size": self.hero_stack,
        }
