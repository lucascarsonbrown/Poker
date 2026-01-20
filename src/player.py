"""
Player classes for poker game.
"""

from typing import List, Optional


class Player:
    """Base player class for human players."""

    def __init__(self, balance: int):
        self.hand: List = []
        self.player_balance: int = balance
        self.current_bet: int = 0
        self.playing_current_round: bool = True
        self.is_AI: bool = False

    def add_card_to_hand(self, card) -> None:
        """Add a card to the player's hand."""
        self.hand.append(card)
        assert len(self.hand) <= 2, "Player cannot have more than 2 hole cards"

    def clear_hand(self) -> None:
        """Clear the player's hand for a new round."""
        self.hand = []

    def place_bet(self, action: str, observed_env) -> Optional[str]:
        """
        Process a betting action from the player.

        Args:
            action: Action string ('f', 'k', 'c', or 'bX' where X is amount)
            observed_env: The game environment

        Returns:
            The action if valid, None if invalid
        """
        if action == "f":
            return action

        elif action == "k":
            # Check - only valid if no bet to call
            if self.current_bet == observed_env.get_highest_current_bet():
                return action
            else:
                print("Cannot check - there is a bet to call.")
                return None

        elif action == "c":
            # Call - match the current highest bet
            self.current_bet = observed_env.get_highest_current_bet()
            return action

        elif action.startswith("b"):
            # Bet/Raise
            try:
                bet_size = int(action[1:])
            except ValueError:
                print("Invalid bet format. Use 'bX' where X is the amount.")
                return None

            highest_bet = observed_env.get_highest_current_bet()

            if bet_size < highest_bet:
                print(f"Must raise to at least {highest_bet}.")
                return None
            elif bet_size > self.player_balance:
                print("Cannot bet more than your balance.")
                return None
            elif bet_size == highest_bet and highest_bet > 0:
                print("Use 'c' to call, not bet.")
                return None
            else:
                self.current_bet = bet_size
                return action

        else:
            print(f"Unknown action: {action}")
            return None
