"""
Poker game environment.
Manages game state, betting rounds, and winner determination.
"""

from typing import List, Optional
from src.evaluator import Card, Deck, CombinedHand, Evaluator
from src.player import Player
from src.ai.ai_player import CFRAIPlayer


class PokerEnvironment:
    """
    Texas Hold'em game environment for 2 players.

    Game Stages:
        1: Initial (call start_new_round to begin)
        2: Preflop betting
        3: Flop betting (3 community cards)
        4: Turn betting (4 community cards)
        5: River betting (5 community cards)
        6: Round over (distribute winnings)
    """

    def __init__(self, input_cards: bool = False):
        """
        Initialize the poker environment.

        Args:
            input_cards: If True, prompt for manual card input (for testing)
        """
        self.players: List[Player] = []
        self.deck = Deck()

        self.game_stage = 1
        self.dealer_button_position = 0
        self.position_in_play = 0

        self.total_pot_balance = 0
        self.stage_pot_balance = 0
        self.community_cards: List[Card] = []

        self.raise_position = 0
        self.showdown = False

        # Game settings
        self.starting_balance = 2500
        self.SMALL_BLIND = 100
        self.BIG_BLIND = 200

        self.input_cards = input_cards
        self.history: List[str] = []
        self.players_balance_history: List[List[int]] = []

        self._ai_player_idx: Optional[int] = None

    def add_player(self) -> None:
        """Add a human player."""
        self.players.append(Player(self.starting_balance))

    def add_ai_player(self, model_path: str = "models") -> None:
        """Add an AI player using CFR strategy."""
        ai_player = CFRAIPlayer(self.starting_balance, model_path=model_path)
        self.players.append(ai_player)
        self._ai_player_idx = len(self.players) - 1

    def get_player(self, idx: int) -> Player:
        """Get player by index."""
        return self.players[idx]

    def valid_actions(self) -> List[str]:
        """Get valid actions for current player."""
        actions = ["f"]  # Fold always valid

        if self.players[0].current_bet == self.players[1].current_bet:
            actions.append("k")  # Check
        else:
            actions.append("c")  # Call

        # Bet/raise always available (subject to balance)
        return actions

    def get_highest_current_bet(self) -> int:
        """Get the current highest bet."""
        highest = 0
        for player in self.players:
            if player.current_bet > highest and player.playing_current_round:
                highest = player.current_bet
        return highest

    def count_remaining_players(self) -> int:
        """Count players still in the round."""
        return sum(1 for p in self.players if p.playing_current_round)

    def start_new_round(self) -> None:
        """Start a new poker round."""
        assert len(self.players) >= 2, "Need at least 2 players"

        if self.input_cards:
            self.starting_balance = int(input("Enter starting balance: "))
            self.BIG_BLIND = int(input("Enter big blind: "))
            self.SMALL_BLIND = self.BIG_BLIND // 2

        # Reset players
        for player in self.players:
            player.playing_current_round = True
            player.current_bet = 0
            player.clear_hand()
            player.player_balance = self.starting_balance

        # Reset game state
        self.deck.reset_deck()
        self.community_cards = []
        self.stage_pot_balance = 0
        self.total_pot_balance = 0
        self.showdown = False
        self.history = []

        # Move dealer button
        self.dealer_button_position = (self.dealer_button_position + 1) % len(self.players)

        # Start preflop
        self.game_stage = 1
        self._move_to_next_game_stage()

    def handle_game_stage(self, action: str = "") -> None:
        """Process a betting action."""
        if self.game_stage in [1, 6]:
            return  # Nothing to do at start/end

        self._play_current_stage(action)

    def end_of_round(self) -> bool:
        """Check if round is over."""
        return self.game_stage == 6

    def get_game_state(self) -> dict:
        """Get current game state for external use."""
        return {
            "stage": self.game_stage,
            "pot": self.total_pot_balance + self.stage_pot_balance,
            "community_cards": [str(c) for c in self.community_cards],
            "player_in_play": self.position_in_play,
            "valid_actions": self.valid_actions(),
            "players": [
                {
                    "balance": p.player_balance,
                    "current_bet": p.current_bet,
                    "in_round": p.playing_current_round,
                    "is_ai": p.is_AI,
                }
                for p in self.players
            ],
        }

    # --- Private Methods ---

    def _update_stage_pot_balance(self) -> None:
        """Update pot balance from player bets."""
        self.stage_pot_balance = sum(p.current_bet for p in self.players)

    def _play_current_stage(self, action: str = "") -> None:
        """Process betting for current stage."""
        self._update_stage_pot_balance()

        current_player = self.players[self.position_in_play]

        if current_player.is_AI:
            action = current_player.place_bet(self)
        else:
            if not action:
                return  # Waiting for human action
            result = current_player.place_bet(action, self)
            if result is None:
                return  # Invalid action
            action = result

        self.history.append(action)

        if action.startswith("b"):
            self.raise_position = self.position_in_play
        elif action == "f":
            current_player.playing_current_round = False

        self._update_stage_pot_balance()

        # Check if round ends
        if self.count_remaining_players() == 1:
            self._end_round()
            return

        self._move_to_next_player()

        if self.position_in_play == self.raise_position:
            self._move_to_next_game_stage()

    def _move_to_next_player(self, from_position: Optional[int] = None) -> None:
        """Move to next active player."""
        assert self.count_remaining_players() > 1

        if from_position is not None:
            self.position_in_play = from_position

        self.position_in_play = (self.position_in_play + 1) % len(self.players)

        while not self.players[self.position_in_play].playing_current_round:
            self.position_in_play = (self.position_in_play + 1) % len(self.players)

    def _update_balances_end_of_stage(self) -> None:
        """Deduct bets from player balances."""
        for player in self.players:
            player.player_balance -= player.current_bet
            player.current_bet = 0

    def _move_stage_to_total_pot(self) -> None:
        """Move stage pot to total pot."""
        self.total_pot_balance += self.stage_pot_balance
        self.stage_pot_balance = 0

    def _move_to_next_game_stage(self) -> None:
        """Advance to next game stage."""
        self._update_balances_end_of_stage()
        self._move_stage_to_total_pot()

        self.game_stage += 1

        if self.game_stage == 2:
            self._play_preflop()
        elif self.game_stage == 3:
            self._play_flop()
        elif self.game_stage == 4:
            self._play_turn()
        elif self.game_stage == 5:
            self._play_river()
        else:
            self._end_round()
            return

        # Check for all-in showdown
        if self.total_pot_balance == len(self.players) * self.starting_balance:
            self._move_to_next_game_stage()

    def _play_preflop(self) -> None:
        """Handle preflop: post blinds and deal hole cards."""
        n = len(self.players)

        # Post blinds
        self.players[(self.dealer_button_position + 1) % n].current_bet = self.BIG_BLIND
        self.players[(self.dealer_button_position + 2) % n].current_bet = self.SMALL_BLIND

        self._update_stage_pot_balance()

        # Set starting position
        if n == 2:
            self.position_in_play = self.dealer_button_position
        else:
            self.position_in_play = (self.dealer_button_position + 3) % n
        self.raise_position = self.position_in_play

        # Deal hole cards
        for i in range(n):
            player_idx = (self.dealer_button_position + 1 + i) % n
            cards_str = ""
            for _ in range(2):
                if self.input_cards and player_idx == 0:
                    card = Card(input("Enter card (e.g., Ah): "))
                else:
                    card = self.deck.draw()
                cards_str += str(card)
                self.players[player_idx].add_card_to_hand(card)
            self.history.append(cards_str)

    def _play_flop(self) -> None:
        """Deal flop (3 community cards)."""
        self.deck.draw()  # Burn card

        for i in range(3):
            if self.input_cards:
                card = Card(input(f"Enter flop card {i+1}: "))
            else:
                card = self.deck.draw()
            self.community_cards.append(card)

        self.history.append("/")
        self.history.append("".join(str(c) for c in self.community_cards))

        self._move_to_next_player(self.dealer_button_position)
        self.raise_position = self.position_in_play

    def _play_turn(self) -> None:
        """Deal turn (4th community card)."""
        self.deck.draw()  # Burn card

        if self.input_cards:
            card = Card(input("Enter turn card: "))
        else:
            card = self.deck.draw()
        self.community_cards.append(card)

        self.history.append("/")
        self.history.append(str(card))

        self._move_to_next_player(self.dealer_button_position)
        self.raise_position = self.position_in_play

    def _play_river(self) -> None:
        """Deal river (5th community card)."""
        self.deck.draw()  # Burn card

        if self.input_cards:
            card = Card(input("Enter river card: "))
        else:
            card = self.deck.draw()
        self.community_cards.append(card)

        self.history.append("/")
        self.history.append(str(card))

        self._move_to_next_player(self.dealer_button_position)
        self.raise_position = self.position_in_play

    def _end_round(self) -> None:
        """End the round and determine winner."""
        self._update_balances_end_of_stage()
        self._move_stage_to_total_pot()

        if self.count_remaining_players() > 1:
            self.showdown = True
            evaluator = Evaluator()
            potential_winner_indices = []

            for idx, player in enumerate(self.players):
                if player.playing_current_round:
                    potential_winner_indices.append(idx)

                    if self.input_cards and idx == 1:
                        self.players[1].clear_hand()
                        self.players[1].add_card_to_hand(Card(input("Opponent card 1: ")))
                        self.players[1].add_card_to_hand(Card(input("Opponent card 2: ")))

                    hand = CombinedHand(self.community_cards + player.hand)
                    evaluator.add_hands(hand)

            winners = evaluator.get_winner()

            # Mark non-winners as out
            for player in self.players:
                player.playing_current_round = False
            for winner_idx in winners:
                self.players[potential_winner_indices[winner_idx]].playing_current_round = True

        self.game_stage = 6
        self._distribute_pot()

    def _distribute_pot(self) -> None:
        """Distribute pot to winner(s)."""
        winners = [p for p in self.players if p.playing_current_round]
        winnings = self.total_pot_balance / len(winners)

        for player in winners:
            player.player_balance += winnings

        # Track balance history
        for idx, player in enumerate(self.players):
            profit = int(player.player_balance - self.starting_balance)
            if idx >= len(self.players_balance_history):
                self.players_balance_history.append([])
            self.players_balance_history[idx].append(profit)

    def get_winner_indices(self) -> List[int]:
        """Get indices of winning players."""
        return [i for i, p in enumerate(self.players) if p.playing_current_round]
