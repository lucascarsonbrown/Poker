"""
Preflop Texas Hold'em trainer using CFR.

Uses lossless card abstraction (169 clusters) and limited action abstraction.
"""

import argparse
import numpy as np
from typing import List
from phevaluator import evaluate_cards

from training.cfr import History, InfoSet, CFR

# Discrete actions for preflop
DISCRETE_ACTIONS = ["k", "bMIN", "bMID", "bMAX", "c", "f"]

# Global data (loaded before training)
player_hands = None
opponent_hands = None
boards = None
winners = None
dataset_size = 0  # Track dataset size for modulo


def get_preflop_cluster_id(two_cards: str) -> int:
    """
    Lossless preflop abstraction into 169 clusters.
    """
    if isinstance(two_cards, list):
        two_cards = "".join(two_cards)

    RANK_VALUES = {
        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13,
    }

    rank1, suit1 = two_cards[0], two_cards[1]
    rank2, suit2 = two_cards[2], two_cards[3]

    def hash_pair(a: str, b: str) -> int:
        first = min(RANK_VALUES[a], RANK_VALUES[b])
        second = max(RANK_VALUES[a], RANK_VALUES[b])

        def sum_to(n):
            if n <= 1:
                return 0
            count = n - 1
            return (count * (12 + 12 - (n - 2))) // 2

        return int(sum_to(first) + (second - first))

    if rank1 == rank2:
        return RANK_VALUES[rank1]
    elif suit1 != suit2:
        return 13 + hash_pair(rank1, rank2)
    else:
        return 91 + hash_pair(rank1, rank2)


class PreflopHistory(History):
    """
    Preflop game history.

    History format: ['AkTh', 'QdKd', 'c', 'bMIN', '/', 'Qh2d3s4h5s']
    """

    def __init__(self, history: List[str] = None, sample_id: int = 0):
        super().__init__(history)
        self.sample_id = sample_id

    def is_terminal(self) -> bool:
        if not self.history:
            return False
        # Terminal if community cards are shown (10 chars for 5 cards)
        return len(self.history[-1]) == 10

    def actions(self) -> List[str]:
        if self.is_chance():
            return []

        if self.is_terminal():
            raise Exception("Cannot get actions for terminal state")

        # Determine valid actions based on betting sequence
        if len(self.history) == 2:  # First to act preflop
            return ["c", "bMIN", "bMID", "bMAX", "f"]
        elif self.history[-1] == "bMIN":
            return ["bMID", "bMAX", "f", "c"]
        elif self.history[-1] == "bMID":
            return ["bMAX", "f", "c"]
        elif self.history[-1] == "bMAX":
            return ["f", "c"]
        else:
            return ["k", "bMIN", "bMID", "bMAX"]

    def player(self) -> int:
        if len(self.history) < 2:
            return -1  # Chance node
        if self._game_stage_ended():
            return -1
        if self.history[-1] == "/":
            return -1
        return (len(self.history) + 1) % 2

    def _game_stage_ended(self) -> bool:
        return (
            (self.history[-1] == "c" and len(self.history) > 3)
            or self.history[-1] == "f"
            or self.history[-2:] == ["c", "k"]
        )

    def sample_chance_outcome(self) -> str:
        idx = self.sample_id % dataset_size  # Wrap around dataset
        if len(self.history) == 0:
            return "".join(player_hands[idx])
        elif len(self.history) == 1:
            return "".join(opponent_hands[idx])
        elif self.history[-1] != "/":
            return "/"
        else:
            return "".join(boards[idx])

    def terminal_utility(self, player: int) -> float:
        idx = self.sample_id % dataset_size  # Wrap around dataset
        winner = winners[idx]
        pot_size, _ = self._get_pot_size(self.history)

        if "f" in self.history:
            fold_idx = self.history.index("f")
            pot_size, latest_bet = self._get_pot_size(self.history[:fold_idx - 1])
            if self.history[-3] in ["bMIN", "bMID"]:
                pot_size += latest_bet

            if len(self.history) % 2 == player:
                return -pot_size / 2
            else:
                return pot_size / 2

        # Showdown
        if winner == 0:
            return 0

        if (winner == 1 and player == 0) or (winner == -1 and player == 1):
            return pot_size / 2
        else:
            return -pot_size / 2

    def _get_pot_size(self, history: List[str]) -> tuple:
        stage_total = 3  # 1 SB + 1 BB
        latest_bet = 2

        for action in history:
            if action == "bMIN":
                old_total = stage_total
                stage_total = latest_bet + stage_total
                latest_bet = old_total
            elif action == "bMID":
                old_total = stage_total
                stage_total = latest_bet + 2 * stage_total
                latest_bet = 2 * old_total
            elif action == "bMAX":
                stage_total = latest_bet + 100
                latest_bet = 100
            elif action == "c":
                stage_total = 2 * latest_bet

        return stage_total, latest_bet

    def __add__(self, action: str) -> "PreflopHistory":
        return PreflopHistory(self.history + [action], self.sample_id)

    def get_infoSet_key(self) -> List[str]:
        player = self.player()
        infoset = [str(get_preflop_cluster_id(self.history[player]))]

        for action in self.history:
            if action in DISCRETE_ACTIONS:
                infoset.append(action)

        return infoset


class PreflopInfoSet(InfoSet):
    """Information set for preflop decisions."""
    pass


def create_infoSet(infoSet_key: List[str], actions: List[str], player: int) -> PreflopInfoSet:
    return PreflopInfoSet(infoSet_key, actions, player)


def create_history(sample_id: int) -> PreflopHistory:
    return PreflopHistory(sample_id=sample_id)


def evaluate_winner(board: List[str], player_hand: List[str], opponent_hand: List[str]) -> int:
    """Determine winner: 1 for player, -1 for opponent, 0 for tie."""
    p1_score = evaluate_cards(*(board + player_hand))
    p2_score = evaluate_cards(*(board + opponent_hand))
    if p1_score < p2_score:
        return 1
    elif p1_score > p2_score:
        return -1
    return 0


def generate_dataset(num_samples: int = 50000) -> tuple:
    """Generate random poker hands for training."""
    import random

    all_boards = []
    all_player_hands = []
    all_opponent_hands = []
    all_winners = []

    for _ in range(num_samples):
        deck = []
        for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
            for suit in ["h", "d", "s", "c"]:
                deck.append(rank + suit)
        random.shuffle(deck)

        board = deck[:5]
        player_hand = deck[5:7]
        opponent_hand = deck[7:9]

        all_boards.append(board)
        all_player_hands.append(player_hand)
        all_opponent_hands.append(opponent_hand)
        all_winners.append(evaluate_winner(board, player_hand, opponent_hand))

    return all_boards, all_player_hands, all_opponent_hands, all_winners


def main():
    global boards, player_hands, opponent_hands, winners, dataset_size

    parser = argparse.ArgumentParser(description="Train preflop CFR strategy")
    parser.add_argument("-i", "--iterations", type=int, default=50000,
                        help="CFR iterations per batch")
    parser.add_argument("-b", "--batches", type=int, default=1,
                        help="Number of training batches")
    parser.add_argument("-s", "--samples", type=int, default=50000,
                        help="Samples per batch")
    parser.add_argument("-o", "--output", type=str, default="models/preflop_infoSets.joblib",
                        help="Output file path")
    args = parser.parse_args()

    cfr = CFR(create_infoSet, create_history, iterations=args.iterations)

    for batch in range(args.batches):
        print(f"\n=== Batch {batch + 1}/{args.batches} ===")

        # Generate training data
        print("Generating dataset...")
        boards, player_hands, opponent_hands, winners = generate_dataset(args.samples)
        dataset_size = len(boards)
        print(f"Dataset size: {dataset_size}")

        # Train
        cfr.solve(method="vanilla")

        # Save after each batch
        cfr.export_infoSets(args.output)

    print(f"\nTraining complete! Model saved to {args.output}")


if __name__ == "__main__":
    main()
