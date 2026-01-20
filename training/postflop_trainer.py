"""
Postflop Texas Hold'em trainer using CFR.

Uses equity-based card abstraction and limited action abstraction.
"""

import argparse
import random
import numpy as np
from typing import List
from phevaluator import evaluate_cards
from joblib import Parallel, delayed
from tqdm import tqdm

from training.cfr import History, InfoSet, CFR

# Discrete actions for postflop
DISCRETE_ACTIONS = ["k", "bMIN", "bMAX", "c", "f"]

# Cluster configuration
NUM_FLOP_CLUSTERS = 50
NUM_TURN_CLUSTERS = 50
NUM_RIVER_CLUSTERS = 10

# Global data (loaded before training)
boards = None
player_hands = None
opponent_hands = None
player_flop_clusters = None
player_turn_clusters = None
player_river_clusters = None
opp_flop_clusters = None
opp_turn_clusters = None
opp_river_clusters = None
winners = None
dataset_size = 0  # Track dataset size for modulo


def calculate_equity(player_cards: List[str], community_cards: List[str], n: int = 1000) -> float:
    """Monte Carlo equity calculation."""
    excluded = set(player_cards + community_cards)
    deck = []
    for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
        for suit in ["h", "d", "s", "c"]:
            card = rank + suit
            if card not in excluded:
                deck.append(card)

    wins = 0
    remaining = 5 - len(community_cards)

    for _ in range(n):
        random.shuffle(deck)
        opp_cards = deck[:2]
        new_comm = deck[2:2 + remaining]
        full_comm = community_cards + new_comm

        p_score = evaluate_cards(*(player_cards + full_comm))
        o_score = evaluate_cards(*(opp_cards + full_comm))

        if p_score <= o_score:
            wins += 1

    return wins / n


def predict_cluster_fast(cards: List[str], total_clusters: int) -> int:
    """Fast equity-based cluster assignment."""
    equity = calculate_equity(cards[:2], cards[2:], n=500)
    return min(total_clusters - 1, int(equity * total_clusters))


class PostflopHistory(History):
    """
    Postflop game history.

    Format: ['AkTh', 'QdKd', '/', 'QhJdKs', 'bMIN', 'c', '/', 'Ah', 'k', 'k', ...]
    """

    def __init__(self, history: List[str] = None, sample_id: int = 0):
        super().__init__(history)
        self.sample_id = sample_id
        self.stage_i = history.count("/") if history else 0

    def is_terminal(self) -> bool:
        if not self.history:
            return False
        folded = self.history[-1] == "f"
        is_showdown = self.stage_i == 3 and self._game_stage_ended()
        return folded or is_showdown

    def actions(self) -> List[str]:
        if self.is_chance():
            return []
        if self.is_terminal():
            raise Exception("Cannot get actions for terminal state")

        # Determine valid actions
        if self.history[-1] == "k":
            return ["k", "bMIN", "bMAX"]
        elif self.history[-2:] == ["k", "bMIN"]:
            return ["f", "c"]
        elif self.history[-1] == "bMIN":
            return ["bMAX", "f", "c"]
        elif self.history[-1] == "bMAX":
            return ["f", "c"]
        else:
            return ["k", "bMIN", "bMAX"]

    def player(self) -> int:
        if len(self.history) <= 3:
            return -1
        if self._game_stage_ended():
            return -1
        if self.history[-1] == "/":
            return -1

        last_stage = self._get_last_game_stage()
        return (len(last_stage) + 1) % 2

    def _game_stage_ended(self) -> bool:
        return (
            self.history[-1] == "c"
            or self.history[-1] == "f"
            or self.history[-2:] == ["k", "k"]
        )

    def _get_last_game_stage(self) -> List[str]:
        last_idx = max(i for i, v in enumerate(self.history) if v == "/")
        return self.history[last_idx + 1:]

    def sample_chance_outcome(self) -> str:
        idx = self.sample_id % dataset_size  # Wrap around dataset
        if len(self.history) == 0:
            return "".join(player_hands[idx])
        elif len(self.history) == 1:
            return "".join(opponent_hands[idx])
        elif self.history[-1] != "/":
            return "/"
        elif self.stage_i == 1:
            return "".join(boards[idx][:3])
        elif self.stage_i == 2:
            return boards[idx][3]
        elif self.stage_i == 3:
            return boards[idx][4]

    def terminal_utility(self, player: int) -> float:
        idx = self.sample_id % dataset_size  # Wrap around dataset
        winner = winners[idx]
        pot_size, _ = self._get_pot_size(self.history)

        if self.history[-1] == "f":
            pot_size, latest_bet = self._get_pot_size(self.history[:-2])
            if self.history[-3] == "bMIN":
                pot_size += latest_bet

            last_stage = self._get_last_game_stage()
            if len(last_stage) % 2 == player:
                return -pot_size / 2
            else:
                return pot_size / 2

        # Showdown
        if winner == 0:
            return 0
        if (winner == 1 and player == 0) or (winner == -1 and player == 1):
            return pot_size / 2
        return -pot_size / 2

    def _get_pot_size(self, history: List[str]) -> tuple:
        total = 0
        stage_total = 4  # Preflop: BB + SB
        latest_bet = 0

        for action in history:
            if action == "/":
                total += stage_total
                stage_total = 0
                latest_bet = 0
            elif action == "bMIN":
                latest_bet = max(2, total // 3)
                stage_total += latest_bet
            elif action == "bMAX":
                latest_bet = total
                stage_total += latest_bet
            elif action == "c":
                stage_total = 2 * latest_bet

        total += stage_total
        return total, latest_bet

    def __add__(self, action: str) -> "PostflopHistory":
        return PostflopHistory(self.history + [action], self.sample_id)

    def get_infoSet_key(self) -> List[str]:
        player = self.player()
        infoset = []
        stage_i = 0
        idx = self.sample_id % dataset_size  # Wrap around dataset

        for action in self.history:
            if action not in DISCRETE_ACTIONS:
                if action == "/":
                    stage_i += 1
                    continue
                if stage_i == 1:  # Flop
                    cluster = player_flop_clusters[idx] if player == 0 else opp_flop_clusters[idx]
                    infoset.append(str(cluster))
                elif stage_i == 2:  # Turn
                    cluster = player_turn_clusters[idx] if player == 0 else opp_turn_clusters[idx]
                    infoset.append(str(cluster))
                elif stage_i == 3:  # River
                    cluster = player_river_clusters[idx] if player == 0 else opp_river_clusters[idx]
                    infoset.append(str(cluster))
            else:
                infoset.append(action)

        return infoset


class PostflopInfoSet(InfoSet):
    """Information set for postflop decisions."""
    pass


def create_infoSet(infoSet_key: List[str], actions: List[str], player: int) -> PostflopInfoSet:
    return PostflopInfoSet(infoSet_key, actions, player)


def create_history(sample_id: int) -> PostflopHistory:
    return PostflopHistory(sample_id=sample_id)


def evaluate_winner(board: List[str], player_hand: List[str], opponent_hand: List[str]) -> int:
    p1_score = evaluate_cards(*(board + player_hand))
    p2_score = evaluate_cards(*(board + opponent_hand))
    if p1_score < p2_score:
        return 1
    elif p1_score > p2_score:
        return -1
    return 0


def generate_dataset(num_samples: int = 50000) -> None:
    """Generate training dataset with pre-computed clusters."""
    global boards, player_hands, opponent_hands, winners, dataset_size
    global player_flop_clusters, player_turn_clusters, player_river_clusters
    global opp_flop_clusters, opp_turn_clusters, opp_river_clusters

    print("Generating hands...")
    all_boards = []
    all_player_hands = []
    all_opponent_hands = []

    for _ in tqdm(range(num_samples), desc="Dealing"):
        deck = []
        for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
            for suit in ["h", "d", "s", "c"]:
                deck.append(rank + suit)
        random.shuffle(deck)

        all_boards.append(deck[:5])
        all_player_hands.append(deck[5:7])
        all_opponent_hands.append(deck[7:9])

    boards = all_boards
    player_hands = all_player_hands
    opponent_hands = all_opponent_hands

    print("Computing clusters (this may take a while)...")

    # Compute clusters for each hand
    def compute_clusters(hand, board):
        flop = predict_cluster_fast(hand + board[:3], NUM_FLOP_CLUSTERS)
        turn = predict_cluster_fast(hand + board[:4], NUM_TURN_CLUSTERS)
        river = predict_cluster_fast(hand + board, NUM_RIVER_CLUSTERS)
        return flop, turn, river

    player_results = Parallel(n_jobs=-1)(
        delayed(compute_clusters)(ph, b)
        for ph, b in tqdm(zip(player_hands, boards), total=num_samples, desc="Player clusters")
    )
    opp_results = Parallel(n_jobs=-1)(
        delayed(compute_clusters)(oh, b)
        for oh, b in tqdm(zip(opponent_hands, boards), total=num_samples, desc="Opponent clusters")
    )

    player_flop_clusters = [r[0] for r in player_results]
    player_turn_clusters = [r[1] for r in player_results]
    player_river_clusters = [r[2] for r in player_results]
    opp_flop_clusters = [r[0] for r in opp_results]
    opp_turn_clusters = [r[1] for r in opp_results]
    opp_river_clusters = [r[2] for r in opp_results]

    print("Computing winners...")
    winners = Parallel(n_jobs=-1)(
        delayed(evaluate_winner)(b, ph, oh)
        for b, ph, oh in tqdm(zip(boards, player_hands, opponent_hands), total=num_samples, desc="Winners")
    )

    dataset_size = num_samples
    print(f"Dataset size: {dataset_size}")


def main():
    parser = argparse.ArgumentParser(description="Train postflop CFR strategy")
    parser.add_argument("-i", "--iterations", type=int, default=50000,
                        help="CFR iterations per batch")
    parser.add_argument("-b", "--batches", type=int, default=1,
                        help="Number of training batches")
    parser.add_argument("-s", "--samples", type=int, default=10000,
                        help="Samples per batch (lower due to computation)")
    parser.add_argument("-o", "--output", type=str, default="models/postflop_infoSets.joblib",
                        help="Output file path")
    args = parser.parse_args()

    cfr = CFR(create_infoSet, create_history, iterations=args.iterations)

    for batch in range(args.batches):
        print(f"\n=== Batch {batch + 1}/{args.batches} ===")
        generate_dataset(args.samples)
        cfr.solve(method="vanilla")
        cfr.export_infoSets(args.output)

    print(f"\nTraining complete! Model saved to {args.output}")


if __name__ == "__main__":
    main()
