"""
Card abstraction functions for poker AI.
Runtime-only code for mapping hands to clusters.
"""

from typing import List
import random

# Use phevaluator for fast equity calculations
from phevaluator import evaluate_cards

# Cluster configuration (must match trained models)
NUM_FLOP_CLUSTERS = 50
NUM_TURN_CLUSTERS = 50
NUM_RIVER_CLUSTERS = 10

# KMeans classifiers (loaded lazily)
_kmeans_flop = None
_kmeans_turn = None


def load_kmeans_classifiers(model_path: str = "models/kmeans"):
    """Load pre-trained KMeans classifiers for card abstraction."""
    global _kmeans_flop, _kmeans_turn
    import joblib
    import os

    flop_path = os.path.join(model_path, "flop")
    turn_path = os.path.join(model_path, "turn")

    # Find latest model files
    def get_latest_file(folder):
        files = [f for f in os.listdir(folder) if f.endswith('.joblib')]
        return sorted(files)[-1] if files else None

    if os.path.exists(flop_path):
        latest = get_latest_file(flop_path)
        if latest:
            _kmeans_flop = joblib.load(os.path.join(flop_path, latest))
            print(f"Loaded flop classifier: {latest}")

    if os.path.exists(turn_path):
        latest = get_latest_file(turn_path)
        if latest:
            _kmeans_turn = joblib.load(os.path.join(turn_path, latest))
            print(f"Loaded turn classifier: {latest}")


def get_preflop_cluster_id(two_cards: str | List[str]) -> int:
    """
    Lossless preflop abstraction into 169 clusters.

    Clusters:
        1-13: Pocket pairs (AA, 22, ..., KK)
        14-91: Unsuited non-pairs
        92-169: Suited non-pairs

    Args:
        two_cards: Card string like "AhKd" or list ["Ah", "Kd"]

    Returns:
        Cluster ID (1-169)
    """
    if isinstance(two_cards, list):
        two_cards = "".join(two_cards)

    assert len(two_cards) == 4, f"Expected 4-char string, got: {two_cards}"

    RANK_VALUES = {
        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13,
    }

    rank1, suit1 = two_cards[0], two_cards[1]
    rank2, suit2 = two_cards[2], two_cards[3]

    def hash_pair(a: str, b: str) -> int:
        """Hash two different ranks to unique value 1-78."""
        first = min(RANK_VALUES[a], RANK_VALUES[b])
        second = max(RANK_VALUES[a], RANK_VALUES[b])

        # Calculate triangular number offset
        def sum_to(n):
            if n <= 1:
                return 0
            count = n - 1
            return (count * (12 + 12 - (n - 2))) // 2

        return int(sum_to(first) + (second - first))

    if rank1 == rank2:  # Pocket pair
        return RANK_VALUES[rank1]
    elif suit1 != suit2:  # Unsuited
        return 13 + hash_pair(rank1, rank2)
    else:  # Suited
        return 91 + hash_pair(rank1, rank2)


def calculate_equity(
    player_cards: List[str],
    community_cards: List[str] = None,
    n: int = 2000
) -> float:
    """
    Calculate hand equity using Monte Carlo simulation.

    Args:
        player_cards: Player's hole cards ["Ah", "Kd"]
        community_cards: Community cards (0-5 cards)
        n: Number of simulations

    Returns:
        Winning probability (0.0 to 1.0)
    """
    if community_cards is None:
        community_cards = []

    # Create deck without known cards
    excluded = set(player_cards + community_cards)
    deck = []
    for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
        for suit in ["h", "d", "s", "c"]:
            card = rank + suit
            if card not in excluded:
                deck.append(card)

    wins = 0
    remaining_community = 5 - len(community_cards)

    for _ in range(n):
        random.shuffle(deck)
        opponent_cards = deck[:2]
        new_community = deck[2:2 + remaining_community]
        full_community = community_cards + new_community

        player_score = evaluate_cards(*(player_cards + full_community))
        opponent_score = evaluate_cards(*(opponent_cards + full_community))

        if player_score < opponent_score:
            wins += 1
        elif player_score == opponent_score:
            wins += 1  # Count ties as wins for simplicity

    return wins / n


def predict_cluster(cards: List[str], use_kmeans: bool = False) -> int:
    """
    Predict cluster for a hand (postflop).

    Args:
        cards: List of cards [hole1, hole2, comm1, ...]
        use_kmeans: Use KMeans classifier (requires trained model)

    Returns:
        Cluster ID
    """
    assert isinstance(cards, list), "Cards must be a list"

    num_cards = len(cards)

    if use_kmeans and (_kmeans_flop is not None or _kmeans_turn is not None):
        if num_cards == 5:  # Flop
            return _predict_cluster_kmeans(_kmeans_flop, cards)
        elif num_cards == 6:  # Turn
            return _predict_cluster_kmeans(_kmeans_turn, cards)
        elif num_cards == 7:  # River
            return predict_cluster_fast(cards, NUM_RIVER_CLUSTERS)
    else:
        # Fall back to fast equity-based clustering
        if num_cards == 5:
            return predict_cluster_fast(cards, NUM_FLOP_CLUSTERS)
        elif num_cards == 6:
            return predict_cluster_fast(cards, NUM_TURN_CLUSTERS)
        elif num_cards == 7:
            return predict_cluster_fast(cards, NUM_RIVER_CLUSTERS)

    raise ValueError(f"Invalid number of cards: {num_cards}")


def predict_cluster_fast(cards: List[str], total_clusters: int = 10, n: int = 2000) -> int:
    """
    Fast cluster prediction using equity only.

    Args:
        cards: List of cards [hole1, hole2, community...]
        total_clusters: Number of clusters
        n: Monte Carlo samples for equity

    Returns:
        Cluster ID (0 to total_clusters-1)
    """
    equity = calculate_equity(cards[:2], cards[2:], n=n)
    cluster = min(total_clusters - 1, int(equity * total_clusters))
    return cluster


def _predict_cluster_kmeans(kmeans_classifier, cards: List[str], n: int = 200) -> int:
    """Predict cluster using pre-trained KMeans classifier."""
    equity_dist = _calculate_equity_distribution(cards[:2], cards[2:], n=n)
    prediction = kmeans_classifier.predict([equity_dist])
    return prediction[0]


def _calculate_equity_distribution(
    player_cards: List[str],
    community_cards: List[str] = None,
    bins: int = 10,
    n: int = 200
) -> List[float]:
    """
    Calculate equity distribution histogram.

    Returns:
        List of bin probabilities (sums to 1.0)
    """
    if community_cards is None:
        community_cards = []

    equity_hist = [0.0] * bins

    # Create deck without known cards
    excluded = set(player_cards + community_cards)
    deck = []
    for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
        for suit in ["h", "d", "s", "c"]:
            card = rank + suit
            if card not in excluded:
                deck.append(card)

    for _ in range(n):
        random.shuffle(deck)

        if len(community_cards) == 0:
            # Sample flop
            sample_community = deck[:3]
            equity = calculate_equity(player_cards, sample_community, n=200)
        elif len(community_cards) < 5:
            # Sample next card
            sample_community = community_cards + deck[:1]
            equity = calculate_equity(player_cards, sample_community, n=100)
        else:
            equity = calculate_equity(player_cards, community_cards, n=100)

        bin_idx = min(int(equity * bins), bins - 1)
        equity_hist[bin_idx] += 1.0

    # Normalize
    total = sum(equity_hist)
    if total > 0:
        equity_hist = [h / total for h in equity_hist]

    return equity_hist


def evaluate_winner(
    board: List[str],
    player_hand: List[str],
    opponent_hand: List[str]
) -> int:
    """
    Determine winner between two hands.

    Args:
        board: 5 community cards
        player_hand: Player's 2 hole cards
        opponent_hand: Opponent's 2 hole cards

    Returns:
        1 if player wins, -1 if opponent wins, 0 if tie
    """
    player_score = evaluate_cards(*(board + player_hand))
    opponent_score = evaluate_cards(*(board + opponent_hand))

    if player_score < opponent_score:
        return 1
    elif player_score > opponent_score:
        return -1
    else:
        return 0
