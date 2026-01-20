"""
Consolidated deck implementation for poker.
Used by both the game environment and training.
"""

import random
from typing import List, Optional

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["c", "d", "h", "s"]  # clubs, diamonds, hearts, spades
SUIT_NAMES = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}

RANK_VALUES = {
    "A": 14, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13
}


def create_deck(excluded_cards: Optional[List[str]] = None, shuffle: bool = True) -> List[str]:
    """
    Create a deck of cards as strings (e.g., "Ah", "Kd", "2c").

    Args:
        excluded_cards: Cards to exclude from the deck (e.g., ["Ah", "Kd"])
        shuffle: Whether to shuffle the deck

    Returns:
        List of card strings
    """
    excluded = set(excluded_cards) if excluded_cards else set()
    deck = []

    for rank in RANKS:
        for suit in SUITS:
            card = rank + suit
            if card not in excluded:
                deck.append(card)

    if shuffle:
        random.shuffle(deck)

    return deck


def parse_card(card_str: str) -> tuple:
    """
    Parse a card string into (rank, suit).

    Args:
        card_str: Card string like "Ah" or "Td"

    Returns:
        Tuple of (rank_char, suit_char)
    """
    if len(card_str) == 2:
        return card_str[0], card_str[1]
    elif len(card_str) == 3:  # Handle "10h" format
        return "T", card_str[2]
    else:
        raise ValueError(f"Invalid card string: {card_str}")


def card_to_index(card_str: str) -> int:
    """
    Convert card string to 0-51 index.
    Order: [Ac, Ad, Ah, As, 2c, 2d, 2h, 2s, ..., Kc, Kd, Kh, Ks]
    """
    rank, suit = parse_card(card_str)
    rank_val = RANK_VALUES[rank]
    if rank_val == 14:  # Ace
        rank_val = 1
    rank_idx = rank_val - 1
    suit_idx = SUITS.index(suit)
    return rank_idx * 4 + suit_idx


def setup_scenario(n: int = 1) -> tuple:
    """
    Set up n poker scenarios with random hands.

    Returns:
        Tuple of (boards, player_hands, opponent_hands)
        Each is a list of n elements, where each element is a list of card strings.
    """
    boards = []
    player_hands = []
    opponent_hands = []

    for _ in range(n):
        deck = create_deck(shuffle=True)
        boards.append(deck[:5])
        player_hands.append(deck[5:7])
        opponent_hands.append(deck[7:9])

    return boards, player_hands, opponent_hands
