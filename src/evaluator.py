"""
Hand evaluation using bit representation for Texas Hold'Em Poker.
Uses bit manipulation for fast hand strength calculation.
"""

from typing import List
import random

# Bit position lookup table for fast evaluation
def generate_table():
    table = {}
    for i in range(57):
        table[1 << i] = i
    return table

BIT_POSITION_TABLE = generate_table()

# Card constants
CARD_SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
CARD_SUITS_DICT = {"Clubs": 0, "Diamonds": 1, "Hearts": 2, "Spades": 3}
CARD_RANKS = list(range(2, 15))  # 2-14 (Ace = 14)

# Bit masks for suit detection
BIT_MASK_1 = int("0x11111111111111", 16)
BIT_MASK_2 = int("0x22222222222222", 16)
BIT_MASK_4 = int("0x44444444444444", 16)
BIT_MASK_8 = int("0x88888888888888", 16)
BIT_MASKS = [BIT_MASK_1, BIT_MASK_2, BIT_MASK_4, BIT_MASK_8]

CARD_BIT_SUITS_DICT = {1: "Clubs", 2: "Diamonds", 4: "Hearts", 8: "Spades"}

RANK_KEY = {
    "A": 14, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "T": 10, "10": 10,
    "J": 11, "Q": 12, "K": 13,
}

INVERSE_RANK_KEY = {
    14: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
    7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K",
}

SUIT_KEY = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}


class Card:
    """
    Immutable card representation.

    Can initialize as:
        Card("Ah")  - Ace of hearts
        Card(rank=10, suit="Spades") - Ten of spades
    """

    def __init__(self, rank_suit=None, rank=14, suit="Spades", generate_random=False):
        if rank_suit:
            self.__rank = RANK_KEY[rank_suit[:-1]]
            self.__suit = SUIT_KEY[rank_suit[-1].lower()]
        else:
            self.__rank = rank
            self.__suit = suit

        if generate_random:
            self.__rank = random.choice(CARD_RANKS)
            self.__suit = random.choice(CARD_SUITS)

        if self.__rank not in CARD_RANKS:
            raise ValueError(f"Invalid rank: {self.__rank}")
        if self.__suit not in CARD_SUITS:
            raise ValueError(f"Invalid suit: {self.__suit}")

    @property
    def rank(self):
        return self.__rank

    @property
    def suit(self):
        return self.__suit

    @property
    def idx(self):
        """Card index 0-51."""
        rank = self.__rank if self.__rank != 14 else 1
        return (rank - 1) * 4 + CARD_SUITS_DICT[self.__suit]

    def __str__(self):
        return INVERSE_RANK_KEY[self.rank] + self.suit[0].lower()

    def __repr__(self):
        return self.__str__()


class Deck:
    """Standard 52-card deck."""

    def __init__(self):
        self.__cards: List[Card] = []
        self.reset_deck()

    def shuffle(self):
        random.shuffle(self.__cards)

    def reset_deck(self):
        """Reset and shuffle the deck."""
        self.__cards = []
        for rank in CARD_RANKS:
            for suit in CARD_SUITS:
                self.__cards.append(Card(rank=rank, suit=suit))
        random.shuffle(self.__cards)

    @property
    def total_remaining_cards(self):
        return len(self.__cards)

    def draw(self) -> Card:
        """Draw a card from the deck."""
        return self.__cards.pop()


class CombinedHand:
    """
    Combined hand representation (hole cards + community cards).
    Uses bit manipulation for fast evaluation.
    """

    def __init__(self, hand: List[Card] = None):
        self.hand: List[Card] = hand if hand else []
        self.hand_strength = 0
        self.h = 0  # Binary representation
        self.comparator = []

        if self.hand:
            self._update_binary_representation()

    def __str__(self):
        return ", ".join(str(c) for c in self.hand)

    def __len__(self):
        return len(self.hand)

    def as_list(self) -> List[str]:
        """Return hand as list of card strings."""
        return [str(c) for c in self.hand]

    def _update_binary_representation(self):
        """Update the binary representation for evaluation."""
        self.h = 0
        for card in self.hand:
            self.h += (1 << int(4 * (card.rank - 1)) << CARD_SUITS_DICT[card.suit])
            if card.rank == 14:  # Ace also at position 1
                self.h += 1 << CARD_SUITS_DICT[card.suit]

    def add_combined_hands(self, *hands):
        """Add cards from other CombinedHand objects."""
        for hand in hands:
            for card in hand.hand:
                self.hand.append(card)
        self._update_binary_representation()

    def add_cards(self, *cards):
        """Add individual cards."""
        for card in cards:
            self.hand.append(card)
        self._update_binary_representation()

    def get_binary_representation(self):
        return bin(self.h)

    def get_hand_strength(self, verbose=False):
        """
        Calculate hand strength (1=Royal Flush, 10=High Card).

        Sets self.hand_strength and self.comparator for tie-breaking.
        """
        h = self.h

        # 1 - Royal Flush
        royal_flush = (h >> 36) & (h >> 40) & (h >> 44) & (h >> 48) & (h >> 52)
        if royal_flush:
            if verbose:
                print("Royal Flush of", CARD_BIT_SUITS_DICT[royal_flush])
            self.hand_strength = 1
            return

        # 2 - Straight Flush
        hh = (h) & (h >> 4) & (h >> 8) & (h >> 12) & (h >> 16)
        if hh:
            highest_low_card = 0
            checker = hh
            for i in range(1, 11):
                if checker & 15:
                    highest_low_card = i
                checker = checker >> 4
            self.hand_strength = 2
            self.comparator = [highest_low_card]
            if verbose:
                print("Straight Flush starting with:", self.comparator[0])
            return

        # 3 - Four of A Kind
        h_shifted = self.h >> 4
        hh = (h_shifted) & (h_shifted >> 1) & (h_shifted >> 2) & (h_shifted >> 3) & BIT_MASK_1
        if hh:
            four_of_a_kind = BIT_POSITION_TABLE[hh] // 4 + 2
            kicker = max(c.rank for c in self.hand if c.rank != four_of_a_kind)
            self.hand_strength = 3
            self.comparator = [four_of_a_kind, kicker]
            if verbose:
                print("Four of a kind:", self.comparator[0], "Kicker:", self.comparator[1])
            return

        # 4 - Full House
        threes, threes_hh = self._check_threes()
        twos = self._check_twos(threes_hh)
        if (len(threes) >= 1 and len(twos) >= 1) or len(threes) > 1:
            self.hand_strength = 4
            if len(threes) > 1:
                max_three = max(threes)
                max_two = max(twos) if twos else 0
                for three in threes:
                    if three != max_three:
                        max_two = max(max_two, three)
                self.comparator = [max_three, max_two]
            else:
                self.comparator = [max(threes), max(twos)]
            if verbose:
                print(f"Full house: {self.comparator[0]}s full of {self.comparator[1]}s")
            return

        # 5 - Flush
        h_shifted = self.h >> 4
        for idx, MASK in enumerate(BIT_MASKS):
            hh = h_shifted & MASK
            if bin(hh).count("1") >= 5:
                suit = CARD_SUITS[idx]
                final_hand = sorted([c.rank for c in self.hand if c.suit == suit], reverse=True)[:5]
                self.hand_strength = 5
                self.comparator = final_hand
                if verbose:
                    print("Flush with hand:", self.comparator)
                return

        # 6 - Straight
        hh1 = h & BIT_MASK_1
        hh1 = (hh1) | (hh1 << 1) | (hh1 << 2) | (hh1 << 3)
        hh2 = h & BIT_MASK_2
        hh2 = (hh2) | (hh2 >> 1) | (hh2 << 1) | (hh2 << 2)
        hh4 = h & BIT_MASK_4
        hh4 = (hh4) | (hh4 << 1) | (hh4 >> 1) | (hh4 >> 2)
        hh8 = h & BIT_MASK_8
        hh8 = (hh8) | (hh8 >> 1) | (hh8 >> 2) | (hh8 >> 3)
        hh = hh1 | hh2 | hh4 | hh8
        hh = (hh) & (hh >> 4) & (hh >> 8) & (hh >> 12) & (hh >> 16)
        if hh:
            low_card = 1
            n = hh
            for curr in range(1, 15):
                if n & 1:
                    low_card = curr
                n = n >> 4
            self.hand_strength = 6
            self.comparator = [low_card]
            if verbose:
                print("Straight starting from:", self.comparator[0])
            return

        # 7 - Three of A Kind
        if len(threes) == 1:
            self.hand_strength = 7
            kickers = sorted([c.rank for c in self.hand if c.rank != threes[0]], reverse=True)[:2]
            self.comparator = [threes[0]] + kickers
            if verbose:
                print("Three of a kind:", self.comparator[0], "Kickers:", self.comparator[1:])
            return

        # 8 - Two Pair / 9 - One Pair
        if len(twos) >= 1:
            twos.sort(reverse=True)
            if len(twos) >= 2:
                self.hand_strength = 8
                kicker = max(c.rank for c in self.hand if c.rank not in twos[:2])
                self.comparator = [twos[0], twos[1], kicker]
                if verbose:
                    print("Two Pair:", self.comparator[:2], "Kicker:", self.comparator[2])
            else:
                self.hand_strength = 9
                kickers = sorted([c.rank for c in self.hand if c.rank != twos[0]], reverse=True)[:3]
                self.comparator = [twos[0]] + kickers
                if verbose:
                    print("One Pair:", self.comparator[0], "Kickers:", self.comparator[1:])
            return

        # 10 - High Card
        self.hand_strength = 10
        self.comparator = sorted([c.rank for c in self.hand], reverse=True)[:5]
        if verbose:
            print("High Card:", self.comparator)

    def _check_threes(self):
        """Find all three-of-a-kind ranks."""
        h = self.h >> 4
        hh = (
            ((h) & (h >> 1) & (h >> 2))
            | ((h >> 1) & (h >> 2) & (h >> 3))
            | ((h) & (h >> 1) & (h >> 3))
            | ((h) & (h >> 2) & (h >> 3))
        ) & BIT_MASK_1

        threes = []
        if hh:
            n = hh
            for low_card in range(2, 15):
                if n & 1:
                    threes.append(low_card)
                n = n >> 4
        return threes, hh

    def _check_twos(self, threes_hh):
        """Find all pair ranks (excluding three-of-a-kind)."""
        h = self.h >> 4
        hh = (
            ((h) & (h >> 1))
            | ((h) & (h >> 2))
            | ((h) & (h >> 3))
            | ((h >> 1) & (h >> 2))
            | ((h >> 1) & (h >> 3))
            | ((h >> 2) & (h >> 3))
        ) & BIT_MASK_1
        hh = hh ^ threes_hh

        twos = []
        if hh:
            n = hh
            for low_card in range(2, 15):
                if n & 1:
                    twos.append(low_card)
                n = n >> 4
        return twos


class Evaluator:
    """Evaluates and compares poker hands to determine winners."""

    def __init__(self):
        self.hands: List[CombinedHand] = []

    def add_hands(self, *combined_hands: CombinedHand):
        """Add hands to evaluate."""
        for hand in combined_hands:
            self.hands.append(hand)

    def clear_hands(self):
        self.hands = []

    def __str__(self):
        return "\n".join(str(h) for h in self.hands)

    def get_winner(self) -> List[int]:
        """
        Determine the winner(s).

        Returns:
            List of indices of winning players (multiple if tie)
        """
        for hand in self.hands:
            hand.get_hand_strength()

        hand_strengths = [h.hand_strength for h in self.hands]
        best_hand_val = min(hand_strengths)
        potential_winners = [i for i, x in enumerate(hand_strengths) if x == best_hand_val]

        if len(potential_winners) == 1:
            return potential_winners

        # Handle ties by comparing comparators
        return self._resolve_tie(potential_winners, best_hand_val)

    def _resolve_tie(self, potential_winners: List[int], best_hand_val: int) -> List[int]:
        """Resolve ties using comparator values."""
        if best_hand_val == 1:  # Royal Flush - always tie
            return potential_winners

        # Compare by comparator values
        comparator_len = len(self.hands[potential_winners[0]].comparator)
        for i in range(comparator_len):
            best_val = max(self.hands[w].comparator[i] for w in potential_winners)
            potential_winners = [w for w in potential_winners if self.hands[w].comparator[i] == best_val]
            if len(potential_winners) == 1:
                return potential_winners

        return potential_winners
