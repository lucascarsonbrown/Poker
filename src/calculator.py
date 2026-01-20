"""
Poker Calculator API.

Clean interface for poker AI functionality. Designed for integration
with external applications (mobile apps, web services, etc.)
"""

from typing import Dict, List, Optional, Tuple
import os
import joblib

from src.ai.abstraction import (
    calculate_equity,
    get_preflop_cluster_id,
    predict_cluster,
    evaluate_winner,
)


class PokerCalculator:
    """
    High-level API for poker AI calculations.

    Provides:
    - Hand equity calculation
    - AI action recommendations
    - Hand strength evaluation

    Example usage:
        calc = PokerCalculator("models/")

        # Get AI action recommendation
        result = calc.get_ai_action(
            hole_cards=["Ah", "Kd"],
            community_cards=["Qh", "Jd", "Ts"],
            pot_size=100,
            to_call=20
        )
        print(result["action"])  # e.g., "raise"
        print(result["strategy"])  # probability distribution

        # Calculate equity
        equity = calc.get_equity(["Ah", "Kd"], ["Qh", "Jd", "Ts"])
        print(f"Win probability: {equity:.1%}")
    """

    def __init__(self, model_path: str = "models/"):
        """
        Initialize the calculator.

        Args:
            model_path: Path to directory containing trained models
        """
        self.model_path = model_path
        self._preflop_infosets = None
        self._postflop_infosets = None
        self._models_loaded = False

    def _ensure_models_loaded(self) -> None:
        """Lazy load models on first use."""
        if self._models_loaded:
            return

        preflop_path = os.path.join(self.model_path, "preflop_infoSets.joblib")
        postflop_path = os.path.join(self.model_path, "postflop_infoSets.joblib")

        if os.path.exists(preflop_path):
            self._preflop_infosets = joblib.load(preflop_path)

        if os.path.exists(postflop_path):
            self._postflop_infosets = joblib.load(postflop_path)

        self._models_loaded = True

    def get_equity(
        self,
        hole_cards: List[str],
        community_cards: Optional[List[str]] = None,
        simulations: int = 2000
    ) -> float:
        """
        Calculate hand equity (winning probability).

        Args:
            hole_cards: Player's hole cards ["Ah", "Kd"]
            community_cards: Community cards (0-5 cards)
            simulations: Number of Monte Carlo simulations

        Returns:
            Probability of winning (0.0 to 1.0)
        """
        if community_cards is None:
            community_cards = []
        return calculate_equity(hole_cards, community_cards, n=simulations)

    def get_hand_cluster(
        self,
        hole_cards: List[str],
        community_cards: Optional[List[str]] = None
    ) -> int:
        """
        Get abstraction cluster ID for a hand.

        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards (determines cluster type)

        Returns:
            Cluster ID for card abstraction
        """
        if community_cards is None or len(community_cards) == 0:
            # Preflop: 169 lossless clusters
            return get_preflop_cluster_id(hole_cards)
        else:
            # Postflop: equity-based clusters
            return predict_cluster(hole_cards + community_cards)

    def get_ai_action(
        self,
        hole_cards: List[str],
        community_cards: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        pot_size: int = 0,
        to_call: int = 0,
        stack_size: int = 1000,
    ) -> Dict:
        """
        Get AI action recommendation.

        Args:
            hole_cards: Player's hole cards ["Ah", "Kd"]
            community_cards: Community cards
            history: Betting history for the hand
            pot_size: Current pot size
            to_call: Amount needed to call
            stack_size: Player's remaining stack

        Returns:
            Dictionary with:
                - action: Recommended action ("fold", "check", "call", "raise")
                - amount: Bet/raise amount (if applicable)
                - strategy: Full strategy distribution
                - equity: Hand equity
        """
        self._ensure_models_loaded()

        if community_cards is None:
            community_cards = []
        if history is None:
            history = []

        equity = self.get_equity(hole_cards, community_cards, simulations=1000)

        # Determine game stage
        is_preflop = len(community_cards) == 0

        # Try to get trained strategy
        strategy = None
        if is_preflop and self._preflop_infosets:
            strategy = self._get_preflop_strategy(hole_cards, history)
        elif not is_preflop and self._postflop_infosets:
            strategy = self._get_postflop_strategy(hole_cards, community_cards, history)

        # Fall back to equity-based strategy if no trained model
        if strategy is None:
            strategy = self._get_equity_based_strategy(equity, to_call, pot_size, stack_size)

        # Convert abstract action to concrete recommendation
        action, amount = self._strategy_to_action(strategy, pot_size, to_call, stack_size)

        return {
            "action": action,
            "amount": amount,
            "strategy": strategy,
            "equity": equity,
        }

    def compare_hands(
        self,
        board: List[str],
        hand1: List[str],
        hand2: List[str]
    ) -> int:
        """
        Compare two hands on a given board.

        Args:
            board: 5 community cards
            hand1: First player's hole cards
            hand2: Second player's hole cards

        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        return evaluate_winner(board, hand1, hand2)

    def get_valid_actions(
        self,
        to_call: int,
        min_raise: int,
        stack_size: int
    ) -> List[Dict]:
        """
        Get list of valid actions.

        Args:
            to_call: Amount to call (0 if can check)
            min_raise: Minimum raise amount
            stack_size: Player's stack

        Returns:
            List of valid action dictionaries
        """
        actions = []

        # Fold is always valid (except when checking is free)
        if to_call > 0:
            actions.append({"action": "fold", "amount": 0})

        # Check/Call
        if to_call == 0:
            actions.append({"action": "check", "amount": 0})
        else:
            actions.append({"action": "call", "amount": min(to_call, stack_size)})

        # Raise (if we have enough chips)
        if stack_size > to_call:
            raise_amount = min(max(min_raise, to_call * 2), stack_size)
            actions.append({"action": "raise", "amount": raise_amount})

            # All-in
            if stack_size > raise_amount:
                actions.append({"action": "all-in", "amount": stack_size})

        return actions

    # --- Private Methods ---

    def _get_preflop_strategy(
        self,
        hole_cards: List[str],
        history: List[str]
    ) -> Optional[Dict[str, float]]:
        """Get preflop strategy from trained model."""
        cluster_id = get_preflop_cluster_id(hole_cards)

        # Build infoset key from history
        # This is simplified - full implementation would need to abstract history
        infoset_key = str(cluster_id)
        for action in history:
            if action in ["k", "c", "f", "bMIN", "bMID", "bMAX"]:
                infoset_key += action

        if infoset_key in self._preflop_infosets:
            data = self._preflop_infosets[infoset_key]
            # Handle both old (InfoSet object) and new (dict) formats
            if isinstance(data, dict):
                return data.get("strategy", data)
            return data.get_average_strategy()
        return None

    def _get_postflop_strategy(
        self,
        hole_cards: List[str],
        community_cards: List[str],
        history: List[str]
    ) -> Optional[Dict[str, float]]:
        """Get postflop strategy from trained model."""
        cluster_id = predict_cluster(hole_cards + community_cards)

        # Build simplified infoset key
        infoset_key = str(cluster_id)
        for action in history:
            if action in ["k", "c", "f", "bMIN", "bMAX"]:
                infoset_key += action

        if infoset_key in self._postflop_infosets:
            data = self._postflop_infosets[infoset_key]
            # Handle both old (InfoSet object) and new (dict) formats
            if isinstance(data, dict):
                return data.get("strategy", data)
            return data.get_average_strategy()
        return None

    def _get_equity_based_strategy(
        self,
        equity: float,
        to_call: int,
        pot_size: int,
        stack_size: int
    ) -> Dict[str, float]:
        """Generate strategy based on equity heuristics."""
        # Pot odds calculation
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0

        if to_call == 0:
            # No bet to face
            check_prob = 1.0 - equity
            bet_prob = equity
            return {"check": check_prob, "raise": bet_prob}
        else:
            # Facing a bet
            if equity > pot_odds + 0.1:
                # Good equity - call or raise
                return {"fold": 0.0, "call": 0.6, "raise": 0.4}
            elif equity > pot_odds:
                # Marginal - mostly call
                return {"fold": 0.2, "call": 0.7, "raise": 0.1}
            else:
                # Poor equity - mostly fold
                return {"fold": 0.8, "call": 0.2, "raise": 0.0}

    def _strategy_to_action(
        self,
        strategy: Dict[str, float],
        pot_size: int,
        to_call: int,
        stack_size: int
    ) -> Tuple[str, int]:
        """Convert strategy to concrete action and amount."""
        import random

        # Sample from strategy
        actions = list(strategy.keys())
        probs = list(strategy.values())

        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)

        chosen = random.choices(actions, probs)[0]

        # Map abstract actions to concrete
        if chosen in ["f", "fold"]:
            return "fold", 0
        elif chosen in ["k", "check"]:
            return "check", 0
        elif chosen in ["c", "call"]:
            return "call", min(to_call, stack_size)
        elif chosen in ["bMIN", "raise"]:
            amount = max(to_call * 2, pot_size // 3)
            return "raise", min(amount, stack_size)
        elif chosen in ["bMID"]:
            amount = pot_size
            return "raise", min(amount, stack_size)
        elif chosen in ["bMAX", "all-in"]:
            return "raise", stack_size
        else:
            # Default to call
            return "call", min(to_call, stack_size)
