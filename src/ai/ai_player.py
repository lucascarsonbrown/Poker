"""
AI Player implementations for poker.
Uses pre-trained CFR strategies or equity-based heuristics.
"""

import copy
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import joblib

from src.player import Player
from src.ai.abstraction import (
    calculate_equity,
    get_preflop_cluster_id,
    predict_cluster,
)


def sample_action(strategy: Dict[str, float]) -> str:
    """Sample an action from a strategy distribution."""
    actions = list(strategy.keys())
    probs = list(strategy.values())
    return np.random.choice(actions, p=probs)


class AIPlayer(Player):
    """Base AI player class."""

    def __init__(self, balance: int):
        super().__init__(balance)
        self.is_AI = True

    def place_bet(self, observed_env) -> str:
        """Override in subclasses."""
        raise NotImplementedError


class EquityAIPlayer(AIPlayer):
    """
    AI player using equity-based heuristics.
    Good for basic play without pre-trained models.
    """

    def __init__(self, balance: int):
        super().__init__(balance)

    def place_bet(self, observed_env) -> str:
        """Calculate action based on hand equity."""
        card_str = [str(card) for card in self.hand]
        community_cards = [str(card) for card in observed_env.community_cards]

        is_dealer = self == observed_env.get_player(observed_env.dealer_button_position)
        check_allowed = "k" in observed_env.valid_actions()

        action = self._get_action(
            card_str=card_str,
            community_cards=community_cards,
            total_pot=observed_env.total_pot_balance,
            highest_bet=observed_env.get_highest_current_bet(),
            big_blind=observed_env.BIG_BLIND,
            balance=self.player_balance,
            is_dealer=is_dealer,
            check_allowed=check_allowed,
        )

        self._process_action(action, observed_env)
        return action

    def _get_action(
        self,
        card_str: List[str],
        community_cards: List[str],
        total_pot: int,
        highest_bet: int,
        big_blind: int,
        balance: int,
        is_dealer: bool,
        check_allowed: bool,
    ) -> str:
        """Determine action based on equity."""
        equity = calculate_equity(card_str, community_cards)

        # Convert equity to action probabilities
        # fold/check, call, raise
        np_strategy = np.abs(np.array([1.0 - (equity + equity / 2.0), equity, equity / 2.0]))
        np_strategy = np_strategy / np.sum(np_strategy)

        if highest_bet == 0:
            # No bet to call
            if is_dealer:
                strategy = {
                    "k": np_strategy[0],
                    f"b{min(max(big_blind, int(total_pot / 3)), balance)}": np_strategy[2],
                    f"b{min(total_pot, balance)}": np_strategy[1],
                }
            else:
                strategy = {
                    "k": equity,
                    f"b{min(total_pot, balance)}": 1 - equity,
                }
        else:
            # There's a bet to call
            if check_allowed:
                strategy = {
                    "k": np_strategy[0],
                    f"b{min(int(1.5 * highest_bet), balance)}": np_strategy[1],
                    f"b{min(2 * highest_bet, balance)}": np_strategy[2],
                }
            elif highest_bet == balance:
                # All-in situation
                strategy = {
                    "f": np_strategy[0],
                    "c": np_strategy[1] + np_strategy[2],
                }
            else:
                strategy = {
                    "f": np_strategy[0],
                    "c": np_strategy[1],
                    f"b{min(2 * highest_bet, balance)}": np_strategy[2],
                }

        # Normalize and sample
        total = sum(strategy.values())
        strategy = {k: v / total for k, v in strategy.items()}
        return sample_action(strategy)

    def _process_action(self, action: str, observed_env) -> None:
        """Update player state based on action."""
        if action == "k":
            if observed_env.game_stage == 2:
                self.current_bet = observed_env.BIG_BLIND
            else:
                self.current_bet = 0
        elif action == "c":
            self.current_bet = observed_env.get_highest_current_bet()
        elif action.startswith("b"):
            self.current_bet = int(action[1:])


class CFRAIPlayer(AIPlayer):
    """
    AI player using pre-trained CFR strategies.
    Loads strategy from joblib files and samples actions.
    """

    def __init__(self, balance: int, model_path: str = "models"):
        super().__init__(balance)
        self.model_path = model_path
        self.preflop_infosets = None
        self.postflop_infosets = None
        self._load_models()

    def _load_models(self) -> None:
        """Load pre-trained strategy models."""
        import os

        preflop_path = os.path.join(self.model_path, "preflop_infoSets.joblib")
        postflop_path = os.path.join(self.model_path, "postflop_infoSets.joblib")

        # Try loading models, fall back gracefully if not found
        try:
            if os.path.exists(preflop_path):
                self.preflop_infosets = joblib.load(preflop_path)
                print(f"Loaded preflop strategy from {preflop_path}")
        except Exception as e:
            print(f"Warning: Could not load preflop model: {e}")

        try:
            if os.path.exists(postflop_path):
                self.postflop_infosets = joblib.load(postflop_path)
                print(f"Loaded postflop strategy from {postflop_path}")
        except Exception as e:
            print(f"Warning: Could not load postflop model: {e}")

    def place_bet(self, observed_env) -> str:
        """Get action from trained strategy."""
        card_str = [str(card) for card in self.hand]
        community_cards = [str(card) for card in observed_env.community_cards]

        is_dealer = self == observed_env.get_player(observed_env.dealer_button_position)
        check_allowed = "k" in observed_env.valid_actions()

        action = self._get_action(
            history=observed_env.history,
            card_str=card_str,
            community_cards=community_cards,
            highest_bet=observed_env.get_highest_current_bet(),
            stage_pot=observed_env.stage_pot_balance,
            total_pot=observed_env.total_pot_balance,
            balance=self.player_balance,
            big_blind=observed_env.BIG_BLIND,
            is_dealer=is_dealer,
            check_allowed=check_allowed,
        )

        self._process_action(action, observed_env)
        return action

    def _get_action(
        self,
        history: List[str],
        card_str: List[str],
        community_cards: List[str],
        highest_bet: int,
        stage_pot: int,
        total_pot: int,
        balance: int,
        big_blind: int,
        is_dealer: bool,
        check_allowed: bool,
    ) -> str:
        """Get action from trained strategy or fall back to heuristics."""
        smallest_bet = big_blind // 2

        if len(community_cards) == 0:  # Preflop
            if self.preflop_infosets is None:
                # Fall back to equity-based play
                return self._get_heuristic_action(
                    card_str, community_cards, total_pot,
                    highest_bet, big_blind, balance, is_dealer, check_allowed
                )

            abstracted_history = self._abstract_preflop_history(history, big_blind)
            infoset_key = self._build_preflop_infoset_key(abstracted_history)

            if infoset_key not in self.preflop_infosets:
                return self._get_heuristic_action(
                    card_str, community_cards, total_pot,
                    highest_bet, big_blind, balance, is_dealer, check_allowed
                )

            data = self.preflop_infosets[infoset_key]
            # Handle both old (InfoSet object) and new (dict) formats
            if isinstance(data, dict):
                strategy = data.get("strategy", data)
            else:
                strategy = data.get_average_strategy()
            abstract_action = sample_action(strategy)

            # Map abstract action to concrete action
            if abstract_action == "bMIN":
                return f"b{max(big_blind, stage_pot)}"
            elif abstract_action == "bMID":
                return f"b{max(big_blind, 2 * stage_pot)}"
            elif abstract_action == "bMAX":
                return f"b{balance}"
            else:
                return abstract_action

        else:  # Postflop
            if self.postflop_infosets is None:
                return self._get_heuristic_action(
                    card_str, community_cards, total_pot,
                    highest_bet, big_blind, balance, is_dealer, check_allowed
                )

            abstracted_history = self._abstract_postflop_history(history, big_blind)
            infoset_key = self._build_postflop_infoset_key(
                abstracted_history, card_str, community_cards
            )

            if infoset_key not in self.postflop_infosets:
                return self._get_heuristic_action(
                    card_str, community_cards, total_pot,
                    highest_bet, big_blind, balance, is_dealer, check_allowed
                )

            data = self.postflop_infosets[infoset_key]
            # Handle both old (InfoSet object) and new (dict) formats
            if isinstance(data, dict):
                strategy = data.get("strategy", data)
            else:
                strategy = data.get_average_strategy()
            abstract_action = sample_action(strategy)

            # Map abstract action to concrete action
            if abstract_action == "bMIN":
                return f"b{max(big_blind, int(total_pot / 3 / smallest_bet) * smallest_bet)}"
            elif abstract_action == "bMAX":
                return f"b{min(total_pot, balance)}"
            else:
                return abstract_action

    def _get_heuristic_action(
        self,
        card_str: List[str],
        community_cards: List[str],
        total_pot: int,
        highest_bet: int,
        big_blind: int,
        balance: int,
        is_dealer: bool,
        check_allowed: bool,
    ) -> str:
        """Fallback to equity-based heuristics."""
        equity_player = EquityAIPlayer(balance)
        return equity_player._get_action(
            card_str, community_cards, total_pot,
            highest_bet, big_blind, balance, is_dealer, check_allowed
        )

    def _process_action(self, action: str, observed_env) -> None:
        """Update player state based on action."""
        if action == "k":
            if observed_env.game_stage == 2:
                self.current_bet = observed_env.BIG_BLIND
            else:
                self.current_bet = 0
        elif action == "c":
            self.current_bet = observed_env.get_highest_current_bet()
        elif action.startswith("b"):
            self.current_bet = int(action[1:])

    def _abstract_preflop_history(self, history: List[str], big_blind: int) -> List[str]:
        """Abstract preflop betting history."""
        stage = copy.deepcopy(history)
        abstracted = stage[:2]  # Hole cards

        if len(stage) >= 6 and stage[3] != "c":
            # Long betting sequence - simplify
            if len(stage) % 2 == 0:
                abstracted += ["bMAX"]
            else:
                abstracted += ["bMIN", "bMAX"]
        else:
            bet_size = big_blind
            pot_total = big_blind + big_blind // 2

            for action in stage[2:]:
                if action.startswith("b"):
                    bet_size = int(action[1:])

                    if abstracted[-1] == "bMIN":
                        if bet_size <= 2 * pot_total:
                            abstracted.append("bMID")
                        else:
                            abstracted.append("bMAX")
                    elif abstracted[-1] == "bMID":
                        abstracted.append("bMAX")
                    elif abstracted[-1] == "bMAX":
                        if len(abstracted) >= 2 and abstracted[-2] == "bMID":
                            abstracted[-2] = "bMIN"
                        abstracted[-1] = "bMID"
                        abstracted.append("bMAX")
                    else:
                        if bet_size <= pot_total:
                            abstracted.append("bMIN")
                        elif bet_size <= 2 * pot_total:
                            abstracted.append("bMID")
                        else:
                            abstracted.append("bMAX")

                    pot_total += bet_size
                elif action == "c":
                    pot_total = 2 * bet_size
                    abstracted.append("c")
                else:
                    abstracted.append(action)

        return abstracted

    def _abstract_postflop_history(self, history: List[str], big_blind: int) -> List[str]:
        """Abstract postflop betting history."""
        history = copy.deepcopy(history)
        pot_total = big_blind * 2

        # Calculate preflop pot
        flop_start = history.index("/") if "/" in history else len(history)
        for action in history[:flop_start]:
            if action.startswith("b"):
                pot_total = 2 * int(action[1:])

        # Abstract postflop
        abstracted = history[:2]  # Hole cards

        if "/" not in history:
            return abstracted

        stage_start = flop_start
        latest_bet = 0

        while True:
            abstracted.append("/")
            stage = self._get_stage(history[stage_start + 1:])

            if len(stage) >= 4 and (len(stage) < 1 or stage[-1] != "c"):
                # Long betting - simplify
                if stage:
                    abstracted.append(stage[0])
                if len(stage) % 2 == 0:
                    abstracted.append("bMAX")
                else:
                    abstracted.extend(["bMIN", "bMAX"])
            else:
                for action in stage:
                    if action.startswith("b"):
                        bet_size = int(action[1:])
                        latest_bet = bet_size

                        if abstracted[-1] == "bMIN":
                            abstracted.append("bMAX")
                        elif abstracted[-1] == "bMAX":
                            abstracted[-1] = "bMIN"
                            abstracted.append("bMAX")
                        else:
                            if bet_size >= pot_total:
                                abstracted.append("bMAX")
                            else:
                                abstracted.append("bMIN")

                        pot_total += bet_size
                    elif action == "c":
                        pot_total += latest_bet
                        abstracted.append("c")
                    else:
                        abstracted.append(action)

            # Check for next stage
            remaining = history[stage_start + 1:]
            if "/" not in remaining:
                break
            stage_start = remaining.index("/") + (stage_start + 1)

        return abstracted

    def _get_stage(self, history: List[str]) -> List[str]:
        """Get betting actions for current stage."""
        if "/" in history:
            return history[:history.index("/")]
        return history

    def _build_preflop_infoset_key(self, abstracted_history: List[str]) -> str:
        """Build infoset key for preflop."""
        DISCRETE_ACTIONS = {"k", "bMIN", "bMID", "bMAX", "c", "f"}

        # Get player's cards (assume we're player based on history length)
        player_idx = len(abstracted_history) % 2
        cards = abstracted_history[player_idx] if player_idx < len(abstracted_history) else ""

        infoset = [str(get_preflop_cluster_id(cards))]
        for action in abstracted_history:
            if action in DISCRETE_ACTIONS:
                infoset.append(action)

        return "".join(infoset)

    def _build_postflop_infoset_key(
        self,
        abstracted_history: List[str],
        card_str: List[str],
        community_cards: List[str]
    ) -> str:
        """Build infoset key for postflop."""
        DISCRETE_ACTIONS = {"k", "bMIN", "bMAX", "c", "f"}

        infoset = []
        stage = 0

        for action in abstracted_history:
            if action not in DISCRETE_ACTIONS and action != "/":
                if action.startswith("/"):
                    continue
                if stage == 1:  # Flop
                    cluster = predict_cluster(card_str + community_cards[:3])
                    infoset.append(str(cluster))
                elif stage == 2:  # Turn
                    cluster = predict_cluster(card_str + community_cards[:4])
                    infoset.append(str(cluster))
                elif stage == 3:  # River
                    cluster = predict_cluster(card_str + community_cards)
                    infoset.append(str(cluster))
            elif action == "/":
                stage += 1
            else:
                infoset.append(action)

        return "".join(infoset)
