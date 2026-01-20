"""
Counterfactual Regret Minimization (CFR) algorithm for poker.
Base classes for training Nash equilibrium strategies.
"""

from typing import Dict, List, Callable
from tqdm import tqdm
import joblib


class History:
    """
    Base class for game history/state.
    Subclass this for specific games (poker variants).
    """

    def __init__(self, history: List[str] = None):
        self.history = history if history else []

    def is_terminal(self) -> bool:
        """Check if this is a terminal (end) state."""
        raise NotImplementedError()

    def actions(self) -> List[str]:
        """Get available actions from this state."""
        raise NotImplementedError()

    def player(self) -> int:
        """
        Get current player to act.
        Returns -1 for chance nodes.
        """
        raise NotImplementedError()

    def is_chance(self) -> bool:
        """Check if this is a chance node (e.g., dealing cards)."""
        return self.player() == -1

    def sample_chance_outcome(self) -> str:
        """Sample an outcome from a chance node."""
        raise NotImplementedError()

    def terminal_utility(self, player: int) -> float:
        """Get utility (payoff) for a player at terminal state."""
        raise NotImplementedError()

    def __add__(self, action: str) -> "History":
        """Create new history by appending an action."""
        raise NotImplementedError()

    def get_infoSet_key(self) -> List[str]:
        """
        Get information set key.
        This abstracts away hidden information (opponent's cards, etc.)
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self.history)


class InfoSet:
    """
    Information Set - represents what a player knows at a decision point.
    Stores regret and strategy values for CFR.
    """

    def __init__(
        self,
        infoSet_key: List[str],
        actions: List[str],
        player: int
    ):
        self.infoSet = infoSet_key
        self._actions = actions
        self._player = player

        # CFR values
        self.regret = {a: 0.0 for a in actions}
        self.strategy = {a: 1.0 / len(actions) for a in actions}
        self.cumulative_strategy = {a: 0.0 for a in actions}

    def __repr__(self) -> str:
        return str(self.infoSet)

    def actions(self) -> List[str]:
        return self._actions

    def player(self) -> int:
        return self._player

    def to_dict(self) -> dict:
        return {
            "infoset": self.infoSet,
            "regret": self.regret,
            "cumulative_strategy": self.cumulative_strategy,
        }

    def get_strategy(self) -> Dict[str, float]:
        """
        Get current strategy using regret matching.
        Updates self.strategy based on regret values.
        """
        # Clamp negative regrets to zero
        positive_regret = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(positive_regret.values())

        if regret_sum > 0:
            self.strategy = {a: r / regret_sum for a, r in positive_regret.items()}
        else:
            # Uniform strategy if no positive regrets
            n = len(self._actions)
            self.strategy = {a: 1.0 / n for a in self._actions}

        return self.strategy

    def get_average_strategy(self) -> Dict[str, float]:
        """
        Get time-averaged strategy (the Nash equilibrium approximation).
        """
        strategy_sum = sum(self.cumulative_strategy.values())

        if strategy_sum > 0:
            return {a: s / strategy_sum for a, s in self.cumulative_strategy.items()}
        else:
            n = len(self._actions)
            return {a: 1.0 / n for a in self._actions}


class CFR:
    """
    Counterfactual Regret Minimization solver.
    Trains Nash equilibrium strategies for two-player zero-sum games.
    """

    def __init__(
        self,
        create_infoSet: Callable,
        create_history: Callable,
        n_players: int = 2,
        iterations: int = 100000,
    ):
        self.n_players = n_players
        self.iterations = iterations
        self.tracker_interval = max(1, iterations // 10)

        self.infoSets: Dict[str, InfoSet] = {}
        self.create_infoSet = create_infoSet
        self.create_history = create_history

    def get_infoSet(self, history: History) -> InfoSet:
        """Get or create an information set for a history."""
        infoSet_key = history.get_infoSet_key()
        actions = history.actions()
        player = history.player()

        key_str = "".join(infoSet_key)
        if key_str not in self.infoSets:
            self.infoSets[key_str] = self.create_infoSet(infoSet_key, actions, player)

        return self.infoSets[key_str]

    def vanilla_cfr(
        self,
        history: History,
        player: int,
        t: int,
        pi_0: float,
        pi_1: float,
        debug: bool = False
    ) -> float:
        """
        Vanilla CFR algorithm (recursive tree traversal).

        Args:
            history: Current game state
            player: Player whose utility we're computing
            t: Current iteration
            pi_0: Player 0's reach probability
            pi_1: Player 1's reach probability
            debug: Print debug info

        Returns:
            Expected utility for the player
        """
        if history.is_terminal():
            return history.terminal_utility(player)

        if history.is_chance():
            action = history.sample_chance_outcome()
            return self.vanilla_cfr(history + action, player, t, pi_0, pi_1, debug)

        infoSet = self.get_infoSet(history)
        assert infoSet.player() == history.player()

        # Get current strategy
        strategy = infoSet.get_strategy()

        v = 0.0
        va = {}

        # Compute counterfactual values for each action
        for a in infoSet.actions():
            if history.player() == 0:
                va[a] = self.vanilla_cfr(
                    history + a, player, t,
                    strategy[a] * pi_0, pi_1, debug
                )
            else:
                va[a] = self.vanilla_cfr(
                    history + a, player, t,
                    pi_0, strategy[a] * pi_1, debug
                )

            v += strategy[a] * va[a]

        # Update regrets and cumulative strategy for the acting player
        if history.player() == player:
            opponent_reach = pi_1 if player == 0 else pi_0
            player_reach = pi_0 if player == 0 else pi_1

            for a in infoSet.actions():
                infoSet.regret[a] += opponent_reach * (va[a] - v)
                infoSet.cumulative_strategy[a] += player_reach * strategy[a]

        if debug:
            print(f"InfoSet: {infoSet.infoSet}, Strategy: {strategy}")

        return v

    def solve(self, method: str = "vanilla", debug: bool = False) -> None:
        """
        Run CFR iterations to solve the game.

        Args:
            method: "vanilla" (only supported method)
            debug: Print debug information
        """
        util_0 = 0.0
        util_1 = 0.0

        for t in tqdm(range(self.iterations), desc="CFR Training"):
            if method == "vanilla":
                for player in range(self.n_players):
                    history = self.create_history(t)
                    if player == 0:
                        util_0 += self.vanilla_cfr(history, player, t, 1, 1, debug)
                    else:
                        util_1 += self.vanilla_cfr(history, player, t, 1, 1, debug)

            if (t + 1) % self.tracker_interval == 0:
                print(f"\nIteration {t + 1}:")
                print(f"  Avg game value P0: {util_0 / (t + 1):.4f}")
                print(f"  Avg game value P1: {util_1 / (t + 1):.4f}")
                print(f"  Total infosets: {len(self.infoSets)}")

    def export_infoSets(self, filename: str = "infoSets.joblib") -> None:
        """Save trained information sets to file."""
        # Export as plain dictionaries to avoid pickle class issues
        export_data = {}
        for key, infoset in self.infoSets.items():
            export_data[key] = {
                "strategy": infoset.get_average_strategy(),
                "actions": infoset.actions(),
            }
        joblib.dump(export_data, filename)
        print(f"Saved {len(export_data)} infosets to {filename}")

    def load_infoSets(self, filename: str) -> None:
        """Load information sets from file."""
        self.infoSets = joblib.load(filename)
        print(f"Loaded {len(self.infoSets)} infosets from {filename}")
