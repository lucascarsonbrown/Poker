import training.cfr as base
from typing import Dict, List, NewType
import copy

Player = NewType("Player", int)
Action = NewType("Action", str)

ACTIONS = ["R", "P", "S"]


class RPSHistory(base.History):
    def __init__(self, history: List[Action] = None):
        super().__init__(history)

    def is_terminal(self) -> bool:
        return len(self.history) == 2

    def actions(self) -> List[Action]:
        return ACTIONS

    def player(self) -> int:
        return len(self.history) % 2

    def is_chance(self) -> bool:
        return False

    def sample_chance_outcome(self) -> str:
        raise NotImplementedError("RPS has no chance nodes")

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()
        p1_idx = ACTIONS.index(self.history[0])
        p2_idx = ACTIONS.index(self.history[1])

        if p1_idx == p2_idx:
            return 0
        wins = (p1_idx + 1) % 3 == p2_idx
        return -1 if (wins and i == 0) or (not wins and i == 1) else 1

    def __add__(self, action: Action) -> "RPSHistory":
        return RPSHistory(self.history + [action])

    def get_infoSet_key(self) -> List[str]:
        history = copy.deepcopy(self.history)
        if len(history) >= 1:
            history[0] = "?"
        return history


class RPSInfoSet(base.InfoSet):
    def __init__(self, infoSet_key: List[str], actions: List[str], player: int):
        super().__init__(infoSet_key, actions, player)


def create_infoSet(infoSet_key: List[str], actions: List[str], player: int) -> RPSInfoSet:
    return RPSInfoSet(infoSet_key, actions, player)


def create_history(sample_id: int = 0) -> RPSHistory:
    return RPSHistory()


if __name__ == "__main__":
    cfr = base.CFR(create_infoSet, create_history, iterations=100_000)
    cfr.solve()

    print("\nNash equilibrium strategies:")
    for key, infoset in cfr.infoSets.items():
        strategy = infoset.get_average_strategy()
        formatted = {a: f"{p:.3f}" for a, p in strategy.items()}
        print(f"  InfoSet {key!r}: {formatted}")
