# Poker AI

Texas Hold'em AI using Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Calculate hand equity

```python
from src.calculator import PokerCalculator

calc = PokerCalculator("models/")

equity = calc.get_equity(
    hole_cards=["Ah", "Kd"],
    community_cards=["Qh", "Jd", "Ts"]
)
print(f"Win probability: {equity:.1%}")
```

### Get an AI action recommendation

```python
result = calc.get_ai_action(
    hole_cards=["Ah", "Kd"],
    community_cards=["Qh", "Jd", "Ts"],
    pot_size=100,
    to_call=20
)
print(f"Recommended: {result['action']}")
print(f"Hand equity: {result['equity']:.1%}")
```

### Run a full game loop

```python
from src.environment import PokerEnvironment

env = PokerEnvironment()
env.add_player()
env.add_ai_player()
env.start_new_round()

while not env.end_of_round():
    state = env.get_game_state()
    if state['player_in_play'] == 0:
        action = input("Action (f/k/c/bX): ")
        env.handle_game_stage(action)
    else:
        env.handle_game_stage()  # AI plays automatically

print(f"Winner: Player {env.get_winner_indices()[0]}")
```

## Training

The AI requires pre-trained strategy models. Without them, `PokerCalculator` falls back to equity-based heuristics.

```bash
# Preflop strategy (~5-10 min)
python -m training.preflop_trainer -i 100000 -b 5 -o models/preflop_infoSets.joblib

# Postflop strategy (~30-60 min, slower due to equity calculations)
python -m training.postflop_trainer -i 50000 -b 3 -s 10000 -o models/postflop_infoSets.joblib
```

See [training/README.md](training/README.md) for all options and how training works.

## Project Structure

```text
Poker/
├── src/          # Runtime library (calculator, environment, evaluator, AI)
├── training/     # CFR training scripts and base algorithm
├── examples/     # Standalone demos (RPS as a CFR sanity check)
└── models/       # Trained strategy files (.joblib)
```

## Card Format

Two-character strings: rank + suit.

| Type  | Values                        |
|-------|-------------------------------|
| Ranks | `A` `2`-`9` `T` `J` `Q` `K` |
| Suits | `h` `d` `c` `s`              |

Examples: `"Ah"` (Ace of hearts), `"Td"` (Ten of diamonds)

## Action Format

| Action | Meaning                         |
|--------|---------------------------------|
| `f`    | Fold                            |
| `k`    | Check                           |
| `c`    | Call                            |
| `bX`   | Bet/raise X chips (e.g. `b100`) |

---

MIT License
