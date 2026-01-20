This is the brain behind the poker calculator. 

The algorithm is Couterfactural Regret Minimization (CFR) to compute Nash equilibrium stratagies to, well, win. 

Here's how to start it:

Installation:

```bash
pip install -r requirements.txt
```

Basic Usage:

```python
from src.calculator import PokerCalculator

# Initialize calculator
calc = PokerCalculator("models/")

# Calculate hand equity
equity = calc.get_equity(
    hole_cards=["Ah", "Kd"],
    community_cards=["Qh", "Jd", "Ts"]
)
print(f"Win probability: {equity:.1%}")

# Get AI action recommendation
result = calc.get_ai_action(
    hole_cards=["Ah", "Kd"],
    community_cards=["Qh", "Jd", "Ts"],
    pot_size=100,
    to_call=20
)
print(f"Recommended: {result['action']}")
print(f"Hand equity: {result['equity']:.1%}")
```

Run a Game: 

```python
from src.environment import PokerEnvironment

# Create your environment
env = PokerEnvironment()
env.add_player()
env.add_ai_player()

# Start a round
env.start_new_round()

# Game loop
while not env.end_of_round():
    state = env.get_game_state()
    print(f"Pot: {state['pot']}, Your turn: {state['player_in_play'] == 0}")

    if state['player_in_play'] == 0:
        action = input("Action (f/k/c/bX): ")
        env.handle_game_stage(action)
    else:
        env.handle_game_stage()  # AI plays automatically


winners = env.get_winner_indices()
print(f"Winner: Player {winners[0]}")
```

HOWEVER, before using the AI, you need to train the strategy models:

Preflop training

```bash
python -m training.preflop_trainer -i 100000 -b 5 -o models/preflop_infoSets.joblib
```

Options:
- `-i`: CFR iterations per batch (default: 50000)
- `-b`: Number of training batches (default: 1)
- `-s`: Samples per batch (default: 50000)
- `-o`: Output file path

Postflop Training

```bash
python -m training.postflop_trainer -i 50000 -b 3 -s 10000 -o models/postflop_infoSets.joblib
```

Note: Postflop training is slower due to equity calculations. Start with fewer samples.

How it works:

CFR is a self-play algorithm that converges to Nash equilibrium in two-player zero-sum games. The AI plays against itself millions of times, tracking "regret" for each action not taken. Over time, the strategy converges to unexploitable play.


MIT License
