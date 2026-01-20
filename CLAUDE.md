# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Texas Hold'em Poker AI using Counterfactual Regret Minimization (CFR) to compute Nash equilibrium strategies. Designed as a Python API backend for poker applications (mobile apps, web services).

## Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models (required before using AI)
```bash
# Preflop strategy (~5-10 min)
python -m training.preflop_trainer -i 100000 -b 5 -o models/preflop_infoSets.joblib

# Postflop strategy (~30-60 min)
python -m training.postflop_trainer -i 50000 -b 3 -s 10000 -o models/postflop_infoSets.joblib
```

### Run Tests
```bash
python -c "from src.calculator import PokerCalculator; print('Import OK')"
python -c "from src.environment import PokerEnvironment; e = PokerEnvironment(); print('Environment OK')"
```

## Architecture

```
Poker/
├── src/                    # Runtime code (for the calculator)
│   ├── calculator.py       # Main API - PokerCalculator class
│   ├── environment.py      # Game state management
│   ├── player.py           # Human player class
│   ├── evaluator.py        # Hand strength evaluation (bit manipulation)
│   ├── deck.py             # Card/deck utilities
│   └── ai/
│       ├── ai_player.py    # CFRAIPlayer, EquityAIPlayer
│       └── abstraction.py  # Card abstraction (equity, clustering)
│
├── training/               # Training code (separate from runtime)
│   ├── cfr.py              # CFR algorithm (History, InfoSet, CFR classes)
│   ├── preflop_trainer.py  # Preflop strategy training
│   └── postflop_trainer.py # Postflop strategy training
│
├── models/                 # Trained model files (.joblib)
└── examples/               # Example code (rps.py)
```

## Key Classes

### Runtime ([src/](src/))

- **PokerCalculator** ([calculator.py](src/calculator.py)) - Main API
  - `get_equity(hole_cards, community_cards)` - Hand winning probability
  - `get_ai_action(hole_cards, ...)` - AI action recommendation
  - `compare_hands(board, hand1, hand2)` - Compare two hands

- **PokerEnvironment** ([environment.py](src/environment.py)) - Game management
  - Game stages: 1=initial, 2=preflop, 3=flop, 4=turn, 5=river, 6=end
  - Player 0 = human, Player 1 = AI (after `add_ai_player()`)

- **CFRAIPlayer** ([ai/ai_player.py](src/ai/ai_player.py)) - Trained AI
  - Loads strategy from joblib files
  - Falls back to equity heuristics if no model

### Training ([training/](training/))

- **CFR** ([cfr.py](training/cfr.py)) - Core algorithm
  - `solve()` - Run CFR iterations
  - `export_infoSets(filename)` - Save trained strategy

- **PreflopHistory/PostflopHistory** - Game state for training
  - 169 preflop clusters (lossless)
  - 50/50/10 flop/turn/river clusters (equity-based)

## Card Format

2-character strings: `Rank + Suit`
- Ranks: `A`, `2`-`9`, `T`, `J`, `Q`, `K`
- Suits: `h`, `d`, `c`, `s`
- Examples: `"Ah"` (Ace of hearts), `"Td"` (Ten of diamonds)

## Action Format

- `f` = fold
- `k` = check
- `c` = call
- `bX` = bet/raise X amount (e.g., `b100`)
- Abstract: `bMIN` (~1/3 pot), `bMID` (pot), `bMAX` (all-in)

## Dependencies

- `phevaluator` - Fast hand evaluation for training
- `joblib` - Model serialization
- `numpy` - Numerical operations
- `scikit-learn` - KMeans clustering (training only)
