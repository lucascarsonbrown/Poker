# Training

This folder contains the CFR algorithm and the scripts that use it to train poker strategies.

## Files

| File                  | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `cfr.py`              | Base `History`, `InfoSet`, and `CFR` classes             |
| `preflop_trainer.py`  | Trains strategy for preflop decisions                    |
| `postflop_trainer.py` | Trains strategy for flop, turn, and river decisions      |

---

## How training works

Training runs CFR iterations over a dataset of randomly dealt hands. Each iteration traverses the game tree for both players, updating regret values at every decision point. After all iterations, the time-averaged strategy at each information set is exported to a `.joblib` file.

The preflop and postflop trainers are separate because they use different card abstractions and action sets. See [src/README.md](../src/README.md) for a full explanation of the algorithm.

---

## Running the trainers

Run from the project root (not from inside `training/`):

### Preflop

```bash
python -m training.preflop_trainer [options]
```

| Flag            | Default                           | Description                          |
|-----------------|-----------------------------------|--------------------------------------|
| `-i, --iterations` | `50000`                        | CFR iterations per batch             |
| `-b, --batches`    | `1`                            | Number of training batches           |
| `-s, --samples`    | `50000`                        | Random hands dealt per batch         |
| `-o, --output`     | `models/preflop_infoSets.joblib` | Output path                        |

Recommended run (~5-10 min):

```bash
python -m training.preflop_trainer -i 100000 -b 5 -o models/preflop_infoSets.joblib
```

### Postflop

```bash
python -m training.postflop_trainer [options]
```

| Flag            | Default                            | Description                          |
|-----------------|------------------------------------|--------------------------------------|
| `-i, --iterations` | `50000`                         | CFR iterations per batch             |
| `-b, --batches`    | `1`                             | Number of training batches           |
| `-s, --samples`    | `10000`                         | Random hands dealt per batch         |
| `-o, --output`     | `models/postflop_infoSets.joblib` | Output path                        |

Recommended run (~30-60 min):

```bash
python -m training.postflop_trainer -i 50000 -b 3 -s 10000 -o models/postflop_infoSets.joblib
```

Postflop training is significantly slower than preflop because each hand requires Monte Carlo equity calculations to assign cluster IDs before training begins. Cluster computation is parallelized across all CPU cores automatically.

---

## Training in batches

Both trainers support batched training via `-b`. Each batch generates a fresh dataset and runs the full set of CFR iterations, accumulating regret into the same model. More batches expose the algorithm to more hand diversity and generally improve strategy quality. The model is saved after every batch, so you can stop early and keep whatever was learned.

---

## Extending the base classes

`cfr.py` provides abstract base classes (`History`, `InfoSet`, `CFR`) that can be subclassed for any two-player zero-sum game. `examples/rps.py` demonstrates this with Rock-Paper-Scissors as a minimal working example.

To implement a new game variant:

1. Subclass `History` and implement `is_terminal()`, `actions()`, `player()`, `sample_chance_outcome()`, `terminal_utility()`, `__add__()`, and `get_infoSet_key()`
2. Subclass `InfoSet` if you need custom behavior (usually not necessary)
3. Pass factory functions to `CFR(create_infoSet, create_history)`
4. Call `cfr.solve()` then `cfr.export_infoSets()`
