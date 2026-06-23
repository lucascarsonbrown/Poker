# Models

This folder holds the trained strategy files produced by the CFR trainers.

## Files

| File                       | Trained by              | Used by                          |
|----------------------------|-------------------------|----------------------------------|
| `preflop_infoSets.joblib`  | `preflop_trainer.py`    | `PokerCalculator.get_ai_action()` |
| `postflop_infoSets.joblib` | `postflop_trainer.py`   | `PokerCalculator.get_ai_action()` |

## Format

Each file is a Python dictionary serialized with `joblib`. The keys are information set strings (cluster ID + betting history), and the values are dicts containing the trained strategy:

```python
{
    "14bMIN": {
        "strategy": {"f": 0.12, "c": 0.61, "bMAX": 0.27},
        "actions": ["f", "c", "bMAX"]
    },
    ...
}
```

The key `"14bMIN"` means: hand cluster 14 (AK offsuit), facing a min-bet. The strategy says fold 12% of the time, call 61%, re-raise all-in 27%.

## Training

If these files are missing or you want to retrain, see [training/README.md](../training/README.md).

`PokerCalculator` loads them lazily on first use and falls back to equity-based heuristics if they are absent.
