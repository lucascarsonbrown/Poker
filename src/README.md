# How the Poker AI Works

## The simple version

The AI plays poker against itself millions of times. Every time it makes a decision that turns out to be wrong — a fold that would have won, a call that would have lost — it remembers that regret. Over time it shifts its strategy away from choices it regrets and toward choices that worked. After enough iterations, the strategy stops changing. That stable point is a **Nash equilibrium**: a strategy that cannot be exploited by any opponent, no matter what they do.

At game time, the AI looks up the current situation in a table of strategies it built during training, then randomly picks an action according to the probabilities in that table — for example, "raise 60% of the time, call 40%."

---

## The game tree

Poker can be represented as a tree. Each node is a game state; each edge is an action (fold, call, raise). The tree branches at every decision point and terminates when the hand ends. A full poker game tree is astronomically large — billions of nodes even for a simplified two-player game — so the algorithm never builds it explicitly. Instead it traverses it recursively, visiting the same nodes across many iterations.

There are three types of nodes:

| Node type    | Meaning                                    |
|--------------|--------------------------------------------|
| **Chance**   | Nature acts — cards are dealt              |
| **Decision** | A player chooses an action                 |
| **Terminal** | The hand is over; payoffs are assigned     |

In the code, `History` represents a node. Subclasses implement `is_chance()`, `actions()`, `player()`, and `terminal_utility()` to define the specific game being solved.

---

## Information sets

In chess, both players see the whole board. In poker, each player has hidden hole cards. This means two different game states can look identical to the player who is acting — they see the same community cards, the same betting history, but not the opponent's hand.

An **information set** groups all states that are indistinguishable from the acting player's perspective. The key insight of CFR is that a rational player's strategy only needs to be defined over information sets, not individual states — because they cannot tell those states apart anyway.

In the code, `get_infoSet_key()` maps a full game history to the information that the acting player actually observes. Two histories that produce the same key belong to the same information set and share the same strategy.

Each `InfoSet` stores three things:

```python
self.regret             # how much the player regrets not taking each action
self.strategy           # current action probabilities (derived from regret)
self.cumulative_strategy  # running sum of past strategies (used at convergence)
```

---

## Regret matching

At each decision point, the current strategy is computed from accumulated regret:

```text
σ(a) = max(R(a), 0) / Σ max(R(a'), 0)
```

Where `R(a)` is the regret for action `a` and `σ(a)` is the probability of choosing it. Actions with positive regret get higher probability; actions with zero or negative regret are suppressed entirely. If all regrets are negative (every option feels bad), the strategy falls back to uniform — the algorithm keeps exploring.

In `InfoSet.get_strategy()`:

```python
positive_regret = {a: max(r, 0) for a, r in self.regret.items()}
regret_sum = sum(positive_regret.values())

if regret_sum > 0:
    self.strategy = {a: r / regret_sum for a, r in positive_regret.items()}
else:
    self.strategy = {a: 1.0 / n for a in self._actions}  # uniform fallback
```

---

## Counterfactual values and regret updates

The word *counterfactual* means "what would have happened." For each action `a` at a decision point, CFR computes `v(a)`: the expected payoff if the player had taken action `a` — regardless of what their current strategy says.

The **counterfactual regret** for action `a` is the difference between the value of taking `a` and the value of following the current strategy:

```text
R(a) += reach_probability_of_opponent × (v(a) − v(current_strategy))
```

The opponent's reach probability acts as a weight. It asks: "in situations where my opponent actually played to reach this state, how much am I leaving on the table by not always playing `a`?"

In `vanilla_cfr()`:

```python
# v = expected value of current strategy (weighted sum over all actions)
v = sum(strategy[a] * va[a] for a in actions)

# Update regret for the acting player
for a in actions:
    infoSet.regret[a] += opponent_reach * (va[a] - v)
    infoSet.cumulative_strategy[a] += player_reach * strategy[a]
```

Each iteration, both players alternately compute their counterfactual values through the full game tree, then update regrets. The `solve()` loop runs this for as many iterations as specified.

---

## Convergence to Nash equilibrium

The **current** strategy bounces around — it is updated every iteration based on fresh regret calculations. The current strategy is not what gets used at game time.

What converges is the **time-averaged strategy**: the average of every strategy the player has ever played across all iterations. This is stored in `cumulative_strategy` and retrieved via `get_average_strategy()`:

```python
return {a: s / strategy_sum for a, s in self.cumulative_strategy.items()}
```

It is a theorem (Zinkevich et al., 2007) that in two-player zero-sum games, if both players independently minimize their cumulative regret, the time-averaged strategy of each player converges to a Nash equilibrium. The **exploitability** of the strategy — how much a perfect opponent could gain above the game value — shrinks toward zero as iterations increase.

---

## Card abstraction

Even a simplified poker game has too many distinct hands to train a unique strategy for each one. Card abstraction reduces the number of information sets by grouping hands that are strategically similar.

### Preflop: lossless abstraction (169 clusters)

Before the flop, only two hole cards are known. In Texas Hold'em, the specific suits do not matter beyond whether they match — `AhKd` and `AsKc` are strategically identical (both are AK offsuit). This allows lossless compression into exactly 169 canonical hand categories:

| Range   | Category                     | Count |
|---------|------------------------------|-------|
| 1–13    | Pocket pairs (22 through AA) | 13    |
| 14–91   | Offsuit non-pairs            | 78    |
| 92–169  | Suited non-pairs             | 78    |

This is lossless because hands in the same cluster are truly equivalent — no information is thrown away.

### Postflop: equity-based clustering

Once community cards appear, suits start to matter (flushes, flush draws) and the cluster space explodes. Exact equity for every possible 5–7 card combination is intractable, so postflop hands are bucketed by their **win probability** (equity), estimated via Monte Carlo simulation.

`calculate_equity()` runs `n` random rollouts: for each rollout it deals a random opponent hand and completes the board, then checks whether the player's hand wins. The fraction of wins across all rollouts is the equity estimate.

That equity value is then mapped to a bucket:

```python
cluster = min(total_clusters - 1, int(equity * total_clusters))
```

So a hand with 70% equity against 50 clusters lands in cluster 35. Hands in the same cluster share the same trained strategy.

This is **lossy** — two hands with similar equity but different textures (a nut flush draw vs. top pair) may land in the same bucket despite playing differently. More clusters reduce this error at the cost of more training time and memory.

---

## Runtime lookup

At game time, `PokerCalculator.get_ai_action()` does the following:

1. Computes equity for the current hand via Monte Carlo simulation.
2. Maps the hand to a cluster ID (preflop: lossless 169-cluster hash; postflop: equity bucket).
3. Builds an information set key from the cluster ID and the betting history so far.
4. Looks up that key in the trained strategy table loaded from `models/`.
5. Samples an action from the probability distribution stored there.
6. If the key is not found (the training didn't cover that exact sequence), falls back to equity-based heuristics: compare hand equity against pot odds to decide fold/call/raise.

---

## Further reading (How I learned)

- Zinkevich et al., *Regret Minimization in Games with Incomplete Information* (2007) — proves convergence of CFR
- Bowling et al., *Heads-up Limit Hold'em Poker is Solved* (Science, 2015) — applies CFR variants to poker at scale
- Neller & Lanctot, *An Introduction to Counterfactual Regret Minimization* (2013) — accessible walkthrough with pseudocode
