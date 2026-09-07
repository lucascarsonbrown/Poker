"""Renders docs/figure.png for the README.

Documentation tooling. Both panels come from this repo's own code:

  * the 13x13 charts read models/preflop_infoSets.joblib — the actual trained
    CFR strategy — and map hands to information sets with the project's own
    src.ai.abstraction.get_preflop_cluster_id
  * the convergence panel runs training.cfr on examples/rps.py live

    python3 docs/render_figure.py
"""
import os, sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.abstraction import get_preflop_cluster_id
import training.cfr as base
import examples.rps as rps

BG, PANEL, FG, MUTED = "#14171c", "#1b1f26", "#e6e4e0", "#8b8f98"
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
BETS = ("bMIN", "bMID", "bMAX")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": MUTED, "axes.edgecolor": "#2b323d",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titlecolor": FG,
})

strat = joblib.load("models/preflop_infoSets.joblib")


def hand_strategy(i, j):
    """Strategy for the hand at row i, col j of the standard 13x13 grid."""
    r1, r2 = RANKS[i], RANKS[j]
    if i == j:
        cards, label = r1 + "h" + r2 + "d", r1 + r2
    elif i < j:                                   # upper triangle: suited
        cards, label = r1 + "h" + r2 + "h", r1 + r2 + "s"
    else:                                         # lower triangle: offsuit
        cards, label = r2 + "h" + r1 + "d", r2 + r1 + "o"
    node = strat[str(get_preflop_cluster_id(cards))]["strategy"]
    return node, label


aggression = np.zeros((13, 13))
folding = np.zeros((13, 13))
labels = [[""] * 13 for _ in range(13)]
for i in range(13):
    for j in range(13):
        node, label = hand_strategy(i, j)
        labels[i][j] = label
        aggression[i, j] = sum(node.get(b, 0.0) for b in BETS)
        folding[i, j] = node.get("f", 0.0)

# a crude hand-strength proxy, used only to report how well the learned
# strategy tracks it
_v = {r: 14 - i for i, r in enumerate(RANKS)}
strength = np.zeros((13, 13))
for i in range(13):
    for j in range(13):
        strength[i, j] = (_v[RANKS[i]] + _v[RANKS[j]]
                          + (12 if i == j else 2 if i < j else 0))

fig = plt.figure(figsize=(13, 7.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1],
                      hspace=0.34, wspace=0.16,
                      left=0.045, right=0.975, top=0.845, bottom=0.125)

# (a) the money shot: raise frequency over all 169 starting hands ------------
ax = fig.add_subplot(gs[:, 0])
cmap = LinearSegmentedColormap.from_list(
    "agg", ["#1b1f26", "#2f3a4a", "#3f6d7a", "#5fb3b3", "#e0a458", "#f2d492"])
ax.imshow(aggression, cmap=cmap, vmin=0, vmax=1)
for i in range(13):
    for j in range(13):
        ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6.6,
                color="#0d1014" if aggression[i, j] > 0.62 else "#c8ccd4")
ax.set_xticks(range(13), RANKS, fontsize=8)
ax.set_yticks(range(13), RANKS, fontsize=8)
ax.set_title("Learned raise frequency over all 169 starting hands\n"
             "upper right suited · diagonal pairs · lower left offsuit", pad=10)
ax.tick_params(length=0)
cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.043, pad=0.02)
cb.set_label("P(bet or raise)", color=MUTED, fontsize=8.5)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.outline.set_edgecolor("#2b323d")

# (b) fold frequency ---------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
fcmap = LinearSegmentedColormap.from_list(
    "fold", ["#1b1f26", "#3a2b33", "#7a3f4a", "#e06c75", "#f2a6ac"])
im = ax.imshow(folding, cmap=fcmap, vmin=0, vmax=max(0.35, folding.max()))
ax.set_xticks(range(13), RANKS, fontsize=6.5)
ax.set_yticks(range(13), RANKS, fontsize=6.5)
ax.tick_params(length=0)
ax.set_title("Fold frequency — same grid", pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.02)
cb.ax.tick_params(colors=MUTED, labelsize=7)
cb.outline.set_edgecolor("#2b323d")

# (c) does the solver actually converge? -------------------------------------
ax = fig.add_subplot(gs[1, 1])
iters = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
curves = {a: [] for a in rps.ACTIONS}
for n in iters:
    cfr = base.CFR(rps.create_infoSet, rps.create_history, iterations=n)
    cfr.solve()
    root = cfr.infoSets[""].get_average_strategy()
    for a in rps.ACTIONS:
        curves[a].append(root[a])
for a, c in zip(rps.ACTIONS, ["#5fb3b3", "#e0a458", "#c678dd"]):
    ax.plot(iters, curves[a], "o-", color=c, ms=4, lw=1.3, label=a)
ax.axhline(1 / 3, color=FG, ls="--", lw=1, alpha=0.6, label="1/3 (equilibrium)")
ax.set_xscale("log")
ax.set_ylim(0, 0.72)
ax.grid(color="#232935", lw=0.6)
ax.set_xlabel("CFR iterations")
ax.set_ylabel("action probability")
ax.set_title("Sanity check: the same solver on rock-paper-scissors", pad=8)
ax.legend(facecolor=PANEL, edgecolor="#2b323d", labelcolor=FG, fontsize=7.5,
          ncol=2, framealpha=0.9)

fig.suptitle("What counterfactual regret minimization actually learned",
             fontsize=13, y=0.968)
corr = np.corrcoef(aggression.ravel(), strength.ravel())[0, 1]
fig.text(0.5, 0.056,
         "Premium hands (AA, KK, QQ, AKs) raise %.0f%% of the time, the worst "
         "offsuit hands %.0f%%; correlation with hand strength is %.2f. The "
         "broad shape is right and the detail is still noisy — 100k iterations "
         "is not convergence."
         % (100 * np.mean([aggression[0, 0], aggression[1, 1], aggression[2, 2],
                           aggression[0, 1]]),
            100 * np.mean([aggression[12, 5], aggression[12, 6],
                           aggression[12, 10], aggression[12, 11]]), corr),
         color=MUTED, fontsize=8.3, ha="center")
fig.text(0.5, 0.024,
         "Left and top-right read models/preflop_infoSets.joblib directly. "
         "Bottom-right runs the solver live on a game whose equilibrium is known "
         "to be uniform — if it misses there, nothing else can be trusted.",
         color="#6b7078", fontsize=8, ha="center")

fig.savefig("docs/figure.png", dpi=140)
print("saved docs/figure.png")
print("raise freq  AA=%.2f  72o=%.2f" % (aggression[0, 0], aggression[12, 5]))
