"""Draws the two charts from a finished run.

    python plots.py --dataset synthetic

Reads what run.py saved, so run that first.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="synthetic")
parser.add_argument("--out", default="results")
args = parser.parse_args()

stem = os.path.join(args.out, args.dataset)
with open(stem + "_results.json") as f:
    results = json.load(f)
history = np.load(stem + "_reward_history.npy")


def rolling(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(history, color="lightsteelblue", linewidth=0.6, label="episode")
smooth = rolling(history)
ax.plot(np.arange(len(smooth)) + len(history) - len(smooth), smooth,
        color="darkblue", linewidth=1.8, label="50 episode average")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="coin flip")
ax.set_xlabel("episode")
ax.set_ylabel("validation AUC of the chosen subset")
ax.set_title("What the agent found while learning")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

ax = axes[1]
rows = sorted(results["rows"], key=lambda r: r["test_auc"])
names = [r["method"] for r in rows]
scores = [r["test_auc"] for r in rows]
colours = ["indianred" if "rl agent" in n else "steelblue" for n in names]

bars = ax.barh(names, scores, color=colours)
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("test AUC")
ax.set_title("Agent against the usual feature selection methods")
ax.set_xlim(0.4, max(scores) + 0.08)
for bar, row in zip(bars, rows):
    label = "{:.3f}".format(row["test_auc"])
    if row["recovery"] is not None:
        label += "  ({:.0f}/{} planted)".format(
            row["recovery"]["found"], row["recovery"]["of"])
    ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8)
ax.grid(axis="x", alpha=0.3)

fig.suptitle("{} dataset, {} features, budget {}".format(
    args.dataset, results["shape"][1], results["budget"]))
fig.tight_layout()
fig.savefig(stem + "_charts.png", dpi=130)
print("Saved", stem + "_charts.png")
