"""Trains the agent on one dataset and compares it against the baselines.

    python run.py --dataset synthetic
    python run.py --dataset madelon --budget 20 --episodes 2500
    python run.py --dataset credit --budget 10
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import baselines
import datasets
from agent import Agent, run_episode
from env import FeatureEnv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "madelon", "credit"])
parser.add_argument("--budget", type=int, default=8)
parser.add_argument("--episodes", type=int, default=1500)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--entropy", type=float, default=0.02)
parser.add_argument("--penalty", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="results")
args = parser.parse_args()


def test_auc(indices, data, seed):
    """Fits on the search train split and scores on the untouched test split."""
    X_tr, y_tr = data["train"]
    X_te, y_te = data["test"]

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed)
    model.fit(X_tr[:, indices], y_tr)
    return roc_auc_score(y_te, model.predict_proba(X_te[:, indices])[:, 1])


def recovery(indices, truth):
    """How many of the planted features the method actually found."""
    if truth is None:
        return None
    return {"found": len(set(indices) & set(truth)), "of": len(truth)}


X, y, truth = datasets.get(args.dataset, args.seed)
data = datasets.split(X, y, args.seed)

print("dataset:", args.dataset)
print("shape:", X.shape, " budget:", args.budget)
print("rows used for the search:", len(data["train"][1]),
      " val:", len(data["val"][1]), " test:", len(data["test"][1]))
if truth is not None:
    print("planted informative features:", len(truth))

env = FeatureEnv(data, args.budget, penalty=args.penalty, seed=args.seed)
agent = Agent(env.n_features, env.n_actions, lr=args.lr,
              entropy_weight=args.entropy, seed=args.seed)

print("\nTraining for", args.episodes, "episodes...")
history = []
best_reward = -np.inf
best_subset = None
start = time.time()

for episode in range(args.episodes):
    log_probs, values, entropies, rewards, val_auc, chosen = run_episode(env, agent)
    agent.learn(log_probs, values, entropies, rewards)
    history.append(val_auc)

    if val_auc > best_reward:
        best_reward = val_auc
        best_subset = chosen

    if (episode + 1) % 100 == 0:
        recent = np.mean(history[-100:])
        print("episode {:5d}  last 100 avg val auc {:.4f}  best {:.4f}  "
              "model fits {:6d}  {:.0f}s".format(
                  episode + 1, recent, best_reward, env.fits, time.time() - start))

elapsed = time.time() - start
print("\nsearch took {:.0f}s, {} model fits and {} cache hits ({:.0%} reused)".format(
    elapsed, env.fits, env.lookups,
    env.lookups / max(env.fits + env.lookups, 1)))

_, _, _, _, greedy_val, greedy_subset = run_episode(env, agent, greedy=True)
print("greedy rollout val auc: {:.4f} with {} features".format(
    greedy_val, len(greedy_subset)))

if greedy_val > best_reward:
    best_reward, best_subset = greedy_val, greedy_subset

rows = []
X_tr, y_tr = data["train"]

for name, method in baselines.METHODS.items():
    if name == "random":
        # One draw can get lucky, so the bar is the average of several.
        picks = [method(X_tr, y_tr, args.budget, args.seed + s) for s in range(5)]
        found = None
        if truth is not None:
            found = {
                "found": float(np.mean([len(set(p) & set(truth)) for p in picks])),
                "of": len(truth),
            }
        rows.append({
            "method": "random (5 draws)",
            "features": args.budget,
            "test_auc": float(np.mean([test_auc(p, data, args.seed) for p in picks])),
            "recovery": found,
        })
        continue

    picked = method(X_tr, y_tr, args.budget, args.seed)
    rows.append({
        "method": name,
        "features": len(picked),
        "test_auc": test_auc(picked, data, args.seed),
        "recovery": recovery(picked, truth),
    })

for label, subset in [("rl agent (best found)", best_subset),
                      ("rl agent (greedy policy)", greedy_subset)]:
    rows.append({
        "method": label,
        "features": len(subset),
        "test_auc": test_auc(subset, data, args.seed),
        "recovery": recovery(subset, truth),
    })

print("\nTest set results, {} features unless stated".format(args.budget))
print("-" * 62)
print("{:26s} {:>6s} {:>10s} {:>16s}".format(
    "method", "feats", "test auc", "planted found"))
for row in sorted(rows, key=lambda r: r["test_auc"], reverse=True):
    found = "-"
    if row["recovery"] is not None:
        found = "{:.1f} / {}".format(
            row["recovery"]["found"], row["recovery"]["of"])
    print("{:26s} {:6d} {:10.4f} {:>16s}".format(
        row["method"], row["features"], row["test_auc"], found))

os.makedirs(args.out, exist_ok=True)
stem = os.path.join(args.out, args.dataset)
with open(stem + "_results.json", "w") as f:
    json.dump({
        "dataset": args.dataset,
        "shape": list(X.shape),
        "budget": args.budget,
        "episodes": args.episodes,
        "seed": args.seed,
        "search_seconds": round(elapsed, 1),
        "model_fits": env.fits,
        "best_val_auc": best_reward,
        "best_subset": [int(i) for i in best_subset],
        "greedy_subset": [int(i) for i in greedy_subset],
        "rows": rows,
    }, f, indent=2)
np.save(stem + "_reward_history.npy", np.array(history))
print("\nSaved", stem + "_results.json")
