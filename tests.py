"""Checks the parts that would fail quietly.

Run with: python tests.py

Most of these guard against bugs that still let the code run and print a
number, which are the ones that waste days. The leakage check and the
telescoping check are the important ones.
"""

import numpy as np
import torch

import baselines
import datasets
from agent import Agent, run_episode
from env import CHANCE, FeatureEnv


def small_data(seed=0):
    X, y, truth = datasets.make_synthetic(n_features=12, n_samples=800, seed=seed)
    return datasets.split(X, y, seed=seed, search_rows=400, val_rows=200), truth


def test_synthetic_truth_is_sane():
    X, y, truth = datasets.make_synthetic(n_features=40, n_samples=1000)
    assert X.shape == (1000, 40)
    assert len(truth) == 4
    assert len(set(truth)) == 4, "planted indices must be distinct"
    assert truth.min() >= 0 and truth.max() < 40
    assert set(np.unique(y)) == {0, 1}


def test_single_features_carry_no_signal():
    """The whole benchmark rests on this.

    If a planted column correlated with the label on its own then ranking
    features one at a time would solve the task and there would be nothing for
    a search method to do. The label is a product of pairs, so each column
    stays symmetric around zero whatever y is.
    """
    X, y, truth = datasets.make_synthetic(n_samples=6000, noise=0.0)
    for col in truth:
        corr = abs(np.corrcoef(X[:, col], y)[0, 1])
        assert corr < 0.05, "planted feature {} leaks on its own: {:.3f}".format(col, corr)


def test_pairs_together_do_carry_signal():
    """And the flip side: the pairs have to be findable at all."""
    X, y, truth = datasets.make_synthetic(n_samples=6000, noise=0.0)
    products = [abs(np.corrcoef(X[:, a] * X[:, b], y)[0, 1])
                for a in truth for b in truth if a < b]
    assert max(products) > 0.3, "no pair product correlates with the label"


def test_split_does_not_leak_rows():
    X = np.random.default_rng(0).normal(size=(600, 5))
    y = (X[:, 0] > 0).astype(int)
    data = datasets.split(X, y, seed=0, search_rows=1000, val_rows=1000)

    rows = {name: {tuple(np.round(r, 6)) for r in data[name][0]}
            for name in ["train", "val", "test"]}
    assert not rows["train"] & rows["test"]
    assert not rows["val"] & rows["test"]
    assert not rows["train"] & rows["val"]


def test_env_respects_budget():
    data, _ = small_data()
    env = FeatureEnv(data, budget=4)
    env.reset()

    steps = 0
    done = False
    while not done:
        legal = env.legal_actions()
        legal[env.stop_action] = False  # never stop early, test the hard cap
        action = np.where(legal)[0][0]
        _, _, done = env.step(action)
        steps += 1
        assert steps <= 4, "episode ran past the budget"

    assert len(env.selected()) == 4


def test_env_blocks_repeat_picks():
    data, _ = small_data()
    env = FeatureEnv(data, budget=5)
    env.reset()
    env.step(3)
    assert not env.legal_actions()[3], "already picked feature is still legal"


def test_env_blocks_stopping_too_early():
    data, _ = small_data()
    env = FeatureEnv(data, budget=5)
    env.reset()
    assert not env.legal_actions()[env.stop_action]
    env.step(0)
    assert not env.legal_actions()[env.stop_action]
    env.step(1)
    assert env.legal_actions()[env.stop_action]


def test_score_cache_is_consistent():
    data, _ = small_data()
    env = FeatureEnv(data, budget=5)

    first = env.score(np.array([0, 1, 2]))
    fits_after_first = env.fits
    second = env.score(np.array([0, 1, 2]))

    assert first == second
    assert env.fits == fits_after_first, "cache hit still refit the model"
    assert env.lookups == 1


def test_rewards_add_up_to_the_final_score():
    """Shaped rewards must telescope back to the plain end of episode score.

    Each step pays score(new) - score(old), so the sum has to collapse to
    final - 0.5. If it does not, the agent is being paid for something other
    than the thing being measured.
    """
    data, _ = small_data()
    env = FeatureEnv(data, budget=5)
    env.reset()

    rng = np.random.default_rng(7)
    total = 0.0
    done = False
    while not done:
        legal = env.legal_actions()
        legal[env.stop_action] = False
        action = rng.choice(np.where(legal)[0])
        _, reward, done = env.step(action)
        total += reward

    assert abs(total - (env.last_score - CHANCE)) < 1e-9


def test_agent_only_picks_legal_actions():
    data, _ = small_data()
    env = FeatureEnv(data, budget=4)
    agent = Agent(env.n_features, env.n_actions, seed=0)

    state = env.reset()
    for _ in range(3):
        legal = env.legal_actions()
        action, _, _, _ = agent.act(state, legal)
        assert legal[action], "agent picked a blocked action"
        state, _, done = env.step(action)
        if done:
            break


def test_learning_changes_the_weights():
    data, _ = small_data()
    env = FeatureEnv(data, budget=4)
    agent = Agent(env.n_features, env.n_actions, seed=0)

    before = [p.clone() for p in agent.net.parameters()]
    log_probs, values, entropies, rewards, _, _ = run_episode(env, agent)
    agent.learn(log_probs, values, entropies, rewards)

    moved = any(not torch.equal(b, a)
                for b, a in zip(before, agent.net.parameters()))
    assert moved, "an update left every weight untouched"


def test_greedy_rollout_is_repeatable():
    data, _ = small_data()
    env = FeatureEnv(data, budget=4)
    agent = Agent(env.n_features, env.n_actions, seed=0)

    first = run_episode(env, agent, greedy=True)[5]
    second = run_episode(env, agent, greedy=True)[5]
    assert list(first) == list(second), "greedy policy is not deterministic"


def test_baselines_return_the_asked_for_count():
    X, y, _ = datasets.make_synthetic(n_features=20, n_samples=600)
    for name, method in baselines.METHODS.items():
        if name == "all features":
            continue
        picked = method(X, y, 6, 0)
        assert len(picked) == 6, "{} returned {} features".format(name, len(picked))
        assert len(set(picked)) == 6, "{} returned duplicates".format(name)
        assert max(picked) < 20


passed = 0
failed = 0
for name, item in sorted(globals().items()):
    if not name.startswith("test_"):
        continue
    try:
        item()
        print("pass  " + name)
        passed += 1
    except AssertionError as err:
        print("FAIL  {}: {}".format(name, err))
        failed += 1

print("\n{} passed, {} failed".format(passed, failed))
