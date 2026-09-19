"""The environment the agent acts in.

An episode starts with nothing selected and adds one feature per step until the
budget runs out or the agent stops.

The reward for a step is how much the validation AUC moved because of that pick.
The first version of this only paid out at the end of the episode and the agent
learnt nothing in 1500 tries. On the synthetic task fewer than one percent of
random subsets contain a complete pair, so nearly every episode returned the
same 0.5 and there was no gradient to follow. Paying per step tells the agent
which pick helped, and the rewards still add up to the same final score.

Scoring a subset means fitting a model, so every score is cached. Episodes share
their opening picks constantly and the cache turns most of those into lookups.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

CHANCE = 0.5  # AUC you get from an empty feature set


class FeatureEnv:
    def __init__(self, data, budget, penalty=0.0, trees=12, seed=0):
        self.X_train, self.y_train = data["train"]
        self.X_val, self.y_val = data["val"]
        self.n_features = self.X_train.shape[1]
        self.budget = budget
        self.penalty = penalty
        self.trees = trees
        self.seed = seed

        self.stop_action = self.n_features
        self.n_actions = self.n_features + 1

        # Fitting the same subset twice is wasted work and the agent revisits
        # subsets constantly, so every score is kept.
        self.cache = {}
        self.fits = 0
        self.lookups = 0

        self.mask = np.zeros(self.n_features, dtype=np.float32)
        self.last_score = CHANCE

    def reset(self):
        self.mask = np.zeros(self.n_features, dtype=np.float32)
        self.last_score = CHANCE
        return self.state()

    def state(self):
        """Mask plus how much of the budget is gone.

        Without the progress number the agent cannot tell an early step from a
        late one, which makes the stop action impossible to learn.
        """
        used = self.mask.sum() / self.budget
        return np.concatenate([self.mask, [used]]).astype(np.float32)

    def legal_actions(self):
        """Features already taken are blocked, and stopping needs two picks."""
        legal = self.mask == 0
        return np.concatenate([legal, [self.mask.sum() >= 2]])

    def score(self, indices):
        """Validation AUC of a small model trained on these columns only."""
        key = tuple(indices)
        if key in self.cache:
            self.lookups += 1
            return self.cache[key]

        model = RandomForestClassifier(
            n_estimators=self.trees,
            min_samples_leaf=10,
            max_depth=8,
            n_jobs=1,
            random_state=self.seed,
        )
        model.fit(self.X_train[:, indices], self.y_train)
        probs = model.predict_proba(self.X_val[:, indices])[:, 1]

        auc = roc_auc_score(self.y_val, probs)
        self.cache[key] = auc
        self.fits += 1
        return auc

    def step(self, action):
        if action == self.stop_action:
            return self.state(), 0.0, True

        self.mask[action] = 1
        new_score = self.score(self.selected())

        reward = new_score - self.last_score - self.penalty
        self.last_score = new_score

        done = self.mask.sum() >= self.budget
        return self.state(), reward, done

    def selected(self):
        return np.where(self.mask > 0)[0]
