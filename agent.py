"""Policy gradient agent (REINFORCE with a learned baseline).

The policy reads the current selection mask and produces a score for every
feature it could add next. The critic guesses how much AUC is still left to gain
from the current state, and the gap between what actually happened and that
guess is what the policy learns from.

The critic earns its place here. Rewards are differences in AUC, so most steps
pay out something near zero and a few pay out a lot. Without a baseline the
agent spends its updates chasing that noise.
"""

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, n_features, n_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        shared = self.body(state)
        return self.actor(shared), self.critic(shared).squeeze(-1)


class Agent:
    def __init__(self, n_features, n_actions, lr=3e-4, entropy_weight=0.01, seed=0):
        torch.manual_seed(seed)
        self.net = PolicyNet(n_features, n_actions)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.entropy_weight = entropy_weight

    def act(self, state, legal, greedy=False):
        """Samples one action, ignoring anything the environment marked illegal."""
        state_t = torch.as_tensor(state).unsqueeze(0)
        logits, value = self.net(state_t)

        blocked = torch.as_tensor(~legal).unsqueeze(0)
        logits = logits.masked_fill(blocked, -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=1) if greedy else dist.sample()

        return (int(action.item()), dist.log_prob(action).squeeze(0),
                value.squeeze(0), dist.entropy().squeeze(0))

    def learn(self, log_probs, values, entropies, rewards, gamma=1.0):
        """One update per episode.

        The return for a step is every reward that came after it. Later picks
        only get credit for what they added, so the agent can tell a pick that
        completed a useful pair from one that happened to follow it.
        """
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        advantage = returns - values

        policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy_bonus = torch.stack(entropies).mean()

        loss = policy_loss + 0.5 * value_loss - self.entropy_weight * entropy_bonus

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimiser.step()

        return float(policy_loss.item()), float(value_loss.item())


def run_episode(env, agent, greedy=False):
    """Plays one episode and returns what the update needs."""
    state = env.reset()
    log_probs, values, entropies, rewards = [], [], [], []
    done = False

    while not done:
        action, log_prob, value, entropy = agent.act(
            state, env.legal_actions(), greedy)
        state, reward, done = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)
        rewards.append(reward)

    return log_probs, values, entropies, rewards, env.last_score, env.selected()
