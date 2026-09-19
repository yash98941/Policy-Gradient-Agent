# Feature selection with a policy gradient agent

Picking a small subset of features is a search problem, and almost every
practical method dodges the search. Ranking methods score each feature on its
own and take the top k, which is fast and cannot see a feature that is only
useful next to another one. This project trains an agent to build the subset
one feature at a time, so the choice of the next feature depends on what it has
already picked.

The agent is REINFORCE with a learned baseline, written in PyTorch. The
environment is the feature selection problem itself: the state is which
features are selected so far, an action adds one, and the reward is how much
the validation AUC moved because of that pick.

It works on one of the two datasets here. On a task built around feature
interactions it recovers every planted feature and beats all six classical
baselines. On the credit default data it loses to simply keeping all the
columns. Both results are below, because which one you get is the useful part.

## Results

### Planted pairs, 40 features, pick 8

Test AUC is measured on a split that took no part in the search. "Planted
found" is how many of the 4 hidden columns the method recovered.

| method | test AUC | planted found |
| --- | --- | --- |
| RL agent, greedy policy | 0.909 | 4 / 4 |
| RL agent, best subset seen | 0.906 | 4 / 4 |
| tree importance | 0.895 | 4 / 4 |
| RFE | 0.729 | 2 / 4 |
| L1 logistic | 0.729 | 2 / 4 |
| ANOVA F-test | 0.729 | 2 / 4 |
| all 40 features | 0.700 | 4 / 4 |
| mutual information | 0.536 | 0 / 4 |
| random, 5 draws | 0.503 | 1.2 / 4 |

The agent recovered all four planted columns. 1500 episodes took 16 minutes and
8,729 model fits, with 26 percent of scores served from the cache.

Some things in that table matter more than the ranking.

Handing the model everything is worse than choosing. All 40 features scores
0.700, below a good subset of 8. The four useful columns are in there, and the
36 noise columns bury them.

RFE, L1 and ANOVA returned the exact same 8 features. Not similar subsets,
identical ones. All three rank by linear association with the label, and on a
task with no linear association they agree on the same wrong answer. Running
all three and seeing them agree would look like confirmation, and it is not.

Tree importance is a genuinely strong baseline. It also found all four, because
a deep forest splits on one feature and then on the other and picks up the
interaction indirectly. The agent beats it by 0.014, which is real but modest.
The honest summary is that the agent matches the best classical method here
rather than leaving it behind.

### Credit default, 23 features, pick 10

The same agent on the UCI credit card default data, where it loses.

| method | test AUC |
| --- | --- |
| all 23 features | 0.762 |
| tree importance | 0.754 |
| RL agent, best subset seen | 0.748 |
| mutual information | 0.748 |
| ANOVA F-test | 0.746 |
| RL agent, greedy policy | 0.745 |
| RFE | 0.742 |
| L1 logistic | 0.736 |
| random, 5 draws | 0.717 |

Keeping every feature wins. No subset of 10 beats the full 23, and the agent
lands mid table, behind tree importance and level with mutual information.

This is the result I expected to be able to hide and could not, so it is worth
being clear about. The method needs two conditions to pay off: features that
only work in combination, and enough noise features that including everything
hurts. The credit data has neither. Twenty three columns is a small space,
most of them carry signal on their own, and a forest handles the leftovers
without help. Spending eight minutes of search to end up 0.014 below the
do-nothing option is a bad trade, and the correct call on this dataset is not
to select features at all.

The agent did still learn. Its average episode score rose from 0.703 to 0.725
and it found a subset scoring 0.778 on validation. It optimised the thing it
was asked to optimise. The reward just was not worth chasing.

## The benchmark has to be honest

The synthetic task is the interesting part, because it is easy to build a
feature selection benchmark that proves nothing.

My first attempt used `make_classification`, which plants features that are
individually predictive. Tree importance recovered 15 out of 15 immediately.
That is a fair result and a useless benchmark: if ranking features one at a
time solves the task, there is nothing for a search method to do, and any win
the agent posts is a win over a strawman.

So the labels here come from products of pairs:

```
y = sign(x_a * x_b  +  x_c * x_d)
```

with the four columns hidden among 36 pure noise columns. Each of those four
columns is symmetric around zero whatever the label is, so its correlation with
y is zero by construction. `tests.py` asserts this rather than assuming it:
`test_single_features_carry_no_signal` fails the build if any planted column
leaks on its own, and `test_pairs_together_do_carry_signal` fails it if the
pairs are not findable at all.

Measured with a model on the search split:

| features given to the model | validation AUC |
| --- | --- |
| 8 random noise columns | 0.498 |
| one complete pair plus 6 noise | 0.730 |
| both pairs plus 4 noise | 0.903 |

Half a pair is worth nothing. That is what makes the task need a search.

## The reward was the hard part

The first version paid out once, at the end of the episode. The agent learnt
nothing in 1500 episodes and sat flat at 0.50.

The reason was not the algorithm. Under 1 percent of random 8 feature subsets
contain a complete pair, so nearly every episode returned the same number, and
a policy gradient with no variation in the return has no gradient to follow.

Paying per step fixed it. The reward for adding a feature is
`AUC(subset after) - AUC(subset before)`, so the pick that completes a pair is
the one that gets paid. The rewards still sum to the same final score, which
`test_rewards_add_up_to_the_final_score` checks by telescoping them.

There was a second version of the same mistake. Before settling on a sum of
products, the label was a majority vote over the signs of three pairs. That
looks similar and behaves completely differently:

| pairs found | AUC, majority of 3 signs | AUC, sum of 2 products |
| --- | --- | --- |
| 0 | 0.494 | 0.498 |
| 1 | 0.496 | 0.730 |
| 2 | 0.571 | 0.903 |

Under the majority label, finding one pair moves the score by 0.002, which is
inside the noise. The agent could only be paid for finding two pairs at once,
which random exploration essentially never does. I spent a while tuning
learning rates and entropy before measuring this, which was the wrong order.
Check that the reward is detectable before blaming the optimiser.

## How it works

The state is the selection mask plus one number for how much of the budget is
used. Without that number the agent cannot tell an early step from a late one,
which makes the stop action impossible to learn.

An action adds any feature not already taken, or stops. The environment masks
illegal actions to `-1e9` before the softmax, so no probability mass is wasted
on repeats.

The reward is the change in validation AUC from that pick, optionally minus a
fixed penalty per feature if you want shorter subsets.

Learning uses the return to go for each step, a critic head predicting the
value of the state, and the difference between them as the advantage. The
critic matters here because most steps pay out near zero and a few pay out a
lot. An entropy bonus keeps the policy exploring instead of locking onto the
first pair it finds.

Every reward means fitting a model, so every subset score is cached. Episodes
share their opening picks constantly, and on the synthetic run a quarter of all
scores came back as lookups.

## Keeping the comparison fair

Three splits. The agent only ever sees search train and search val. The test
split is untouched until the end, so the reported number is not the same
quantity the agent was optimising. Without this the agent wins by overfitting
the reward signal and the result means nothing.

Every baseline picks exactly the same number of features. On the synthetic data
the table also reports how many of the four planted columns each method found,
which is a cleaner comparison than AUC because it does not depend on the
downstream model.

## Running it

```
pip install -r requirements.txt

python tests.py                                    # 13 checks, about a minute
python run.py --dataset synthetic                  # the planted pairs task
python run.py --dataset madelon --budget 20        # OpenML 1485, 500 features
python run.py --dataset credit --budget 10         # UCI credit default
python plots.py --dataset synthetic                # learning curve and comparison
```

Datasets download themselves on first use and cache under `data/`.

Options:

```
--episodes 3000      longer search
--penalty 0.002      charge per feature to push for smaller subsets
--entropy 0.05       explore more
```

## Files

| file | what it does |
| --- | --- |
| `datasets.py` | the planted pairs task, Madelon, credit, and the three way split |
| `env.py` | the selection environment, shaped reward, score cache |
| `agent.py` | policy and critic network, REINFORCE update, episode loop |
| `baselines.py` | random, mutual info, ANOVA, L1, tree importance, RFE |
| `run.py` | trains, evaluates on test, prints and saves the comparison |
| `plots.py` | learning curve and the method comparison chart |
| `tests.py` | environment rules, reward telescoping, split leakage, benchmark sanity |

## Datasets

Madelon is from the NIPS 2003 feature selection challenge: 500 features of
which 20 matter, built around a five dimensional XOR, which is the real version
of what the synthetic task imitates.
