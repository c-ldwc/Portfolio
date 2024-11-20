from scipy.stats import beta, binom
import numpy as np

from copy import deepcopy

from typing import Callable
from dataclasses import dataclass, field
from tqdm import tqdm
# from .reward import reward_generator
# from .stats import bound


@dataclass
class bandit:
    N: int = 1  # num units
    K: int = None  # num actions
    reward_fn: Callable = None  # accepts an action and returns a reward
    T: int = 100
    t: int = 0

    def __post_init__(self):
        self.running_means = [0] * self.K  # means for each action
        self.running_counts = [0] * self.K  # Counts for each action
        self.action = [0] * self.N  # most recent actions


@dataclass
class thompsonSampling(bandit):
    prior: list[int] = field(default_factory=[1, 1])  # shared prior for actions
    initial: bool = True  # Is this the first trial?

    def __post_init__(self):
        self.act_priors = [
            deepcopy(self.prior) for a in range(self.K)
        ]  # prior for each action
        super().__post_init__()

    def choose_action(self):
        sampled_rewards = [None] * self.K
        for unit in range(self.N):
            for action in range(self.K):
                prior = self.act_priors[action]
                sampled_rewards[action] = beta(prior[0], prior[1]).rvs(1)[0]
            self.action[unit] = np.argmax(sampled_rewards)
        self.t += 1

    def update(self):
        """
        Update bandit priors
        """
        for unit in range(self.N):
            this_prior = self.act_priors[self.action[unit]]
            this_reward = self.reward_fn(self.action[unit])
            self.act_priors[
                self.action[unit]
            ] = [  # conjugate prior update for beta prior with binomial data
                this_prior[0] + this_reward,
                this_prior[1] + 1 - this_reward,
            ]


@dataclass
class UCB(bandit):
    bound_fn: Callable = None

    def __post_init__(self):
        self.bounds = np.array(
            [deepcopy([None, None]) for i in range(self.K)]
        )  # bounds for each action
        super().__post_init__()

        for arm in range(self.K):
            # print(arm)
            rwd = self.reward_fn(arm)

            self.running_means[arm] += rwd
            self.running_counts[arm] += 1

            bound = self.bound_fn(
                self.running_means[arm], self.running_counts[arm], self.T
            )
            self.bounds[arm] = bound
            self.t += 1

    def choose_action(self):
        best_action = np.argmax(
            [b[1] for b in self.bounds]
        )  # np argmax will select earliest winner if multiple maxes
        self.action = best_action

    def update(self):
        self.reward = self.reward_fn(self.action)

        self.running_means[self.action] = (
            self.running_means[self.action] * self.running_counts[self.action]
            + self.reward
        ) / (self.running_counts[self.action] + 1)
        self.running_counts[self.action] += 1

        bound = self.bound_fn(
            self.running_means[self.action], self.running_counts[self.action], self.T
        )
        self.bounds[self.action] = bound
        self.t += 1
