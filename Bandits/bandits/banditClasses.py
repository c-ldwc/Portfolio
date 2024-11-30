from scipy.stats import beta
import numpy as np

from copy import deepcopy

from typing import Callable
from dataclasses import dataclass, field


@dataclass
class bandit:
    N: int = 1  # num units
    K: int = None  # num actions
    reward_fn: Callable = None  # accepts an action and returns a reward
    T: int = 100
    t: int = 0

    def __post_init__(self):
        self.running_means = np.zeros(self.K)  # means for each action
        self.running_counts = np.zeros(self.K)  # Counts for each action
        self.action = np.ones(self.N, dtype=int)  # most recent actions
        self.rewards = np.zeros((self.K, 2), dtype=int)
        self.last_rewards = np.ones(self.N, dtype=int) * -99


@dataclass
class thompsonSampling(bandit):
    prior: list[int] = field(default=list)  # shared prior for actions
    initial: bool = True  # Is this the first trial?

    def __post_init__(self):
        self.act_priors = np.array(
            [deepcopy(self.prior) for a in range(self.K)]
        ).astype(int)  # prior for each action
        super().__post_init__()

    def choose_action(self):
        sampled_rewards = [None] * self.K
        for unit in range(self.N):
            for action in range(self.K):
                prior = self.act_priors[action]
                sampled_rewards[action] = beta(prior[0], prior[1]).rvs(1)[0]
            self.action[unit] = np.argmax(sampled_rewards)

    def observe_reward(self):
        assert np.all(
            self.rewards.shape == self.act_priors.shape
        ), f"prior dims({self.act_priors.shape}) and reward dims ({self.rewards.shape}) differ"
        for unit in range(self.N):
            this_reward = int(self.reward_fn(self.action[unit], self.t))
            self.rewards[self.action[unit], 1 - this_reward] += 1
            self.last_rewards[unit] = this_reward

    def update(self):
        """
        Update bandit priors,
        This is the ordinary binomial likelihood beta prior on p update
        """
        Ns = self.rewards.sum(axis=1)
        # print(f"Ns {Ns}")
        # print(f't = {self.t}, act = {self.action}, last rwd = {self.last_rewards}')
        # print(self.act_priors)

        # print(self.rewards)
        self.act_priors[:, 0] += self.rewards[:, 0]
        self.act_priors[:, 1] += Ns - self.rewards[:, 0]
        # reset rewards until next update
        # print(self.act_priors)
        self.rewards = np.zeros((self.K, 2), dtype=int)
        self.t += 1


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
            rwd = self.reward_fn(arm, self.t)

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
        self.reward = self.reward_fn(self.action, self.t)

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


if __name__ == "__main__":
    from numpy import ndarray
    from scipy.stats import binom

    @dataclass
    class binom_reward_generator:
        """
        Beta distributed reward generator
        mus: list of means, len(mus) is the number of arms
        a: alpha params for reward distributions. same length as number of arms
        b parameters are calculated such that we get means equal to mu
        """

        mus: ndarray[float] = None
        N: int = 1

        def reward(self, act, t):
            """
            Get a reward for action `act` with mean self.mus[act]
            """
            if self.N == 1:
                return binom(p=self.mus[act], n=1).rvs(self.N)[0]
            else:
                return binom(p=self.mus[act], n=1).rvs(self.N)

    rg = binom_reward_generator(mus=[0.2, 0.5])
    b = thompsonSampling(N=1, K=2, reward_fn=rg.reward, T=10, prior=[1, 1])

    while b.t < b.T:
        b.choose_action()
        b.observe_reward()
        b.update()
