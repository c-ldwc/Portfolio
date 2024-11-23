from scipy.stats import beta, binom
from numpy import sin, pi, ndarray
from dataclasses import dataclass


@dataclass
class beta_reward_generator:
    """
    Beta distributed reward generator
    mus: list of means, len(mus) is the number of arms
    a: alpha params for reward distributions. same length as number of arms
    b parameters are calculated such that we get means equal to mu
    """

    mus: ndarray[float] = None
    a: ndarray[float] = None
    N: int = 1

    def __post_init__(self):
        self.b = self.a / self.mus - self.a

    def reward(self, act):
        """
        Get a reward for action `act` with mean self.mus[act]
        """
        if self.N == 1:
            return beta(self.a[act], self.b[act]).rvs(self.N)[0]
        else:
            return beta(self.a[act], self.b[act]).rvs(self.N)


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

    def reward(self, act):
        """
        Get a reward for action `act` with mean self.mus[act]
        """
        if self.N == 1:
            return binom(p=self.mus[act], n=1).rvs(self.N)[0]
        else:
            return binom(p=self.mus[act], n=1).rvs(self.N)


# def non_stationary_reward(mu, t, N=1, amp=0.01, delay=0):
#     if mu + amp >= 1:
#         amp = (1 - mu) / 2
#     elif mu - amp <= 0:
#         amp = mu / 2
#     mu = mu + 0.01 * sin(2 * pi / 7.0 * t)
#     a = 2
#     b = a / mu - a  # To get E[X] = mu for beta dist with fixed a parameter
#     if N == 1:
#         return beta(a, b).rvs(N)[0]
#     else:
#         return beta(a, b).rvs(N)
