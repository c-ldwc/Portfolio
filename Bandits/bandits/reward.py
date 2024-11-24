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

    def reward(self,act,t):
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

    def reward(self, act,t):
        """
        Get a reward for action `act` with mean self.mus[act]
        """
        if self.N == 1:
            return binom(p=self.mus[act], n=1).rvs(self.N)[0]
        else:
            return binom(p=self.mus[act], n=1).rvs(self.N)


@dataclass
class binom_sin_reward:
    mus: ndarray[float] = None
    N: int = 1
    period: int = 7
    amp: float = .1

    def reward(self, act,t):
        mu = self.mus[act] + 0.01 * sin(2 * pi * t / self.period)
        if mu > 1: mu = 1
        if mu < 0: mu = 0
        if self.N == 1:
            return binom(p=mu, n=1).rvs(self.N)[0]
        else:
            return binom(p=mu, n=1).rvs(self.N)
        
@dataclass
class binom_weekend_reward:
    mus: ndarray[float] = None
    N: int = 1
    amp: float = .1

    def reward(self, act,t):
        mu = self.mus[act] 
        if t in (6,7):
            mu = mu + self.amp
        if mu > 1: mu = 1
        if mu < 0: mu = 0
        if self.N == 1:
            return binom(p=mu, n=1).rvs(self.N)[0]
        else:
            return binom(p=mu, n=1).rvs(self.N)

