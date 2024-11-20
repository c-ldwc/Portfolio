from numpy import sqrt as _sqrt, log as _log


def radius(T):
    return [_sqrt(2 * _log(T) / i) for i in range(T)]


def bound(mean: float, t_a: int, T: int) -> list[int]:
    """
    Calculate confidence bound for an arm
    mean is mean for this arm
    t_a is number of time arm sampled
    T is time horizon
    """
    assert (
        t_a <= T
    ), "You have a list of observations that is longer than the time horizon"
    conf_rad = _sqrt(2 * _log(T) / t_a)  # confidence radius
    return [mean - conf_rad, mean + conf_rad]


def regret(a: list[int], arm_mu: list[int]) -> float:
    """
    regret for the current sequence
    obs is reward observations so far
    a is list of actions
    arm_mu is list of true arm reward means
    """
    best_mu = max(arm_mu)
    t = len(a)
    return t * best_mu - sum([arm_mu[a_s] for a_s in a])


# I think this fn is O(n) whereas
# [bandit_mod.regret(x[:i], arm_mu) for i in range(1500)] is O(n(n+1)/2) = O(n^2)
def cumulative_reg(act, arm_mu):
    reg = [0] + [None] * (len(act) - 1)
    best_mu = max(arm_mu)
    for i in range(1, len(act)):
        reg[i] = reg[i - 1] + best_mu - arm_mu[act.iloc[i]]
    return reg


def running_mean(obs):
    means = [obs.iloc[0]] + [None] * (len(obs) - 1)
    N = 1
    for i in range(1, len(obs)):
        means[i] = (means[i - 1] * N + obs.iloc[i]) / (N + 1)
        N += 1
    return means
