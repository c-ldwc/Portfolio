import bandits
from scipy.stats import beta
import numpy as np

from typing import Callable
from copy import deepcopy

from pandas import DataFrame
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def bandit_summaries(sims, arm_mu):
    """
    Generate summaries for simulations, grouped by source
    """
    regret = sims.groupby(["sim", "source"]).act.apply(
        lambda x: bandits.regret(x, arm_mu)
    )
    regret_trajectories = (
        sims.groupby(["sim", "source"])
        .act.apply(lambda x: bandits.cumulative_reg(x, arm_mu))
        .explode()
    )

    running_means = (
        sims.sort_values(["source", "sim", "bandit_iter"])
        .groupby(
            [
                "source",
                "sim",
                "act",
            ]
        )
        .obs.apply(bandits.running_mean)
        .explode()
        .reset_index()
    )

    running_means["i"] = running_means.groupby(["source", "sim", "act"]).cumcount()

    mean_obs = sims.groupby(["bandit_iter", "source"], as_index=False).agg(
        {"obs": "mean"}
    )

    return {
        "regret": regret,
        "regret_trajectories": regret_trajectories,
        "running_means": running_means,
        "mean_obs": mean_obs,
    }


def ucb_single_sim(T, arm_mu, reward_fn=None):
    n_arms = len(arm_mu)
    ucb = bandits.UCB(K=n_arms, reward_fn=reward_fn, bound_fn=bandits.bound, T=T)

    obs = np.ones(ucb.T - ucb.K) * -999

    params = np.ones((ucb.T - ucb.K, ucb.K, 2)) * -999
    actions = np.ones((ucb.T - ucb.K))
    while ucb.t < ucb.T:
        ucb.choose_action()
        ucb.update()
        params[ucb.t - 1 - ucb.K, :, :] = np.asanyarray(deepcopy(ucb.bounds))
        actions[ucb.t - 1 - ucb.K] = ucb.action
        obs[ucb.t - 1 - ucb.K] = ucb.reward

    return obs, actions, ucb.running_means, params


def TS_single_sim(T, arm_mu, reward_fn=None):
    n_arms = len(arm_mu)
    bandit = bandits.thompsonSampling(K=n_arms, reward_fn=reward_fn, N=1, prior=[1, 1])

    obs = np.ones(T) * -999

    params = np.ones((T, bandit.K, 2)) * -999
    actions = np.ones(T) * -999
    while bandit.t < T:
        bandit.choose_action()
        bandit.observe_reward()
        bandit.update()
        # Saving params as 95% bounds on the posterior. Can reconstruct beta params from obs and act later
        params[bandit.t - 1, 0, :] = beta(
            bandit.act_priors[0][0], bandit.act_priors[0][1]
        ).ppf([0.025, 0.975])
        params[bandit.t - 1, 1, :] = beta(
            bandit.act_priors[1][0], bandit.act_priors[1][1]
        ).ppf([0.025, 0.975])
        actions[bandit.t - 1] = bandit.action[0]
        obs[bandit.t - 1] = bandit.last_rewards[0]

    return obs, actions, bandit.running_means, params


def simulate(
    N: int = 1000,
    T: int = 1000,
    n_out=None,
    arm_mu=[0.5, 0.3],
    single_sim: Callable = None,
    reward_fn: Callable = None,
):
    if n_out is None:
        n_out = T

    data = DataFrame(
        {
            "obs": [None] * (n_out) * N,
            "act": [None] * (n_out) * N,
            "bounds_0_low": [None] * (n_out) * N,
            "bounds_0_high": [None] * (n_out) * N,
            "bounds_1_low": [None] * (n_out) * N,
            "bounds_1_high": [None] * (n_out) * N,
            "bandit_iter": np.tile(range((n_out)), N),
            "sim": np.repeat(range(N), (n_out)),
        }
    )
    for sim_i in trange(N):
        obs, act, _, bounds = single_sim(T, arm_mu, reward_fn=reward_fn)
        data.loc[data.sim == sim_i, "obs"] = obs
        data.loc[data.sim == sim_i, "act"] = act
        data.loc[data.sim == sim_i, "bounds_0_low"] = bounds[:, 0, 0]
        data.loc[data.sim == sim_i, "bounds_0_high"] = bounds[:, 0, 1]
        data.loc[data.sim == sim_i, "bounds_1_low"] = bounds[:, 1, 0]
        data.loc[data.sim == sim_i, "bounds_1_high"] = bounds[:, 1, 1]
    data["act"] = data.act.astype(int)
    return data


def mean_plots(running_means):
    fig, ax = plt.subplots(2, 2, sharex= True)
    running_means.loc[(running_means.obs < 0) | (running_means.obs > 1), "obs"] = np.nan
    for j, source in enumerate(running_means.source.unique()):
        data = running_means.loc[running_means.source == source]
        for i, act in enumerate(running_means.act.unique()):
            act_data = data.loc[data.act == act]
            ax[j, i].set_title(f"{source} k = {act}")
            for sim in trange(running_means.sim.max() + 1):
                sim_data = act_data.loc[data.sim == sim]
                if not sim_data.empty:
                    sim_data.plot(
                        x="i", y="obs", ax=ax[j, i], c="b", alpha=0.05, legend=None
                    )
    fig.tight_layout()

    return fig


def regret_plots(df, T):
    df = df.reset_index()
    df["t"] = df.groupby(["source", "sim"]).cumcount()
    reg_means = df.pivot(index=["source", "t"], columns="sim").mean(
        axis=1
    )  # means across simulations at time t

    n_sims = df.sim.max()

    df = df.set_index(["sim", "source"])
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].set_title("Thompson Sampling")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("regret")

    ax[1].set_title("UCB")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("regret")

    # df = df.set_index(['sim', 'source'])
    for i in trange(n_sims):
        ax[0].plot(range(T), df.loc[(i, "Thompson Sampling"), "act"], c="k", alpha=0.1)

        ax[1].plot(range(2, T), df.loc[(i, "UCB"), "act"], c="k", alpha=0.1)

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].plot(range(T), reg_means[("Thompson Sampling", slice(None))], c="r")

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].plot(range(2, T), reg_means[("UCB", slice(None))], c="r")
    fig.tight_layout()

    return fig


def bound_plots(df):
    algos = df.source.unique()
    n_algos = len(algos)
    n_sims = df.sim.max()

    fig, ax = plt.subplots(n_algos, figsize=(8, 4.5))

    for k, algo in enumerate(algos):
        ax[k].set_title(algo)
        theseSims = df[df.source == algo]
        for sim in trange(n_sims):
            thisSim = theseSims[(theseSims.sim == sim)]
            ax[k].plot(  # arm_0 lower
                range(thisSim.shape[0]), thisSim.bounds_0_low, alpha=0.01, c="b"
            )

            ax[k].plot(  # arm_0 upper
                range(thisSim.shape[0]), thisSim.bounds_0_high, alpha=0.01, c="b"
            )

            ax[k].plot(  # arm_1 lower
                range(thisSim.shape[0]), thisSim.bounds_1_low, alpha=0.01, c="r"
            )

            ax[k].plot(  # arm_1 upper
                range(thisSim.shape[0]), thisSim.bounds_1_high, alpha=0.01, c="r"
            )

        ax[k].set_ylim(-0.4, 1)

        legend_elements = [
        Line2D([0], [0], color='red', label='μ = 0.2'),
        Line2D([0], [0], color='blue', label='μ = 0.3')
    ]
    ax[0].legend(handles = legend_elements)
    return fig
