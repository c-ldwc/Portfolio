from copy import deepcopy
from typing import Callable

import numpy as np
from numpy import ndarray
from scipy.stats import multivariate_normal, uniform
from tqdm import tqdm


def hmc(
    M: np.ndarray,
    data: dict,
    n_iter: int,
    log_prob: Callable,
    grad: Callable,
    starting: ndarray,
    progress: bool = True,
    tune_accept: bool = True,
    adapt: bool = True,
    L: int = 10,
    eps: float = 1e-6,
):
    if not isinstance(starting, float):
        n_param = len(starting)
    else:
        n_param = 1

    samples = np.zeros((n_iter, n_param))  

    samples[0, :] = starting


    def g(x, covar):
        return multivariate_normal(x, covar)

    mmntm = multivariate_normal(np.zeros(n_param), M)

    if progress:
        iterator = tqdm(range(1, n_iter))
    else:
        iterator = range(1, n_iter)

    if type(M) == np.ndarray:
        M_inv = np.linalg.inv(M)
    elif type(M) == float:
        M_inv = 1 / M

    # Dict for last sample params for log_prob
    last_prop = deepcopy(data)

    grad_params = deepcopy(data)

    accept_rate = 1

    for i in (pbar := iterator):
        last_sample = samples[i - 1, :]
        proposal = last_sample.copy().reshape(-1, 1)
        # print(f"starting prop: {proposal}")
        mmntm_prop = mmntm.rvs(1).reshape(-1, 1)
        start_mmntm = mmntm_prop.copy()
        for l in range(L):  # leapfrog
            grad_params["proposal"] = proposal
            # print(f"l: {l}, proposal: {proposal}, mmntm_prop:{mmntm_prop}, grad:{grad(**grad_params)}")

            mmntm_prop += 1 / 2 * eps * grad(**grad_params)
            proposal += eps * np.dot(M_inv, mmntm_prop)
            grad_params["proposal"] = proposal
            mmntm_prop += 1 / 2 * eps * grad(**grad_params)
            if any(np.isnan(proposal)) or any(np.isnan(mmntm_prop)):
                raise ValueError('Nans detected in leapfrog')
        data["proposal"] = proposal
        last_prop["proposal"] = last_sample
        # print(mmntm_prop)

        r = (
            log_prob(**data)
            + mmntm.logpdf(mmntm_prop.flatten())
            - log_prob(**last_prop)
            - mmntm.logpdf(start_mmntm.flatten())
        )

        if np.isnan(r):
            raise ValueError('r is nan')

        # print(r)
        # print(proposal)
        u = np.log(uniform(0, 1).rvs(1)[0])
        accept = int(u < r)
        samples[i, :] = ([last_sample, proposal][accept]).flatten()
        accept_rate = (accept_rate * i + accept) / (i + 1)
        if (i + 1) % 50 == 0:
            pbar.set_description(f"accept rate: {np.round(accept_rate,3)}")
    return samples


def metro_hastings(
    data: dict,
    n_iter: int,
    log_prob: Callable,
    starting: list,
    progress: bool = True,
    bounds=False,
    sigma: np.ndarray = False,
    tune_accept: bool = True,
    adapt: bool = True,
) -> ndarray:
    """ """

    if not isinstance(starting, float):
        n_param = len(starting)
    else:
        n_param = 1

    samples = np.zeros((n_iter, n_param))

    samples[0, :] = starting

    # def g(center, width=.05):
    #     return uniform(center-width/2, width)

    def g(x, covar):
        return multivariate_normal(x, covar)

    if progress:
        iterator = tqdm(range(1, n_iter))
    else:
        iterator = range(1, n_iter)

    if bounds:
        bounded_params = [i for i in range(n_param) if bounds[i] is not None]

    # Dict for last sample params for log_prob
    last_prop = deepcopy(data)

    accept_rate = 1

    scale = 1

    for i in iterator:
        last_sample = samples[i - 1, :]

        if adapt:
            if i > 200:
                # use sample covariance matrix as approximation to posterior
                sigma = 2.4**2 / n_param * np.cov(samples[:i, :].T)

        proposal = g(last_sample, scale * sigma).rvs(1)

        data["proposal"] = proposal
        last_prop["proposal"] = last_sample
        # print(data)
        a1 = log_prob(**data) - log_prob(**last_prop)
        # a2 = np.sum(np.log(g(proposal).pdf(last_sample))) - np.sum(
        #     np.log(g(last_sample).pdf(proposal))
        # )

        a = a1  # + a2

        accept = int(np.log(uniform(0, 1).rvs(1)[0]) < a)

        samples[i, :] = [last_sample, proposal][accept]

        accept_rate = (accept_rate * i + accept) / (i + 1)  # running mean

        if (i + 1) % 20 == 0:
            iterator.set_description(f"acceptance = {np.round(accept_rate,3)}")

        if tune_accept:
            if i > 200:
                if accept_rate > 0.44:
                    scale = scale * 1.01
                if accept_rate < 0.44:
                    scale = scale * 0.99

    return samples


if __name__ == "__main__":
    import plotly.express as px
    from scipy.stats import poisson

    # def log_prob(data, X, proposal):
    #     mu = X @ proposal
    #     log_lik = np.sum(-((data - mu) ** 2) / 6)
    #     prior = -np.dot(proposal.T, proposal) / 8
    #     return log_lik + prior

    # X = multivariate_normal([0, 0], np.eye(2, 2) * 3).rvs(1000)
    # coef = [1, 3.0]
    # y = (X @ coef + multivariate_normal([0], [3]).rvs(1000)).reshape(-1, 1)

    # def grad(data, X, proposal):
    #     return ((data - X @ proposal) / 3 * X).sum(axis=0).reshape(-1, 1) - data.shape[
    #         0
    #     ] * proposal / 4

    # hmc_param = hmc(
    #     M=np.eye(2, 2) * 2,
    #     data={"data": y, "X": X},
    #     grad=grad,
    #     n_iter=5000,
    #     log_prob=log_prob,
    #     starting=[2, 2],
    #     eps=0.02,
    #     L=10,
    # )

    # print("mean of samples")
    # print(hmc_param[500:, :].mean(axis=0))
    # print("variance of samples")
    # print(hmc_param[500:, :].var(axis=0))
    # print("90% quantiles")
    # print(np.quantile(hmc_param[500:, :], (0.025, 0.975), axis=0))

    # fig = px.line(x=range(5000), y=hmc_param[:, 0])
    # fig.show()

    # fig = px.scatter(x=hmc_param[:, 1], y=hmc_param[:, 0])
    # fig.show()
    def regression_unnormed_posterior(data, X, proposal):
        data = data.flatten()
        mu = X @ proposal
        lam = np.exp(mu)
        return np.sum(data*mu - lam) - np.dot(proposal.T, proposal)/2

    def grad(proposal, data, X):
        """
        gradient of log posterior prob for poisson regression
        """
        return ((data - np.exp(X @ proposal))*X ).sum(axis = 0).reshape(-1,1) - proposal/2


    def create_regression(N=1000, p=2):
        coef = np.array([.4,2.2])#uniform(-.5,1).rvs(p+1)

        X = multivariate_normal(np.zeros(p), np.eye(p, p) * 1).rvs(N)
        X = np.c_[X, np.ones(N)]

        y = poisson(np.exp(X @ coef)).rvs(N)
        y = y.reshape(-1,1)
        coef = coef.reshape(-1,1)
        return X, y, coef


    X, y, coef = create_regression(p = 1)

    hmc_param = hmc(
        M=np.eye(2,2)*1.5,
        data={"data": y, "X": X},
        grad=grad,
        n_iter=1000,
        log_prob=regression_unnormed_posterior,
        starting=[0, 0],
        eps=.002,
        L=500,
    )

    'here'