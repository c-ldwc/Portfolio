import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def R_hat(chains, warmup=2000):
    """
    chains is a n_chains X n_iterations X n_params array
    """
    n = chains.shape[1] - warmup
    m = chains.shape[0]
    p = chains.shape[2]

    chains = chains[:, warmup:, :]
    phi_j = chains.mean(
        axis=1
    )  # means for each parameter for each chain n_chains X n_params

    phi_bar = phi_j.mean(axis=0).reshape(
        1, -1
    )  # means of the above quantity over the chains 1 X n_params

    B = (
        ((phi_j - phi_bar) ** 2).sum(axis=0) * 1 / (m - 1)
    )  # between seq var: 1 X n_params
    W_var = np.sum((chains - phi_j.reshape(m, 1, p)) ** 2, axis=1) / (
        n - 1
    )  # within chain variance: n_chain x n_params
    W = np.mean(W_var, axis=0)

    var_estimand = (n - 1) / n * W + B
    return np.sqrt(var_estimand / W)


def plot_chains(chains, warmup, names):
    """
    chains is a n_chains X n_iterations X n_params array
    """

    chains = chains[:, warmup:, :]
    fig,ax = plt.subplots(chains.shape[2], 1)
    for p in range(chains.shape[2]):
        for chain in range(chains.shape[0]):
            ax[p].plot(
                    [i for i in range(chains.shape[1])],
                    chains[chain, :, p],
                    alpha=0.6,
                    c = 'blue'
                )
            
            ax[p].set_ylabel(names[p])
    fig.tight_layout()

    return None


def param_scatter(params, names, warmup=1000, plot_params = {'c':'k', 's':4}):
    data = params[warmup:, :]

    fig, ax = plt.subplots(
        params.shape[1], params.shape[1]
    )  
    for i in range(params.shape[1]):
        for j in range(params.shape[1]):
            if j >= i:
                
                plot_params['x'] = data[:, j]
                plot_params['y'] = data[:, i]
                ax[i, j].scatter(**plot_params)
            if j == i:
                ax[i,i].set_xlabel(names[i])
                ax[i,i].set_ylabel(names[i])

    fig.tight_layout()
