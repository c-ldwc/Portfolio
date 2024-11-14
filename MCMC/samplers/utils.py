import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_chains(chains, warmup):
    """
    chains is a n_chains X n_iterations X n_params array
    """

    chains = chains[:, warmup:, :]
    fig = make_subplots(chains.shape[2], 1)
    for p in range(chains.shape[2]):
        for chain in range(chains.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(chains.shape[1])],
                    y=chains[chain, :, p],
                    mode="lines",
                    name=f"chain {chain}",
                    opacity=0.6,
                    showlegend = False
                ),
                p + 1,
                1,
            )
    return fig


def param_scatter(params, names, warmup=1000):
    data = params[warmup:, :]

    fig = make_subplots(
        params.shape[1], params.shape[1]
    )  # , vertical_spacing = .1, horizontal_spacing = .1)
    for i in range(params.shape[1]):
        for j in range(params.shape[1]):
            if j >= i:
                fig.add_trace(
                    go.Scatter(
                        x=data[:, j],
                        y=data[:, i],
                        mode="markers",
                        marker=dict(
                            color="White", size=5, line=dict(color="Black", width=1)
                        ),
                        showlegend=False,
                    ),
                    i + 1,
                    j + 1,
                )
            if j == i:
                fig.update_yaxes(title_text=names[i], row=i + 1, col=j + 1)
                fig.update_xaxes(title_text=names[i], row=i + 1, col=j + 1)

    return fig
