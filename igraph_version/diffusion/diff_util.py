import math
import numpy as np
import torch
from masked_diffusion import GaussianDiffusion
from masked_diffusion import ModelMeanType, ModelVarType, LossType
import igraph as ig

def cosine_beta_schedule(T, s=0.008, max_beta=0.999):
    """
    Cosine beta schedule from Nichol & Dhariwal (2021).

    Decays ᾱ_t more slowly than the linear schedule, preserving more signal
    at intermediate timesteps.  The offset `s` prevents betas from being
    too small near t=0.
    """
    alpha_bar = lambda t: math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(T):
        betas.append(min(1.0 - alpha_bar((i + 1) / T) / alpha_bar(i / T), max_beta))
    return np.array(betas, dtype=np.float64)


def linear_beta_schedule(T):
    return torch.linspace(1e-4, 2e-2, T, dtype=torch.float64).numpy()


def create_diffusion(T=1000, schedule="cosine"):
    """
    Build a GaussianDiffusion object.

    Parameters
    ----------
    T        : total diffusion timesteps
    schedule : "cosine" (default — Nichol & Dhariwal 2021, slower signal decay)
               or "linear" (Ho et al. 2020)
    """
    betas = cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=True,
    )

    return diffusion

def network_to_dense(net):
    """
    Convert a single network dict to (x [n,7], adj [n,n]) dense tensors.
    Shared by preprocess() and the SimCLR diffusion augmentation path.
    """
    graph     = net["graph"]
    n         = graph.vcount()
    laundering = net["laundering_nodes"]
    depths    = net["node_depths"]

    x = torch.zeros(n, 6)

    degrees_raw     = graph.degree()
    max_deg         = max(max(degrees_raw), 1)
    betweenness_raw = graph.betweenness(directed=False)
    betw_denom      = max(1.0, (n - 1) * (n - 2) / 2)
    clustering_raw  = graph.transitivity_local_undirected(mode="zero")
    pagerank_raw    = graph.pagerank()
    _assort         = graph.assortativity_degree(directed=False)
    assortativity   = 0.0 if (_assort is None or math.isnan(_assort)) else _assort

    for i in range(n):
        if "name" in graph.vs.attributes():
            node_id = int(graph.vs[i]["name"])
        else:
            node_id = i

        x[i, 0] = int(node_id in laundering)                      # binary
        x[i, 1] = degrees_raw[i] / max_deg                        # normalised degree [0,1]
        x[i, 2] = betweenness_raw[i] / betw_denom                 # betweenness [0,1]
        x[i, 3] = clustering_raw[i]                               # clustering [0,1]
        x[i, 4] = pagerank_raw[i]                                 # pagerank
        x[i, 5] = assortativity                                   # assortativity [-1,1]

    adj = torch.from_numpy(np.array(graph.get_adjacency().data, dtype=np.float32))

    return x, adj


def preprocess(networks, save_path="cached_dataset.pt"):

    data = [network_to_dense(net) for net in networks]

    torch.save(data, save_path)

    print(f"Saved {len(data)} graphs to {save_path}")


def build_igraph_from_transactions(tx_df):
    """
    Build an undirected igraph graph from transactions dataframe.
    """
    g = ig.Graph.DataFrame(
        tx_df[["From_Account_int", "To_Account_int"]],
        directed=False,
        use_vids=False
    )
    return g
