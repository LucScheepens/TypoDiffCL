import torch
from masked_diffusion import GaussianDiffusion
from masked_diffusion import ModelMeanType, ModelVarType, LossType
import igraph as ig

def linear_beta_schedule(T):

    return torch.linspace(
        1e-4,
        2e-2,
        T,
        dtype=torch.float64
    ).numpy()


def create_diffusion(T=1000):

    betas = linear_beta_schedule(T)

    diffusion = GaussianDiffusion(

        betas=betas,

        model_mean_type=ModelMeanType.EPSILON,

        model_var_type=ModelVarType.FIXED_SMALL,

        loss_type=LossType.MSE,

        rescale_timesteps=True,
    )

    return diffusion

def preprocess(networks, save_path="cached_dataset.pt"):

    data = []

    for net in networks:

        graph = net["graph"]

        n = graph.vcount()

        laundering = net["laundering_nodes"]
        depths = net["node_depths"]

        # ----------------------------
        # Node features
        # ----------------------------

        x = torch.zeros(n, 6)

        # Graph-level structural features (computed once per graph)
        betweenness_raw = graph.betweenness(directed=False)
        betw_denom      = max(1.0, (n - 1) * (n - 2) / 2)
        clustering_raw  = graph.transitivity_local_undirected(mode="zero")
        pagerank_raw    = graph.pagerank()

        for i in range(n):

            if "name" in graph.vs.attributes():
                node_id = int(graph.vs[i]["name"])
            else:
                node_id = i

            x[i, 0] = int(node_id in laundering)          # binary — separate noise
            x[i, 1] = graph.degree(i)                     # degree
            x[i, 2] = depths.get(node_id, 0)             # depth
            x[i, 3] = betweenness_raw[i] / betw_denom    # betweenness centrality [0,1]
            x[i, 4] = clustering_raw[i]                   # local clustering coeff [0,1]
            x[i, 5] = pagerank_raw[i]                     # pagerank score


        # ----------------------------
        # Adjacency
        # ----------------------------

        adj = torch.zeros(n, n)

        for i, j in graph.get_edgelist():
            adj[i, j] = 1
            adj[j, i] = 1


        data.append((x, adj))


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
