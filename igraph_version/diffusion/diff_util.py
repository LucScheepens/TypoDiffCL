import torch
from masked_diffusion import GaussianDiffusion
from masked_diffusion import ModelMeanType, ModelVarType, LossType


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

        x = torch.zeros(n, 3)

        for i in range(n):

            if "name" in graph.vs.attributes():
                node_id = int(graph.vs[i]["name"])
            else:
                node_id = i

            # laundering
            x[i, 0] = int(node_id in laundering)

            # degree
            x[i, 1] = graph.degree(i)

            # depth
            x[i, 2] = depths.get(node_id, 0)


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



