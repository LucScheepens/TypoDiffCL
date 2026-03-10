from igraph import Graph, plot
from pathlib import Path
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

from torch_geometric.data import Batch

from augmentation import augment_network_view_fast
from simclr import GraphEncoder, prepare_networks, network_to_pyg_data_fast


def plot_graph_from_dict_igraph(graph_dict, save_path=None, show=True, dpi=300):
    """
    Parameters
    ----------
    graph_dict : dict
        Contains 'start_node', 'nodes', 'transactions' (DataFrame)

    save_path : str or Path or None
        If provided, saves the figure to this path

    show : bool
        Whether to display the plot

    dpi : int
        Resolution for saved image
    """

    df = graph_dict["transactions"]

    laundering_nodes = set(graph_dict.get("laundering_nodes", []))
    collapsed_nodes = set(graph_dict.get("collapsed_nodes", []))
    start = graph_dict.get("start_node")

    edges = []
    laundering_edges = set()
    observed_nodes = set()

    # ---- Build edge list + collect nodes ----
    for _, row in df.iterrows():
        src = row["From_Account_int"]
        dst = row["To_Account_int"]

        if src != dst:
            edges.append((src, dst))
            observed_nodes.update([src, dst])

        if row["Is Laundering"] == 1 and src != dst:
            laundering_edges.add((src, dst))

    # ---- Final node set (union of everything) ----
    all_nodes = set(graph_dict.get("nodes", [])) \
        | set(graph_dict.get("collapsed_nodes", [])) \
        | set(graph_dict.get("laundering_nodes", [])) \
        | observed_nodes

    # ---- Build mapping: node ID -> igraph index ----
    all_nodes = list(all_nodes)
    node_id_to_idx = {node_id: i for i, node_id in enumerate(all_nodes)}

    # ---- Remap edges to igraph indices ----
    edges_idx = [(node_id_to_idx[src], node_id_to_idx[dst]) for src, dst in edges]

    # ---- Create igraph graph ----
    G = Graph(directed=True)
    G.add_vertices(len(all_nodes))
    G.add_edges(edges_idx)

    # Set vertex names for lookup + labeling
    G.vs["name"] = all_nodes

    # ---- Vertex colors ----
    vertex_colors = []
    for v in G.vs["name"]:
        if v in laundering_nodes:
            vertex_colors.append("red")
        elif v in collapsed_nodes:
            vertex_colors.append("gray")
        else:
            vertex_colors.append("lightblue")

    # Highlight start node
    vertex_sizes = [10] * G.vcount()
    if start in node_id_to_idx:
        start_idx = node_id_to_idx[start]
        vertex_colors[start_idx] = "yellow"
        vertex_sizes[start_idx] = 20

    G.vs["color"] = vertex_colors
    G.vs["size"] = vertex_sizes
    G.vs["label"] = [""] * G.vcount()

    # ---- Edge colors (aligned with edges list) ----
    edge_colors = [
        "red" if (src, dst) in laundering_edges else "black"
        for src, dst in edges
    ]
    G.es["color"] = edge_colors
    G.es["arrow_size"] = [0.6] * G.ecount()

    # ---- Layout ----
    layout = G.layout_fruchterman_reingold()

    # ---- Plot ----
    visual_style = {
        "layout": layout,
        "vertex_color": G.vs["color"],
        "vertex_size": G.vs["size"],
        "vertex_label": G.vs["label"],
        "edge_color": G.es["color"],
        "edge_arrow_size": G.es["arrow_size"],
        "bbox": (1400, 1400),
        "margin": 40,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot(G, save_path, **visual_style, dpi=dpi)

    if show:
        plot(G, **visual_style)



def plot_simclr_latent_space_laundering_vs_clean(
    networks,
    df_full
):
    """
    Plot the SimCLR encoder latent space, colored by presence of laundering nodes.
    """ 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("model_checkpoints/best_model_run1.pt", map_location=device)

    encoder = GraphEncoder(in_dim=3, hidden_dim=64, out_dim=128).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    full_graph = prepare_networks(networks, df_full)

    graphs = []
    labels = []   # 1 = laundering present, 0 = none

    for net in networks:
        v = augment_network_view_fast(net, full_graph)
        graphs.append(network_to_pyg_data_fast(v))

        has_laundering = len(net["laundering_nodes"]) > 0
        labels.append(int(has_laundering))

    labels = np.array(labels)

    data = Batch.from_data_list(graphs).to(device)

    with torch.no_grad():
        h = encoder(data)

    H = h.cpu().numpy()

    # Normalize (important for contrastive models)
    H = H / np.linalg.norm(H, axis=1, keepdims=True)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    H_2d = reducer.fit_transform(H)

    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        H_2d[:, 0],
        H_2d[:, 1],
        c=labels,
        cmap="coolwarm",   # blue = non-laundering, red = laundering
        s=30,
        alpha=0.85
    )

    plt.title("SimCLR Encoder Latent Space\nRed = Laundering Networks | Blue = Clean Networks")
    plt.xticks([])
    plt.yticks([])

    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(["No laundering nodes", "Has laundering nodes"])

    plt.tight_layout()
    plt.show()

    plt.savefig("simclr_latent_space_laundering_vs_clean.png", dpi=300)
