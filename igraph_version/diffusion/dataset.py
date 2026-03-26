import torch
from torch.utils.data import Dataset


class NetworkDataset(Dataset):

    def __init__(self, network_list):

        self.networks = network_list


    def __len__(self):
        return len(self.networks)


    def __getitem__(self, idx):

        data = self.networks[idx]

        graph = data["graph"]

        n = graph.vcount()   # number of vertices


        # --------------------------------
        # Node features
        # --------------------------------

        x = torch.zeros(n, 2)

        laundering = data["laundering_nodes"]

        # igraph vertex indices: 0..n-1
        for i in range(n):

            # Try to recover original node id (if stored)
            if "name" in graph.vs.attributes():
                node_id = int(graph.vs[i]["name"])
            else:
                node_id = i

            # Feature 1: laundering
            x[i, 0] = int(node_id in laundering)

            # Feature 2: degree
            x[i, 1] = graph.degree(i)


        # --------------------------------
        # Adjacency

        adj = torch.zeros(n, n)

        edges = graph.get_edgelist()

        for i, j in edges:

            adj[i, j] = 1
            adj[j, i] = 1


        return x.float(), adj.float()

import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):

    def __init__(self, path, max_nodes=None):

        data = torch.load(path)

        if max_nodes is not None:
            before = len(data)
            data = [(x, adj) for x, adj in data if x.shape[0] <= max_nodes]
            print(f"Filtered dataset: {before} → {len(data)} graphs (max_nodes={max_nodes})")

        self.data = data


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        return self.data[idx]