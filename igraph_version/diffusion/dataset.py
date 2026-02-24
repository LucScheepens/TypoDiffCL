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

        x = torch.zeros(n, 3)


        laundering = data["laundering_nodes"]
        depths = data["node_depths"]


        # igraph vertex indices: 0..n-1
        for i in range(n):

            # Try to recover original node id (if stored)
            if "name" in graph.vs.attributes():
                node_id = int(graph.vs[i]["name"])
            else:
                node_id = i


            # Feature 1: laundering
            x[i, 0] = int(node_id in laundering)

            # Feature 2: degree (safe now)
            x[i, 1] = graph.degree(i)

            # Feature 3: depth
            x[i, 2] = depths.get(node_id, 0)


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

    def __init__(self, path):

        self.data = torch.load(path)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        return self.data[idx]