import torch

def collate_fn(batch):

    xs, adjs = zip(*batch)

    B = len(xs)
    F = xs[0].shape[1]

    max_nodes = max(x.shape[0] for x in xs)

    x_pad = torch.zeros(B, max_nodes, F)
    adj_pad = torch.zeros(B, max_nodes, max_nodes)
    node_mask = torch.zeros(B, max_nodes)


    for i, (x, adj) in enumerate(zip(xs, adjs)):

        n = x.shape[0]

        x_pad[i, :n] = x
        adj_pad[i, :n, :n] = adj

        node_mask[i, :n] = 1


    return x_pad, adj_pad, node_mask