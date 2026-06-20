"""
graphsmote_augmentation.py
──────────────────────────
Graph-level SMOTE augmentation for laundering graph oversampling.

Adapted from GraphSMOTE (Zhao et al., NeurIPS 2021) to the graph-classification
setting used in this thesis.  The original paper operates at node level inside a
single large graph; here each sample is an ego-subgraph, so we interpolate
graph-level summary features and reuse real topologies.

Algorithm
─────────
1. Encode each laundering graph as its mean-pooled node feature vector (18-dim).
2. Build a k-NN index in this space (sklearn NearestNeighbors, Euclidean).
3. For each requested synthetic graph:
   a. Sample a seed graph i uniformly from the laundering set.
   b. Sample a random neighbour j from i's k nearest neighbours.
   c. Draw α ~ Uniform(0, 1).
   d. Compute the target mean: μ_syn = μ_i + α * (μ_j − μ_i).
   e. Pick the real graph (i or j) whose mean is closest to μ_syn as the
      topology donor — this preserves realistic adjacency structure.
   f. Shift the donor's node features by a constant so their column-wise mean
      matches μ_syn (shift-only; no per-node noise to avoid artefacts).
4. Return a list of PyG Data objects with label y=1 (laundering).

Compatibility
─────────────
Output Data objects match the format produced by evaluate_classifiers.network_to_pyg:
  x            : [n, 18]  (laundering flag col excluded)
  edge_index   : [2, E]
  y            : tensor([1])
  timestamp_val: -1.0

Usage (standalone)
──────────────────
    from experiments.graphsmote_augmentation import GraphSMOTEAugmenter
    from torch_geometric.data import Data

    augmenter = GraphSMOTEAugmenter(k=5, random_state=42)
    augmenter.fit(train_laundering_graphs)   # list[Data] with y==1
    synthetic = augmenter.generate(n=50)    # list[Data]
"""

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


def _graph_mean_features(data: Data) -> np.ndarray:
    """Mean-pool node features to a single graph-level vector."""
    return data.x.float().mean(dim=0).numpy()


class GraphSMOTEAugmenter:
    """
    Graph-level SMOTE for laundering ego-subgraph augmentation.

    Parameters
    ----------
    k : int
        Number of nearest neighbours used during interpolation (default 5).
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(self, k: int = 5, random_state: int | None = 42):
        if not _SKLEARN_OK:
            raise ImportError("sklearn is required: pip install scikit-learn")
        self.k = k
        self.rng = np.random.default_rng(random_state)
        self._fitted = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, laundering_graphs: list[Data]) -> "GraphSMOTEAugmenter":
        """
        Fit the k-NN index on the provided laundering graphs.

        Parameters
        ----------
        laundering_graphs : list[Data]
            PyG Data objects with y == 1.  Non-laundering graphs are silently
            filtered out.
        """
        graphs = [g for g in laundering_graphs if g.y.item() == 1]
        if len(graphs) < 2:
            raise ValueError(
                f"Need at least 2 laundering graphs to fit GraphSMOTE, got {len(graphs)}."
            )

        self._graphs = graphs
        self._embeddings = np.stack([_graph_mean_features(g) for g in graphs])  # [N, 18]

        k_actual = min(self.k, len(graphs) - 1)
        self._nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", algorithm="ball_tree")
        self._nn.fit(self._embeddings)
        self._fitted = True
        return self

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(self, n: int) -> list[Data]:
        """
        Generate `n` synthetic laundering graphs.

        Returns
        -------
        list[Data]
            Synthetic graphs, each with y=1 and x.shape[1] == self._graphs[0].x.shape[1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")

        _, nn_indices = self._nn.kneighbors(self._embeddings)  # [N, k]
        synthetic = []

        for _ in range(n):
            # ── pick seed and neighbour ──────────────────────────────────────
            i = int(self.rng.integers(len(self._graphs)))
            j = int(self.rng.choice(nn_indices[i]))
            alpha = float(self.rng.uniform(0.0, 1.0))

            mu_i = self._embeddings[i]
            mu_j = self._embeddings[j]
            mu_syn = mu_i + alpha * (mu_j - mu_i)

            # ── pick topology donor ──────────────────────────────────────────
            # Whichever of {i, j} has mean closer to mu_syn donates the graph
            # structure; this keeps the edge pattern realistic.
            dist_i = float(np.linalg.norm(mu_syn - mu_i))
            dist_j = float(np.linalg.norm(mu_syn - mu_j))
            donor = self._graphs[i] if dist_i <= dist_j else self._graphs[j]

            # ── shift donor node features to match μ_syn ─────────────────────
            x_donor = donor.x.float()          # [n_donor, 18]
            mu_donor = x_donor.mean(dim=0)     # [18]
            shift = torch.tensor(mu_syn, dtype=torch.float) - mu_donor
            x_syn = x_donor + shift.unsqueeze(0)

            synthetic.append(Data(
                x=x_syn,
                edge_index=donor.edge_index.clone(),
                y=torch.tensor([1], dtype=torch.long),
                timestamp_val=-1.0,
            ))

        return synthetic

    # ── convenience: fit_generate ─────────────────────────────────────────────

    def fit_generate(self, laundering_graphs: list[Data], n: int) -> list[Data]:
        """Fit on laundering_graphs then generate n synthetic samples."""
        return self.fit(laundering_graphs).generate(n)

    def __repr__(self) -> str:
        status = f"fitted on {len(self._graphs)} graphs" if self._fitted else "unfitted"
        return f"GraphSMOTEAugmenter(k={self.k}, {status})"
