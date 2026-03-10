"""
graph.py — Build correlation graph from training data.

Stocks are nodes. An edge connects stocks whose historical
returns have |correlation| > threshold.
"""

import numpy as np
import torch
import config


def build_graph(log_returns_df):
    """
    Build edge_index from Pearson correlations of training data.
    Returns: edge_index [2, E] tensor
    """
    # Use only training portion
    split = int(len(log_returns_df) * config.TRAIN_RATIO)
    train_data = log_returns_df.values[:split]

    corr = np.corrcoef(train_data.T)   # [N, N]
    N = corr.shape[0]

    src, tgt = [], []
    for i in range(N):
        for j in range(N):
            if abs(corr[i, j]) > config.CORR_THRESHOLD:
                src.append(i)
                tgt.append(j)

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    return edge_index
