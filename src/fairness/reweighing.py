import numpy as np

def compute_group_weights(protected, group_col=None):
    if group_col is None:
        groups = protected
    else:
        groups = protected[group_col]

    weights = np.ones(len(groups))
    for g in groups.unique():
        idx = groups == g
        weights[idx] = len(groups) / sum(idx)

    return weights
