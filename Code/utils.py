import numpy as np

def gaussian_beta_nll(y_true, y_mean, y_std, beta=0.5, eps=1e-6, reduction="mean"):
    y_true = np.asarray(y_true)
    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)

    var = np.maximum(y_std, eps) ** 2
    nll = 0.5 * (y_true - y_mean) ** 2 / var + 0.5 * np.log(2 * np.pi * var)
    weighted = (var ** beta) * nll   # sigma^{2*beta} weighting

    if reduction == "sum":
        return np.sum(weighted)
    if reduction == "none":
        return weighted
    return np.mean(weighted)
