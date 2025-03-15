import numpy as np
from scipy.stats import f as f_distrib


def hotelling_t2(x, y):
    nx, _, p = x.shape
    ny, _, _ = y.shape
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    centred_x = x - mean_x
    centred_y = y - mean_y
    cov_x = np.matmul(
        centred_x[:, :, :, None], centred_x[:, :, :, None].transpose(0, 1, 3, 2)).sum(0)
    cov_y = np.matmul(
        centred_y[:, :, :, None], centred_y[:, :, :, None].transpose(0, 1, 3, 2)).sum(0)
    cov = (cov_x + cov_y) / (nx + ny - 2)
    precision = np.linalg.inv(cov)
    t2 = np.einsum(
        '...ij,...i,...j', precision, mean_x - mean_y, mean_x - mean_y) * nx * ny / (nx + ny)
    f_stat = t2 * (nx + ny - p - 1) / ((nx + ny - 2) * p)
    pval = 1. - f_distrib.cdf(f_stat, p, nx + ny - p - 1)
    return pval, t2
