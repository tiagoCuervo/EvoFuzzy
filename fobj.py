import numpy as np


# Root Mean Square Error
def rmse(pred, real):
    return np.sqrt(np.mean((pred - real) ** 2))


# Mean Absolute Difference
def abs_diff(pred, real):
    return np.mean(np.abs(pred - real))


# Pearson correlation coefficient
def corrcoeff(pred, real):
    pcorr = (np.sum(pred * real) - pred.size * np.mean(pred) * np.mean(real)) / (
                (pred.size - 1) * np.std(pred) * np.std(real))
    return pcorr


# Prediction of Change in Direction
def pocid(pred, real):
    d = np.diff(real)
    dp = np.diff(pred)
    return np.mean(np.sign(d) == np.sign(dp))


# Nash-Sutcliffe Efficiency Index
def nse(pred, real):
    return 1 - (np.sum((real - pred) ** 2)) / (np.sum((real - np.mean(real)) ** 2))


# Mean Absolute Error
def mae(pred, real):
    return np.mean(np.abs((real - pred) / real))


# Entropy
def entropy(p, q):
    return -np.sum(p * np.log(q.clip(1e-12, None)))


# Cross-entropy
def ce(pred, real):
    f_real, intervals = np.histogram(real, bins=len(np.unique(real))-1)
    intervals[0] = min(min(pred), min(real))
    intervals[-1] = max(max(pred), max(real))
    f_real = f_real / real.size
    f_pred = np.histogram(pred, bins=intervals)[0] / pred.size
    return entropy(f_real, f_pred)


# Kullback-Leibler divergence
def kldiv(pred, real):
    f_real, intervals = np.histogram(real, bins=len(np.unique(real))-1)
    intervals[0] = min(min(pred), min(real))
    intervals[-1] = max(max(pred), max(real))
    f_real = f_real / real.size
    f_pred = np.histogram(pred, bins=intervals)[0] / pred.size
    # kl = entropy(f_pred, f_real) - entropy(f_pred, f_pred)
    kl = entropy(f_real, f_pred) - entropy(f_real, f_real)
    return kl
