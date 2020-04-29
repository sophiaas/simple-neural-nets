import numpy as np
import math


def center(X):
    return X - np.mean(X)


def normalize(X, max_val=None):
    X += np.min(X)
    if max_val is None:
        X /= np.max(X)
    else:
        X /= max_val
    return X


def standardize(X):
    mean = np.mean(X)
    std = np.std(X)
    X -= mean
    X /= std
    return X
