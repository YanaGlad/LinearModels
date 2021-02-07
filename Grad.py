import numpy as np


def f(x):
    return np.sum(np.sin(x) ** 2, axis=0)


def grad_f(x):
    return np.array([2 * np.sin(x[0]) * np.cos(x[0]),
                     2 * np.sin(x[1]) * np.cos(x[1])])


