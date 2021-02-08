from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def logit(x, w):
    return np.dot(x, w)


def sigmoid(h):
    return 1. / (1 + np.exp(-h))


class MyLogisticRegression(object):
    def __init__(self, l1_coef, l2_coef):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)
