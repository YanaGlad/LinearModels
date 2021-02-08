from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from BatchGenerator import generate_batches


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

        X_train = np.concatenate((np.ones(n, 1)), X, axis=1)
        losses = []

        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                prediction = sigmoid(logit(X_batch, self.w))
                step = lr * self.get_grad(X_batch, y_batch, prediction)
                self.w -= step
                losses.append(self.___loss(y_batch, prediction))
        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        wc = np.copy(self.w)
        wc[0] = 0

        grad = np.dot(X_batch.T, predictions - y_batch) / len(y_batch)

        l1 = self.l1_coef * np.sign(wc)
        l2 = 2 * self.l1_coef * np.eye(wc.shape[0]) @ wc

        return grad + l1 + l2
