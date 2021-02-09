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

    def predic_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def predict(self, X, threhold=0.5):
        return self.predic_proba(X) >= threhold

    def get_weight(self):
        return self.w

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=[[-2, 0.5], [3, -0.5]], cluster_std=1, random_state=42)

colors = ("red", "green")
colored_y = np.zeros(y.size, dtype=str)

for i, cl in enumerate([0, 1]):
    colored_y[y == cl] = str(colors[i])

plt.figure(figsize=(15, 10))
plt.scatter(X[:, 0], X[:, 1], c=colored_y)
plt.show()

clf = MyLogisticRegression(0.1, 0.1)
clf.fit(X, y, epochs=1000)
w = clf.get_weight()

from matplotlib.colors import ListedColormap

plt.figure(figsize=(15, 8))

eps = 0.1
xx, yy = np.meshgrid(np.linspace(np.min(X[:, 0]) - eps, np.max(X[:, 0]) + eps, 200),
                     np.linspace(np.min(X[:, 1]) - eps, np.max(X[:, 1]) + eps, 200))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=colored_y)
plt.show()