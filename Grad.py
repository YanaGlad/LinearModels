import numpy as np


def f(x):
    return np.sum(np.sin(x) ** 2, axis=0)


def grad_f(x):
    return np.array([2 * np.sin(x[0]) * np.cos(x[0]),
                     2 * np.sin(x[1]) * np.cos(x[1])])


def grad_descent_2d(f, grag_f, lr, num_iter=100, x0=None):
    if x0 is None:
        x0 = np.random.random(2) # 2 - [x,y], 3 - [x,y,z]...

    history = []

    curr_x = x0.copy()

    for iter_num in range(num_iter):
        entry = np.hstack((curr_x, f(curr_x)))
        history.append(entry)
        grad = grad_f(curr_x)
        curr_x -= grad * lr

    return np.vstack(history)

steps = grad_descent_2d(f, grad_f, lr=0.1, num_iter=20)
