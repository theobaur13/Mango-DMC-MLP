import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def squared_error(y, y_hat):
    return 0.5 * np.sum((y - y_hat) ** 2)

def absolute_error(y, y_hat):
    return np.sum(np.abs(y - y_hat))