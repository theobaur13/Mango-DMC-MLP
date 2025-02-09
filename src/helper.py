import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def mean_squared_error(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mean_absolute_error(y, y_hat):
    return np.mean(np.abs(y - y_hat))