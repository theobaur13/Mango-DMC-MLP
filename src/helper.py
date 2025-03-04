import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(q):
    return np.where(q > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(q, alpha=0.01):
    return np.where(q > 0, 1, alpha)

def identity(x):
    return x

def mean_squared_error(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mean_absolute_error(y, y_hat):
    return np.mean(np.abs(y - y_hat))