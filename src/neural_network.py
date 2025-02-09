import numpy as np
from src.helper import sigmoid, relu

def init_weights(p, U, L, seed):
    np.random.seed(seed)
    
    # Randomly initialise weight matrix W. W(1) has dimensions of U1 x p, W(l) has dimensions of Ul x Ul-1. W(L) has dimensions of 1 x UL.
    weights = []
    for i in range(L):
        if i == 0 and L == 1:
            weights.append(np.random.rand(1, p))                # Randomly initialise input layer weight matrix of shape    W(1) = (p, 1)
        elif i == 0:
            weights.append(np.random.rand(U[i], p))             # Randomly initialise input layer weight matrix of shape    W(1) = (p, U)
        elif i == L-1:
            weights.append(np.random.rand(1, U[i-1]))           # Randomly initialise output layer weight matrix of shape   W(L) = (1, UL-1)
        else:
            weights.append(np.random.rand(U[i], U[i-1]))        # Randomly initialise hidden layer weight matrix of shape   W(l) = (Ul, Ul-1)
    return weights

def init_biases(U, L, seed=0):
    np.random.seed(seed)
    biases = []
    if L == 1:
        biases.append(np.random.rand())
    else:
        for i in range(L):
            biases.append(np.random.rand(U[i]))
    return biases

# h = activation function
# W = weights
# x = input
# b = bias
# p = number of features
# U = number of hidden layer neurons
# L = number of layers

def nn_matrix(x, L, weights, biases):
    # Calculate variables for input layer
    activations = []

    # q(l) = h(W(l) * q(l-1) + b(l))
    q = x
    for i in range(L):
        q = np.dot(q, weights[i].T) + biases[i]
        q = relu(q)
        activations.append(q)

    return q, activations

def backpropogation(y, y_hat, activations, weights, biases, L, x, learning_rate, regularisation_lambda=1):
    layer_errors = []

    # Calculate neuron error for output layer (neuron error = activation(1 - activation)(y - y_hat))
    output_errors = activations[-1] * (1 - activations[-1]) * (y - y_hat)
    layer_errors.append(output_errors)

    # Calculate neuron error for hidden layers (neuron error = activation(1 - activation)(sum(forward weight from neuron * error forward neuron)))
    errors = output_errors

    for i in range(L-2, -1, -1):
        hidden_errors = activations[i] * (1 - activations[i]) * np.dot(errors, weights[i+1])
        errors = hidden_errors
        layer_errors.append(hidden_errors)
    
    layer_errors = layer_errors[::-1]

    # Calculate weight change for each layer (delta = learning rate * forward neuron error * activation of previous neuron)
    deltas = [np.zeros_like(w) for w in weights]
    bias_deltas = [np.zeros_like(b) for b in biases]

    for row in range(x.shape[0]):
        for i in range(L):
            if i == 0:
                delta = learning_rate * np.outer(layer_errors[i][row], x[row])
                bias = learning_rate * layer_errors[i][row]
            else:
                delta = learning_rate * np.outer(layer_errors[i][row], activations[i-1][row])
                bias = learning_rate * layer_errors[i][row]

            deltas[i] += delta
            bias_deltas[i] += bias

    # Update weights
    for i in range(L):
        weights[i] -= deltas[i] - regularisation_lambda * weights[i]
        biases[i] -= bias_deltas[i]

    return weights, biases