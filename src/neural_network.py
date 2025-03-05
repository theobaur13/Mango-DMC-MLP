import numpy as np
from src.helper import sigmoid, relu, identity, relu_derivative, leaky_relu, leaky_relu_derivative

def init_weights(p, U, L, seed, activation="relu"):
    np.random.seed(seed)
    
    # Randomly initialise weight matrix W. W(1) has dimensions of U1 x p, W(l) has dimensions of Ul x Ul-1. W(L) has dimensions of 1 x UL.
    weights = []
    for i in range(L):
        if activation == "relu":
            if i == 0 and L == 1:
                weights.append(np.random.rand(1, p) * np.sqrt(2 / p))                # Randomly initialise input layer weight matrix of shape    W(1) = (p, 1)
            elif i == 0:
                weights.append(np.random.rand(U[i], p) * np.sqrt(2 / p))             # Randomly initialise input layer weight matrix of shape    W(1) = (p, U)
            elif i == L-1:
                weights.append(np.random.rand(1, U[i-1]) * np.sqrt(2 / U[i-1]))           # Randomly initialise output layer weight matrix of shape   W(L) = (1, UL-1)
            else:
                weights.append(np.random.rand(U[i], U[i-1]) * np.sqrt(2 / U[i-1]))        # Randomly initialise hidden layer weight matrix of shape   W(l) = (Ul, Ul-1)
        elif activation == "sigmoid":
            if i == 0 and L == 1:
                weights.append(np.random.randn(1, p) * np.sqrt(1 / p))
            elif i == 0:
                weights.append(np.random.randn(U[i], p) * np.sqrt(1 / p))
            elif i == L-1:
                weights.append(np.random.randn(1, U[i-1]) * np.sqrt(1 / U[i-1]))
            else:
                weights.append(np.random.randn(U[i], U[i-1]) * np.sqrt(1 / U[i-1]))
    return weights

def init_biases(U, L, seed=0):
    np.random.seed(seed)
    biases = []
    if L == 1:
        biases.append(np.zeros(1))
    else:
        for i in range(L):
            biases.append(np.zeros(U[i]))
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
        # Hidden layers
        if i != L-1:
            q = np.dot(q, weights[i].T) + biases[i]
            q = relu(q)
            activations.append(q)
        # Output layer
        else:
            q = np.dot(q, weights[i].T) + biases[i]
            q = identity(q)
            activations.append(q)
    
    return q, activations

def clip_gradients(gradients, threshold=0.001):
    for i in range(len(gradients)):
        gradients[i] = np.clip(gradients[i], -threshold, threshold)
    return gradients

def backpropogation(y, y_hat, activations, weights, biases, L, x, learning_rate=0.001, regularisation_lambda=0.1, regularisation=2):
    layer_errors = []

    # Calculate neuron error for output layer (neuron error = activation(1 - activation)(y - y_hat))
    # output_errors = activations[-1] * (1 - activations[-1]) * (y - y_hat)   # Sigmoid derivative
    output_errors = y - y_hat                                             # Identity derivative
    layer_errors.append(output_errors)

    # Calculate neuron error for hidden layers (neuron error = activation(1 - activation)(sum(forward weight from neuron * error forward neuron)))
    errors = output_errors

    for i in range(L-2, -1, -1):
        # hidden_errors = activations[i] * (1 - activations[i]) * np.dot(errors, weights[i+1])    # Sigmoid derivative
        hidden_errors = relu_derivative(activations[i]) * np.dot(errors, weights[i+1])        # ReLU derivative
        errors = hidden_errors
        layer_errors.append(hidden_errors)

    layer_errors = layer_errors[::-1]

    # Calculate weight change for each layer (delta = learning rate * forward neuron error * activation of previous neuron)
    deltas = [np.zeros_like(w) for w in weights]
    bias_deltas = [np.zeros_like(b) for b in biases]

    for i in range(L):
        if i == 0:
            delta = learning_rate * np.dot(layer_errors[i].T, x)
            bias = learning_rate * np.sum(layer_errors[i], axis=0)
        else:
            delta = learning_rate * np.dot(layer_errors[i].T, activations[i-1])
            bias = learning_rate * np.sum(layer_errors[i], axis=0)

        deltas[i] += delta
        bias_deltas[i] += bias

    # Clip gradients
    deltas = clip_gradients(deltas)
    bias_deltas = clip_gradients(bias_deltas)

    # Update weights
    for i in range(L):
        weights[i] += deltas[i]
        if regularisation == 1:
            weights[i] += regularisation_lambda * np.sign(weights[i])  # L1 regularization
        elif regularisation == 2:
            weights[i] += regularisation_lambda * weights[i]    # L2 Regularisation
        
        biases[i] += bias_deltas[i]
        
    return weights, biases