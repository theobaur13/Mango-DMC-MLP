import numpy as np
from src.helper import sigmoid, relu

# prediction = h(W1 * x1 + W2 * x2 + ... + Wp * xp + b)
# h = activation function
# W = weights
# x = input
# b = bias
# p = number of features
def single_layer_nn_matrix(data, seed=0):
    # Initialize weights and bias
    np.random.seed(seed)
    weights = np.random.rand(data.shape[1])
    bias = np.random.rand()

    # Calculate prediction
    inner = np.dot(data, weights) + bias
    prediction = sigmoid(inner)

    return prediction

def two_layer_nn_matrix(data, seed=0, U=3):
    # Calculate variables for input layer
    p = data.shape[1]                           # Number of features set to number of spectroscopy columns
    x = np.transpose(data)                      # Initialise x matrix as column vector

    # Initialise weights and bias
    np.random.seed(seed)
    weights_input = np.random.rand(p, U)        # Randomly initialise input layer weight matrix of shape    W(1) = (p, U)  
    bias_input = np.random.rand(U)              # Randomly initialise bias vector of shape                  b(1) =(U,)
    weights_hidden = np.random.rand(U)          # Randomly initialise hidden layer weight vector of shape   W(2) = (U,)
    bias_hidden = np.random.rand()              # Randomly initialise hidden layer bias                     b(2)

    # q = h(W(1) * x + b(1))
    print(np.dot(np.transpose(weights_input), x))
    print(bias_input)
    q = sigmoid(np.dot(np.transpose(weights_input), x) + bias_input)

    # prediction = W(2) * q + b(2)
    prediction = np.dot(np.transpose(weights_hidden), q) + bias_hidden

    return prediction

def deep_nn_matrix(data, rand_U=True, seed=0, L=4, U_limit=10):
    # Calculate variables for input layer
    p = data.shape[1]                           # Number of features set to number of spectroscopy columns
    x = data                                    # Initialise x matrix as column vector

    np.random.seed(seed)    
    if rand_U:
        U = np.random.randint(1, U_limit-1, L-1)    # Randomly initialise U in shape [U1, ..., UL-1] if rand_U is True
        U = np.append(U, 1)                         # Add 1 to the end of U to match the output layer

    # Initialise weights and bias
    # Randomly initialise weight matrix W. W(1) has dimensions of U1 x p, W(l) has dimensions of Ul x Ul-1. W(L) has dimensions of 1 x UL.
    weights = []
    for i in range(L):
        if i == 0:
            weights.append(np.random.rand(U[i], p))
        elif i == L-1:
            weights.append(np.random.rand(1, U[i-1]))
        else:
            weights.append(np.random.rand(U[i], U[i-1]))

    # Randomly initialise bias vector b. b(1) has dimensions of U1, b(l) has dimensions of Ul. b(L) has dimensions of 1.
    biases = []
    for i in range(L):
        biases.append(np.random.rand(U[i]))

    # q(l) = h(W(l) * q(l-1) + b(l))
    q = x
    for i in range(L):
        q = np.dot(weights[i], q.T).T + biases[i]
        q = sigmoid(q)

    return q