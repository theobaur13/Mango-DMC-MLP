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
    p = data.shape[1]                           # Number of features set to number of spectroscopy columns
    x = np.transpose(data)                      # Initialise x matrix as column vector

    # Initialise weights and bias
    np.random.seed(seed)
    weights_input = np.random.rand(p, U)        # Randomly initialise input layer weight matrix of shape    W(1) = (p, U)  
    bias_input = np.random.rand(U)              # Randomly initialise bias vector of shape                  b(1) =(U,)
    weights_hidden = np.random.rand(U)          # Randomly initialise hidden layer weight vector of shape   W(2) = (U,)
    bias_hidden = np.random.rand()              # Randomly initialise hidden layer bias                     b(2)

    # q = h(W(1) * x + b(1))
    q = sigmoid(np.dot(np.transpose(weights_input), x) + bias_input)

    # prediction = W(2) * q + b(2)
    prediction = np.dot(np.transpose(weights_hidden), q) + bias_hidden

    return prediction