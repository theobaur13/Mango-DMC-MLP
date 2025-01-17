import os
import pandas as pd
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), "data", "test.csv")

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def clean_data(data):
    # Drop columns that are not needed
    data = data.drop(columns=["Set","Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp"])
    dry_matter = data["DM"].to_numpy()
    spectral_data = data.drop(columns=["DM"]).to_numpy()
    
    dry_matter = dry_matter.reshape(-1, 1)

    return spectral_data, dry_matter
 
def main():
    # Load data
    data = load_data(data_path)
    spectral_data, dry_matter = clean_data(data)

    # Single layer neural network
    prediction = two_layer_nn_loop(spectral_data)
    for i in range(3):
        print(f"Prediction: {prediction[i]}, Actual: {dry_matter[i]}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

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

def single_layer_nn_loop(data, seed=0):
    # Initialize weights and bias
    np.random.seed(seed)

    p = data.shape[1]                                   # Number of features set to number of spectroscopy columns
    bias = np.random.rand()
    weights = []

    for i in range(p):
        weights.append(np.random.rand())                # Randomly initialize weights

    # Calculate prediction
    prediction = []
    for i in range(data.shape[0]):                      # Loop through each row
        inner = 0
        for j in range(p):                              # Loop through each column (feature)
            inner += weights[j] * data[i][j]            # Calculate weight * feature
        inner += bias
        prediction.append(sigmoid(inner))               # Apply sigmoid activation function

    return prediction

def two_layer_nn_loop(data, seed=0, U=3):
    # Initialize weights and bias
    np.random.seed(seed)

    p = data.shape[1]                                   # Number of features set to number of spectroscopy columns
    
    # Initialize weights and bias for input layer
    weights_input = []
    bias_input = []
    for i in range(U):                                  # Loop through each hidden layer neuron
        bias = np.random.rand()                         # Randomly initialize one bias
        
        weights = []
        for j in range(p):
            weights.append(np.random.rand())            # Randomly initialize p weights (one for each column)

        weights_input.append(weights)
        bias_input.append(bias)
    
    # Initialize weights and bias for hidden layer
    weights_hidden = []
    bias_hidden = np.random.rand()

    for i in range(U):
        weights_hidden.append(np.random.rand())         # Randomly initialize U weights (one for each hidden layer neuron)

    prediction = []

    # Calculate input layer predictions
    for i in range(data.shape[0]):                      # Loop through each row
        q = []
        row = data[i]
        for k in range(U):                              # Loop through each hidden layer neuron
            q_prediction = two_layer_nn_loop_input_layer(row, weights_input[k], bias_input[k])
            q.append(q_prediction)

        # Calculate hidden layer prediction
        y = two_layer_nn_loop_hidden_layer(q, weights_hidden, bias_hidden)
        prediction.append(y)

    return prediction

# q_k = h(W_k1 * x_1 + W_k2 * x_2 + ... + W_kp * x_p + b_k)
def two_layer_nn_loop_input_layer(data_row, weights, bias):
    inner = 0
    
    for i in range(len(data_row)):
        inner += weights[i] * data_row[i]
    inner += bias

    prediction = sigmoid(inner)
    return prediction

# y = W_1 * q_1 + W_2 * q_2 + ... + W_U * q_U + b
def two_layer_nn_loop_hidden_layer(q, weights, bias):
    inner = 0

    for i in range(len(q)):
        inner += weights[i] * q[i]
    inner += bias

    prediction = sigmoid(inner)
    return prediction

if __name__ == "__main__":
    main()