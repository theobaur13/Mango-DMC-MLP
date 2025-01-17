import os
import pandas as pd
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), "data", "NAnderson2020MendeleyMangoNIRData.csv")

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
    prediction = single_layer_nn_loop(spectral_data)
    for i in range(10):
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

    p = data.shape[1]
    bias = np.random.rand()
    weights = []

    for i in range(p):
        weights.append(np.random.rand())

    # Calculate prediction
    prediction = []
    for i in range(data.shape[0]):
        inner = 0
        for j in range(p):
            inner += weights[j] * data[i][j]
        inner += bias
        prediction.append(sigmoid(inner))

    return prediction

if __name__ == "__main__":
    main()