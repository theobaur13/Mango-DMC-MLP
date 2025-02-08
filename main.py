import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.neural_network import nn_matrix, init_weights, init_biases, backpropogation
from src.helper import squared_error, absolute_error

file_name = "test.csv"
file_name = "NAnderson2020MendeleyMangoNIRData.csv"
data_path = os.path.join(os.path.dirname(__file__), "data", file_name)
model_path = os.path.join(os.path.dirname(__file__), "model")

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

def save_model(weights, biases, path, file_prefix="model"):
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Save weights and biases using pickle
    with open(f"{path}/{file_prefix}_weights.pkl", "wb") as f:
        pickle.dump(weights, f)
    
    with open(f"{path}/{file_prefix}_biases.pkl", "wb") as f:
        pickle.dump(biases, f)

def load_model(path, file_prefix="model", L=None):
    # Load weights and biases using pickle
    with open(f"{path}/{file_prefix}_weights.pkl", "rb") as f:
        weights = pickle.load(f)
    
    with open(f"{path}/{file_prefix}_biases.pkl", "rb") as f:
        biases = pickle.load(f)
    
    return weights, biases

def analyse_error(squared_error, absolute_error):
    # Plot squared error and absolute error on the same graph
    plt.plot(squared_error, label="Squared Error")
    plt.plot(absolute_error, label="Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def main():
    # Load data
    data = load_data(data_path)
    spectral_data, dry_matter = clean_data(data)

    # Hyperparameters
    learning_rate = 0.00000001                  # Learning rate
    # epochs = 14768                              # Number of epochs = 1000
    epochs = 10
    seed = 1                                    # Seed for random number generator
    L = 5                                       # Number of layers
    # U = [5, 8, 1]                             # Shape of neural network U
    U_limit = 8                                 # Upper limit for number of hidden layer neurons
    print_level = 1                             # Print progress every print_level epochs

    # Create random shape of neural network U
    np.random.seed(seed)
    U = np.random.randint(1, U_limit-1, L-1)    # Randomly initialise U in shape [U1, ..., UL-1] if rand_U is True
    U = np.append(U, 1)                         # Add 1 to the end of U to match the output layer
    print(f"U: {U}")

    # Initialise error storage
    squared_error_values = []
    absolute_error_values = []

    # Initialise weights and biases
    weights = init_weights(spectral_data.shape[1], U, L, seed)
    biases = init_biases(U, L, seed)

    for epoch in tqdm(range(epochs)):
        # Forward pass
        prediction, activations = nn_matrix(spectral_data, U, L, weights, biases, seed)
        
        # Backpropogation
        weights, biases = backpropogation(dry_matter, prediction, activations, weights, biases, L, spectral_data, learning_rate)

        # Every 100 epochs, print the progress
        if epoch % print_level == 0:
            squared_error_value = squared_error(dry_matter, prediction)
            absolute_error_value = absolute_error(dry_matter, prediction)

            squared_error_values.append(squared_error_value)
            absolute_error_values.append(absolute_error_value)
            print(f"Epoch {epoch}: Squared Error = {squared_error_value}, Absolute Error = {absolute_error_value}")
    
    # Save model
    save_model(weights, biases, model_path)

    # Print prediction and actual values
    if prediction is not None:
        for i in range(3):
            print(f"Prediction: {prediction[i]}, Actual: {dry_matter[i]}")

    squared_error_value = squared_error(dry_matter, prediction)
    absolute_error_value = absolute_error(dry_matter, prediction)

    print(f"Squared error: {squared_error_value}")
    print(f"Absolute error: {absolute_error_value}")

    # Run error analysis
    analyse_error(squared_error_values, absolute_error_values)

if __name__ == "__main__":
    main()