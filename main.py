import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.neural_network import nn_matrix, init_weights, init_biases, backpropogation
from src.helper import mean_squared_error, mean_absolute_error

file_name = "test.csv"
# file_name = "NAnderson2020MendeleyMangoNIRData.csv"
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
    
    # Standardise the spectral data to be between 0 and 1
    spectral_data = (spectral_data + 1) / 2

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

def load_model(path, file_prefix="model"):
    # Load weights and biases using pickle
    with open(f"{path}/{file_prefix}_weights.pkl", "rb") as f:
        weights = pickle.load(f)
    
    with open(f"{path}/{file_prefix}_biases.pkl", "rb") as f:
        biases = pickle.load(f)
    
    return weights, biases

def analyse_error(squared_error, absolute_error):
    # Remove the first value from the error arrays as it is always very large
    squared_error = squared_error[1:]
    absolute_error = absolute_error[1:]

    # Plot squared error and absolute error on the same graph but with different y-axis
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Squared Error', color=color)
    ax1.plot(squared_error, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Absolute Error', color=color)
    ax2.plot(absolute_error, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def main():
    # Load data
    data = load_data(data_path)
    spectral_data, dry_matter = clean_data(data)

    # Hyperparameters
    learning_rate = 0.001                    # Learning rate
    epochs = 50                                  # Number of epochs
    regularisation_lambda = 0.1              # Regularisation parameter
    seed = 1                                    # Seed for random number generator
    L = 3                                       # Number of layers
    U = [5, 2, 1]                                # Shape of neural network U
    print_level = 1                             # Print progress every print_level epochs

    print(f"U: {U}")

    # Initialise error storage
    squared_error_values = []
    absolute_error_values = []

    # Randomly Initialise weights and biases
    np.random.seed(seed)
    weights = init_weights(spectral_data.shape[1], U, L, seed)
    biases = init_biases(U, L, seed)

    # weights, biases = load_model(model_path, file_prefix="optimal")

    # for i in range(L):
    #     weights[i] = weights[i].T

    for epoch in tqdm(range(epochs)):
        # Forward pass
        prediction, activations = nn_matrix(spectral_data, L, weights, biases)

        # Backpropogation
        weights, biases = backpropogation(dry_matter, prediction, activations, weights, biases, L, spectral_data, learning_rate, regularisation_lambda)

        # Calculate error
        squared_error_value = mean_squared_error(dry_matter, prediction)
        absolute_error_value = mean_absolute_error(dry_matter, prediction)

        squared_error_values.append(squared_error_value)
        absolute_error_values.append(absolute_error_value)

        # Every 100 epochs, print the progress
        if epoch % print_level == 0:
            print(f"Epoch {epoch}: Mean Squared Error = {squared_error_value}, Mean Absolute Error = {absolute_error_value}")

            # # Print prediction and actual values
            # if prediction is not None:
            #     for i in range(3):
            #         print(f"Prediction: {prediction[i]}, Actual: {dry_matter[i]}")
    
    # Print weights and biases
    print(f"Weights: {weights}")
    print(f"Biases: {biases}")

    # Save model
    save_model(weights, biases, model_path)

    # Print prediction and actual values
    if prediction is not None:
        for i in range(3):
            print(f"Prediction: {prediction[i]}, Actual: {dry_matter[i]}")

    squared_error_value = mean_squared_error(dry_matter, prediction)
    absolute_error_value = mean_absolute_error(dry_matter, prediction)

    print(f"Mean Squared error: {squared_error_value}")
    print(f"Mean Absolute error: {absolute_error_value}")

    # Run error analysis
    analyse_error(squared_error_values, absolute_error_values)

if __name__ == "__main__":
    main()