import os
import numpy as np
from tqdm import tqdm
from src.neural_network import nn_matrix, init_weights, init_biases, backpropogation
from src.helper import mean_squared_error, mean_absolute_error
from src.utils import load_data, clean_data, save_model, load_model, analyse_error

# file_name = "test.csv"
file_name = "NAnderson2020MendeleyMangoNIRData.csv"
data_path = os.path.join(os.path.dirname(__file__), "data", file_name)
model_path = os.path.join(os.path.dirname(__file__), "model")

def main():
    # Load data
    data = load_data(data_path)
    spectral_data, dry_matter = clean_data(data)

    # Hyperparameters
    learning_rate = 0.001                       # Learning rate
    epochs = 50                                 # Number of epochs
    regularisation_lambda = 0.1                 # Regularisation parameter
    seed = 1                                    # Seed for random number generator
    L = 3                                       # Number of layers
    U = [306, 5, 1]                               # Shape of neural network U includes the input layer and output neuron
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