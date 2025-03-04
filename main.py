import os
import sys
import numpy as np
from tqdm import tqdm
from src.neural_network import nn_matrix, init_weights, init_biases, backpropogation
from src.helper import mean_squared_error, mean_absolute_error
from src.utils import load_data, clean_data, split_data, save_model, load_model, analyse_error, analyse_prediction

def main(data_path, model_path):
    # Load data
    data = load_data(data_path)
    spectral_data, dry_matter = clean_data(data)
    train_spectral_data, test_spectral_data, train_dry_matter, test_dry_matter = split_data(spectral_data, dry_matter)

    # Hyperparameters
    learning_rate = 0.0000000001                # Learning rate
    epochs = 150                                # Number of epochs
    regularisation_lambda = 0.0000000001        # Regularisation parameter
    seed = 4                                    # Seed for random number generator
    L = 4                                       # Number of layers
    U = [100, 50, 10, 1]                        # Shape of neural network U includes the input layer and output neuron
    print_level = 50                            # Print progress every print_level epochs

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
        prediction, activations = nn_matrix(train_spectral_data, L, weights, biases)

        # Backpropogation
        weights, biases = backpropogation(train_dry_matter, prediction, activations, weights, biases, L, train_spectral_data, learning_rate, regularisation_lambda)

        # Calculate error
        squared_error_value = mean_squared_error(train_dry_matter, prediction)
        absolute_error_value = mean_absolute_error(train_dry_matter, prediction)

        squared_error_values.append(squared_error_value)
        absolute_error_values.append(absolute_error_value)

        # Every 100 epochs, print the progress
        if epoch % print_level == 0:
            print(f"Epoch {epoch}: Mean Squared Error = {squared_error_value}, Mean Absolute Error = {absolute_error_value}")
            print(f"Prediction: {prediction[0]}, Actual: {dry_matter[0]}")

    # Save model
    save_model(weights, biases, model_path)

    # Print prediction and actual values with test data
    prediction, _ = nn_matrix(test_spectral_data, L, weights, biases)
    for i in range(3):
        print(f"Prediction: {prediction[i]}, Actual: {test_dry_matter[i]}")

    squared_error_value = mean_squared_error(test_dry_matter, prediction)
    absolute_error_value = mean_absolute_error(test_dry_matter, prediction)

    print(f"Mean Squared error: {squared_error_value}")
    print(f"Mean Absolute error: {absolute_error_value}")

    # Run error analysis
    analyse_error(squared_error_values, absolute_error_values)

    # Run prediction analysis
    analyse_prediction(prediction, test_dry_matter)

if __name__ == "__main__":
    file_name = sys.argv[1]
    data_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    model_path = os.path.join(os.path.dirname(__file__), "model")

    main(data_path, model_path)