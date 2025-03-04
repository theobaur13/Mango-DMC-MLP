import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def clean_data(data):
    # Drop columns that are not needed
    columns_to_drop = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp"]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=columns_to_drop)

    spectral_data = data.drop(columns=["DM"]).to_numpy()
    dry_matter = data["DM"].to_numpy()
    dry_matter = dry_matter.reshape(-1, 1)
    
    # Remove columns with all zeros
    print(f"Columns before removing all zeros: {spectral_data.shape[1]}")
    spectral_data = spectral_data[:, ~np.all(spectral_data == 0, axis=0)]
    print(f"Columns after removing all zeros: {spectral_data.shape[1]}")

    # Remove rows that contain more than x zeros and their associated dry matter value
    print(f"Rows before removing zeros: {spectral_data.shape[0]}")
    zeros = np.sum(spectral_data == 0, axis=1)
    mask = zeros <= 10
    spectral_data = spectral_data[mask]
    dry_matter = dry_matter[mask]
    print(f"Rows after removing zeros: {spectral_data.shape[0]}")

    # Standardise the spectral data to be between 0 and 1
    spectral_data = (spectral_data + 1) / 2
    
    return spectral_data, dry_matter

def split_data(spectral_data, dry_matter, train_size=0.8):
    # Split data into training and testing sets
    split_index = int(spectral_data.shape[0] * train_size)
    train_spectral_data = spectral_data[:split_index]
    test_spectral_data = spectral_data[split_index:]
    train_dry_matter = dry_matter[:split_index]
    test_dry_matter = dry_matter[split_index:]
    
    return train_spectral_data, test_spectral_data, train_dry_matter, test_dry_matter

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

def analyse_prediction(prediction, actual):
    # Plot the prediction and actual values
    plt.plot(prediction, label="Prediction")
    plt.plot(actual, label="Actual")
    plt.legend()
    plt.show()