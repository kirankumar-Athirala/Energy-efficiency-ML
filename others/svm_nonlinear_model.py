import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_data():
    """
    Load data from CSV files using a file dialog.

    Returns:
    numpy.ndarray: Loaded data.
    """
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(title="Select data files", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    if not file_paths:
        print("No files selected. Exiting.")
        raise SystemExit

    try:
        data = np.vstack([np.genfromtxt(file_path, delimiter=',') for file_path in file_paths])
    except Exception as e:
        print(f"Error loading data: {e}")
        raise SystemExit

    root.destroy()
    return data

def preprocess_data(data):
    """
    Preprocess data by splitting it into training and testing sets,
    scaling features, and converting to PyTorch tensors.

    Args:
    data (numpy.ndarray): Raw data.

    Returns:
    tuple: Tuple containing X_train, X_test, y_train, y_test, and raw y_test.
    """
    labels = data[:, -1]
    features = data[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_test

def plot_confusion_matrix(conf_matrix, y_test, accuracy, f1):
    """
    Plot and save the confusion matrix.

    Args:
    conf_matrix (numpy.ndarray): Confusion matrix.
    y_test (numpy.ndarray): True labels.
    accuracy (float): Model accuracy.
    f1 (float): Model F1 score.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    # Use current working directory and append 'images' folder
    current_working_dir = os.getcwd()
    save_path = os.path.join(current_working_dir, 'images')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save Confusion Matrix plot as .png
    conf_matrix_filename = os.path.join(save_path, f'svn_nonlinear_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(conf_matrix_filename)
    print(f"Confusion Matrix plot saved to {conf_matrix_filename}")

class NonLinearSVM(nn.Module):
    """
    Non-linear Support Vector Machine model.

    Args:
    input_size (int): Number of input features.
    """
    def __init__(self, input_size):
        super(NonLinearSVM, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SVMTrainer:
    """
    Trainer for the Non-linear Support Vector Machine model.

    Args:
    model (torch.nn.Module): The SVM model.
    criterion (torch.nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimization algorithm.
    """
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, X_train_tensor, y_train_tensor, num_epochs=500, learning_rate=0.01, weight_decay=0.001):
        """
        Train the SVM model.

        Args:
        X_train_tensor (torch.Tensor): Training features.
        y_train_tensor (torch.Tensor): Training labels.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization strength.

        Returns:
        torch.nn.Module: Trained SVM model.
        """
        self.model.to(device)
        criterion = self.criterion()
        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

        return self.model

def evaluate_model(svm_model, X_test_tensor, y_test):
    """
    Evaluate the SVM model and print metrics.

    Args:
    svm_model (torch.nn.Module): Trained SVM model.
    X_test_tensor (torch.Tensor): Testing features.
    y_test (numpy.ndarray): True labels.
    """
    with torch.no_grad():
        y_pred_tensor = svm_model(X_test_tensor)
        y_pred = (y_pred_tensor >= 0).cpu().numpy().astype(int).squeeze()

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    plot_confusion_matrix(conf_matrix, y_test, accuracy, f1)

def save_model(svm_model, model_filename='best_svm_model.pth'):
    """
    Save the trained SVM model.

    Args:
    svm_model (torch.nn.Module): Trained SVM model.
    model_filename (str): Filename for the saved model.
    """
    torch.save(svm_model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    data = load_data()
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_test = preprocess_data(data)

    svm_model = NonLinearSVM(input_size=X_train_tensor.shape[1])
    svm_trainer = SVMTrainer(svm_model, nn.BCEWithLogitsLoss, optim.Adam)
    trained_svm_model = svm_trainer.train(X_train_tensor, y_train_tensor)

    evaluate_model(trained_svm_model, X_test_tensor, y_test)
    save_model(trained_svm_model)
