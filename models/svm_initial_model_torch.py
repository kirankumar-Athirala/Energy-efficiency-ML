import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog
import tkinter as tk
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_data():
    # Create the root window for the file dialog
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog for selecting multiple files
    file_paths = filedialog.askopenfilenames(title="Select data files", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    # Load data from multiple files
    data = np.vstack([np.genfromtxt(file_path, delimiter=',') for file_path in file_paths])

    # Close the root window
    root.destroy()

    return data

def preprocess_data(data):
    y = data[:, -1]
    X = data[:, :-1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_test

def plot_confusion_matrix(conf_matrix, y_test):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save Confusion Matrix plot as .png
    conf_matrix_filename = '/Users/kirankumarathirala/Documents/Energy-Efficiency/code/SVM_initial_confusion_matrix.png'
    plt.savefig(conf_matrix_filename)
    print(f"Confusion Matrix plot saved to {conf_matrix_filename}")
    
def train_svm_model(X_train_tensor, y_train_tensor):
    # Define a simple SVM model using PyTorch
    class SVMModel(nn.Module):
        def __init__(self, input_size):
            super(SVMModel, self).__init__()
            self.fc = nn.Linear(input_size, 1)

        def forward(self, x):
            return self.fc(x)

    # Create the SVM model and move it to the specified device
    svm_model = SVMModel(input_size=X_train_tensor.shape[1]).to(device)

    # Hyperparameter tuning
    learning_rate = 0.01
    num_epochs = 500
    weight_decay = 0.001

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(svm_model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    # Train the SVM model
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = svm_model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

    return svm_model

def evaluate_model(svm_model, X_test_tensor, y_test):
    with torch.no_grad():
        y_pred_tensor = svm_model(X_test_tensor)
        y_pred = (y_pred_tensor >= 0).cpu().numpy().astype(int).squeeze()

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix, y_test)


def save_model(svm_model):
    # Save the model to a file
    model_filename = 'best_svm_model.pth'
    torch.save(svm_model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    data = load_data()
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_test = preprocess_data(data)
    svm_model = train_svm_model(X_train_tensor, y_train_tensor)
    evaluate_model(svm_model, X_test_tensor, y_test)
    save_model(svm_model)
