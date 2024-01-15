import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Function for data loading and preprocessing
def load_and_preprocess_data():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(title="Select data files", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    data = np.vstack([np.genfromtxt(file_path, delimiter=',') for file_path in file_paths])

    root.destroy()

    y = data[:, -1]
    X = data[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    input_size = X_train_tensor.size(-1)
    kernel_size = 3
    padding = 0
    stride = 1

    conv_output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    pool_output_size = conv_output_size // 2

    input_size_after_conv = pool_output_size

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_size_after_conv


# Function for evaluating the model with F1 score and confusion matrix
def evaluate_model(model, X_test, y_test):
    X_test = X_test.unsqueeze(1)
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = (test_outputs > 0.5).float().cpu().numpy()

    y_test_numpy = y_test.unsqueeze(1).cpu().numpy()

    # F1 Score
    f1 = f1_score(y_test_numpy, test_predictions)

    # Confusion Matrix
    cm = confusion_matrix(y_test_numpy, test_predictions)

    print(f'Test F1 Score: {f1}')
    print('Confusion Matrix:')
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


# Function for training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(1))
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# Function for creating the model
def create_model(input_size_after_conv):
    class Net(nn.Module):
        def __init__(self, input_size_after_conv):
            super(Net, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
            self.pool = nn.MaxPool1d(2)
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.5)  # Adding dropout layer
            self.fc1 = nn.Linear(32 * input_size_after_conv, 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.flatten(x)
            x = self.dropout(x)  # Applying dropout
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = torch.sigmoid(x)
            return x

    return Net(input_size_after_conv).to(device)


# Main code
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, input_size_after_conv = load_and_preprocess_data()

    model = create_model(input_size_after_conv).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    train_model(model, train_loader, criterion, optimizer, num_epochs=100)

    evaluate_model(model, X_test, y_test)
