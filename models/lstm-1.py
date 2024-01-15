import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split 


refBuffer = 4000

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        
        out = self.dropout1(out)
        out, _ = self.lstm2(out)

        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Function to generate dataset
def genDataSet():
    y = []
    x = []

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(title=f"Select csv data files")
    for file_path in file_paths:
        signal = np.array(np.genfromtxt(file_path, delimiter=','))
        data = np.delete(signal, np.s_[:refBuffer], 1)
        x.append(data)
        y.extend([1 if label == 1 else 0 for label in signal[:, -1]])

    root.destroy()

    x_data = np.vstack(x)
    y_data = np.vstack(y).reshape(-1, 1)

    # Reshape x_data to have three dimensions [batch_size, sequence_length, input_size]
    sequence_length = 1  # You may need to adjust this based on your data
    x_data = x_data.reshape(-1, sequence_length, x_data.shape[1])

    # Convert to PyTorch tensors
    x_data = torch.FloatTensor(x_data)
    y_data = torch.FloatTensor(y_data)

    return x_data, y_data


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

def estimate(model, inputSignal):
    model.eval()
    with torch.no_grad():
        outputs = model(inputSignal)
        y_pred = (outputs > 0.5).float()

    return y_pred

if __name__ == '__main__':
    # Select device based on availability
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    x_data, y_data = genDataSet('Train')
    # Split data into training and testing sets
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Further split the training data into training and validation sets
    x_test, x_val, y_test, y_val = train_test_split(x_test_temp, y_test_temp, test_size=0.2, random_state=42)    


    # Convert data to PyTorch DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    input_size = x_train.shape[-1]
    hidden_size = 128
    output_size = 1
    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

    # Convert test data to PyTorch tensor
    x_test = torch.FloatTensor(x_test).to(device)

    # Make predictions on the test set
    y_pred = estimate(model, x_test)

    # Move predictions to CPU for accuracy calculation
    y_pred = y_pred.cpu()

    test_acc = (y_pred == y_test).sum().item() / len(y_test)

    print('\nTest accuracy:', test_acc)
