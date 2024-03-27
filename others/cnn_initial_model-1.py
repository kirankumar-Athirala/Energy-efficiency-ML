import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

device = "mps" if torch.backends.mps.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self, input_size_after_conv):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * input_size_after_conv, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

def prepare_data():
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

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, optimizer, criterion, scheduler, batch_size, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X.unsqueeze(1))
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor.unsqueeze(1))
            val_predictions = (val_outputs > 0.5).float()

        val_accuracy = (val_predictions == y_test_tensor.unsqueeze(1)).float().mean().item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')

        scheduler.step(val_accuracy)

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.unsqueeze(1))
        test_predictions = (test_outputs > 0.5).float()

    test_accuracy = (test_predictions == y_test_tensor.unsqueeze(1)).float().mean().item()
    print(f'Test Accuracy: {test_accuracy}')

    # Confusion Matrix
    cm = confusion_matrix(y_test_tensor.cpu().numpy(), test_predictions.cpu().numpy())
    print("Confusion Matrix:")
    print(cm)

    # F1 Score
    f1 = f1_score(y_test_tensor.cpu().numpy(), test_predictions.cpu().numpy())
    print(f'F1 Score: {f1}')

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('/Users/kirankumarathirala/Documents/Energy-Efficiency/code/CNN_confusion_matrix_plot.png')
    plt.show()


def main():
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data()

    input_size = X_train_tensor.size(-1)
    kernel_size = 3
    padding = 0
    stride = 1
    conv_output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    pool_output_size = conv_output_size // 2
    input_size_after_conv = pool_output_size

    model = Net(input_size_after_conv).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    batch_size = 64

    train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, optimizer, criterion, scheduler, batch_size)

    evaluate_model(model, X_test_tensor, y_test_tensor)

if __name__ == "__main__":
    main()