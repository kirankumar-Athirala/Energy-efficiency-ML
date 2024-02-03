import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras.layers import BatchNormalization

def load_data():
    y = []
    x = []

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(title=f"Select csv data files")
    for file_path in file_paths:
        signal = np.array(np.genfromtxt(file_path, delimiter=','))
        x.append(signal)
        y.extend([1 if label == 1 else 0 for label in signal[:, -1]])

    root.destroy()

    x_data = np.vstack(x)
    y_data = np.vstack(y).reshape(-1, 1)

    sequence_length = 1
    x_data = x_data.reshape(-1, sequence_length, x_data.shape[1])

    return x_data, y_data

def split_data(x_data, y_data):
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test_temp, y_test_temp, test_size=0.2, random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test

def build_model(input_shape, hidden_size):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=input_shape, return_sequences=True, bias_regularizer=L1L2(0.01, 0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
    return history

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # F1 Score
    f1 = f1_score(y_test, y_pred_binary)
    print(f'Test F1 Score: {f1}')
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy}")
    test_acc = (y_pred_binary == y_test).sum().item() / len(y_test)
    return y_pred_binary, test_acc

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Plot and save confusion matrix as PNG
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'/Users/kirankumarathirala/Documents/Energy-Efficiency/code/images/confusion_matrix_lstm.png')
    
    

if __name__ == '__main__':
    x_data, y_data = load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data)
    
    input_shape = (x_train.shape[1], x_train.shape[2])

    hidden_size = 128
    
    model = build_model(input_shape, hidden_size)
    print("Model Summary:")
    model.summary()
    
    epochs = 30
    batch_size = hidden_size
    
    history = train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size)
    
    y_pred_binary, test_acc = evaluate_model(model, x_test, y_test)
    print('\nTest accuracy:', test_acc)
    
    plot_confusion_matrix(y_test, y_pred_binary)