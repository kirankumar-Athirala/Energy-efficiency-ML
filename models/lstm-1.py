import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import backend as K

# Definition of the CustomCallback
class CustomCallback(Callback):
    def __init__(self, file_path):
        super(CustomCallback, self).__init__()
        self.file_path = file_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Initialize the file if it does not exist
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                f.write("Epoch, Loss, Val Loss\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        with open(self.file_path, 'a') as f:
            f.write(f"{epoch + 1}, {loss}, {val_loss}\n")
        print(f"CustomCallback: Epoch {epoch + 1} completed. Loss: {loss}, Val Loss: {val_loss}")

           
def load_data():
    y = []
    x = []

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(title=f"Select csv data files")
    for file_path in file_paths:
        signal = np.array(np.genfromtxt(file_path, delimiter=','))
        x.append(signal[:, :-1])
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


def train_model_and_save(model, x_train, y_train, x_val, y_val, epochs, batch_size, model_save_path, initial_epoch=0):
    checkpoint_path = model_save_path.replace('.hs', '_checkpoint.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    current_working_dir = os.getcwd()
    metadata_file_path = os.path.join(current_working_dir, 'trained-models','training_metadata.txt')
    
    custom_callback = CustomCallback(file_path=metadata_file_path)  # Custom callback instance

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint, custom_callback],
        initial_epoch=initial_epoch
    )
    model.save(model_save_path)
    return history

def evaluate_model_with_saved_model(model_path, x_test, y_test):
    # Load the saved model
    model = load_model(model_path)
    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # F1 Score
    f1 = f1_score(y_test, y_pred_binary)
    print(f'Test F1 Score: {f1}')
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy}")
    test_acc = (y_pred_binary == y_test).sum().item() / len(y_test)
    return y_pred_binary, test_acc

def plot_confusion_matrix(save_folder, y_true, y_pred):
    cm_image_path = os.path.join(save_folder, 'lstm_confusion_matrix.png')
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Plot and save confusion matrix as PNG
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('LSTM Confusion Matrix')
    plt.savefig(cm_image_path)
    plt.close()
    print(f'Confusion Matrix image saved at: {cm_image_path}')
def get_initial_epoch():
    try:
        with open('training_metadata.txt', 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            initial_epoch = int(last_line.split(',')[0].split(': ')[1])
            return initial_epoch
    except FileNotFoundError:
        return 0
    

if __name__ == '__main__':
    x_data, y_data = load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data, y_data)
    input_shape = (x_train.shape[1], x_train.shape[2])
    hidden_size = 128

    current_working_dir = os.getcwd()
    model_folder = os.path.join(current_working_dir, 'trained-models')
    model_save_path = os.path.join(model_folder, 'LSTM_saved_model.hs')
    checkpoint_path = model_save_path.replace('.hs', '_checkpoint.h5')

    if os.path.exists(checkpoint_path):
        print("Loading model from checkpoint.")
        model = load_model(checkpoint_path)
    else:
        print("Building new model.")
        model = build_model(input_shape, hidden_size)
    
    model.summary()

    epochs = 30
    batch_size = 128

    save_folder = os.path.join(current_working_dir, 'images')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
            
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    initial_epoch = get_initial_epoch()

    history = train_model_and_save(model, x_train, y_train, x_val, y_val, epochs, batch_size, model_save_path, initial_epoch=initial_epoch)
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    y_pred_binary, test_acc = evaluate_model_with_saved_model(model_save_path, x_test, y_test)
    print('\nTest accuracy:', test_acc)
    plot_confusion_matrix(save_folder, y_test, y_pred_binary)
