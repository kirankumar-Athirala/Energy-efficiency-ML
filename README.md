# Energy-efficiency-ML
# LSTM Binary Classification Model

This repository contains code for building, training, and evaluating an LSTM model for binary classification tasks using TensorFlow and Keras.

## Overview

The LSTM model is built using TensorFlow and Keras libraries. It is designed to perform binary classification tasks based on time-series data.

### Features

- **Custom Callback**: The code includes a custom Keras callback to log training metadata such as loss and validation loss after each epoch.

- **Data Loading**: Data is loaded from CSV files selected by the user using a file dialog.

- **Model Architecture**: The model architecture consists of LSTM layers followed by BatchNormalization, Dropout, and Dense layers.

- **Training**: The model is trained using a training set and validated using a validation set. Training progress and validation metrics are logged.

- **Evaluation**: The trained model is evaluated using a separate test set, and metrics such as accuracy and F1 score are calculated.

- **Confusion Matrix**: The repository includes functionality to plot and save the confusion matrix.

## Usage

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- tkinter
- matplotlib
- scikit-learn

### Installation

1. Clone the repository:
    https://github.com/kirankumar-Athirala/Energy-efficiency-ML.git
2. Install dependencies
   pip install -r requirements.txt

### Running the Code

1. Navigate to the repository directory:
   Energy-efficiency-ML/models
2. Run the main script:
   python lstm-1.py
3. Follow the prompts to select CSV data files and observe the model training and evaluation process.

## License

This project is licensed under the MIT License.
