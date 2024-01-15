#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:13:33 2023

@author: kirankumarathirala
"""

import csv
import tkinter as tk
from tkinter import filedialog
import os

def process_csv(input_file_path):
    # Determine the label based on the input file name
    if 'without_person' in input_file_path:
        label = '0'
    elif 'with_person' in input_file_path:
        label = '1'
    else:
        print(f"Warning: File name doesn't contain 'without_person' or 'with_person'. Defaulting to label '0'.")
        label = '0'

    # Generate output file path based on the input file path
    output_file_path = os.path.splitext(input_file_path)[0] + '_labeled.csv'

    with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        for row in csv_reader:
            row.append(label)
            csv_writer.writerow(row)

    print(f'Values added to the last column in {output_file_path} with label {label}')

def select_files():
    # Ask the user to select an input file
    input_file_paths = filedialog.askopenfilenames(title="Select Input CSV File", filetypes=[("CSV files", "*.csv")])
    if not input_file_paths:
        return  # User canceled file selection
    for input_file_path in input_file_paths:
        # Process the CSV file
        process_csv(input_file_path)

# Create a Tkinter window
root = tk.Tk()
root.title("CSV File Processor")

# Create a button to trigger file selection and processing
process_button = tk.Button(root, text="Select CSV File and Process", command=select_files)
process_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
