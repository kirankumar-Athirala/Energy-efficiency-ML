#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:40:17 2023

@author: kirankumarathirala
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:43:21 2023

@author: kirankumarathirala
"""
import tkinter as tk
from tkinter import filedialog
import csv
import pandas as pd

def convert_to_csv(input_file_path):
    df = pd.read_csv(input_file_path, sep='\t', header=None)
    df_subset = df.iloc[:, 16:]
    output_file_path = input_file_path.replace('.txt', '.csv')
    df_subset.to_csv(output_file_path, index=False, header=False)
    print(f'Conversion completed. CSV file saved as {output_file_path}')

def browse_files():
    file_paths = filedialog.askopenfilenames(title='Select TXT Files', filetypes=[('Text Files', '*.txt')])
    for file_path in file_paths:
        convert_to_csv(file_path)

# Create the main Tkinter window
root = tk.Tk()
root.title('TXT to CSV Converter')

# Create a button to browse and select files
browse_button = tk.Button(root, text='Browse Files', command=browse_files)
browse_button.pack(pady=20)


# Start the Tkinter event loop
root.mainloop()
