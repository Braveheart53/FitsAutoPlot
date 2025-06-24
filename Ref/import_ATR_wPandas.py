# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:57:34 2024

@author: wwallace
William W. Wallace
wwallace@nrao.edu
"""

import pandas as pd

def process_data(file_path, line_number):
    # Read the entire file into a list of lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract header information
    header_lines = lines[:line_number]
    header_info = ''.join(header_lines).strip()
    
    # Extract the specified line of numerical data
    data_line = lines[line_number].strip()
    
    # Remove the last character if it's '#'
    if data_line.endswith('#'):
        data_line = data_line[:-1]
    
    # Convert the line of numerical data into a list of numbers
    data_numbers = list(map(float, data_line.split()))
    
    # Select every other item from the list of numbers
    selected_data = data_numbers[::2]
    
    return header_info, selected_data

# Example usage
file_path = 'data.txt'
line_number = 5  # Specify the line number of the numerical data
header_info, selected_data = process_data(file_path, line_number)

print("Header Information:")
print(header_info)
print("\nSelected Data:")
print(selected_data)
