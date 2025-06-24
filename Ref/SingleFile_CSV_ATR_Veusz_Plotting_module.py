# -*- coding: utf-8 -*-
# %% Header Info
"""
=============================================================================
# Header Info
--------

Created on %(date)s

@author: %(username)s

# Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

0.0.1: Script to run in consol description
    Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing
    More Info:

0.0.2: NaN

# Variable Descriptions
Define all utilized variables
    file_path: path(s) to selected files for processing

=============================================================================
"""

# event loop QApplication.exec_()
# %% Import all required modules
# %%% GUI Uses
import tkinter as tk
from tkinter import filedialog
# Py Qt imports
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
# from PyQt6.QtCore import Qt, QSize
# PySide6 Imports
# from PySide6.QtGui import *
# from PySide6.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
# from PySide6.QtCore import Qt, QSize
# qtpy imports
from qtpy.QtGui import *
from qtpy.QtWidgets import (QApplication, QPushButton, QMainWindow, QLabel,
                            QWidget)
from qtpy.QtCore import Qt, QSize

# %%% Math Modules
# import pandas as pd
import xarray as xr
import numpy as np
import skrf as rf

# %%% System Interface Modules
import os
import time as time
import sys

# %%% Plotting Environment
import veusz.embed as vz
import pprint

# %%% File type Export/Import
# import h5py as h5
# from scipy.io import savemat

# %%% Parallelization
# from multiprocessing import Pool  # udpate when you learn it!
# from multiprocessing import Process
# from multiprocess import Process
# from multiprocess import Pool

# %%% GPU Implementation
# %%%% Cuda
# import cupy as cp
# from cuda import cuda, nvrtc  # need to learn this as well, see class below
# %%%% Rocm Hips
# from hip import hip
# from hip import hiprtc

# %% Functions and Classes
# %%% First Class or Function list
# =============================================================================
# class GPUFun:
#     """
#     GPU functions for use of Cuda
#     See: https://nvidia.github.io/cuda-python/overview.html
#
#     It does take setup of the kernal
#     """
#     def _cudaGetErrorEnum(error):
#         if isinstance(error, cuda.CUresult):
#             err, name = cuda.cuGetErrorName(error)
#             return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
#         elif isinstance(error, nvrtc.nvrtcResult):
#             return nvrtc.nvrtcGetErrorString(error)[1]
#         else:
#             raise RuntimeError('Unknown error type: {}'.format(error))
#
#     def checkCudaErrors(result):
#         if result[0].value:
#             raise RuntimeError("CUDA error code={}({})".format(
#                 result[0].value, _cudaGetErrorEnum(result[0])))
#         if len(result) == 1:
#             return None
#         elif len(result) == 2:
#             return result[1]
#         else:
#             return result[1:]
#
#     def hip_check(call_result):
#         err = call_result[0]
#         result = call_result[1:]
#         if len(result) == 1:
#             result = result[0]
#         if isinstance(err, hip.hipError_t) and err !=
#         hip.hipError_t.hipSuccess:
#             raise RuntimeError(str(err))
#         return result
# =============================================================================

# %%% Second Class or Function List


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

# %% # Main Function


def main():
    """
    execution of main function
    """
    # =============================================================================
    # Description
    # =============================================================================

    # file selection dialog, need to update for any files other than current
    # outdoor range files
    # =============================================================================
    #     filename = filedialog.askopenfilenames(filetypes=
    #                                               [("Antenna Range Files",
    #                                             ".ATR")])
    #
    #     # start the main loop for processing the selected files
    #     for mainLooper in range(len(filename)):
    #         # this loop processes the files selected one at a time,
    #           while combining
    #         # the data as it progresses
    #
    #         # get the file parts for further use of the current file.
    #         fileParts= os.path.split(filename[mainLooper])
    #
    #     # After the mainloop, I need to combine all the data into a
    #       multi-dimensional
    #     # array. Then call Veusz and parse the data into that gui.
    # =============================================================================

# =============================================================================
#     props = hip.hipDeviceProp_t()
#     hip_check(hip.hipGetDeviceProperties(props, 0))
#
#     for attrib in sorted(props.PROPERTIES()):
#         print(f"props.{attrib}={getattr(props,attrib)}")
#     print("ok")
#
# =============================================================================

class PlotATR:
    
    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)
    
    
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
        selected_phase_data = data_numbers[::2]
        selected_magnitude_data = data_numbers[1::2]
        selected_data = np.array([[selected_magnitude_data],
                                  [selected_phase_data]])
        selected_data = np.array([selected_magnitude_data,
                                  selected_phase_data])
    
        return header_info, selected_data
    
    
    def create_plot():
        file_path = file_entry.get()
        plot_name = plot_name_entry.get()
        dataset_name = dataset_name_entry.get()
    
        if not file_path or not plot_name or not dataset_name:
            status_label.config(text="Please fill in all fields")
            return
    
        try:
            # Read ATR file
            line_number = 13  # remeber it is zero indexed
            header_info, selected_data = process_data(file_path, line_number)
    
            # take the numpy array and create a pandas array
            df = selected_data
    
            # Create Veusz embedded window
            embed = vz.Embedded('Veusz')
            embed.EnableToolbar()
    
            # Create a new document
            # doc = embed.Root.AddDocument()
    
            # Create a new page
            page = embed.Root.Add('page')
    
            # Create a new graph
            graph = page.Add('graph', name='graph1', autoadd=False)
    
            # Set plot title
            # incorrect graph.title.val = plot_name
    
            # Assume the first column is freq and the second column is mag
            freq_data = df.iloc[:, 0].tolist()
            mag_data = df.iloc[:, 1].tolist()
            phase_data = df.iloc[:, 3].tolist()
    
            # Create datasets
            embed.SetData(dataset_name + '_freq', freq_data)
            embed.SetData(dataset_name + '_mag', mag_data)
            embed.SetData(dataset_name + '_phase', phase_data)
    
            # Create xy plot
            xy = graph.Add('xy', name=dataset_name)
            xy.xData.val = dataset_name + '_x'
            xy.yData.val = dataset_name + '_y'
    
            # Create a new Axis
            # note that in xy.xAxis, this can be changed to match the give name
            x_axis = graph.Add('axis', name='frequency')
            y_axis = graph.Add('axis', name='counts')
            x_axis.label.val = 'frequency (MHz)'
            y_axis.label.val = 'Counts'
            y_axis.direction.val = 'vertical'
            # x_axis.childnames gives you all the settable parameters
            x_axis.autoRange.val = 'Auto'
            xy.xAxis.val = 'frequency'
            xy.yAxis.val = 'counts'
            xy.marker.val = 'none'
            xy.PlotLine.color.val = 'red'
            xy.PlotLine.width.val = '1pt'
    
            # Add Page or Graph Title
    
            # Save The Veusz file
            saveDir = os.path.dirname(file_path)
            embed.Save(saveDir + '/' + dataset_name + '.vsz')
    
            # Show the plot
            embed.WaitForClose()
    
            status_label.config(text="Plot created successfully")
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}")


    def PlotATRMain():
        """The execution of main function."""
        # =============================================================================
        # The main script is below. First we import the files through a dialog. Then
        # utilize the classes and their functions to process and plot the data
        # utilizing Veusz.
        # =============================================================================
        # Create main window
        root = tk.Tk()
        root.title("ATR Plotter")
    
        # File selection
        file_label = tk.Label(root, text="Select ATR file:")
        file_label.pack()
        file_entry = tk.Entry(root, width=50)
        file_entry.pack()
        file_button = tk.Button(root, text="Browse", command=select_file)
        file_button.pack()
    
        # Plot name input
        plot_name_label = tk.Label(root, text="Enter plot name:")
        plot_name_label.pack()
        plot_name_entry = tk.Entry(root, width=50)
        plot_name_entry.pack()
    
        # Dataset name input
        dataset_name_label = tk.Label(root, text="Enter dataset name:")
        dataset_name_label.pack()
        dataset_name_entry = tk.Entry(root, width=50)
        dataset_name_entry.pack()
    
        # Create plot button
        create_button = tk.Button(root, text="Create Plot", command=create_plot)
        create_button.pack()
    
        # Status label
        status_label = tk.Label(root, text="")
        status_label.pack()
    
        # may need to remove this
        root.mainloop()
        # =============================================================================
        #     # file selection dialog, need to update for any files other than current
        #     # outdoor range files
        #     filename = askopenfilenames(filetypes=[("Antenna Range Files",
        #                                             ".ATR")])
        #
        #     # start the main loop for processing the selected files
        #     for mainLooper in range(len(filename)):
        #         # this loop processes the files selected one at a time, while combining
        #         # the data as it progresses
        #
        #         # get the file parts for further use of the current file.
        #         fileParts = os.path.split(filename[mainLooper])
        #
        #     # After the mainloop, I need to combine all the data into a multi-dimensional
        #     # array. Then call Veusz and parse the data into that gui.
        # =============================================================================

class PlotCSV:
        
    
    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)
    
    
    def create_plot():
        file_path = file_entry.get()
        plot_name = plot_name_entry.get()
        dataset_name = dataset_name_entry.get()
    
        if not file_path or not plot_name or not dataset_name:
            status_label.config(text="Please fill in all fields")
            return
    
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
    
            # Create Veusz embedded window
            embed = vz.Embedded('Veusz')
            embed.EnableToolbar()
    
            # Create a new document
            # doc = embed.Root.AddDocument()
    
            # Create a new page
            page = embed.Root.Add('page')
    
            # Create a new graph
            graph = page.Add('graph', name='graph1', autoadd=False)
    
            # Set plot title
            # incorrect graph.title.val = plot_name
    
            # Assume the first column is x and the second column is y
            x_data = df.iloc[:, 0].tolist()
            y_data = df.iloc[:, 1].tolist()
    
            # Create datasets
            embed.SetData(dataset_name + '_x', x_data)
            embed.SetData(dataset_name + '_y', y_data)
    
            # Create xy plot
            xy = graph.Add('xy', name=dataset_name)
            xy.xData.val = dataset_name + '_x'
            xy.yData.val = dataset_name + '_y'
    
            # Create a new Axis
            # note that in xy.xAxis, this can be changed to match the give name
            x_axis = graph.Add('axis', name='frequency')
            y_axis = graph.Add('axis', name='counts')
            x_axis.label.val = 'frequency (MHz)'
            y_axis.label.val = 'Counts'
            y_axis.direction.val = 'vertical'
            # x_axis.childnames gives you all the settable parameters
            x_axis.autoRange.val = 'Auto'
            xy.xAxis.val = 'frequency'
            xy.yAxis.val = 'counts'
            xy.marker.val = 'none'
            xy.PlotLine.color.val = 'red'
            xy.PlotLine.width.val = '1pt'
    
            # Add Page or Graph Title
    
            # Save The Veusz file
            saveDir = os.path.dirname(file_path)
            embed.Save(saveDir + '/' + dataset_name + '.vsz')
    
            # Show the plot
            embed.WaitForClose()
    
            status_label.config(text="Plot created successfully")
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}")
    
    
    # Create main window
    root = tk.Tk()
    root.title("CSV Plotter")
    
    # File selection
    file_label = tk.Label(root, text="Select CSV file:")
    file_label.pack()
    file_entry = tk.Entry(root, width=50)
    file_entry.pack()
    file_button = tk.Button(root, text="Browse", command=select_file)
    file_button.pack()
    
    # Plot name input
    plot_name_label = tk.Label(root, text="Enter plot name:")
    plot_name_label.pack()
    plot_name_entry = tk.Entry(root, width=50)
    plot_name_entry.pack()
    
    # Dataset name input
    dataset_name_label = tk.Label(root, text="Enter dataset name:")
    dataset_name_label.pack()
    dataset_name_entry = tk.Entry(root, width=50)
    dataset_name_entry.pack()
    
    # Create plot button
    create_button = tk.Button(root, text="Create Plot", command=create_plot)
    create_button.pack()
    
    # Status label
    status_label = tk.Label(root, text="")
    status_label.pack()

root.mainloop()

# %%% If _name_ statement
if __name__ == "__main__":
    main()
