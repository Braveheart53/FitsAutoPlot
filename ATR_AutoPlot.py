# -*- coding: utf-8 -*-
# %% Header
# =============================================================================
# Author: William W. Wallace
#
#
# TODO
# update the ATR script to plot multiple files at once, and remove it from here
# and plaec it is own file to import / call
# atr NOT working, need to correct immediately!!
#
# TODO
# Update CSV to auto plot multiple files, extract it to its own file
# =============================================================================

# event loop QApplication.exec_()
# %% Import all required modules
# %%% GUI Uses
# %%%% Tkinter
import tkinter as tk
from tkinter.filedialog import askopenfilenames
from tkinter import filedialog
from tkinter import messagebox
# %%%% Py Qt imports
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
# from PyQt6.QtCore import Qt, QSize
# %%%% PySide6 Imports
# from PySide6.QtGui import *
# from PySide6.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
# from PySide6.QtCore import Qt, QSize
# %%%% qtpy imports
from qtpy.QtGui import *
from qtpy.QtWidgets import (QApplication, QPushButton, QMainWindow, QLabel,
                            QWidget)
from qtpy.QtCore import Qt, QSize

# %%% Math Modules
# import pandas as pd
import xarray as xr
import numpy as np
import skrf as rf
import mpmath as mpm
import scipy.constants as cons
from astropy.time import Time as tme
# look up how to use data classes, this is how one can create a matlab type
# structure, in addition to my own codes for such
import dataclasses as dataclass

# %%% Unit Conversion
# import pint as pint
# ureg = pint.UnitRegistry()

# %%% System Interface Modules
import os
import os.path as pathCheck
import time as time
import sys
# add something to the python sys path
# sys.path.append(os.path.abspath("something"))


# %%% Plotting Environment
import veusz.embed as vz
import pprint

# %%% File type Export/Import
import h5py as h5
from scipy.io import savemat
from fastest_ascii_import import fastest_file_parser as fparser

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

# %% Add to Python Sysytem Path for calling in scripts
# add_subdirs_to_path("somedir")

# %% Locally Global constants
c_0 = cons.speed_of_light  # in meters
pi = mpm.pi  # using mpmath versus numpy, there is a reason for this, just
# not sure what that reason is at the moment

# %% Class Definitions
# Begin Class definitions based upon use cases for range and data


class PlotATR:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ATR Plotter")

        # File selection
        self.file_label = tk.Label(self.root, text="Select ATR file:")
        self.file_label.pack()
        self.file_entry = tk.Entry(self.root, width=150)
        self.file_entry.pack()
        self.file_button = tk.Button(
            self.root, text="Browse",
            command=PlotATR._select_atr_files(self)
        )
        self.file_button.pack()

        # Plot name input
        self.plot_name_label = tk.Label(self.root, text="Enter plot name:")
        self.plot_name_label.pack()
        self.plot_name_entry = tk.Entry(self.root, width=150)
        self.plot_name_entry.config(state='normal')
        self.plot_name_entry.pack()

        # Dataset name input
        self.dataset_name_label = tk.Label(
            self.root, text="Enter dataset name:")
        self.dataset_name_label.pack()
        self.dataset_name_entry = tk.Entry(self.root, width=150)
        self.dataset_name_entry.config(state='normal')
        self.dataset_name_entry.pack()

        # Create plot button
        create_button = tk.Button(
            self.root, text="Create Plot", command=self.create_plot
        )
        create_button.pack()

        # Status label
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()

        # File Info
        self.fileParts = None
        self.fileNames = None

        # Veusz Object
        self.doc = vz.Embedded('GBO ATR Autoplotter', hidden=False)
        self.doc.EnableToolbar()

        # Plot Info

        # requireds @dataclass defintion
        # self.plotInfo = plotDescInfo(
        #     xAxis_label='Frequency (Hz)',
        #     yAxis_label='Uncalibrated (dBm)',
        #     graph_notes=None,
        #     graph_title='Title',
        #     base_name=None,
        #     first_plot=True
        # )

# =============================================================================
#     def _select_atr_file(self):
#         """
#         Select A single ATR file for processing
#
#         Returns
#         -------
#         file_path : TYPE
#             DESCRIPTION.
#         file_entry : TYPE
#             DESCRIPTION.
#
#         """
#         file_path = filedialog.askopenfilename(
#             filetypes=[("GBO Outdoor Test Range files", "*.atr")])
#         if file_path:
#             self.file_entry.delete(0, tk.END)
#             self.file_entry.insert(0, file_path)
#         # return file_path, file_entry
# =============================================================================

    def _select_atr_files(self):

        # file selection dialog, need to update for any files other than current
        # outdoor range files
        # import os
        # Create a root window
        root2 = tk.Tk()

        # Hide the root window
        root2.withdraw()
        # root.iconify()

        self.filenames = askopenfilenames(filetypes=[("Antenna Range Files",
                                                      ".ATR")])
        root2.destroy()
        # start the main loop for processing the selected files
        self.fileParts = [None] * len(self.filenmames)
        for mainLooper in range(len(self.filenames)):
            # this loop processes the files selected one at a time, while combining
            # the data as it progresses

            # get the file parts for further use of the current file.
            self.fileParts[mainLooper] = os.path.split(
                self.filenames[mainLooper]
            )

        # After the mainloop, I need to combine all the data into a multi-dimensional
        # array. Then call Veusz and parse the data into that gui.
        if fileParts[0][0]:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, self.filenames)

        return filenames, fileParts

    def create_plot(self):
        for index in enumerate(range(len(self.filenames))):
            breakpoint

            file_path = self.filenames[index]
            if self.plot_name_entry.get():
                plot_name = self.plot_name_entry.get()
            else:
                plot_name = self.fileParts[index][1]

            if self.dataset_name_entry.get():
                dataset_name = self.dataset_name_entry.get()
            else:
                dataset_name = self.fileParts[index][1]

            if not file_path or not plot_name or not dataset_name:
                # self.status_label.config(text="Please fill in all fields")
                self.status_label.config(
                    text="Blank Fields Will Be Autogenerated."
                )

            if not file_path:
                # self.status_label.config(text="Please fill in all fields")
                self.status_label.config(
                    text="User MUST select at least one file."
                )
                return

            try:
                # Read ATR file
                line_number = 13  # remeber it is zero indexed
                header_info, selected_data = (
                    PlotATR.process_data(file_path,
                                         line_number, self))

                # take the numpy and call it df
                df = selected_data

                # partse header info for future use, remember if the ATR
                # format changes this MUST be changed
                header_lines = header_info.splitlines()
                # freqs are ion MHz in the ATR file
                freq_max = float(header_lines[8].split(":")[-1].strip())
                freq_min = float(header_lines[9].split(":")[-1].strip())
                freq_step = float(header_lines[10].split(":")[-1].strip())

                if freq_min != freq_max:
                    print("This script is not yet capable of reducing " +
                          "data that consists of multiple frequencies. " +
                          "Contact William Wallace at x2216 and ask him " +
                          "to implement this feature.")
                    return
                else:
                    # once plotting of multiple files at once is made, this
                    # array will have to index with the various data sets
                    # by length
                    freq_array = [freq_max]

                Polarization_plane = float(
                    header_lines[5].split(":")[-1].split()[0])  # in deg

                # As of 2025-02-21 The outdoor range is only capable of single
                # elevation scans (at 0 el)
                Az_min = float(header_lines[6].split(":")[-1].strip())
                Az_max = float(header_lines[7].split(":")[-1].strip())
                # All steps in az for the GBO outdoor range are by default
                # 1 degrees in a continual scan
                Az_angles = np.arange(Az_min, Az_max + 1, 1, dtype=float)

                # Create Veusz embedded window
                # embed = vz.Embedded('Veusz', hidden=False)
                # self.doc.EnableToolbar()

                # Create a new document
                # doc = self.doc.Root.AddDocument()

                # Create a new page
                page = self.doc.Root.Add('page', name=dataset_name)

                # Create a new graph
                graph = page.Add('graph', name=dataset_name, autoadd=False)

                # Set plot title
                # incorrect graph.title.val = plot_name

                # First add an dimension to the 2D array in the first axis
                # then parse out the data
                # df = np.expand_dims(df, axis=3)

                # add frequency to the matrix
                data_freq = freq_array
                data_mag = df[:, 0]
                data_phase = df[:, 1]
                data_Az_angle = Az_angles

                # iloc is a pandas function....
                # freq_data = df.iloc[:, 0].tolist()
                # mag_data = df.iloc[:, 1].tolist()
                # phase_data = df.iloc[:, 3].tolist()

                data_col_stack = np.column_stack(
                    (data_Az_angle, data_mag, data_phase)
                )

                # trying xr.dataarray
                data_full_2 = xr.DataArray(data_col_stack,
                                           coords=[data_Az_angle, data_freq],
                                           dims=["Azimuth", "frequnecy"],
                                           attrs=dict(description=(
                                               "Full Dataset for " +
                                               "the associated " +
                                               "outdoor " +
                                               "range measurement.")
                                           )
                                           )

                # Use Xarray to create a dataset
                # a dataset in xarray may not be needed here, a dataarray
                # may be best, but also astropy tables may work better here
                # looking into this, leaving any items in work
                data_full = xr.Dataset(
                    data_vars=dict(magnitude=(["freq", "Az"], data_mag),
                                   phase=(["freq", "Az"], data_phase)
                                   ),
                    coords=dict(
                        frequency=("freq", data_freq),
                        Azimuth=("Az", data_Az_angle)
                    ),
                    attrs=dict(description=(
                        "Full Dataset for the associated " + "outdoor " +
                        "range measurement.")
                    )
                )

                # Create datasets
                self.doc.SetData(dataset_name + '_freq', data_freq)
                self.doc.SetData(dataset_name + '_mag', data_mag)
                self.doc.SetData(dataset_name + '_phase', data_phase)

                # Create xy plot
                xy = graph.Add('xy', name=dataset_name)
                xy.xData.val = dataset_name + '_x'
                xy.yData.val = dataset_name + '_y'
                xy.notes.val = header_lines

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
                # self.doc.Save(saveDir + '/' + dataset_name + '.vsz')
                filenameSave = saveDir + '/' + dataset_name + '.vszh5'
                self.save(self, filenameSave)

                # Show the plot
                self.doc.WaitForClose()

                self.status_label.config(text="Plot created successfully")
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")

    def process_data(file_path, line_number, self):
        """Read the entire file into a list of lines."""
        filesize = os.path.getsize(file_path)

        # Use appropriate method based on file size
        if filesize < 10**7:  # < 10MB
            # Use fast binary reading
            with open(file_path, 'rb') as file:
                content = file.read().decode('ascii')
                lines = content.splitlines()
        else:
            # Use memory-efficient line iteration
            with open(file_path, 'r', encoding='ascii') as file:
                lines = file.readlines()

        # with open(file_path, 'r') as file:
        #     lines = file.readlines()

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
        # selected_data = np.array([[selected_magnitude_data],
        #                           [selected_phase_data]])
        selected_data = np.array([selected_magnitude_data,
                                 selected_phase_data])
        selected_data_transpose = selected_data.transpose()
        return header_info, selected_data_transpose

    def save(self, filename: str):
        """Save Veusz document to specified file."""
        # there might be a precision argument or format string
        # TODO
        # MUST find a way to save higher precision!!!!
        # a work around would be to save all data to np.float64 arrays
        # in files and link them in the veusz document, I would really prefer
        # not do it this way.
        # self.doc.Save(filename, 'vsz')
        filename_root = os.path.splitext(filename)[0]
        filenameHP = filename_root + '.vszh5'
        fileSplit = os.path.split(filename)
        filenameVSZ = (fileSplit[0] + '/Beware_oldVersion/' +
                       os.path.splitext(fileSplit[1])[0] + '_BEWARE.vsz')
        # filename3 = filename_root + '_hdf5.hdf5'
        self.doc.Save(filenameHP, mode='hdf5')
        os.makedirs(fileSplit[0] + '/Beware_oldVersion/', exist_ok=True)
        self.doc.Save(filenameVSZ, mode='vsz')

    def run(self):

        # run the GUI!
        self.root.mainloop()
        # root.destroy()


class Astronomy:
    """
    Used at the moment to create various functions from the AstroPy module.
    """

    def __init__(self):
        pass

    def mjdconvert_time(mjd_time, *args):
        """


        Parameters
        ----------
        mjd_time : TYPE
            Take in 'mjd'

        *args : Varargin
            Various variables or arguments taken for format and scales

        Returns
        -------
        iso_time : String
            return iso time in a string or object

        """

        # =============================================================================
        #         Python 3.8 does not have a built-in switch or case statement
        # like some other programming languages. However, the functionality of
        # a case statement can be achieved through several alternative methods:
        # 1. Dictionary Mapping
        # This approach uses a dictionary to map cases to their corresponding
        # actions.
        # def case_statement(argument):
        #     options = {
        #         'case1': lambda: "Action for case 1",
        #         'case2': lambda: "Action for case 2",
        #         'case3': lambda: "Action for case 3"
        #     }
        #     return options.get(argument, lambda: "Default action")()
        #
        # print(case_statement('case2'))
        # # Expected output: Action for case 2
        #
        # print(case_statement('case4'))
        # # Expected output: Default action
        #
        # 2. If-elif-else Statements
        # This is the most straightforward way to mimic a case statement.
        # def case_statement(argument):
        #     if argument == 'case1':
        #         return "Action for case 1"
        #     elif argument == 'case2':
        #         return "Action for case 2"
        #     elif argument == 'case3':
        #         return "Action for case 3"
        #     else:
        #         return "Default action"
        #
        # 3. Class-based approach
        # For more complex scenarios, a class can be used to handle cases.
        # class CaseStatement:
        #     def case1(self):
        #         return "Action for case 1"
        #
        #     def case2(self):
        #         return "Action for case 2"
        #
        #     def default(self):
        #         return "Default action"
        #
        #     def execute(self, argument):
        #         method_name = 'case' + argument[4:] if argument.startswith('case') else 'default'
        #         method = getattr(self, method_name, self.default)
        #         return method()
        #
        # case_executor = CaseStatement()
        # print(case_executor.execute('case1'))
        # # Expected output: Action for case 1
        #
        # print(case_executor.execute('case4'))
        # # Expected output: Default action
        #
        # Python 3.10 and later: match-case statement
        # It's worth noting that Python 3.10 introduced the match-case
        # statement, which provides a more direct way to implement case-like
        # logic. However, this is not available in Python 3.8.
        # def case_statement(argument):
        #     match argument:
        #         case 'case1':
        #             return "Action for case 1"
        #         case 'case2':
        #             return "Action for case 2"
        #         case 'case3':
        #             return "Action for case 3"
        #         case _:
        #             return "Default action"
        #
        #
        # Generative AI is experimental.
        #
        # =============================================================================

        # --> if MJD not in UTC, can add scale flag: scale='tdb'
        t = tme(mjd_time, format="mjd")
        t_out = []
        if len(args) > 0:
            for arg in args:
                # case statement, as match and case is not introduced until Python
                # 3.10
                # go through the args and create a t_out for each
                # match arg:
                #     case 'iso':
                #         t_out.append(t.iso)
                #     case _:
                #         t_out.append(t.iso)
                #         return t_out
                t_out.append(Astronomy.case_4_mjd_timeFormat(t, t_out, arg))
                print(t_out)
        else:
            t_out.append(Astronomy.case_4_mjd_timeFormat(t, t_out, "iso"))

        # print the returned format(s)
        print(t_out)
        return t_out

    def case_4_mjd_timeFormat(timeObjIn, timeObjOut, optionString):
        """
        Case statement for changing time FORMATs.

        Annother case statement for changing time scales is required.Subformats
        are not yet implemented, but should be as **kwargs for ease of use.

        Parameters
        ----------
        timeObjIn : TYPE
            DESCRIPTION.
        timeObjIn : TYPE
            DESCRIPTION.
        optionString : TYPE
            DESCRIPTION.

        Returns
        -------
        Case option for given input optionString
            DESCRIPTION.

        References
        -------
        https://docs.astropy.org/en/stable/time/index.html
        """
        timeList = []
        options = {
            'iso': lambda: timeList.append(timeObjIn.iso),
            'fits': lambda: timeList.append(timeObjIn.fits),
            'isot': lambda: timeList.append(timeObjIn.isot),
            'gps': lambda: timeList.append(timeObjIn.gps),
            'cxcsec': lambda: timeList.append(timeObjIn.cxcsec),
            'decimalyear': lambda: timeList.append(timeObjIn.decimalyear),
            'jd': lambda: timeList.append(timeObjIn.jd),
            'julian': lambda: timeList.append(timeObjIn.jd),
            'jyear': lambda: timeList.append(timeObjIn.jyear),
            'jyear_str': lambda: timeList.append(timeObjIn.jyear_str),
            'mjd': lambda: timeList.append(timeObjIn.mjd),
            'plot_date': lambda: timeList.append(timeObjIn.plot_date),
            'unix': lambda: timeList.append(timeObjIn.unix),
            'unix_tai': lambda: timeList.append(timeObjIn.unix_tai),
            'yday': lambda: timeList.append(timeObjIn.yday),
            'ymdhms': lambda: timeList.append(timeObjIn.ymdhms),
            'yyymmddHHmmss': lambda: timeList.append(timeObjIn.ymdhms),
            'datetime64': lambda: timeList.append(timeObjIn.datetime64)
        }
        # options.setdefault('iso')
        # print(options.get(optionString,
        # lambda: timeObjOut.append(timeObjIn.iso))())
        timeObjOut = options.get(optionString,
                                 lambda: timeList.append(timeObjIn.iso))()
        return timeList


class CSVFun:
    """
    Used for processing CSV Data
    """

    def __init__(self):
        pass

    def select_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)


def add_subdirs_to_path(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        sys.path.append(dirpath)


def main():
    """
    execution of main function.
    """
    # =============================================================================
    # The main script is below. First we import the files through a dialog. Then
    # utilize the classes and their functions to process the data.
    # =============================================================================

    # Just showing use of the mjd conversion function.
    mjd = 60686.3539
    newTime = Astronomy.mjdconvert_time(mjd, "iso", "ymdhms", "fits")
    # newTime = Astronomy.mjdconvert_time(mjd)
    print(newTime)

    ATR_Example = PlotATR()
    ATR_Example.run()

    # sys.path.append(
    #     os.path.abspath("C:\\Users\\wwallace\\OneDrive - National Radio Astronomy Observatory" +
    #                     "\\Documents\\Python_Vault")
    # )


if __name__ == "__main__":
    main()
