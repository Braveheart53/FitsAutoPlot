# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------

Created on 2025-06-17

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216


# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Script to run in consol description
Date: 2025-06-17
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing

# %%%%% Variable Descriptions
    Define all utilized variables
        file_path: path(s) to selected files for processing

# %%%%% More Info
=============================================================================
"""

# event loop QApplication.exec_()
# %% Import all required modules
# %%% GUI Module Imports
# %%%% QtPy
# from qtpy.QtGui import *
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QMessageBox
)
# %%%% PyQt 6 GUI
# =============================================================================
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import (QApplication, QPushButton, QMainWindow, QLabel)
# from PyQt6.QtCore import Qt, QSize
# =============================================================================
# %%%% PySide6
# =============================================================================
# from PySide6.QtWidgets import QApplication, QWidget
# =============================================================================
# %%%% tkinter (essentially a Tcl wrapper)
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
# %%% Astronomy Modules
# =============================================================================
# from astropy.io import fits as pyfits
# from astropy.table import QTable as astroQTable
# from astropy import units as astroU
# from astropy import coordinates as astroCoord
# from astropy.cosmology import WMAP7
# from astropy.table import Table as astroTable
# from astropy.wcs import WCS
# =============================================================================
# %%% Math Modules
# import pandas as pd
# import xarray as xr
import numpy as np
# import skrf as rf
# %%% System Interface Modules
import os
# import time as time
import sys
import subprocess
# %%% Plotting Environment
import veusz.embed as embed
# from FITS_AutoPlot import VZPlot
# import pydoc
# %%% File type Export/Import
# =============================================================================
# import h5py as h5
# from scipy.io import savemat
from fastest_ascii_import import fastest_file_parser as fparser
# =============================================================================
# %%% Debug and Console Display
# =============================================================================
import pdir
from rich import inspect as richinspect
from rich import pretty
import inspect
from pprint import pprint
# =============================================================================
# %%% Parallel Processing Modules
# =============================================================================
# from multiprocessing import Pool  # udpate when you learn it!
# from multiprocessing import Process
# from multiprocess import Pool
# from multiprocess import Process
# =============================================================================
# %%% GPU Acceleration
# =============================================================================
# # Cupy is numpy implementation in CUDA for GPU use, need to learn more
# import cupy
# from cuda import cuda, nvrtc  # need to learn this as well, see class below
# =============================================================================

# %% Class and Function Definitons


class switch:
    """"Creates a case or switch style statement."""
    """
    This is utilized as follows:

        for case in switch('b'):
            if case('a'):
                # print or do whatever one wants
                print("Case A")
                break
            if case('b'):
                print("Case B")  # Output: "Case B"
                break
    """

    def __init__(self, value):
        self.value = value

    def __iter__(self):
        """interate and find the match."""
        yield self.match

    def match(self, *args):
        return self.value in args


class VZPlotRnS:
    """
        Used to create vuesz plots using data fron Rns
    """

    def __init__(self):
        self.doc = embed.Embedded('Rhode & Schwarz SFT File Plotter')
        # self.page = self.doc.Root.Add('page')
        # self.grid = self.page.Add('grid', columns=2)
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)
        # strings to search for and parse data by, section would contain
        # the date time after a ';'
        # then the next line is data
        self.searchData_strings = {
            'version': 'VERSION',
            'type': 'TYPE',
            'mode': 'MODE',
            'center freq': 'CENTER FREQ',
            'freq offset': 'FREQ OFFSET',
            'span': 'SPAN',
            'x-axis': 'X-AXIS',
            'start': 'START',
            'stop': 'START',
            'ref level': 'REF LEVEL',
            'level offset': 'LEVEL OFFSET',
            'ref psotion': 'REF POSITION',
            'y-axis': 'Y-AXIS',
            'level range': 'LEVEL RANGE',
            'rf att': 'RF ATT',
            'rbw': 'RBW',
            'vbw': 'VBW',
            'swt': 'SWT',
            'trace mode': 'TRACE MODE',
            'detector': 'DETECTOR',
            'sweep count': "SWEEP COUNT",
            'trace': 'TRACE',
            'x-unit': 'X-UNIT',
            'y-unit': 'Y-UNIT',
            'preamplifier': 'PREAMPLIFIER',
            'transducer': 'TRANSDUCER',
            'values': 'VALUES',
            'section': 'SECTION'
        }

        # from file examples, there are many blank lines in the header
        # just grab the header data with this as well as the strings
        # but really the string search should work as well, this will
        # grab the entire heading lines
        self.sft_lines = [1, 2, 3] + list(range(5, 58, 2))

    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)

    def create_plots(self):
        """Generate automated plots based on dataset dimensions.
            Only after all data sets are created in veusz"""
        for ds in self.doc.GetDatasets():
            ds_type = self.doc.GetDataType(ds)
            # create a new page for each plot and name it by ds
            self._create_page(ds)

            # use a case or switch statement to plot according to dataset
            # veusz documentation does not address nd dataset
            # if ds == 'data':
            #     print('Found the nd Array Data')

            for case in switch(ds_type):
                if case('1d'):
                    # somehow nd datatype  is not being passed
                    currentDataDims = self.doc.GetData(ds)[0].ndim
                    if currentDataDims == 1:
                        self._plot_1d(ds)
                    else:
                        # This means it is ndimensional data...
                        ds_type = 'nD'
                        self._plot_nd(ds)
                    del currentDataDims
                    break
                if case('2d'):
                    self._plot_2d(ds)
                    break
                if case('text'):
                    pass
                    break
                if case('datetime'):
                    pass
                    break
                if case('nD'):
                    print('An ND case was found in ' + ds)
                    break

    def _plot_1d(self, dataset: str):
        """Create line plot for 1D datasets with red initial trace."""
        try:
            graph = self.grid.Add('graph')
            xy = graph.Add('xy')
            xy.yData.val = dataset
            # if 'date' in dataset:
            #     # get the dataset from Veusz and change it to a datetime list
            #     # need to figure out exactly the time string format utilized
            #     # self.doc.SetDataDateTime, likely will overwrite or need to
            #     # use this earlier in the script
            #     pass
            xy.marker.val = 'none'
            xy.PlotLine.color.val = 'red'

            # make the dataset and the x axis aer of the same length
            if dataset.size == xAxis.size:
                # xy.xData.val = set x axis data
                currentYDataTuple = self.doc.GetData(xy.yData.val)
                if isinstance(currentYDataTuple[0], np.ndarray):
                    if currentYDataTuple[0].ndim == 1:
                        length = currentYDataTuple[0].size
                        xy.xData.val = range(length)
                    else:
                        QMessageBox.critical(
                            None,
                            "Length of Data Array is non-integer",
                            "This should not happen. \n"
                            "Contact the Author and inform him of this error."
                        )
                        return
                else:
                    QMessageBox.warning(
                        None,
                        "The Data array is non-numeric \n",
                        "X axis value not defined."
                    )

            if self.first_1d:
                self.first_1d = False

        except Exception as e:
            QMessageBox.critical(
                None,
                "1D Plotting Error",
                f"Failed to Plot 1D Data Set: {str(e)}"
            )

    def save(self, filename: str):
        """Save Veusz document to specified file."""
        self.doc.Save(filename)

    def open_veusz_gui(filename: str):
        """Launch Veusz GUI with generated project file."""
        if sys.platform.startswith('win'):
            veusz_exe = os.path.join(sys.prefix, 'Scripts', 'veusz.exe')
        else:
            veusz_exe = os.path.join(sys.prefix, 'bin', 'veusz')

        if not os.path.exists(veusz_exe):
            QMessageBox.critical(
                None,
                "Veusz Not Found",
                "Veusz not found in Python environment.\n",
                "Install with: [pip OR conda OR mamba] install veusz"
            )
            return

        try:
            subprocess.Popen([veusz_exe, filename])
        except Exception as e:
            QMessageBox.critical(
                None,
                "Launch Error",
                f"Failed to start Veusz: {str(e)}"
            )

    def RnS_sft_plot(self, fileName: str = None):
        """
        Used to import all data into the Vuesz environment. Then pass to
        create_plots to plot based on data within the veusz environment.

        Parameters
        ----------
        dataIn : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # check is a fileName was passed and if it is a list or tuple, then
        # process accordingly.
        if fileName is None:
            fileName, fileParts = self._select_sft_file()
        elif isinstance(fileName, list) and os.path.splitext(fileName[0])[1] == '.sft':
            fileParts = [None]*len(fileName)
            for mainLooper in range(len(fileName)):
                # this loop processes the files passed one at a time,
                # while combining
                # the data as it progresses

                # get the file parts for further use of the current file.
                fileParts[mainLooper] = os.path.split(fileName[mainLooper])
        elif fileName.isascii() and os.path.splitext(fileName[0])[1] == '.sft':
            fileParts = os.path.split(fileName)
        else:
            # add a prompt gui to user and exit
            QMessageBox.critical(
                None,
                "File Selected or Passed is not the correct file type. \n",
                "Ensure the extension .sft is included and \n",
                "that you are using the ASCII type of that file."
            )
            return

        # parse all the data through a general parser
        if isinstance(fileName, list):
            for currentFile in fileName:
                dataReturned = fparser(currentFile, line_targets=self.sft_lines,
                                       string_patterns=self.searchData_strings)

                # used for data set name and label
                base_name = os.path.splitext(os.path.basename(fileName))[0]

                # now we have all the data in dataReturned of type dict
                # use this and VZPlotRns to create Veusz plots

                for item in dataReturned:
                    dataSetName = 'test'
                    data = [1, 2, 3]
                    description = 'testing  this'
                    # Put all of the header information in the notes of the plot
                    self.doc.SetNote()

                    # Put all data into a dataset with a name and label
                    self.doc.SetData(dataSetName, data)
                    self.doc.SetDataLabel(dataSetName,
                                          f"{description} [{base_name}]")
        else:
            dataReturned = fparser(fileName, line_targets=self.sft_lines,
                                   string_patterns=self.searchData_strings)

            # used for data set name and label
            base_name = os.path.splitext(os.path.basename(fileName))[0]

            # now we have all the data in dataReturned of type dict
            # use this and VZPlotRns to create Veusz plots

            for item in dataReturned:
                dataSetName = 'test'
                data = [1, 2, 3]
                description = 'testing  this'
                # Put all of the header information in the notes of the plot
                self.doc.SetNote()

                # Put all data into a dataset with a name and label
                self.doc.SetData(dataSetName, data)
                self.doc.SetDataLabel(dataSetName,
                                      f"{description} [{base_name}]")

    def _select_sft_file(self):
        """
        GUI using tkiner for sft file selection. Does work with
        multiple files.

        Returns
        -------
        filename : TYPE
            DESCRIPTION.
        fileParts : TYPE
            DESCRIPTION.

        """
        # import os
        filename = askopenfilenames(filetypes=[("R&S SFT Files",
                                                ".SFT")])

        # start the main loop for processing the selected files
        fileParts = [None] * len(filename)
        for mainLooper in range(len(filename)):
            # this loop processes the files selected one at a time, while combining
            # the data as it progresses

            # get the file parts for further use of the current file.
            fileParts[mainLooper] = os.path.split(filename[mainLooper])

        return filename, fileParts

        # if file_path:
        #     file_entry.delete(0, tk.END)
        #     file_entry.insert(0, file_path)


# %% Main Execution
if __name__ == '__main__':
    """
    execution of main function
    """
    vz = VZPlotRnS()
    vz.RnS_sft_plot()
    pass


# if __name__ == "__main__":
#     main()
