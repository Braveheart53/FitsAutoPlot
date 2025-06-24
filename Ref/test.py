# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------

Created on %(date)s

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216


# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Script to run in consol description
Date: 
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing

# %%%%% Variable Descriptions
    Define all utilized variables
        file_path: path(s) to selected files for processing

# %%%%% More Info

# %%%% 0.0.2: NaN
Date: 
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing
    More Info:
# %%%%% Variable Descriptions
    Define all utilized variables
        file_path: path(s) to selected files for processing
# %%%%% More Info
=============================================================================
"""

# event loop QApplication.exec_()
import plots_csvfile_inVeusz as pltCSV
# %% Import all required modules
# %%% GUI Module Imports
# %%%% QtPy
from qtpy.QtGui import *
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt, QSize
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
# =============================================================================
# import tkinter as tk
# from tkinter import filedialog
# from tkinter.filedialog import askopenfilenames
# =============================================================================
# %%% Astronomy Modules
# =============================================================================
from astropy.io import fits as pyfits
from astropy.table import QTable as astroQTable
from astropy import units as astroU
# from astropy import coordinates as astroCoord
# from astropy.cosmology import WMAP7
# from astropy.table import Table as astroTable
# from astropy.wcs import WCS
# =============================================================================
# %%% Math Modules
# =============================================================================
# import pandas as pd
# import xarray as xr
import numpy as np
import skrf as rf
# =============================================================================
# %%% System Interface Modules
import os
# import time as time
import sys
# import subprocess
# %%% Plotting Environment
import veusz.embed as vz
import pydoc
# %%% File type Export/Import
# =============================================================================
# import h5py as h5
# from scipy.io import savemat
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


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

# %% Main Execution


def main():
    """
    execution of main function
    """
    pltCSV()
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


if __name__ == "__main__":
    main()
