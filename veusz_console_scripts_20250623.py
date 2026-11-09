# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------

Created on 2025-06-23

Template of small console helpers used while building the AutoPlot tools
(switch statement, file selection, Cartesian to polar conversion).

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216


# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.3: select_file works; history newest-first.
Date: 2026-11-05
# %%%%% Function Descriptions
        select_file: used tkinter names (filedialog, file_entry, tk) that
            were never imported or defined, so any call raised NameError.
            It now opens a tkinter file dialog (imported when called) and
            returns the chosen paths; an optional Entry widget is filled
            with the first path.
        main: unchanged (commented outline only).
# %%%% 0.0.2: NaN
Date:
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing
# %%%% 0.0.1: Script to run in consol description
Date:
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing
=============================================================================
"""

# event loop QApplication.exec_()
# %% Import all required modules
# %%% GUI Module Imports
# %%%% QtPy
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
# from astropy import coordinates as astroCoord
# from astropy.cosmology import WMAP7
# from astropy.table import Table as astroTable
# from astropy.wcs import WCS
# =============================================================================
# %%% Math Modules
# =============================================================================
# import pandas as pd
# import xarray as xr
# =============================================================================
# %%% System Interface Modules
# import time as time
# import subprocess
# %%% Plotting Environment
# %%% File type Export/Import
# =============================================================================
# import h5py as h5
# from scipy.io import savemat
# =============================================================================
# %%% Debug and Console Display
# =============================================================================


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


def select_file(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
                multiple=True, entry=None):
    """
    Ask for one or more files with a tkinter dialog.

    Parameters
    ----------
    filetypes : sequence of (label, pattern)
        Dialog filters.
    multiple : bool
        Allow several files (default True).
    entry : tkinter.Entry, optional
        When given, its text is replaced with the first selected path.

    Returns
    -------
    list of str
        Selected paths (empty when cancelled).
    """
    import tkinter as tk
    from tkinter import filedialog
    owner = None
    if tk._default_root is None:          # no Tk window yet: a hidden one
        owner = tk.Tk()
        owner.withdraw()
    try:
        if multiple:
            paths = list(filedialog.askopenfilenames(filetypes=list(filetypes)))
        else:
            path = filedialog.askopenfilename(filetypes=list(filetypes))
            paths = [path] if path else []
    finally:
        if owner is not None:
            owner.destroy()
    if entry is not None and paths:
        entry.delete(0, tk.END)
        entry.insert(0, paths[0])
    return paths


# %% Main Execution


def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.

    Returns
    -------
    r : float
        Radial coordinate.
    theta : float
        Angular coordinate in radians.
    """
    import math
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


# TODO: Check out PyAntenna


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


if __name__ == "__main__":
    main()
