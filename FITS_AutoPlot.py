# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------

Created on 2025-05-30

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216


# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Script to run in consol description
Date: 2025-05-30
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing
# %%%%% Variable Descriptions
    Define all utilized variables
        file_path: path(s) to selected files for processing
# %%%%% More Info
    See https://fits.gsfc.nasa.gov/fits_samples.html for sampel fits files
# %%%% 0.0.2: Previous version put all the plots on a single page.
Date: 2025-05-30
# %%%%% Function Descriptions
        main: main script body
        select_file: utilzing module os, select multiple files for processing
    More Info: Previous Version put all plots on a single page, this version
    will create a page per plot
# %%%%% Variable Descriptions
    Define all utilized variables
        file_path: path(s) to selected files for processing
# %%%%% More Info
    This puts all plots on their own page and grid. Colorbars are not working
    and will need to be corrected.
    Also need to find out how to apply color maps to the contour or otherwise

    Also need to figure out how to plot multi-dimensional data and correctly
    associate the X-axis in each plot.

    Have not yet test astropy import method. This shall be done in 0.0.3
=============================================================================
"""

# event loop QApplication.exec_()
# %% Import all required modules
# %%% GUI Module Imports
# %%%% QtPy
from qtpy.QtGui import *
# from qtpy.QtWidgets import (
#     QApplication,
#     QLabel,
#     QLineEdit,
#     QMainWindow,
#     QVBoxLayout,
#     QWidget,
# )
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
# =============================================================================
# import tkinter as tk
# from tkinter import filedialog
# from tkinter.filedialog import askopenfilenames
# =============================================================================
# %%% Astronomy Modules
# =============================================================================
from astropy.io import fits
from astropy.table import QTable
# from astropy import units as astroU
# from astropy import coordinates as astroCoord
# from astropy.cosmology import WMAP7
# from astropy.table import Table as astroTable
# from astropy.wcs import WCS
# import PyFITS
# =============================================================================
# %%% Math Modules
# =============================================================================
# import pandas as pd
# import xarray as xr
import numpy as np
# import skrf as rf
# =============================================================================
# %%% System Interface Modules
import os
# import time as time
from datetime import date
import sys
import subprocess
import pdir
# %%% Console Interaction Improvement
import pdir
from rich import inspect as richinspect
from rich import pretty
import inspect
from pprint import pprint
# %%% Plotting Environment
import veusz.embed as embed
# %%% File type Export/Import
# =============================================================================
# import h5py as h5
# from scipy.io import savemat
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

# %% Internal Functional Variables
createLinkedFits = True
createArbitraryXaxis = True

# %% Class and Function Definitons


class switch:
    """Creates a case or switch style statement."""

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
        """Interate and find the match."""
        yield self.match

    def match(self, *args):
        return self.value in args


class qtGUI:
    """Handles all Qt-based user interactions."""

    def __init__(self):
        self.app = QApplication(sys.argv)

    def get_fits_file_and_method(self):
        """Display dialog for FITS file selection and import method choice."""
        dialog = QDialog()
        dialog.setWindowTitle("Open FITS File and Select Import Method")
        layout = QVBoxLayout()

        # File selection components
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_btn = QPushButton("Select FITS File")
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)

        # Import method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Import Method:")
        self.veusz_radio = QRadioButton("Veusz Native")
        self.astropy_radio = QRadioButton("AstroPy")
        self.veusz_radio.setChecked(True)
        method_group = QButtonGroup(dialog)
        method_group.addButton(self.veusz_radio)
        method_group.addButton(self.astropy_radio)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.veusz_radio)
        method_layout.addWidget(self.astropy_radio)
        layout.addLayout(method_layout)

        # Dialog buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        dialog.setLayout(layout)
        self.selected_file = None

        # Connect signals
        file_btn.clicked.connect(lambda: self._select_file(dialog))
        ok_btn.clicked.connect(lambda: self._validate_selection(dialog))
        cancel_btn.clicked.connect(dialog.reject)

        result = dialog.exec_()
        if result == QDialog.Accepted:
            method = 'astropy' if self.astropy_radio.isChecked() else 'veusz'
            return self.selected_file, method
        return None, None

    def _select_file(self, dialog):
        """Handle file selection button click."""
        fname, _ = QFileDialog.getOpenFileName(
            dialog, "Open FITS File", "", "FITS Files (*.fits *.fit)"
        )
        if fname:
            self.selected_file = fname
            self.file_label.setText(fname)

    def _validate_selection(self, dialog):
        """Validate file selection before accepting dialog."""
        if not self.selected_file:
            self.file_label.setText("Please select a file!")
            return
        dialog.accept()

    def get_save_filename(self):
        """Display file save dialog for Veusz project."""
        return QFileDialog.getSaveFileName(
            None, "Save Veusz Project", "",
            "Veusz Files (*.vsz)")[0]

    def ask_open_veusz(self):
        """Display dialog to open created file in Veusz GUI."""
        msg = QMessageBox()
        msg.setWindowTitle("Open in Veusz")
        msg.setText("Would you like to open the file in Veusz?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return msg.exec_() == QMessageBox.Yes


class VZPlotFITS:
    """Manages Veusz document creation and FITS data visualization."""

    def __init__(self):
        self.doc = embed.Embedded('FITS Visualizer')
        # self.page = self.doc.Root.Add('page')
        # self.grid = self.page.Add('grid', columns=2)
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)

    def import_via_veusz(self, filename: str):
        """Import FITS data using Veusz native importer."""
        base_name = os.path.splitext(os.path.basename(filename))[0]

        # Import all HDUs and columns from FITS file
        self.doc.ImportFileFITS(
            filename=filename,
            items=['/'],  # Import all items
            namemap={},
            linked=createLinkedFits
        )

        # Tag datasets with source filename
        for ds in self.doc.GetDatasets():
            # original_label = self.doc.GetDataLabel(ds) or ds
            original_label = ds
            # self.doc.SetDataLabel(ds, f"{original_label} [{base_name}]")
            self.doc.TagDatasets(base_name, [original_label])

    def import_via_astropy(self, filename: str):
        """Import FITS data using Astropy with enhanced metadata handling."""
        """"Testing this function to begin in version 0.0.3"""
        base_name = os.path.splitext(os.path.basename(filename))[0]

        with fits.open(filename) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is None:
                    continue

                if isinstance(hdu, fits.BinTableHDU):
                    self._import_astropy_table(hdu, base_name)
                else:
                    self._import_astropy_image(hdu, i, base_name)

    def _import_astropy_table(self, hdu, base_name):
        """Process table HDUs using QTable."""
        qt = QTable.read(hdu)
        for colname in qt.colnames:
            description = hdu.header.get(f'TCOMM{colname}', colname)
            self.doc.SetData(colname, qt[colname])
            # self.doc.SetDataLabel(colname, f"{description} [{base_name}]")
            self.doc.TagDatasets(base_name, [original_label])

    def _import_astropy_image(self, hdu, index, base_name):
        """Process image HDUs with filename tagging."""
        header = hdu.header
        ds_name = header.get('EXTNAME', f'hdu{index}_data')
        label = header.get('EXTNAME', f'HDU {index} Data')

        self.doc.SetData(ds_name, hdu.data)
        # self.doc.SetDataLabel(ds_name, f"{label} [{base_name}]")
        self.doc.TagDatasets(base_name, [ds_name])

    def create_plots(self):
        """Generate automated plots based on dataset dimensions."""
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

    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)

    def _plot_2d(self, dataset: str):
        """Create contour plot for 2D datasets."""
        # to create a second plot view of the same project
        secondView = self.doc.StartSecondView('2D Dataset View of: ' +
                                              dataset)
        secondView.EnableToolbar(enable=True)
        # secondView.To(self.page.path)
        secondView.MoveToPage(self.doc.GetDatasets().index(self.page.name) + 1)
        graph = self.grid.Add('graph')
        # see if a move up can be done with Root page\graph\colorbar1\move up
        # must associated the colorbar with the widget of the image or contour
        colorbar = graph.Add('colorbar', direction='vertical',
                             name='colorbar1', widgetName='imageIn')
        contour = graph.Add('contour')
        contour.data.val = dataset
        image = graph.Add('image', name='imageIn',
                          data=dataset, colorMap='plasma')

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
            if createArbitraryXaxis:
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

    def _plot_nd(self, dataset: str):
        """Create line plot for nD datasets with red initial trace."""
        try:
            graph = self.grid.Add('graph')
            # the following is from _plot_1d... adjust as needed for nd
# =============================================================================
#             xy = graph.Add('xy')
#             xy.yData.val = dataset
#             # if 'date' in dataset:
#             #     # get the dataset from Veusz and change it to a datetime list
#             #     # need to figure out exactly the time string format utilized
#             #     # self.doc.SetDataDateTime, likely will overwrite or need to
#             #     # use this earlier in the script
#             #     pass
#             xy.marker.val = 'none'
#             xy.PlotLine.color.val = 'red'
#             if createArbitraryXaxis:
#                 # xy.xData.val = set x axis data
#                 currentYDataTuple = self.doc.GetData(xy.yData.val)
#                 if isinstance(currentYDataTuple[0], np.ndarray):
#                     if currentYDataTuple[0].ndim == 1:
#                         length = currentYDataTuple[0].size
#                         xy.xData.val = range(length)
#                     else:
#                         QMessageBox.critical(
#                             None,
#                             "Length of Data Array is non-integer",
#                             "This should not happen. \n"
#                             "Contact the Author and inform him of this error."
#                         )
#                         return
#                 else:
#                     QMessageBox.warning(
#                         None,
#                         "The Data array is non-numeric \n",
#                         "X axis value not defined."
#                     )
#
#             if self.first_1d:
#                 self.first_1d = False
# =============================================================================

        except Exception as e:
            QMessageBox.critical(
                None,
                "nD Plotting Error",
                f"Failed to Plot nD Data Set: {str(e)}"
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


# %% Main Func
if __name__ == '__main__':
    gui = qtGUI()
    fits_path, import_method = gui.get_fits_file_and_method()

    if not fits_path or not import_method:
        sys.exit()

    vz = VZPlotFITS()
    getattr(vz, f'import_via_{import_method}')(fits_path)
    vz.create_plots()

    if save_path := gui.get_save_filename():
        vz.save(save_path)
        if gui.ask_open_veusz():
            VZPlotFITS.open_veusz_gui(save_path)

    sys.exit(gui.app.exec_())
