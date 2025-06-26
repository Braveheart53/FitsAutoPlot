# -*- coding: utf-8 -*-
# %% Header
# =============================================================================
# Author: William W. Wallace
#
#
# TODO: update the ATR script to plot multiple files at once, and remove it from here
# and plaec it is own file to import / call
# atr NOT working, need to correct immediately!!
#
# TODO: Update CSV to auto plot multiple files, extract it to its own file
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

# %%% Math Modules
# import pandas as pd
import xarray as xr
import numpy as np
# import skrf as rf
import mpmath as mpm
import scipy.constants as cons
from astropy.time import Time as tme
# look up how to use data classes, this is how one can create a matlab type
# structure, in addition to my own codes for such
# import dataclasses as dataclass

# %%% Unit Conversion
# import pint as pint
# ureg = pint.UnitRegistry()

# %%% System Interface Modules
import os
import os.path as pathCheck
# import time as time
import sys
# add something to the python sys path
# sys.path.append(os.path.abspath("something"))
from operator import itemgetter
import subprocess

# %%% Plotting Environment
import veusz.embed as vz
# import pprint

# %%% File type Export/Import
# import h5py as h5
# from scipy.io import savemat
# from fastest_ascii_import import fastest_file_parser as fparser

# %%% Parallelization
# from multiprocessing import Pool  # udpate when you learn it!
# from multiprocessing import Process
# from multiprocess import Process
# from multiprocess import Pool

# %%% Debugging Help
from rich import inspect as richinspect
import pdir
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


class qtGUI:
    """Handles all Qt-based user interactions."""

    def __init__(self):
        self.app = QApplication(sys.argv)

    def closeEvent(self, event):
        QApplication.closeAllWindows()
        event.accept()

    def _select_sft_file(self, dialog):
        """Handle file selection button click."""
        fname, _ = QFileDialog.getOpenFileName(
            dialog, "Open SFT File", "", "R&S SFT Files (*.sft)"
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
            "Veusz High Precision Files (*.vszh5)")[0]

    def ask_open_veusz(self):
        """Display dialog to open created file in Veusz GUI."""
        msg = QMessageBox()
        msg.setWindowTitle("Open in Veusz")
        msg.setText("Would you like to open the file in Veusz?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return msg.exec_() == QMessageBox.Yes


class Wrap4With:
    """Used to add context management to a given object."""

    def __init__(self, resource):
        self._resource = resource

    def __enter__(self):
        """Return the wrapped resource upon entering the context."""
        return self._resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the resource upon exiting the context.

        If the resource has a .close() method, it will be called.
        """
        if hasattr(self._resource, 'close'):
            self._resource.close()
        # Optionally, add other cleanup logic here.
        # Return None to propagate exceptions, or True to suppress them.
        return None


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

        # Close and save
        create_button = tk.Button(
            self.root, text="Close and Save", command=self._save_Veusz
        )
        create_button.pack()

        # Status label
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()

        # File Info
        if not self.fileParts:
            self.fileParts = None

        if not self.filenames:
            self.filenames = None

        # Plot Info
        # if not self.plotTitle:
        self.plotTitle = 'GBO Outdoor Antenna Range Pattern'
        self.freq_label = 'Frequency (MHz)'
        self.az_label = 'Azimuth (degrees)'
        self.phase_label = 'Phase (degrees)'
        self.mag_label = 'Magnitude (dBm)'

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
    def _save_Veusz(self):
        """Save the generated file and ask to open Veusz Interface. """
        self.root.destroy()
        gui = qtGUI()
        if save_path := gui.get_save_filename():
            self.save(save_path)
            if gui.ask_open_veusz():
                self.open_veusz_gui(save_path)

        sys.exit(gui.app.exec_())
        # QApplication.closeAllWindows()
        gui.closeEvent()

    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)
        return self.page

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
        self.fileParts = [None] * len(self.filenames)
        for mainLooper in range(len(self.filenames)):
            # this loop processes the files selected one at a time, while combining
            # the data as it progresses

            # get the file parts for further use of the current file.
            self.fileParts[mainLooper] = os.path.split(
                self.filenames[mainLooper]
            )

        # After the mainloop, I need to combine all the data into a multi-dimensional
        # array. Then call Veusz and parse the data into that gui.
        if self.fileParts[0][0]:
            self.file_entry.delete(0, tk.END)
            filenames_only = list(map(itemgetter(1), self.fileParts))
            self.file_entry.insert(0, ' ; '.join(filenames_only))

        return self.filenames, self.fileParts

    def create_plot(self):
        """Create the 2D plots for all data."""
        data_freq_all = np.empty(len(self.filenames), dtype=object)
        data_mag_all = np.empty(len(self.filenames), dtype=object)
        data_phase_all = np.empty(len(self.filenames), dtype=object)
        data_Az_angle_all = np.empty(len(self.filenames), dtype=object)
        data_Polarization_all = np.array(['H', 'V', str(45), str(-45)])
        measurement_Types = ['Mag', 'Phase']

        for index, file_path in enumerate(self.filenames):

            # file_path = element
            if self.plot_name_entry.get():
                plot_name = self.plot_name_entry.get()
            else:
                plot_name = self.fileParts[index][1]

            if self.dataset_name_entry.get():
                dataset_name = self.dataset_name_entry.get()
            else:
                dataset_name = self.fileParts[index][1]

            if (
                    not self.plot_name_entry.get() or not
                    self.dataset_name_entry.get()
            ):
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
                header_info, selected_data, header_lines = (
                    PlotATR.process_data(file_path,
                                         line_number, self))

                # take the numpy and call it df
                df = selected_data

                # partse header info for future use, remember if the ATR
                # format changes this MUST be changed
                # header_lines = header_info.splitlines()
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

                Polarization_plane = (
                    header_lines[5].split(":")[-1].split()[0]
                )  # string

                # Make and index for placing the data in the correct area
                # for a combined data structure based on the read in
                # polarization
                try:
                    # note it cannot differentiate between H plane (mag field)
                    # and Hpol, horizontal to ground....
                    # Treats E plane and Vpol as the same thing
                    for case in switch(Polarization_plane):
                        if case('H'):
                            # print or do whatever one wants
                            print("H")
                            Polarization_index = 0
                            break
                        if case('V'):
                            print("V")
                            Polarization_index = 1
                            break
                        if case(str(45)):
                            print("45")
                            Polarization_index = 2
                            break
                        if case(str(-45)):
                            print("-45")
                            Polarization_index = 3
                            break
                        if case('E'):
                            # print or do whatever one wants
                            print("E")
                            Polarization_index = 1
                            break
                        raise ValueError('Polarization String is ' +
                                         'incorrect. See file: ' +
                                         dataset_name)

                except ValueError as e:
                    print(f"Error Raised: {e}")
                    break

                # As of 2025-02-21 The outdoor range is only capable of single
                # elevation scans (at 0 el)
                Az_min = float(header_lines[6].split(":")[-1].strip())
                Az_max = float(header_lines[7].split(":")[-1].strip())
                # All steps in az for the GBO outdoor range are by default
                # 1 degrees in a continual scan
                Az_angles = np.arange(Az_min, Az_max + 1, 1, dtype=float)

                # First add an dimension to the 2D array in the first axis
                # then parse out the data
                # df = np.expand_dims(df, axis=3)

                # add frequency to the matrix
                data_freq = freq_array
                data_mag = df[:, 0]
                data_phase = df[:, 1]
                data_Az_angle = Az_angles

                # TODO: Make a data structure that contains all phase, mag
                # and polarization
                # this is a little inefficient as it written over each pass
                #

                # TODO: When upgrading to multi-frequency files
                # this will need to be changed
                data_freq_all[index] = data_freq[0]
                # TODO: Finish the 4D data matrix creation in xarray
                if Polarization_index:
                    # combine all in a 4D matrix such as:
                    # (4D: Az × Freq × Pol × Var)
                    # Row follows Az
                    # Column follows freq
                    # internal item list follows pol
                    #       Mag = ([
                    #                    ---  magnitude at freq and angle             magnitude at freq and angle ---
                    # Az angle 1    [[mag@pol0, mag@pol1, mag@pol2, mag@pol3], [mag@pol0, mag@pol1, mag@pol2, mag@pol3]],
                    # Az angle 2    [[mag@pol0, mag@pol1, mag@pol2, mag@pol3], [mag@pol0, mag@pol1, mag@pol2, mag@pol3]]
                    #               ])
                    pass

                data_mag_all[index] = data_mag
                data_phase_all[index] = data_phase
                data_Az_angle_all[index] = data_Az_angle

                data_col_stack = np.column_stack(
                    (data_Az_angle, data_mag, data_phase)
                )

                # All Data is now parsed into various variables
                # TODO: Combine All Data in 5D array
                # TODO: Save all combined data in a numpy or xarray file
                # for future use.
                # Data array example
# =============================================================================
#                 # Define coordinates
#                 azimuth = np.array([0, 10, 20, 30])          # Azimuth points
#                 frequency = np.array([100, 200, 300])         # Frequency points
#                 polarization = np.array(['H', 'V'])           # Polarization states
#                 variables = ['Mag', 'Phase']                  # Measurement types
#
#                 # Generate synthetic data (4D: Az × Freq × Pol × Var)
#                 mag_data = np.array([
#                     [[1.2, 1.1], [1.5, 1.4], [1.7, 1.6]],   # Az=0°
#                     [[2.3, 2.2], [2.5, 2.4], [2.7, 2.6]],   # Az=10°
#                     [[1.8, 1.7], [2.0, 1.9], [2.2, 2.1]],   # Az=20°
#                     [[3.0, 2.9], [3.2, 3.1], [3.4, 3.3]]    # Az=30°
#                 ])
#
#                 phase_data = np.array([
#                     [[0.1, 0.11], [0.12, 0.13], [0.14, 0.15]],
#                     [[0.2, 0.21], [0.22, 0.23], [0.24, 0.25]],
#                     [[0.15, 0.16], [0.17, 0.18], [0.19, 0.20]],
#                     [[0.3, 0.31], [0.32, 0.33], [0.34, 0.35]]
#                 ])
#
#                 # Combine into single 4D array (Az, Freq, Pol, Var)
#                 combined_data = np.stack([mag_data, phase_data], axis=3)
#
#                 # Create DataArray
#                 da = xr.DataArray(
#                     data=combined_data,
#                     dims=['Az', 'Frequency', 'Polarization', 'Variable'],
#                     coords={
#                         'Az': azimuth,
#                         'Frequency': frequency,
#                         'Polarization': polarization,
#                         'Variable': variables,
#                         'Magnitude': (['Az', 'Frequency', 'Polarization'], mag_data),
#                         'Phase': (['Az', 'Frequency', 'Polarization'], phase_data)
#                     },
#                     name='Antenna_Measurements',
#                     attrs={
#                         'description': 'Full antenna measurements',
#                         'units': 'Mag(dB), Phase(radians)'
#                     }
#                 )
# =============================================================================
# =============================================================================
# import numpy as np
# import xarray as xr
#
# # Define measurement coordinates
# azimuth = np.array([0, 10, 20, 30])          # Azimuth points (degrees)
# elevation = np.array([0, 15, 30])             # Elevation points (degrees)
# frequency = np.array([100, 200, 300])         # Frequency points (MHz)
# polarization = np.array(['H', 'V'])           # Polarization states
# variables = ['Mag', 'Phase']                  # Measurement types
#
# # Initialize empty 5D arrays (Az, El, Freq, Pol)
# mag_data = np.empty((len(azimuth), len(elevation), len(frequency), len(polarization)))
# phase_data = np.empty((len(azimuth), len(elevation), len(frequency), len(polarization)))
#
# # Generate synthetic 5D antenna pattern data
# for i, az in enumerate(azimuth):
#     for j, el in enumerate(elevation):
#         for k, freq in enumerate(frequency):
#             for l, pol in enumerate(polarization):
#                 # Realistic antenna pattern calculations
#                 # Magnitude: combines azimuth, elevation, frequency, and polarization effects
#                 base_mag = 1.0 + 0.5 * (freq/100 - 1)  # Frequency scaling
#                 az_pattern = 1 + 0.1 * np.sin(np.radians(az))  # Azimuth variation
#                 el_pattern = 1 + 0.05 * np.cos(np.radians(el))  # Elevation variation
#                 pol_factor = 1.0 if pol == 'H' else 0.9  # Polarization difference
#
#                 mag_data[i, j, k, l] = base_mag * az_pattern * el_pattern * pol_factor
#
#                 # Phase: realistic phase progression
#                 phase_base = 0.1 + 0.02 * (freq/100 - 1)  # Frequency dependence
#                 phase_az = 0.01 * az  # Azimuth contribution
#                 phase_el = 0.005 * el  # Elevation contribution
#                 phase_pol = 0.02 if pol == 'V' else 0.0  # Polarization offset
#
#                 phase_data[i, j, k, l] = phase_base + phase_az + phase_el + phase_pol
#
# # Combine into single 5D array (Az, El, Freq, Pol, Var)
# combined_data = np.stack([mag_data, phase_data], axis=4)
#
# # Create 5D DataArray
# da = xr.DataArray(
#     data=combined_data,
#     dims=['Az', 'El', 'Frequency', 'Polarization', 'Variable'],
#     coords={
#         'Az': azimuth,
#         'El': elevation,
#         'Frequency': frequency,
#         'Polarization': polarization,
#         'Variable': variables
#     },
#     name='Antenna_Measurements',
#     attrs={
#         'description': '5D antenna measurements with elevation dimension',
#         'units': 'Magnitude (linear), Phase (radians)',
#         'generation_method': 'Synthetic antenna pattern model'
#     }
# )
#
# # Add magnitude and phase as non-dimensional coordinates
# da = da.assign_coords({
#     'Magnitude': (['Az', 'El', 'Frequency', 'Polarization'], mag_data),
#     'Phase': (['Az', 'El', 'Frequency', 'Polarization'], phase_data)
# })
#
# print("Created 5D DataArray with dimensions:", da.dims)
# print("Data shape:", da.shape)
#
# # Example data access
# print("\nSample values:")
# print(f"Mag @ Az=0°, El=0°, Freq=100MHz, H: {da.sel(Az=0, El=0, Frequency=100, Polarization='H', Variable='Mag').item():.3f}")
# print(f"Phase @ Az=30°, El=30°, Freq=300MHz, V: {da.sel(Az=30, El=30, Frequency=300, Polarization='V', Variable='Phase').item():.3f}")
#
# =============================================================================

# =============================================================================
#                 data_full_DA = xr.DataArray(data_col_stack,
#                                             dims=["Azimuth",
#                                                   "Magnitude", 'Phase'],
#                                             coords=dict(
#                                                 Az=data_Az_angle,
#                                                 Magnitude=(['Azimuth',
#                                                            'Magnitdue'],
#                                                            data_mag),
#                                                 Phase=(['Azimuth',
#                                                        'Phase'],
#                                                        data_phase)
#                                             ),
#                                             attrs=dict(
#                                                 description=(
#                                                     "Full Dataset for " +
#                                                     "the associated " +
#                                                     "outdoor " +
#                                                     "range measurement."
#                                                 ),
#                                                 name='Data Array All'
#                                             )
#                                             )
#
#                 # Use Xarray to create a dataset
#                 # a dataset in xarray may not be needed here, a dataarray
#                 # may be best, but also astropy tables may work better here
#                 # looking into this, leaving any items in work
#                 data_full_DS = xr.Dataset(
#                     data_vars=dict(magnitude=(["freq", "Az"], data_mag),
#                                    phase=(["freq", "Az"], data_phase)
#                                    ),
#                     coords=dict(
#                         frequency=("freq", data_freq),
#                         Azimuth=("Az", data_Az_angle)
#                     ),
#                     attrs=dict(description=(
#                         "Full Dataset for the associated " + "outdoor " +
#                         "range measurement.")
#                     ),
#                     name='Data Set All'
#                 )
# =============================================================================

                # Create datasets
                dataset = os.path.splitext(dataset_name)[0]
                freqName = (dataset +
                            '_freq')
                magName = (dataset +
                           '_mag')
                phaseName = (dataset +
                             '_phase')
                azName = (dataset +
                          '_Az')
                self.plotTitle = dataset
                self.doc.SetData(freqName, data_freq)
                self.doc.SetData(magName, data_mag)
                self.doc.SetData(phaseName, data_phase)
                self.doc.SetData(azName, data_Az_angle)
                self.doc.TagDatasets(dataset,
                                     [freqName, magName, phaseName, azName])

                # % Overlay Plot
                if 'Overlay_All_mag' not in self.doc.Root.childnames:
                    # Create Pages and Graphs for Overlays
                    pageAll_mag = self.doc.Root.Add('page',
                                                    name='Overlay_All_mag')
                    gridAll_mag = pageAll_mag.Add('grid', columns=2)
                    graphAll_mag = gridAll.Add('graph',
                                               name='Overlay_All_mag')

                    pageAll_phase = self.doc.Root.Add('page',
                                                      name='Overlay_All_phase')
                    gridAll_phase = pageAll_phase.Add('grid', columns=2)
                    graphAll_phase = gridAll_phase.Add('graph',
                                                       name='Overlay_All_phase')

                    # pageAll_Polar_mag = self._create_page('Overlay_All_mag')
                    # graphAll_Polar_mag = self.grid.Add('graph',
                    #                              name='Overlay_All_mag')

                    # pageAll_Polar_phase = self._create_page('Overlay_All_phase')
                    # graphAll_Polar_phase = self.grid.Add('polar',
                    #                              name='Overlay_All_phase')

                    # Add notes to the overlay pages
                    pageAll_mag.notes.val = ("All Imported " +
                                             "and Plottable Data Overlay")
                    pageAll_phase.notes.val = ("All Imported " +
                                               "and Plottable Data Overlay")

                    with Wrap4With(graphAll_mag) as graph:
                        graph.Add('label', name='plotTitle')
                        graph.topMargin.val = '1cm'
                        graph.plotTitle.Text.size.val = '10pt'
                        graph.plotTitle.label.val = 'Overlay of Imported Magnitude'
                        graph.plotTitle.alignHorz.val = 'centre'
                        graph.plotTitle.yPos.val = 1.05
                        graph.plotTitle.xPos.val = 0.5
                        graph.notes.val = 'All Imported Data Overlayed'
                        # set graph axis labels
                        graph.x.label.val = self.az_label
                        graph.y.label.val = self.mag_label

                        # grid lines
                        graph.x.GridLines.hide.val = False
                        graph.y.GridLines.hide.val = False
                        graph.x.MinorGridLines.hide.val = False
                        graph.y.MinorGridLines.hide.val = False

                        # Extents
                        graph.y.min.val = -60
                        graph.y.max.val = 20
                        graph.x.min.val = -180
                        graph.x.max.val = 180

                    with Wrap4With(graphAll_phase) as graph:
                        graph.Add('label', name='plotTitle')
                        graph.topMargin.val = '1cm'
                        graph.plotTitle.Text.size.val = '10pt'
                        graph.plotTitle.label.val = 'Overlay of Imported Phase'
                        graph.plotTitle.alignHorz.val = 'centre'
                        graph.plotTitle.yPos.val = 1.05
                        graph.plotTitle.xPos.val = 0.5
                        graph.notes.val = 'All Imported Data Overlayed'
                        # set graph axis labels
                        graph.x.label.val = self.az_label
                        graph.y.label.val = self.phase_label

                        # grid lines
                        graph.x.GridLines.hide.val = False
                        graph.y.GridLines.hide.val = False
                        graph.x.MinorGridLines.hide.val = False
                        graph.y.MinorGridLines.hide.val = False

                        # Extents
                        graph.y.min.val = -180
                        graph.y.max.val = 180
                        graph.x.min.val = -180
                        graph.x.max.val = 180

                    # set auto color theme for the files
                    self.doc.Root.colorTheme.val = 'max128'

                    # Create xy plot for magnitude on page, graph, and grid ALL
                    xy_All_mag = graphAll_mag.Add(
                        'xy', name=magName)
                    with Wrap4With(xy_All_mag) as xy:
                        xy.xData.val = azName
                        xy.yData.val = magName
                        xy.nanHandling = 'break-on'
                        xy.marker.val = 'circle'
                        xy.markerSize.val = '2pt'
                        xy.MarkerLine.color.val = 'transparent'
                        xy.MarkerFill.color.val = 'auto'
                        xy.MarkerFill.transparency.val = 80
                        xy.MarkerFill.style.val = 'solid'
                        xy.FillBelow.transparency.val = 90
                        xy.FillBelow.style.val = 'solid'
                        xy.FillBelow.fillto.val = 'bottom'
                        xy.FillBelow.color.val = 'darkgreen'
                        xy.FillBelow.hide.val = True
                        xy.PlotLine.color.val = 'auto'

                    xy_All_phase = graphAll_phase.Add(
                        'xy', name=phaseName)
                    with Wrap4With(xy_All_phase) as xy:
                        xy.xData.val = azName
                        xy.yData.val = phaseName
                        xy.nanHandling = 'break-on'
                        xy.marker.val = 'circle'
                        xy.markerSize.val = '2pt'
                        xy.MarkerLine.color.val = 'transparent'
                        xy.MarkerFill.color.val = 'auto'
                        xy.MarkerFill.transparency.val = 80
                        xy.MarkerFill.style.val = 'solid'
                        xy.FillBelow.transparency.val = 90
                        xy.FillBelow.style.val = 'solid'
                        xy.FillBelow.fillto.val = 'bottom'
                        xy.FillBelow.color.val = 'darkgreen'
                        xy.FillBelow.hide.val = True
                        xy.PlotLine.color.val = 'auto'

                    # TODO: Create new phase plot on a new page

                    # TODO: Create Polar Plots (Mag and Phase)
                    # pageAll_Polar_mag = self._create_page(
                    #     'Overlay_All_Polar_mag')

                else:
                    # TODO: repat all overlays as needed
                    pageAll_mag = self.doc.Root.Overlay_All_mag
                    graphAll_mag = (

                        self.doc.Root.Overlay_All_mag.grid1.Overlay_All_mag
                    )

                    pageAll_phase = self.doc.Root.Overlay_All_phase
                    graphAll_phase = (
                        self.doc.Root.Overlay_All_mag.grid1.Overlay_All_phase
                    )

                    xy_All_mag = graphAll_mag.Add(
                        'xy', name=magName)
                    with Wrap4With(xy_All_mag) as xy:
                        xy.xData.val = azName
                        xy.yData.val = magName
                        xy.nanHandling = 'break-on'
                        xy.marker.val = 'circle'
                        xy.markerSize.val = '2pt'
                        xy.MarkerLine.color.val = 'transparent'
                        xy.MarkerFill.color.val = 'auto'
                        xy.MarkerFill.transparency.val = 80
                        xy.MarkerFill.style.val = 'solid'
                        xy.FillBelow.transparency.val = 90
                        xy.FillBelow.style.val = 'solid'
                        xy.FillBelow.fillto.val = 'bottom'
                        xy.FillBelow.color.val = 'darkgreen'
                        xy.FillBelow.hide.val = True
                        xy.PlotLine.color.val = 'auto'

                    xy_All_phase = graphAll_phase.Add(
                        'xy', name=phaseName)
                    with Wrap4With(xy_All_phase) as xy:
                        xy.xData.val = azName
                        xy.yData.val = phaseName
                        xy.nanHandling = 'break-on'
                        xy.marker.val = 'circle'
                        xy.markerSize.val = '2pt'
                        xy.MarkerLine.color.val = 'transparent'
                        xy.MarkerFill.color.val = 'auto'
                        xy.MarkerFill.transparency.val = 80
                        xy.MarkerFill.style.val = 'solid'
                        xy.FillBelow.transparency.val = 90
                        xy.FillBelow.style.val = 'solid'
                        xy.FillBelow.fillto.val = 'bottom'
                        xy.FillBelow.color.val = 'darkgreen'
                        xy.FillBelow.hide.val = True
                        xy.PlotLine.color.val = 'auto'

                # Create a new single plot for magnitude
                # TODO: All pages and graph creation
                # Magnitude
                page_mag = self.doc.Root.Add('page', name=magName)
                grid_mag = page_mag.Add('grid', columns=2)
                graph_mag = grid_mag.Add(
                    'graph',
                    name=dataset + ' Mag')
                page_mag.notes.val = '\n'.join(header_lines)

                # Phase
                page_phase = self.doc.Root.Add('page', name=phaseName)
                grid_phase = page_phase.Add('grid', columns=2)
                graph_phase = grid_phase.Add(
                    'graph',
                    name=dataset + ' Phase')
                page_phase.notes.val = '\n'.join(header_lines)

                # # Magnitude Polar
                # # Phase Polar

                with Wrap4With(graph_mag) as graph:
                    graph.Add('label', name='plotTitle')
                    graph.topMargin.val = '1cm'
                    graph.plotTitle.Text.size.val = '10pt'
                    graph.plotTitle.label.val = self.plotTitle.replace(
                        '_', " ")
                    graph.plotTitle.alignHorz.val = 'centre'
                    graph.plotTitle.yPos.val = 1.05
                    graph.plotTitle.xPos.val = 0.5
                    graph.notes.val = '\n'.join(header_lines)
                    # set graph axis labels
                    graph.x.label.val = self.az_label
                    graph.y.label.val = self.mag_label

                    # grid lines
                    graph.x.GridLines.hide.val = False
                    graph.y.GridLines.hide.val = False
                    graph.x.MinorGridLines.hide.val = False
                    graph.y.MinorGridLines.hide.val = False

                    # Extents
                    graph.y.min.val = -60
                    graph.y.max.val = 20
                    graph.x.min.val = -180
                    graph.x.max.val = 180

                    # Create xy plot
                    xy_mag = graph.Add(
                        'xy', name=dataset
                    )

                with Wrap4With(xy_mag) as xy:
                    # Assign Data
                    xy.xData.val = azName
                    xy.yData.val = magName

                    # Create a new Axis
                    # note that in xy.xAxis, this can be changed to match the give name
                    # x_axis = graph.Add('axis', name='frequency')
                    # y_axis = graph.Add('axis', name='counts')
                    # x_axis.label.val = 'Frequency (MHz)'
                    # y_axis.label.val = 'Uncalibrated Gain (dB)'
                    # y_axis.direction.val = 'vertical'
                    # # x_axis.childnames gives you all the settable parameters
                    # x_axis.autoRange.val = 'Auto'
                    # xy.xAxis.val = 'frequency'
                    # xy.yAxis.val = 'Uncalibrated Gain'
                    # xy.marker.val = 'none'
                    # xy.PlotLine.color.val = 'red'
                    # xy.PlotLine.width.val = '1pt'
                    xy.nanHandling = 'break-on'

                    # set marker and colors for overlay plot
                    xy.marker.val = 'circle'
                    xy.markerSize.val = '2pt'
                    # xy.MarkerLine.transparency.val =
                    xy.MarkerLine.color.val = 'transparent'
                    xy.MarkerFill.color.val = 'auto'
                    xy.MarkerFill.transparency.val = 80
                    xy.MarkerFill.style.val = 'solid'
                    xy.FillBelow.transparency.val = 90
                    xy.FillBelow.style.val = 'solid'
                    xy.FillBelow.fillto.val = 'bottom'
                    xy.FillBelow.color.val = 'darkgreen'
                    xy.FillBelow.hide.val = False
                    xy.PlotLine.color.val = 'red'

                # TODO: Add Page and graph for Phase
                with Wrap4With(graph_phase) as graph:
                    graph.Add('label', name='plotTitle')
                    graph.topMargin.val = '1cm'
                    graph.plotTitle.Text.size.val = '10pt'
                    graph.plotTitle.label.val = self.plotTitle.replace(
                        '_', " ")
                    graph.plotTitle.alignHorz.val = 'centre'
                    graph.plotTitle.yPos.val = 1.05
                    graph.plotTitle.xPos.val = 0.5
                    graph.notes.val = '\n'.join(header_lines)
                    # set graph axis labels
                    graph.x.label.val = self.az_label
                    graph.y.label.val = self.phase_label

                    # grid lines
                    graph.x.GridLines.hide.val = False
                    graph.y.GridLines.hide.val = False
                    graph.x.MinorGridLines.hide.val = False
                    graph.y.MinorGridLines.hide.val = False

                    # Extents
                    graph.y.min.val = -180
                    graph.y.max.val = 180
                    graph.x.min.val = -180
                    graph.x.max.val = 180

                    # Create xy plot
                    xy_phase = graph.Add(
                        'xy', name=dataset
                    )

                with Wrap4With(xy_phase) as xy:
                    # Assign Data
                    xy.xData.val = azName
                    xy.yData.val = phaseName

                    # Axis control
                    xy.nanHandling = 'break-on'

                    # set marker and colors for overlay plot
                    xy.marker.val = 'circle'
                    xy.markerSize.val = '2pt'
                    # xy.MarkerLine.transparency.val =
                    xy.MarkerLine.color.val = 'transparent'
                    xy.MarkerFill.color.val = 'auto'
                    xy.MarkerFill.transparency.val = 80
                    xy.MarkerFill.style.val = 'solid'
                    xy.FillBelow.transparency.val = 90
                    xy.FillBelow.style.val = 'solid'
                    xy.FillBelow.fillto.val = 'bottom'
                    xy.FillBelow.color.val = 'darkgreen'
                    xy.FillBelow.hide.val = False
                    xy.PlotLine.color.val = 'red'

                # TODO: Add a page for Magnitude Polar

                # TODO: Add a page for Phase Polar

            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")

            # Save The Veusz file after loop completed
            # saveDir = os.path.dirname(file_path)
            # # self.doc.Save(saveDir + '/' + dataset_name + '.vsz')
            # filenameSave = saveDir + '/' + os.path.splitext(
            #     dataset_name)[0] + '.vszh5'
            # self._save(filenameSave)

            # # Show the plot
            # self.doc.WaitForClose()
            # self.status_label.config(text="Plot created successfully")
            # self.root.destroy()

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
        return header_info, selected_data_transpose, header_lines

    def save(self, filename: str):
        """Save Veusz document to specified file."""
        # there might be a precision argument or format string
        # Found, just save in hdf veusz format, vszh5
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

    def open_veusz_gui(self, filename: str):
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

    def run(self):

        # run the GUI!
        self.root.mainloop()
        # self.root.destroy()
        pass


class Astronomy:
    """
    Used at the moment to create various functions from the AstroPy module.
    """
    # TODO: Break out into own module

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
        ----------
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
    # TODO: Make this multi file and put in own module
    """ Used for processing CSV Data."""

    def __init__(self):
        pass

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, self.file_path)


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
    # mjd = 60686.3539
    # newTime = Astronomy.mjdconvert_time(mjd, "iso", "ymdhms", "fits")
    # # newTime = Astronomy.mjdconvert_time(mjd)
    # print(newTime)

    ATR_Example = PlotATR()
    ATR_Example.run()
    pass
    # gui = qtGUI()
    # if save_path := gui.get_save_filename():
    #     ATR_Example.save(save_path)
    #     if gui.ask_open_veusz():
    #         PlotATR.open_veusz_gui(save_path)

    # sys.exit(gui.app.exec_())
    # # QApplication.closeAllWindows()
    # gui.closeEvent()

    # saveDir = os.path.dirname(file_path)
    # # self.doc.Save(saveDir + '/' + dataset_name + '.vsz')
    # filenameSave = saveDir + '/' + os.path.splitext(
    #     dataset_name)[0] + '.vszh5'
    # self._save(filenameSave)

    # # Show the plot
    # self.doc.WaitForClose()
    # self.status_label.config(text="Plot created successfully")

    # sys.path.append(
    #     os.path.abspath("C:\\Users\\wwallace\\OneDrive - National Radio Astronomy Observatory" +
    #                     "\\Documents\\Python_Vault")
    # )


if __name__ == "__main__":
    main()
