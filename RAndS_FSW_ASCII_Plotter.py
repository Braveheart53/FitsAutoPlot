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
# %%% Math Modules and others
# import pandas as pd
# import xarray as xr
import numpy as np
# import skrf as rf
from operator import itemgetter
import re
from dataclasses import dataclass
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


@dataclass
class plotDescInfo:
    """
    Setting up general plot info class to update as needed
    """
    xAxis_label: str
    yAxis_label: str
    graph_notes: str
    graph_title: str
    base_name: str


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
            'stop': 'STOP',
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

        # setup the shared plot info are that to be changed and passed
        # with self
        self.plotInfo = plotDescInfo(
            xAxis_label='Frequency [Hz]',
            yAxis_label='Uncalibrated (dBm)',
            graph_notes=None,
            graph_title='Title',
            base_name=None
        )

    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)

# TODO
# look at removing this function as it may not be needed her
# =============================================================================
#     def create_plots(self):
#         """Generate automated plots based on dataset dimensions.
#             Only after all data sets are created in veusz"""
#         for ds in self.doc.GetDatasets():
#             ds_type = self.doc.GetDataType(ds)
#             # create a new page for each plot and name it by ds
#             self._create_page(ds)
#
#             # use a case or switch statement to plot according to dataset
#             # veusz documentation does not address nd dataset
#             # if ds == 'data':
#             #     print('Found the nd Array Data')
#
#             for case in switch(ds_type):
#                 if case('1d'):
#                     # somehow nd datatype  is not being passed
#                     currentDataDims = self.doc.GetData(ds)[0].ndim
#                     if currentDataDims == 1:
#                         self._plot_1d(ds)
#                     else:
#                         # This means it is ndimensional data...
#                         ds_type = 'nD'
#                         self._plot_nd(ds)
#                     del currentDataDims
#                     break
#                 if case('2d'):
#                     self._plot_2d(ds)
#                     break
#                 if case('text'):
#                     pass
#                     break
#                 if case('datetime'):
#                     pass
#                     break
#                 if case('nD'):
#                     print('An ND case was found in ' + ds)
#                     break
# =============================================================================

    def _plot_1d(self, dataset: str):
        """Create line plot for 1D datasets with red initial trace."""
        try:
            self._create_page(dataset)
            graph = self.grid.Add('graph')

            # add graph title
            graph.Add('label', name='plotTitle')
            graph.topMargin.val = '1cm'
            graph.plotTitle.Text.size.val = '10pt'
            graph.plotTitle.label.val = self.plotInfo.graph_title
            graph.plotTitle.alignHorz.val = 'left'
            graph.plotTitle.yPos.val = 1.05
            graph.plotTitle.xPos.val = -0.3

            # add notes to graph
            graph.notes.val = self.plotInfo.graph_notes

            # add xy scatter plot
            xy = graph.Add('xy')

            # set the datasets being used for the plot
            # TODO
            # may want to look into crating separate files for high precision
            # such as np.save('high_precision_data.npy', x_data)
            # then just linking those files instead of using local datasets
            xy.yData.val = dataset
            xy.xData.val = self.plotInfo.base_name + '_freq'

            # set graph axis labels
            graph.x.label.val = self.plotInfo.xAxis_label
            graph.y.label.val = self.plotInfo.yAxis_label
            # breakpoint

            # set marker and colors
            xy.marker.val = 'none'
            xy.PlotLine.color.val = 'red'

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

    def rns_sft_parser(self, fileName: str = None):
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
            fileName, fileParts = self._select_sft_files()
        elif ((isinstance(fileName, list) or isinstance(fileName, tuple)) and
              os.path.splitext(fileName[0])[1] == '.sft'):
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
        if isinstance(fileName, list) or isinstance(fileName, tuple):
            pass
        else:
            # just make it a list so the following applies without issue
            fileName = [fileName]

        for currentFile in fileName:
            dataReturned = fparser(currentFile, line_targets=self.sft_lines,
                                   string_patterns=self.searchData_strings)

            # used for data set name and label
            base_name = os.path.splitext(os.path.basename(currentFile))[0]
            self.plotInfo.base_name = base_name
            # based on section string in the sft file, extract on the
            # section data defintions
            data_Sections = dict(filter(
                lambda item: 'section' in item[0],
                dataReturned['pattern_matches'].items()
            ))

            # test length of data sections and data_located
            if len(data_Sections) != len(dataReturned['data_matches']):
                # error, these should match
                QMessageBox.critical(
                    None,
                    "Data Sections and the extracted number of data \n",
                    "lists do not match. \n",
                    "Unable to proceed at the moment. \n",
                    "Tell William to stop being lazy and correct this."
                )
                # TODO
                # make this more robust, it really should match, but at
                # least do some error checking on the data itself.
                return

            # now we have all the data in dataReturned of type dict
            # use this and VZPlotRns to create Veusz plots

            # all_items = [(k, v) for subdict in parent_dict.values() for
            #               k, v in subdict.items()]
            #
            # using operator itemgetter for better performance
            # being 0 based index, but all data matches the index of the
            # dict data_Sections for date info
            # line numbers will be used to ensure we are using the
            # correct data.
            data_y_values = list(map(itemgetter('extracted_value'),
                                     dataReturned['data_matches'].values())
                                 )
            data_line_numbers = list(map(itemgetter('line_number'),
                                         dataReturned['data_matches'].values())
                                     )
            data_Section_line_numbers = list(map(itemgetter('line_number'),
                                                 data_Sections.values()))
            data_Section_content = list(map(itemgetter('content'),
                                            data_Sections.values()))
            # breakpoint
            # so now we have to make the X axis. For each file this
            # remains the same.
            # use pattern_matches for this
            data_fields = dataReturned['pattern_matches']

            # get number of points
            numPts = extract_with_regex(
                data_fields['values']['extracted_value'])

            if len(numPts) != 1:
                QMessageBox.critical(
                    None,
                    "Values Line of SFT File Incorrect. \n",
                    "Check file " + str(currentFile)
                )
                # TODO
                # make this more robust, it really should match, but at
                # least do some error checking on the data itself.
                return
            else:
                numPts = int(numPts[0])

            # get span
            freqStart = extract_with_regex(
                data_fields['start']['extracted_value'])
            freqStop = extract_with_regex(
                data_fields['stop_2']['extracted_value'])
            freqCenter = extract_with_regex(
                data_fields['center freq']['extracted_value'])
            freqSpan = extract_with_regex(
                data_fields['span']['extracted_value'])
            freqStart = float(freqStart[0])
            freqStop = float(freqStop[0])
            freqCenter = float(freqCenter[0])
            freqSpan = float(freqSpan[0])
            # TODO
            # look at these values and ensure they are not trucated here
            freqRange = np.linspace(freqStart, freqStop, num=numPts,
                                    endpoint=True, dtype=np.float64)
            # data_x_value = freqRange.tolist()
            data_x_value = freqRange

            if len(data_x_value) != len(set(data_x_value)):
                QMessageBox.critical(
                    None,
                    "Something happened with frequency list \n",
                    "Generation from file  ", str(currentFile), ".\n",
                    "See ", "data_fields in the script."
                )

            # step through line data to create header notes to be
            # put in each graph.
            data_header = dataReturned['line_data']
            data_notes = '\n'.join(data_header.values())

            for index, label in enumerate(data_Section_content):
                dataSetName = label
                if index == 0:
                    x_data = data_x_value
                    x_data_name = base_name + '_freq'
                    # TODO
                    # may want to look into crating separate files for high precision
                    # such as np.save('high_precision_data.npy', x_data)
                    # then just linking those files instead of using local datasets
                    self.doc.SetData(name=x_data_name, val=x_data)
                    # self.doc.SetDataLabel(dataSetName,
                    #                       f"Frequency [{base_name}]")
                    self.doc.TagDatasets(base_name, [x_data_name])
                description = label
                # Put all of the header information in the notes of the plot
                # self.doc.SetNote()

                # Put all data into a dataset with a name and label and tag
                if (data_line_numbers[index]-1
                        != data_Section_line_numbers[index]):
                    QMessageBox.critical(
                        None,
                        "The data and its labels are not aligned. \n",
                        "Check file " + str(currentFile)
                    )
                    # TODO
                    # make this more robust, it really should match, but at
                    # least do some error checking on the data itself.
                    return

                # TODO
                # may want to look into crating separate files for high precision
                # such as np.save('high_precision_data.npy', x_data)
                # then just linking those files instead of using local datasets
                # by using the keywork linked=True
                self.doc.SetData(name=dataSetName, val=data_y_values[index])
                # setDataText?!
                # self.doc.SetDataLabel(dataSetName,
                #                       f"{description} [{base_name}]")
                self.doc.TagDatasets(base_name, [dataSetName])
                # self.doc.TagDatasets(label, [dataSetName])

                # ok, all the data is now in Veusz, so we need to create
                # the pages, grids and graphs. I would like to embed
                # the header information is each graph.
                # best to do per file and data section so information
                # is not lost and exterion indexing is not needed
                self.plotInfo.graph_notes = data_notes
                self.plotInfo.graph_title = base_name + '::' + dataSetName
                self.plotInfo.graph_title = (
                    self.plotInfo.graph_title.replace('_', ' ')
                )

                # plotting directly instead of using self.create_plots(self)
                self._plot_1d(dataSetName)
                # breakpoint

    def _select_sft_files(self):
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
        # Create a root window
        root = tk.Tk()

        # Hide the root window
        root.withdraw()
        # root.iconify()

        filename = askopenfilenames(filetypes=[("Rhode & Schwarz SFT Files",
                                                ".SFT")])
        root.destroy()
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


def extract_between(inputText: str, start_char: str, end_char: str):
    """
    Extract a string between two characters and return a list of found.

    Parameters
    ----------
    text : TYPE
        DESCRIPTION.
    start_char : TYPE
        DESCRIPTION.
    end_char : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    pattern = re.escape(start_char) + r"(.*?)" + re.escape(end_char)
    return re.findall(pattern, text)


def extract_with_regex(inputText: str, delim: str = ';'):
    """
    Extract all substrings enclosed by the same delimiter using regex.
    """
    # Escape delim if it’s a regex special character
    esc = re.escape(delim)
    pattern = rf"{esc}(.*?){esc}"        # Non-greedy capture between delim
    return re.findall(pattern, inputText)     # List of all matches


# %% Main Execution
if __name__ == '__main__':
    """
    execution of main function
    """
    vz = VZPlotRnS()
    vz.rns_sft_parser()

    # this is done in the data parsing to ensure multiple traces are labeled
    # correctly
    # vz.create_plots()

    gui = qtGUI()
    if save_path := gui.get_save_filename():
        vz.save(save_path)
        if gui.ask_open_veusz():
            VZPlotRnS.open_veusz_gui(save_path)

    sys.exit(gui.app.exec_())
    # QApplication.closeAllWindows()
    gui.closeEvent()


# if __name__ == "__main__":
#     main()
