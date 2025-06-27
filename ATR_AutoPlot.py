"""Main Script for GBO Outdoor Antenna Range Data Files."""
# -*- coding: utf-8 -*-
# %% Header
# =============================================================================
# Author: William W. Wallace
#
#
#
#
# TODO: Update CSV to auto plot multiple files, extract it to its own file
# TODO: Checkout PyAntenna and stats calcs for the data
# =============================================================================

# event loop QApplication.exec_()
# %% Import all required modules
import subprocess
from operator import itemgetter
import sys
from qtpy.QtGui import *
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton
)
from rich import inspect as richinspect
import pdir
import veusz.embed as vz
import numpy as np

# %%% System Interface Modules
import os
os.environ['QT_API'] = 'pyside6'

# %%% GUI Uses
# %%%% qtpy imports


# %%% Plotting Environment

# %%% Debug help

# %% Class Definitions
# Begin Class definitions based upon use cases for range and data


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
        """Return the match."""
        return self.value in args


class qtGUI_Save:
    """Handles all Qt-based user interactions."""

    def __init__(self):
        self.appSave = QApplication(sys.argv)

    def closeEvent(self, event):
        """Close the Event."""
        QApplication.closeAllWindows()
        event.accept()

    def _select_sft_file(self, dialog):
        """Handle file selection button click."""
        fname, _ = QFileDialog.getOpenFileName(
            dialog, "Open ATR File", "", "GBO ATR Files (*.atr)"
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
    """Class Utilized to import, parse, and plot GBO Outdoor Range Data."""

    def __init__(self):
        """Initialize the PlotATR Class."""
        """

        Returns
        -------
        None.

        """
        if not hasattr(self, 'plotApp'):
            self.plotapp = (
                QApplication.instance() or QApplication(sys.argv)
            )
            self.plotwindow = QWidget()
            self.plotwindow.setWindowTitle('ATR Plot Interface')
            # Set initial self.plotwindow size
            self.plotwindow.resize(500, 300)

            # Create labels
            self.label_filename = QLabel('Filename(s):')
            self.label_plot_title = QLabel('Plot Title:')
            self.label_data_set = QLabel('Data Set Name:')

            # Create input fields
            self.lineedit_filename = QLineEdit()
            self.lineedit_plot_title = QLineEdit()
            self.lineedit_data_set = QLineEdit()

            # Create buttons
            self.button_browse = QPushButton('Browse')
            self.button_create_plots = QPushButton('Create Plots')
            self.button_save_close = QPushButton('Save and Close')

            # Filename field with Browse button
            self.filename_layout = QHBoxLayout()
            self.filename_layout.addWidget(self.lineedit_filename)
            self.filename_layout.addWidget(self.button_browse)

            # Main layout
            self.main_layout = QVBoxLayout()

            # Add filename section
            self.main_layout.addWidget(self.label_filename)
            self.main_layout.addLayout(self.filename_layout)

            # Add plot title field
            self.main_layout.addWidget(self.label_plot_title)
            self.main_layout.addWidget(self.lineedit_plot_title)

            # Add dataset name field
            self.main_layout.addWidget(self.label_data_set)
            self.main_layout.addWidget(self.lineedit_data_set)

            # Add action buttons
            self.main_layout.addSpacing(20)  # Add space before buttons
            self.main_layout.addWidget(self.button_create_plots)
            self.main_layout.addWidget(self.button_save_close)

            # Set main layout
            self.plotwindow.setLayout(self.main_layout)

            # Connect buttons to functionality
            self.button_create_plots.clicked.connect(self.create_plot)
            self.button_save_close.clicked.connect(self.save_Veusz)
            self.button_browse.clicked.connect(self._select_atr_files)

        # File Info
        if not hasattr(self, 'fileParts'):
            self.fileParts = None

        if not hasattr(self, 'filenames'):
            self.filenames = None

        # Plot Info
        # if not self.plotTitle:
        self.plotTitle = 'GBO Outdoor Antenna Range Pattern'
        self.freq_label = 'Frequency (MHz)'
        self.az_label = 'Azimuth (degrees)'
        self.phase_label = 'Phase (degrees)'
        self.mag_label = 'Magnitude (dB)'

        # Veusz Object
        if not hasattr(self, 'doc'):
            self.doc = vz.Embedded('GBO ATR Autoplotter', hidden=False)
            self.doc.EnableToolbar()

    def save_Veusz(self):
        """Save the generated file and ask to open Veusz Interface."""
        sys.exit(self.plotApp.exec_())
        gui = qtGUI_Save()
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

    def _select_atr_files(self, parent: QWidget = None,
                          caption: str = "Select Files",
                          directory: str = "",
                          filter: str = "GBO ATR Files (*.atr)"
                          ):
        """QtPy or Pyside6 GUI for File Select, Multi File for ATRs."""
        """Open a file dialog for multiple file selection using qtpy.
        select_multiple_files(
            parent: QWidget = None, 
            caption: str = "Select Files", 
            directory: str = "", 
            filter: str = "All Files (*)"):

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget for the dialog.
        caption : str, optional
            The dialog window title.
        directory : str, optional
            The initial directory shown in the dialog.
        filter : str, optional
            File type filter string, e.g. "Images (*.png *.jpg);;Text files (*.txt)".
    
        Returns
        -------
        list of str
            List of selected file paths. Empty if cancelled.
        """
        breakpoint
        if parent is None or not parent:
            parent = QWidget()

        self.self.filenames, _ = QFileDialog.getOpenFileNames(
            parent, caption, directory, filter)

        # start the main loop for processing the selected files
        self.fileParts = [None] * len(self.filenames)
        for mainLooper in range(len(self.filenames)):
            # this loop processes the files selected one at a time,
            # while combining
            # the data as it progresses

            # get the file parts for further use of the current file.
            self.fileParts[mainLooper] = os.path.split(
                self.filenames[mainLooper]
            )

        # After the mainloop, I need to combine all the data into a
        # multi-dimensional
        # array. Then call Veusz and parse the data into that gui.
        if self.fileParts[0][0]:
            filenames_only = list(map(itemgetter(1), self.fileParts))
            # update the file listing

        return self.filenames, self.fileParts

    def create_plot(self):
        """Create the 2D plots for all data."""
        data_freq_all = np.empty(len(self.filenames), dtype=object)
        data_mag_all = np.empty(len(self.filenames), dtype=object)
        data_phase_all = np.empty(len(self.filenames), dtype=object)
        data_Az_angle_all = np.empty(len(self.filenames), dtype=object)
        # data_Polarization_all = np.array(['H', 'V', str(45), str(-45)])
        # measurement_Types = ['Mag', 'Phase']

        for index, file_path in enumerate(self.filenames):

            # file_path = element
            # if self.plot_name_entry.get():
            #     plot_name = self.plot_name_entry.get()
            # else:
            #     plot_name = self.fileParts[index][1]

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
                # freq_step = float(header_lines[10].split(":")[-1].strip())

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

                data_freq_all[index] = data_freq[0]

                data_mag_all[index] = data_mag
                data_phase_all[index] = data_phase
                data_Az_angle_all[index] = data_Az_angle

                # data_col_stack = np.column_stack(
                #     (data_Az_angle, data_mag, data_phase)
                # )

                # Create datasets
                dataset = os.path.splitext(dataset_name)[0]
                freqName = (dataset +
                            '_freq')
                magName = (dataset +
                           '_mag')
                polarMagName = (dataset + '_polar'
                                '_mag')
                # radName = (dataset +
                #            '_r')
                # thetaName = (dataset +
                #              '_theta')
                phaseName = (dataset +
                             '_phase')
                polarPhaseName = (dataset + '_polar'
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
                # TODO: Create a mag pattern overlay plot
                if 'Overlay_All_mag' not in self.doc.Root.childnames:
                    # Create Pages and Graphs for Overlays
                    pageAll_mag = self.doc.Root.Add('page',
                                                    name='Overlay_All_mag')
                    gridAll_mag = pageAll_mag.Add('grid', columns=2)
                    graphAll_mag = gridAll_mag.Add('graph',
                                                   name='Overlay_All_mag')

                    pageAll_phase = self.doc.Root.Add('page',
                                                      name='Overlay_All_phase')
                    gridAll_phase = pageAll_phase.Add('grid', columns=2)
                    graphAll_phase = gridAll_phase.Add(
                        'graph', name='Overlay_All_phase'
                    )
                    # Add notes to the overlay pages
                    pageAll_mag.notes.val = ("All Imported " +
                                             "and Plottable Data Overlay")
                    pageAll_phase.notes.val = ("All Imported " +
                                               "and Plottable Data Overlay")

                    with Wrap4With(graphAll_mag) as graph:
                        graph.Add('label', name='plotTitle')
                        graph.topMargin.val = '1cm'
                        graph.plotTitle.Text.size.val = '10pt'
                        graph.plotTitle.label.val = (
                            'Overlay of Imported Magnitude'
                        )
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

                else:
                    pageAll_mag = self.doc.Root.Overlay_All_mag
                    graphAll_mag = (

                        self.doc.Root.Overlay_All_mag.grid1.Overlay_All_mag
                    )

                    pageAll_phase = self.doc.Root.Overlay_All_phase
                    graphAll_phase = (
                        self.doc.Root.Overlay_All_phase.grid1.Overlay_All_phase
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

                page_mag = self.doc.Root.Add('page', name=magName)
                grid_mag = page_mag.Add('grid', columns=2)
                graph_mag = grid_mag.Add(
                    'graph',
                    name=dataset + '_Mag')
                page_mag.notes.val = '\n'.join(header_lines)

                # Phase
                page_phase = self.doc.Root.Add('page', name=phaseName)
                grid_phase = page_phase.Add('grid', columns=2)
                graph_phase = grid_phase.Add(
                    'graph',
                    name=dataset + '_Phase')
                page_phase.notes.val = '\n'.join(header_lines)

                page_Polar_mag = self.doc.Root.Add('page',
                                                   name=polarMagName)
                with Wrap4With(page_Polar_mag) as page:
                    page.Add('label', name='plotTitle')
                    page.plotTitle.Text.size.val = '10pt'
                    page.plotTitle.label.val = self.plotTitle.replace(
                        '_', " ")
                    page.plotTitle.alignHorz.val = 'centre'
                    page.plotTitle.yPos.val = 0.95
                    page.plotTitle.xPos.val = 0.5

                grid_Polar_mag = page_Polar_mag.Add('grid', columns=2)
                graph_Polar_mag = grid_Polar_mag.Add(
                    'polar',
                    name=dataset + '_Polar_mag')
                page_Polar_mag.notes.val = '\n'.join(header_lines)

                # Phase Polar
                page_Polar_phase = self.doc.Root.Add('page',
                                                     name=polarPhaseName)

                with Wrap4With(page_Polar_phase) as page:
                    page.Add('label', name='plotTitle')
                    page.plotTitle.Text.size.val = '10pt'
                    page.plotTitle.label.val = self.plotTitle.replace(
                        '_', " ")
                    page.plotTitle.alignHorz.val = 'centre'
                    page.plotTitle.yPos.val = 0.95
                    page.plotTitle.xPos.val = 0.5

                grid_Polar_phase = page_Polar_phase.Add('grid', columns=2)

                graph_Polar_phase = grid_Polar_phase.Add(
                    'polar',
                    name=dataset + '_Polar_Phase')
                page_Polar_phase.notes.val = '\n'.join(header_lines)

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

                # TODO: Polar 2 - Add a page for Magnitude Polar
                with Wrap4With(graph_Polar_mag) as pGraph:

                    pGraph.topMargin.val = '1cm'
                    pGraph.units.val = 'degrees'
                    pGraph.direction.val = 'clockwise'
                    pGraph.position0.val = 'top'

                    # Extents
                    pGraph.minradius.val = -60
                    pGraph.maxradius.val = 20

                    # Create nonorthogonal point plot
                    rtheta_mag = pGraph.Add(
                        'nonorthpoint', name=dataset
                    )

                with Wrap4With(rtheta_mag) as nonortho:
                    # Assign Data
                    nonortho.data1.val = magName
                    nonortho.data2.val = azName

                    # set marker and colors for overlay plot
                    nonortho.PlotLine.color.val = 'red'
                    nonortho.PlotLine.width.val = '2pt'
                    nonortho.MarkerLine.transparency.val = 75
                    nonortho.MarkerFill.transparency.val = 75
                    nonortho.Fill1.transparency.val = 65
                    nonortho.Fill1.color.val = 'green'
                    nonortho.Fill1.filltype.val = 'center'
                    nonortho.Fill1.hide.val = False

                # TODO: Polar 3 - Add a page for Phase Polar
                with Wrap4With(graph_Polar_phase) as pGraph:

                    pGraph.topMargin.val = '1cm'
                    pGraph.units.val = 'degrees'
                    pGraph.direction.val = 'clockwise'
                    pGraph.position0.val = 'top'

                    # Extents
                    pGraph.minradius.val = -180
                    pGraph.maxradius.val = 180

                    # Create nonorthogonal point plot
                    rtheta_phase = pGraph.Add(
                        'nonorthpoint', name=dataset
                    )

                with Wrap4With(rtheta_phase) as nonortho:
                    # Assign Data
                    nonortho.data1.val = phaseName
                    nonortho.data2.val = azName

                    # set marker and colors for overlay plot
                    nonortho.PlotLine.color.val = 'red'
                    nonortho.PlotLine.width.val = '2pt'
                    nonortho.MarkerLine.transparency.val = 75
                    nonortho.MarkerFill.transparency.val = 75
                    nonortho.Fill1.transparency.val = 65
                    nonortho.Fill1.color.val = 'green'
                    nonortho.Fill1.filltype.val = 'center'
                    nonortho.Fill1.hide.val = False

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
        """Use to run the ATR Plotting Routines.

        Returns
        -------
        None.

        """
        # run the GUI!
        # Show the window
        self.plotwindow.show()

        # Start the application event loop
        if QApplication.instance():
            self.plotapp.exec_()


def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : array_like
        x-coordinate(s).
    y : array_like
        y-coordinate(s).

    Returns
    -------
    r : ndarray
        Radial coordinate(s).
    theta : ndarray
        Angular coordinate(s) in radians.
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta


def main():
    """Execute main function."""
    # =============================================================================

    ATR_Example = PlotATR()
    ATR_Example.run()


if __name__ == "__main__":
    main()
