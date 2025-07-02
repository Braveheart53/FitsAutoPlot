# -*- coding: utf-8 -*-

"""
=============================================================================
Enhanced R&S FSW ASCII Plotter with Multiprocessing and GPU Support

Created on 2025-06-28
Enhanced version with parallel processing and GPU acceleration capabilities

Author: William W. Wallace (Enhanced)
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

Version: 1.0.0 - Enhanced with multiprocessing and GPU support
=============================================================================
"""
# TODO: Trouble shoot processing lag and ensure file saving works
# TODO: change status on check box change for enable or disable GPU
# and multiprocessing
# TODO: compare GPU based processing on this one to Touchstone
# TODO: change state of all text boxes to be checked on start

# %% Import all required modules
# %%% System Interface Modules
from dataclasses import dataclass
import re
from operator import itemgetter
import os
import sys
import subprocess
import psutil  # For CPU count detection
# %%% GUI Module Imports - QtPy for cross-platform compatibility
# from qtpy.QtGui import *
# from qtpy.QtCore import Qt, QSize, QThread, Signal
# from qtpy.QtWidgets import (
#     QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
#     QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
#     QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
#     QSpinBox, QGroupBox, QListWidget, QSplitter
# )
if getattr(sys, 'frozen', False):
    #     # Running as compiled executable - use PySide6 directly
    # System Interface Modules
    os.environ['QT_API'] = 'pyside6'
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
    from PySide6.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QSplitter
    )
    # from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
    # from PyQt6.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    # from PyQt6.QtWidgets import (
    #     QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    #     QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
    #     QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
    #     QSpinBox, QGroupBox, QListWidget, QSplitter
    # )
else:
    # Development environment - use QtPy
    from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize
    from qtpy.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    from qtpy.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QSplitter
    )


# %%% Math and Processing Modules
import numpy as np
from fastest_ascii_import import fastest_file_parser as fparser


# %%% Parallel Processing Modules
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
# %%% GPU Acceleration Modules
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

# %%% Plotting Environment
import veusz.embed as embed
from veusz.windows.simplewindow import SimpleWindow
from veusz.document import CommandInterface
# %%% File Processing

# %% Configuration and Data Classes


@dataclass
class ProcessingConfig:
    """Configuration class for processing settings."""
    enable_multiprocessing: bool = False
    enable_gpu_processing: bool = False
    use_opencl: bool = True  # Prefer OpenCL for cross-platform compatibility
    num_processes: int = cpu_count()
    max_workers: int = cpu_count()
    chunk_size: int = 1000


@dataclass
class plotDescInfo:
    """Setting up general plot info class to update as needed."""
    xAxis_label: str
    yAxis_label: str
    graph_notes: str
    graph_title: str
    base_name: str
    first_plot: bool

# %% GPU Processing Classes


class GPUProcessor:
    """Handles GPU acceleration using either CuPy or PyOpenCL."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize GPU processor based on available libraries.

        Parameters
        ----------
        config : ProcessingConfig
            Configuration object containing GPU settings.
        """
        self.config = config
        self.gpu_available = False
        self.context = None
        self.queue = None

        if config.enable_gpu_processing:
            self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize GPU context based on available libraries."""
        if self.config.use_opencl and PYOPENCL_AVAILABLE:
            self._initialize_opencl()
        elif CUPY_AVAILABLE:
            self._initialize_cupy()
        else:
            print("No GPU libraries available. Falling back to CPU processing.")

    def _initialize_opencl(self):
        """Initialize OpenCL context for cross-platform GPU support."""
        try:
            platforms = cl.get_platforms()
            if platforms:
                # Select first available platform and device
                platform = platforms[0]
                devices = platform.get_devices(cl.device_type.GPU)
                if not devices:
                    devices = platform.get_devices(cl.device_type.CPU)

                if devices:
                    self.context = cl.Context(devices=[devices[0]])
                    self.queue = cl.CommandQueue(self.context)
                    self.gpu_available = True
                    print(f"OpenCL initialized with device: {devices[0].name}")
        except Exception as e:
            print(f"OpenCL initialization failed: {e}")

    def _initialize_cupy(self):
        """Initialize CuPy for NVIDIA GPU acceleration."""
        try:
            cp.cuda.Device(0).use()
            self.gpu_available = True
            print("CuPy initialized successfully")
        except Exception as e:
            print(f"CuPy initialization failed: {e}")

    def process_array_gpu(self, data_array):
        """
        Process numpy array using GPU acceleration.

        Parameters
        ----------
        data_array : numpy.ndarray
            Input data array to process.

        Returns
        -------
        numpy.ndarray
            Processed data array.
        """
        if not self.gpu_available:
            return data_array

        try:
            if self.config.use_opencl and self.context:
                return self._process_opencl(data_array)
            elif CUPY_AVAILABLE:
                return self._process_cupy(data_array)
        except Exception as e:
            print(f"GPU processing failed: {e}. Falling back to CPU.")

        return data_array

    def _process_opencl(self, data_array):
        """Process data using OpenCL."""
        # Create OpenCL buffers
        mf = cl.mem_flags
        data_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=data_array.astype(np.float32))

        # Simple kernel for demonstration (can be replaced with more complex operations)
        kernel_source = """
        __kernel void process_data(__global float* data) {
            int gid = get_global_id(0);
            // Example: apply some mathematical operation
            data[gid] = data[gid] * 1.0f;  // Identity operation for now
        }
        """

        program = cl.Program(self.context, kernel_source).build()
        kernel = program.process_data

        # Execute kernel
        kernel(self.queue, (len(data_array),), None, data_buffer)

        # Read back results
        result = np.empty_like(data_array, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, data_buffer)
        self.queue.finish()

        return result.astype(data_array.dtype)

    def _process_cupy(self, data_array):
        """Process data using CuPy."""
        gpu_array = cp.asarray(data_array)
        # Example processing (can be replaced with more complex operations)
        processed_gpu = gpu_array * 1.0  # Identity operation for now
        return cp.asnumpy(processed_gpu)

# %% Multiprocessing Worker Functions


def process_file_worker(file_info):
    """
    Worker function for processing individual SFT files in parallel.

    Parameters
    ----------
    file_info : tuple
        Tuple containing (filename, search_strings, sft_lines, config).

    Returns
    -------
    dict
        Processed file data.
    """
    filename, search_strings, sft_lines, config = file_info

    try:
        # Parse the file
        data_returned = fparser(filename, line_targets=sft_lines,
                                string_patterns=search_strings)

        # Initialize GPU processor if enabled
        if config.enable_gpu_processing:
            gpu_processor = GPUProcessor(config)

            # Process data arrays with GPU if available
            for key, data_match in data_returned['data_matches'].items():
                if 'extracted_value' in data_match:
                    data_array = np.array(data_match['extracted_value'])
                    processed_array = gpu_processor.process_array_gpu(
                        data_array)
                    data_returned['data_matches'][key]['extracted_value'] = processed_array.tolist(
                    )

        return {
            'filename': filename,
            'data': data_returned,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'filename': filename,
            'data': None,
            'success': False,
            'error': str(e)
        }


def extract_with_regex(inputText: str, delim: str = ';'):
    """
    Extract all substrings enclosed by the same delimiter using regex.

    Parameters
    ----------
    inputText : str
        Input text to search.
    delim : str
        Delimiter character.

    Returns
    -------
    list
        List of extracted strings.
    """
    esc = re.escape(delim)
    pattern = rf"{esc}(.*?){esc}"
    return re.findall(pattern, inputText)

# %% Enhanced Qt GUI Classes


class FileProcessingThread(QThread):
    """Thread for handling file processing without blocking the GUI."""

    progress_updated = Signal(int)
    processing_finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, file_list, config, search_strings, sft_lines):
        """
        Initialize processing thread.

        Parameters
        ----------
        file_list : list
            List of files to process.
        config : ProcessingConfig
            Processing configuration.
        search_strings : dict
            Search patterns for file parsing.
        sft_lines : list
            Line targets for file parsing.
        """
        super().__init__()
        self.file_list = file_list
        self.config = config
        self.search_strings = search_strings
        self.sft_lines = sft_lines

    def run(self):
        """Execute file processing in separate thread."""
        try:
            if self.config.enable_multiprocessing and len(self.file_list) > 1:
                results = self._process_files_parallel()
            else:
                results = self._process_files_sequential()

            self.processing_finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_files_parallel(self):
        """Process files using multiprocessing."""
        file_info_list = [
            (filename, self.search_strings, self.sft_lines, self.config)
            for filename in self.file_list
        ]

        results = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(process_file_worker, file_info): file_info[0]
                for file_info in file_info_list
            }

            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1
                progress = int((completed / len(self.file_list)) * 100)
                self.progress_updated.emit(progress)

        return results

    def _process_files_sequential(self):
        """Process files sequentially."""
        results = []
        for i, filename in enumerate(self.file_list):
            file_info = (filename, self.search_strings,
                         self.sft_lines, self.config)
            result = process_file_worker(file_info)
            results.append(result)
            progress = int(((i + 1) / len(self.file_list)) * 100)
            self.progress_updated.emit(progress)

        return results


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with modern Qt interface."""

    def __init__(self):
        """Initialize the enhanced main window."""
        super().__init__()
        self.setWindowTitle("Enhanced R&S SFT File Plotter")
        self.setGeometry(100, 100, 800, 600)

        # Initialize configuration
        self.config = ProcessingConfig()

        # Initialize VZPlotRnS
        self.vzplot = VZPlotRnS()

        # Setup UI
        self._setup_ui()

        # File list
        self.selected_files = []

    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        # File list widget
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)

        # Browse button
        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.clicked.connect(self._browse_files)
        browse_layout.addWidget(self.browse_button)

        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        browse_layout.addWidget(self.clear_button)

        file_layout.addLayout(browse_layout)
        main_layout.addWidget(file_group)

        # Processing options section
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)

        # Multiprocessing options
        self.enable_mp_checkbox = QCheckBox("Enable Multiprocessing")
        self.enable_mp_checkbox.setChecked(self.config.enable_multiprocessing)
        self.enable_mp_checkbox.stateChanged.connect(self._update_mp_config)
        options_layout.addWidget(self.enable_mp_checkbox)

        # CPU cores selection
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU Cores:"))
        self.cpu_spinbox = QSpinBox()
        self.cpu_spinbox.setMinimum(1)
        self.cpu_spinbox.setMaximum(cpu_count())
        self.cpu_spinbox.setValue(self.config.num_processes)
        self.cpu_spinbox.valueChanged.connect(self._update_cpu_config)
        cpu_layout.addWidget(self.cpu_spinbox)
        cpu_layout.addStretch()
        options_layout.addLayout(cpu_layout)

        # GPU options
        self.enable_gpu_checkbox = QCheckBox("Enable GPU Processing")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.stateChanged.connect(self._update_gpu_config)
        options_layout.addWidget(self.enable_gpu_checkbox)

        # OpenCL preference
        self.use_opencl_checkbox = QCheckBox("Prefer OpenCL (Cross-platform)")
        self.use_opencl_checkbox.setChecked(self.config.use_opencl)
        self.use_opencl_checkbox.stateChanged.connect(
            self._update_opencl_config)
        options_layout.addWidget(self.use_opencl_checkbox)

        main_layout.addWidget(options_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        main_layout.addWidget(self.status_text)

        # Control buttons
        button_layout = QHBoxLayout()

        self.plot_button = QPushButton("Process and Plot")
        self.plot_button.clicked.connect(self._process_and_plot)
        button_layout.addWidget(self.plot_button)

        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self._save_project)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def _browse_files(self):
        """Open file dialog to select multiple SFT files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("R&S SFT Files (*.sft)")
        file_dialog.setWindowTitle("Select SFT Files")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self._update_file_list()
            self._log_message(f"Selected {len(selected_files)} files")

    def _clear_files(self):
        """Clear the selected files list."""
        self.selected_files.clear()
        self._update_file_list()
        self._log_message("File list cleared")

    def _update_file_list(self):
        """Update the file list widget."""
        self.file_list_widget.clear()
        for file_path in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(file_path))

    def _update_mp_config(self, state):
        """Update multiprocessing configuration."""
        self.config.enable_multiprocessing = state == Qt.Checked
        self._log_message(
            f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}")

    def _update_cpu_config(self, value):
        """Update CPU cores configuration."""
        self.config.num_processes = value
        self.config.max_workers = value
        self._log_message(f"CPU cores set to: {value}")

    def _update_gpu_config(self, state):
        """Update GPU processing configuration."""
        self.config.enable_gpu_processing = state == Qt.Checked
        self._log_message(
            f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}")

    def _update_opencl_config(self, state):
        """Update OpenCL preference configuration."""
        self.config.use_opencl = state == Qt.Checked
        self._log_message(
            f"OpenCL preference: {'Enabled' if self.config.use_opencl else 'Disabled'}")

    def _log_message(self, message):
        """Add message to status text."""
        self.status_text.append(f"[{self._get_timestamp()}] {message}")

    def _get_timestamp(self):
        """Get current timestamp string."""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _process_and_plot(self):
        """Process selected files and create plots."""
        if not self.selected_files:
            QMessageBox.warning(
                self, "Warning", "Please select SFT files first.")
            return

        self.plot_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start processing thread
        self.processing_thread = FileProcessingThread(
            self.selected_files,
            self.config,
            self.vzplot.searchData_strings,
            self.vzplot.sft_lines
        )

        self.processing_thread.progress_updated.connect(
            self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(
            self._on_processing_finished)
        self.processing_thread.error_occurred.connect(
            self._on_processing_error)

        self.processing_thread.start()
        self._log_message("Processing started...")

    def _on_processing_finished(self, results):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        self._log_message(
            f"Processing completed: {len(successful_results)} successful, {len(failed_results)} failed")

        if failed_results:
            error_msg = "\n".join(
                [f"{r['filename']}: {r['error']}" for r in failed_results])
            QMessageBox.warning(self, "Processing Errors",
                                f"Some files failed to process:\n{error_msg}")

        if successful_results:
            self._create_plots(successful_results)
            self.save_button.setEnabled(True)

    def _on_processing_error(self, error_message):
        """Handle processing error."""
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)
        self._log_message(f"Processing error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)

    def _create_plots(self, results):
        """Create Veusz plots from processed results."""
        self._log_message("Creating plots...")

        for result in results:
            filename = result['filename']
            data_returned = result['data']

            try:
                self.vzplot._process_file_data(filename, data_returned)
            except Exception as e:
                self._log_message(f"Plot creation failed for {filename}: {e}")

        self._log_message("Plot creation completed")

    def _save_project(self):
        """Save Veusz project."""
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self, "Save Veusz Project", "",
            "Veusz High Precision Files (*.vszh5)"
        )

        if save_path:
            try:
                self.vzplot.save(save_path)
                self._log_message(f"Project saved: {save_path}")

                # Ask to open in Veusz
                reply = QMessageBox.question(
                    self, "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    VZPlotRnS.open_veusz_gui(save_path)

            except Exception as e:
                QMessageBox.critical(self, "Save Error",
                                     f"Failed to save project: {e}")

# %% Auto Plotter Class


class VZPlotRnS:
    """Enhanced Veusz plotting class with multiprocessing support."""

    def __init__(self):
        """Initialize VZPlotRnS with enhanced capabilities."""
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)

        # Search strings for data parsing
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
            'stop_2': 'STOP',  # Added for compatibility
            'ref level': 'REF LEVEL',
            'level offset': 'LEVEL OFFSET',
            'ref position': 'REF POSITION',
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

        # Line targets for header parsing
        self.sft_lines = [1, 2, 3] + list(range(5, 58, 2))

        # Plot info initialization
        self.plotInfo = plotDescInfo(
            xAxis_label='Frequency (Hz)',
            yAxis_label='Uncalibrated (dBm)',
            graph_notes=None,
            graph_title='Title',
            base_name=None,
            first_plot=True
        )

    def _process_file_data(self, filename, data_returned):
        """
        Process individual file data and create plots.

        Parameters
        ----------
        filename : str
            Path to the processed file.
        data_returned : dict
            Parsed file data.
        """
        base_name = os.path.splitext(os.path.basename(filename))[0]
        self.plotInfo.base_name = base_name

        # Extract section data
        data_sections = dict(filter(
            lambda item: 'section' in item[0],
            data_returned['pattern_matches'].items()
        ))

        if len(data_sections) != len(data_returned['data_matches']):
            raise ValueError(f"Data sections mismatch in file {filename}")

        # Process data values
        data_y_values = list(map(itemgetter('extracted_value'),
                                 data_returned['data_matches'].values()))
        data_line_numbers = list(map(itemgetter('line_number'),
                                     data_returned['data_matches'].values()))
        data_section_line_numbers = list(map(itemgetter('line_number'),
                                             data_sections.values()))
        data_section_content = list(map(itemgetter('content'),
                                        data_sections.values()))

        # Create frequency range
        data_fields = data_returned['pattern_matches']
        num_pts = extract_with_regex(data_fields['values']['extracted_value'])
        if len(num_pts) != 1:
            raise ValueError(f"Invalid VALUES field in {filename}")
        num_pts = int(num_pts[0])

        # Extract frequency parameters
        freq_start = float(extract_with_regex(
            data_fields['start']['extracted_value'])[0])
        freq_stop = float(extract_with_regex(
            data_fields['stop_2']['extracted_value'])[0])

        freq_range = np.linspace(freq_start, freq_stop, num=num_pts,
                                 endpoint=True, dtype=np.float64)

        # Create header notes
        data_header = data_returned['line_data']
        data_notes = '\n'.join(data_header.values())
        data_notes = os.path.split(filename)[1] + '\n\n' + data_notes

        # Process each data section
        for index, label in enumerate(data_section_content):
            dataset_name = label

            if index == 0:
                x_data_name = base_name + '_freq'
                self.doc.SetData(name=x_data_name, val=freq_range)
                self.doc.TagDatasets(base_name, [x_data_name])

            # Verify data alignment
            if (data_line_numbers[index] - 1 != data_section_line_numbers[index]):
                raise ValueError(f"Data alignment error in {filename}")

            # Set data in Veusz
            self.doc.SetData(name=dataset_name, val=data_y_values[index])
            self.doc.TagDatasets(base_name, [dataset_name])

            # Update plot info and create plot
            self.plotInfo.graph_notes = data_notes
            self.plotInfo.graph_title = base_name + '::' + dataset_name
            self.plotInfo.graph_title = self.plotInfo.graph_title.replace(
                '_', ' ')

            self._plot_1d(dataset_name)

    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)

    def _plot_1d(self, dataset: str):
        """Create line plot for 1D datasets with enhanced styling."""
        try:
            # Create overlay plot if it doesn't exist
            if 'AllImported' not in self.doc.Root.childnames:
                self._create_page('AllImported')
                self.page.notes.val = "All Imported and Plottable Data Overlay"

                graph_all = self.grid.Add('graph', name='Imported_Overlay')
                graph_all.Add('label', name='plotTitle')
                graph_all.topMargin.val = '1cm'
                graph_all.plotTitle.Text.size.val = '10pt'
                graph_all.plotTitle.label.val = 'Overlay of All Imported'
                graph_all.plotTitle.alignHorz.val = 'centre'
                graph_all.plotTitle.yPos.val = 1.05
                graph_all.plotTitle.xPos.val = 0.5
                graph_all.notes.val = 'All imported overlay, see individual plots for specifics.'

                # Set axis labels
                graph_all.x.label.val = self.plotInfo.xAxis_label
                graph_all.y.label.val = self.plotInfo.yAxis_label
                self.doc.Root.colorTheme.val = 'max128'
            else:
                self.page = self.doc.Root.AllImported
                graph_all = self.doc.Root.AllImported.grid1.Imported_Overlay

            # Add overlay plot
            all_overlay_xy = graph_all.Add('xy', name=dataset)
            all_overlay_xy.yData.val = dataset
            all_overlay_xy.xData.val = self.plotInfo.base_name + '_freq'
            all_overlay_xy.nanHandling = 'break-on'

            # Style overlay plot
            all_overlay_xy.marker.val = 'circle'
            all_overlay_xy.markerSize.val = '2pt'
            all_overlay_xy.MarkerLine.color.val = 'transparent'
            all_overlay_xy.MarkerFill.color.val = 'auto'
            all_overlay_xy.MarkerFill.transparency.val = 80
            all_overlay_xy.MarkerFill.style.val = 'solid'
            all_overlay_xy.FillBelow.transparency.val = 90
            all_overlay_xy.FillBelow.style.val = 'solid'
            all_overlay_xy.FillBelow.fillto.val = 'bottom'
            all_overlay_xy.FillBelow.color.val = 'darkgreen'
            all_overlay_xy.FillBelow.hide.val = True
            all_overlay_xy.PlotLine.color.val = 'auto'

            self.plotInfo.first_plot = False

            # Create individual plot
            self._create_page(self.plotInfo.graph_title)
            self.page.notes.val = self.plotInfo.graph_notes

            graph = self.grid.Add('graph', name=dataset)
            graph.Add('label', name='plotTitle')
            graph.topMargin.val = '1cm'
            graph.plotTitle.Text.size.val = '10pt'
            graph.plotTitle.label.val = self.plotInfo.graph_title
            graph.plotTitle.alignHorz.val = 'left'
            graph.plotTitle.yPos.val = 1.05
            graph.plotTitle.xPos.val = -0.3
            graph.notes.val = self.plotInfo.graph_notes

            # Add individual plot
            xy = graph.Add('xy', name=dataset)
            xy.yData.val = dataset
            xy.xData.val = self.plotInfo.base_name + '_freq'
            xy.nanHandling = 'break-on'

            # Set axis labels
            graph.x.label.val = self.plotInfo.xAxis_label
            graph.y.label.val = self.plotInfo.yAxis_label

            # Style individual plot
            xy.marker.val = 'circle'
            xy.markerSize.val = '2pt'
            xy.MarkerLine.color.val = 'transparent'
            xy.MarkerFill.color.val = 'foreground'
            xy.MarkerFill.transparency.val = 80
            xy.MarkerFill.style.val = 'solid'
            xy.FillBelow.transparency.val = 90
            xy.FillBelow.style.val = 'solid'
            xy.FillBelow.fillto.val = 'bottom'
            xy.FillBelow.color.val = 'darkgreen'
            xy.FillBelow.hide.val = False
            xy.PlotLine.color.val = 'red'

            if self.first_1d:
                self.first_1d = False

        except Exception as e:
            raise RuntimeError(f"Failed to create 1D plot: {e}")

    def save(self, filename: str):
        """Save Veusz document with high precision support."""
        filename_root = os.path.splitext(filename)[0]
        filename_hp = filename_root + '.vszh5'
        file_split = os.path.split(filename)
        filename_vsz = (file_split[0] + '/Beware_oldVersion/' +
                        os.path.splitext(file_split[1])[0] + '_BEWARE.vsz')

        # Save high precision version
        self.doc.Save(filename_hp, mode='hdf5')

        # Save legacy version
        os.makedirs(file_split[0] + '/Beware_oldVersion/', exist_ok=True)
        self.doc.Save(filename_vsz, mode='vsz')

    @staticmethod
    def open_veusz_gui(filename: str):
        """Launch Veusz GUI with generated project file."""
        if sys.platform.startswith('win'):
            veusz_exe = os.path.join(sys.prefix, 'Scripts', 'veusz.exe')
        else:
            veusz_exe = os.path.join(sys.prefix, 'bin', 'veusz')

        if not os.path.exists(veusz_exe):
            QMessageBox.critical(
                None, "Veusz Not Found",
                "Veusz not found in Python environment.\n"
                "Install with: [pip OR conda OR mamba] install veusz"
            )
            return

        try:
            subprocess.Popen([veusz_exe, filename])
        except Exception as e:
            QMessageBox.critical(
                None, "Launch Error",
                f"Failed to start Veusz: {e}"
            )

# %% Veusz Example for Embedding
class VeuszWin(SimpleWindow):
    """A veusz window displaying a sin function."""

    def __init__(self, title):
        SimpleWindow.__init__(self, title)

        # send commands to this object to modify the window
        # the commands are from the standard veusz api
        ifc = self.interface = CommandInterface(self.document)

        # a basic plot win a sin function
        ifc.To( ifc.Add('page') )
        ifc.To( ifc.Add('graph') )
        ifc.Add('function', name='myfunc')

        ifc.Set( 'myfunc/function', 'sin(x)' )
        ifc.Set( 'x/max', 3.14*2 )

class MainWindow(QWidget):
    """Put veusz window in layout with push button."""

    def __init__(self):
        QWidget.__init__(self)

        lt = QVBoxLayout()
        self.veuszwin = VeuszWin("")
        lt.addWidget(self.veuszwin)
        self.button = QPushButton("hi there")
        lt.addWidget(self.button)
        self.setLayout(lt)
        self.connect(self.button, Signal('clicked()'),
                     self.slotClicked)

    def slotClicked(self):
        filename = 'out.png'
        print("Writing", filename)
        self.veuszwin.interface.Export(filename)
# %% Utility Functions


def setup_qt_plugins():
    """
    Setup Qt platform plugin paths for compiled applications.
    """
    try:
        import PySide6
        dirname = os.path.dirname(PySide6.__file__)
        plugin_path = os.path.join(dirname, 'plugins', 'platforms')
        if os.path.exists(plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    except ImportError:
        pass


# %% Main Application


def main():
    """Main application entry point."""
    # Set multiprocessing start method for cross-platform compatibility
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)
        # Call this before any Qt imports
    if getattr(sys, 'frozen', False):  # Check if running as compiled executable
        setup_qt_plugins()
    app = QApplication(sys.argv)

    # Create and show main window
    window = EnhancedMainWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
