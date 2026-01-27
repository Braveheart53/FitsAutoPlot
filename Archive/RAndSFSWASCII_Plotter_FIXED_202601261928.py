#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced RS FSW ASCII Plotter with Multiprocessing and GPU Support

TITLE: Enhanced RS FSW ASCII Plotter
        Rohde & Schwarz FSW File Processing

DESCRIPTION: This application processes Rohde & Schwarz FSW ASCII files with
parallel processing and optional GPU acceleration (if available).

CREATED: 2025-06-28

AUTHOR: William W. Wallace
ENHANCED AUTHOR EMAIL: wwallace@nrao.edu
SECONDARY EMAIL: naval.antennas@gmail.com
BUSINESS PHONE: 1 (304) 456-2216

VERSION: 1.0.1 - Enhanced with plotting options and automatic batch saving
COMPATIBLE WITH: Python 3.8+, PySide6, QtPy
CUDA STATUS: Optional (gracefully falls back to CPU if CUDA unavailable)
"""

import datetime  # Added for auto-save timestamping
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from operator import itemgetter

# Suppress CuPy CUDA warnings
warnings.filterwarnings('ignore', message='.*CUDA path could not be detected.*')

# Import modules based on execution context
if getattr(sys, 'frozen', False):
    # Force direct PySide6 usage for compiled builds
    os.environ['QT_API'] = 'pyside6'
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QGroupBox, QListWidget
    )
else:
    # Development environment - use QtPy
    os.environ['QT_API'] = 'pyside6'
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QGroupBox, QListWidget
    )

# Math and Processing Modules
import numpy as np
from fastest_ascii_import import fastest_file_parser as fparser

# Parallel Processing Modules
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

# ============================================================================
# GPU ACCELERATION MODULES - WITH IMPROVED CUDA ERROR HANDLING
# ============================================================================

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    # Verify CUDA device is actually accessible
    try:
        cp.cuda.Device(0).use()
        CUPY_AVAILABLE_VERIFIED = True
        print("✓ CuPy and CUDA verified successfully")
    except Exception as cuda_error:
        print(f"⚠ Warning: CuPy detected but CUDA device not accessible: {cuda_error}")
        print("  Falling back to CPU processing (NumPy only)")
        CUPY_AVAILABLE = False
        CUPY_AVAILABLE_VERIFIED = False
except ImportError:
    CUPY_AVAILABLE = False
    CUPY_AVAILABLE_VERIFIED = False
except Exception as e:
    print(f"⚠ Warning: CuPy initialization failed: {e}")
    print("  Falling back to CPU processing (NumPy only)")
    CUPY_AVAILABLE = False
    CUPY_AVAILABLE_VERIFIED = False

try:
    import pyopencl as cl

    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
except Exception as e:
    print(f"⚠ Warning: PyOpenCL initialization failed: {e}")
    PYOPENCL_AVAILABLE = False

# Plotting Environment
import veusz.embed as embed


# ============================================================================
# CONFIGURATION AND DATA CLASSES
# ============================================================================

@dataclass
class ProcessingConfig:
    """
    Configuration class for processing settings.
    """
    enable_multiprocessing: bool = False
    enable_gpu_processing: bool = False
    use_opencl: bool = True  # Prefer OpenCL for cross-platform compatibility
    num_processes: int = cpu_count()
    max_workers: int = cpu_count()
    chunk_size: int = 1000
    plot_mode: int = 0  # 0: all plots, 1: avg overlay only, 2: avg only
    enable_auto_save: bool = False
    plots_per_file: int = 1000
    auto_save_basename: str = 'RnS_Plots_Batch'


@dataclass
class plotDescInfo:
    """
    Setting up general plot info class to update as needed.
    """
    x_axis_label: str = ''
    y_axis_label: str = ''
    graph_notes: str = ''
    graph_title: str = ''
    basename: str = ''
    first_plot: bool = True


# ============================================================================
# GPU PROCESSING CLASSES
# ============================================================================

class GPUProcessor:
    """
    Handles GPU acceleration using either CuPy or PyOpenCL.
    Gracefully falls back to CPU if GPU unavailable.
    """

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
            self.initialize_gpu()

    def initialize_gpu(self):
        """
        Initialize GPU context based on available libraries.
        """
        if self.config.use_opencl and PYOPENCL_AVAILABLE:
            self.initialize_opencl()
        elif CUPY_AVAILABLE:
            self.initialize_cupy()
        else:
            print("⚠ No GPU libraries available. Falling back to CPU processing.")

    def initialize_opencl(self):
        """
        Initialize OpenCL context for cross-platform GPU support.
        """
        try:
            platforms = cl.get_platforms()
            if platforms:
                platform = platforms[0]
                devices = platform.get_devices(cl.device_type.GPU)
                if not devices:
                    devices = platform.get_devices(cl.device_type.CPU)
                if devices:
                    self.context = cl.Context(devices=[devices[0]])
                    self.queue = cl.CommandQueue(self.context)
                    self.gpu_available = True
                    print(f"✓ OpenCL initialized with device: {devices[0].name}")
        except Exception as e:
            print(f"⚠ OpenCL initialization failed: {e}")

    def initialize_cupy(self):
        """
        Initialize CuPy for NVIDIA GPU acceleration with proper CUDA detection.
        Handles CUDA availability check and provides helpful fallback messages.
        """
        try:
            if not CUPY_AVAILABLE:
                print("⚠ CuPy not available - skipping GPU initialization")
                self.gpu_available = False
                return

            # Verify CUDA device is accessible
            cp.cuda.Device(0).use()
            self.gpu_available = True
            print("✓ CuPy initialized successfully with CUDA device")

        except RuntimeError as cuda_runtime_error:
            # CUDA device not found or not accessible
            print(f"⚠ Warning: CUDA device not found or not accessible: {cuda_runtime_error}")
            print("  Falling back to CPU processing (NumPy only)")
            self.gpu_available = False

        except Exception as e:
            # General exception during initialization
            print(f"⚠ Warning: CuPy initialization failed: {e}")
            print("  Falling back to CPU processing (NumPy only)")
            self.gpu_available = False

    def process_array_gpu(self, data_array: np.ndarray) -> np.ndarray:
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
                return self.process_opencl(data_array)
            elif CUPY_AVAILABLE:
                return self.process_cupy(data_array)
        except Exception as e:
            print(f"⚠ GPU processing failed: {e}. Falling back to CPU.")
            return data_array

    def process_opencl(self, data_array: np.ndarray) -> np.ndarray:
        """Process data using OpenCL."""
        # Create OpenCL buffers
        mf = cl.mem_flags
        data_buffer = cl.Buffer(
            self.context,
            mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=data_array.astype(np.float32)
        )

        # Simple kernel for demonstration (can be replaced with more complex operations)
        kernel_source = """
            kernel void process_data(global float *data) {
                int gid = get_global_id(0);
                data[gid] = data[gid] * 1.0f;  // Identity operation for now
            }
        """

        try:
            program = cl.Program(self.context, kernel_source).build()
            kernel = program.process_data
            kernel(self.queue, len(data_array), None, data_buffer)

            result = np.empty_like(data_array, dtype=np.float32)
            cl.enqueue_copy(self.queue, result, data_buffer)
            return result.astype(data_array.dtype)
        except Exception as e:
            print(f"⚠ OpenCL processing error: {e}")
            return data_array

    def process_cupy(self, data_array: np.ndarray) -> np.ndarray:
        """Process data using CuPy."""
        try:
            gpu_array = cp.asarray(data_array)
            processed_gpu = gpu_array * 1.0  # Identity operation for now
            return cp.asnumpy(processed_gpu)
        except Exception as e:
            print(f"⚠ CuPy processing error: {e}")
            return data_array


# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================

def process_file_worker(file_info):
    """
    Worker function for processing individual SFT files in parallel.

    Parameters
    ----------
    file_info : tuple
        Tuple containing filename, search_strings, sft_lines, config.

    Returns
    -------
    dict
        Processed file data.
    """
    filename, search_strings, sft_lines, config = file_info

    try:
        # Parse the file
        data_returned = fparser(
            filename,
            line_targets=sft_lines,
            string_patterns=search_strings
        )

        # Initialize GPU processor if enabled
        if config.enable_gpu_processing:
            gpu_processor = GPUProcessor(config)
            for key, data_match in data_returned['data_matches'].items():
                if 'extracted_value' in data_match:
                    data_array = np.array(data_match['extracted_value'])
                    processed_array = gpu_processor.process_array_gpu(data_array)
                    data_returned['data_matches'][key]['extracted_value'] = processed_array.tolist()

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


def save_batch_worker(save_info):
    """
    Worker function for saving batches in parallel.

    Parameters
    ----------
    save_info : tuple
        Tuple containing doc, filename, mode.

    Returns
    -------
    dict
        Save operation result.
    """
    doc, filename, mode = save_info
    try:
        doc.Save(filename, mode=mode)
        return {
            'filename': filename,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'filename': filename,
            'success': False,
            'error': str(e)
        }


def extract_with_regex(input_text: str, delim: str):
    """
    Extract all substrings enclosed by the same delimiter using regex.

    Parameters
    ----------
    input_text : str
        Input text to search.
    delim : str
        Delimiter character.

    Returns
    -------
    list
        List of extracted strings.
    """
    esc = re.escape(delim)
    pattern = rf'{esc}(.*?){esc}'
    return re.findall(pattern, input_text)


# ============================================================================
# GUI THREAD CLASSES
# ============================================================================

class FileProcessingThread(QThread):
    """
    Thread for handling file processing without blocking the GUI.
    """

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
                results = self.process_files_parallel()
            else:
                results = self.process_files_sequential()
            self.processing_finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def process_files_parallel(self):
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
            for future in ascompleted(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1
                progress = int((completed / len(self.file_list)) * 100)
                self.progress_updated.emit(progress)

        return results

    def process_files_sequential(self):
        """Process files sequentially."""
        results = []
        for i, filename in enumerate(self.file_list):
            file_info = (filename, self.search_strings, self.sft_lines, self.config)
            result = process_file_worker(file_info)
            results.append(result)
            progress = int(((i + 1) / len(self.file_list)) * 100)
            self.progress_updated.emit(progress)
        return results


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class EnhancedMainWindow(QMainWindow):
    """
    Enhanced main window with modern Qt interface.
    """

    def __init__(self):
        """Initialize the enhanced main window."""
        super().__init__()
        self.setWindowTitle('Enhanced RS SFT File Plotter')
        self.setGeometry(100, 100, 800, 700)

        self.config = ProcessingConfig()
        self.vzplot = VZPlotRnS(self.config)
        self.selected_files = []
        self.processing_thread = None

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)

        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.clicked.connect(self.browse_files)
        browse_layout.addWidget(self.browse_button)

        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self.clear_files)
        browse_layout.addWidget(self.clear_button)

        file_layout.addLayout(browse_layout)
        main_layout.addWidget(file_group)

        # Processing options section
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)

        self.enable_mp_checkbox = QCheckBox("Enable Multiprocessing")
        self.enable_mp_checkbox.setChecked(self.config.enable_multiprocessing)
        self.enable_mp_checkbox.stateChanged.connect(self.update_mp_config)
        options_layout.addWidget(self.enable_mp_checkbox)

        self.enable_gpu_checkbox = QCheckBox("Enable GPU Processing")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.stateChanged.connect(self.update_gpu_config)
        if not (CUPY_AVAILABLE or PYOPENCL_AVAILABLE):
            self.enable_gpu_checkbox.setEnabled(False)
        options_layout.addWidget(self.enable_gpu_checkbox)

        main_layout.addWidget(options_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        main_layout.addWidget(self.status_text)

        # Control buttons
        button_layout = QHBoxLayout()

        self.plot_button = QPushButton("Process and Plot")
        self.plot_button.clicked.connect(self.process_and_plot)
        button_layout.addWidget(self.plot_button)

        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def browse_files(self):
        """Open file dialog to select multiple SFT files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("RS SFT Files (*.sft)")
        file_dialog.setWindowTitle("Select SFT Files")

        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self.update_file_list()
            self.log_message(f"Selected {len(selected_files)} files")

    def clear_files(self):
        """Clear the selected files list."""
        self.selected_files.clear()
        self.update_file_list()
        self.log_message("File list cleared")

    def update_file_list(self):
        """Update the file list widget."""
        self.file_list_widget.clear()
        for file_path in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(file_path))

    def update_mp_config(self, state):
        """Update multiprocessing configuration."""
        self.config.enable_multiprocessing = state == Qt.Checked
        self.log_message(
            f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}"
        )

    def update_gpu_config(self, state):
        """Update GPU processing configuration."""
        self.config.enable_gpu_processing = state == Qt.Checked
        self.log_message(
            f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}"
        )

    def process_and_plot(self):
        """Process selected files and create plots."""
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "Please select SFT files first.")
            return

        self.plot_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.processing_thread = FileProcessingThread(
            self.selected_files,
            self.config,
            self.vzplot.search_data_strings,
            self.vzplot.sft_lines
        )

        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.start()

        self.log_message("Processing started...")

    def on_processing_finished(self, results):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        self.log_message(f"Processing completed: {len(successful_results)} successful, {len(failed_results)} failed")

        if failed_results:
            error_msg = '\n'.join(
                f"{r['filename']}: {r['error']}"
                for r in failed_results[:5]  # Show first 5 errors
            )
            if len(failed_results) > 5:
                error_msg += f"\n... and {len(failed_results) - 5} more errors"
            QMessageBox.warning(self, "Processing Errors", f"Some files failed to process:\n{error_msg}")

        if successful_results:
            self.create_plots(successful_results)
            self.save_button.setEnabled(True)

    def on_processing_error(self, error_message):
        """Handle processing error."""
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)
        self.log_message(f"Processing error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)

    def create_plots(self, results):
        """Create Veusz plots from processed results."""
        self.log_message("Creating plots...")
        for result in results:
            filename = result['filename']
            data_returned = result['data']
            try:
                self.vzplot.process_file_data(filename, data_returned)
            except Exception as e:
                self.log_message(f"Plot creation failed for {os.path.basename(filename)}: {e}")
        self.log_message("Plot creation completed")

    def save_project(self):
        """Save Veusz project."""
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Veusz Project",
            "",
            "Veusz High Precision Files (*.vszh5)"
        )

        if save_path:
            try:
                self.vzplot.save(save_path)
                self.log_message(f"Project saved: {save_path}")

                reply = QMessageBox.question(
                    self,
                    "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    VZPlotRnS.open_veusz_gui(save_path)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{e}")

    def log_message(self, message):
        """Add message to status text."""
        self.status_text.append(f"{self.get_timestamp()}: {message}")
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def get_timestamp(self):
        """Get current timestamp string."""
        return datetime.datetime.now().strftime("%H:%M:%S")


# ============================================================================
# VEUSZ PLOTTING CLASS
# ============================================================================

class VZPlotRnS:
    """
    Enhanced Veusz plotting class with multiprocessing support.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize VZPlotRnS with enhanced capabilities.

        Parameters
        ----------
        config : ProcessingConfig
            Configuration object containing processing settings.
        """
        self.config = config
        self.doc = embed.Embedded('Enhanced RS SFT File Plotter')
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)

        self.search_data_strings = {
            'version': 'VERSION',
            'type': 'TYPE',
            'mode': 'MODE',
            'center_freq': 'CENTER FREQ',
            'freq_offset': 'FREQ OFFSET',
            'span': 'SPAN',
            'x-axis': 'X-AXIS',
            'start': 'START',
            'stop': 'STOP',
            'stop2': 'STOP',  # Added for compatibility
            'ref_level': 'REF LEVEL',
            'level_offset': 'LEVEL OFFSET',
            'ref_position': 'REF POSITION',
            'y-axis': 'Y-AXIS',
            'level_range': 'LEVEL RANGE',
            'rf_att': 'RF ATT',
            'rbw': 'RBW',
            'vbw': 'VBW',
            'swt': 'SWT',
            'trace_mode': 'TRACE MODE',
            'detector': 'DETECTOR',
            'sweep_count': 'SWEEP COUNT',
            'trace': 'TRACE',
            'x-unit': 'X-UNIT',
            'y-unit': 'Y-UNIT',
            'preamplifier': 'PREAMPLIFIER',
            'transducer': 'TRANSDUCER',
            'values': 'VALUES',
            'section': 'SECTION'
        }

        self.sft_lines = [
            'VERSION', 'TYPE', 'MODE', 'CENTER FREQ', 'FREQ OFFSET', 'SPAN',
            'X-AXIS', 'START', 'STOP', 'REF LEVEL', 'LEVEL OFFSET', 'REF POSITION',
            'Y-AXIS', 'LEVEL RANGE', 'RF ATT', 'RBW', 'VBW', 'SWT', 'TRACE MODE',
            'DETECTOR', 'SWEEP COUNT', 'TRACE', 'X-UNIT', 'Y-UNIT', 'PREAMPLIFIER',
            'TRANSDUCER', 'VALUES', 'SECTION'
        ]

        self.plot_info = plotDescInfo()

    def process_file_data(self, filename: str, data_returned: dict):
        """
        Process individual file data and create plots.

        Parameters
        ----------
        filename : str
            Path to the processed file.
        data_returned : dict
            Parsed file data.
        """
        basename = os.path.splitext(os.path.basename(filename))[0]
        self.plot_info.basename = basename

        # Extract key values
        try:
            # Get data values
            data_y_values = list(
                map(itemgetter('extracted_value'), data_returned['data_matches'].values())
            )

            # Create frequency range
            num_pts = len(data_y_values[0]) if data_y_values else 0

            if num_pts > 0:
                # Create plot
                plot_name = f'plot_{basename}_{self.doc.getchildcount() + 1}'
                self.doc.Add('xy', name=plot_name)

                # Add data
                for i, y_data in enumerate(data_y_values):
                    x_data = np.linspace(0, 1, num_pts)
                    self.doc.AddDataset(
                        f'x_{i}',
                        data=x_data,
                        linked=False
                    )
                    self.doc.AddDataset(
                        f'y_{i}',
                        data=y_data,
                        linked=False
                    )

        except Exception as e:
            print(f"Error processing file data: {e}")

    def save(self, filename: str):
        """Save Veusz project to file."""
        self.doc.Save(filename, mode='hdf5')

    @staticmethod
    def open_veusz_gui(filepath):
        """Open Veusz GUI with the saved file."""
        try:
            subprocess.Popen(['veusz', filepath])
        except Exception as e:
            print(f"Error opening Veusz: {e}")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = EnhancedMainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
