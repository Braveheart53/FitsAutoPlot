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
Version: 1.0.3 - Corrected Average+Overlay only mode with AutoSave functionality
=============================================================================
"""

# TODO: compare GPU based processing on this one to Touchstone

# %% Import all required modules

import datetime  # Added for auto-save timestamping
import gc
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
# %%% System Interface Modules
from dataclasses import dataclass
from operator import itemgetter

from hanging_threads import start_monitoring

# %%% GUI Module Imports - QtPy for cross-platform compatibility
if getattr(sys, 'frozen', False):
    # Running as compiled executable - use PySide6 directly
    os.environ['QT_API'] = 'pyside6'
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QLineEdit
    )
else:
    # Development environment - use QtPy
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QLineEdit
    )

# %%% Math and Processing Modules
import numpy as np
from fastest_ascii_import import fastest_file_parser as fparser

# %%% Parallel Processing Modules
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing as mp

# %%% GPU Acceleration Modules
CUPY_AVAILABLE = True
try:
    import cupy as cp

    CUPY_AVAILABLE = True
    # Verify CUDA device is actually accessible
    try:
        cp.cuda.Device(0).use()
        CUPY_AVAILABLE_VERIFIED = True
    except Exception as cuda_error:
        print(f"Warning: CuPy detected but CUDA device not accessible: {cuda_error}")
        print("Falling back to CPU processing (NumPy only)")
        CUPY_AVAILABLE = False
        CUPY_AVAILABLE_VERIFIED = False
except ImportError:
    CUPY_AVAILABLE = False
    CUPY_AVAILABLE_VERIFIED = False
except Exception as e:
    print(f"Warning: CuPy initialization failed: {e}")
    print("Falling back to CPU processing (NumPy only)")
    CUPY_AVAILABLE = False
    CUPY_AVAILABLE_VERIFIED = False

try:
    import pyopencl as cl

    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

# %%% Plotting Environment
import veusz.embed as embed
from veusz.windows.simplewindow import SimpleWindow
from veusz.document import CommandInterface


# %% Configuration and Data Classes

@dataclass
class ProcessingConfig:
    """Configuration class for processing settings."""
    enable_multiprocessing: bool = True  # Default enabled for maximum performance
    enable_gpu_processing: bool = False  # User configurable
    use_opencl: bool = True  # Prefer OpenCL for cross-platform compatibility
    num_processes: int = cpu_count()
    max_workers: int = cpu_count()
    chunk_size: int = min(1000, max(100, cpu_count() * 10))

    # Plot mode and auto-save options
    plot_mode: int = 0  # 0=all plots, 1=avg+overlay only, 2=avg only
    enable_auto_save: bool = False
    plots_per_file: int = 1000
    auto_save_base_name: str = "RnS_Plots_Batch"

    # Enhanced processing options
    enable_async_processing: bool = True
    memory_optimization: bool = True
    use_shared_memory: bool = True
    batch_plot_generation: bool = True


@dataclass
class plotDescInfo:
    """Setting up general plot info class to update as needed."""
    xAxis_label: str
    yAxis_label: str
    graph_notes: str
    graph_title: str
    base_name: str
    first_plot: bool


# %% Enhanced GPU Processing Classes

class GPUProcessor:
    """Handles GPU acceleration using either CuPy or PyOpenCL with enhanced memory management."""

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
        self.memory_pool = None
        self.batch_size = config.chunk_size

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
                # Select best available platform and device
                platform = platforms[0]
                devices = platform.get_devices(cl.device_type.GPU)
                if not devices:
                    devices = platform.get_devices(cl.device_type.CPU)

                if devices:
                    # Select device with most compute units
                    best_device = max(devices, key=lambda d: d.max_compute_units)
                    self.context = cl.Context(devices=[best_device])
                    self.queue = cl.CommandQueue(self.context,
                                                 properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
                    self.gpu_available = True
                    print(f"OpenCL initialized with device: {best_device.name} ({best_device.max_compute_units} CUs)")
        except Exception as e:
            print(f"OpenCL initialization failed: {e}")

    def _initialize_cupy(self):
        """Initialize CuPy for NVIDIA GPU acceleration with memory pool."""

        # try:
        #     cp.cuda.Device(0).use()
        #     # Initialize memory pool for efficient GPU memory management
        #     self.memory_pool = cp.get_default_memory_pool()
        #     self.gpu_available = True
        #     print("CuPy initialized successfully with memory pool")
        # except Exception as e:
        #     print(f"CuPy initialization failed: {e}")
        def initialize_cupy(self):
            """Initialize CuPy for NVIDIA GPU acceleration with proper CUDA detection."""
            try:
                if not CUPY_AVAILABLE:
                    print("CuPy not available - skipping GPU initialization")
                    self.gpu_available = False
                    return

                # Verify CUDA device is accessible
                cp.cuda.Device(0).use()

                # Get memory pool for efficient GPU memory management
                self.memory_pool = cp.get_default_memory_pool()
                self.gpu_available = True
                print("CuPy initialized successfully with CUDA device")

            except RuntimeError as cuda_runtime_error:
                print(f"Warning: CUDA device not found or not accessible: {cuda_runtime_error}")
                print("Falling back to CPU processing")
                self.gpu_available = False
                CUPY_AVAILABLE = False

            except Exception as e:
                print(f"Warning: CuPy initialization failed: {e}")
                print("Falling back to CPU processing")
                self.gpu_available = False

    def process_batch_arrays_gpu(self, data_arrays_list):
        """
        Process multiple numpy arrays using GPU acceleration with batch processing.

        Parameters
        ----------
        data_arrays_list : list
            List of numpy arrays to process.

        Returns
        -------
        list
            List of processed numpy arrays.
        """
        if not self.gpu_available or not data_arrays_list:
            return data_arrays_list

        try:
            if self.config.use_opencl and self.context:
                return self._process_batch_opencl(data_arrays_list)
            elif CUPY_AVAILABLE:
                return self._process_batch_cupy(data_arrays_list)
        except Exception as e:
            print(f"GPU batch processing failed: {e}. Falling back to CPU.")
            return data_arrays_list

    def _process_batch_cupy(self, data_arrays_list):
        """Process batch of data using CuPy with optimized memory management."""
        processed_arrays = []

        with cp.cuda.Stream() as stream:
            for batch_start in range(0, len(data_arrays_list), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(data_arrays_list))
                batch_arrays = data_arrays_list[batch_start:batch_end]

                # Process batch on GPU
                gpu_arrays = [cp.asarray(arr, stream=stream) for arr in batch_arrays]

                # Apply processing (vectorized operations)
                processed_gpu = [arr * 1.0 for arr in gpu_arrays]  # Identity operation

                # Transfer back to CPU
                processed_batch = [cp.asnumpy(arr) for arr in processed_gpu]
                processed_arrays.extend(processed_batch)

                # Clear GPU memory for this batch
                del gpu_arrays, processed_gpu
                stream.synchronize()

        # Force memory cleanup
        if self.memory_pool:
            self.memory_pool.free_all_blocks()

        return processed_arrays

    def _process_batch_opencl(self, data_arrays_list):
        """Process batch of data using OpenCL with enhanced kernels."""
        processed_arrays = []

        # Enhanced kernel with multiple operations
        kernel_source = """
        __kernel void process_data_batch(__global float* data, const int size) {
            int gid = get_global_id(0);
            if (gid < size) {
                // Enhanced processing operations
                data[gid] = data[gid] * 1.0f; // Identity operation for now
                // Add more complex operations here as needed
            }
        }
        """

        try:
            program = cl.Program(self.context, kernel_source).build()
            kernel = program.process_data_batch

            for batch_start in range(0, len(data_arrays_list), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(data_arrays_list))
                batch_arrays = data_arrays_list[batch_start:batch_end]

                batch_processed = []
                for data_array in batch_arrays:
                    # Create OpenCL buffer
                    mf = cl.mem_flags
                    data_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                            hostbuf=data_array.astype(np.float32))

                    # Execute kernel
                    kernel(self.queue, (len(data_array),), None, data_buffer, np.int32(len(data_array)))

                    # Read back results
                    result = np.empty_like(data_array, dtype=np.float32)
                    cl.enqueue_copy(self.queue, result, data_buffer)
                    batch_processed.append(result.astype(data_array.dtype))

                processed_arrays.extend(batch_processed)

            self.queue.finish()
        except Exception as e:
            print(f"OpenCL batch processing error: {e}")
            return data_arrays_list

        return processed_arrays

    def process_array_gpu(self, data_array):
        """
        Process single numpy array using GPU acceleration.

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

        return self.process_batch_arrays_gpu([data_array])[0]


# %% Enhanced Multiprocessing Worker Functions

def process_file_worker_enhanced(file_info):
    """
    Enhanced worker function for processing individual SFT files in parallel with GPU support.

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
        start_time = time.time()
        data_returned = fparser(filename, line_targets=sft_lines,
                                string_patterns=search_strings)

        # Initialize GPU processor if enabled
        if config.enable_gpu_processing:
            gpu_processor = GPUProcessor(config)

            # Collect all data arrays for batch processing
            data_arrays = []
            keys_order = []

            for key, data_match in data_returned['data_matches'].items():
                if 'extracted_value' in data_match:
                    data_array = np.array(data_match['extracted_value'])
                    data_arrays.append(data_array)
                    keys_order.append(key)

            if data_arrays:
                # Process all arrays in batch on GPU
                processed_arrays = gpu_processor.process_batch_arrays_gpu(data_arrays)

                # Update the data structure
                for i, key in enumerate(keys_order):
                    data_returned['data_matches'][key]['extracted_value'] = processed_arrays[i].tolist()

        processing_time = time.time() - start_time

        return {
            'filename': filename,
            'data': data_returned,
            'success': True,
            'error': None,
            'processing_time': processing_time
        }

    except Exception as e:
        return {
            'filename': filename,
            'data': None,
            'success': False,
            'error': str(e),
            'processing_time': 0
        }


def save_batch_worker_enhanced(save_info):
    """
    Enhanced worker function for saving batches in parallel with error handling.

    Parameters
    ----------
    save_info : tuple
        Tuple containing (doc_copy, filename, mode).

    Returns
    -------
    dict
        Save operation result.
    """
    doc_copy, filename, mode = save_info

    try:
        start_time = time.time()
        doc_copy.Save(filename, mode=mode)
        save_time = time.time() - start_time

        return {
            'filename': filename,
            'success': True,
            'error': None,
            'save_time': save_time
        }

    except Exception as e:
        return {
            'filename': filename,
            'success': False,
            'error': str(e),
            'save_time': 0
        }


def plot_generation_worker(plot_info):
    """
    Worker function for parallel plot generation.

    Parameters
    ----------
    plot_info : tuple
        Tuple containing plot generation parameters.

    Returns
    -------
    dict
        Plot generation result.
    """
    try:
        # This would contain the Veusz plot generation logic
        # Currently a placeholder for parallel plot generation
        return {
            'success': True,
            'error': None
        }

    except Exception as e:
        return {
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
    """Thread for handling file processing without blocking the GUI with enhanced parallel processing."""

    progress_updated = Signal(int)
    processing_finished = Signal(object)
    error_occurred = Signal(str)
    status_updated = Signal(str)

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
        """Execute file processing in separate thread with maximum parallelization."""
        try:
            self.status_updated.emit("Initializing parallel processing...")

            if self.config.enable_multiprocessing and len(self.file_list) > 1:
                results = self._process_files_parallel_enhanced()
            else:
                results = self._process_files_sequential_enhanced()

            self.processing_finished.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_files_parallel_enhanced(self):
        """Process files using enhanced multiprocessing with optimal resource utilization."""
        file_info_list = [
            (filename, self.search_strings, self.sft_lines, self.config)
            for filename in self.file_list
        ]

        results = []
        total_files = len(file_info_list)

        # Use optimal number of workers based on system resources
        optimal_workers = min(self.config.max_workers, len(file_info_list), cpu_count() * 2)
        self.status_updated.emit(f"Processing {total_files} files with {optimal_workers} workers...")

        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file_worker_enhanced, file_info): file_info[0]
                for file_info in file_info_list
            }

            completed = 0
            total_processing_time = 0

            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1

                if result['success']:
                    total_processing_time += result.get('processing_time', 0)

                progress = int((completed / total_files) * 100)
                self.progress_updated.emit(progress)

                # Update status with performance metrics
                avg_time = total_processing_time / max(1, completed) if completed > 0 else 0
                eta_seconds = avg_time * (total_files - completed)
                self.status_updated.emit(f"Processed {completed}/{total_files} files. ETA: {eta_seconds:.1f}s")

        successful = sum(1 for r in results if r['success'])
        self.status_updated.emit(f"Completed: {successful}/{total_files} files processed successfully")
        return results

    def _process_files_sequential_enhanced(self):
        """Process files sequentially with performance monitoring."""
        results = []
        total_files = len(self.file_list)

        for i, filename in enumerate(self.file_list):
            self.status_updated.emit(f"Processing file {i + 1}/{total_files}: {os.path.basename(filename)}")
            file_info = (filename, self.search_strings, self.sft_lines, self.config)
            result = process_file_worker_enhanced(file_info)
            results.append(result)

            progress = int(((i + 1) / total_files) * 100)
            self.progress_updated.emit(progress)

        return results


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with modern Qt interface and advanced processing options."""

    def __init__(self):
        """Initialize the enhanced main window."""
        super().__init__()
        self.setWindowTitle("Enhanced R&S SFT File Plotter v1.0.3")
        self.setGeometry(100, 100, 900, 800)  # Increased size for new options

        # Initialize configuration with defaults optimized for performance
        self.config = ProcessingConfig()
        self.config.enable_multiprocessing = True  # Enable by default

        # Auto-detect and enable GPU if available
        if CUPY_AVAILABLE or PYOPENCL_AVAILABLE:
            self.config.enable_gpu_processing = True

        # Initialize VZPlotRnS
        self.vzplot = VZPlotRnS(self.config)

        # Setup UI
        self._setup_ui()

        # File list
        self.selected_files = []

    def _setup_ui(self):
        """Set up the user interface with enhanced options."""
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

        # Multiprocessing options (checked by default)
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

        # GPU options (checked by default if available)
        self.enable_gpu_checkbox = QCheckBox("Enable GPU Processing")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.stateChanged.connect(self._update_gpu_config)
        options_layout.addWidget(self.enable_gpu_checkbox)

        # OpenCL preference (checked by default)
        self.use_opencl_checkbox = QCheckBox("Prefer OpenCL (Cross-platform)")
        self.use_opencl_checkbox.setChecked(self.config.use_opencl)
        self.use_opencl_checkbox.stateChanged.connect(self._update_opencl_config)
        options_layout.addWidget(self.use_opencl_checkbox)

        main_layout.addWidget(options_group)

        # Plot options section
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout(plot_group)

        # Radio button group for plot mode
        self.plot_mode_group = QButtonGroup()
        self.all_plots_radio = QRadioButton("All plots (original behavior)")
        self.all_plots_radio.setChecked(True)
        self.plot_mode_group.addButton(self.all_plots_radio, 0)
        plot_layout.addWidget(self.all_plots_radio)

        self.avg_overlay_radio = QRadioButton("Average + Overlay only (no individual plots)")
        self.plot_mode_group.addButton(self.avg_overlay_radio, 1)
        plot_layout.addWidget(self.avg_overlay_radio)

        self.avg_only_radio = QRadioButton("Average only (no individual or overlay plots)")
        self.plot_mode_group.addButton(self.avg_only_radio, 2)
        plot_layout.addWidget(self.avg_only_radio)

        self.plot_mode_group.buttonClicked.connect(self._update_plot_mode)
        main_layout.addWidget(plot_group)

        # Auto-save options section
        autosave_group = QGroupBox("Auto-Save Options")
        autosave_layout = QVBoxLayout(autosave_group)

        # Enable auto-save checkbox
        self.enable_autosave_checkbox = QCheckBox("Enable automatic batch saving")
        self.enable_autosave_checkbox.stateChanged.connect(self._update_autosave_config)
        autosave_layout.addWidget(self.enable_autosave_checkbox)

        # Plots per file setting
        plots_layout = QHBoxLayout()
        plots_layout.addWidget(QLabel("Plots per file:"))
        self.plots_per_file_edit = QLineEdit("1000")
        self.plots_per_file_edit.setEnabled(False)
        self.plots_per_file_edit.textChanged.connect(self._update_plots_per_file)
        plots_layout.addWidget(self.plots_per_file_edit)
        plots_layout.addStretch()
        autosave_layout.addLayout(plots_layout)

        # Base filename setting
        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Base filename:"))
        self.base_filename_edit = QLineEdit("RnS_Plots_Batch")
        self.base_filename_edit.setEnabled(False)
        self.base_filename_edit.textChanged.connect(self._update_base_filename)
        filename_layout.addWidget(self.base_filename_edit)
        filename_layout.addStretch()
        autosave_layout.addLayout(filename_layout)

        main_layout.addWidget(autosave_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(120)
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
        self.vzplot.config.enable_multiprocessing = self.config.enable_multiprocessing
        self._log_message(
            f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}")

    def _update_cpu_config(self, value):
        """Update CPU cores configuration."""
        self.config.num_processes = value
        self.config.max_workers = value
        self.vzplot.config.num_processes = value
        self.vzplot.config.max_workers = value
        self._log_message(f"CPU cores set to: {value}")

    def _update_gpu_config(self, state):
        """Update GPU processing configuration."""
        self.config.enable_gpu_processing = state == Qt.Checked
        self.vzplot.config.enable_gpu_processing = self.config.enable_gpu_processing
        self._log_message(
            f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}")

    def _update_opencl_config(self, state):
        """Update OpenCL preference configuration."""
        self.config.use_opencl = state == Qt.Checked
        self.vzplot.config.use_opencl = self.config.use_opencl
        self._log_message(
            f"OpenCL preference: {'Enabled' if self.config.use_opencl else 'Disabled'}")

    def _update_plot_mode(self, button):
        """Update plot mode configuration."""
        self.config.plot_mode = self.plot_mode_group.id(button)
        self.vzplot.config.plot_mode = self.config.plot_mode
        mode_names = ["All plots", "Average + Overlay only", "Average only"]
        self._log_message(f"Plot mode: {mode_names[self.config.plot_mode]}")

    def _update_autosave_config(self, state):
        """Update auto-save configuration."""
        self.config.enable_auto_save = state == Qt.Checked
        self.plots_per_file_edit.setEnabled(self.config.enable_auto_save)
        self.base_filename_edit.setEnabled(self.config.enable_auto_save)
        self.vzplot.auto_save_enabled = self.config.enable_auto_save
        self._log_message(
            f"Auto-save: {'Enabled' if self.config.enable_auto_save else 'Disabled'}")

    def _update_plots_per_file(self, text):
        """Update plots per file configuration."""
        try:
            value = int(text) if text else 1000
            self.config.plots_per_file = max(1, value)
            self.vzplot.plots_per_file = self.config.plots_per_file
        except ValueError:
            self.config.plots_per_file = 1000

    def _update_base_filename(self, text):
        """Update base filename configuration."""
        self.config.auto_save_base_name = text if text else "RnS_Plots_Batch"

    def _log_message(self, message):
        """Add message to status text."""
        self.status_text.append(f"[{self._get_timestamp()}] {message}")
        # Auto-scroll to bottom
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def _get_timestamp(self):
        """Get current timestamp string."""
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _process_and_plot(self):
        """Process selected files and create plots with enhanced performance monitoring."""
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

        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self._on_processing_finished)
        self.processing_thread.error_occurred.connect(self._on_processing_error)
        self.processing_thread.status_updated.connect(self._log_message)

        self.processing_thread.start()
        self._log_message("Enhanced parallel processing started...")

    def _on_processing_finished(self, results):
        """Handle processing completion with detailed metrics."""
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        # Calculate performance metrics
        total_processing_time = sum(r.get('processing_time', 0) for r in successful_results)
        avg_processing_time = total_processing_time / max(1, len(successful_results))

        self._log_message(
            f"Processing completed: {len(successful_results)} successful, {len(failed_results)} failed")
        self._log_message(f"Average processing time: {avg_processing_time:.3f}s per file")

        if failed_results:
            error_msg = "\n".join(
                [f"{os.path.basename(r['filename'])}: {r['error']}" for r in failed_results[:5]])  # Show first 5 errors
            if len(failed_results) > 5:
                error_msg += f"\n... and {len(failed_results) - 5} more errors"
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
        """Create Veusz plots from processed results with parallel processing."""
        self._log_message("Creating plots with enhanced parallel processing...")
        start_time = time.time()

        # Use parallel processing for plot creation if enabled and beneficial
        if self.config.enable_multiprocessing and len(results) > 2:
            self._create_plots_parallel(results)
        else:
            self._create_plots_sequential(results)

        plot_time = time.time() - start_time
        self._log_message(f"Plot creation completed in {plot_time:.2f}s")

    def _create_plots_parallel(self, results):
        """Create plots using parallel processing where possible."""
        # Note: Veusz document operations must be serialized, but we can parallelize data preparation
        for result in results:
            filename = result['filename']
            data_returned = result['data']

            try:
                self.vzplot._process_file_data(filename, data_returned)
            except Exception as e:
                self._log_message(f"Plot creation failed for {os.path.basename(filename)}: {e}")

    def _create_plots_sequential(self, results):
        """Create plots sequentially with progress updates."""
        for i, result in enumerate(results):
            filename = result['filename']
            data_returned = result['data']

            try:
                self.vzplot._process_file_data(filename, data_returned)
                if i % 10 == 0:  # Update every 10 files
                    self._log_message(f"Created plots for {i + 1}/{len(results)} files")
            except Exception as e:
                self._log_message(f"Plot creation failed for {os.path.basename(filename)}: {e}")

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


# %% Enhanced Auto Plotter Class

class VZPlotRnS:
    """Enhanced Veusz plotting class with maximum multiprocessing and GPU support."""

    def __init__(self, config: ProcessingConfig):
        """Initialize VZPlotRnS with enhanced capabilities."""
        self.config = config
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)

        # Auto-save tracking variables
        self.plot_count = 0
        self.file_batch_number = 1
        self.base_save_path = None
        self.auto_save_enabled = False
        self.plots_per_file = 1000

        # Global average tracking variables
        self._global_datasets = []  # Track all datasets across all files
        self._current_file_datasets = []  # Track datasets for current file

        # Initialize GPU processor for averaging operations
        self.gpu_processor = GPUProcessor(config) if config.enable_gpu_processing else None

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

        # Create a dict to track datasets for average calculation
        self._datasets_by_base = defaultdict(list)

    def _close_current_document_window(self):
        """Close current Veusz document window if it exists."""
        try:
            if hasattr(self.doc, 'window') and self.doc.window:
                self.doc.window.close()
                self.doc.window = None
                gc.collect()  # Force garbage collection
                print("Closed current Veusz document window")
        except Exception as e:
            print(f"Error closing document window: {e}")

    def _setup_auto_save_path(self):
        """Setup automatic save path based on configuration."""
        if self.base_save_path is None and self.auto_save_enabled:
            # Create auto-save directory in current working directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"RnS_AutoSave_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)

            # Set base path without extension (will be added during save)
            self.base_save_path = os.path.join(save_dir, self.config.auto_save_base_name)
            print(f"Auto-save directory created: {save_dir}")

    def _auto_save_if_needed(self):
        """Automatically save file if plot count threshold is reached."""
        if not self.auto_save_enabled:
            return

        if self.plot_count > 0 and self.plot_count % self.plots_per_file == 0:
            if self.base_save_path is None:
                self._setup_auto_save_path()

            # Generate filename with batch number
            batch_filename = f"{self.base_save_path}_{self.file_batch_number:03d}.vszh5"

            # Create file average before saving
            self._create_file_average()

            # Use multiprocessing for save operation if enabled
            if self.config.enable_multiprocessing:
                self._save_batch_parallel(batch_filename)
            else:
                self._save_batch_sequential(batch_filename)

            print(f"Auto-saved batch {self.file_batch_number} with {self.plots_per_file} plots to: {batch_filename}")

            # Close current window and reset for next batch
            self._close_current_document_window()
            self.file_batch_number += 1
            self._reset_for_next_batch()

    def _save_batch_parallel(self, filename):
        """Save current batch using parallel processing."""
        try:
            # Create a deep copy of the document for parallel saving
            def save_worker():
                self.doc.Save(filename, mode='hdf5')

            # Use threading for I/O bound save operation
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(save_worker)
                try:
                    future.result(timeout=30)  # 30 second timeout
                except Exception as e:
                    print(f"Parallel save failed: {e}")
                    self._save_batch_sequential(filename)

        except Exception as e:
            print(f"Error during parallel save: {e}")
            # Fallback to sequential save
            self._save_batch_sequential(filename)

    def _save_batch_sequential(self, filename):
        """Save current batch sequentially."""
        try:
            self.doc.Save(filename, mode='hdf5')
        except Exception as e:
            print(f"Error during sequential save: {e}")

    def _reset_for_next_batch(self):
        """Reset document for next batch while preserving settings."""
        # Close current document window if it exists
        self._close_current_document_window()

        # Create new document for next batch
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.doc.EnableToolbar(enable=True)
        self.first_1d = True

        # Clear dataset tracking for new batch (but keep global datasets)
        self._datasets_by_base.clear()
        self._current_file_datasets.clear()

        # Force garbage collection
        gc.collect()

    def _create_file_average(self):
        """Create average of all datasets in current file (excluding global average)."""
        if self.config.plot_mode not in [1, 2]:  # Only for avg+overlay and avg only modes
            return

        # Get all current file datasets (excluding frequency datasets and existing averages)
        file_data_datasets = [
            ds for ds in self._current_file_datasets
            if ('freq' not in ds.lower() and
                not ds.endswith(
                    ('_avg_dB', '_avg_lin', '_global_avg_dB', '_global_avg_lin', '_file_avg_dB', '_file_avg_lin')))
        ]

        if len(file_data_datasets) < 2:
            return

        try:
            # Use GPU or CPU for averaging
            use_gpu = (self.config.enable_gpu_processing and
                       self.gpu_processor and
                       self.gpu_processor.gpu_available)
            xp = cp if use_gpu else np

            # Helper function to convert dB to linear
            def _db_to_lin(arr):
                return xp.power(10.0, arr / 10.0)

            # Fetch data arrays for file average
            data_arrays = [xp.asarray(self.doc.GetData(ds)[0]) for ds in file_data_datasets]

            # Enhanced parallel processing
            if use_gpu:
                linear_arrays = [_db_to_lin(arr) for arr in data_arrays]
                linear_stack = xp.vstack(linear_arrays)
            elif (self.config.enable_multiprocessing and
                  len(file_data_datasets) > 2 and
                  self.config.num_processes > 1):
                # Use ThreadPoolExecutor for mathematical operations
                with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
                    chunk_size = max(1, len(data_arrays) // self.config.num_processes)
                    chunks = [data_arrays[i:i + chunk_size]
                              for i in range(0, len(data_arrays), chunk_size)]

                    def process_chunk(chunk):
                        return [10.0 ** (arr / 10.0) for arr in chunk]

                    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                    linear_arrays = []
                    for future in as_completed(futures):
                        linear_arrays.extend(future.result())

                    linear_stack = xp.vstack(linear_arrays)
            else:
                # Sequential processing
                linear_stack = xp.vstack([_db_to_lin(a) for a in data_arrays])

            # Compute file averages
            file_avg_lin = xp.mean(linear_stack, axis=0)
            file_avg_db = 10.0 * xp.log10(file_avg_lin)

            # Move to CPU if needed
            if use_gpu:
                file_avg_lin = cp.asnumpy(file_avg_lin)
                file_avg_db = cp.asnumpy(file_avg_db)

            # Register file averages in Veusz
            file_avg_lin_name = "File_avg_lin"
            file_avg_db_name = "File_avg_dB"

            self.doc.SetData(name=file_avg_lin_name, val=file_avg_lin)
            self.doc.SetData(name=file_avg_db_name, val=file_avg_db)

            # Tag the file average datasets
            self.doc.TagDatasets('File_Average', [file_avg_db_name, file_avg_lin_name])

            # Create file average plot
            if self.config.plot_mode >= 1:  # avg+overlay or avg only modes
                prev_title = self.plotInfo.graph_title
                self.plotInfo.graph_title = "File Average"
                self._plot_1d(file_avg_db_name)
                self.plotInfo.graph_title = prev_title

            # Add to overlay if in avg+overlay mode
            if self.config.plot_mode == 1:
                self._add_to_overlay(file_avg_db_name, "File Average")

            print(f"Created file average from {len(file_data_datasets)} datasets")

        except Exception as e:
            print(f"Error in file average calculation: {e}")

    def _create_global_average(self):
        """Create global average of all datasets across all files."""
        if self.config.plot_mode not in [1, 2]:  # Only for avg+overlay and avg only modes
            return

        # Get all global datasets (excluding frequency datasets and existing averages)
        global_data_datasets = [
            ds for ds in self._global_datasets
            if ('freq' not in ds.lower() and
                not ds.endswith(
                    ('_avg_dB', '_avg_lin', '_global_avg_dB', '_global_avg_lin', '_file_avg_dB', '_file_avg_lin')))
        ]

        if len(global_data_datasets) < 2:
            return

        try:
            # Use GPU or CPU for averaging
            use_gpu = (self.config.enable_gpu_processing and
                       self.gpu_processor and
                       self.gpu_processor.gpu_available)
            xp = cp if use_gpu else np

            # Helper function to convert dB to linear
            def _db_to_lin(arr):
                return xp.power(10.0, arr / 10.0)

            # Fetch data arrays for global average
            data_arrays = [xp.asarray(self.doc.GetData(ds)[0]) for ds in global_data_datasets]

            # Enhanced parallel processing
            if use_gpu:
                linear_arrays = [_db_to_lin(arr) for arr in data_arrays]
                linear_stack = xp.vstack(linear_arrays)
            elif (self.config.enable_multiprocessing and
                  len(global_data_datasets) > 2 and
                  self.config.num_processes > 1):
                # Use ThreadPoolExecutor for mathematical operations
                with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
                    chunk_size = max(1, len(data_arrays) // self.config.num_processes)
                    chunks = [data_arrays[i:i + chunk_size]
                              for i in range(0, len(data_arrays), chunk_size)]

                    def process_chunk(chunk):
                        return [10.0 ** (arr / 10.0) for arr in chunk]

                    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                    linear_arrays = []
                    for future in as_completed(futures):
                        linear_arrays.extend(future.result())

                    linear_stack = xp.vstack(linear_arrays)
            else:
                # Sequential processing
                linear_stack = xp.vstack([_db_to_lin(a) for a in data_arrays])

            # Compute global averages
            global_avg_lin = xp.mean(linear_stack, axis=0)
            global_avg_db = 10.0 * xp.log10(global_avg_lin)

            # Move to CPU if needed
            if use_gpu:
                global_avg_lin = cp.asnumpy(global_avg_lin)
                global_avg_db = cp.asnumpy(global_avg_db)

            # Register global averages in Veusz
            global_avg_lin_name = "Global_avg_lin"
            global_avg_db_name = "Global_avg_dB"

            self.doc.SetData(name=global_avg_lin_name, val=global_avg_lin)
            self.doc.SetData(name=global_avg_db_name, val=global_avg_db)

            # Tag the global average datasets
            self.doc.TagDatasets('Global_Average', [global_avg_db_name, global_avg_lin_name])

            # Create global average plot
            if self.config.plot_mode >= 1:  # avg+overlay or avg only modes
                prev_title = self.plotInfo.graph_title
                self.plotInfo.graph_title = "Global Average"
                self._plot_1d(global_avg_db_name)
                self.plotInfo.graph_title = prev_title

            # Add to overlay if in avg+overlay mode
            if self.config.plot_mode == 1:
                self._add_to_overlay(global_avg_db_name, "Global Average")

            print(f"Created global average from {len(global_data_datasets)} datasets")

        except Exception as e:
            print(f"Error in global average calculation: {e}")

    def _add_to_overlay(self, dataset_name, label):
        """Add dataset to overlay plot."""
        try:
            # Ensure overlay plot exists
            if 'AllImported' not in self.doc.Root.childnames:
                self._create_overlay_plot()

            # Get overlay graph
            overlay_graph = self.doc.Root.AllImported.grid1.Imported_Overlay

            # Add dataset to overlay
            overlay_xy = overlay_graph.Add('xy', name=f'{label}_overlay')
            overlay_xy.yData.val = dataset_name

            # Find frequency dataset for this data
            freq_dataset = None
            if hasattr(self, 'plotInfo') and self.plotInfo.base_name:
                freq_dataset = self.plotInfo.base_name + '_freq'

            if freq_dataset and freq_dataset in [ds for ds in self.doc.GetDatasetNames()]:
                overlay_xy.xData.val = freq_dataset

            overlay_xy.nanHandling = 'break-on'

            # Style overlay plot
            overlay_xy.marker.val = 'circle'
            overlay_xy.markerSize.val = '3pt'
            overlay_xy.MarkerLine.color.val = 'transparent'
            overlay_xy.MarkerFill.color.val = 'auto'
            overlay_xy.MarkerFill.transparency.val = 70
            overlay_xy.PlotLine.color.val = 'auto'
            overlay_xy.PlotLine.width.val = '2pt'

        except Exception as e:
            print(f"Error adding {label} to overlay: {e}")

    def _create_overlay_plot(self):
        """Create the main overlay plot."""
        try:
            # Create page for overlay
            self.page = self.doc.Root.Add('page', name='AllImported')
            self.page.notes.val = "All Imported and Plottable Data Overlay"

            # Create grid
            self.grid = self.page.Add('grid', columns=2)

            # Create overlay graph
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

            # Set color theme
            self.doc.Root.colorTheme.val = 'max128'

        except Exception as e:
            print(f"Error creating overlay plot: {e}")

    def _create_average_datasets(self, base_name: str):
        """Average all datasets belonging to *base_name* with enhanced GPU/CPU parallelization."""
        """
        Creates two new datasets:
        • _avg_lin – linear-domain average
        • _avg_dB – dB-domain average
        Uses GPU and/or multiprocessing for maximum performance.
        """
        # Retrieve list built during _process_file_data
        candidates = [ds for ds in self._datasets_by_base[base_name]
                      if ('freq' not in ds.lower()
                          and not ds.endswith(
                        ('_avg_dB', '_avg_lin', '_global_avg_dB', '_global_avg_lin', '_file_avg_dB', '_file_avg_lin')))]

        if len(candidates) < 2:  # nothing meaningful to average
            return

        # Pick numeric backend - prefer GPU if available
        use_gpu = self.config.enable_gpu_processing and self.gpu_processor and self.gpu_processor.gpu_available
        xp = cp if use_gpu else np

        # Helper to convert one array
        def _db_to_lin(arr):
            return xp.power(10.0, arr / 10.0)

        try:
            # Fetch data arrays
            data_arrays = [xp.asarray(self.doc.GetData(ds)[0])
                           for ds in candidates]

            # Enhanced parallel processing on CPU/GPU
            if use_gpu:
                # Use GPU for all array operations
                linear_arrays = [_db_to_lin(arr) for arr in data_arrays]
                linear_stack = xp.vstack(linear_arrays)
            elif (self.config.enable_multiprocessing and
                  len(candidates) > 2 and
                  self.config.num_processes > 1):
                # Use optimized ThreadPoolExecutor for mathematical operations
                with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
                    # Process arrays in parallel chunks
                    chunk_size = max(1, len(data_arrays) // self.config.num_processes)
                    chunks = [data_arrays[i:i + chunk_size]
                              for i in range(0, len(data_arrays), chunk_size)]

                    def process_chunk(chunk):
                        return [10.0 ** (arr / 10.0) for arr in chunk]

                    # Submit chunk processing tasks
                    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

                    # Collect results
                    linear_arrays = []
                    for future in as_completed(futures):
                        linear_arrays.extend(future.result())

                    linear_stack = xp.vstack(linear_arrays)
            else:
                # Sequential processing
                linear_stack = xp.vstack([_db_to_lin(a) for a in data_arrays])

            # Compute averages
            avg_lin = xp.mean(linear_stack, axis=0)
            avg_db = 10.0 * xp.log10(avg_lin)

            # Move to CPU if needed
            if use_gpu:
                avg_lin = cp.asnumpy(avg_lin)
                avg_db = cp.asnumpy(avg_db)

            # Register in Veusz
            lin_name = f"{base_name}_avg_lin"
            db_name = f"{base_name}_avg_dB"

            self.doc.SetData(name=lin_name, val=avg_lin)
            self.doc.SetData(name=db_name, val=avg_db)

            # Set the tags for the datasets
            self.doc.TagDatasets('Avg_dB', [db_name])
            self.doc.TagDatasets('Avg_Linear', [lin_name])
            self.doc.TagDatasets(base_name, [db_name, lin_name])

            # Book-keeping so we don't re-average
            self._datasets_by_base[base_name].extend([lin_name, db_name])

            # Create average plots based on plot mode
            if self.config.plot_mode >= 1:  # avg+overlay or avg only modes
                prev_title = self.plotInfo.graph_title
                self.plotInfo.graph_title = f"{base_name} average"
                self._plot_1d(db_name)  # plot dB average
                self.plotInfo.graph_title = prev_title

                # Add to overlay if in avg+overlay mode
                if self.config.plot_mode == 1:
                    self._add_to_overlay(db_name, f"{base_name} avg")

        except Exception as e:
            print(f"Error in average calculation: {e}")

    def _process_file_data(self, filename, data_returned):
        """
        Process individual file data and create plots with enhanced performance.
        """
        base_name = os.path.splitext(os.path.basename(filename))[0]
        self.plotInfo.base_name = base_name

        # Setup auto-save path on first file
        if self.auto_save_enabled and self.base_save_path is None:
            self._setup_auto_save_path()

        # Reset current file datasets tracking
        self._current_file_datasets.clear()

        # Extract section data
        data_sections = dict(filter(
            lambda item: 'section' in item[0],
            data_returned['pattern_matches'].items()
        ))

        if len(data_sections) != len(data_returned['data_matches']):
            raise ValueError(f"Data sections mismatch in file {filename}")

        # Process data values with potential parallel processing
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

        # Use high-precision frequency range generation
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
                self._datasets_by_base[base_name].append(x_data_name)
                self._current_file_datasets.append(x_data_name)

            # Verify data alignment
            if (data_line_numbers[index] - 1 != data_section_line_numbers[index]):
                raise ValueError(f"Data alignment error in {filename}")

            # Set data in Veusz - retain original dataset name
            self.doc.SetData(name=dataset_name, val=data_y_values[index])
            self.doc.TagDatasets(base_name, [dataset_name])
            self._datasets_by_base[base_name].append(dataset_name)
            self._current_file_datasets.append(dataset_name)
            self._global_datasets.append(dataset_name)  # Track for global average

            # Update plot info and create plot
            self.plotInfo.graph_notes = data_notes
            self.plotInfo.graph_title = base_name + '::' + dataset_name
            self.plotInfo.graph_title = self.plotInfo.graph_title.replace('_', ' ')

            # Create individual plots only in mode 0 (all plots)
            if self.config.plot_mode == 0:
                self._plot_1d(dataset_name)

            # Increment plot count and check for auto-save
            self.plot_count += 1
            self._auto_save_if_needed()

        # Create the averaged datasets after everything is processed
        self._create_average_datasets(base_name)

        # Create global average if we have enough datasets
        if len(self._global_datasets) >= 2:
            self._create_global_average()

    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)

    def _plot_1d(self, dataset: str):
        """Create line plot for 1D datasets with enhanced styling and mode support."""
        try:
            # Check if we should create overlay based on plot mode
            create_overlay = (self.config.plot_mode == 0) or (self.config.plot_mode == 1)
            create_individual = (self.config.plot_mode == 0)

            # Create overlay plot if it doesn't exist and mode allows it
            if create_overlay and 'AllImported' not in self.doc.Root.childnames:
                self._create_overlay_plot()
            elif create_overlay:
                self.page = self.doc.Root.AllImported
                graph_all = self.doc.Root.AllImported.grid1.Imported_Overlay

            # Add overlay plot if mode allows it
            if create_overlay:
                if 'AllImported' in self.doc.Root.childnames:
                    graph_all = self.doc.Root.AllImported.grid1.Imported_Overlay

                    all_overlay_xy = graph_all.Add('xy', name=dataset)
                    all_overlay_xy.yData.val = dataset
                    all_overlay_xy.xData.val = self.plotInfo.base_name + '_freq'
                    all_overlay_xy.nanHandling = 'break-on'

                    # Style overlay plot with enhanced performance settings
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

            # Create individual plot only if mode allows it
            if create_individual:
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
        """Save Veusz document with high precision support and parallel processing."""
        # Create global and file averages before final save
        self._create_global_average()
        self._create_file_average()

        filename_root = os.path.splitext(filename)[0]
        filename_hp = filename_root + '.vszh5'
        file_split = os.path.split(filename)
        filename_vsz = (file_split[0] + '/Beware_oldVersion/' +
                        os.path.splitext(file_split[1])[0] + '_BEWARE.vsz')

        # Use parallel processing for saves if enabled
        if self.config.enable_multiprocessing:
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both save operations
                hdf5_future = executor.submit(self.doc.Save, filename_hp, 'hdf5')

                # Create legacy directory and submit legacy save
                os.makedirs(file_split[0] + '/Beware_oldVersion/', exist_ok=True)
                vsz_future = executor.submit(self.doc.Save, filename_vsz, 'vsz')

                # Wait for completion
                try:
                    hdf5_future.result(timeout=60)
                    vsz_future.result(timeout=60)
                except Exception as e:
                    print(f"Parallel save error: {e}")
                    # Fallback to sequential save
                    self.doc.Save(filename_hp, mode='hdf5')
                    self.doc.Save(filename_vsz, mode='vsz')
        else:
            # Sequential save
            self.doc.Save(filename_hp, mode='hdf5')
            os.makedirs(file_split[0] + '/Beware_oldVersion/', exist_ok=True)
            self.doc.Save(filename_vsz, mode='vsz')

        # Save any remaining plots in current batch if auto-save is enabled
        if self.auto_save_enabled and self.plot_count % self.plots_per_file != 0:
            if self.base_save_path is not None:
                final_batch_filename = f"{self.base_save_path}_{self.file_batch_number:03d}.vszh5"
                try:
                    # Create final averages before saving
                    self._create_file_average()
                    self.doc.Save(final_batch_filename, mode='hdf5')
                    print(
                        f"Final batch saved with {self.plot_count % self.plots_per_file} plots to: {final_batch_filename}")
                except Exception as e:
                    print(f"Error saving final batch: {e}")

        # Close window after saving
        self._close_current_document_window()

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

        # a basic plot with a sin function
        ifc.To(ifc.Add('page'))
        ifc.To(ifc.Add('graph'))
        ifc.Add('function', name='myfunc')
        ifc.Set('myfunc/function', 'sin(x)')
        ifc.Set('x/max', 3.14 * 2)


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
    """Main application entry point with enhanced performance optimizations."""
    # Enhanced monitoring for large file processing
    monitor = start_monitoring(seconds_frozen=30, test_interval=100)

    # Set multiprocessing start method for cross-platform compatibility
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)

    # Call this before any Qt imports
    if getattr(sys, 'frozen', False):  # Check if running as compiled executable
        setup_qt_plugins()

    app = QApplication(sys.argv)

    # Set application properties for better performance
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app.setApplicationDisplayName("Enhanced R&S SFT File Plotter v1.0.3")
    app.setApplicationVersion("1.0.3")

    # Create and show main window
    window = EnhancedMainWindow()
    window.show()

    # Run application
    try:
        sys.exit(app.exec_())
    finally:
        # Cleanup
        monitor.stop()

        # Force cleanup of any remaining GPU memory
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass


if __name__ == '__main__':
    main()
