# CORRECTED FILE 2: RAndS_FSW_ASCII_Plotter_lrgFiles.py
# Full corrected version with all CUDA and Veusz fixes applied
# Date: January 26, 2026
# All issues fixed: CUDA detection + Data registration sequence + Validation
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
# %%% System Interface Modules
import os
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter
warnings.filterwarnings('ignore', message='.*CUDA path could not be detected.*')
from hanging_threads import start_monitoring
import datetime
# %%% GUI Module Imports - QtPy for cross-platform compatibility
if getattr(sys, 'frozen', False):
    os.environ['QT_API'] = 'pyside6'
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QLineEdit
    )
else:
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
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing as mp
# %%% GPU Acceleration Modules
# Multi-level GPU detection with fallback
CUPY_AVAILABLE = False
CUPY_AVAILABLE_VERIFIED = False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use()
        CUPY_AVAILABLE = True
        CUPY_AVAILABLE_VERIFIED = True
        print("✓ CuPy/CUDA available and verified")
    except RuntimeError as e:
        print(f"⚠ CuPy found but CUDA not accessible: {e}")
        print("  Falling back to CPU processing")
        CUPY_AVAILABLE = False
except ImportError:
    print("ℹ CuPy not installed. Using CPU processing.")
    CUPY_AVAILABLE = False
except Exception as e:
    print(f"⚠ CuPy error: {e}")
    print("  Falling back to CPU processing")
    CUPY_AVAILABLE = False
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
# %%% Plotting Environment
import veusz.embed as embed
# %% Configuration and Data Classes
@dataclass
class ProcessingConfig:
    """Configuration class for processing settings."""
    enable_multiprocessing: bool = True
    enable_gpu_processing: bool = False
    use_opencl: bool = True
    num_processes: int = cpu_count()
    max_workers: int = cpu_count()
    chunk_size: int = min(1000, max(100, cpu_count() * 10))
    plot_mode: int = 0
    enable_auto_save: bool = False
    plots_per_file: int = 1000
    auto_save_base_name: str = "RnS_Plots_Batch"
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
# %% GPU Processing Classes
class GPUProcessor:
    """Handles GPU acceleration using either CuPy or PyOpenCL."""
    def __init__(self, config: ProcessingConfig):
        """Initialize GPU processor based on available libraries."""
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
        """Initialize CuPy for NVIDIA GPU acceleration with proper CUDA detection."""
        try:
            if not CUPY_AVAILABLE:
                print("CuPy not available - skipping GPU initialization")
                self.gpu_available = False
                return
            cp.cuda.Device(0).use()
            self.memory_pool = cp.get_default_memory_pool()
            self.gpu_available = True
            print("CuPy initialized successfully with CUDA device")
        except RuntimeError as cuda_runtime_error:
            print(f"Warning: CUDA device not found or not accessible: {cuda_runtime_error}")
            print("Falling back to CPU processing")
            self.gpu_available = False
        except Exception as e:
            print(f"Warning: CuPy initialization failed: {e}")
            print("Falling back to CPU processing")
            self.gpu_available = False
    def process_array_gpu(self, data_array):
        """Process numpy array using GPU acceleration."""
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
        mf = cl.mem_flags
        data_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=data_array.astype(np.float32))
        kernel_source = """
        __kernel void process_data(__global float* data) {
            int gid = get_global_id(0);
            data[gid] = data[gid] * 1.0f;
        }
        """
        program = cl.Program(self.context, kernel_source).build()
        kernel = program.process_data
        kernel(self.queue, (len(data_array),), None, data_buffer)
        result = np.empty_like(data_array, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, data_buffer)
        self.queue.finish()
        return result.astype(data_array.dtype)
    def _process_cupy(self, data_array):
        """Process data using CuPy."""
        gpu_array = cp.asarray(data_array)
        processed_gpu = gpu_array * 1.0
        return cp.asnumpy(processed_gpu)
# %% Multiprocessing Worker Functions
def process_file_worker(file_info):
    """Worker function for processing individual SFT files in parallel."""
    filename, search_strings, sft_lines, config = file_info
    try:
        data_returned = fparser(filename, line_targets=sft_lines,
                                string_patterns=search_strings)
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
    """Worker function for saving batches in parallel."""
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
def extract_with_regex(inputText: str, delim: str = ';'):
    """Extract all substrings enclosed by the same delimiter using regex."""
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
        """Initialize processing thread."""
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
        self.setWindowTitle("Enhanced R&S SFT File Plotter - Large Files")
        self.setGeometry(100, 100, 800, 850)
        self.config = ProcessingConfig()
        self.vzplot = VZPlotRnS(self.config)
        self.selected_files = []
        self._setup_ui()
    def _setup_ui(self):
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
        self.browse_button.clicked.connect(self._browse_files)
        browse_layout.addWidget(self.browse_button)
        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        browse_layout.addWidget(self.clear_button)
        file_layout.addLayout(browse_layout)
        main_layout.addWidget(file_group)
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        self.enable_mp_checkbox = QCheckBox("Enable Multiprocessing")
        self.enable_mp_checkbox.setChecked(self.config.enable_multiprocessing)
        self.enable_mp_checkbox.stateChanged.connect(self._update_mp_config)
        options_layout.addWidget(self.enable_mp_checkbox)
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
        self.enable_gpu_checkbox = QCheckBox("Enable GPU Processing")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.stateChanged.connect(self._update_gpu_config)
        options_layout.addWidget(self.enable_gpu_checkbox)
        self.use_opencl_checkbox = QCheckBox("Prefer OpenCL (Cross-platform)")
        self.use_opencl_checkbox.setChecked(self.config.use_opencl)
        self.use_opencl_checkbox.stateChanged.connect(self._update_opencl_config)
        options_layout.addWidget(self.use_opencl_checkbox)
        self.memory_opt_checkbox = QCheckBox("Memory Optimization")
        self.memory_opt_checkbox.setChecked(self.config.memory_optimization)
        self.memory_opt_checkbox.stateChanged.connect(self._update_memory_config)
        options_layout.addWidget(self.memory_opt_checkbox)
        main_layout.addWidget(options_group)
        # Plot options
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout(plot_group)
        self.plot_mode_group = QButtonGroup()
        self.all_plots_radio = QRadioButton("All plots (original behavior)")
        self.all_plots_radio.setChecked(True)
        self.plot_mode_group.addButton(self.all_plots_radio, 0)
        plot_layout.addWidget(self.all_plots_radio)
        self.avg_overlay_radio = QRadioButton("Average + Overlay only")
        self.plot_mode_group.addButton(self.avg_overlay_radio, 1)
        plot_layout.addWidget(self.avg_overlay_radio)
        self.avg_only_radio = QRadioButton("Average only")
        self.plot_mode_group.addButton(self.avg_only_radio, 2)
        plot_layout.addWidget(self.avg_only_radio)
        self.plot_mode_group.buttonClicked.connect(self._update_plot_mode)
        main_layout.addWidget(plot_group)
        # Auto-save options
        autosave_group = QGroupBox("Auto-Save Options")
        autosave_layout = QVBoxLayout(autosave_group)
        self.enable_autosave_checkbox = QCheckBox("Enable automatic batch saving")
        self.enable_autosave_checkbox.stateChanged.connect(self._update_autosave_config)
        autosave_layout.addWidget(self.enable_autosave_checkbox)
        plots_layout = QHBoxLayout()
        plots_layout.addWidget(QLabel("Plots per file:"))
        self.plots_per_file_edit = QLineEdit("1000")
        self.plots_per_file_edit.setEnabled(False)
        self.plots_per_file_edit.textChanged.connect(self._update_plots_per_file)
        plots_layout.addWidget(self.plots_per_file_edit)
        plots_layout.addStretch()
        autosave_layout.addLayout(plots_layout)
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
        self._log_message(f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}")
    def _update_cpu_config(self, value):
        """Update CPU cores configuration."""
        self.config.num_processes = value
        self.config.max_workers = value
        self._log_message(f"CPU cores set to: {value}")
    def _update_gpu_config(self, state):
        """Update GPU processing configuration."""
        self.config.enable_gpu_processing = state == Qt.Checked
        self._log_message(f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}")
    def _update_opencl_config(self, state):
        """Update OpenCL preference configuration."""
        self.config.use_opencl = state == Qt.Checked
        self._log_message(f"OpenCL preference: {'Enabled' if self.config.use_opencl else 'Disabled'}")
    def _update_memory_config(self, state):
        """Update memory optimization configuration."""
        self.config.memory_optimization = state == Qt.Checked
        self._log_message(f"Memory optimization: {'Enabled' if self.config.memory_optimization else 'Disabled'}")
    def _update_plot_mode(self, button):
        """Update plot mode configuration."""
        self.config.plot_mode = self.plot_mode_group.id(button)
        mode_names = ["All plots", "Average + Overlay only", "Average only"]
        self._log_message(f"Plot mode: {mode_names[self.config.plot_mode]}")
    def _update_autosave_config(self, state):
        """Update auto-save configuration."""
        self.config.enable_auto_save = state == Qt.Checked
        self.plots_per_file_edit.setEnabled(self.config.enable_auto_save)
        self.base_filename_edit.setEnabled(self.config.enable_auto_save)
        self.vzplot.auto_save_enabled = self.config.enable_auto_save
        self._log_message(f"Auto-save: {'Enabled' if self.config.enable_auto_save else 'Disabled'}")
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
    def _get_timestamp(self):
        """Get current timestamp string."""
        return datetime.datetime.now().strftime("%H:%M:%S")
    def _process_and_plot(self):
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
            self.vzplot.searchData_strings,
            self.vzplot.sft_lines
        )
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self._on_processing_finished)
        self.processing_thread.error_occurred.connect(self._on_processing_error)
        self.processing_thread.start()
        self._log_message("Processing started...")
    def _on_processing_finished(self, results):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        self._log_message(f"Processing completed: {len(successful_results)} successful, {len(failed_results)} failed")
        if failed_results:
            error_msg = "\n".join([f"{r['filename']}: {r['error']}" for r in failed_results])
            QMessageBox.warning(self, "Processing Errors", f"Some files failed:\n{error_msg}")
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
                reply = QMessageBox.question(
                    self, "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    VZPlotRnS.open_veusz_gui(save_path)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project: {e}")
# %% Auto Plotter Class
class VZPlotRnS:
    """Enhanced Veusz plotting class with multiprocessing support."""
    def __init__(self, config: ProcessingConfig):
        """Initialize VZPlotRnS with enhanced capabilities."""
        self.config = config
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)
        # NEW: Initialize auto-save tracking variables
        self.plot_count = 0
        self.file_batch_number = 1
        self.base_save_path = None
        self.auto_save_enabled = False
        self.plots_per_file = 1000
        # NEW: Dataset tracking for validation
        self.registered_datasets = set()
        self.frequency_data = None
        self.frequency_registered = False
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
            'stop_2': 'STOP',
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
        # Track datasets for average calculation
        self._datasets_by_base = defaultdict(list)
    def _setup_auto_save_path(self):
        """Setup automatic save path based on configuration."""
        if self.base_save_path is None and self.auto_save_enabled:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"RnS_AutoSave_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            self.base_save_path = os.path.join(save_dir, self.config.auto_save_base_name)
            print(f"Auto-save directory created: {save_dir}")
    def _auto_save_if_needed(self):
        """Automatically save file if plot count threshold is reached."""
        if not self.auto_save_enabled:
            return
        if self.plot_count > 0 and self.plot_count % self.plots_per_file == 0:
            if self.base_save_path is None:
                self._setup_auto_save_path()
            batch_filename = f"{self.base_save_path}_{self.file_batch_number:03d}.vszh5"
            if self.config.enable_multiprocessing:
                self._save_batch_parallel(batch_filename)
            else:
                self._save_batch_sequential(batch_filename)
            print(f"Auto-saved batch {self.file_batch_number} with {self.plots_per_file} plots to: {batch_filename}")
            self.file_batch_number += 1
            self._reset_for_next_batch()
    def _save_batch_parallel(self, filename):
        """Save current batch using parallel processing."""
        try:
            def save_worker():
                self.doc.Save(filename, mode='hdf5')
            save_thread = threading.Thread(target=save_worker)
            save_thread.start()
            save_thread.join(timeout=30)
            if save_thread.is_alive():
                print(f"Warning: Save operation for {filename} timed out")
        except Exception as e:
            print(f"Error during parallel save: {e}")
            self._save_batch_sequential(filename)
    def _save_batch_sequential(self, filename):
        """Save current batch sequentially."""
        try:
            self.doc.Save(filename, mode='hdf5')
        except Exception as e:
            print(f"Error during sequential save: {e}")
    def _reset_for_next_batch(self):
        """Reset document for next batch while preserving settings."""
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.doc.EnableToolbar(enable=True)
        self.first_1d = True
        self._datasets_by_base.clear()
    def _create_average_datasets(self, base_name: str):
        """Average all datasets belonging to base_name."""
        candidates = [ds for ds in self._datasets_by_base[base_name]
                      if ('freq' not in ds.lower()
                          and not ds.endswith(('_avg_dB', '_avg_lin')))]
        if len(candidates) < 2:
            return
        use_gpu = self.config.enable_gpu_processing and CUPY_AVAILABLE
        xp = cp if use_gpu else np
        def _db_to_lin(arr):
            return xp.power(10.0, arr / 10.0)
        data_arrays = [xp.asarray(self.doc.GetData(ds)[0])
                       for ds in candidates]
        if (self.config.enable_multiprocessing and not use_gpu
                and len(candidates) > 1 and self.config.num_processes > 1):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.num_processes) as exe:
                linear_arrays = list(exe.map(lambda a: 10.0 ** (a / 10.0), data_arrays))
            linear_stack = xp.vstack(linear_arrays)
        else:
            linear_stack = xp.vstack([_db_to_lin(a) for a in data_arrays])
        avg_lin = xp.mean(linear_stack, axis=0)
        avg_db = 10.0 * xp.log10(avg_lin)
        if use_gpu:
            avg_lin = cp.asnumpy(avg_lin)
            avg_db = cp.asnumpy(avg_db)
        lin_name = f"{base_name}_avg_lin"
        db_name = f"{base_name}_avg_dB"
        self.doc.SetData(name=lin_name, val=avg_lin)
        self.doc.SetData(name=db_name, val=avg_db)
        self.doc.TagDatasets('Avg_dB', [db_name])
        self.doc.TagDatasets('Avg_Linear', [lin_name])
        self.doc.TagDatasets(base_name, [db_name, lin_name])
        self._datasets_by_base[base_name].extend([lin_name, db_name])
        if self.config.plot_mode >= 1:
            prev_title = self.plotInfo.graph_title
            self.plotInfo.graph_title = f"{base_name} average"
            self._plot_1d(db_name)
            self.plotInfo.graph_title = prev_title
    def _process_file_data(self, filename, data_returned):
        """Process individual file data and create plots."""
        base_name = os.path.splitext(os.path.basename(filename))[0]
        self.plotInfo.base_name = base_name
        if self.auto_save_enabled and self.base_save_path is None:
            self._setup_auto_save_path()
        data_sections = dict(filter(
            lambda item: 'section' in item[0],
            data_returned['pattern_matches'].items()
        ))
        if len(data_sections) != len(data_returned['data_matches']):
            raise ValueError(f"Data sections mismatch in file {filename}")
        data_y_values = list(map(itemgetter('extracted_value'),
                                 data_returned['data_matches'].values()))
        data_line_numbers = list(map(itemgetter('line_number'),
                                     data_returned['data_matches'].values()))
        data_section_line_numbers = list(map(itemgetter('line_number'),
                                             data_sections.values()))
        data_section_content = list(map(itemgetter('content'),
                                        data_sections.values()))
        data_fields = data_returned['pattern_matches']
        num_pts = extract_with_regex(data_fields['values']['extracted_value'])
        if len(num_pts) != 1:
            raise ValueError(f"Invalid VALUES field in {filename}")
        num_pts = int(num_pts[0])
        freq_start = float(extract_with_regex(
            data_fields['start']['extracted_value'])[0])
        freq_stop = float(extract_with_regex(
            data_fields['stop_2']['extracted_value'])[0])
        freq_range = np.linspace(freq_start, freq_stop, num=num_pts,
                                 endpoint=True, dtype=np.float64)
        data_header = data_returned['line_data']
        data_notes = '\n'.join(data_header.values())
        data_notes = os.path.split(filename)[1] + '\n\n' + data_notes
        # ========== CRITICAL SECTION: REGISTER DATA BEFORE PLOTTING ==========
        for index, label in enumerate(data_section_content):
            dataset_name = label
            # STEP 1: Register frequency data (only once)
            if index == 0:
                x_data_name = base_name + '_freq'
                self.doc.SetData(name=x_data_name, val=freq_range)  # ← REGISTER
                self.doc.TagDatasets(base_name, [x_data_name])
                self._datasets_by_base[base_name].append(x_data_name)
                self.registered_datasets.add(x_data_name)
                self.frequency_registered = True
                self.frequency_data = freq_range
            # Verify data alignment
            if (data_line_numbers[index] - 1 != data_section_line_numbers[index]):
                raise ValueError(f"Data alignment error in {filename}")
            # STEP 2: Register Y data
            self.doc.SetData(name=dataset_name, val=data_y_values[index])  # ← REGISTER
            self.doc.TagDatasets(base_name, [dataset_name])
            self._datasets_by_base[base_name].append(dataset_name)
            self.registered_datasets.add(dataset_name)
            # Update plot info
            self.plotInfo.graph_notes = data_notes
            self.plotInfo.graph_title = base_name + '::' + dataset_name
            self.plotInfo.graph_title = self.plotInfo.graph_title.replace('_', ' ')
            # STEP 3: Create plot (data now exists!)
            if self.config.plot_mode == 0:
                self._plot_1d(dataset_name)
            # Increment plot count and check for auto-save
            self.plot_count += 1
            self._auto_save_if_needed()
        # Create the averaged datasets after everything is processed
        self._create_average_datasets(base_name)
    def _create_page(self, dataset: str):
        """Create a new page and grid."""
        self.page = self.doc.Root.Add('page', name=dataset)
        self.grid = self.page.Add('grid', columns=2)
    def _plot_1d(self, dataset: str):
        """Create line plot for 1D datasets with enhanced styling."""
        try:
            # NEW: Validation at the beginning
            if dataset not in self.registered_datasets:
                print(f"Warning: Dataset '{dataset}' not registered. Skipping plot.")
                return
            if not self.frequency_registered:
                print(f"Warning: Frequency data not registered. Skipping plot.")
                return
            create_overlay = (self.config.plot_mode == 0) or (self.config.plot_mode == 1)
            create_individual = (self.config.plot_mode == 0)
            if create_overlay and 'AllImported' not in self.doc.Root.childnames:
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
                graph_all.x.label.val = self.plotInfo.xAxis_label
                graph_all.y.label.val = self.plotInfo.yAxis_label
                self.doc.Root.colorTheme.val = 'max128'
            elif create_overlay:
                self.page = self.doc.Root.AllImported
                # graph_all = self.doc.Root.AllImported.grid1.Imported_Overlay
                graph_all = self.page.grid1.Imported_Overlay
            if create_overlay:
                all_overlay_xy = graph_all.Add('xy', name=dataset)
                all_overlay_xy.yData.val = dataset
                all_overlay_xy.xData.val = self.plotInfo.base_name + '_freq'
                all_overlay_xy.nanHandling = 'break-on'
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
                xy = graph.Add('xy', name=dataset)
                xy.yData.val = dataset
                xy.xData.val = self.plotInfo.base_name + '_freq'
                xy.nanHandling = 'break-on'
                graph.x.label.val = self.plotInfo.xAxis_label
                graph.y.label.val = self.plotInfo.yAxis_label
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
        self.doc.Save(filename_hp, mode='hdf5')
        os.makedirs(file_split[0] + '/Beware_oldVersion/', exist_ok=True)
        self.doc.Save(filename_vsz, mode='vsz')
        if self.auto_save_enabled and self.plot_count % self.plots_per_file != 0:
            if self.base_save_path is not None:
                final_batch_filename = f"{self.base_save_path}_{self.file_batch_number:03d}.vszh5"
                try:
                    self.doc.Save(final_batch_filename, mode='hdf5')
                    print(
                        f"Final batch saved with {self.plot_count % self.plots_per_file} plots to: {final_batch_filename}")
                except Exception as e:
                    print(f"Error saving final batch: {e}")
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
# %% Utility Functions
def setup_qt_plugins():
    """Setup Qt platform plugin paths for compiled applications."""
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
    monitor = start_monitoring(seconds_frozen=20, test_interval=100)
    if __name__ == '__main__':
        mp.set_start_method("spawn", force=True)
    if getattr(sys, 'frozen', False):
        setup_qt_plugins()
    app = QApplication(sys.argv)
    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec_())
    monitor.stop()
if __name__ == '__main__':
    main()
