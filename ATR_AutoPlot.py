"""Enhanced ATR AutoPlot with Modern Qt GUI Interface.

This version integrates the modern Qt GUI interface from the R&S plotter
with the ATR processing capabilities, including multiprocessing and GPU acceleration.

Author: William W. Wallace
Last updated: 2025-06-28
"""

# -*- coding: utf-8 -*-

# %% Import Modules

import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

# %%% Import Math and Plotting Modules
import numpy as np
import veusz.embed as vz

# %%% Import GUI Modules
from qtpy.QtGui import *
from qtpy.QtCore import Qt, QSize, QThread, Signal
from qtpy.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
    QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
    QSpinBox, QGroupBox, QListWidget, QSplitter, QLineEdit,
)

# %%% Debugging Modules
# from rich import inspect as richinspect
# import pdir

# %% GPU Computing imports with fallback support

try:
    import cupy as cp
    GPU_AVAILABLE = "cupy"
    print("CuPy detected - NVIDIA/AMD GPU acceleration available")
except ImportError:
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        GPU_AVAILABLE = "opencl"
        print("PyOpenCL detected - Cross-platform GPU acceleration available")
    except ImportError:
        try:
            import taichi as ti
            GPU_AVAILABLE = "taichi"
            print("Taichi detected - Cross-platform GPU acceleration available")
        except ImportError:
            GPU_AVAILABLE = None
            print("No GPU acceleration libraries available - using CPU only")

# System Interface Modules
os.environ['QT_API'] = 'pyside6'

# %% Configuration Classes


@dataclass
class ProcessingConfig:
    """Configuration class for processing settings."""
    enable_multiprocessing: bool = True
    enable_gpu_processing: bool = True
    use_opencl: bool = True
    num_processes: int = multiprocessing.cpu_count()
    max_workers: int = multiprocessing.cpu_count()
    chunk_size: int = 1000

# %% GPU Acceleration Classes


class GPUAccelerator:
    """Handles GPU-accelerated computations with cross-platform support."""

    def __init__(self, enable_gpu: bool = True):
        """Initialize GPU accelerator.

        Parameters
        ----------
        enable_gpu : bool, optional
            Whether to enable GPU acceleration. Default is True.
        """
        self.gpu_enabled = enable_gpu and GPU_AVAILABLE is not None
        self.backend = GPU_AVAILABLE if self.gpu_enabled else None

        if self.gpu_enabled:
            self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize the appropriate GPU backend."""
        if self.backend == "cupy":
            try:
                cp.cuda.Device(0).use()
                print(f"CuPy initialized on device: {cp.cuda.Device()}")
            except Exception as e:
                print(f"CuPy initialization failed: {e}")
                self.gpu_enabled = False

        elif self.backend == "opencl":
            try:
                self.cl_context = cl.create_some_context()
                self.cl_queue = cl.CommandQueue(self.cl_context)
                print(f"OpenCL initialized: {self.cl_context.devices}")
            except Exception as e:
                print(f"OpenCL initialization failed: {e}")
                self.gpu_enabled = False

        elif self.backend == "taichi":
            try:
                ti.init(arch=ti.gpu)
                print("Taichi GPU backend initialized")
            except Exception as e:
                print(f"Taichi GPU initialization failed: {e}")
                ti.init(arch=ti.cpu)
                print("Taichi fallback to CPU")
                self.gpu_enabled = False

    def array_operations(self, data: np.ndarray, apply_db_conversion: bool = False) -> np.ndarray:
        """Perform array operations with GPU acceleration if available.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        apply_db_conversion : bool, optional
            Whether to apply dB conversion. Default is False for ATR magnitude data.

        Returns
        -------
        np.ndarray
            Processed data array.
        """
        if not self.gpu_enabled or not apply_db_conversion:
            return self._cpu_passthrough(data)

        try:
            if self.backend == "cupy":
                return self._cupy_operations(data, apply_db_conversion)
            elif self.backend == "opencl":
                return self._opencl_operations(data, apply_db_conversion)
            elif self.backend == "taichi":
                return self._taichi_operations(data, apply_db_conversion)
        except Exception as e:
            print(f"GPU operation failed, falling back to CPU: {e}")
            return self._cpu_passthrough(data)

        return self._cpu_passthrough(data)

    def _cpu_passthrough(self, data: np.ndarray) -> np.ndarray:
        """CPU-based passthrough - no transformation for ATR magnitude data."""
        return np.copy(data)

    def _cupy_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """CuPy-based GPU operations with optional dB conversion."""
        gpu_data = cp.asarray(data)
        if apply_db_conversion:
            gpu_result = cp.where(
                gpu_data != 0,
                20 * cp.log10(cp.abs(gpu_data)),
                -60
            )
        else:
            gpu_result = gpu_data
        return cp.asnumpy(gpu_result)

    def _opencl_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """OpenCL-based GPU operations with optional dB conversion."""
        if not apply_db_conversion:
            return np.copy(data)

        data_gpu = cl_array.to_device(self.cl_queue, data.astype(np.float32))
        result_gpu = cl_array.empty_like(data_gpu)

        kernel_code = """
        __kernel void log_transform(__global float* input,
                                   __global float* output,
                                   int n) {
            int i = get_global_id(0);
            if (i < n) {
                if (input[i] != 0.0f) {
                    output[i] = 20.0f * log10(fabs(input[i]));
                } else {
                    output[i] = -60.0f;
                }
            }
        }
        """

        program = cl.Program(self.cl_context, kernel_code).build()
        program.log_transform(
            self.cl_queue,
            (data.size,),
            None,
            data_gpu.data,
            result_gpu.data,
            np.int32(data.size)
        )

        return result_gpu.get()

    def _taichi_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """Taichi-based GPU operations with optional dB conversion."""
        if not apply_db_conversion:
            return np.copy(data)

        @ti.kernel
        def log_transform(input_field: ti.template(),
                          output_field: ti.template()) -> None:
            for i in input_field:
                if input_field[i] != 0.0:
                    output_field[i] = 20.0 * ti.log10(ti.abs(input_field[i]))
                else:
                    output_field[i] = -60.0

        input_field = ti.field(dtype=ti.f32, shape=data.shape)
        output_field = ti.field(dtype=ti.f32, shape=data.shape)

        input_field.from_numpy(data.astype(np.float32))
        log_transform(input_field, output_field)

        return output_field.to_numpy()

# %% Worker Functions


def process_single_atr_file(file_info: Tuple[str, int, object]) -> Tuple[str, Dict[str, Any]]:
    """Process a single ATR file with multiprocessing support.

    This function is designed to be used with multiprocessing pools.

    Parameters
    ----------
    file_info : Tuple[str, int, object]
        Tuple containing (file_path, line_number, gpu_accelerator).

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing (filename, processed_data_dict).
    """
    file_path, line_number, gpu_accelerator = file_info

    try:
        # Read file with optimized approach based on size
        filesize = os.path.getsize(file_path)
        if filesize < 10**7:  # < 10MB
            with open(file_path, 'rb') as file:
                content = file.read().decode('ascii')
                lines = content.splitlines()
        else:
            with open(file_path, 'r', encoding='ascii') as file:
                lines = file.readlines()

        # Extract header information
        header_lines = lines[:line_number]
        header_info = ''.join(header_lines).strip()

        # Extract numerical data
        data_line = lines[line_number].strip()
        if data_line.endswith('#'):
            data_line = data_line[:-1]

        # Convert to numbers and separate phase/magnitude
        data_numbers = list(map(float, data_line.split()))
        selected_phase_data = data_numbers[::2]
        selected_magnitude_data = data_numbers[1::2]

        # Create numpy array - magnitude data is already in dB format in ATR files
        selected_data = np.array(
            [selected_magnitude_data, selected_phase_data])
        selected_data_transpose = selected_data.transpose()

        # Parse header for frequency and azimuth info
        freq_max = float(header_lines[8].split(":")[-1].strip())
        freq_min = float(header_lines[9].split(":")[-1].strip())
        az_min = float(header_lines[6].split(":")[-1].strip())
        az_max = float(header_lines[7].split(":")[-1].strip())

        # Create azimuth angles array
        az_angles = np.arange(az_min, az_max + 1, 1, dtype=float)

        filename = os.path.basename(file_path)

        return filename, {
            'header_lines': header_lines,
            'data': selected_data_transpose,
            'frequency': freq_max,
            'azimuth_angles': az_angles,
            'magnitude': selected_data[0],  # Already in dB format
            'phase': selected_data[1]
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return os.path.basename(file_path), {}

# %% Threading Classes


class ATRProcessingThread(QThread):
    """Thread for handling ATR file processing without blocking the GUI."""

    progress_updated = Signal(int)
    processing_finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, file_list, config):
        """Initialize ATR processing thread.

        Parameters
        ----------
        file_list : list
            List of ATR files to process.
        config : ProcessingConfig
            Processing configuration.
        """
        super().__init__()
        self.file_list = file_list
        self.config = config
        self.gpu_accelerator = GPUAccelerator(config.enable_gpu_processing)

    def run(self):
        """Execute ATR file processing in separate thread."""
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
        line_number = 13  # Zero-indexed line number for data

        file_info_list = [
            (file_path, line_number, self.gpu_accelerator)
            for file_path in self.file_list
        ]

        results = {}

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_atr_file, file_info): file_info[0]
                for file_info in file_info_list
            }

            completed = 0
            for future in as_completed(future_to_file):
                filename, data = future.result()
                if data:
                    results[filename] = data

                completed += 1
                progress = int((completed / len(self.file_list)) * 100)
                self.progress_updated.emit(progress)

        return results

    def _process_files_sequential(self):
        """Process files sequentially."""
        line_number = 13
        results = {}

        for i, file_path in enumerate(self.file_list):
            filename, data = process_single_atr_file(
                (file_path, line_number, self.gpu_accelerator)
            )

            if data:
                results[filename] = data

            progress = int(((i + 1) / len(self.file_list)) * 100)
            self.progress_updated.emit(progress)

        return results

# %% Utility Classes


class switch:
    """Creates a case or switch style statement."""

    def __init__(self, value):
        """Initialize switch with value."""
        self.value = value

    def __iter__(self):
        """Iterate and find the match."""
        yield self.match

    def match(self, *args):
        """Return the match."""
        return self.value in args


class Wrap4With:
    """Used to add context management to a given object."""

    def __init__(self, resource):
        """Initialize wrapper with resource."""
        self._resource = resource

    def __enter__(self):
        """Return the wrapped resource upon entering the context."""
        return self._resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the resource upon exiting the context."""
        if hasattr(self._resource, 'close'):
            self._resource.close()
        return None

# %% Enhanced Main Window Class


class EnhancedATRMainWindow(QMainWindow):
    """Enhanced main window with modern Qt interface for ATR plotting."""

    def __init__(self):
        """Initialize the enhanced ATR main window."""
        super().__init__()
        self.setWindowTitle("Enhanced ATR AutoPlot with Modern GUI")
        self.setGeometry(100, 100, 900, 700)

        # Initialize configuration
        self.config = ProcessingConfig()

        # Initialize ATR plotter
        self.atr_plotter = None

        # Setup UI
        self._setup_ui()

        # File list
        self.selected_files = []

        # Processing results
        self.processed_data = {}

    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # File selection section
        file_group = QGroupBox("ATR File Selection")
        file_layout = QVBoxLayout(file_group)

        # File list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumHeight(150)
        file_layout.addWidget(self.file_list_widget)

        # Browse button
        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse ATR Files")
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
        self.cpu_spinbox.setMaximum(multiprocessing.cpu_count())
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

        # Plot configuration section
        plot_group = QGroupBox("Plot Configuration")
        plot_layout = QVBoxLayout(plot_group)

        # Plot title
        plot_title_layout = QHBoxLayout()
        plot_title_layout.addWidget(QLabel("Plot Title:"))
        self.plot_title_edit = QLineEdit("GBO Outdoor Antenna Range Pattern")
        plot_title_layout.addWidget(self.plot_title_edit)
        plot_layout.addLayout(plot_title_layout)

        # Dataset name
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_edit = QLineEdit("ATR_Dataset")
        dataset_layout.addWidget(self.dataset_name_edit)
        plot_layout.addLayout(dataset_layout)

        main_layout.addWidget(plot_group)

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

        self.process_button = QPushButton("Process and Plot ATR Files")
        self.process_button.clicked.connect(self._process_and_plot)
        button_layout.addWidget(self.process_button)

        self.save_button = QPushButton("Save Veusz Project")
        self.save_button.clicked.connect(self._save_project)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

        # Initialize status
        self._log_message("Enhanced ATR AutoPlot initialized")
        self._log_message(f"GPU Support: {GPU_AVAILABLE or 'None'}")
        self._log_message(
            f"CPU Cores Available: {multiprocessing.cpu_count()}")

    def _browse_files(self):
        """Open file dialog to select multiple ATR files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("GBO ATR Files (*.atr)")
        file_dialog.setWindowTitle("Select ATR Files")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self._update_file_list()
            self._log_message(f"Selected {len(selected_files)} ATR files")

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
            f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}"
        )

    def _update_cpu_config(self, value):
        """Update CPU cores configuration."""
        self.config.num_processes = value
        self.config.max_workers = value
        self._log_message(f"CPU cores set to: {value}")

    def _update_gpu_config(self, state):
        """Update GPU processing configuration."""
        self.config.enable_gpu_processing = state == Qt.Checked
        self._log_message(
            f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}"
        )

    def _update_opencl_config(self, state):
        """Update OpenCL preference configuration."""
        self.config.use_opencl = state == Qt.Checked
        self._log_message(
            f"OpenCL preference: {'Enabled' if self.config.use_opencl else 'Disabled'}"
        )

    def _log_message(self, message):
        """Add message to status text."""
        self.status_text.append(f"[{self._get_timestamp()}] {message}")

    def _get_timestamp(self):
        """Get current timestamp string."""
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _process_and_plot(self):
        """Process selected ATR files and create plots."""
        if not self.selected_files:
            QMessageBox.warning(
                self, "Warning", "Please select ATR files first.")
            return

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Initialize ATR plotter
        self.atr_plotter = ATRPlotter(
            plot_title=self.plot_title_edit.text(),
            dataset_name=self.dataset_name_edit.text()
        )

        # Start processing thread
        self.processing_thread = ATRProcessingThread(
            self.selected_files,
            self.config
        )

        self.processing_thread.progress_updated.connect(
            self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(
            self._on_processing_finished)
        self.processing_thread.error_occurred.connect(
            self._on_processing_error)

        self.processing_thread.start()
        self._log_message("ATR processing started...")

    def _on_processing_finished(self, results):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)

        successful_files = len([r for r in results.values() if r])
        failed_files = len(results) - successful_files

        self._log_message(
            f"Processing completed: {successful_files} successful, {failed_files} failed"
        )

        if successful_files > 0:
            self.processed_data = results
            self._create_plots(results)
            self.save_button.setEnabled(True)

        if failed_files > 0:
            QMessageBox.warning(
                self, "Processing Warnings",
                f"{failed_files} files failed to process. Check status log for details."
            )

    def _on_processing_error(self, error_message):
        """Handle processing error."""
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self._log_message(f"Processing error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)

    def _create_plots(self, results):
        """Create Veusz plots from processed results."""
        self._log_message("Creating ATR plots...")

        try:
            for filename, data in results.items():
                if data:
                    self.atr_plotter.create_plots_from_data(filename, data)

            self._log_message("ATR plot creation completed")

        except Exception as e:
            self._log_message(f"Plot creation error: {e}")
            QMessageBox.critical(self, "Plot Creation Error", str(e))

    def _save_project(self):
        """Save Veusz project."""
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self, "Save ATR Veusz Project", "",
            "Veusz High Precision Files (*.vszh5)"
        )

        if save_path and self.atr_plotter:
            try:
                self.atr_plotter.save(save_path)
                self._log_message(f"Project saved: {save_path}")

                # Ask to open in Veusz
                reply = QMessageBox.question(
                    self, "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self._open_veusz_gui(save_path)

            except Exception as e:
                QMessageBox.critical(self, "Save Error",
                                     f"Failed to save project: {e}")

    def _open_veusz_gui(self, filename: str):
        """Launch Veusz GUI with generated project file."""
        if sys.platform.startswith('win'):
            veusz_exe = os.path.join(sys.prefix, 'Scripts', 'veusz.exe')
        else:
            veusz_exe = os.path.join(sys.prefix, 'bin', 'veusz')

        if not os.path.exists(veusz_exe):
            QMessageBox.critical(
                self, "Veusz Not Found",
                "Veusz not found in Python environment.\n"
                "Install with: [pip OR conda OR mamba] install veusz"
            )
            return

        try:
            subprocess.Popen([veusz_exe, filename])
        except Exception as e:
            QMessageBox.critical(
                self, "Launch Error", f"Failed to start Veusz: {str(e)}"
            )

# %% ATR Plotter Class


class ATRPlotter:
    """Enhanced ATR plotting class with Veusz integration."""

    def __init__(self, plot_title: str = "GBO Outdoor Antenna Range Pattern",
                 dataset_name: str = "ATR_Dataset"):
        """Initialize ATR plotter.

        Parameters
        ----------
        plot_title : str
            Title for plots.
        dataset_name : str
            Base name for datasets.
        """
        self.plot_title = plot_title
        self.dataset_name = dataset_name

        # Initialize Veusz
        self.doc = vz.Embedded('Enhanced ATR AutoPlotter', hidden=False)
        self.doc.EnableToolbar()

        # Labels for plots
        self.freq_label = 'Frequency (MHz)'
        self.az_label = 'Azimuth (degrees)'
        self.phase_label = 'Phase (degrees)'
        self.mag_label = 'Magnitude (dB)'

        # Track whether overlay pages exist
        self.overlay_created = False

    def create_plots_from_data(self, filename: str, data: Dict[str, Any]):
        """Create Veusz plots from processed ATR data.

        Parameters
        ----------
        filename : str
            Name of the processed file.
        data : Dict[str, Any]
            Processed ATR data dictionary.
        """
        if not data:
            return

        # Create overlay pages if they don't exist
        if not self.overlay_created:
            self._create_overlay_pages()
            self.overlay_created = True

        dataset_name = os.path.splitext(filename)[0]

        # Create individual plots
        self._create_individual_plots(dataset_name, data)

        # Add to overlay plots
        self._add_to_overlay_plots(dataset_name, data)

    def _create_overlay_pages(self):
        """Create overlay pages for magnitude and phase plots."""
        # Create Pages and Graphs for Overlays
        page_all_mag = self.doc.Root.Add('page', name='Overlay_All_mag')
        grid_all_mag = page_all_mag.Add('grid', columns=2)
        graph_all_mag = grid_all_mag.Add('graph', name='Overlay_All_mag')

        page_all_phase = self.doc.Root.Add('page', name='Overlay_All_phase')
        grid_all_phase = page_all_phase.Add('grid', columns=2)
        graph_all_phase = grid_all_phase.Add('graph', name='Overlay_All_phase')

        # Configure magnitude overlay graph
        self._configure_overlay_graph(
            graph_all_mag, 'Overlay of Imported Magnitude',
            self.az_label, self.mag_label, -60, 20, -180, 180
        )

        # Configure phase overlay graph
        self._configure_overlay_graph(
            graph_all_phase, 'Overlay of Imported Phase',
            self.az_label, self.phase_label, -180, 180, -180, 180
        )

        # Set auto color theme
        self.doc.Root.colorTheme.val = 'max128'

    def _configure_overlay_graph(self, graph, title: str, x_label: str,
                                 y_label: str, y_min: float, y_max: float,
                                 x_min: float, x_max: float):
        """Configure an overlay graph with standard settings."""
        with Wrap4With(graph) as g:
            g.Add('label', name='plotTitle')
            g.topMargin.val = '1cm'
            g.plotTitle.Text.size.val = '10pt'
            g.plotTitle.label.val = title
            g.plotTitle.alignHorz.val = 'centre'
            g.plotTitle.yPos.val = 1.05
            g.plotTitle.xPos.val = 0.5

            # Set axis labels
            g.x.label.val = x_label
            g.y.label.val = y_label

            # Grid lines
            g.x.GridLines.hide.val = False
            g.y.GridLines.hide.val = False
            g.x.MinorGridLines.hide.val = False
            g.y.MinorGridLines.hide.val = False

            # Set extents
            g.y.min.val = y_min
            g.y.max.val = y_max
            g.x.min.val = x_min
            g.x.max.val = x_max

    def _create_individual_plots(self, dataset_name: str, data: Dict[str, Any]):
        """Create individual plots for a dataset."""
        # Create dataset names
        freq_name = f"{dataset_name}_freq"
        mag_name = f"{dataset_name}_mag"
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        # Set data in Veusz
        self.doc.SetData(freq_name, [data['frequency']])
        self.doc.SetData(mag_name, data['magnitude'])
        self.doc.SetData(phase_name, data['phase'])
        self.doc.SetData(az_name, data['azimuth_angles'])

        # Tag datasets
        self.doc.TagDatasets(
            dataset_name,
            [freq_name, mag_name, phase_name, az_name]
        )

        # Create individual pages and plots
        self._create_magnitude_page(dataset_name, data)
        self._create_phase_page(dataset_name, data)
        self._create_polar_pages(dataset_name, data)

    def _create_magnitude_page(self, dataset_name: str, data: Dict[str, Any]):
        """Create magnitude plot page."""
        mag_name = f"{dataset_name}_mag"
        az_name = f"{dataset_name}_Az"

        page_mag = self.doc.Root.Add('page', name=mag_name)
        grid_mag = page_mag.Add('grid', columns=2)
        graph_mag = grid_mag.Add('graph', name=f"{dataset_name}_Mag")
        page_mag.notes.val = '\n'.join(data['header_lines'])

        # Configure graph
        self._configure_standard_graph(
            graph_mag, dataset_name.replace('_', ' '),
            self.az_label, self.mag_label, -60, 20, -180, 180
        )

        # Create XY plot
        xy_mag = graph_mag.Add('xy', name=dataset_name)
        self._configure_xy_plot(xy_mag, az_name, mag_name, 'red')

    def _create_phase_page(self, dataset_name: str, data: Dict[str, Any]):
        """Create phase plot page."""
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        page_phase = self.doc.Root.Add('page', name=phase_name)
        grid_phase = page_phase.Add('grid', columns=2)
        graph_phase = grid_phase.Add('graph', name=f"{dataset_name}_Phase")
        page_phase.notes.val = '\n'.join(data['header_lines'])

        # Configure graph
        self._configure_standard_graph(
            graph_phase, dataset_name.replace('_', ' '),
            self.az_label, self.phase_label, -180, 180, -180, 180
        )

        # Create XY plot
        xy_phase = graph_phase.Add('xy', name=dataset_name)
        self._configure_xy_plot(xy_phase, az_name, phase_name, 'red')

    def _create_polar_pages(self, dataset_name: str, data: Dict[str, Any]):
        """Create polar plot pages for magnitude and phase."""
        mag_name = f"{dataset_name}_mag"
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        # Magnitude polar plot
        polar_mag_name = f"{dataset_name}_polar_mag"
        page_polar_mag = self.doc.Root.Add('page', name=polar_mag_name)
        self._add_page_title(page_polar_mag, dataset_name.replace('_', ' '))
        grid_polar_mag = page_polar_mag.Add('grid', columns=2)
        graph_polar_mag = grid_polar_mag.Add(
            'polar', name=f"{dataset_name}_Polar_mag")
        page_polar_mag.notes.val = '\n'.join(data['header_lines'])

        # Configure polar graph for magnitude
        self._configure_polar_graph(graph_polar_mag, -60, 20)
        rtheta_mag = graph_polar_mag.Add('nonorthpoint', name=dataset_name)
        self._configure_polar_plot(rtheta_mag, mag_name, az_name)

        # Phase polar plot
        polar_phase_name = f"{dataset_name}_polar_phase"
        page_polar_phase = self.doc.Root.Add('page', name=polar_phase_name)
        self._add_page_title(page_polar_phase, dataset_name.replace('_', ' '))
        grid_polar_phase = page_polar_phase.Add('grid', columns=2)
        graph_polar_phase = grid_polar_phase.Add(
            'polar', name=f"{dataset_name}_Polar_Phase")
        page_polar_phase.notes.val = '\n'.join(data['header_lines'])

        # Configure polar graph for phase
        self._configure_polar_graph(graph_polar_phase, -180, 180)
        rtheta_phase = graph_polar_phase.Add('nonorthpoint', name=dataset_name)
        self._configure_polar_plot(rtheta_phase, phase_name, az_name)

    def _configure_standard_graph(self, graph, title: str, x_label: str,
                                  y_label: str, y_min: float, y_max: float,
                                  x_min: float, x_max: float):
        """Configure a standard XY graph."""
        with Wrap4With(graph) as g:
            g.Add('label', name='plotTitle')
            g.topMargin.val = '1cm'
            g.plotTitle.Text.size.val = '10pt'
            g.plotTitle.label.val = title
            g.plotTitle.alignHorz.val = 'centre'
            g.plotTitle.yPos.val = 1.05
            g.plotTitle.xPos.val = 0.5

            # Set axis labels
            g.x.label.val = x_label
            g.y.label.val = y_label

            # Grid lines
            g.x.GridLines.hide.val = False
            g.y.GridLines.hide.val = False
            g.x.MinorGridLines.hide.val = False
            g.y.MinorGridLines.hide.val = False

            # Set extents
            g.y.min.val = y_min
            g.y.max.val = y_max
            g.x.min.val = x_min
            g.x.max.val = x_max

    def _configure_xy_plot(self, xy, x_data: str, y_data: str, color: str):
        """Configure an XY plot."""
        with Wrap4With(xy) as plot:
            plot.xData.val = x_data
            plot.yData.val = y_data
            plot.nanHandling = 'break-on'
            plot.marker.val = 'circle'
            plot.markerSize.val = '2pt'
            plot.MarkerLine.color.val = 'transparent'
            plot.MarkerFill.color.val = 'auto'
            plot.MarkerFill.transparency.val = 80
            plot.MarkerFill.style.val = 'solid'
            plot.FillBelow.transparency.val = 90
            plot.FillBelow.style.val = 'solid'
            plot.FillBelow.fillto.val = 'bottom'
            plot.FillBelow.color.val = 'darkgreen'
            plot.FillBelow.hide.val = False
            plot.PlotLine.color.val = color

    def _configure_polar_graph(self, graph, min_radius: float, max_radius: float):
        """Configure a polar graph."""
        with Wrap4With(graph) as g:
            g.topMargin.val = '1cm'
            g.units.val = 'degrees'
            g.direction.val = 'clockwise'
            g.position0.val = 'top'
            g.minradius.val = min_radius
            g.maxradius.val = max_radius

    def _configure_polar_plot(self, plot, data1: str, data2: str):
        """Configure a polar plot."""
        with Wrap4With(plot) as p:
            p.data1.val = data1
            p.data2.val = data2
            p.PlotLine.color.val = 'red'
            p.PlotLine.width.val = '2pt'
            p.MarkerLine.transparency.val = 75
            p.MarkerFill.transparency.val = 75
            p.Fill1.transparency.val = 65
            p.Fill1.color.val = 'green'
            p.Fill1.filltype.val = 'center'
            p.Fill1.hide.val = False

    def _add_page_title(self, page, title: str):
        """Add a title to a page."""
        with Wrap4With(page) as p:
            p.Add('label', name='plotTitle')
            p.plotTitle.Text.size.val = '10pt'
            p.plotTitle.label.val = title
            p.plotTitle.alignHorz.val = 'centre'
            p.plotTitle.yPos.val = 0.95
            p.plotTitle.xPos.val = 0.5

    def _add_to_overlay_plots(self, dataset_name: str, data: Dict[str, Any]):
        """Add dataset to overlay plots."""
        mag_name = f"{dataset_name}_mag"
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        # Get overlay graphs
        page_all_mag = self.doc.Root.Overlay_All_mag
        graph_all_mag = page_all_mag.grid1.Overlay_All_mag

        page_all_phase = self.doc.Root.Overlay_All_phase
        graph_all_phase = page_all_phase.grid1.Overlay_All_phase

        # Add to magnitude overlay
        xy_all_mag = graph_all_mag.Add('xy', name=mag_name)
        self._configure_overlay_xy_plot(xy_all_mag, az_name, mag_name)

        # Add to phase overlay
        xy_all_phase = graph_all_phase.Add('xy', name=phase_name)
        self._configure_overlay_xy_plot(xy_all_phase, az_name, phase_name)

    def _configure_overlay_xy_plot(self, xy, x_data: str, y_data: str):
        """Configure an overlay XY plot."""
        with Wrap4With(xy) as plot:
            plot.xData.val = x_data
            plot.yData.val = y_data
            plot.nanHandling = 'break-on'
            plot.marker.val = 'circle'
            plot.markerSize.val = '2pt'
            plot.MarkerLine.color.val = 'transparent'
            plot.MarkerFill.color.val = 'auto'
            plot.MarkerFill.transparency.val = 80
            plot.MarkerFill.style.val = 'solid'
            plot.FillBelow.transparency.val = 90
            plot.FillBelow.style.val = 'solid'
            plot.FillBelow.fillto.val = 'bottom'
            plot.FillBelow.color.val = 'darkgreen'
            plot.FillBelow.hide.val = True
            plot.PlotLine.color.val = 'auto'

    def save(self, filename: str):
        """Save Veusz document to specified file."""
        filename_root = os.path.splitext(filename)[0]
        filename_hp = filename_root + '.vszh5'
        file_split = os.path.split(filename)
        filename_vsz = (
            file_split[0] + '/Beware_oldVersion/' +
            os.path.splitext(file_split[1])[0] + '_BEWARE.vsz'
        )

        # Save high precision version
        self.doc.Save(filename_hp, mode='hdf5')

        # Save legacy version
        os.makedirs(file_split[0] + '/Beware_oldVersion/', exist_ok=True)
        self.doc.Save(filename_vsz, mode='vsz')

# %% Utility Functions


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : np.ndarray
        x-coordinate(s).
    y : np.ndarray
        y-coordinate(s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (r, theta) where r is radial coordinate(s)
        and theta is angular coordinate(s) in radians.
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta

# %% Main Execution


def main():
    """Execute main function."""
    # Set multiprocessing start method for cross-platform compatibility
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)

    app = QApplication(sys.argv)

    # Create and show main window
    window = EnhancedATRMainWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
