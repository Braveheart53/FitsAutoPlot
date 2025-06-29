"""Enhanced Touchstone AutoPlot with Modern Qt GUI Interface, Time Domain Analysis, and Smith Chart Plotting.
# %% Header Info
This version integrates modern Qt GUI interface with comprehensive Touchstone file processing
capabilities using scikit-rf, including multiprocessing, GPU acceleration, advanced
time domain analysis with gating functionality, and Smith Chart plotting.
# %%% Author
Author: William W. Wallace
Last updated: 2025-06-28

# %%% Nuitka Project Options for Optimal Compilation
# nuitka-project: --enable-plugin=pyside6
nuitka-project: --enable-plugin=pylint-warnings
nuitka-project: --include-package=veusz
nuitka-project: --include-package-data=veusz
nuitka-project: --include-package=qtpy
nuitka-project: --include-package=numpy
nuitka-project: --include-package=scipy
nuitka-project: --include-package=skrf
nuitka-project: --include-package=scikit-rf
nuitka-project: --include-package=matplotlib
nuitka-project: --include-package=multiprocessing
nuitka-project: --include-package=subprocess
nuitka-project: --include-qt-plugins=all
nuitka-project: --nofollow-import-to=pyqt5
nuitka-project: --nofollow-import-to=PyQt5
nuitka-project: --nofollow-import-to=pyqt6
nuitka-project: --nofollow-import-to=PyQt6
nuitka-project: --nofollow-import-to=tkinter
nuitka-project: --nofollow-import-to=tk-inter
nuitka-project: --nofollow-import-to=Tkinter
nuitka-project: --python-flag=no_site
nuitka-project: --python-flag=-O
nuitka-project: --lto=yes

# %%% Nuitka Command Line
nuitka ^
  --mingw64 ^
  --standalone ^
  --onefile ^
  --follow-imports ^
  --enable-plugin=pyside6 ^
  --enable-plugin=pylint-warnings ^
  --nofollow-import-to=pyqt5 ^
  --nofollow-import-to=PyQt5 ^
  --nofollow-import-to=pyqt6 ^
  --nofollow-import-to=PyQt6 ^
  --nofollow-import-to=tkinter ^
  --nofollow-import-to=tk-inter ^
  --nofollow-import-to=Tkinter ^
  --include-package=veusz ^
  --include-package-data=veusz ^
  --include-package=qtpy ^
  --include-package=numpy ^
  --include-package=scipy ^
  --include-package=skrf ^
  --include-package=scikit-rf ^
  --include-package=matplotlib ^
  --include-package=multiprocessing ^
  --include-package=subprocess ^
  --include-qt-plugins=all ^
  --include-data-files="*.py=./" ^
  --python-flag=no_site ^
  --python-flag=-O ^
  --lto=yes ^
  --remove-output ^
  --show-progress ^
  --show-memory ^
  --show-modules ^
  --output-filename=TouchstoneAutoPlot ^
  --output-dir=dist ^
  --windows-console-mode=disable ^
  --assume-yes-for-downloads ^
  TouchstoneAutoPlot.py

# TODO: Fix Smith Chart creation and plotting in veusz and fix time domain
plotting.
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

# %%% Import Math and Scientific Computing Modules
import numpy as np
import scipy.signal.windows as windows
import veusz.embed as vz

# %%% Import RF/Microwave Engineering Modules
import skrf as rf
from skrf import Network
from skrf.time import time_gate

# %%% Import GUI Modules
from qtpy.QtGui import *
from qtpy.QtCore import Qt, QSize, QThread, Signal, QTimer
from qtpy.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
    QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
    QSpinBox, QGroupBox, QListWidget, QSplitter, QLineEdit,
    QTabWidget, QComboBox, QSlider, QDoubleSpinBox, QGridLayout,
    QFormLayout, QFrame
)

# %%% Matplotlib for embedded plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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


@dataclass
class TimeDomainConfig:
    """Configuration class for time domain analysis settings."""

    window_type: str = 'kaiser'
    window_param: float = 6.0
    gate_start: float = 0.0
    gate_stop: float = 1.0
    gate_center: float = 0.5
    gate_span: float = 0.2
    mode: str = 'bandpass'
    method: str = 'fft'
    t_unit: str = 'ns'
    auto_gate: bool = True


@dataclass
class SmithChartConfig:
    """Configuration class for Smith Chart plotting settings."""

    chart_type: str = 'z'  # 'z' for impedance, 'y' for admittance
    draw_labels: bool = True
    draw_vswr: bool = True
    reference_impedance: float = 50.0
    show_legend: bool = True
    grid_color: str = 'gray'
    trace_color: str = 'blue'
    marker_style: str = 'circle'
    marker_size: int = 4


# %% GPU Acceleration Classes
class GPUAccelerator:
    """Handles GPU-accelerated computations with cross-platform support."""

    def __init__(self, enable_gpu=True):
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
                print("CuPy initialized on device: {}".format(cp.cuda.Device()))
            except Exception as e:
                print("CuPy initialization failed: {}".format(e))
                self.gpu_enabled = False

        elif self.backend == "opencl":
            try:
                self.cl_context = cl.create_some_context()
                self.cl_queue = cl.CommandQueue(self.cl_context)
                print("OpenCL initialized: {}".format(self.cl_context.devices))
            except Exception as e:
                print("OpenCL initialization failed: {}".format(e))
                self.gpu_enabled = False

        elif self.backend == "taichi":
            try:
                ti.init(arch=ti.gpu)
                print("Taichi GPU backend initialized")
            except Exception as e:
                print("Taichi GPU initialization failed: {}".format(e))
                ti.init(arch=ti.cpu)
                print("Taichi fallback to CPU")
                self.gpu_enabled = False

    def process_s_parameters(self, s_data):
        """Process S-parameter data with GPU acceleration if available.

        Parameters
        ----------
        s_data : np.ndarray
            S-parameter data array.

        Returns
        -------
        np.ndarray
            Processed S-parameter data.
        """
        if not self.gpu_enabled:
            return np.copy(s_data)

        try:
            if self.backend == "cupy":
                return self._cupy_process(s_data)
            elif self.backend == "opencl":
                return self._opencl_process(s_data)
            elif self.backend == "taichi":
                return self._taichi_process(s_data)
        except Exception as e:
            print("GPU processing failed, falling back to CPU: {}".format(e))
            return np.copy(s_data)

        return np.copy(s_data)

    def _cupy_process(self, data):
        """CuPy-based GPU processing.

        Parameters
        ----------
        data : np.ndarray
            Input data array.

        Returns
        -------
        np.ndarray
            Processed data array.
        """
        gpu_data = cp.asarray(data)
        # Perform any GPU-accelerated computations here
        result = cp.copy(gpu_data)
        return cp.asnumpy(result)

    def _opencl_process(self, data):
        """OpenCL-based GPU processing.

        Parameters
        ----------
        data : np.ndarray
            Input data array.

        Returns
        -------
        np.ndarray
            Processed data array.
        """
        # Basic passthrough with OpenCL memory management
        data_gpu = cl_array.to_device(self.cl_queue, data.astype(np.complex64))
        result = data_gpu.get()
        return result

    def _taichi_process(self, data):
        """Taichi-based GPU processing.

        Parameters
        ----------
        data : np.ndarray
            Input data array.

        Returns
        -------
        np.ndarray
            Processed data array.
        """
        # Basic passthrough with Taichi
        return np.copy(data)


# %% Worker Functions for Multiprocessing
def process_single_touchstone_file(file_info):
    """Process a single Touchstone file with multiprocessing support.

    This function is designed to be used with multiprocessing pools.

    Parameters
    ----------
    file_info : Tuple[str, object]
        Tuple containing (file_path, gpu_accelerator).

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing (filename, processed_data_dict).
    """
    file_path, gpu_accelerator = file_info

    try:
        # Load Touchstone file using scikit-rf
        network = rf.Network(file_path)

        # Extract basic information
        filename = os.path.basename(file_path)
        n_ports = network.nports
        frequency = network.frequency
        s_parameters = network.s

        # Process with GPU if available
        if gpu_accelerator:
            s_processed = gpu_accelerator.process_s_parameters(s_parameters)
        else:
            s_processed = s_parameters

        # Extract header information from Touchstone file
        header_info = []
        try:
            with open(file_path, 'r', encoding='ascii') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('!') or line.startswith('#'):
                        header_info.append(line.strip())
                    elif line.strip() and not line[0].isdigit():
                        header_info.append(line.strip())
        except Exception as e:
            print("Warning: Could not extract header from {}: {}".format(
                filename, e))

        return filename, {
            'network': network,
            'header_info': header_info,
            'n_ports': n_ports,
            'frequency': frequency,
            's_parameters': s_processed,
            'freq_hz': frequency.f,
            'freq_ghz': frequency.f / 1e9,
            's_mag_db': 20 * np.log10(np.abs(s_processed) + 1e-12),
            's_phase_deg': np.angle(s_processed, deg=True),
            'z0': network.z0
        }

    except Exception as e:
        print("Error processing Touchstone file {}: {}".format(
            file_path, str(e)))
        return os.path.basename(file_path), {}


# %% Threading Classes
class TouchstoneProcessingThread(QThread):
    """Thread for handling Touchstone file processing without blocking the GUI."""

    progress_updated = Signal(int)
    processing_finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, file_list, config):
        """Initialize Touchstone processing thread.

        Parameters
        ----------
        file_list : list
            List of Touchstone files to process.
        config : ProcessingConfig
            Processing configuration.
        """
        super().__init__()
        self.file_list = file_list
        self.config = config
        self.gpu_accelerator = GPUAccelerator(config.enable_gpu_processing)

    def run(self):
        """Execute Touchstone file processing in separate thread."""
        try:
            if self.config.enable_multiprocessing and len(self.file_list) > 1:
                results = self._process_files_parallel()
            else:
                results = self._process_files_sequential()

            self.processing_finished.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_files_parallel(self):
        """Process files using multiprocessing.

        Returns
        -------
        dict
            Dictionary containing processed results.
        """
        file_info_list = [
            (file_path, self.gpu_accelerator)
            for file_path in self.file_list
        ]

        results = {}
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_touchstone_file, file_info): file_info[0]
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
        """Process files sequentially.

        Returns
        -------
        dict
            Dictionary containing processed results.
        """
        results = {}

        for i, file_path in enumerate(self.file_list):
            filename, data = process_single_touchstone_file(
                (file_path, self.gpu_accelerator)
            )

            if data:
                results[filename] = data

            progress = int(((i + 1) / len(self.file_list)) * 100)
            self.progress_updated.emit(progress)

        return results


# %% Time Domain Analysis Classes
class TimeDomainProcessor:
    """Handles time domain analysis and gating of S-parameter data."""

    def __init__(self, config):
        """Initialize time domain processor.

        Parameters
        ----------
        config : TimeDomainConfig
            Time domain analysis configuration.
        """
        self.config = config

    def process_network(self, network):
        """Process a network for time domain analysis.

        Parameters
        ----------
        network : Network
            scikit-rf Network object.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing time domain analysis results.
        """
        results = {}

        try:
            # Get time domain representation
            network_td = network.copy()

            # Apply time domain gating
            for i in range(network.nports):
                for j in range(network.nports):
                    s_param = network.s[:, i, j]

                    # Create single-port network for gating
                    temp_network = network.copy()
                    temp_network.s = s_param.reshape(-1, 1, 1)

                    # Apply gating
                    if self.config.auto_gate:
                        gated_network = time_gate(
                            temp_network,
                            mode=self.config.mode,
                            window=(self.config.window_type,
                                    self.config.window_param),
                            method=self.config.method,
                            t_unit=self.config.t_unit
                        )
                    else:
                        gated_network = time_gate(
                            temp_network,
                            start=self.config.gate_start,
                            stop=self.config.gate_stop,
                            mode=self.config.mode,
                            window=(self.config.window_type,
                                    self.config.window_param),
                            method=self.config.method,
                            t_unit=self.config.t_unit
                        )

                    # Store results
                    param_name = "S{}{}".format(i+1, j+1)
                    results["{}_original".format(param_name)] = temp_network
                    results["{}_gated".format(param_name)] = gated_network
                    results["{}_td".format(param_name)
                            ] = temp_network.s_time_db
                    results["{}_td_filtered".format(
                        param_name)] = gated_network.s_time_db

            # Store time vector
            results['time'] = network.frequency.t_ns

        except Exception as e:
            print("Time domain processing error: {}".format(e))
            results['error'] = str(e)

        return results


# %% Smith Chart Processing Classes
class SmithChartProcessor:
    """Handles Smith Chart analysis and plotting of S-parameter data."""

    def __init__(self, config):
        """Initialize Smith Chart processor.

        Parameters
        ----------
        config : SmithChartConfig
            Smith Chart plotting configuration.
        """
        self.config = config

    def process_network_for_smith_chart(self, network, apply_td_filter=False, td_processor=None):
        """Process a network for Smith Chart plotting.

        Parameters
        ----------
        network : Network
            scikit-rf Network object.
        apply_td_filter : bool, optional
            Whether to apply time domain filtering. Default is False.
        td_processor : TimeDomainProcessor, optional
            Time domain processor for filtering.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing Smith Chart data.
        """
        results = {}

        try:
            # Use time domain filtered data if requested
            if apply_td_filter and td_processor:
                td_results = td_processor.process_network(network)
                if 'error' not in td_results:
                    # Create filtered networks for each S-parameter
                    for i in range(network.nports):
                        for j in range(network.nports):
                            param_name = "S{}{}".format(i+1, j+1)
                            gated_key = "{}_gated".format(param_name)
                            if gated_key in td_results:
                                results[param_name] = td_results[gated_key]
                else:
                    # Fall back to original data if filtering failed
                    apply_td_filter = False

            if not apply_td_filter:
                # Use original S-parameter data
                for i in range(network.nports):
                    for j in range(network.nports):
                        param_name = "S{}{}".format(i+1, j+1)
                        # Create single S-parameter network
                        temp_network = network.copy()
                        temp_network.s = network.s[:, i, j].reshape(-1, 1, 1)
                        results[param_name] = temp_network

            # Store metadata
            results['frequency'] = network.frequency
            results['reference_impedance'] = self.config.reference_impedance
            results['filtered'] = apply_td_filter

        except Exception as e:
            print("Smith Chart processing error: {}".format(e))
            results['error'] = str(e)

        return results


# %% Matplotlib Canvas for Embedded Plots
class TouchstonePlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for Touchstone plots."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """Initialize plot canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        width : float, optional
            Figure width in inches. Default is 8.
        height : float, optional
            Figure height in inches. Default is 6.
        dpi : int, optional
            Figure DPI. Default is 100.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        # Configure matplotlib for better appearance
        plt.style.use('default')
        self.fig.patch.set_facecolor('white')

    def plot_s_parameters(self, frequency, s_data, title="S-Parameters"):
        """Plot S-parameters on the canvas.

        Parameters
        ----------
        frequency : np.ndarray
            Frequency array.
        s_data : np.ndarray
            S-parameter data.
        title : str, optional
            Plot title. Default is "S-Parameters".
        """
        self.fig.clear()

        n_ports = int(np.sqrt(s_data.shape[1]))

        # Create subplots
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        # Plot magnitude
        for i in range(n_ports):
            for j in range(n_ports):
                idx = i * n_ports + j
                mag_db = 20 * np.log10(np.abs(s_data[:, idx]) + 1e-12)
                ax1.plot(frequency / 1e9, mag_db,
                         label='S{}{}'.format(i+1, j+1))

        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('{} - Magnitude'.format(title))
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot phase
        for i in range(n_ports):
            for j in range(n_ports):
                idx = i * n_ports + j
                phase_deg = np.angle(s_data[:, idx], deg=True)
                ax2.plot(frequency / 1e9, phase_deg,
                         label='S{}{}'.format(i+1, j+1))

        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('{} - Phase'.format(title))
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        self.draw()

    def plot_time_domain(self, time, td_data, td_filtered_data=None, title="Time Domain"):
        """Plot time domain data on the canvas.

        Parameters
        ----------
        time : np.ndarray
            Time array.
        td_data : np.ndarray
            Time domain data (unfiltered).
        td_filtered_data : np.ndarray, optional
            Time domain data (filtered).
        title : str, optional
            Plot title. Default is "Time Domain".
        """
        self.fig.clear()

        ax = self.fig.add_subplot(111)

        # Plot unfiltered data
        ax.plot(time, np.abs(td_data), 'o:', alpha=0.7, markersize=2,
                label='Unfiltered', linestyle='dotted')

        # Plot filtered data if available
        if td_filtered_data is not None:
            ax.plot(time, np.abs(td_filtered_data), '-', linewidth=2,
                    label='Filtered')

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Magnitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        self.draw()

    def plot_smith_chart(self, network, title="Smith Chart", chart_type='z',
                         draw_labels=True, draw_vswr=True):
        """Plot Smith chart on the canvas.

        Parameters
        ----------
        network : Network
            scikit-rf Network object.
        title : str, optional
            Plot title. Default is "Smith Chart".
        chart_type : str, optional
            Chart type ('z' for impedance, 'y' for admittance). Default is 'z'.
        draw_labels : bool, optional
            Whether to draw labels. Default is True.
        draw_vswr : bool, optional
            Whether to draw VSWR circles. Default is True.
        """
        self.fig.clear()

        ax = self.fig.add_subplot(111)

        try:
            # Use scikit-rf's built-in Smith chart plotting
            network.plot_s_smith(
                ax=ax,
                chart_type=chart_type,
                draw_labels=draw_labels,
                draw_vswr=draw_vswr,
                show_legend=True
            )

            ax.set_title(title)

        except Exception as e:
            # Fallback to basic complex plot if Smith chart fails
            print("Smith chart plotting failed: {}. Using complex plot.".format(e))

            for i in range(network.nports):
                for j in range(network.nports):
                    s_param = network.s[:, i, j]
                    ax.plot(s_param.real, s_param.imag,
                            label='S{}{}'.format(i+1, j+1),
                            marker='o', markersize=3)

            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.set_title('{} (Complex Plane)'.format(title))
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')

        self.draw()


# %% Main Application Classes
class TouchstoneMainWindow(QMainWindow):
    """Main window for Touchstone AutoPlot application."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle(
            "Enhanced Touchstone AutoPlot with Modern GUI, Time Domain Analysis, and Smith Chart Plotting")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize configurations
        self.config = ProcessingConfig()
        self.td_config = TimeDomainConfig()
        self.smith_config = SmithChartConfig()

        # Initialize components
        self.touchstone_plotter = None
        self.td_processor = TimeDomainProcessor(self.td_config)
        self.smith_processor = SmithChartProcessor(self.smith_config)

        # Data storage
        self.selected_files = []
        self.processed_data = {}
        self.td_results = {}
        self.smith_results = {}

        # Setup UI
        self._setup_ui()

        # Initialize status
        self._log_message("Enhanced Touchstone AutoPlot initialized")
        self._log_message("GPU Support: {}".format(GPU_AVAILABLE or 'None'))
        self._log_message("CPU Cores Available: {}".format(
            multiprocessing.cpu_count()))

    def _setup_ui(self):
        """Set up the user interface with tab widget."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Setup tabs
        self._setup_main_tab()
        self._setup_time_domain_tab()
        self._setup_smith_chart_tab()

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

        self.process_button = QPushButton("Process Touchstone Files")
        self.process_button.clicked.connect(self._process_files)
        button_layout.addWidget(self.process_button)

        self.save_button = QPushButton("Save Veusz Project")
        self.save_button.clicked.connect(self._save_project)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def _setup_main_tab(self):
        """Setup the main processing tab."""
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "Main Processing")

        layout = QVBoxLayout(main_tab)

        # File selection section
        file_group = QGroupBox("Touchstone File Selection")
        file_layout = QVBoxLayout(file_group)

        # File list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumHeight(150)
        file_layout.addWidget(self.file_list_widget)

        # Browse buttons
        browse_layout = QHBoxLayout()

        self.browse_button = QPushButton("Browse Touchstone Files")
        self.browse_button.clicked.connect(self._browse_files)
        browse_layout.addWidget(self.browse_button)

        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        browse_layout.addWidget(self.clear_button)

        file_layout.addLayout(browse_layout)
        layout.addWidget(file_group)

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

        layout.addWidget(options_group)

        # Plot configuration section
        plot_group = QGroupBox("Plot Configuration")
        plot_layout = QVBoxLayout(plot_group)

        # Plot title
        plot_title_layout = QHBoxLayout()
        plot_title_layout.addWidget(QLabel("Plot Title:"))
        self.plot_title_edit = QLineEdit("Touchstone S-Parameter Analysis")
        plot_title_layout.addWidget(self.plot_title_edit)
        plot_layout.addLayout(plot_title_layout)

        # Dataset name
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_edit = QLineEdit("Touchstone_Dataset")
        dataset_layout.addWidget(self.dataset_name_edit)
        plot_layout.addLayout(dataset_layout)

        layout.addWidget(plot_group)

    def _setup_time_domain_tab(self):
        """Setup the time domain analysis tab."""
        td_tab = QWidget()
        self.tab_widget.addTab(td_tab, "Time Domain Analysis")

        layout = QHBoxLayout(td_tab)

        # Left panel for controls
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)

        # File selection for preview
        file_select_group = QGroupBox("File Selection for Preview")
        file_select_layout = QVBoxLayout(file_select_group)

        self.td_file_combo = QComboBox()
        self.td_file_combo.currentTextChanged.connect(self._update_td_preview)
        file_select_layout.addWidget(self.td_file_combo)

        controls_layout.addWidget(file_select_group)

        # Window settings
        window_group = QGroupBox("Window Settings")
        window_layout = QFormLayout(window_group)

        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(
            ['kaiser', 'hamming', 'hann', 'blackman', 'boxcar'])
        self.window_type_combo.setCurrentText(self.td_config.window_type)
        self.window_type_combo.currentTextChanged.connect(
            self._update_window_config)
        window_layout.addRow("Window Type:", self.window_type_combo)

        self.window_param_spin = QDoubleSpinBox()
        self.window_param_spin.setRange(0.1, 20.0)
        self.window_param_spin.setValue(self.td_config.window_param)
        self.window_param_spin.valueChanged.connect(self._update_window_param)
        window_layout.addRow("Window Parameter:", self.window_param_spin)

        controls_layout.addWidget(window_group)

        # Gating settings
        gate_group = QGroupBox("Gating Settings")
        gate_layout = QFormLayout(gate_group)

        self.auto_gate_checkbox = QCheckBox("Auto Gate")
        self.auto_gate_checkbox.setChecked(self.td_config.auto_gate)
        self.auto_gate_checkbox.stateChanged.connect(self._update_auto_gate)
        gate_layout.addRow(self.auto_gate_checkbox)

        self.gate_start_spin = QDoubleSpinBox()
        self.gate_start_spin.setRange(-100.0, 100.0)
        self.gate_start_spin.setValue(self.td_config.gate_start)
        self.gate_start_spin.setSuffix(' ns')
        self.gate_start_spin.valueChanged.connect(self._update_gate_start)
        gate_layout.addRow("Gate Start:", self.gate_start_spin)

        self.gate_stop_spin = QDoubleSpinBox()
        self.gate_stop_spin.setRange(-100.0, 100.0)
        self.gate_stop_spin.setValue(self.td_config.gate_stop)
        self.gate_stop_spin.setSuffix(' ns')
        self.gate_stop_spin.valueChanged.connect(self._update_gate_stop)
        gate_layout.addRow("Gate Stop:", self.gate_stop_spin)

        # Enable/disable manual gating based on auto_gate
        self.gate_start_spin.setEnabled(not self.td_config.auto_gate)
        self.gate_stop_spin.setEnabled(not self.td_config.auto_gate)

        controls_layout.addWidget(gate_group)

        # Mode and method settings
        method_group = QGroupBox("Processing Settings")
        method_layout = QFormLayout(method_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['bandpass', 'bandstop'])
        self.mode_combo.setCurrentText(self.td_config.mode)
        self.mode_combo.currentTextChanged.connect(self._update_mode)
        method_layout.addRow("Mode:", self.mode_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(['fft', 'rfft', 'convolution'])
        self.method_combo.setCurrentText(self.td_config.method)
        self.method_combo.currentTextChanged.connect(self._update_method)
        method_layout.addRow("Method:", self.method_combo)

        controls_layout.addWidget(method_group)

        # Processing scope selection
        scope_group = QGroupBox("Processing Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.td_process_selected_only = QCheckBox(
            "Process only selected file from dropdown")
        self.td_process_selected_only.setChecked(False)
        scope_layout.addWidget(self.td_process_selected_only)

        controls_layout.addWidget(scope_group)

        # Process button for time domain
        self.td_process_button = QPushButton("Process Time Domain")
        self.td_process_button.clicked.connect(self._process_time_domain)
        self.td_process_button.setEnabled(False)
        controls_layout.addWidget(self.td_process_button)

        controls_layout.addStretch()

        # Right panel for plot
        self.td_plot_canvas = TouchstonePlotCanvas(td_tab, width=8, height=6)

        # Add to layout
        layout.addWidget(controls_widget)
        layout.addWidget(self.td_plot_canvas)

    def _setup_smith_chart_tab(self):
        """Setup the Smith Chart analysis tab."""
        smith_tab = QWidget()
        self.tab_widget.addTab(smith_tab, "Smith Chart Analysis")

        layout = QHBoxLayout(smith_tab)

        # Left panel for controls
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls_widget)

        # File selection for preview
        file_select_group = QGroupBox("File Selection for Preview")
        file_select_layout = QVBoxLayout(file_select_group)

        self.smith_file_combo = QComboBox()
        self.smith_file_combo.currentTextChanged.connect(
            self._update_smith_preview)
        file_select_layout.addWidget(self.smith_file_combo)

        controls_layout.addWidget(file_select_group)

        # Smith Chart settings
        smith_group = QGroupBox("Smith Chart Settings")
        smith_layout = QFormLayout(smith_group)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(['z (Impedance)', 'y (Admittance)'])
        self.chart_type_combo.setCurrentText('z (Impedance)')
        self.chart_type_combo.currentTextChanged.connect(
            self._update_chart_type)
        smith_layout.addRow("Chart Type:", self.chart_type_combo)

        self.ref_impedance_spin = QDoubleSpinBox()
        self.ref_impedance_spin.setRange(1.0, 1000.0)
        self.ref_impedance_spin.setValue(self.smith_config.reference_impedance)
        self.ref_impedance_spin.setSuffix(' Ω')
        self.ref_impedance_spin.valueChanged.connect(
            self._update_ref_impedance)
        smith_layout.addRow("Reference Impedance:", self.ref_impedance_spin)

        self.draw_labels_checkbox = QCheckBox("Draw Labels")
        self.draw_labels_checkbox.setChecked(self.smith_config.draw_labels)
        self.draw_labels_checkbox.stateChanged.connect(
            self._update_draw_labels)
        smith_layout.addRow(self.draw_labels_checkbox)

        self.draw_vswr_checkbox = QCheckBox("Draw VSWR Circles")
        self.draw_vswr_checkbox.setChecked(self.smith_config.draw_vswr)
        self.draw_vswr_checkbox.stateChanged.connect(self._update_draw_vswr)
        smith_layout.addRow(self.draw_vswr_checkbox)

        controls_layout.addWidget(smith_group)

        # Time domain filtering option
        td_filter_group = QGroupBox("Time Domain Filtering")
        td_filter_layout = QVBoxLayout(td_filter_group)

        self.smith_enable_td_filter = QCheckBox(
            "Enable time domain filtering on Smith Chart data")
        self.smith_enable_td_filter.setChecked(False)
        self.smith_enable_td_filter.stateChanged.connect(
            self._update_smith_preview)
        td_filter_layout.addWidget(self.smith_enable_td_filter)

        controls_layout.addWidget(td_filter_group)

        # Processing scope selection
        scope_group = QGroupBox("Processing Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.smith_process_selected_only = QCheckBox(
            "Generate Smith Chart only for selected file from dropdown")
        self.smith_process_selected_only.setChecked(False)
        scope_layout.addWidget(self.smith_process_selected_only)

        controls_layout.addWidget(scope_group)

        # Process button for Smith Charts
        self.smith_process_button = QPushButton(
            "Generate Smith Charts in Veusz")
        self.smith_process_button.clicked.connect(self._process_smith_charts)
        self.smith_process_button.setEnabled(False)
        controls_layout.addWidget(self.smith_process_button)

        controls_layout.addStretch()

        # Right panel for plot
        self.smith_plot_canvas = TouchstonePlotCanvas(
            smith_tab, width=8, height=6)

        # Add to layout
        layout.addWidget(controls_widget)
        layout.addWidget(self.smith_plot_canvas)

    def _browse_files(self):
        """Open file dialog to select multiple Touchstone files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Touchstone Files (*.s*p)")
        file_dialog.setWindowTitle("Select Touchstone Files")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self._update_file_list()
            self._log_message(
                "Selected {} Touchstone files".format(len(selected_files)))

    def _clear_files(self):
        """Clear the selected files list."""
        self.selected_files.clear()
        self.processed_data.clear()
        self.td_results.clear()
        self.smith_results.clear()
        self._update_file_list()
        self._update_td_file_combo()
        self._update_smith_file_combo()
        self._log_message("File list cleared")

    def _update_file_list(self):
        """Update the file list widget."""
        self.file_list_widget.clear()
        for file_path in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(file_path))

    def _update_td_file_combo(self):
        """Update the time domain file selection combo box."""
        self.td_file_combo.clear()
        if self.processed_data:
            self.td_file_combo.addItems(list(self.processed_data.keys()))
        self.td_process_button.setEnabled(len(self.processed_data) > 0)

    def _update_smith_file_combo(self):
        """Update the Smith Chart file selection combo box."""
        self.smith_file_combo.clear()
        if self.processed_data:
            self.smith_file_combo.addItems(list(self.processed_data.keys()))
        self.smith_process_button.setEnabled(len(self.processed_data) > 0)

    def _update_mp_config(self, state):
        """Update multiprocessing configuration.

        Parameters
        ----------
        state : int
            Checkbox state.
        """
        self.config.enable_multiprocessing = state == Qt.Checked
        status_msg = "Multiprocessing: {}".format(
            'Enabled' if self.config.enable_multiprocessing else 'Disabled'
        )
        self._log_message(status_msg)

    def _update_cpu_config(self, value):
        """Update CPU cores configuration.

        Parameters
        ----------
        value : int
            Number of CPU cores.
        """
        self.config.num_processes = value
        self.config.max_workers = value
        self._log_message("CPU cores set to: {}".format(value))

    def _update_gpu_config(self, state):
        """Update GPU processing configuration.

        Parameters
        ----------
        state : int
            Checkbox state.
        """
        self.config.enable_gpu_processing = state == Qt.Checked
        status_msg = "GPU processing: {}".format(
            'Enabled' if self.config.enable_gpu_processing else 'Disabled'
        )
        self._log_message(status_msg)

    def _update_window_config(self, window_type):
        """Update window type configuration.

        Parameters
        ----------
        window_type : str
            Window type selection.
        """
        self.td_config.window_type = window_type
        self._update_td_preview()

    def _update_window_param(self, value):
        """Update window parameter configuration.

        Parameters
        ----------
        value : float
            Window parameter value.
        """
        self.td_config.window_param = value
        self._update_td_preview()

    def _update_auto_gate(self, state):
        """Update auto gate configuration.

        Parameters
        ----------
        state : int
            Checkbox state.
        """
        self.td_config.auto_gate = state == Qt.Checked
        self.gate_start_spin.setEnabled(not self.td_config.auto_gate)
        self.gate_stop_spin.setEnabled(not self.td_config.auto_gate)
        self._update_td_preview()

    def _update_gate_start(self, value):
        """Update gate start configuration.

        Parameters
        ----------
        value : float
            Gate start value.
        """
        self.td_config.gate_start = value
        self._update_td_preview()

    def _update_gate_stop(self, value):
        """Update gate stop configuration.

        Parameters
        ----------
        value : float
            Gate stop value.
        """
        self.td_config.gate_stop = value
        self._update_td_preview()

    def _update_mode(self, mode):
        """Update processing mode configuration.

        Parameters
        ----------
        mode : str
            Processing mode.
        """
        self.td_config.mode = mode
        self._update_td_preview()

    def _update_method(self, method):
        """Update processing method configuration.

        Parameters
        ----------
        method : str
            Processing method.
        """
        self.td_config.method = method
        self._update_td_preview()

    def _update_chart_type(self, chart_type_text):
        """Update Smith Chart type configuration.

        Parameters
        ----------
        chart_type_text : str
            Chart type selection text.
        """
        if 'Impedance' in chart_type_text:
            self.smith_config.chart_type = 'z'
        else:
            self.smith_config.chart_type = 'y'
        self._update_smith_preview()

    def _update_ref_impedance(self, value):
        """Update reference impedance configuration.

        Parameters
        ----------
        value : float
            Reference impedance value.
        """
        self.smith_config.reference_impedance = value
        self._update_smith_preview()

    def _update_draw_labels(self, state):
        """Update draw labels configuration.

        Parameters
        ----------
        state : int
            Checkbox state.
        """
        self.smith_config.draw_labels = state == Qt.Checked
        self._update_smith_preview()

    def _update_draw_vswr(self, state):
        """Update draw VSWR configuration.

        Parameters
        ----------
        state : int
            Checkbox state.
        """
        self.smith_config.draw_vswr = state == Qt.Checked
        self._update_smith_preview()

    def _update_td_preview(self):
        """Update the time domain preview plot."""
        if not self.processed_data:
            return

        current_file = self.td_file_combo.currentText()
        if not current_file or current_file not in self.processed_data:
            return

        try:
            # Get current file data
            data = self.processed_data[current_file]
            network = data['network']

            # Update time domain processor config
            self.td_processor.config = self.td_config

            # Process for preview (just S11 or first available parameter)
            if network.nports >= 1:
                # Create single-port network for preview
                preview_network = network.copy()
                preview_network.s = network.s[:, 0, 0].reshape(-1, 1, 1)

                # Get time domain data
                time_ns = network.frequency.t_ns
                td_unfiltered = preview_network.s_time_db

                # Apply gating for preview
                try:
                    if self.td_config.auto_gate:
                        gated_network = time_gate(
                            preview_network,
                            mode=self.td_config.mode,
                            window=(self.td_config.window_type,
                                    self.td_config.window_param),
                            method=self.td_config.method,
                            t_unit=self.td_config.t_unit
                        )
                    else:
                        gated_network = time_gate(
                            preview_network,
                            start=self.td_config.gate_start,
                            stop=self.td_config.gate_stop,
                            mode=self.td_config.mode,
                            window=(self.td_config.window_type,
                                    self.td_config.window_param),
                            method=self.td_config.method,
                            t_unit=self.td_config.t_unit
                        )

                    td_filtered = gated_network.s_time_db

                    # Update plot
                    plot_title = "{} - S11 Time Domain Preview".format(
                        current_file)
                    self.td_plot_canvas.plot_time_domain(
                        time_ns, td_unfiltered[:, 0, 0], td_filtered[:, 0, 0],
                        plot_title
                    )

                except Exception as e:
                    self._log_message(
                        "Time domain preview error: {}".format(e))
                    # Plot only unfiltered data
                    plot_title = "{} - S11 Time Domain (Unfiltered)".format(
                        current_file)
                    self.td_plot_canvas.plot_time_domain(
                        time_ns, td_unfiltered[:, 0, 0], None,
                        plot_title
                    )

        except Exception as e:
            self._log_message("Preview update error: {}".format(e))

    def _update_smith_preview(self):
        """Update the Smith Chart preview plot."""
        if not self.processed_data:
            return

        current_file = self.smith_file_combo.currentText()
        if not current_file or current_file not in self.processed_data:
            return

        try:
            # Get current file data
            data = self.processed_data[current_file]
            network = data['network']

            # Update Smith Chart processor config
            self.smith_processor.config = self.smith_config

            # Apply time domain filtering if requested
            apply_td_filter = self.smith_enable_td_filter.isChecked()

            if apply_td_filter:
                smith_data = self.smith_processor.process_network_for_smith_chart(
                    network, apply_td_filter=True, td_processor=self.td_processor
                )

                if 'error' not in smith_data and 'S11' in smith_data:
                    preview_network = smith_data['S11']
                    plot_title = "{} - S11 Smith Chart (Time Domain Filtered)".format(
                        current_file)
                else:
                    # Fall back to original if filtering failed
                    preview_network = network.copy()
                    preview_network.s = network.s[:, 0, 0].reshape(-1, 1, 1)
                    plot_title = "{} - S11 Smith Chart (Original)".format(
                        current_file)
            else:
                # Use original network
                preview_network = network.copy()
                preview_network.s = network.s[:, 0, 0].reshape(-1, 1, 1)
                plot_title = "{} - S11 Smith Chart".format(current_file)

            # Update plot
            self.smith_plot_canvas.plot_smith_chart(
                preview_network,
                title=plot_title,
                chart_type=self.smith_config.chart_type,
                draw_labels=self.smith_config.draw_labels,
                draw_vswr=self.smith_config.draw_vswr
            )

        except Exception as e:
            self._log_message("Smith Chart preview error: {}".format(e))

    def _process_files(self):
        """Process selected Touchstone files."""
        if not self.selected_files:
            QMessageBox.warning(
                self, "Warning", "Please select Touchstone files first.")
            return

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Initialize Touchstone plotter
        self.touchstone_plotter = TouchstonePlotter(
            plot_title=self.plot_title_edit.text(),
            dataset_name=self.dataset_name_edit.text()
        )

        # Start processing thread
        self.processing_thread = TouchstoneProcessingThread(
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
        self._log_message("Touchstone processing started...")

    def _on_processing_finished(self, results):
        """Handle processing completion.

        Parameters
        ----------
        results : dict
            Processing results dictionary.
        """
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)

        successful_files = len([r for r in results.values() if r])
        failed_files = len(results) - successful_files

        completion_msg = "Processing completed: {} successful, {} failed".format(
            successful_files, failed_files
        )
        self._log_message(completion_msg)

        if successful_files > 0:
            self.processed_data = results
            self._update_td_file_combo()
            self._update_smith_file_combo()
            self._create_plots(results)
            self.save_button.setEnabled(True)

        if failed_files > 0:
            warning_msg = "{} files failed to process. Check status log for details.".format(
                failed_files
            )
            QMessageBox.warning(self, "Processing Warnings", warning_msg)

    def _on_processing_error(self, error_message):
        """Handle processing error.

        Parameters
        ----------
        error_message : str
            Error message string.
        """
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self._log_message("Processing error: {}".format(error_message))
        QMessageBox.critical(self, "Processing Error", error_message)

    def _process_time_domain(self):
        """Process time domain analysis for all files."""
        if not self.processed_data:
            QMessageBox.warning(
                self, "Warning", "Please process Touchstone files first.")
            return

        self._log_message("Starting time domain analysis...")

        try:
            # Update processor config
            self.td_processor.config = self.td_config

            # Determine which files to process
            if self.td_process_selected_only.isChecked():
                current_file = self.td_file_combo.currentText()
                if current_file and current_file in self.processed_data:
                    files_to_process = {
                        current_file: self.processed_data[current_file]}
                else:
                    QMessageBox.warning(
                        self, "Warning", "No file selected for processing.")
                    return
            else:
                files_to_process = self.processed_data

            # Process each file
            for filename, data in files_to_process.items():
                network = data['network']
                td_result = self.td_processor.process_network(network)
                self.td_results[filename] = td_result

                # Create time domain plots in Veusz
                self._create_time_domain_plots(filename, data, td_result)

            scope_msg = "selected file" if self.td_process_selected_only.isChecked() else "all files"
            self._log_message(
                "Time domain analysis completed for {}".format(scope_msg))

        except Exception as e:
            self._log_message("Time domain analysis error: {}".format(e))
            QMessageBox.critical(self, "Time Domain Error", str(e))

    def _process_smith_charts(self):
        """Process Smith Chart analysis for files."""
        if not self.processed_data:
            QMessageBox.warning(
                self, "Warning", "Please process Touchstone files first.")
            return

        self._log_message("Starting Smith Chart analysis...")

        try:
            # Update processor config
            self.smith_processor.config = self.smith_config

            # Determine which files to process
            if self.smith_process_selected_only.isChecked():
                current_file = self.smith_file_combo.currentText()
                if current_file and current_file in self.processed_data:
                    files_to_process = {
                        current_file: self.processed_data[current_file]}
                else:
                    QMessageBox.warning(
                        self, "Warning", "No file selected for processing.")
                    return
            else:
                files_to_process = self.processed_data

            # Process each file
            apply_td_filter = self.smith_enable_td_filter.isChecked()

            for filename, data in files_to_process.items():
                network = data['network']
                smith_result = self.smith_processor.process_network_for_smith_chart(
                    network, apply_td_filter=apply_td_filter, td_processor=self.td_processor
                )
                self.smith_results[filename] = smith_result

                # Create Smith Chart plots in Veusz
                self._create_smith_chart_plots(filename, data, smith_result)

            scope_msg = "selected file" if self.smith_process_selected_only.isChecked() else "all files"
            filter_msg = " (with time domain filtering)" if apply_td_filter else ""
            self._log_message(
                "Smith Chart analysis completed for {}{}".format(scope_msg, filter_msg))

        except Exception as e:
            self._log_message("Smith Chart analysis error: {}".format(e))
            QMessageBox.critical(self, "Smith Chart Error", str(e))

    def _create_plots(self, results):
        """Create Veusz plots from processed results.

        Parameters
        ----------
        results : dict
            Processing results dictionary.
        """
        self._log_message("Creating Touchstone plots...")

        try:
            for filename, data in results.items():
                if data:
                    self.touchstone_plotter.create_plots_from_data(
                        filename, data)

            self._log_message("Touchstone plot creation completed")

        except Exception as e:
            self._log_message("Plot creation error: {}".format(e))
            QMessageBox.critical(self, "Plot Creation Error", str(e))

    def _create_time_domain_plots(self, filename, data, td_result):
        """Create time domain plots in Veusz.

        Parameters
        ----------
        filename : str
            Name of the file.
        data : dict
            Original data dictionary.
        td_result : dict
            Time domain analysis results.
        """
        if not self.touchstone_plotter or 'error' in td_result:
            return

        try:
            self.touchstone_plotter.create_time_domain_plots(
                filename, data, td_result)

        except Exception as e:
            self._log_message("Time domain plot creation error: {}".format(e))

    def _create_smith_chart_plots(self, filename, data, smith_result):
        """Create Smith Chart plots in Veusz.

        Parameters
        ----------
        filename : str
            Name of the file.
        data : dict
            Original data dictionary.
        smith_result : dict
            Smith Chart analysis results.
        """
        if not self.touchstone_plotter or 'error' in smith_result:
            return

        try:
            self.touchstone_plotter.create_smith_chart_plots(
                filename, data, smith_result)

        except Exception as e:
            self._log_message("Smith Chart plot creation error: {}".format(e))

    def _save_project(self):
        """Save Veusz project."""
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self, "Save Touchstone Veusz Project", "",
            "Veusz High Precision Files (*.vszh5)"
        )

        if save_path and self.touchstone_plotter:
            try:
                self.touchstone_plotter.save(save_path)
                self._log_message("Project saved: {}".format(save_path))

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
                                     "Failed to save project: {}".format(e))

    def _open_veusz_gui(self, filename):
        """Launch Veusz GUI with generated project file.

        Parameters
        ----------
        filename : str
            Path to the Veusz project file.
        """
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
                self, "Launch Error", "Failed to start Veusz: {}".format(
                    str(e))
            )

    def _log_message(self, message):
        """Add message to status text.

        Parameters
        ----------
        message : str
            Message to log.
        """
        timestamp = self._get_timestamp()
        self.status_text.append("[{}] {}".format(timestamp, message))

    def _get_timestamp(self):
        """Get current timestamp string.

        Returns
        -------
        str
            Formatted timestamp string.
        """
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")


# %% Touchstone Plotter Class
class TouchstonePlotter:
    """Enhanced Touchstone plotting class with Veusz integration."""

    def __init__(self, plot_title="Touchstone S-Parameter Analysis",
                 dataset_name="Touchstone_Dataset"):
        """Initialize Touchstone plotter.

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
        self.doc = vz.Embedded('Enhanced Touchstone AutoPlotter', hidden=False)
        self.doc.EnableToolbar()

        # Labels for plots
        self.freq_label = 'Frequency (GHz)'
        self.mag_label = 'Magnitude (dB)'
        self.phase_label = 'Phase (degrees)'
        self.time_label = 'Time (ns)'

        # Track overlay pages
        self.overlay_created = False

    def create_plots_from_data(self, filename, data):
        """Create Veusz plots from processed Touchstone data.

        Parameters
        ----------
        filename : str
            Name of the processed file.
        data : Dict[str, Any]
            Processed Touchstone data dictionary.
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

    def create_time_domain_plots(self, filename, data, td_result):
        """Create time domain plots in Veusz.

        Parameters
        ----------
        filename : str
            Name of the file.
        data : Dict[str, Any]
            Original data dictionary.
        td_result : Dict[str, Any]
            Time domain analysis results.
        """
        # TODO: This needs fixed
        if 'error' in td_result:
            return

        dataset_name = os.path.splitext(filename)[0]

        # Create time domain pages for each S-parameter
        network = data['network']
        n_ports = network.nports

        for i in range(n_ports):
            for j in range(n_ports):
                param_name = "S{}{}".format(i+1, j+1)
                td_key = "{}_td".format(param_name)
                tdf_key = "{}_td_filtered".format(param_name)

                if td_key in td_result and tdf_key in td_result:
                    self._create_time_domain_page(
                        dataset_name, param_name, data, td_result
                    )

    def create_smith_chart_plots(self, filename, data, smith_result):
        """Create Smith Chart plots in Veusz.

        Parameters
        ----------
        filename : str
            Name of the file.
        data : Dict[str, Any]
            Original data dictionary.
        smith_result : Dict[str, Any]
            Smith Chart analysis results.
        """
        # TODO: Something amiss here, check it out
        if 'error' in smith_result:
            return

        dataset_name = os.path.splitext(filename)[0]

        # Create Smith Chart pages for each S-parameter
        network = data['network']
        n_ports = network.nports

        for i in range(n_ports):
            for j in range(n_ports):
                param_name = "S{}{}".format(i+1, j+1)
                if param_name in smith_result:
                    self._create_smith_chart_page(
                        dataset_name, param_name, data, smith_result
                    )

    def _create_overlay_pages(self):
        """Create overlay pages for magnitude and phase plots."""
        # Create overlay pages for S-parameters
        page_all_mag = self.doc.Root.Add('page', name='Overlay_All_Magnitude')
        grid_all_mag = page_all_mag.Add('grid', columns=2)
        graph_all_mag = grid_all_mag.Add('graph', name='Overlay_All_Magnitude')

        page_all_phase = self.doc.Root.Add('page', name='Overlay_All_Phase')
        grid_all_phase = page_all_phase.Add('grid', columns=2)
        graph_all_phase = grid_all_phase.Add('graph', name='Overlay_All_Phase')

        # Configure overlay graphs
        self._configure_overlay_graph(
            graph_all_mag, 'Overlay of All S-Parameter Magnitudes',
            self.freq_label, self.mag_label, -60, 20, 0, 20
        )

        self._configure_overlay_graph(
            graph_all_phase, 'Overlay of All S-Parameter Phases',
            self.freq_label, self.phase_label, -180, 180, 0, 20
        )

        # Set auto color theme
        self.doc.Root.colorTheme.val = 'max128'

    def _configure_overlay_graph(self, graph, title, x_label,
                                 y_label, y_min, y_max,
                                 x_min, x_max):
        """Configure an overlay graph with standard settings.

        Parameters
        ----------
        graph : veusz graph object
            Graph to configure.
        title : str
            Graph title.
        x_label : str
            X-axis label.
        y_label : str
            Y-axis label.
        y_min : float
            Y-axis minimum.
        y_max : float
            Y-axis maximum.
        x_min : float
            X-axis minimum.
        x_max : float
            X-axis maximum.
        """
        with self._wrap_widget(graph) as g:
            g.Add('label', name='plotTitle')
            g.topMargin.val = '1cm'
            g.plotTitle.Text.size.val = '12pt'
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

    def _create_individual_plots(self, dataset_name, data):
        """Create individual plots for a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        data : dict
            Data dictionary.
        """
        network = data['network']
        n_ports = network.nports

        # Create frequency dataset
        freq_name = "{}_freq".format(dataset_name)
        self.doc.SetData(freq_name, data['freq_ghz'])

        # Create datasets for each S-parameter
        for i in range(n_ports):
            for j in range(n_ports):
                # S-parameter names
                param_name = "S{}{}".format(i+1, j+1)
                mag_name = "{}_{}_mag".format(dataset_name, param_name)
                phase_name = "{}_{}_phase".format(dataset_name, param_name)

                # Extract S-parameter data
                s_param = network.s[:, i, j]
                mag_db = 20 * np.log10(np.abs(s_param) + 1e-12)
                phase_deg = np.angle(s_param, deg=True)

                # Set data in Veusz
                self.doc.SetData(mag_name, mag_db)
                self.doc.SetData(phase_name, phase_deg)

                # Tag datasets
                tag_name = "{}_{}".format(dataset_name, param_name)
                self.doc.TagDatasets(
                    tag_name,
                    [freq_name, mag_name, phase_name]
                )

        # Create individual pages
        self._create_magnitude_phase_pages(dataset_name, data)

    def _create_magnitude_phase_pages(self, dataset_name, data):
        """Create magnitude and phase plot pages.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        data : dict
            Data dictionary.
        """
        network = data['network']
        n_ports = network.nports

        # Create magnitude page
        mag_page_name = "{}_Magnitude".format(dataset_name)
        page_mag = self.doc.Root.Add('page', name=mag_page_name)
        grid_mag = page_mag.Add('grid', columns=2)
        graph_mag = grid_mag.Add(
            'graph', name="{}_Mag_Graph".format(dataset_name))

        # Add header info to page notes
        if 'header_info' in data:
            page_mag.notes.val = '\n'.join(data['header_info'])

        # Configure magnitude graph
        self._configure_standard_graph(
            graph_mag, "{} - Magnitude".format(dataset_name.replace('_', ' ')),
            self.freq_label, self.mag_label, -60, 20
        )

        # Add S-parameter plots to magnitude graph
        freq_name = "{}_freq".format(dataset_name)
        for i in range(n_ports):
            for j in range(n_ports):
                param_name = "S{}{}".format(i+1, j+1)
                mag_name = "{}_{}_mag".format(dataset_name, param_name)

                xy_mag = graph_mag.Add('xy', name="{}_mag".format(param_name))
                self._configure_xy_plot(xy_mag, freq_name, mag_name, 'auto')

        # Create phase page
        phase_page_name = "{}_Phase".format(dataset_name)
        page_phase = self.doc.Root.Add('page', name=phase_page_name)
        grid_phase = page_phase.Add('grid', columns=2)
        graph_phase = grid_phase.Add(
            'graph', name="{}_Phase_Graph".format(dataset_name))

        # Add header info to page notes
        if 'header_info' in data:
            page_phase.notes.val = '\n'.join(data['header_info'])

        # Configure phase graph
        self._configure_standard_graph(
            graph_phase, "{} - Phase".format(dataset_name.replace('_', ' ')),
            self.freq_label, self.phase_label, -180, 180
        )

        # Add S-parameter plots to phase graph
        for i in range(n_ports):
            for j in range(n_ports):
                param_name = "S{}{}".format(i+1, j+1)
                phase_name = "{}_{}_phase".format(dataset_name, param_name)

                xy_phase = graph_phase.Add(
                    'xy', name="{}_phase".format(param_name))
                self._configure_xy_plot(
                    xy_phase, freq_name, phase_name, 'auto')

    def _create_time_domain_page(self, dataset_name, param_name,
                                 data, td_result):
        """Create time domain plot page for a specific S-parameter.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        param_name : str
            S-parameter name.
        data : dict
            Original data dictionary.
        td_result : dict
            Time domain results.
        """
        # Create time domain datasets
        time_name = "{}_{}_time".format(dataset_name, param_name)
        td_unfilt_name = "{}_{}_td".format(dataset_name, param_name)
        td_filt_name = "{}_{}_tdf".format(dataset_name, param_name)

        # Get time domain data
        time_data = td_result.get('time', np.array([]))
        td_unfilt_data = td_result.get(
            "{}_td".format(param_name), np.array([]))
        td_filt_data = td_result.get(
            "{}_td_filtered".format(param_name), np.array([]))

        # Set data in Veusz
        self.doc.SetData(time_name, time_data)
        self.doc.SetData(td_unfilt_name, np.abs(td_unfilt_data))
        self.doc.SetData(td_filt_name, np.abs(td_filt_data))

        # Create time domain page
        td_page_name = "{}_{}_TimeDomain".format(dataset_name, param_name)
        page_td = self.doc.Root.Add('page', name=td_page_name)
        grid_td = page_td.Add('grid', columns=2)
        graph_td = grid_td.Add(
            'graph', name="{}_{}_TD_Graph".format(dataset_name, param_name))

        # Add header info to page notes
        if 'header_info' in data:
            page_td.notes.val = '\n'.join(data['header_info'])

        # Configure time domain graph
        graph_title = "{} - {} Time Domain".format(
            dataset_name.replace('_', ' '), param_name)
        self._configure_standard_graph(
            graph_td, graph_title,
            self.time_label, 'Magnitude', 0, 1
        )

        # Add unfiltered time domain plot (dotted line, no markers)
        xy_unfilt = graph_td.Add('xy', name="{}_td_unfilt".format(param_name))
        with self._wrap_widget(xy_unfilt) as plot:
            plot.xData.val = time_name
            plot.yData.val = td_unfilt_name
            plot.PlotLine.style.val = 'dotted'
            plot.PlotLine.width.val = '1pt'
            plot.PlotLine.color.val = 'blue'
            plot.marker.val = 'none'

        # Add filtered time domain plot (solid line)
        xy_filt = graph_td.Add('xy', name="{}_td_filt".format(param_name))
        self._configure_xy_plot(xy_filt, time_name, td_filt_name, 'red')

    def _create_smith_chart_page(self, dataset_name, param_name, data, smith_result):
        """Create Smith Chart plot page for a specific S-parameter.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        param_name : str
            S-parameter name.
        data : dict
            Original data dictionary.
        smith_result : dict
            Smith Chart results.
        """
        # Get the network for this parameter
        param_network = smith_result.get(param_name)
        if param_network is None:
            return

        # Create datasets for Smith Chart
        real_name = "{}_{}_real".format(dataset_name, param_name)
        imag_name = "{}_{}_imag".format(dataset_name, param_name)

        # Extract real and imaginary parts of S-parameters
        s_param = param_network.s[:, 0, 0]  # Single port network
        real_part = np.real(s_param)
        imag_part = np.imag(s_param)

        # Set data in Veusz
        self.doc.SetData(real_name, real_part)
        self.doc.SetData(imag_name, imag_part)

        # Create Smith Chart page using polar plot
        smith_page_name = "{}_{}_SmithChart".format(dataset_name, param_name)
        page_smith = self.doc.Root.Add('page', name=smith_page_name)
        grid_smith = page_smith.Add('grid', columns=2)

        # Use polar plot to simulate Smith Chart
        graph_smith = grid_smith.Add(
            'polar', name="{}_{}_Smith_Graph".format(dataset_name, param_name))

        # Add header info to page notes
        if 'header_info' in data:
            page_smith.notes.val = '\n'.join(data['header_info'])

        # Configure polar graph for Smith Chart appearance
        self._configure_smith_polar_graph(
            graph_smith,
            "{} - {} Smith Chart{}".format(
                dataset_name.replace('_', ' '),
                param_name,
                " (Time Domain Filtered)" if smith_result.get(
                    'filtered', False) else ""
            )
        )

        # Convert to polar coordinates for polar plot
        magnitude = np.abs(s_param)
        phase_rad = np.angle(s_param)

        # Create datasets for polar representation
        mag_name = "{}_{}_smith_mag".format(dataset_name, param_name)
        phase_name = "{}_{}_smith_phase".format(dataset_name, param_name)

        self.doc.SetData(mag_name, magnitude)
        self.doc.SetData(phase_name, phase_rad)

        # Add polar plot for Smith Chart
        polar_plot = graph_smith.Add(
            'nonorthpoint', name="{}_smith".format(param_name))
        self._configure_smith_polar_plot(polar_plot, mag_name, phase_name)

    def _configure_standard_graph(self, graph, title, x_label,
                                  y_label, y_min, y_max):
        """Configure a standard XY graph.

        Parameters
        ----------
        graph : veusz graph object
            Graph to configure.
        title : str
            Graph title.
        x_label : str
            X-axis label.
        y_label : str
            Y-axis label.
        y_min : float
            Y-axis minimum.
        y_max : float
            Y-axis maximum.
        """
        with self._wrap_widget(graph) as g:
            g.Add('label', name='plotTitle')
            g.topMargin.val = '1cm'
            g.plotTitle.Text.size.val = '12pt'
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

    def _configure_xy_plot(self, xy, x_data, y_data, color):
        """Configure an XY plot.

        Parameters
        ----------
        xy : veusz xy plot object
            XY plot to configure.
        x_data : str
            X data name.
        y_data : str
            Y data name.
        color : str
            Plot color.
        """
        with self._wrap_widget(xy) as plot:
            plot.xData.val = x_data
            plot.yData.val = y_data
            plot.PlotLine.color.val = color
            plot.PlotLine.width.val = '2pt'
            plot.marker.val = 'circle'
            plot.markerSize.val = '2pt'
            plot.MarkerFill.color.val = 'auto'
            plot.MarkerFill.transparency.val = 80

    def _configure_smith_polar_graph(self, graph, title):
        """Configure a polar graph for Smith Chart representation.

        Parameters
        ----------
        graph : veusz polar graph object
            Polar graph to configure.
        title : str
            Graph title.
        """
        with self._wrap_widget(graph) as g:
            g.Add('label', name='plotTitle')
            g.topMargin.val = '1cm'
            g.plotTitle.Text.size.val = '12pt'
            g.plotTitle.label.val = title
            g.plotTitle.alignHorz.val = 'centre'
            g.plotTitle.yPos.val = 1.05
            g.plotTitle.xPos.val = 0.5

            # Configure polar settings for Smith Chart
            g.units.val = 'radians'
            g.direction.val = 'anticlockwise'
            g.position0.val = 'right'
            g.minradius.val = 0
            g.maxradius.val = 1

    def _configure_smith_polar_plot(self, plot, mag_data, phase_data):
        """Configure a polar plot for Smith Chart representation.

        Parameters
        ----------
        plot : veusz polar plot object
            Polar plot to configure.
        mag_data : str
            Magnitude data name.
        phase_data : str
            Phase data name.
        """
        with self._wrap_widget(plot) as p:
            p.data1.val = mag_data
            p.data2.val = phase_data
            p.PlotLine.color.val = 'blue'
            p.PlotLine.width.val = '2pt'
            p.marker.val = 'circle'
            p.markerSize.val = '3pt'
            p.MarkerFill.color.val = 'blue'
            p.MarkerFill.transparency.val = 60

    def _add_to_overlay_plots(self, dataset_name, data):
        """Add dataset to overlay plots.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        data : dict
            Data dictionary.
        """
        network = data['network']
        n_ports = network.nports

        # Get overlay graphs
        try:
            page_all_mag = self.doc.Root.Overlay_All_Magnitude
            graph_all_mag = page_all_mag.grid1.Overlay_All_Magnitude

            page_all_phase = self.doc.Root.Overlay_All_Phase
            graph_all_phase = page_all_phase.grid1.Overlay_All_Phase

            freq_name = "{}_freq".format(dataset_name)

            # Add each S-parameter to overlays
            for i in range(n_ports):
                for j in range(n_ports):
                    param_name = "S{}{}".format(i+1, j+1)
                    mag_name = "{}_{}_mag".format(dataset_name, param_name)
                    phase_name = "{}_{}_phase".format(dataset_name, param_name)

                    # Add to magnitude overlay
                    xy_mag_name = "{}_{}_mag".format(dataset_name, param_name)
                    xy_mag = graph_all_mag.Add('xy', name=xy_mag_name)
                    self._configure_xy_plot(
                        xy_mag, freq_name, mag_name, 'auto')

                    # Add to phase overlay
                    xy_phase_name = "{}_{}_phase".format(
                        dataset_name, param_name)
                    xy_phase = graph_all_phase.Add('xy', name=xy_phase_name)
                    self._configure_xy_plot(
                        xy_phase, freq_name, phase_name, 'auto')

        except Exception as e:
            print("Error adding to overlay plots: {}".format(e))

    def _wrap_widget(self, widget):
        """Wrapper for Veusz widget context management.

        Parameters
        ----------
        widget : veusz widget object
            Widget to wrap.

        Returns
        -------
        WidgetWrapper
            Context manager for the widget.
        """
        class WidgetWrapper:
            def __init__(self, widget):
                self.widget = widget

            def __enter__(self):
                return self.widget

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

        return WidgetWrapper(widget)

    def save(self, filename):
        """Save Veusz document to specified file.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        filename_root = os.path.splitext(filename)[0]
        filename_hp = filename_root + '.vszh5'
        file_split = os.path.split(filename)
        beware_dir = file_split[0] + '/Beware_oldVersion/'
        filename_vsz = (
            beware_dir +
            os.path.splitext(file_split[1])[0] + '_BEWARE.vsz'
        )

        # Save high precision version
        self.doc.Save(filename_hp, mode='hdf5')

        # Save legacy version
        os.makedirs(beware_dir, exist_ok=True)
        self.doc.Save(filename_vsz, mode='vsz')


# %%

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


class switch:
    """Creates a case or switch style statement."""

    def __init__(self, value):
        """Initialize switch with value.

        Parameters
        ----------
        value : Any
            Value to switch on.
        """
        self.value = value

    def __iter__(self):
        """Iterate and find the match.

        Yields
        ------
        function
            Match function.
        """
        yield self.match

    def match(self, *args):
        """Return the match.

        Parameters
        ----------
        *args : Any
            Values to match against.

        Returns
        -------
        bool
            True if value matches any of the arguments.
        """
        return self.value in args


class Wrap4With:
    """Used to add context management to a given object."""

    def __init__(self, resource):
        """Initialize wrapper with resource.

        Parameters
        ----------
        resource : Any
            Resource to wrap.
        """
        self._resource = resource

    def __enter__(self):
        """Return the wrapped resource upon entering the context.

        Returns
        -------
        Any
            The wrapped resource.
        """
        return self._resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the resource upon exiting the context.

        Parameters
        ----------
        exc_type : type
            Exception type.
        exc_val : Exception
            Exception value.
        exc_tb : traceback
            Exception traceback.

        Returns
        -------
        None
            Returns None to propagate exceptions.
        """
        if hasattr(self._resource, 'close'):
            self._resource.close()
        return None


# %% Main Execution
def main():
    """Execute main function."""
    # Set multiprocessing start method for cross-platform compatibility
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)

    # Call this before any Qt imports
    if getattr(sys, 'frozen', False):  # Check if running as compiled executable
        setup_qt_plugins()
    app = QApplication(sys.argv)

    # Create and show main window
    window = TouchstoneMainWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
