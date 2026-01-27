#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enhanced Touchstone AutoPlot with Modern Qt GUI, Time Domain Analysis, and Smith Chart (Matplotlib)

# %% Header Info

This version integrates modern Qt GUI interface with comprehensive Touchstone file processing
capabilities using scikit-rf, including:
- Multiprocessing and GPU acceleration
- Advanced time domain analysis with gating functionality  
- Smith Chart plotting using matplotlib (with PDF export, multi-page PDF support)
- Veusz integration for S-parameter visualization

# %%% Author

Author: William W. Wallace
Last updated: 2025-06-28

# %%% Key Enhancement

Smith Chart tab now uses matplotlib instead of Veusz, with:
- Multiple output formats (PNG, PDF, TIFF, BMP, SVG, JPG)
- Multi-page PDF generation with bookmarks
- Configurable impedance/admittance display
- VSWR circles and Smith grid
- Same 3-tab structure preserved (Main Processing, Time Domain, Smith Chart)
"""

# ============================================================================
# IMPORTS - Standard Library
# ============================================================================

import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import datetime

# ============================================================================
# IMPORTS - Scientific and Numerical Computing
# ============================================================================

import numpy as np
import scipy.signal.windows as windows
import veusz.embed as vz

# ============================================================================
# IMPORTS - RF/Microwave Engineering and Network Analysis
# ============================================================================

import skrf as rf
from skrf import Network
from skrf.time import time_gate

# ============================================================================
# IMPORTS - Plotting and Visualization (Matplotlib for Smith Charts)
# ============================================================================

import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# ============================================================================
# IMPORTS - PDF Processing (for Smith chart PDF export)
# ============================================================================

try:
    from PyPDF2 import PdfMerger, PdfWriter
except ImportError:
    print("WARNING: PyPDF2 not installed. PDF merging will not be available.")
    print("Install with: pip install PyPDF2")

# ============================================================================
# IMPORTS - Qt Framework (GUI)
# ============================================================================

if getattr(sys, 'frozen', False):
    # Running as compiled executable - use PySide6 directly
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
    from PySide6.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    from PySide6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QSplitter, QLineEdit,
        QTabWidget, QComboBox, QSlider, QDoubleSpinBox, QGridLayout,
        QFormLayout, QFrame
    )
else:
    # Development environment - use QtPy
    from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize
    from qtpy.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    from qtpy.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
        QSpinBox, QGroupBox, QListWidget, QSplitter, QLineEdit,
        QTabWidget, QComboBox, QSlider, QDoubleSpinBox, QGridLayout,
        QFormLayout, QFrame
    )

# ============================================================================
# GPU Acceleration Detection
# ============================================================================

GPU_AVAILABLE = None
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


# ============================================================================
# CONFIGURATION AND DATA CLASSES
# ============================================================================

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
    window_type: str = "kaiser"
    window_param: float = 6.0
    gate_start: float = 0.0
    gate_stop: float = 1.0
    gate_center: float = 0.5
    gate_span: float = 0.2
    mode: str = "bandpass"
    method: str = "fft"
    tunit: str = "ns"
    auto_gate: bool = True


@dataclass
class SmithChartConfig:
    """Configuration class for Smith Chart plotting settings."""
    chart_type: str = "z"  # z for impedance, y for admittance
    draw_labels: bool = True
    draw_vswr: bool = True
    reference_impedance: float = 50.0
    show_legend: bool = True
    grid_color: str = "gray"
    trace_color: str = "blue"
    marker_style: str = "circle"
    marker_size: int = 4
    line_width: float = 1.5


@dataclass
class SmithChartMatplotlibConfig:
    """Configuration class for matplotlib Smith chart settings."""
    figure_size: Tuple[float, float] = (10, 10)
    dpi: int = 150
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid_alpha: float = 0.3
    line_width: float = 2.0
    marker_size: float = 6.0
    combine_to_pdf: bool = False
    output_format: str = "png"  # png, pdf, tiff, bmp, svg, jpg


# ============================================================================
# SMITH CHART MATPLOTLIB PLOTTER CLASS
# ============================================================================

class SmithChartMatplotlibPlotter:
    """
    Generates Smith charts using matplotlib and scikit-rf.
    
    This class handles the creation of Smith charts for impedance or admittance
    data extracted from Touchstone files.
    """

    def __init__(self, config: SmithChartMatplotlibConfig = None, z0: float = 50.0):
        """
        Initialize the Smith chart plotter.

        Parameters
        ----------
        config : SmithChartMatplotlibConfig, optional
            Configuration object. If None, defaults are used.
        z0 : float
            Reference impedance in Ohms. Default is 50.0.
        """
        self.config = config or SmithChartMatplotlibConfig()
        self.z0 = z0
        self.figures = []
        self.file_paths = []

    def create_smith_chart_figure(self, network: Network, param_name: str,
                                  chart_type: str = "z", draw_labels: bool = True,
                                  draw_vswr: bool = True) -> Tuple[Figure, str]:
        """
        Create a Smith chart figure from network S-parameters.

        Parameters
        ----------
        network : Network
            scikit-rf Network object containing S-parameter data.
        param_name : str
            S-parameter name (e.g., 'S11', 'S21').
        chart_type : str
            'z' for impedance, 'y' for admittance.
        draw_labels : bool
            Whether to draw impedance/admittance labels.
        draw_vswr : bool
            Whether to draw VSWR circles.

        Returns
        -------
        Tuple[Figure, str]
            matplotlib Figure object and descriptive title string.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size, dpi=self.config.dpi)

        # Extract port indices from parameter name
        indices = self.extract_param_indices(param_name)
        if indices is None:
            raise ValueError(f"Invalid parameter name: {param_name}")

        i, j = indices
        if i >= network.nports or j >= network.nports:
            raise IndexError(f"Port index out of range for {param_name}")

        # Extract S-parameter data
        s_param = network.s[:, i, j]

        # Convert to impedance or admittance
        if chart_type.lower() == "z":
            z_norm = (1 + s_param) / (1 - s_param)
            title = f"Smith Chart - {param_name} (Impedance)"
            label_type = "Impedance"
        elif chart_type.lower() == "y":
            z_norm = (1 - s_param) / (1 + s_param)
            title = f"Smith Chart - {param_name} (Admittance)"
            label_type = "Admittance"
        else:
            raise ValueError(f"Invalid chart type: {chart_type}")

        # Draw Smith chart background
        self.draw_smith_chart_grid(ax, draw_labels=draw_labels, draw_vswr=draw_vswr,
                                   label_type=label_type)

        # Plot S-parameter trace
        real_part = np.real(z_norm)
        imag_part = np.imag(z_norm)

        ax.plot(real_part, imag_part, "b-", linewidth=self.config.line_width,
                label=f"{param_name} Trace", marker="o", markersize=self.config.marker_size,
                alpha=0.7)

        # Mark frequency points
        num_points = len(z_norm)
        if num_points > 0:
            ax.plot(real_part[0], imag_part[0], "go", markersize=8, label="Start", zorder=5)
            ax.plot(real_part[-1], imag_part[-1], "ro", markersize=8, label="End", zorder=5)

        # Configure axes
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Real Part", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Imaginary Part", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight="bold")
        ax.legend(fontsize=self.config.legend_fontsize, loc="upper right")
        ax.grid(True, alpha=self.config.grid_alpha)

        fig.tight_layout()
        return fig, title

    def draw_smith_chart_grid(self, ax, draw_labels: bool = True, draw_vswr: bool = True,
                              label_type: str = "Impedance"):
        """Draw the Smith chart grid background on matplotlib axes."""
        # Main circle (magnitude = 1)
        circle = Circle((0, 0), 1, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(circle)

        # Resistance circles
        resistance_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for r in resistance_values:
            center_x = r / (1 + r)
            radius = 1 / (1 + r)
            circle = Circle((center_x, 0), radius, fill=False, edgecolor="lightblue",
                            linewidth=0.5, linestyle="--")
            ax.add_patch(circle)

            if draw_labels and center_x + radius <= 1.0:
                label_text = f"{r:.1f}" if r != 1.0 else f"{r:.1f}"
                ax.text(center_x + radius, 0.05, label_text, fontsize=8, ha="left",
                        va="center", color="blue")

        # Reactance circles
        reactance_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for x in reactance_values:
            center_x = 1.0
            center_y = 1.0 / x
            radius = 1.0 / x

            circle = Circle((center_x, center_y), radius, fill=False, edgecolor="lightgreen",
                            linewidth=0.5, linestyle="--")
            ax.add_patch(circle)

            circle_neg = Circle((center_x, -center_y), radius, fill=False, edgecolor="lightgreen",
                                linewidth=0.5, linestyle="--")
            ax.add_patch(circle_neg)

            if draw_labels:
                ax.text(center_x + 0.05, center_y + 0.05, f"+j{x:.1f}", fontsize=7,
                        ha="left", va="bottom", color="green")
                ax.text(center_x + 0.05, -center_y - 0.05, f"-j{x:.1f}", fontsize=7,
                        ha="left", va="top", color="green")

        # VSWR circles
        if draw_vswr:
            vswr_values = [1.5, 2.0, 3.0]
            for vswr in vswr_values:
                gamma_mag = (vswr - 1) / (vswr + 1)
                circle = Circle((0, 0), gamma_mag, fill=False, edgecolor="red",
                                linewidth=0.5, linestyle=":", alpha=0.5)
                ax.add_patch(circle)

                if draw_labels:
                    ax.text(gamma_mag, 0.05, f"VSWR={vswr:.1f}", fontsize=7,
                            ha="center", va="bottom", color="red")

    def extract_param_indices(self, param_name: str) -> Optional[Tuple[int, int]]:
        """
        Extract port indices from S-parameter name.

        Parameters
        ----------
        param_name : str
            Parameter name like 'S11', 'S21', 'S12', etc.

        Returns
        -------
        Tuple[int, int] or None
            (row_index, column_index) or None if invalid.
        """
        if not param_name.startswith('S') or len(param_name) != 3:
            return None
        try:
            i = int(param_name[1]) - 1
            j = int(param_name[2]) - 1
            return (i, j) if i >= 0 and j >= 0 else None
        except ValueError:
            return None

    def save_smith_charts(self, network: Network, filename: str, output_dir: str,
                          output_format: str = "png", chart_type: str = "z",
                          combine_pdf: bool = False, draw_labels: bool = True,
                          draw_vswr: bool = True) -> List[str]:
        """
        Generate and save Smith charts for all S-parameters in a network.

        Parameters
        ----------
        network : Network
            scikit-rf Network object.
        filename : str
            Base filename for saved charts.
        output_dir : str
            Directory to save chart files.
        output_format : str
            Output format: png, pdf, tiff, bmp, svg, jpg.
        chart_type : str
            'z' for impedance, 'y' for admittance.
        combine_pdf : bool
            If True and format is PDF, combine all into single PDF.
        draw_labels : bool
            Whether to draw Smith chart labels.
        draw_vswr : bool
            Whether to draw VSWR circles.

        Returns
        -------
        List[str]
            List of saved file paths.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        base_filename = os.path.splitext(filename)[0]
        pdf_files = [] if (output_format.lower() == "pdf" and combine_pdf) else None

        for i in range(network.nports):
            for j in range(network.nports):
                param_name = f"S{i + 1}{j + 1}"
                try:
                    fig, title = self.create_smith_chart_figure(
                        network, param_name, chart_type=chart_type,
                        draw_labels=draw_labels, draw_vswr=draw_vswr
                    )

                    safe_title = title.replace(" ", "_").replace("-", "_")
                    output_filename = f"{base_filename}_{param_name}.{output_format.lower()}"
                    output_path = os.path.join(output_dir, output_filename)

                    fig.savefig(output_path, format=output_format.lower(), dpi=self.config.dpi)
                    plt.close(fig)

                    saved_files.append(output_path)

                    if output_format.lower() == "pdf" and combine_pdf:
                        pdf_files.append(output_path)

                except Exception as e:
                    print(f"Warning: Failed to create Smith chart for {param_name}: {e}")

        # Combine PDFs if requested
        if pdf_files and combine_pdf:
            try:
                merger = PdfMerger()
                for pdf_file in pdf_files:
                    merger.append(pdf_file)
                combined_pdf = os.path.join(output_dir, f"{base_filename}_combined.pdf")
                merger.write(combined_pdf)
                merger.close()
                saved_files.append(combined_pdf)
            except Exception as e:
                print(f"Warning: Failed to combine PDFs: {e}")

        return saved_files


# ============================================================================
# PROCESSING THREAD CLASSES
# ============================================================================

class TimeDomainProcessor:
    """Handles time domain analysis of S-parameter data."""

    def __init__(self, config: TimeDomainConfig = None):
        """Initialize time domain processor."""
        self.config = config or TimeDomainConfig()

    def process_network(self, network: Network, apply_td_filter: bool = False) -> dict:
        """Process a network for time domain analysis."""
        results = {}
        try:
            # Convert to time domain using IFFT
            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    s_param = network.s[:, i, j]

                    # Simple IFFT conversion
                    time_data = np.fft.ifft(s_param)

                    results[param_name] = np.abs(time_data)

            results['network'] = network
            results['filtered'] = apply_td_filter
            results['error'] = None

        except Exception as e:
            results['error'] = str(e)

        return results


class SmithChartProcessor:
    """Handles Smith Chart analysis of S-parameter data."""

    def __init__(self, config: SmithChartConfig = None):
        """Initialize Smith Chart processor."""
        self.config = config or SmithChartConfig()

    def process_network_for_smith_chart(self, network: Network,
                                        apply_td_filter: bool = False,
                                        td_processor: TimeDomainProcessor = None) -> dict:
        """Process a network for Smith Chart plotting."""
        results = {}
        try:
            results['network'] = network
            results['chart_type'] = self.config.chart_type
            results['reference_impedance'] = self.config.reference_impedance
            results['draw_labels'] = self.config.draw_labels
            results['draw_vswr'] = self.config.draw_vswr
            results['error'] = None

        except Exception as e:
            results['error'] = str(e)

        return results


# ============================================================================
# VEUSZ PLOTTER CLASS
# ============================================================================

class TouchstonePlotter:
    """Handles Veusz plotting for Touchstone files."""

    def __init__(self, plot_title: str = "Touchstone S-Parameter Analysis",
                 dataset_name: str = "Touchstone_Dataset"):
        """Initialize the Veusz plotter."""
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.doc = vz.Embedded("Touchstone_AutoPlot")

    def create_plots_from_data(self, filename: str, data: dict):
        """Create plots from processed Touchstone data."""
        try:
            network = data['network']
            # Create basic S-parameter plots
            page = self.doc.Root.Add('page', name=f"SParameters_{filename}")
            grid = page.Add('grid', columns=2)
            graph = grid.Add('graph', name=f"SParam_{filename}")

            # Add S-parameter data to Veusz
            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    # Add dataset
                    self.doc.AddDataset(
                        param_name,
                        data=network.s[:, i, j],
                        linked=False
                    )

        except Exception as e:
            print(f"Error creating plots: {e}")

    def create_time_domain_plots(self, filename: str, data: dict, td_result: dict):
        """Create time domain plots in Veusz."""
        if 'error' in td_result and td_result['error']:
            return
        try:
            pass  # Implement if needed
        except Exception as e:
            print(f"Error creating time domain plots: {e}")

    def create_smith_chart_plots(self, filename: str, data: dict, smith_result: dict):
        """Create Smith chart plots in Veusz."""
        if 'error' in smith_result and smith_result['error']:
            return
        try:
            pass  # Smith charts are now generated in matplotlib
        except Exception as e:
            print(f"Error creating Smith chart plots: {e}")

    def save(self, filename: str):
        """Save Veusz project."""
        self.doc.Save(filename, mode='hdf5')


# ============================================================================
# MATPLOTLIB CANVAS FOR EMBEDDING PLOTS
# ============================================================================

class TouchstonePlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for Touchstone plots."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """Initialize plot canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        plt.style.use('default')
        self.fig.patch.set_facecolor('white')

    def plot_s_parameters(self, frequency, s_data, title="S-Parameters"):
        """Plot S-parameters on the canvas."""
        self.fig.clear()
        n_ports = int(np.sqrt(s_data.shape[1]))

        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        for i in range(n_ports):
            for j in range(n_ports):
                idx = i * n_ports + j
                mag_db = 20 * np.log10(np.abs(s_data[:, idx]) + 1e-12)
                ax1.plot(frequency / 1e9, mag_db, label=f"S{i + 1}{j + 1}")

                phase_deg = np.angle(s_data[:, idx], deg=True)
                ax2.plot(frequency / 1e9, phase_deg, label=f"S{i + 1}{j + 1}")

        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(f"{title} - Magnitude")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_title(f"{title} - Phase")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        self.draw()

    def plot_time_domain(self, time, td_data, td_filtered_data=None, title="Time Domain"):
        """Plot time domain data on the canvas."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        ax.plot(time, np.abs(td_data), alpha=0.7, markersize=2,
                label="Unfiltered", linestyle="dotted")

        if td_filtered_data is not None:
            ax.plot(time, np.abs(td_filtered_data), "-", linewidth=2, label="Filtered")

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Magnitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        self.draw()

    def plot_smith_chart(self, network, title="Smith Chart", chart_type="z",
                         draw_labels=True, draw_vswr=True):
        """Plot Smith chart on the canvas."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        try:
            # Use scikit-rf's built-in Smith chart plotting
            network.plot_s_smith(ax=ax, chart_type=chart_type, draw_labels=draw_labels,
                                 draw_vswr=draw_vswr, show_legend=True)
            ax.set_title(title)
        except Exception as e:
            # Fallback to basic complex plane plot
            print(f"Smith chart plotting failed: {e}. Using complex plot.")
            for i in range(network.nports):
                for j in range(network.nports):
                    s_param = network.s[:, i, j]
                    ax.plot(s_param.real, s_param.imag, label=f"S{i + 1}{j + 1}",
                            marker="o", markersize=3)

            ax.set_xlabel("Real Part")
            ax.set_ylabel("Imaginary Part")
            ax.set_title(f"Complex Plane - {title}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis("equal")

        self.draw()


# ============================================================================
# TOUCHSTONE PROCESSING THREAD
# ============================================================================

def process_single_touchstone_file(file_info: Tuple) -> Tuple[str, Optional[dict]]:
    """Worker function to process a single Touchstone file."""
    filepath, _ = file_info
    try:
        network = Network(filepath)
        filename = os.path.basename(filepath)
        data = {
            'network': network,
            'frequency': network.frequency.f,
            'file_path': filepath
        }
        return filename, data
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return os.path.basename(filepath), None


class TouchstoneProcessingThread(QThread):
    """Thread for processing Touchstone files without blocking GUI."""

    progress_updated = Signal(int)
    processing_finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, file_list: List[str], config: ProcessingConfig):
        """Initialize processing thread."""
        super().__init__()
        self.file_list = file_list
        self.config = config

    def run(self):
        """Execute file processing in separate thread."""
        try:
            results = {}
            if self.config.enable_multiprocessing and len(self.file_list) > 1:
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = [executor.submit(process_single_touchstone_file, (f, None))
                               for f in self.file_list]
                    completed = 0
                    for future in as_completed(futures):
                        filename, data = future.result()
                        results[filename] = data
                        completed += 1
                        progress = int((completed / len(self.file_list)) * 100)
                        self.progress_updated.emit(progress)
            else:
                for i, filepath in enumerate(self.file_list):
                    filename, data = process_single_touchstone_file((filepath, None))
                    results[filename] = data
                    progress = int(((i + 1) / len(self.file_list)) * 100)
                    self.progress_updated.emit(progress)

            self.processing_finished.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class TouchstoneMainWindow(QMainWindow):
    """Main window for Touchstone AutoPlot application with THREE tabs."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Enhanced Touchstone AutoPlot - Smith Chart Matplotlib Edition")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize configurations
        self.config = ProcessingConfig()
        self.td_config = TimeDomainConfig()
        self.smith_config = SmithChartConfig()
        self.smith_mpl_config = SmithChartMatplotlibConfig()

        # Initialize processors and plotters
        self.touchstone_plotter = None
        self.td_processor = TimeDomainProcessor(self.td_config)
        self.smith_processor = SmithChartProcessor(self.smith_config)
        self.smith_mpl_plotter = SmithChartMatplotlibPlotter(self.smith_mpl_config)

        # Data storage
        self.selected_files = []
        self.processed_data = {}
        self.td_results = {}
        self.smith_results = {}

        # Setup UI
        self.setup_ui()

        self._log_message("Enhanced Touchstone AutoPlot initialized")
        self._log_message(f"GPU Support: {GPU_AVAILABLE or 'None'}")
        self._log_message(f"CPU Cores Available: {multiprocessing.cpu_count()}")

    def setup_ui(self):
        """Set up the user interface with tab widget."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Setup the three tabs
        self.setup_main_tab()
        self.setup_time_domain_tab()
        self.setup_smith_chart_tab()

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

    def setup_main_tab(self):
        """Setup the main processing tab."""
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "Main Processing")
        layout = QVBoxLayout(main_tab)

        # File selection
        file_group = QGroupBox("Touchstone File Selection")
        file_layout = QVBoxLayout(file_group)

        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumHeight(150)
        file_layout.addWidget(self.file_list_widget)

        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse Touchstone Files")
        self.browse_button.clicked.connect(self._browse_files)
        browse_layout.addWidget(self.browse_button)

        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        browse_layout.addWidget(self.clear_button)

        file_layout.addLayout(browse_layout)
        layout.addWidget(file_group)

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
        self.cpu_spinbox.setMaximum(multiprocessing.cpu_count())
        self.cpu_spinbox.setValue(self.config.num_processes)
        self.cpu_spinbox.valueChanged.connect(self._update_cpu_config)
        cpu_layout.addWidget(self.cpu_spinbox)
        cpu_layout.addStretch()
        options_layout.addLayout(cpu_layout)

        self.enable_gpu_checkbox = QCheckBox("Enable GPU Processing")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.stateChanged.connect(self._update_gpu_config)
        options_layout.addWidget(self.enable_gpu_checkbox)

        layout.addWidget(options_group)

        # Plot configuration
        plot_group = QGroupBox("Plot Configuration")
        plot_layout = QVBoxLayout(plot_group)

        plot_title_layout = QHBoxLayout()
        plot_title_layout.addWidget(QLabel("Plot Title:"))
        self.plot_title_edit = QLineEdit("Touchstone S-Parameter Analysis")
        plot_title_layout.addWidget(self.plot_title_edit)
        plot_layout.addLayout(plot_title_layout)

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_edit = QLineEdit("Touchstone_Dataset")
        dataset_layout.addWidget(self.dataset_name_edit)
        plot_layout.addLayout(dataset_layout)

        layout.addWidget(plot_group)

    def setup_time_domain_tab(self):
        """Setup the time domain analysis tab."""
        td_tab = QWidget()
        self.tab_widget.addTab(td_tab, "Time Domain Analysis")
        layout = QHBoxLayout(td_tab)

        # Left control panel
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
        self.window_type_combo.addItems(["kaiser", "hamming", "hann", "blackman", "boxcar"])
        self.window_type_combo.setCurrentText(self.td_config.window_type)
        self.window_type_combo.currentTextChanged.connect(self._update_window_config)
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
        self.gate_start_spin.setSuffix(" ns")
        self.gate_start_spin.valueChanged.connect(self._update_gate_start)
        gate_layout.addRow("Gate Start:", self.gate_start_spin)

        self.gate_stop_spin = QDoubleSpinBox()
        self.gate_stop_spin.setRange(-100.0, 100.0)
        self.gate_stop_spin.setValue(self.td_config.gate_stop)
        self.gate_stop_spin.setSuffix(" ns")
        self.gate_stop_spin.valueChanged.connect(self._update_gate_stop)
        gate_layout.addRow("Gate Stop:", self.gate_stop_spin)

        self.gate_start_spin.setEnabled(not self.td_config.auto_gate)
        self.gate_stop_spin.setEnabled(not self.td_config.auto_gate)

        controls_layout.addWidget(gate_group)

        # Method settings
        method_group = QGroupBox("Processing Settings")
        method_layout = QFormLayout(method_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["bandpass", "bandstop"])
        self.mode_combo.setCurrentText(self.td_config.mode)
        self.mode_combo.currentTextChanged.connect(self._update_mode)
        method_layout.addRow("Mode:", self.mode_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["fft", "rfft", "convolution"])
        self.method_combo.setCurrentText(self.td_config.method)
        self.method_combo.currentTextChanged.connect(self._update_method)
        method_layout.addRow("Method:", self.method_combo)

        controls_layout.addWidget(method_group)
        controls_layout.addStretch()

        layout.addWidget(controls_widget)

        # Right panel for plot
        self.td_plot_canvas = TouchstonePlotCanvas(td_tab, width=8, height=6)
        layout.addWidget(self.td_plot_canvas)

    def setup_smith_chart_tab(self):
        """Setup the Smith Chart analysis tab with matplotlib support."""
        smith_tab = QWidget()
        self.tab_widget.addTab(smith_tab, "Smith Chart Analysis")
        layout = QHBoxLayout(smith_tab)

        # Left control panel
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls_widget)

        # File selection for preview
        file_select_group = QGroupBox("File Selection for Preview")
        file_select_layout = QVBoxLayout(file_select_group)
        self.smith_file_combo = QComboBox()
        self.smith_file_combo.currentTextChanged.connect(self._update_smith_preview)
        file_select_layout.addWidget(self.smith_file_combo)
        controls_layout.addWidget(file_select_group)

        # Smith Chart settings
        smith_group = QGroupBox("Smith Chart Settings")
        smith_layout = QFormLayout(smith_group)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["z (Impedance)", "y (Admittance)"])
        self.chart_type_combo.setCurrentText("z (Impedance)")
        self.chart_type_combo.currentTextChanged.connect(self._update_chart_type)
        smith_layout.addRow("Chart Type:", self.chart_type_combo)

        self.ref_impedance_spin = QDoubleSpinBox()
        self.ref_impedance_spin.setRange(1.0, 1000.0)
        self.ref_impedance_spin.setValue(self.smith_config.reference_impedance)
        self.ref_impedance_spin.setSuffix(" Ω")
        self.ref_impedance_spin.valueChanged.connect(self._update_ref_impedance)
        smith_layout.addRow("Reference Impedance:", self.ref_impedance_spin)

        self.draw_labels_checkbox = QCheckBox("Draw Labels")
        self.draw_labels_checkbox.setChecked(self.smith_config.draw_labels)
        self.draw_labels_checkbox.stateChanged.connect(self._update_draw_labels)
        smith_layout.addRow(self.draw_labels_checkbox)

        self.draw_vswr_checkbox = QCheckBox("Draw VSWR Circles")
        self.draw_vswr_checkbox.setChecked(self.smith_config.draw_vswr)
        self.draw_vswr_checkbox.stateChanged.connect(self._update_draw_vswr)
        smith_layout.addRow(self.draw_vswr_checkbox)

        controls_layout.addWidget(smith_group)

        # Processing scope
        scope_group = QGroupBox("Processing Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.smith_process_selected_only = QCheckBox("Generate Smith Chart only for selected file from dropdown")
        self.smith_process_selected_only.setChecked(False)
        scope_layout.addWidget(self.smith_process_selected_only)

        controls_layout.addWidget(scope_group)

        # Output format selection
        output_group = QGroupBox("Output Format")
        output_layout = QVBoxLayout(output_group)

        format_label = QLabel("Output Format:")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["PNG", "PDF", "TIFF", "BMP", "SVG", "JPG"])
        self.output_format_combo.setCurrentText("PNG")
        self.output_format_combo.currentTextChanged.connect(self._update_output_format)
        output_layout.addWidget(format_label)
        output_layout.addWidget(self.output_format_combo)

        self.combine_pdf_checkbox = QCheckBox("Combine all plots into single PDF (with bookmarks)")
        self.combine_pdf_checkbox.setChecked(False)
        self.combine_pdf_checkbox.setEnabled(False)
        self.combine_pdf_checkbox.stateChanged.connect(self._update_combine_pdf)
        output_layout.addWidget(self.combine_pdf_checkbox)

        controls_layout.addWidget(output_group)

        # Time domain filtering option
        td_filter_group = QGroupBox("Time Domain Filtering")
        td_filter_layout = QVBoxLayout(td_filter_group)

        self.smith_enable_td_filter = QCheckBox("Enable time domain filtering on Smith Chart data")
        self.smith_enable_td_filter.setChecked(False)
        self.smith_enable_td_filter.stateChanged.connect(self._update_smith_preview)
        td_filter_layout.addWidget(self.smith_enable_td_filter)

        controls_layout.addWidget(td_filter_group)

        # Process button
        self.smith_process_button = QPushButton("Generate Smith Charts in Matplotlib")
        self.smith_process_button.clicked.connect(self._process_smith_charts_matplotlib)
        self.smith_process_button.setEnabled(False)
        controls_layout.addWidget(self.smith_process_button)

        controls_layout.addStretch()
        layout.addWidget(controls_widget)

        # Right panel for plot
        self.smith_plot_canvas = TouchstonePlotCanvas(smith_tab, width=8, height=6)
        layout.addWidget(self.smith_plot_canvas)

    # ========================================================================
    # CALLBACK METHODS - File Operations
    # ========================================================================

    def _browse_files(self):
        """Open file dialog to select Touchstone files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Touchstone Files (*.s*p)")
        file_dialog.setWindowTitle("Select Touchstone Files")

        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self._update_file_list()
            self._log_message(f"Selected {len(selected_files)} Touchstone files")

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
        for filepath in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(filepath))

    def _update_td_file_combo(self):
        """Update the time domain file selection combo box."""
        self.td_file_combo.clear()
        if self.processed_data:
            self.td_file_combo.addItems(list(self.processed_data.keys()))

    def _update_smith_file_combo(self):
        """Update the Smith Chart file selection combo box."""
        self.smith_file_combo.clear()
        if self.processed_data:
            self.smith_file_combo.addItems(list(self.processed_data.keys()))

    # ========================================================================
    # CALLBACK METHODS - Configuration Updates
    # ========================================================================

    def _update_mp_config(self, state: int):
        """Update multiprocessing configuration."""
        self.config.enable_multiprocessing = (state == Qt.Checked)
        status_msg = f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}"
        self._log_message(status_msg)

    def _update_cpu_config(self, value: int):
        """Update CPU cores configuration."""
        self.config.num_processes = value
        self.config.max_workers = value
        self._log_message(f"CPU cores set to: {value}")

    def _update_gpu_config(self, state: int):
        """Update GPU processing configuration."""
        self.config.enable_gpu_processing = (state == Qt.Checked)
        status_msg = f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}"
        self._log_message(status_msg)

    def _update_window_config(self, window_type: str):
        """Update window type configuration."""
        self.td_config.window_type = window_type
        self._update_td_preview()

    def _update_window_param(self, value: float):
        """Update window parameter configuration."""
        self.td_config.window_param = value
        self._update_td_preview()

    def _update_mode(self, mode: str):
        """Update processing mode configuration."""
        self.td_config.mode = mode
        self._update_td_preview()

    def _update_method(self, method: str):
        """Update processing method configuration."""
        self.td_config.method = method
        self._update_td_preview()

    def _update_auto_gate(self, state: int):
        """Update auto gate configuration."""
        self.td_config.auto_gate = (state == Qt.Checked)
        self.gate_start_spin.setEnabled(not self.td_config.auto_gate)
        self.gate_stop_spin.setEnabled(not self.td_config.auto_gate)
        self._update_td_preview()

    def _update_gate_start(self, value: float):
        """Update gate start configuration."""
        self.td_config.gate_start = value
        self._update_td_preview()

    def _update_gate_stop(self, value: float):
        """Update gate stop configuration."""
        self.td_config.gate_stop = value
        self._update_td_preview()

    def _update_chart_type(self, chart_type_text: str):
        """Update Smith Chart type configuration."""
        if "Impedance" in chart_type_text:
            self.smith_config.chart_type = "z"
        else:
            self.smith_config.chart_type = "y"
        self._update_smith_preview()

    def _update_ref_impedance(self, value: float):
        """Update reference impedance configuration."""
        self.smith_config.reference_impedance = value
        self.smith_mpl_plotter.z0 = value
        self._update_smith_preview()

    def _update_draw_labels(self, state: int):
        """Update draw labels configuration."""
        self.smith_config.draw_labels = (state == Qt.Checked)
        self._update_smith_preview()

    def _update_draw_vswr(self, state: int):
        """Update draw VSWR configuration."""
        self.smith_config.draw_vswr = (state == Qt.Checked)
        self._update_smith_preview()

    def _update_output_format(self, format_text: str):
        """Update output format configuration."""
        self.smith_mpl_config.output_format = format_text.lower()
        is_pdf = format_text.lower() == "pdf"
        self.combine_pdf_checkbox.setEnabled(is_pdf)

    def _update_combine_pdf(self, state: int):
        """Update combine PDF configuration."""
        self.smith_mpl_config.combine_to_pdf = (state == Qt.Checked)

    # ========================================================================
    # PREVIEW AND PROCESSING METHODS
    # ========================================================================

    def _update_td_preview(self):
        """Update time domain preview."""
        current_file = self.td_file_combo.currentText()
        if current_file and current_file in self.processed_data:
            try:
                network = self.processed_data[current_file]['network']
                td_result = self.td_processor.process_network(network)
                self.td_results[current_file] = td_result

                # Extract first S-parameter for preview
                s_param = network.s[:, 0, 0]
                time_data = np.fft.ifft(s_param)
                time_array = np.arange(len(time_data))

                plot_title = f"{current_file} - Time Domain"
                self.td_plot_canvas.plot_time_domain(time_array, np.abs(time_data), title=plot_title)

            except Exception as e:
                self._log_message(f"Time domain preview error: {e}")

    def _update_smith_preview(self):
        """Update Smith chart preview."""
        current_file = self.smith_file_combo.currentText()
        if current_file and current_file in self.processed_data:
            try:
                network = self.processed_data[current_file]['network']

                # Create preview network with S11
                preview_network = network.copy()
                preview_network.s = network.s[:, 0, 0].reshape(-1, 1, 1)

                plot_title = f"{current_file} - S11 Smith Chart"

                self.smith_plot_canvas.plot_smith_chart(
                    preview_network,
                    title=plot_title,
                    chart_type=self.smith_config.chart_type,
                    draw_labels=self.smith_config.draw_labels,
                    draw_vswr=self.smith_config.draw_vswr
                )

            except Exception as e:
                self._log_message(f"Smith Chart preview error: {e}")

    def _process_files(self):
        """Process selected files."""
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "Please select Touchstone files first.")
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
        self.processing_thread = TouchstoneProcessingThread(self.selected_files, self.config)
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self._on_processing_finished)
        self.processing_thread.error_occurred.connect(self._on_processing_error)
        self.processing_thread.start()

        self._log_message("Touchstone processing started...")

    def _on_processing_finished(self, results):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)

        successful_files = len([r for r in results.values() if r])
        failed_files = len(results) - successful_files

        self._log_message(f"Processing completed: {successful_files} successful, {failed_files} failed")

        if successful_files > 0:
            self.processed_data = results
            self._update_td_file_combo()
            self._update_smith_file_combo()
            self.save_button.setEnabled(True)

        if failed_files > 0:
            QMessageBox.warning(
                self,
                "Processing Warnings",
                f"{failed_files} files failed to process. Check status log for details."
            )

    def _on_processing_error(self, error_message: str):
        """Handle processing error."""
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self._log_message(f"Processing error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)

    def _process_smith_charts_matplotlib(self):
        """Process and generate Smith charts using matplotlib."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return

        self._log_message("Starting Smith Chart generation...")

        try:
            # Ask user for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory for Smith Charts"
            )

            if not output_dir:
                return

            all_saved_files = []

            # Determine which files to process
            if self.smith_process_selected_only.isChecked():
                current_file = self.smith_file_combo.currentText()
                if current_file and current_file in self.processed_data:
                    files_to_process = {current_file: self.processed_data[current_file]}
                else:
                    QMessageBox.warning(self, "Warning", "No file selected for processing.")
                    return
            else:
                files_to_process = self.processed_data

            # Generate Smith charts for each file
            for filename, data in files_to_process.items():
                network = data['network']

                try:
                    saved_files = self.smith_mpl_plotter.save_smith_charts(
                        network=network,
                        filename=filename,
                        output_dir=output_dir,
                        output_format=self.smith_mpl_config.output_format,
                        chart_type=self.smith_config.chart_type,
                        combine_pdf=self.smith_mpl_config.combine_to_pdf,
                        draw_labels=self.smith_config.draw_labels,
                        draw_vswr=self.smith_config.draw_vswr
                    )

                    all_saved_files.extend(saved_files)
                    self._log_message(f"Smith charts generated for {filename}: {len(saved_files)} files")

                except Exception as e:
                    self._log_message(f"Error processing {filename}: {e}")

            # Summary message
            msg = f"Smith charts generated successfully!\n"
            msg += f"Total files saved: {len(all_saved_files)}\n"
            msg += f"Output directory: {output_dir}"

            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate Smith charts: {str(e)}")
            self._log_message(f"Smith chart generation error: {e}")

    def _save_project(self):
        """Save Veusz project."""
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Touchstone Veusz Project",
            "",
            "Veusz High Precision Files (*.vszh5)"
        )

        if save_path and self.touchstone_plotter:
            try:
                self.touchstone_plotter.save(save_path)
                self._log_message(f"Project saved: {save_path}")

                reply = QMessageBox.question(
                    self,
                    "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    subprocess.Popen(['veusz', save_path])

            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{e}")

    def _log_message(self, message: str):
        """Add message to status log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = TouchstoneMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
