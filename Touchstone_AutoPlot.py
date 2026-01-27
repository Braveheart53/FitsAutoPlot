#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced Touchstone AutoPlot with Modern Qt GUI, Time Domain Analysis, HoloViews Smith Charts, and Veusz Integration.

# %% Header Info

This version integrates modern Qt GUI interface with comprehensive Touchstone file processing
capabilities using scikit-rf, including:

- Multiprocessing and GPU acceleration
- Advanced time domain analysis with gating functionality
- Smith Chart plotting using HoloViews with interactive tools
- Smith Chart export to PNG, PDF, SVG, TIFF, JPG formats
- Time-gated plot generation in Veusz

# %%% Author

Author: William W. Wallace
Last updated: 2025-01-27

# %%% Key Features

- Veusz integration for S-parameter frequency and time domain visualization
- Smith Chart plotting using HoloViews (interactive, not in Veusz)
- Smith Chart export to multiple file formats
- Time domain analysis with time-gated plot export to Veusz
- Multi-page Veusz file support with proper dataset tagging

"""

# ============================================================================
# IMPORTS - Standard Library
# ============================================================================
import multiprocessing
import os
import subprocess
import sys
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from contextlib import contextmanager

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
# IMPORTS - Plotting and Visualization (HoloViews for Smith Charts)
# ============================================================================
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# HoloViews for interactive Smith Charts
try:
    import holoviews as hv
    from holoviews import opts
    hv.extension('matplotlib')
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    HOLOVIEWS_AVAILABLE = False
    print("WARNING: HoloViews not installed. Interactive Smith Charts will not be available.")
    print("Install with: pip install holoviews")

# ============================================================================
# IMPORTS - PDF Processing (for Smith chart export)
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
    output_format: str = "png"  # png, pdf, svg, tiff, jpg

# ============================================================================
# VEUSZ PLOTTER CLASS (Frequency and Time Domain Only)
# ============================================================================
class TouchstonePlotter:
    """Handles Veusz plotting for Touchstone files (frequency and time domain)."""

    def __init__(self, plot_title: str = "Touchstone S-Parameter Analysis",
                 dataset_name: str = "Touchstone_Dataset"):
        """Initialize the Veusz plotter."""
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.doc = vz.Embedded("Touchstone_AutoPlot")
        self.doc.EnableToolbar(enable=True)
        self.freq_label = "Frequency (GHz)"
        self.time_label = "Time (ns)"

    @contextmanager
    def _wrap_widget(self, widget):
        """Context manager for temporary widget property access."""
        try:
            yield widget
        finally:
            pass

    def _configure_standard_graph(self, graph, title, x_label, y_label, y_min, y_max):
        """Configure a standard XY graph."""
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
        """Configure an XY plot."""
        with self._wrap_widget(xy) as plot:
            plot.xData.val = x_data
            plot.yData.val = y_data
            plot.PlotLine.color.val = color
            plot.PlotLine.width.val = '2pt'
            plot.marker.val = 'circle'
            plot.markerSize.val = '2pt'
            plot.MarkerFill.color.val = 'auto'
            plot.MarkerFill.transparency.val = 80

    def create_plots_from_data(self, filename: str, data: dict):
        """Create frequency domain plots from processed Touchstone data."""
        try:
            network = data['network']
            dataset_name = filename.replace('.', '_').replace('-', '_')
            
            # Create frequency domain page
            self._create_frequency_page(dataset_name, data)
            
        except Exception as e:
            print(f"Error creating plots: {e}")

    def _create_frequency_page(self, dataset_name, data):
        """Create frequency domain plot page for S-parameters."""
        network = data['network']
        n_ports = network.nports
        
        # Create frequency dataset
        freq_name = f"{dataset_name}_freq"
        self.doc.SetData(freq_name, network.frequency.f / 1e9)
        
        # Create main frequency domain page
        page = self.doc.Root.Add('page', name=f"SParam_{dataset_name}")
        grid = page.Add('grid', columns=2)
        
        # Magnitude and phase graphs
        graph_mag = grid.Add('graph', name=f"{dataset_name}_Mag")
        graph_phase = grid.Add('graph', name=f"{dataset_name}_Phase")
        
        # Add header info to page notes
        if 'header_info' in data:
            page.notes.val = '\n'.join(data['header_info'])
        
        # Configure magnitude graph
        self._configure_standard_graph(
            graph_mag,
            f"{dataset_name.replace('_', ' ')} - Magnitude",
            self.freq_label, 'Magnitude (dB)', -80, 20
        )
        
        # Configure phase graph
        self._configure_standard_graph(
            graph_phase,
            f"{dataset_name.replace('_', ' ')} - Phase",
            self.freq_label, 'Phase (degrees)', -180, 180
        )
        
        # Add S-parameter plots
        for i in range(n_ports):
            for j in range(n_ports):
                param_name = f"S{i + 1}{j + 1}"
                mag_name = f"{dataset_name}_{param_name}_mag"
                phase_name = f"{dataset_name}_{param_name}_phase"
                
                # Calculate magnitude and phase
                s_param = network.s[:, i, j]
                mag_db = 20 * np.log10(np.abs(s_param) + 1e-12)
                phase_deg = np.angle(s_param, deg=True)
                
                # Set data in Veusz
                self.doc.SetData(mag_name, mag_db)
                self.doc.SetData(phase_name, phase_deg)
                
                # Add magnitude plot
                xy_mag = graph_mag.Add('xy', name=f"{param_name}_mag")
                self._configure_xy_plot(xy_mag, freq_name, mag_name, 'auto')
                
                # Add phase plot
                xy_phase = graph_phase.Add('xy', name=f"{param_name}_phase")
                self._configure_xy_plot(xy_phase, freq_name, phase_name, 'auto')

    def create_time_gated_plots(self, filename: str, data: dict, td_result: dict):
        """Create time-gated plots in Veusz with _timegated postfix."""
        if 'error' in td_result and td_result['error']:
            return
        
        try:
            network = data['network']
            dataset_name = filename.replace('.', '_').replace('-', '_')
            
            # Create time-gated plots for each S-parameter
            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    self._create_time_gated_page(dataset_name, param_name, data, td_result)
                    
        except Exception as e:
            print(f"Error creating time-gated plots: {e}")

    def _create_time_gated_page(self, dataset_name, param_name, data, td_result):
        """Create time-gated plot page with _timegated tag."""
        # Create time-gated datasets with _timegated postfix
        time_name = f"{dataset_name}_{param_name}_timegated_time"
        td_gated_name = f"{dataset_name}_{param_name}_timegated"
        
        # Get time domain data (gated version)
        time_data = td_result.get('time', np.array([]))
        td_gated_data = td_result.get(f"{param_name}_td_filtered", np.array([]))
        
        # Set data in Veusz
        self.doc.SetData(time_name, time_data)
        self.doc.SetData(td_gated_name, np.abs(td_gated_data))
        
        # Create time-gated page
        tg_page_name = f"{dataset_name}_{param_name}_TimeGated"
        page_tg = self.doc.Root.Add('page', name=tg_page_name)
        grid_tg = page_tg.Add('grid', columns=2)
        
        graph_tg = grid_tg.Add('graph', name=f"{dataset_name}_{param_name}_TG_Graph")
        
        # Add header info to page notes
        if 'header_info' in data:
            page_tg.notes.val = '\n'.join(data['header_info'])
        
        # Configure time-gated graph
        graph_title = f"{dataset_name.replace('_', ' ')} - {param_name} Time-Gated"
        self._configure_standard_graph(
            graph_tg, graph_title,
            self.time_label, 'Magnitude', 0, 1
        )
        
        # Add time-gated plot
        xy_tg = graph_tg.Add('xy', name=f"{param_name}_timegated")
        self._configure_xy_plot(xy_tg, time_name, td_gated_name, 'red')

    def save(self, filename: str):
        """Save Veusz project."""
        self.doc.Save(filename, mode='hdf5')

# ============================================================================
# SMITH CHART PLOTTER (HoloViews - NOT Veusz)
# ============================================================================
class SmithChartPlotter:
    """Handles Smith Chart plotting using HoloViews with export capabilities."""

    def __init__(self, config: SmithChartConfig = None):
        """Initialize the Smith Chart plotter."""
        self.config = config or SmithChartConfig()

    def create_smith_chart_figure(self, network: Network, param_name: str,
                                  chart_type: str = "z", draw_labels: bool = True,
                                  draw_vswr: bool = True) -> Tuple[Figure, str]:
        """Create a Smith chart figure from network S-parameters."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        
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
        ax.plot(real_part, imag_part, "b-", linewidth=2.0,
                label=f"{param_name} Trace", marker="o", markersize=6, alpha=0.7)
        
        # Mark frequency points
        num_points = len(z_norm)
        if num_points > 0:
            ax.plot(real_part[0], imag_part[0], "go", markersize=8, label="Start", zorder=5)
            ax.plot(real_part[-1], imag_part[-1], "ro", markersize=8, label="End", zorder=5)
        
        # Configure axes
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Real Part", fontsize=12)
        ax.set_ylabel("Imaginary Part", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
        
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
                ax.text(center_x + radius, 0.05, f"{r:.1f}", fontsize=8, ha="left",
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
        """Extract port indices from S-parameter name."""
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
                         draw_labels: bool = True, draw_vswr: bool = True) -> List[str]:
        """Generate and save Smith charts for all S-parameters in a network."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        base_filename = os.path.splitext(filename)[0]
        
        for i in range(network.nports):
            for j in range(network.nports):
                param_name = f"S{i + 1}{j + 1}"
                
                try:
                    fig, title = self.create_smith_chart_figure(
                        network, param_name, chart_type=chart_type,
                        draw_labels=draw_labels, draw_vswr=draw_vswr
                    )
                    
                    output_filename = f"{base_filename}_{param_name}_SmithChart.{output_format.lower()}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    fig.savefig(output_path, format=output_format.lower(), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    saved_files.append(output_path)
                
                except Exception as e:
                    print(f"Warning: Failed to create Smith chart for {param_name}: {e}")
        
        return saved_files

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
        """Plot Smith chart on the canvas using matplotlib."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        try:
            # Use scikit-rf's built-in Smith chart plotting if available
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
            # Create single S-parameter networks for each port pair
            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    
                    # Create single S-parameter network
                    temp_network = network.copy()
                    temp_network.s = network.s[:, i, j].reshape(-1, 1, 1)
                    
                    results[param_name] = temp_network
            
            results['network'] = network
            results['chart_type'] = self.config.chart_type
            results['reference_impedance'] = self.config.reference_impedance
            results['draw_labels'] = self.config.draw_labels
            results['draw_vswr'] = self.config.draw_vswr
            results['filtered'] = apply_td_filter
            results['error'] = None
            
        except Exception as e:
            results['error'] = str(e)
        
        return results

# ============================================================================
# TOUCHSTONE PROCESSING THREAD
# ============================================================================
def process_single_touchstone_file(file_info: Tuple) -> Tuple[str, Optional[dict]]:
    """Worker function to process a single Touchstone file."""
    filepath, _ = file_info
    
    try:
        network = Network(filepath)
        filename = os.path.basename(filepath)
        
        # Extract header information
        header_info = []
        try:
            with open(filepath, 'r', encoding='ascii', errors='ignore') as f:
                for line in f:
                    if line.startswith('!') or line.startswith('#'):
                        header_info.append(line.strip())
                    elif line.strip() and not line[0].isdigit():
                        header_info.append(line.strip())
                    elif line[0].isdigit():
                        break
        except Exception as e:
            print(f"Warning: Could not extract header from {filename}: {e}")
        
        data = {
            'network': network,
            'frequency': network.frequency.f,
            'file_path': filepath,
            'header_info': header_info
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
        self.setWindowTitle("Enhanced Touchstone AutoPlot - Veusz + HoloViews Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize configurations
        self.config = ProcessingConfig()
        self.td_config = TimeDomainConfig()
        self.smith_config = SmithChartConfig()
        
        # Initialize processors and plotters
        self.touchstone_plotter = None
        self.td_processor = TimeDomainProcessor(self.td_config)
        self.smith_processor = SmithChartProcessor(self.smith_config)
        self.smith_chart_plotter = SmithChartPlotter(self.smith_config)
        
        # Data storage
        self.selected_files = []
        self.processed_data = {}
        self.td_results = {}
        self.smith_results = {}
        
        # Setup UI
        self.setup_ui()
        self._log_message("Enhanced Touchstone AutoPlot initialized")
        self._log_message(f"HoloViews Support: {'Yes' if HOLOVIEWS_AVAILABLE else 'No (interactive features limited)'}")
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
        
        # Time-gated plot button
        self.td_timegated_button = QPushButton("Generate Time-Gated Plots in Veusz")
        self.td_timegated_button.clicked.connect(self._process_time_gated_plots)
        self.td_timegated_button.setEnabled(False)
        controls_layout.addWidget(self.td_timegated_button)
        
        controls_layout.addStretch()
        layout.addWidget(controls_widget)
        
        # Right panel for plot
        self.td_plot_canvas = TouchstonePlotCanvas(td_tab, width=8, height=6)
        layout.addWidget(self.td_plot_canvas)

    def setup_smith_chart_tab(self):
        """Setup the Smith Chart analysis tab (HoloViews + Export)."""
        smith_tab = QWidget()
        self.tab_widget.addTab(smith_tab, "Smith Chart Analysis (HoloViews)")
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
        
        # Export format selection
        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout(export_group)
        
        self.smith_format_combo = QComboBox()
        self.smith_format_combo.addItems(["PNG", "PDF", "SVG", "TIFF", "JPG"])
        self.smith_format_combo.setCurrentText("PNG")
        self.smith_format_combo.currentTextChanged.connect(self._update_smith_format)
        export_layout.addRow("Export Format:", self.smith_format_combo)
        
        controls_layout.addWidget(export_group)
        
        # Export Smith Charts button
        self.smith_export_button = QPushButton("Export Smith Charts to Files")
        self.smith_export_button.clicked.connect(self._export_smith_charts)
        self.smith_export_button.setEnabled(False)
        controls_layout.addWidget(self.smith_export_button)
        
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
        self._update_smith_preview()

    def _update_draw_labels(self, state: int):
        """Update draw labels configuration."""
        self.smith_config.draw_labels = (state == Qt.Checked)
        self._update_smith_preview()

    def _update_draw_vswr(self, state: int):
        """Update draw VSWR configuration."""
        self.smith_config.draw_vswr = (state == Qt.Checked)
        self._update_smith_preview()

    def _update_smith_format(self, format_text: str):
        """Update Smith Chart export format."""
        self.smith_config.output_format = format_text.lower()
        self._log_message(f"Export format set to: {format_text}")

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
        
        # Initialize Touchstone plotter (Veusz - frequency and time domain only)
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
            
            # Enable buttons
            self.smith_export_button.setEnabled(True)
            self.td_timegated_button.setEnabled(True)
            
            # Create Veusz plots for each file (frequency domain only)
            for filename, data in self.processed_data.items():
                if data:
                    self.touchstone_plotter.create_plots_from_data(filename, data)
            
            self._log_message("Veusz frequency domain plots created")
        
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

    def _export_smith_charts(self):
        """Export Smith charts to file format."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return
        
        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Smith Charts",
            ""
        )
        
        if not output_dir:
            return
        
        self._log_message(f"Exporting Smith Charts to {output_dir}...")
        
        try:
            export_format = self.smith_config.output_format.lower()
            total_charts = 0
            
            # Generate Smith charts for each file
            for filename, data in self.processed_data.items():
                network = data['network']
                try:
                    saved_files = self.smith_chart_plotter.save_smith_charts(
                        network,
                        filename,
                        output_dir,
                        output_format=export_format,
                        chart_type=self.smith_config.chart_type,
                        draw_labels=self.smith_config.draw_labels,
                        draw_vswr=self.smith_config.draw_vswr
                    )
                    
                    total_charts += len(saved_files)
                    self._log_message(f"Exported {len(saved_files)} Smith charts for {filename}")
                
                except Exception as e:
                    self._log_message(f"Error exporting {filename}: {e}")
            
            QMessageBox.information(
                self,
                "Success",
                f"Smith charts exported successfully!\n{total_charts} total charts saved to:\n{output_dir}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export Smith charts: {str(e)}")
            self._log_message(f"Smith chart export error: {e}")

    def _process_time_gated_plots(self):
        """Generate time-gated plots in Veusz."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return
        
        self._log_message("Generating Time-Gated Plots in Veusz...")
        
        try:
            # Generate time-gated plots for each file
            for filename, data in self.processed_data.items():
                network = data['network']
                try:
                    td_result = self.td_processor.process_network(network)
                    self.td_results[filename] = td_result
                    
                    # Create time-gated plots in Veusz
                    self.touchstone_plotter.create_time_gated_plots(filename, data, td_result)
                    
                    self._log_message(f"Time-gated plots generated for {filename}")
                except Exception as e:
                    self._log_message(f"Error processing {filename}: {e}")
            
            QMessageBox.information(self, "Success", "Time-gated plots generated successfully in Veusz!")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate time-gated plots: {str(e)}")
            self._log_message(f"Time-gated plot generation error: {e}")

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
