#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Enhanced Touchstone AutoPlot - Complete Production Version with Gate Control Fix.

This version integrates modern Qt GUI interface with comprehensive Touchstone file processing
capabilities using scikit-rf, including:

- Multiprocessing and GPU acceleration (CuPy, PyOpenCL, Taichi)
- Advanced time domain analysis with gating functionality using scikit-rf
- Smith Chart plotting using scikit-rf native charts (interactive with PDF bookmarks)
- Time-gated plot generation in Veusz
- PDF export with bookmarks for Smith charts
- CORRECTED: Proper scikit-rf time-domain conversion using IFFT and windowing
- FIXED: Gate controls always enabled for arbitrary adjustment
- FIXED: Marker colors applied to both lines and markers in Veusz plots

Author: William W. Wallace
Last updated: 2026-01-27 (Gate Control & Color Application Fixes)
"""

# ============================================================================
# IMPORTS - Standard Library
# ============================================================================

import multiprocessing
import os
import subprocess
import sys
import datetime
import tempfile
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
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

# ============================================================================
# IMPORTS - Plotting and Visualization
# ============================================================================

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    import mpld3
    from mpld3 import plugins
    MPLD3_AVAILABLE = True
except ImportError:
    MPLD3_AVAILABLE = False
    print("WARNING: mpld3 not installed. Interactive Smith charts will not be available.")
    print("Install with: pip install mpld3")

# ============================================================================
# IMPORTS - PDF Processing
# ============================================================================

try:
    from PyPDF2 import PdfWriter, PdfReader
except ImportError:
    print("WARNING: PyPDF2 not installed. PDF merging will not be available.")
    print("Install with: pip install PyPDF2")

# ============================================================================
# IMPORTS - Qt Framework
# ============================================================================

if getattr(sys, 'frozen', False):
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QMessageBox, QMainWindow, QWidget,
        QTextEdit, QProgressBar, QCheckBox, QSpinBox, QGroupBox,
        QListWidget, QLineEdit, QTabWidget, QComboBox, QDoubleSpinBox,
        QFormLayout, QListWidgetItem, QSlider
    )
else:
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QMessageBox, QMainWindow, QWidget,
        QTextEdit, QProgressBar, QCheckBox, QSpinBox, QGroupBox,
        QListWidget, QLineEdit, QTabWidget, QComboBox, QDoubleSpinBox,
        QFormLayout, QListWidgetItem, QSlider
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
            print("No GPU acceleration libraries detected - using CPU only")

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for file processing."""
    enable_multiprocessing: bool = True
    enable_gpu_processing: bool = True
    num_processes: int = multiprocessing.cpu_count()
    max_workers: int = multiprocessing.cpu_count()

@dataclass
class TimeDomainConfig:
    """Configuration for time domain analysis."""
    window_type: str = "kaiser"
    window_param: float = 6.0
    gate_center: float = 0.5
    gate_span: float = 0.2
    mode: str = "bandpass"
    tunit: str = "ns"
    auto_gate: bool = True

@dataclass
class SmithChartConfig:
    """Configuration for Smith Chart plotting."""
    chart_type: str = "z"
    draw_labels: bool = True
    draw_vswr: bool = True
    reference_impedance: float = 50.0

# ============================================================================
# S-PARAMETER TO COLOR MAPPING FUNCTION
# ============================================================================

def get_sparam_color(param_name: str) -> str:
    """Get consistent color for S-parameter based on parameter name.
    
    Provides deterministic color mapping for reflection and transmission parameters
    to ensure consistency across all plot types (frequency, time domain, time-gated).
    
    Parameters
    ----------
    param_name : str
        S-parameter name (e.g., "S11", "S21", "S34").
        
    Returns
    -------
    str
        Veusz color name for the parameter.
    """
    color_map = {
        # Reflection parameters (diagonal elements)
        'S11': 'blue',
        'S22': 'red',
        'S33': 'green',
        'S44': 'magenta',
        'S55': 'orange',
        'S66': 'brown',
        'S77': 'pink',
        'S88': 'grey',
        
        # Forward transmission parameters
        'S21': 'darkblue',
        'S31': 'darkgreen',
        'S41': 'darkmagenta',
        'S51': 'darkcyan',
        'S61': 'olive',
        'S71': 'navy',
        'S81': 'teal',
        
        # Other transmission parameters
        'S12': 'cyan',
        'S13': 'purple',
        'S14': 'darkcyan',
        'S23': 'lightblue',
        'S24': 'lightgreen',
        'S34': 'indigo',
        'S43': 'maroon',
        'S52': 'coral',
        'S62': 'khaki',
        'S72': 'lavender',
        'S82': 'lightgrey',
    }
    return color_map.get(param_name, 'auto')

# ============================================================================
# TIME DOMAIN PROCESSOR
# ============================================================================

class TimeDomainProcessor:
    """Handles time domain analysis of S-parameter data using scikit-rf.
    
    Uses windowed IFFT for frequency-to-time conversion with proper scaling
    and supports time-domain gating with configurable window types.
    """

    def __init__(self, config: TimeDomainConfig = None):
        """Initialize time domain processor.
        
        Parameters
        ----------
        config : TimeDomainConfig
            Configuration for time domain analysis.
        """
        self.config = config or TimeDomainConfig()

    def process_network(self, network: Network, apply_window: bool = True) -> dict:
        """Process a network for time domain analysis using windowed IFFT.
        
        Converts frequency-domain S-parameters to time-domain impulse response
        using windowed IFFT with optional time-domain gating.
        
        Parameters
        ----------
        network : Network
            scikit-rf Network object with equally-spaced frequency points.
        apply_window : bool
            Whether to apply windowing in frequency domain before IFFT.
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'frequency': Frequency array (Hz)
            - 'time': Time array (ns)
            - 'S{i}{j}_td': Time-domain unfiltered impulse response
            - 'S{i}{j}_td_filtered': Time-domain gated impulse response
            - 'error': Error message if processing failed, None otherwise.
        """
        results = {}
        try:
            freq_hz = network.frequency.f
            num_freq = len(freq_hz)
            freq_spacing_hz = np.mean(np.diff(freq_hz))

            # Calculate time axis: Δt = 1 / (N * Δf)
            delta_t = 1.0 / (num_freq * freq_spacing_hz)
            time_ns = np.arange(num_freq) * delta_t * 1e9

            # Process each S-parameter
            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    s_param = network.s[:, i, j]

                    # Apply frequency-domain windowing if requested
                    if apply_window:
                        try:
                            if self.config.window_type == "kaiser":
                                window = windows.kaiser(num_freq, self.config.window_param)
                            elif self.config.window_type == "hamming":
                                window = windows.hamming(num_freq)
                            elif self.config.window_type == "hann":
                                window = windows.hann(num_freq)
                            elif self.config.window_type == "blackman":
                                window = windows.blackman(num_freq)
                            else:  # boxcar
                                window = np.ones(num_freq)
                            s_windowed = s_param * window
                        except Exception as e:
                            print(f"Warning: Windowing failed for {param_name}: {e}")
                            s_windowed = s_param
                    else:
                        s_windowed = s_param

                    # Convert to time domain using IFFT with proper scaling
                    # Scaling factor: multiply by N for energy conservation
                    s_time_unfiltered = np.fft.ifft(s_windowed) * num_freq
                    results[f"{param_name}_td"] = s_time_unfiltered

                    # Apply time-domain gating if auto_gate enabled
                    if self.config.auto_gate:
                        try:
                            s_time_gated = self._apply_time_gate(
                                s_time_unfiltered,
                                time_ns,
                                self.config.gate_center,
                                self.config.gate_span
                            )
                            results[f"{param_name}_td_filtered"] = s_time_gated
                        except Exception as e:
                            print(f"Warning: Time gating failed for {param_name}: {e}")
                            results[f"{param_name}_td_filtered"] = s_time_unfiltered
                    else:
                        results[f"{param_name}_td_filtered"] = s_time_unfiltered

            # Store frequency and time information
            results['frequency'] = freq_hz
            results['time'] = time_ns
            results['error'] = None

        except Exception as e:
            print(f"Error in time domain processing: {e}")
            results['error'] = str(e)

        return results

    def _apply_time_gate(self, time_domain_data: np.ndarray, time_array: np.ndarray,
                         gate_center: float, gate_span: float) -> np.ndarray:
        """Apply time-domain gating with Hamming window smoothing.
        
        Parameters
        ----------
        time_domain_data : np.ndarray
            Time-domain impulse response.
        time_array : np.ndarray
            Time axis in nanoseconds.
        gate_center : float
            Gate center time in nanoseconds.
        gate_span : float
            Gate span (width) in nanoseconds.
            
        Returns
        -------
        np.ndarray
            Gated time-domain data.
        """
        gate_start = gate_center - gate_span / 2
        gate_stop = gate_center + gate_span / 2

        # Find indices corresponding to gate window
        gate_indices = np.where((time_array >= gate_start) & (time_array <= gate_stop))[0]

        if len(gate_indices) < 2:
            return time_domain_data  # Return unmodified if gate too small

        # Create smooth gate with Hamming window
        gate = np.zeros_like(time_domain_data)
        gate[gate_indices] = windows.hamming(len(gate_indices))

        return time_domain_data * gate

    def apply_gpu_acceleration(self, s_data: np.ndarray) -> np.ndarray:
        """Apply GPU-accelerated IFFT to S-parameter data if available.
        
        Parameters
        ----------
        s_data : np.ndarray
            S-parameter data array (complex).
            
        Returns
        -------
        np.ndarray
            Time-domain result (complex), GPU or CPU computed.
        """
        if not GPU_AVAILABLE:
            return np.fft.ifft(s_data) * len(s_data)

        try:
            if GPU_AVAILABLE == "cupy":
                import cupy as cp

                # Transfer to GPU, perform IFFT with scaling, transfer back
                gpu_data = cp.asarray(s_data, dtype=cp.complex128)
                gpu_result = cp.fft.ifft(gpu_data) * len(s_data)
                return cp.asnumpy(gpu_result)

            elif GPU_AVAILABLE == "opencl":
                # OpenCL doesn't have built-in FFT in standard libraries
                # Fall back to NumPy
                print("Note: OpenCL lacks production FFT support, using NumPy CPU")
                return np.fft.ifft(s_data) * len(s_data)

            elif GPU_AVAILABLE == "taichi":
                # Taichi is better for custom kernels, not FFT
                # Fall back to NumPy
                print("Note: Taichi is designed for custom kernels, using NumPy CPU for FFT")
                return np.fft.ifft(s_data) * len(s_data)

        except Exception as e:
            print(f"Warning: GPU acceleration failed, falling back to CPU: {e}")

        return np.fft.ifft(s_data) * len(s_data)

# ============================================================================
# SMITH CHART PLOTTER - SCIKIT-RF NATIVE
# ============================================================================

class SmithChartPlottermpld3:
    """Handles interactive Smith Chart plotting using scikit-rf native implementation."""

    def __init__(self, reference_impedance: float = 50.0):
        """Initialize the Smith Chart plotter."""
        self.reference_impedance = reference_impedance
        self.plots = {}

    def create_smith_chart(self, network: Network, param_name: str = "S11",
                          chart_type: str = "z") -> Tuple[Figure, Network]:
        """Create an interactive Smith Chart visualization.
        
        Parameters
        ----------
        network : Network
            Scikit-rf Network object with single S-parameter data (1x1 network).
        param_name : str
            Parameter name for labeling (e.g., "S11", "S21").
        chart_type : str
            Chart type: "z" for impedance, "y" for admittance.
            
        Returns
        -------
        Tuple[Figure, Network]
            Matplotlib figure with Smith chart and network object.
        """
        try:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            # Use scikit-rf's native Smith chart plotting
            network.plot_s_smith(
                ax=ax,
                chart_type=chart_type,
                draw_labels=True,
                draw_vswr=True,
                show_legend=True
            )

            # Customize title
            chart_label = "Impedance" if chart_type == "z" else "Admittance"
            ax.set_title(f'{param_name} - Smith Chart ({chart_label})',
                        fontsize=14, fontweight='bold', pad=20)

            # Add mpld3 hover tooltips if available
            if MPLD3_AVAILABLE:
                try:
                    freq_ghz = network.frequency.f / 1e9
                    s_data = network.s[:, 0, 0]
                    labels = []

                    for i in range(len(freq_ghz)):
                        freq = freq_ghz[i]
                        s_val = s_data[i]
                        mag = np.abs(s_val)
                        phase = np.angle(s_val, deg=True)

                        if mag < 1.0:
                            vswr = (1 + mag) / (1 - mag + 1e-12)
                        else:
                            vswr = 10.0

                        label = (f"{param_name}\n"
                                f"Frequency: {freq:.2f} GHz\n"
                                f"Magnitude: {mag:.4f}\n"
                                f"Phase: {phase:.1f}°\n"
                                f"VSWR: {vswr:.2f}")
                        labels.append(label)

                    for artist in ax.get_children():
                        if hasattr(artist, 'get_offsets') and len(artist.get_offsets()) > 0:
                            tooltip = plugins.PointHTMLTooltip(artist, labels, voffset=10, hoffset=10)
                            mpld3.plugins.connect(fig, tooltip)
                            break
                        elif hasattr(artist, 'get_xydata') and len(artist.get_xydata()) > 0:
                            xy_data = artist.get_xydata()
                            if len(xy_data) == len(labels):
                                tooltip = plugins.LineLabelTooltip(artist, labels)
                                mpld3.plugins.connect(fig, tooltip)
                                break

                except Exception as e:
                    print(f"Warning: Could not add tooltips: {e}")

            self.plots[param_name] = fig
            return fig, network

        except Exception as e:
            print(f"Error creating Smith chart: {e}")
            raise

    def export_smith_chart(self, network: Network, param_name: str, chart_type: str,
                          output_path: str, export_format: str) -> bool:
        """Export a Smith chart to file.
        
        Parameters
        ----------
        network : Network
            Single S-parameter network (1x1).
        param_name : str
            Parameter name (e.g., "S11").
        chart_type : str
            "z" for impedance, "y" for admittance.
        output_path : str
            Output file path.
        export_format : str
            Format: "html", "png", "svg", or "pdf".
            
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            fig, _ = self.create_smith_chart(network, param_name, chart_type)

            if export_format.lower() == "html":
                if not MPLD3_AVAILABLE:
                    print("Warning: mpld3 not available, saving as PNG instead")
                    fig.savefig(output_path.replace('.html', '.png'),
                              format='png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    return True

                html_str = mpld3.fig_to_html(fig)
                with open(output_path, 'w') as f:
                    f.write(html_str)

            elif export_format.lower() == "png":
                fig.savefig(output_path, format='png', dpi=150, bbox_inches='tight')

            elif export_format.lower() == "svg":
                fig.savefig(output_path, format='svg', bbox_inches='tight')

            elif export_format.lower() == "pdf":
                fig.savefig(output_path, format='pdf', bbox_inches='tight')

            else:
                print(f"Unknown export format: {export_format}")
                plt.close(fig)
                return False

            plt.close(fig)
            return True

        except Exception as e:
            print(f"Error exporting Smith chart: {e}")
            return False

    @staticmethod
    def create_pdf_with_bookmarks(pdf_files: List[Tuple[str, str]], output_path: str) -> bool:
        """Merge PDFs with bookmarks for navigation.
        
        Parameters
        ----------
        pdf_files : List[Tuple[str, str]]
            List of (filepath, bookmark_name) tuples.
        output_path : str
            Output PDF path.
            
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            writer = PdfWriter()
            current_page = 0

            for pdf_path, bookmark_name in pdf_files:
                try:
                    reader = PdfReader(pdf_path)
                    num_pages = len(reader.pages)

                    for page in reader.pages:
                        writer.add_page(page)

                    writer.add_outline_item(bookmark_name, current_page)
                    current_page += num_pages

                except Exception as e:
                    print(f"Warning: Could not process {pdf_path}: {e}")
                    continue

            writer.page_mode = "/UseOutlines"

            with open(output_path, 'wb') as f:
                writer.write(f)

            return True

        except Exception as e:
            print(f"Error creating PDF with bookmarks: {e}")
            return False

# ============================================================================
# TOUCHSTONE PLOTTER - VEUSZ
# ============================================================================

class TouchstonePlotter:
    """Handles Veusz plotting for Touchstone files."""

    def __init__(self, plot_title: str = "Touchstone S-Parameter Analysis",
                 dataset_name: str = "Touchstone_Dataset"):
        """Initialize the Veusz plotter."""
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.doc = vz.Embedded("Touchstone_AutoPlot")
        self.doc.EnableToolbar()

    @contextmanager
    def _wrap_widget(self, widget):
        """Context manager for widget access."""
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
            g.x.label.val = x_label
            g.y.label.val = y_label
            g.x.GridLines.hide.val = False
            g.y.GridLines.hide.val = False
            g.x.MinorGridLines.hide.val = False
            g.y.MinorGridLines.hide.val = False
            g.y.min.val = y_min
            g.y.max.val = y_max

    def _configure_xy_plot(self, xy, x_data, y_data, color):
        """Configure an XY plot with consistent color for line AND markers."""
        with self._wrap_widget(xy) as plot:
            plot.xData.val = x_data
            plot.yData.val = y_data
            plot.PlotLine.color.val = color
            plot.PlotLine.width.val = '2pt'
            plot.marker.val = 'circle'
            plot.markerSize.val = '2pt'
            # FIXED: Apply color to markers, not 'auto'
            plot.MarkerFill.color.val = color
            plot.MarkerFill.transparency.val = 20
            # Also set marker outline color for consistency
            plot.MarkerLine.color.val = color
            plot.MarkerLine.width.val = '1pt'

    def create_plots_from_data(self, filename: str, data: dict):
        """Create frequency domain plots from Touchstone data."""
        try:
            network = data['network']
            dataset_name = filename.replace('.', '_').replace('-', '_')
            self._create_frequency_page(dataset_name, data)
        except Exception as e:
            print(f"Error creating plots: {e}")

    def _create_frequency_page(self, dataset_name, data):
        """Create frequency domain plot page."""
        network = data['network']
        n_ports = network.nports
        freq_name = f"{dataset_name}_freq"

        self.doc.SetData(freq_name, network.frequency.f / 1e9)

        page = self.doc.Root.Add('page', name=f"SParam_{dataset_name}")
        grid = page.Add('grid', columns=2)

        if 'header_info' in data:
            page.notes.val = '\n'.join(data['header_info'])

        graph_mag = grid.Add('graph', name=f"{dataset_name}_Mag")
        graph_phase = grid.Add('graph', name=f"{dataset_name}_Phase")

        self._configure_standard_graph(
            graph_mag,
            f"{dataset_name.replace('_', ' ')} - Magnitude",
            'Frequency (GHz)', 'Magnitude (dB)', -80, 20
        )

        self._configure_standard_graph(
            graph_phase,
            f"{dataset_name.replace('_', ' ')} - Phase",
            'Frequency (GHz)', 'Phase (degrees)', -180, 180
        )

        for i in range(n_ports):
            for j in range(n_ports):
                param_name = f"S{i + 1}{j + 1}"
                mag_name = f"{dataset_name}_{param_name}_mag"
                phase_name = f"{dataset_name}_{param_name}_phase"

                s_param = network.s[:, i, j]
                mag_db = 20 * np.log10(np.abs(s_param) + 1e-12)
                phase_deg = np.angle(s_param, deg=True)

                self.doc.SetData(mag_name, mag_db)
                self.doc.SetData(phase_name, phase_deg)

                xy_mag = graph_mag.Add('xy', name=f"{param_name}_mag")
                # FIXED: Use consistent color for both lines and markers
                self._configure_xy_plot(xy_mag, freq_name, mag_name, get_sparam_color(param_name))

                xy_phase = graph_phase.Add('xy', name=f"{param_name}_phase")
                # FIXED: Use consistent color for both lines and markers
                self._configure_xy_plot(xy_phase, freq_name, phase_name, get_sparam_color(param_name))

    def create_time_domain_plots(self, filename: str, data: dict, td_result: dict):
        """Create time domain plots in Veusz."""
        if 'error' in td_result and td_result['error']:
            return

        try:
            network = data['network']
            dataset_name = filename.replace('.', '_').replace('-', '_')

            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    self._create_time_domain_page(dataset_name, param_name, data, td_result)

        except Exception as e:
            print(f"Error creating time domain plots: {e}")

    def _create_time_domain_page(self, dataset_name, param_name, data, td_result):
        """Create time domain plot page for a specific S-parameter."""
        time_name = f"{dataset_name}_{param_name}_time"
        td_unfilt_name = f"{dataset_name}_{param_name}_td"
        td_filt_name = f"{dataset_name}_{param_name}_tdf"

        time_data = td_result.get('time', np.array([]))
        td_unfilt_data = td_result.get(f"{param_name}_td", np.array([]))
        td_filt_data = td_result.get(f"{param_name}_td_filtered", np.array([]))

        self.doc.SetData(time_name, time_data)
        self.doc.SetData(td_unfilt_name, np.abs(td_unfilt_data))
        self.doc.SetData(td_filt_name, np.abs(td_filt_data))

        td_page_name = f"{dataset_name}_{param_name}_TimeDomain"
        page_td = self.doc.Root.Add('page', name=td_page_name)
        grid_td = page_td.Add('grid', columns=2)
        graph_td = grid_td.Add('graph', name=f"{dataset_name}_{param_name}_TD_Graph")

        if 'header_info' in data:
            page_td.notes.val = '\n'.join(data['header_info'])

        graph_title = f"{dataset_name.replace('_', ' ')} - {param_name} Time Domain"

        self._configure_standard_graph(
            graph_td, graph_title,
            'Time (ns)', 'Magnitude', 0, 1
        )

        xy_unfilt = graph_td.Add('xy', name=f"{param_name}_td_unfilt")
        with self._wrap_widget(xy_unfilt) as plot:
            plot.xData.val = time_name
            plot.yData.val = td_unfilt_name
            plot.PlotLine.style.val = 'dotted'
            plot.PlotLine.width.val = '1pt'
            plot.PlotLine.color.val = 'blue'
            plot.marker.val = 'none'

        xy_filt = graph_td.Add('xy', name=f"{param_name}_td_filt")
        # FIXED: Use consistent color for both lines and markers
        self._configure_xy_plot(xy_filt, time_name, td_filt_name, get_sparam_color(param_name))

    def create_time_gated_plots(self, filename: str, data: dict, td_result: dict):
        """Create time-gated plots in Veusz with _timegated postfix."""
        if 'error' in td_result and td_result['error']:
            return

        try:
            network = data['network']
            dataset_name = filename.replace('.', '_').replace('-', '_')

            for i in range(network.nports):
                for j in range(network.nports):
                    param_name = f"S{i + 1}{j + 1}"
                    self._create_time_gated_page(dataset_name, param_name, data, td_result)

        except Exception as e:
            print(f"Error creating time-gated plots: {e}")

    def _create_time_gated_page(self, dataset_name, param_name, data, td_result):
        """Create time-gated plot page with _timegated tag."""
        time_name = f"{dataset_name}_{param_name}_timegated_time"
        td_gated_name = f"{dataset_name}_{param_name}_timegated"

        time_data = td_result.get('time', np.array([]))
        td_gated_data = td_result.get(f"{param_name}_td_filtered", np.array([]))

        self.doc.SetData(time_name, time_data)
        self.doc.SetData(td_gated_name, np.abs(td_gated_data))

        tg_page_name = f"{dataset_name}_{param_name}_TimeGated"
        page_tg = self.doc.Root.Add('page', name=tg_page_name)
        grid_tg = page_tg.Add('grid', columns=2)
        graph_tg = grid_tg.Add('graph', name=f"{dataset_name}_{param_name}_TG_Graph")

        if 'header_info' in data:
            page_tg.notes.val = '\n'.join(data['header_info'])

        graph_title = f"{dataset_name.replace('_', ' ')} - {param_name} Time-Gated"

        self._configure_standard_graph(
            graph_tg, graph_title,
            'Time (ns)', 'Magnitude', 0, 1
        )

        xy_tg = graph_tg.Add('xy', name=f"{param_name}_timegated")
        # FIXED: Use consistent color for both lines and markers
        self._configure_xy_plot(xy_tg, time_name, td_gated_name, get_sparam_color(param_name))

    def save(self, filename: str):
        """Save Veusz project."""
        self.doc.Save(filename, mode='hdf5')

# ============================================================================
# MATPLOTLIB CANVAS FOR PLOTS
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

    def plot_smith_chart(self, network, title="Smith Chart", chart_type="z",
                        draw_labels=True, draw_vswr=True):
        """Plot Smith chart using scikit-rf native implementation."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        try:
            network.plot_s_smith(
                ax=ax,
                chart_type=chart_type,
                draw_labels=draw_labels,
                draw_vswr=draw_vswr,
                show_legend=True
            )

            chart_label = "Impedance" if chart_type == "z" else "Admittance"
            ax.set_title(f"{title} - Smith Chart ({chart_label})", fontweight='bold', fontsize=12)

        except Exception as e:
            print(f"Smith chart plotting error: {e}")

            # Fallback to complex plane
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

    def plot_time_domain(self, time, td_data, td_filtered_data=None, title="Time Domain"):
        """Plot time domain data."""
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

# ============================================================================
# PROCESSING THREAD
# ============================================================================

def process_single_touchstone_file(file_info: Tuple) -> Tuple[str, Optional[dict]]:
    """Worker function to process a single Touchstone file."""
    filepath, _ = file_info

    try:
        network = Network(filepath)
        filename = os.path.basename(filepath)
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
    """Thread for processing Touchstone files."""

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
        self.setWindowTitle("Enhanced Touchstone AutoPlot - Complete Edition")
        self.setGeometry(100, 100, 1600, 1000)

        self.config = ProcessingConfig()
        self.td_config = TimeDomainConfig()
        self.smith_config = SmithChartConfig()
        self.touchstone_plotter = None
        self.td_processor = TimeDomainProcessor(self.td_config)
        self.smith_plotter_mpld3 = SmithChartPlottermpld3()
        self.selected_files = []
        self.processed_data = {}
        self.td_results = {}
        self.smith_networks = {}

        self.setup_ui()

        self._log_message("Enhanced Touchstone AutoPlot initialized")
        self._log_message(f"GPU Support: {GPU_AVAILABLE or 'None'}")
        self._log_message(f"CPU Cores Available: {multiprocessing.cpu_count()}")
        self._log_message(f"mpld3 Support: {'Available' if MPLD3_AVAILABLE else 'Not installed'}")

    def setup_ui(self):
        """Set up the user interface with tab widget."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.setup_main_tab()
        self.setup_time_domain_tab()
        self.setup_smith_chart_tab()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(120)
        self.status_text.setReadOnly(True)
        main_layout.addWidget(self.status_text)

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

        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)

        file_select_group = QGroupBox("File Selection for Preview")
        file_select_layout = QVBoxLayout(file_select_group)

        self.td_file_combo = QComboBox()
        self.td_file_combo.currentTextChanged.connect(self._update_td_preview)
        file_select_layout.addWidget(self.td_file_combo)

        controls_layout.addWidget(file_select_group)

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

        gate_group = QGroupBox("Gating Settings")
        gate_layout = QFormLayout(gate_group)

        self.auto_gate_checkbox = QCheckBox("Auto Gate")
        self.auto_gate_checkbox.setChecked(self.td_config.auto_gate)
        self.auto_gate_checkbox.stateChanged.connect(self._update_auto_gate)
        gate_layout.addRow(self.auto_gate_checkbox)

        self.gate_center_spin = QDoubleSpinBox()
        self.gate_center_spin.setRange(-100.0, 100.0)
        self.gate_center_spin.setValue(self.td_config.gate_center)
        self.gate_center_spin.setSuffix(" ns")
        self.gate_center_spin.valueChanged.connect(self._update_gate_center)
        gate_layout.addRow("Gate Center:", self.gate_center_spin)

        self.gate_span_spin = QDoubleSpinBox()
        self.gate_span_spin.setRange(0.01, 100.0)
        self.gate_span_spin.setValue(self.td_config.gate_span)
        self.gate_span_spin.setSuffix(" ns")
        self.gate_span_spin.valueChanged.connect(self._update_gate_span)
        gate_layout.addRow("Gate Span:", self.gate_span_spin)

        # FIXED: Gate controls are always enabled for user adjustment.
        # The checkbox controls whether gating is APPLIED, not whether controls are accessible.
        # Removed: self.gate_center_spin.setEnabled(self.td_config.auto_gate)
        # Removed: self.gate_span_spin.setEnabled(self.td_config.auto_gate)

        controls_layout.addWidget(gate_group)

        method_group = QGroupBox("Processing Settings")
        method_layout = QFormLayout(method_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["bandpass", "bandstop"])
        self.mode_combo.setCurrentText(self.td_config.mode)
        self.mode_combo.currentTextChanged.connect(self._update_mode)
        method_layout.addRow("Mode:", self.mode_combo)

        controls_layout.addWidget(method_group)

        self.td_timegated_button = QPushButton("Generate Time-Gated Plots in Veusz")
        self.td_timegated_button.clicked.connect(self._process_time_gated_plots)
        self.td_timegated_button.setEnabled(False)
        controls_layout.addWidget(self.td_timegated_button)

        controls_layout.addStretch()

        layout.addWidget(controls_widget)

        self.td_plot_canvas = TouchstonePlotCanvas(td_tab, width=8, height=6)
        layout.addWidget(self.td_plot_canvas)

    def setup_smith_chart_tab(self):
        """Setup the Smith Chart analysis tab."""
        smith_tab = QWidget()
        self.tab_widget.addTab(smith_tab, "Smith Chart Analysis")

        layout = QHBoxLayout(smith_tab)

        controls_widget = QWidget()
        controls_widget.setMaximumWidth(380)
        controls_layout = QVBoxLayout(controls_widget)

        file_select_group = QGroupBox("File Selection for Preview")
        file_select_layout = QVBoxLayout(file_select_group)

        self.smith_file_combo = QComboBox()
        self.smith_file_combo.currentTextChanged.connect(self._update_smith_preview)
        file_select_layout.addWidget(self.smith_file_combo)

        controls_layout.addWidget(file_select_group)

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

        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout(export_group)

        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG", "SVG", "PDF", "HTML (Interactive)"])
        export_layout.addRow("Format:", self.export_format_combo)

        self.pdf_bookmarks_checkbox = QCheckBox("PDF with Bookmarks")
        self.pdf_bookmarks_checkbox.setChecked(False)
        export_layout.addRow(self.pdf_bookmarks_checkbox)

        controls_layout.addWidget(export_group)

        self.smith_process_button = QPushButton("Generate Smith Charts")
        self.smith_process_button.clicked.connect(self._process_smith_charts_mpld3)
        self.smith_process_button.setEnabled(False)
        controls_layout.addWidget(self.smith_process_button)

        self.smith_export_button = QPushButton("Export Smith Charts")
        self.smith_export_button.clicked.connect(self._export_smith_charts)
        self.smith_export_button.setEnabled(False)
        controls_layout.addWidget(self.smith_export_button)

        controls_layout.addStretch()

        layout.addWidget(controls_widget)

        right_layout = QVBoxLayout()

        self.smith_plot_canvas = TouchstonePlotCanvas(smith_tab, width=10, height=10)
        self.smith_toolbar = NavigationToolbar(self.smith_plot_canvas, smith_tab)

        right_layout.addWidget(self.smith_toolbar)
        right_layout.addWidget(self.smith_plot_canvas)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        layout.addWidget(right_widget)

    # ========================================================================
    # CALLBACKS
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
        self.smith_networks.clear()
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

    def _update_mp_config(self, state: int):
        """Update multiprocessing configuration."""
        self.config.enable_multiprocessing = (state == Qt.Checked)
        self._log_message(f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}")

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

    def _update_auto_gate(self, state: int):
        """Update auto gate configuration.
        
        FIXED: Gate controls are always enabled for user adjustment.
        The checkbox controls whether gating is APPLIED, not whether controls are accessible.
        """
        self.td_config.auto_gate = (state == Qt.Checked)
        # Gate controls remain enabled - user can adjust them regardless of checkbox state
        self._update_td_preview()

    def _update_gate_center(self, value: float):
        """Update gate center configuration."""
        self.td_config.gate_center = value
        self._update_td_preview()

    def _update_gate_span(self, value: float):
        """Update gate span configuration."""
        self.td_config.gate_span = value
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

    def _update_draw_labels(self, state: int):
        """Update draw labels configuration."""
        self.smith_config.draw_labels = (state == Qt.Checked)
        self._update_smith_preview()

    def _update_draw_vswr(self, state: int):
        """Update draw VSWR configuration."""
        self.smith_config.draw_vswr = (state == Qt.Checked)
        self._update_smith_preview()

    def _update_td_preview(self):
        """Update time domain preview."""
        current_file = self.td_file_combo.currentText()
        if current_file and current_file in self.processed_data:
            try:
                network = self.processed_data[current_file]['network']
                td_result = self.td_processor.process_network(network)
                self.td_results[current_file] = td_result

                time_data = td_result.get('time', np.array([]))
                td_data = td_result.get('S11_td', np.array([]))
                td_filtered = td_result.get('S11_td_filtered', np.array([]))

                plot_title = f"{current_file} - Time Domain"

                self.td_plot_canvas.plot_time_domain(
                    time_data, td_data, td_filtered, title=plot_title
                )

            except Exception as e:
                self._log_message(f"Time domain preview error: {e}")

    def _update_smith_preview(self):
        """Update Smith chart preview."""
        current_file = self.smith_file_combo.currentText()
        if current_file and current_file in self.processed_data:
            try:
                network = self.processed_data[current_file]['network']
                preview_network = network.copy()
                preview_network.s = network.s[:, 0, 0].reshape(-1, 1, 1)

                plot_title = f"{current_file} - S11"

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

        self.touchstone_plotter = TouchstonePlotter(
            plot_title=self.plot_title_edit.text(),
            dataset_name=self.dataset_name_edit.text()
        )

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
            self.smith_process_button.setEnabled(True)
            self.td_timegated_button.setEnabled(True)

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

    def _process_smith_charts_mpld3(self):
        """Generate interactive Smith charts."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return

        self._log_message("Generating Smith Charts...")

        try:
            self.smith_networks.clear()

            for filename, data in self.processed_data.items():
                network = data['network']
                try:
                    for i in range(network.nports):
                        for j in range(network.nports):
                            param_name = f"S{i + 1}{j + 1}"
                            single_param_network = network.copy()
                            single_param_network.s = network.s[:, i, j].reshape(-1, 1, 1)

                            key = f"{filename}_{param_name}"
                            self.smith_networks[key] = {
                                'network': single_param_network,
                                'param_name': param_name,
                                'filename': filename
                            }

                    self._log_message(f"Smith chart networks prepared for {filename}")

                except Exception as e:
                    self._log_message(f"Error processing {filename}: {e}")

            if self.smith_networks:
                first_key = list(self.smith_networks.keys())[0]
                first_network = self.smith_networks[first_key]['network']
                first_param = self.smith_networks[first_key]['param_name']

                self.smith_plot_canvas.plot_smith_chart(
                    first_network,
                    title=f"{self.smith_networks[first_key]['filename']} - {first_param}",
                    chart_type=self.smith_config.chart_type,
                    draw_labels=self.smith_config.draw_labels,
                    draw_vswr=self.smith_config.draw_vswr
                )

                self.smith_export_button.setEnabled(True)
                self._log_message(f"Displayed: {first_key}")

                QMessageBox.information(self, "Success",
                    f"Prepared {len(self.smith_networks)} Smith charts for export.\n"
                    "Click Export to save in selected format.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare Smith charts: {str(e)}")
            self._log_message(f"Smith chart preparation error: {e}")

    def _export_smith_charts(self):
        """Export Smith charts in selected format."""
        if not self.smith_networks:
            QMessageBox.warning(self, "No Charts", "Please generate Smith charts first.")
            return

        export_format = self.export_format_combo.currentText()
        use_pdf_bookmarks = self.pdf_bookmarks_checkbox.isChecked()

        if "PNG" in export_format:
            ext = ".png"
            fmt = "png"
        elif "SVG" in export_format:
            ext = ".svg"
            fmt = "svg"
        elif "PDF" in export_format:
            ext = ".pdf"
            fmt = "pdf"
        else:
            ext = ".html"
            fmt = "html"

        try:
            if fmt == "pdf" and use_pdf_bookmarks:
                file_dialog = QFileDialog()
                pdf_output, _ = file_dialog.getSaveFileName(
                    self,
                    "Save Combined Smith Chart PDF with Bookmarks",
                    "SmithCharts_Combined.pdf",
                    "PDF Files (*.pdf)"
                )

                if not pdf_output:
                    return

                temp_dir = tempfile.mkdtemp()
                pdf_files = []
                exported_count = 0

                try:
                    for key, chart_data in self.smith_networks.items():
                        filename = chart_data['filename']
                        param_name = chart_data['param_name']
                        network = chart_data['network']
                        base_name = os.path.splitext(filename)[0]
                        temp_pdf = os.path.join(temp_dir, f"{base_name}_{param_name}_temp.pdf")

                        success = self.smith_plotter_mpld3.export_smith_chart(
                            network,
                            param_name,
                            self.smith_config.chart_type,
                            temp_pdf,
                            "pdf"
                        )

                        if success:
                            exported_count += 1
                            pdf_files.append((temp_pdf, f"{param_name}"))
                            self._log_message(f"Created temporary PDF: {param_name}")
                        else:
                            self._log_message(f"Failed to create PDF: {param_name}")

                    if pdf_files:
                        if SmithChartPlottermpld3.create_pdf_with_bookmarks(pdf_files, pdf_output):
                            self._log_message(f"Created bookmarked PDF: {os.path.basename(pdf_output)}")
                            QMessageBox.information(
                                self, "Export Complete",
                                f"Successfully created combined PDF with {exported_count} bookmarked pages:\n{pdf_output}"
                            )
                            self._log_message(f"Smith chart PDF export completed: {pdf_output}")
                        else:
                            QMessageBox.critical(self, "Error", "Failed to merge PDFs with bookmarks")
                    else:
                        QMessageBox.warning(self, "No Charts", "No Smith charts could be exported.")

                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            else:
                save_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select directory to save Smith charts"
                )

                if not save_dir:
                    return

                exported_count = 0

                for key, chart_data in self.smith_networks.items():
                    filename = chart_data['filename']
                    param_name = chart_data['param_name']
                    network = chart_data['network']
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}_{param_name}{ext}"
                    output_path = os.path.join(save_dir, output_filename)

                    success = self.smith_plotter_mpld3.export_smith_chart(
                        network,
                        param_name,
                        self.smith_config.chart_type,
                        output_path,
                        fmt
                    )

                    if success:
                        exported_count += 1
                        self._log_message(f"Exported: {output_filename}")
                    else:
                        self._log_message(f"Failed to export: {output_filename}")

                QMessageBox.information(
                    self, "Export Complete",
                    f"Successfully exported {exported_count} Smith charts to:\n{save_dir}"
                )

                self._log_message(f"Smith chart export completed: {exported_count} files")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export Smith charts:\n{e}")
            self._log_message(f"Smith chart export error: {e}")

    def _process_time_gated_plots(self):
        """Generate time-gated plots in Veusz."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return

        self._log_message("Generating Time-Gated Plots in Veusz...")

        try:
            for filename, data in self.processed_data.items():
                network = data['network']
                try:
                    td_result = self.td_processor.process_network(network)
                    self.td_results[filename] = td_result
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
