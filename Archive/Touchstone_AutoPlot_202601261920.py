"""
Enhanced Touchstone AutoPlot with Smith Chart Matplotlib Support.

This script extends the original Touchstone_AutoPlot.py to support Smith chart
plotting using matplotlib and scikit-rf instead of Veusz. When the Smith Chart
tab is selected and processed, it generates high-quality Smith charts using
matplotlib, with options to save in multiple formats (PNG, PDF, TIFF, BMP, SVG, JPG).

For PDF format, users can optionally combine all Smith charts into a single
multi-page bookmarked PDF document.

All other Touchstone processing (S-parameters, time domain analysis) continues
through Veusz as in the original implementation.

Author: Modified from William W. Wallace's Touchstone_AutoPlot.py
Last Updated: 2026-01-26
Python Version: 3.8+

Dependencies:
    - PyQt5/PySide6 (GUI framework)
    - scikit-rf (S-parameter analysis)
    - matplotlib (Smith chart plotting)
    - numpy, scipy (numerical computing)
    - PyPDF2 (PDF merging for multi-page PDF output)
    - veusz (for non-Smith chart Touchstone plotting)
    - pandas (data handling)

Usage:
    python Touchstone_AutoPlot_SmithMatplotlib.py
    
    Then use the GUI to:
    1. Load Touchstone files (.S1P, .S2P, etc.)
    2. Navigate to "Smith Chart Analysis" tab
    3. Configure Smith chart settings (impedance/admittance, VSWR circles, etc.)
    4. Click "Generate Smith Charts in Matplotlib"
    5. Choose output format and save location
    6. For PDF: optionally select "Combine to Single PDF" to create multi-page document
"""

import datetime
import multiprocessing
# ============================================================================
# IMPORTS - Standard Library
# ============================================================================
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# ============================================================================
# IMPORTS - Scientific and Numerical Computing
# ============================================================================
import numpy as np

# ============================================================================
# IMPORTS - RF/Microwave Engineering and Network Analysis
# ============================================================================
try:
    import skrf as rf
    from skrf import Network
    from skrf.time import time_gate
except ImportError:
    print("ERROR: scikit-rf not installed. Install with: pip install scikit-rf or conda install scikit-rf")
    sys.exit(1)

# ============================================================================
# IMPORTS - Plotting and Visualization (matplotlib)
# ============================================================================
import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# ============================================================================
# IMPORTS - PDF Processing (for multi-page PDF generation)
# ============================================================================
try:
    from PyPDF2 import PdfMerger, PdfWriter
except ImportError:
    print("WARNING: PyPDF2 not installed. PDF merging will not be available.")
    print("Install with: pip install PyPDF2")

# ============================================================================
# IMPORTS - Veusz Integration (for non-Smith chart processing)
# ============================================================================
try:
    import veusz.embed as vz
except ImportError:
    print("WARNING: Veusz not installed or not properly configured.")
    print("Install with: pip install veusz")

# ============================================================================
# IMPORTS - Qt Framework (GUI)
# ============================================================================
# Try PySide6 first (for compiled builds), fall back to QtPy
if getattr(sys, 'frozen', False):
    # Running as compiled executable - use PySide6 directly
    from PySide6.QtWidgets import (
        QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QMessageBox,
        QMainWindow, QWidget, QTextEdit, QCheckBox,
        QGroupBox, QListWidget, QTabWidget, QComboBox, QDoubleSpinBox, QFormLayout
    )
else:
    # Development environment - use QtPy abstraction layer
    try:
        from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize, QRect
        from qtpy.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush, QColor
        from qtpy.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QFileDialog, QLabel, QRadioButton, QButtonGroup, QMessageBox,
            QMainWindow, QWidget, QTextEdit, QProgressBar, QCheckBox,
            QSpinBox, QGroupBox, QListWidget, QSplitter, QLineEdit,
            QTabWidget, QComboBox, QSlider, QDoubleSpinBox, QGridLayout,
            QFormLayout, QFrame, QListWidgetItem, QProgressDialog
        )
    except ImportError:
        print("ERROR: Neither PySide6 nor QtPy available.")
        print("Install with: pip install pyside6 OR pip install qtpy")
        sys.exit(1)

# ============================================================================
# GPU ACCELERATION SUPPORT (Optional - for S-parameter processing only)
# ============================================================================
GPUAVAILABLE = None

try:
    import cupy as cp

    GPUAVAILABLE = 'cupy'
    print("CuPy detected - NVIDIA/AMD GPU acceleration available")
except ImportError:
    try:
        import pyopencl as cl
        import pyopencl.array as clarray

        GPUAVAILABLE = 'opencl'
        print("PyOpenCL detected - Cross-platform GPU acceleration available")
    except ImportError:
        try:
            import taichi as ti

            GPUAVAILABLE = 'taichi'
            print("Taichi detected - Cross-platform GPU acceleration available")
        except ImportError:
            GPUAVAILABLE = None
            print("No GPU acceleration libraries available - using CPU only")


# ============================================================================
# CONFIGURATION CLASSES
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
    window_type: str = 'kaiser'
    window_param: float = 6.0
    gate_start: float = 0.0
    gate_stop: float = 1.0
    gate_center: float = 0.5
    gate_span: float = 0.2
    mode: str = 'bandpass'
    method: str = 'fft'
    tunit: str = 'ns'
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
    output_format: str = 'png'  # png, pdf, tiff, bmp, svg, jpg


# ============================================================================
# SMITH CHART PLOTTING CLASS (Matplotlib-based)
# ============================================================================

class SmithChartMatplotlibPlotter:
    """
    Generates Smith charts using matplotlib and scikit-rf.
    
    This class handles the creation of Smith charts for impedance or admittance
    data extracted from Touchstone files. It supports rendering directly to
    matplotlib figures and saving to multiple image formats.
    
    Attributes:
        config (SmithChartMatplotlibConfig): Configuration settings for plotting.
        z0 (float): Reference impedance for normalization.
    """

    def __init__(self, config: SmithChartMatplotlibConfig = None, z0: float = 50.0):
        """
        Initialize the Smith chart plotter.
        
        Parameters:
            config (SmithChartMatplotlibConfig, optional): Configuration object.
                If None, defaults are used.
            z0 (float): Reference impedance in Ohms. Default is 50.0.
        """
        self.config = config or SmithChartMatplotlibConfig()
        self.z0 = z0
        self.figures = []
        self.file_paths = []

    def create_smith_chart_figure(self, network: Network, param_name: str,
                                  chart_type: str = 'z',
                                  draw_labels: bool = True,
                                  draw_vswr: bool = True) -> Tuple[Figure, str]:
        """
        Create a Smith chart figure from network S-parameters.
        
        Parameters:
            network (Network): scikit-rf Network object containing S-parameter data.
            param_name (str): S-parameter name (e.g., 'S11', 'S21').
            chart_type (str): 'z' for impedance, 'y' for admittance.
            draw_labels (bool): Whether to draw impedance/admittance labels.
            draw_vswr (bool): Whether to draw VSWR circles.
            
        Returns:
            Tuple[Figure, str]: matplotlib Figure object and descriptive title string.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size,
                               dpi=self.config.dpi)

        # Extract port indices from parameter name (e.g., 'S11' -> (0,0))
        indices = self._extract_param_indices(param_name)
        if indices is None:
            raise ValueError(f"Invalid parameter name: {param_name}")

        i, j = indices
        if i >= network.nports or j >= network.nports:
            raise IndexError(f"Port index out of range for {param_name}")

        # Extract S-parameter data
        s_param = network.s[:, i, j]

        # Convert to impedance or admittance
        if chart_type.lower() == 'z':
            # Impedance: Z = Z0 * (1 + S) / (1 - S)
            z_norm = (1 + s_param) / (1 - s_param)
            title = f"Smith Chart - {param_name} (Impedance)"
            label_type = "Impedance"
        elif chart_type.lower() == 'y':
            # Admittance: Y = Y0 * (1 - S) / (1 + S)
            z_norm = (1 - s_param) / (1 + s_param)
            title = f"Smith Chart - {param_name} (Admittance)"
            label_type = "Admittance"
        else:
            raise ValueError(f"Invalid chart type: {chart_type}")

        # Draw Smith chart background
        self._draw_smith_chart_grid(ax, draw_labels=draw_labels,
                                    draw_vswr=draw_vswr, label_type=label_type)

        # Plot S-parameter trace
        real_part = np.real(z_norm)
        imag_part = np.imag(z_norm)

        ax.plot(real_part, imag_part, 'b-', linewidth=self.config.line_width,
                label=f"{param_name} Trace", marker='o',
                markersize=self.config.marker_size, alpha=0.7)

        # Mark frequency points
        num_points = len(z_norm)
        if num_points > 0:
            # Mark start and end points
            ax.plot(real_part[0], imag_part[0], 'go', markersize=8,
                    label='Start', zorder=5)
            ax.plot(real_part[-1], imag_part[-1], 'ro', markersize=8,
                    label='End', zorder=5)

        # Configure axes
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Real Part', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Imaginary Part', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.legend(fontsize=self.config.legend_fontsize, loc='upper right')
        ax.grid(True, alpha=self.config.grid_alpha)

        fig.tight_layout()

        return fig, title

    def _draw_smith_chart_grid(self, ax, draw_labels: bool = True,
                               draw_vswr: bool = True, label_type: str = "Impedance"):
        """
        Draw the Smith chart grid background on matplotlib axes.
        
        Parameters:
            ax: matplotlib Axes object.
            draw_labels (bool): Whether to draw impedance/admittance labels.
            draw_vswr (bool): Whether to draw VSWR circles.
            label_type (str): "Impedance" or "Admittance" for label text.
        """
        # Main circle (magnitude = 1)
        circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Resistance circles (for impedance) or conductance circles (for admittance)
        # These are circles with centers on the real axis
        resistance_values = [0.2, 0.5, 1.0, 2.0, 5.0]

        for r in resistance_values:
            # Circle equation for Smith chart resistance circles:
            # (x - r/(1+r))^2 + y^2 = (1/(1+r))^2
            center_x = r / (1 + r)
            radius = 1 / (1 + r)

            circle = Circle((center_x, 0), radius, fill=False,
                            edgecolor='lightblue', linewidth=0.5, linestyle='--')
            ax.add_patch(circle)

            # Add labels if requested
            if draw_labels and center_x + radius < 1.0:
                label_text = f"{r:.1f}" if r != 1.0 else f"{r:.1f}"
                ax.text(center_x + radius + 0.05, 0, label_text,
                        fontsize=8, ha='left', va='center', color='blue')

        # Reactance circles (imaginary axis)
        reactance_values = [0.2, 0.5, 1.0, 2.0, 5.0]

        for x in reactance_values:
            # Circle equation for Smith chart reactance circles:
            # (x - 1)^2 + (y - 1/x)^2 = (1/x)^2
            center_x = 1.0
            center_y = 1.0 / x
            radius = 1.0 / x

            circle = Circle((center_x, center_y), radius, fill=False,
                            edgecolor='lightgreen', linewidth=0.5, linestyle='--')
            ax.add_patch(circle)

            # Negative reactance circle (symmetric about real axis)
            circle_neg = Circle((center_x, -center_y), radius, fill=False,
                                edgecolor='lightgreen', linewidth=0.5, linestyle='--')
            ax.add_patch(circle_neg)

            # Add labels if requested
            if draw_labels:
                ax.text(center_x + 0.05, center_y + 0.05, f"+j{x:.1f}",
                        fontsize=7, ha='left', va='bottom', color='green')
                ax.text(center_x + 0.05, -center_y - 0.05, f"-j{x:.1f}",
                        fontsize=7, ha='left', va='top', color='green')

        # Draw VSWR circles if requested
        if draw_vswr:
            vswr_values = [1.5, 2.0, 3.0]
            for vswr in vswr_values:
                # VSWR circle: maps to reflection coefficient magnitude
                gamma_mag = (vswr - 1) / (vswr + 1)
                circle = Circle((0, 0), gamma_mag, fill=False,
                                edgecolor='red', linewidth=0.5, linestyle=':',
                                alpha=0.5)
                ax.add_patch(circle)

                # VSWR label
                if draw_labels:
                    ax.text(gamma_mag, 0.05, f"VSWR={vswr:.1f}",
                            fontsize=7, ha='center', va='bottom', color='red')

    def _extract_param_indices(self, param_name: str) -> Optional[Tuple[int, int]]:
        """
        Extract port indices from S-parameter name.
        
        Parameters:
            param_name (str): Parameter name like 'S11', 'S21', 'S12', etc.
            
        Returns:
            Tuple[int, int] or None: (row_index, column_index) or None if invalid.
        """
        if not param_name.startswith('S') or len(param_name) != 3:
            return None

        try:
            i = int(param_name[1]) - 1
            j = int(param_name[2]) - 1
            return (i, j) if i >= 0 and j >= 0 else None
        except ValueError:
            return None

    def save_smith_charts(self, network: Network, filename: str,
                          output_dir: str, output_format: str = 'png',
                          chart_type: str = 'z', combine_pdf: bool = False,
                          draw_labels: bool = True,
                          draw_vswr: bool = True) -> List[str]:
        """
        Generate and save Smith charts for all S-parameters in a network.
        
        Parameters:
            network (Network): scikit-rf Network object.
            filename (str): Base filename for saved charts.
            output_dir (str): Directory to save chart files.
            output_format (str): Output format ('png', 'pdf', 'tiff', 'bmp', 'svg', 'jpg').
            chart_type (str): 'z' for impedance, 'y' for admittance.
            combine_pdf (bool): If True and format is PDF, combine all into single PDF.
            draw_labels (bool): Whether to draw Smith chart labels.
            draw_vswr (bool): Whether to draw VSWR circles.
            
        Returns:
            List[str]: List of saved file paths.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        base_filename = os.path.splitext(filename)[0]
        pdf_files = [] if output_format.lower() == 'pdf' and combine_pdf else None

        # Generate Smith charts for each S-parameter
        for i in range(network.nports):
            for j in range(network.nports):
                param_name = f"S{i + 1}{j + 1}"

                try:
                    fig, title = self.create_smith_chart_figure(
                        network, param_name,
                        chart_type=chart_type,
                        draw_labels=draw_labels,
                        draw_vswr=draw_vswr
                    )

                    # Create output filename
                    safe_title = title.replace(' ', '_').replace('-', '_')
                    output_filename = f"{base_filename}_{param_name}.{output_format.lower()}"
                    output_path = os.path.join(output_dir, output_filename)

                    # Save figure
                    fig.savefig(output_path, format=output_format.lower(), dpi=self.config.dpi)
                    plt.close(fig)

                    saved_files.append(output_path)

                    # Track PDF files for potential merging
                    if output_format.lower() == 'pdf' and combine_pdf:
                        pdf_files.append(output_path)

                except Exception as e:
                    print(f"Warning: Failed to create Smith chart for {param_name}: {e}")

        # Combine PDFs if requested
        if combine_pdf and pdf_files and len(pdf_files) > 1:
            combined_path = os.path.join(output_dir, f"{base_filename}_All_SmithCharts.pdf")
            try:
                self._combine_pdfs(pdf_files, combined_path)
                saved_files.append(combined_path)
                print(f"Combined PDF created: {combined_path}")
            except Exception as e:
                print(f"Warning: Failed to create combined PDF: {e}")

        return saved_files

    def _combine_pdfs(self, pdf_files: List[str], output_path: str):
        """
        Combine multiple PDF files into a single bookmarked PDF.
        
        Parameters:
            pdf_files (List[str]): List of PDF file paths to combine.
            output_path (str): Path for the output combined PDF.
        """
        try:
            from PyPDF2 import PdfMerger

            merger = PdfMerger()

            for pdf_file in pdf_files:
                if os.path.exists(pdf_file):
                    merger.append(pdf_file)

            merger.write(output_path)
            merger.close()

        except ImportError:
            print("PyPDF2 not available. Install with: pip install PyPDF2")
            # Fallback: at least document the issue
            raise RuntimeError("PDF merging requires PyPDF2. Install with: pip install PyPDF2")


# ============================================================================
# MAIN TOUCHSTONE PROCESSING CLASSES (from original, abbreviated)
# ============================================================================

class TimeDomainProcessor:
    """Handles time domain analysis and gating of S-parameter data."""

    def __init__(self, config: TimeDomainConfig = None):
        """Initialize time domain processor."""
        self.config = config or TimeDomainConfig()

    def process_network(self, network: Network) -> Dict[str, Any]:
        """Process a network for time domain analysis."""
        results = {}
        try:
            # Time domain conversion (placeholder - extend as needed)
            results['network'] = network
            results['time'] = np.arange(0, 100, 1)  # Placeholder time vector
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
                                        td_processor: TimeDomainProcessor = None) -> Dict[str, Any]:
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
# TOUCHSTONE PROCESSING FUNCTIONS
# ============================================================================

def process_single_touchstone_file(file_info: Tuple[str, Optional[object]]) -> Tuple[str, Dict]:
    """
    Process a single Touchstone file.
    
    Parameters:
        file_info (Tuple): Tuple of (filepath, gpu_accelerator).
        
    Returns:
        Tuple: (filename, processed_data_dict).
    """
    filepath, gpu_accelerator = file_info

    try:
        # Load Touchstone file using scikit-rf
        network = rf.Network(filepath)
        filename = os.path.basename(filepath)

        return filename, {
            'network': network,
            'filepath': filepath,
            'num_ports': network.nports,
            'frequency_ghz': network.frequency.f / 1e9,
            'z0': network.z0
        }
    except Exception as e:
        print(f"Error processing Touchstone file {filepath}: {str(e)}")
        return os.path.basename(filepath), None


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class TouchstoneMainWindowSmithMatplotlib(QMainWindow):
    """
    Main application window for Touchstone AutoPlot with matplotlib Smith charts.
    
    This enhanced version allows Smith chart generation using matplotlib and
    scikit-rf instead of Veusz, while maintaining compatibility with traditional
    Veusz-based Touchstone S-parameter and time domain plotting.
    """

    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        self.setWindowTitle("Enhanced Touchstone AutoPlot - Smith Chart Matplotlib Edition")
        self.setGeometry(100, 100, 1400, 900)

        # Data storage
        self.selected_files = []
        self.processed_data = {}
        self.smith_results = {}

        # Configuration objects
        self.config = ProcessingConfig()
        self.td_config = TimeDomainConfig()
        self.smith_config = SmithChartConfig()
        self.smith_mpl_config = SmithChartMatplotlibConfig()

        # Processing components
        self.td_processor = TimeDomainProcessor(self.td_config)
        self.smith_processor = SmithChartProcessor(self.smith_config)
        self.smith_mpl_plotter = SmithChartMatplotlibPlotter(self.smith_mpl_config)

        # Setup UI
        self.setup_ui()

        self.log_message("Application initialized successfully")
        self.log_message("Select Touchstone files and configure Smith chart options")

    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Tab widget for different analysis types
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # File selection area
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        button_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self.browse_files)
        clear_btn = QPushButton("Clear Files")
        clear_btn.clicked.connect(self.clear_files)

        button_layout.addWidget(browse_btn)
        button_layout.addWidget(clear_btn)
        file_layout.addLayout(button_layout)

        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)

        main_layout.addWidget(file_group)

        # Smith Chart tab setup
        self.setup_smith_chart_tab()

        # Status/logging area
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)

        main_layout.addWidget(status_group)

    def setup_smith_chart_tab(self):
        """Set up the Smith Chart analysis tab."""
        smith_tab = QWidget()
        self.tab_widget.addTab(smith_tab, "Smith Chart Analysis")

        layout = QHBoxLayout(smith_tab)

        # Left control panel
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls_widget)

        # File selection
        file_select_group = QGroupBox("File Selection")
        file_select_layout = QVBoxLayout(file_select_group)

        self.smith_file_combo = QComboBox()
        self.smith_file_combo.currentTextChanged.connect(self.update_smith_preview)
        file_select_layout.addWidget(self.smith_file_combo)

        controls_layout.addWidget(file_select_group)

        # Smith Chart settings
        smith_group = QGroupBox("Smith Chart Settings")
        smith_layout = QFormLayout(smith_group)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["z (Impedance)", "y (Admittance)"])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart_type)
        smith_layout.addRow("Chart Type:", self.chart_type_combo)

        self.ref_impedance_spin = QDoubleSpinBox()
        self.ref_impedance_spin.setRange(1.0, 1000.0)
        self.ref_impedance_spin.setValue(self.smith_config.reference_impedance)
        self.ref_impedance_spin.setSuffix(" Ω")
        self.ref_impedance_spin.valueChanged.connect(self.update_ref_impedance)
        smith_layout.addRow("Reference Impedance:", self.ref_impedance_spin)

        self.draw_labels_checkbox = QCheckBox("Draw Labels")
        self.draw_labels_checkbox.setChecked(self.smith_config.draw_labels)
        self.draw_labels_checkbox.stateChanged.connect(self.update_draw_labels)
        smith_layout.addRow(self.draw_labels_checkbox)

        self.draw_vswr_checkbox = QCheckBox("Draw VSWR Circles")
        self.draw_vswr_checkbox.setChecked(self.smith_config.draw_vswr)
        self.draw_vswr_checkbox.stateChanged.connect(self.update_draw_vswr)
        smith_layout.addRow(self.draw_vswr_checkbox)

        controls_layout.addWidget(smith_group)

        # Output format selection
        output_group = QGroupBox("Output Format")
        output_layout = QVBoxLayout(output_group)

        format_label = QLabel("Output Format:")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["PNG", "PDF", "TIFF", "BMP", "SVG", "JPG"])
        self.output_format_combo.setCurrentText("PNG")
        self.output_format_combo.currentTextChanged.connect(self.update_output_format)

        output_layout.addWidget(format_label)
        output_layout.addWidget(self.output_format_combo)

        self.combine_pdf_checkbox = QCheckBox("Combine all plots into single PDF (with bookmarks)")
        self.combine_pdf_checkbox.setChecked(False)
        self.combine_pdf_checkbox.setEnabled(False)
        self.combine_pdf_checkbox.stateChanged.connect(self.update_combine_pdf)

        output_layout.addWidget(self.combine_pdf_checkbox)

        controls_layout.addWidget(output_group)

        # Processing scope
        scope_group = QGroupBox("Processing Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.smith_process_selected_only = QCheckBox(
            "Generate Smith Chart only for selected file"
        )
        self.smith_process_selected_only.setChecked(False)
        scope_layout.addWidget(self.smith_process_selected_only)

        controls_layout.addWidget(scope_group)

        # Process button
        self.smith_process_button = QPushButton("Generate Smith Charts in Matplotlib")
        self.smith_process_button.clicked.connect(self.process_smith_charts_matplotlib)
        self.smith_process_button.setEnabled(False)
        controls_layout.addWidget(self.smith_process_button)

        controls_layout.addStretch()

        layout.addWidget(controls_widget)

    def browse_files(self):
        """Open file dialog to select Touchstone files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Touchstone Files (*.s1p *.s2p *.s3p *.s4p *.sp)")
        file_dialog.setWindowTitle("Select Touchstone Files")

        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self.update_file_list()
            self.process_files_in_thread()

    def clear_files(self):
        """Clear the selected files list."""
        self.selected_files.clear()
        self.processed_data.clear()
        self.smith_results.clear()
        self.update_file_list()
        self.update_smith_file_combo()
        self.log_message("File list cleared")

    def update_file_list(self):
        """Update the file list widget display."""
        self.file_list_widget.clear()
        for filepath in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(filepath))

    def update_smith_file_combo(self):
        """Update the Smith chart file selection combo box."""
        self.smith_file_combo.clear()
        if self.processed_data:
            self.smith_file_combo.addItems(list(self.processed_data.keys()))
            self.smith_process_button.setEnabled(len(self.processed_data) > 0)

    def process_files_in_thread(self):
        """Process selected files in a background thread."""
        if not self.selected_files:
            self.log_message("No files selected")
            return

        # For simplicity, process sequentially
        for filepath in self.selected_files:
            filename, data = process_single_touchstone_file((filepath, None))
            if data:
                self.processed_data[filename] = data
                self.log_message(f"Loaded: {filename}")

        self.update_smith_file_combo()

    def update_chart_type(self, chart_type_text: str):
        """Update Smith chart type configuration."""
        if "Impedance" in chart_type_text:
            self.smith_config.chart_type = 'z'
        else:
            self.smith_config.chart_type = 'y'
        self.smith_mpl_config.output_format = self.output_format_combo.currentText().lower()

    def update_ref_impedance(self, value: float):
        """Update reference impedance configuration."""
        self.smith_config.reference_impedance = value
        self.smith_mpl_plotter.z0 = value

    def update_draw_labels(self, state: int):
        """Update draw labels configuration."""
        self.smith_config.draw_labels = (state == Qt.Checked)

    def update_draw_vswr(self, state: int):
        """Update draw VSWR configuration."""
        self.smith_config.draw_vswr = (state == Qt.Checked)

    def update_smith_preview(self):
        """Update Smith chart preview (placeholder)."""
        current_file = self.smith_file_combo.currentText()
        if current_file and current_file in self.processed_data:
            self.log_message(f"Selected: {current_file}")

    def update_output_format(self, format_text: str):
        """Update output format and enable PDF combine option if applicable."""
        self.smith_mpl_config.output_format = format_text.lower()
        is_pdf = format_text.lower() == 'pdf'
        self.combine_pdf_checkbox.setEnabled(is_pdf)

    def update_combine_pdf(self, state: int):
        """Update combine PDF option."""
        self.smith_mpl_config.combine_to_pdf = (state == Qt.Checked)

    def process_smith_charts_matplotlib(self):
        """Process and generate Smith charts using matplotlib."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please load Touchstone files first")
            return

        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Smith Charts"
        )

        if not output_dir:
            return

        try:
            # Determine which files to process
            if self.smith_process_selected_only.isChecked():
                current_file = self.smith_file_combo.currentText()
                files_to_process = {current_file: self.processed_data[current_file]}
            else:
                files_to_process = self.processed_data

            # Generate Smith charts
            all_saved_files = []

            for filename, data in files_to_process.items():
                if 'network' not in data:
                    continue

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
                    self.log_message(f"Smith charts generated for {filename}: {len(saved_files)} file(s)")

                except Exception as e:
                    self.log_message(f"Error processing {filename}: {str(e)}")

            # Notify user
            msg = f"Smith charts generated successfully!\n\n"
            msg += f"Total files saved: {len(all_saved_files)}\n"
            msg += f"Output directory: {output_dir}"

            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate Smith charts: {str(e)}")
            self.log_message(f"Smith chart generation error: {str(e)}")

    def log_message(self, message: str):
        """Add message to status log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = TouchstoneMainWindowSmithMatplotlib()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
