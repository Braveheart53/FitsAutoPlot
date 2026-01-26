"""
CSV/TSV AutoPlot - Automatic data visualization from delimited text files.

This script provides a complete GUI application for loading CSV, TSV, and other
delimited text files, configuring custom delimiters, and generating publication-
quality plots using Veusz. It mirrors the functionality and interface design of
the original Touchstone_AutoPlot.py but adapts it for arbitrary delimited data files.

Key Features:
    - Support for arbitrary delimiter characters/strings (comma, tab, semicolon, etc.)
    - Automatic data type detection (numeric vs. categorical)
    - Interactive preview of loaded data
    - Multi-column selection for X and Y axes
    - Automatic plot generation with Veusz
    - Batch processing of multiple files
    - Configurable plot styles and formatting
    - Comprehensive error handling and logging

Typical Workflow:
    1. Launch the application
    2. Browse and select CSV/TSV files
    3. Specify delimiter character(s)
    4. Preview data to verify correct parsing
    5. Select columns for plotting
    6. Configure plot options (title, labels, legend, etc.)
    7. Generate plots in Veusz or save as images

Author: Based on William W. Wallace's Touchstone_AutoPlot.py framework
Last Updated: 2026-01-26
Python Version: 3.8+

Dependencies:
    - PyQt5/PySide6 (GUI framework)
    - pandas (data loading and manipulation)
    - numpy (numerical computing)
    - matplotlib (preview plots)
    - veusz (final plot generation)
    - scipy (optional, for advanced analysis)

Installation:
    pip install pandas numpy matplotlib veusz pyside6

Usage:
    python CSV_TSV_AutoPlot.py
"""

# ============================================================================
# IMPORTS - Standard Library
# ============================================================================
import os
import sys
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ascompleted
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import datetime
import traceback

# ============================================================================
# IMPORTS - Data Processing and Numerical Computing
# ============================================================================
import numpy as np
import pandas as pd

try:
    import scipy.stats as stats
except ImportError:
    stats = None

# ============================================================================
# IMPORTS - Visualization (Matplotlib for preview)
# ============================================================================
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ============================================================================
# IMPORTS - Veusz Integration (for final plot generation)
# ============================================================================
try:
    import veusz.embed as vz
except ImportError:
    print("WARNING: Veusz not installed. Plot generation will not work.")
    print("Install with: pip install veusz")
    vz = None

# ============================================================================
# IMPORTS - Qt Framework (GUI)
# ============================================================================
if getattr(sys, 'frozen', False):
    # Running as compiled executable - use PySide6 directly
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize, QRect
    from PySide6.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush, QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QMessageBox, QTextEdit,
        QProgressBar, QCheckBox, QSpinBox, QGroupBox, QListWidget,
        QLineEdit, QTabWidget, QComboBox, QDoubleSpinBox, QGridLayout,
        QFormLayout, QFrame, QListWidgetItem, QTableWidget, QTableWidgetItem,
        QHeaderView, QDialog, QProgressDialog, QDialogButtonBox, QScrollArea,
        QSplitter
    )
else:
    # Development environment - use QtPy abstraction layer
    try:
        from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize, QRect
        from qtpy.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush, QColor
        from qtpy.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QPushButton, QFileDialog, QLabel, QMessageBox, QTextEdit,
            QProgressBar, QCheckBox, QSpinBox, QGroupBox, QListWidget,
            QLineEdit, QTabWidget, QComboBox, QDoubleSpinBox, QGridLayout,
            QFormLayout, QFrame, QListWidgetItem, QTableWidget, QTableWidgetItem,
            QHeaderView, QDialog, QProgressDialog, QDialogButtonBox, QScrollArea,
            QSplitter
        )
    except ImportError:
        print("ERROR: Neither PySide6 nor QtPy available.")
        print("Install with: pip install pyside6 OR pip install qtpy")
        sys.exit(1)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class CSVProcessingConfig:
    """Configuration class for CSV/TSV processing settings."""
    enable_multiprocessing: bool = True
    num_processes: int = multiprocessing.cpu_count()
    max_workers: int = multiprocessing.cpu_count()
    chunk_size: int = 1000
    encoding: str = 'utf-8'
    skip_rows: int = 0
    skip_footer: int = 0
    skip_empty_lines: bool = True


@dataclass
class PlotConfig:
    """Configuration class for plot formatting."""
    title: str = "Data Plot"
    x_label: str = "X Axis"
    y_label: str = "Y Axis"
    x_scale: str = "linear"  # 'linear' or 'log'
    y_scale: str = "linear"  # 'linear' or 'log'
    show_grid: bool = True
    show_legend: bool = True
    legend_position: str = "best"
    line_style: str = "-"
    line_width: float = 2.0
    marker_style: str = "o"
    marker_size: float = 6.0


# ============================================================================
# CSV/TSV LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def detect_delimiter(file_path: str, sample_size: int = 5) -> str:
    """
    Attempt to auto-detect the delimiter in a delimited text file.
    
    This function reads the first few lines of the file and analyzes
    common delimiters (comma, tab, semicolon, pipe) to determine which
    is most likely being used.
    
    Parameters:
        file_path (str): Path to the delimited text file.
        sample_size (int): Number of lines to sample for detection.
        
    Returns:
        str: Detected delimiter character (most commonly ',', '\t', ';', or '|').
    """
    delimiters = [',', '\t', ';', '|']
    delimiter_counts = {d: 0 for d in delimiters}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(sample_size):
                line = f.readline()
                if not line:
                    break
                
                # Skip empty lines and comments
                if not line.strip() or line.startswith('#'):
                    continue
                
                # Count occurrences of each delimiter
                for delimiter in delimiters:
                    delimiter_counts[delimiter] += line.count(delimiter)
        
        # Return the delimiter with the highest count
        if max(delimiter_counts.values()) > 0:
            return max(delimiter_counts, key=delimiter_counts.get)
        else:
            return ','  # Default to comma
    
    except Exception as e:
        print(f"Error detecting delimiter: {e}")
        return ','  # Default to comma


def load_csv_file(file_path: str, delimiter: str = ',',
                  config: CSVProcessingConfig = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV/TSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV/TSV file.
        delimiter (str): Delimiter character or string.
        config (CSVProcessingConfig, optional): Processing configuration.
        
    Returns:
        Tuple[DataFrame or None, str]: Loaded DataFrame and status message.
    """
    if config is None:
        config = CSVProcessingConfig()
    
    try:
        # Load the CSV file with specified delimiter
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=config.encoding,
            skiprows=config.skip_rows,
            skip_blank_lines=config.skip_empty_lines,
            engine='python'  # Use python engine for better delimiter support
        )
        
        status_msg = f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns"
        return df, status_msg
    
    except Exception as e:
        error_msg = f"Error loading {file_path}: {str(e)}"
        return None, error_msg


def load_multiple_csv_files(file_paths: List[str], delimiter: str = ',',
                            config: CSVProcessingConfig = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV/TSV files.
    
    Parameters:
        file_paths (List[str]): List of file paths.
        delimiter (str): Delimiter character or string.
        config (CSVProcessingConfig, optional): Processing configuration.
        
    Returns:
        Dict[str, DataFrame]: Dictionary mapping filenames to DataFrames.
    """
    results = {}
    
    for file_path in file_paths:
        df, status_msg = load_csv_file(file_path, delimiter, config)
        if df is not None:
            filename = os.path.basename(file_path)
            results[filename] = df
    
    return results


def infer_numeric_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Infer which columns in a DataFrame are numeric.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        Dict[str, bool]: Dictionary mapping column names to numeric status.
    """
    numeric_status = {}
    
    for col in df.columns:
        try:
            # Try to convert to numeric
            pd.to_numeric(df[col], errors='coerce')
            non_null_count = df[col].notna().sum()
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            
            # Consider numeric if more than 90% of values are numeric
            is_numeric = (numeric_count / non_null_count) > 0.9 if non_null_count > 0 else False
            numeric_status[col] = is_numeric
        except Exception:
            numeric_status[col] = False
    
    return numeric_status


# ============================================================================
# VEUSZ PLOTTING CLASS
# ============================================================================

class VeuszPlotter:
    """
    Handles plot generation in Veusz for CSV/TSV data.
    
    This class wraps the Veusz embedding API to create publication-quality
    plots from pandas DataFrames loaded from CSV/TSV files.
    """
    
    def __init__(self, plot_title: str = "CSV Data Plot",
                 dataset_name: str = "CSVDataset"):
        """
        Initialize Veusz plotter.
        
        Parameters:
            plot_title (str): Title for the plot window.
            dataset_name (str): Base name for datasets in Veusz.
        """
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.doc = None
        
        try:
            if vz is not None:
                self.doc = vz.Embedded(plot_title, hidden=False)
                self.doc.EnableToolbar()
        except Exception as e:
            print(f"Warning: Could not initialize Veusz: {e}")
    
    def create_xy_plot(self, df: pd.DataFrame, x_column: str, y_columns: List[str],
                      plot_config: PlotConfig = None) -> bool:
        """
        Create an XY plot in Veusz from DataFrame columns.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            x_column (str): Name of column to use for X axis.
            y_columns (List[str]): Names of columns to use for Y axes.
            plot_config (PlotConfig, optional): Plot configuration.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.doc is None:
            return False
        
        if plot_config is None:
            plot_config = PlotConfig()
        
        try:
            # Create page and grid
            page = self.doc.Root.Add('page', name='XYPlot')
            grid = page.Add('grid', columns=1, rows=1)
            
            # Create graph
            graph = grid.Add('graph', name='MainPlot')
            
            # Add data to Veusz
            # X data
            x_data = df[x_column].dropna().values.tolist()
            self.doc.SetData(f'{self.dataset_name}_X', x_data)
            
            # Y data for each column
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for idx, y_col in enumerate(y_columns):
                y_data = df[y_col].dropna().values.tolist()
                self.doc.SetData(f'{self.dataset_name}_Y{idx}', y_data)
                
                # Create XY plot
                xy_plot = graph.Add('xy', name=f'Plot_{y_col}',
                                   xData=f'{self.dataset_name}_X',
                                   yData=f'{self.dataset_name}_Y{idx}')
                
                # Configure appearance
                with self._wrap_widget(xy_plot) as plot:
                    plot.PlotLine.color.val = colors[idx % len(colors)]
                    plot.PlotLine.style.val = plot_config.line_style
                    plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                    plot.marker.val = plot_config.marker_style
                    plot.markerSize.val = f'{plot_config.marker_size}pt'
            
            # Configure graph
            with self._wrap_widget(graph) as g:
                g.title.val = plot_config.title
                g.xlabel.val = plot_config.x_label
                g.ylabel.val = plot_config.y_label
                g.xLog.val = (plot_config.x_scale == 'log')
                g.yLog.val = (plot_config.y_scale == 'log')
                g.showLegend.val = plot_config.show_legend
                if plot_config.show_legend:
                    g.legend.pos.val = plot_config.legend_position
            
            return True
        
        except Exception as e:
            print(f"Error creating XY plot: {e}")
            traceback.print_exc()
            return False
    
    def _wrap_widget(self, widget):
        """Context manager for widget configuration in Veusz."""
        class WidgetWrapper:
            def __init__(self, w):
                self.widget = w
            
            def __enter__(self):
                return self.widget
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return WidgetWrapper(widget)
    
    def save_project(self, file_path: str) -> bool:
        """
        Save Veusz project to file.
        
        Parameters:
            file_path (str): Path where to save the Veusz project.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.doc is None:
            return False
        
        try:
            self.doc.Save(file_path)
            return True
        except Exception as e:
            print(f"Error saving Veusz project: {e}")
            return False
    
    def export_plot(self, file_path: str, format_type: str = 'png') -> bool:
        """
        Export plot to image file.
        
        Parameters:
            file_path (str): Output file path.
            format_type (str): Format ('png', 'pdf', 'svg', etc.).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.doc is None:
            return False
        
        try:
            # Find the first page
            pages = [c for c in self.doc.Root.children if c.typename == 'page']
            if pages:
                page = pages[0]
                # Export using Veusz's export function
                self.doc.Export(file_path, format=format_type, page=page.name)
            return True
        except Exception as e:
            print(f"Error exporting plot: {e}")
            return False
    
    def open_gui(self, file_path: str = None):
        """
        Open the Veusz GUI window.
        
        Parameters:
            file_path (str, optional): Optional file path to open.
        """
        if self.doc is None:
            return
        
        try:
            if sys.platform.startswith('win'):
                veusz_exe = os.path.join(sys.prefix, 'Scripts', 'veusz.exe')
            else:
                veusz_exe = os.path.join(sys.prefix, 'bin', 'veusz')
            
            if os.path.exists(veusz_exe):
                if file_path:
                    subprocess.Popen([veusz_exe, file_path])
                else:
                    subprocess.Popen([veusz_exe])
        except Exception as e:
            print(f"Error opening Veusz GUI: {e}")


# ============================================================================
# MATPLOTLIB PREVIEW CANVAS
# ============================================================================

class PreviewCanvas(FigureCanvas):
    """
    Matplotlib canvas for previewing data plots.
    
    This canvas displays a live preview of selected columns from the
    loaded CSV/TSV data to help users verify their selections before
    generating final plots in Veusz.
    """
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Initialize preview canvas.
        
        Parameters:
            parent (QWidget, optional): Parent widget.
            width (float): Figure width in inches.
            height (float): Figure height in inches.
            dpi (int): Figure DPI.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_data(self, df: pd.DataFrame, x_column: str, y_columns: List[str],
                 plot_config: PlotConfig = None):
        """
        Plot data on the canvas.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            x_column (str): Column name for X axis.
            y_columns (List[str]): Column names for Y axes.
            plot_config (PlotConfig, optional): Plot configuration.
        """
        if plot_config is None:
            plot_config = PlotConfig()
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Get X data
        x_data = pd.to_numeric(df[x_column], errors='coerce').dropna()
        
        # Plot each Y column
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx, y_col in enumerate(y_columns):
            y_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
            
            # Match indices where both X and Y are valid
            valid_idx = np.intersect1d(x_data.index, y_data.index)
            
            if len(valid_idx) > 0:
                ax.plot(x_data[valid_idx], y_data[valid_idx],
                       color=colors[idx % len(colors)],
                       marker=plot_config.marker_style,
                       markersize=plot_config.marker_size,
                       linestyle=plot_config.line_style,
                       linewidth=plot_config.line_width,
                       label=y_col, alpha=0.7)
        
        # Configure axes
        ax.set_xlabel(plot_config.x_label, fontsize=12)
        ax.set_ylabel(plot_config.y_label, fontsize=12)
        ax.set_title(plot_config.title, fontsize=14, fontweight='bold')
        ax.set_xscale(plot_config.x_scale)
        ax.set_yscale(plot_config.y_scale)
        ax.grid(plot_config.show_grid, alpha=0.3)
        
        if plot_config.show_legend:
            ax.legend(fontsize=10, loc=plot_config.legend_position)
        
        self.draw()
    
    def clear_plot(self):
        """Clear the canvas."""
        self.fig.clear()
        self.draw()


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class CSVAutoPlotMainWindow(QMainWindow):
    """
    Main application window for CSV/TSV AutoPlot.
    
    Provides a comprehensive GUI for loading delimited text files, configuring
    parsing options, previewing data, and generating plots in Veusz.
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("CSV/TSV AutoPlot - Data Visualization Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.selected_files = []
        self.loaded_data = {}  # Dict[filename, DataFrame]
        
        # Configuration
        self.csv_config = CSVProcessingConfig()
        self.plot_config = PlotConfig()
        
        # Setup UI
        self.setup_ui()
        
        self.log_message("CSV/TSV AutoPlot initialized successfully")
        self.log_message("Browse for CSV or TSV files to begin")
    
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # ====== FILE SELECTION AREA ======
        file_group = QGroupBox("File Selection and Delimiter Configuration")
        file_layout = QHBoxLayout(file_group)
        
        # Left side: File list
        left_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self.browse_files)
        clear_btn = QPushButton("Clear Files")
        clear_btn.clicked.connect(self.clear_files)
        
        button_layout.addWidget(browse_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addStretch()
        
        left_layout.addLayout(button_layout)
        
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(self.on_file_selection_changed)
        left_layout.addWidget(QLabel("Loaded Files:"))
        left_layout.addWidget(self.file_list_widget)
        
        # Right side: Delimiter configuration
        right_layout = QVBoxLayout()
        
        delimiter_group = QGroupBox("Delimiter Configuration")
        delimiter_form = QFormLayout(delimiter_group)
        
        auto_detect_btn = QPushButton("Auto-Detect Delimiter")
        auto_detect_btn.clicked.connect(self.auto_detect_delimiter)
        delimiter_form.addRow("Auto-Detection:", auto_detect_btn)
        
        self.delimiter_input = QLineEdit()
        self.delimiter_input.setText(',')
        self.delimiter_input.setMaximumWidth(100)
        self.delimiter_input.textChanged.connect(self.on_delimiter_changed)
        delimiter_form.addRow("Delimiter:", self.delimiter_input)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        for name, delim in [("Comma", ","), ("Tab", "\t"), ("Semicolon", ";"), ("Pipe", "|")]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, d=delim: self.set_delimiter(d))
            btn.setMaximumWidth(80)
            preset_layout.addWidget(btn)
        preset_layout.addStretch()
        delimiter_form.addRow("Presets:", QWidget())  # placeholder
        # Add preset layout to form properly
        form_layout = QVBoxLayout()
        form_layout.addWidget(delimiter_group)
        
        delimiter_group2 = QGroupBox("Delimiter Presets")
        preset_form = QVBoxLayout(delimiter_group2)
        preset_form.addLayout(preset_layout)
        form_layout.addWidget(delimiter_group2)
        
        right_layout.addLayout(form_layout)
        right_layout.addStretch()
        
        file_layout.addLayout(left_layout, 1)
        file_layout.addLayout(right_layout, 0)
        
        main_layout.addWidget(file_group)
        
        # ====== DATA PREVIEW AREA ======
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Table widget for data preview
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setMaximumHeight(200)
        preview_layout.addWidget(QLabel("First 10 rows:"))
        preview_layout.addWidget(self.data_preview_table)
        
        main_layout.addWidget(preview_group)
        
        # ====== COLUMN SELECTION FOR PLOTTING ======
        column_group = QGroupBox("Column Selection for Plotting")
        column_layout = QVBoxLayout(column_group)
        
        selection_form = QFormLayout()
        
        self.x_column_combo = QComboBox()
        self.x_column_combo.currentTextChanged.connect(self.on_plot_config_changed)
        selection_form.addRow("X-Axis Column:", self.x_column_combo)
        
        # Y-columns: use list widget for multiple selection
        y_label = QLabel("Y-Axis Columns (select multiple):")
        self.y_columns_list = QListWidget()
        self.y_columns_list.setSelectionMode(
            self.y_columns_list.SelectionMode.MultiSelection
        )
        self.y_columns_list.itemSelectionChanged.connect(self.on_plot_config_changed)
        self.y_columns_list.setMaximumHeight(120)
        
        column_layout.addLayout(selection_form)
        column_layout.addWidget(y_label)
        column_layout.addWidget(self.y_columns_list)
        
        main_layout.addWidget(column_group)
        
        # ====== PLOT CONFIGURATION ======
        plot_group = QGroupBox("Plot Configuration")
        plot_layout = QFormLayout(plot_group)
        
        self.title_input = QLineEdit()
        self.title_input.setText(self.plot_config.title)
        self.title_input.textChanged.connect(self.on_plot_config_changed)
        plot_layout.addRow("Plot Title:", self.title_input)
        
        self.x_label_input = QLineEdit()
        self.x_label_input.setText(self.plot_config.x_label)
        self.x_label_input.textChanged.connect(self.on_plot_config_changed)
        plot_layout.addRow("X-Axis Label:", self.x_label_input)
        
        self.y_label_input = QLineEdit()
        self.y_label_input.setText(self.plot_config.y_label)
        self.y_label_input.textChanged.connect(self.on_plot_config_changed)
        plot_layout.addRow("Y-Axis Label:", self.y_label_input)
        
        self.show_legend_checkbox = QCheckBox("Show Legend")
        self.show_legend_checkbox.setChecked(self.plot_config.show_legend)
        self.show_legend_checkbox.stateChanged.connect(self.on_plot_config_changed)
        plot_layout.addRow(self.show_legend_checkbox)
        
        main_layout.addWidget(plot_group)
        
        # ====== PREVIEW PLOT AREA ======
        self.preview_canvas = PreviewCanvas(width=8, height=4)
        main_layout.addWidget(QLabel("Plot Preview:"))
        main_layout.addWidget(self.preview_canvas)
        
        # ====== ACTION BUTTONS ======
        button_group = QGroupBox("")
        button_layout = QHBoxLayout(button_group)
        
        generate_btn = QPushButton("Generate Plots in Veusz")
        generate_btn.clicked.connect(self.generate_plots)
        
        export_btn = QPushButton("Export Preview as Image")
        export_btn.clicked.connect(self.export_preview)
        
        button_layout.addWidget(generate_btn)
        button_layout.addWidget(export_btn)
        button_layout.addStretch()
        
        main_layout.addWidget(button_group)
        
        # ====== STATUS/LOGGING AREA ======
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120)
        status_layout.addWidget(self.status_text)
        
        main_layout.addWidget(status_group)
    
    def browse_files(self):
        """Open file dialog to select CSV/TSV files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Delimited Files (*.csv *.tsv *.txt);;All Files (*)")
        file_dialog.setWindowTitle("Select CSV/TSV Files")
        
        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self.load_files()
    
    def load_files(self):
        """Load selected files into memory."""
        delimiter = self.delimiter_input.text() or ','
        
        for file_path in self.selected_files:
            if file_path not in [os.path.join(d, f) for d, f in
                                [os.path.split(self._get_file_path(name))
                                 for name in self.loaded_data.keys()]]:
                df, msg = load_csv_file(file_path, delimiter, self.csv_config)
                if df is not None:
                    filename = os.path.basename(file_path)
                    self.loaded_data[filename] = df
                    self.log_message(msg)
                else:
                    self.log_message(msg)
        
        self.update_file_list()
    
    def _get_file_path(self, filename: str) -> str:
        """Get full file path from filename."""
        for file_path in self.selected_files:
            if os.path.basename(file_path) == filename:
                return file_path
        return ""
    
    def update_file_list(self):
        """Update the file list widget."""
        self.file_list_widget.clear()
        for filename in self.loaded_data.keys():
            self.file_list_widget.addItem(filename)
    
    def clear_files(self):
        """Clear the selected files list."""
        self.selected_files.clear()
        self.loaded_data.clear()
        self.update_file_list()
        self.update_column_selectors()
        self.preview_canvas.clear_plot()
        self.log_message("File list cleared")
    
    def on_file_selection_changed(self):
        """Handle file selection change."""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            filename = current_item.text()
            self.update_column_selectors(filename)
            self.display_data_preview(filename)
            self.log_message(f"Selected: {filename}")
    
    def update_column_selectors(self, filename: str = None):
        """Update column selection dropdowns."""
        self.x_column_combo.clear()
        self.y_columns_list.clear()
        
        if filename and filename in self.loaded_data:
            df = self.loaded_data[filename]
            columns = df.columns.tolist()
            
            self.x_column_combo.addItems(columns)
            self.y_columns_list.addItems(columns)
            
            # Pre-select first column as X if available
            if len(columns) > 0:
                self.x_column_combo.setCurrentIndex(0)
    
    def display_data_preview(self, filename: str):
        """Display a preview of the loaded data in the table."""
        if filename not in self.loaded_data:
            return
        
        df = self.loaded_data[filename]
        
        # Show first 10 rows
        preview_df = df.head(10)
        
        self.data_preview_table.setRowCount(preview_df.shape[0])
        self.data_preview_table.setColumnCount(preview_df.shape[1])
        self.data_preview_table.setHorizontalHeaderLabels(preview_df.columns)
        
        for i, row in enumerate(preview_df.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_preview_table.setItem(i, j, item)
        
        # Resize columns to content
        self.data_preview_table.resizeColumnsToContents()
    
    def auto_detect_delimiter(self):
        """Auto-detect delimiter from currently selected file."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No File Selected", "Please select a file first")
            return
        
        filename = current_item.text()
        file_path = self._get_file_path(filename)
        
        if file_path:
            detected_delim = detect_delimiter(file_path)
            self.delimiter_input.setText(detected_delim)
            self.log_message(f"Auto-detected delimiter: {repr(detected_delim)}")
    
    def set_delimiter(self, delimiter: str):
        """Set the delimiter and reload files."""
        self.delimiter_input.setText(delimiter)
        self.loaded_data.clear()
        self.load_files()
        self.log_message(f"Delimiter set to: {repr(delimiter)}")
    
    def on_delimiter_changed(self):
        """Handle delimiter change."""
        # Reload files with new delimiter
        self.loaded_data.clear()
        self.load_files()
    
    def on_plot_config_changed(self):
        """Handle plot configuration change."""
        # Update preview
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            return
        
        filename = current_item.text()
        if filename not in self.loaded_data:
            return
        
        df = self.loaded_data[filename]
        
        # Get selected columns
        x_column = self.x_column_combo.currentText()
        y_columns = [item.text() for item in self.y_columns_list.selectedItems()]
        
        if not x_column or not y_columns:
            return
        
        # Update plot config
        self.plot_config.title = self.title_input.text()
        self.plot_config.x_label = self.x_label_input.text()
        self.plot_config.y_label = self.y_label_input.text()
        self.plot_config.show_legend = self.show_legend_checkbox.isChecked()
        
        # Plot preview
        try:
            self.preview_canvas.plot_data(df, x_column, y_columns, self.plot_config)
        except Exception as e:
            self.log_message(f"Error updating preview: {str(e)}")
    
    def generate_plots(self):
        """Generate plots in Veusz."""
        if not self.loaded_data:
            QMessageBox.warning(self, "No Data", "Please load CSV/TSV files first")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Veusz Projects"
        )
        
        if not output_dir:
            return
        
        try:
            current_item = self.file_list_widget.currentItem()
            if not current_item:
                QMessageBox.warning(self, "No File Selected", "Please select a file")
                return
            
            filename = current_item.text()
            df = self.loaded_data[filename]
            
            x_column = self.x_column_combo.currentText()
            y_columns = [item.text() for item in self.y_columns_list.selectedItems()]
            
            if not x_column or not y_columns:
                QMessageBox.warning(self, "No Columns Selected",
                                  "Please select X and Y columns")
                return
            
            # Create Veusz plot
            base_filename = os.path.splitext(filename)[0]
            plotter = VeuszPlotter(
                plot_title=self.plot_config.title,
                dataset_name=f"{base_filename}_Data"
            )
            
            # Create XY plot
            success = plotter.create_xy_plot(df, x_column, y_columns, self.plot_config)
            
            if success:
                # Save project
                project_path = os.path.join(output_dir, f"{base_filename}_plot.vszh5")
                plotter.save_project(project_path)
                
                self.log_message(f"Plot created successfully: {project_path}")
                
                # Ask to open in Veusz
                reply = QMessageBox.question(
                    self, "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    plotter.open_gui(project_path)
            else:
                QMessageBox.critical(self, "Error", "Failed to create plot in Veusz")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating plots: {str(e)}")
            self.log_message(f"Plot generation error: {str(e)}")
    
    def export_preview(self):
        """Export the preview plot as an image."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Plot Preview",
            "", "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.preview_canvas.fig.savefig(file_path, dpi=150)
                self.log_message(f"Preview exported: {file_path}")
                QMessageBox.information(self, "Success", "Plot preview exported successfully")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
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
    window = CSVAutoPlotMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
