"""
CSV/TSV AutoPlot - Automatic data visualization with Veusz Embedded API.

This script provides a complete GUI application for loading CSV, TSV, and other
delimited text files, configuring custom delimiters, and generating publication-
quality plots using Veusz's embedded API (separate window with toolbar).

NEW FEATURES IN THIS VERSION:
- Veusz window opens as separate window with full toolbar (Veusz embedded API)
- Save Veusz files button (HDF5 .vszh5 format)
- Support for metadata/header lines
- Data start line configuration
- Dual preview panes (metadata + data)
- Multi-file support with flexible column matching
- ALL loaded files are plotted (file selection only affects preview)
- Overlaid plot + individual plots for each file
- Datasets tagged with original file names
- Page names use file names for easy navigation
- Logging of missing columns to text file
- Metadata stored in page notes (not displayed on plot)

Key Features:
- Support for arbitrary delimiter characters/strings (comma, tab, semicolon, etc.)
- Automatic delimiter detection
- Handle files with metadata headers
- Interactive preview of loaded data
- Multi-column selection for X and Y axes
- Multi-file plotting with flexible column matching by name
- Overlaid plot showing all files + individual plots per file
- Veusz plot window with full toolbar (separate window)
- Save plots as .vszh5 (Veusz HDF5 format)
- Metadata stored in page notes (accessible via Veusz)
- Configurable plot styles and formatting
- Comprehensive error handling and logging
- Optional log file generation for missing columns

Typical Workflow:
1. Launch the application
2. Browse and select CSV/TSV files (can have different column structures)
3. Configure delimiter (auto-detect or manual)
4. Set data start line (where actual data begins)
5. Preview metadata and data (select file in list for preview)
6. Select columns for plotting (matched by column name across files)
7. Configure plot options (title, labels, legend, etc.)
8. Enable logging if you want missing column reports
9. Generate plots in separate Veusz window (ALL loaded files plotted)
10. Navigate between overlaid plot and individual file plots
11. Save Veusz file for later use (.vszh5 format)

Author: Based on William W. Wallace's framework
Last Updated: 2026-02-03
Python Version: 3.8+

Dependencies:
- PyQt5/PySide6 (GUI framework)
- pandas (data loading and manipulation)
- numpy (numerical computing)
- matplotlib (preview plots)
- veusz (plot generation with embedded API)
- scipy (optional, for advanced analysis)

Installation:
    pip install pandas numpy matplotlib veusz pyside6 qtpy

Usage:
    python CSV_TSV_AutoPlot_Embedded.py
"""

import datetime
import multiprocessing
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ============================================================================
# IMPORTS - Veusz Integration (embedded API for separate window)
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
try:
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QMessageBox, QTextEdit,
        QCheckBox, QSpinBox, QGroupBox, QListWidget,
        QLineEdit, QComboBox, QFormLayout, QTableWidget, QTableWidgetItem,
        QSplitter
    )
except ImportError:
    print("ERROR: QtPy not available.")
    print("Install with: pip install pyside6 qtpy")
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
    data_start_line: int = 1  # Line where data begins
    skip_footer: int = 0
    skip_empty_lines: bool = True
    enable_logging: bool = False  # Enable logging to file

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
# LOGGING CLASS
# ============================================================================

class ColumnLogger:
    """
    Handles logging of missing columns across multiple files.
    """

    def __init__(self, log_file_path: str):
        """
        Initialize column logger.

        Parameters:
            log_file_path (str): Path to log file.
        """
        self.log_file_path = log_file_path
        self.log_entries = []

    def log_missing_column(self, filename: str, column: str):
        """
        Log a missing column for a file.

        Parameters:
            filename (str): Name of file.
            column (str): Name of missing column.
        """
        entry = f"File: {filename} | Missing column: {column}"
        self.log_entries.append(entry)

    def log_available_columns(self, filename: str, columns: List[str]):
        """
        Log available columns for a file.

        Parameters:
            filename (str): Name of file.
            columns (List[str]): List of available columns.
        """
        entry = f"File: {filename} | Available columns: {', '.join(columns)}"
        self.log_entries.append(entry)

    def write_log(self):
        """Write accumulated log entries to file."""
        if not self.log_entries:
            return

        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("CSV/TSV AutoPlot - Column Matching Log\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")

                for entry in self.log_entries:
                    f.write(entry + "\n")

                f.write("\n" + "="*80 + "\n")
                f.write(f"Total log entries: {len(self.log_entries)}\n")
                f.write("="*80 + "\n")
        except Exception as e:
            print(f"Error writing log file: {e}")

# ============================================================================
# CSV/TSV LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def detect_delimiter(file_path: str, sample_size: int = 5, 
                    start_line: int = 1) -> str:
    """
    Attempt to auto-detect the delimiter in a delimited text file.

    Parameters:
        file_path (str): Path to the delimited text file.
        sample_size (int): Number of lines to sample for detection.
        start_line (int): Line number where data starts (1-indexed).

    Returns:
        str: Detected delimiter character.
    """
    delimiters = [',', '\t', ';', '|']
    delimiter_counts = {d: 0 for d in delimiters}

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip to start line
            for _ in range(start_line - 1):
                f.readline()

            # Sample data lines
            for _ in range(sample_size):
                line = f.readline()
                if not line:
                    break

                if not line.strip():
                    continue

                for delimiter in delimiters:
                    delimiter_counts[delimiter] += line.count(delimiter)

        if max(delimiter_counts.values()) > 0:
            return max(delimiter_counts, key=delimiter_counts.get)
        else:
            return ','
    except Exception as e:
        print(f"Error detecting delimiter: {e}")
        return ','

def load_csv_file(file_path: str, delimiter: str = ',',
                 config: CSVProcessingConfig = None) -> Tuple[Optional[pd.DataFrame], str, List[str]]:
    """
    Load a CSV/TSV file into a pandas DataFrame with metadata extraction.

    Parameters:
        file_path (str): Path to the CSV/TSV file.
        delimiter (str): Delimiter character or string.
        config (CSVProcessingConfig, optional): Processing configuration.

    Returns:
        Tuple[DataFrame or None, str, List[str]]: Loaded DataFrame, status message, and metadata lines.
    """
    if config is None:
        config = CSVProcessingConfig()

    try:
        # Extract metadata lines (before data start line)
        metadata_lines = []
        if config.data_start_line > 1:
            with open(file_path, 'r', encoding=config.encoding, errors='ignore') as f:
                for i in range(config.data_start_line - 1):
                    line = f.readline()
                    if line:
                        metadata_lines.append(line.rstrip('\n\r'))

        # Load the CSV file with specified delimiter, skipping metadata lines
        skip_rows = config.data_start_line - 1 if config.data_start_line > 1 else 0

        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=config.encoding,
            skiprows=skip_rows,
            skip_blank_lines=config.skip_empty_lines,
            engine='python',
            on_bad_lines='warn'
        )

        # Clean column names (strip whitespace and trailing commas)
        df.columns = df.columns.str.strip().str.rstrip(',')

        status_msg = f"Successfully loaded {os.path.basename(file_path)}: {df.shape[0]} rows, {df.shape[1]} columns"
        if metadata_lines:
            status_msg += f" (skipped {len(metadata_lines)} metadata lines)"

        return df, status_msg, metadata_lines

    except Exception as e:
        error_msg = f"Error loading {file_path}: {str(e)}"
        return None, error_msg, []

# ============================================================================
# VEUSZ PLOTTING CLASS WITH EMBEDDED API (SEPARATE WINDOW)
# ============================================================================

class VeuszPlotter:
    """
    Handles plot generation in Veusz using embedded API (separate window).

    This class wraps the Veusz embedding API to create publication-quality
    plots from pandas DataFrames. The Veusz window opens separately with
    full toolbar enabled.

    Supports multiple files with different column structures - columns are
    matched by name, and missing columns are filled with NaN.

    Metadata is stored in page notes (not displayed on plot).

    Creates both overlaid plot (all files) and individual plots per file.
    """

    def __init__(self, plot_title: str = "CSV Data Plot",
                 dataset_name: str = "CSVDataset"):
        """
        Initialize Veusz plotter with embedded API.

        Parameters:
            plot_title (str): Title for the plot window.
            dataset_name (str): Base name for datasets in Veusz.
        """
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.doc = None

        try:
            if vz is not None:
                # Create embedded document with separate window (hidden=False)
                self.doc = vz.Embedded(plot_title, hidden=False)
                self.doc.EnableToolbar()
        except Exception as e:
            print(f"Warning: Could not initialize Veusz: {e}")

    def create_multi_file_plot(self, file_data_dict: Dict[str, Tuple[pd.DataFrame, List[str]]], 
                               x_column: str, y_columns: List[str],
                               plot_config: PlotConfig = None,
                               logger: Optional[ColumnLogger] = None) -> bool:
        """
        Create XY plots from multiple files with flexible column matching.

        Creates:
        1. Overlaid plot (all files on one graph) on "All Files" page
        2. Individual plots for each file on separate pages

        Metadata from all files is stored in page notes (not displayed on plot).
        All datasets are tagged with their original file names.

        Parameters:
            file_data_dict (Dict): Dictionary mapping filename to (DataFrame, metadata).
            x_column (str): Name of column to use for X axis.
            y_columns (List[str]): Names of columns to use for Y axes.
            plot_config (PlotConfig, optional): Plot configuration.
            logger (ColumnLogger, optional): Logger for missing columns.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.doc is None:
            return False

        if plot_config is None:
            plot_config = PlotConfig()

        try:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

            # ====================================================================
            # PREPARE METADATA NOTES (SHARED ACROSS ALL PAGES)
            # ====================================================================
            metadata_notes = []
            metadata_notes.append("="*80)
            metadata_notes.append("CSV/TSV AutoPlot - File Metadata")
            metadata_notes.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            metadata_notes.append("="*80)
            metadata_notes.append("")

            for filename, (df, metadata) in file_data_dict.items():
                metadata_notes.append(f"File: {filename}")
                metadata_notes.append("-" * 80)

                if metadata and len(metadata) > 0:
                    metadata_notes.append("Header/Metadata lines:")
                    for line in metadata:
                        metadata_notes.append(f"  {line}")
                else:
                    metadata_notes.append("(No metadata - data starts at line 1)")

                metadata_notes.append(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
                metadata_notes.append(f"Columns: {', '.join(df.columns.tolist())}")
                metadata_notes.append("")

            metadata_notes.append("="*80)
            metadata_notes.append(f"Plot Configuration:")
            metadata_notes.append(f"  X-Axis: {x_column}")
            metadata_notes.append(f"  Y-Axes: {', '.join(y_columns)}")
            metadata_notes.append(f"  Title: {plot_config.title}")
            metadata_notes.append("="*80)

            metadata_str = "\n".join(metadata_notes)

            # ====================================================================
            # PAGE 1: OVERLAID PLOT (ALL FILES)
            # ====================================================================
            print("Creating overlaid plot page...")
            page_overlaid = self.doc.Root.Add('page', name='All_Files_Overlaid')
            page_overlaid.notes.val = metadata_str

            grid_overlaid = page_overlaid.Add('grid', columns=1, rows=1)
            graph_overlaid = grid_overlaid.Add('graph', name='AllFilesPlot')

            plot_index = 0

            # Process each file for overlaid plot
            for file_idx, (filename, (df, metadata)) in enumerate(file_data_dict.items()):
                available_columns = df.columns.tolist()

                if logger:
                    logger.log_available_columns(filename, available_columns)

                # Check if X column exists
                if x_column not in available_columns:
                    if logger:
                        logger.log_missing_column(filename, x_column)
                    print(f"Warning: X column '{x_column}' not found in {filename}, skipping file")
                    continue

                # Get X data
                x_data = pd.to_numeric(df[x_column], errors='coerce').dropna().values.tolist()
                x_dataset_name = f'{self.dataset_name}_X_{file_idx}_{filename}'
                self.doc.SetData(x_dataset_name, x_data)

                # Tag dataset with file name
                self.doc.TagDatasets(x_dataset_name, [filename, 'X-axis', x_column])

                # Process each Y column
                for y_col in y_columns:
                    if y_col not in available_columns:
                        if logger:
                            logger.log_missing_column(filename, y_col)
                        print(f"Warning: Y column '{y_col}' not found in {filename}, skipping")
                        continue

                    # Get Y data
                    y_data = pd.to_numeric(df[y_col], errors='coerce').dropna().values.tolist()
                    y_dataset_name = f'{self.dataset_name}_Y_{file_idx}_{filename}_{y_col}'
                    self.doc.SetData(y_dataset_name, y_data)

                    # Tag dataset with file name
                    self.doc.TagDatasets(y_dataset_name, [filename, 'Y-axis', y_col])

                    # Create XY plot
                    plot_name = f'{filename}_{y_col}'
                    xy_plot = graph_overlaid.Add('xy', name=plot_name,
                                                xData=x_dataset_name,
                                                yData=y_dataset_name)

                    # Configure appearance
                    xy_plot.PlotLine.color.val = colors[plot_index % len(colors)]
                    xy_plot.PlotLine.style.val = plot_config.line_style
                    xy_plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                    xy_plot.marker.val = plot_config.marker_style
                    xy_plot.markerSize.val = f'{plot_config.marker_size}pt'

                    plot_index += 1

            # Configure overlaid graph
            graph_overlaid.title.val = f"{plot_config.title} (All Files)"
            graph_overlaid.xlabel.val = plot_config.x_label
            graph_overlaid.ylabel.val = plot_config.y_label
            graph_overlaid.xLog.val = (plot_config.x_scale == 'log')
            graph_overlaid.yLog.val = (plot_config.y_scale == 'log')
            graph_overlaid.showLegend.val = plot_config.show_legend
            if plot_config.show_legend:
                graph_overlaid.legend.pos.val = plot_config.legend_position

            # ====================================================================
            # PAGES 2+: INDIVIDUAL PLOTS FOR EACH FILE
            # ====================================================================
            print("Creating individual file plot pages...")

            for file_idx, (filename, (df, metadata)) in enumerate(file_data_dict.items()):
                available_columns = df.columns.tolist()

                # Check if X column exists
                if x_column not in available_columns:
                    print(f"Skipping individual plot for {filename}: X column not found")
                    continue

                # Create page for this file (use filename without extension as page name)
                page_name = os.path.splitext(filename)[0].replace(' ', '_').replace('.', '_')
                page_individual = self.doc.Root.Add('page', name=page_name)

                # Add file-specific metadata to page notes
                file_notes = []
                file_notes.append("="*80)
                file_notes.append(f"File: {filename}")
                file_notes.append("="*80)
                file_notes.append("")

                if metadata and len(metadata) > 0:
                    file_notes.append("Header/Metadata lines:")
                    for line in metadata:
                        file_notes.append(f"  {line}")
                    file_notes.append("")

                file_notes.append(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
                file_notes.append(f"Columns: {', '.join(df.columns.tolist())}")
                file_notes.append("")
                file_notes.append("="*80)

                page_individual.notes.val = "\n".join(file_notes)

                # Create grid and graph
                grid_individual = page_individual.Add('grid', columns=1, rows=1)
                graph_individual = grid_individual.Add('graph', name=f'{page_name}_Plot')

                # Get X data (reuse from overlaid plot)
                x_dataset_name = f'{self.dataset_name}_X_{file_idx}_{filename}'

                # Process each Y column for this file
                plot_color_index = 0
                for y_col in y_columns:
                    if y_col not in available_columns:
                        continue

                    # Reuse Y data from overlaid plot
                    y_dataset_name = f'{self.dataset_name}_Y_{file_idx}_{filename}_{y_col}'

                    # Create XY plot
                    plot_name = f'{y_col}'
                    xy_plot = graph_individual.Add('xy', name=plot_name,
                                                  xData=x_dataset_name,
                                                  yData=y_dataset_name)

                    # Configure appearance
                    xy_plot.PlotLine.color.val = colors[plot_color_index % len(colors)]
                    xy_plot.PlotLine.style.val = plot_config.line_style
                    xy_plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                    xy_plot.marker.val = plot_config.marker_style
                    xy_plot.markerSize.val = f'{plot_config.marker_size}pt'

                    plot_color_index += 1

                # Configure individual graph
                graph_individual.title.val = f"{plot_config.title} - {filename}"
                graph_individual.xlabel.val = plot_config.x_label
                graph_individual.ylabel.val = plot_config.y_label
                graph_individual.xLog.val = (plot_config.x_scale == 'log')
                graph_individual.yLog.val = (plot_config.y_scale == 'log')
                graph_individual.showLegend.val = (len(y_columns) > 1)  # Show legend only if multiple Y columns
                if len(y_columns) > 1:
                    graph_individual.legend.pos.val = plot_config.legend_position

            print(f"Created overlaid plot + {len(file_data_dict)} individual file plots")
            return True

        except Exception as e:
            print(f"Error creating multi-file plot: {e}")
            traceback.print_exc()
            return False

    def save_project(self, file_path: str) -> bool:
        """
        Save Veusz project to file in HDF5 format (.vszh5).

        Parameters:
            file_path (str): Path where to save the Veusz project.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.doc is None:
            return False

        try:
            # Ensure .vszh5 extension
            if not file_path.lower().endswith('.vszh5'):
                # Remove .vsz if present and add .vszh5
                file_path = os.path.splitext(file_path)[0] + '.vszh5'

            # Save with HDF5 mode
            self.doc.Save(file_path, mode='hdf5')
            return True
        except Exception as e:
            print(f"Error saving Veusz project: {e}")
            traceback.print_exc()
            return False

# ============================================================================
# MATPLOTLIB PREVIEW CANVAS
# ============================================================================

class PreviewCanvas(FigureCanvas):
    """
    Matplotlib canvas for previewing data plots.
    """

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        """Initialize preview canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_data(self, df: pd.DataFrame, x_column: str, y_columns: List[str],
                 plot_config: PlotConfig = None):
        """Plot data on the canvas."""
        if plot_config is None:
            plot_config = PlotConfig()

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Check if X column exists
        if x_column not in df.columns:
            ax.text(0.5, 0.5, f"X column '{x_column}' not found in current file",
                   ha='center', va='center', transform=ax.transAxes)
            self.draw()
            return

        x_data = pd.to_numeric(df[x_column], errors='coerce').dropna()

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for idx, y_col in enumerate(y_columns):
            # Check if Y column exists
            if y_col not in df.columns:
                print(f"Warning: Y column '{y_col}' not found in current file for preview")
                continue

            y_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
            valid_idx = np.intersect1d(x_data.index, y_data.index)

            if len(valid_idx) > 0:
                ax.plot(x_data[valid_idx], y_data[valid_idx],
                       color=colors[idx % len(colors)],
                       marker=plot_config.marker_style,
                       markersize=plot_config.marker_size,
                       linestyle=plot_config.line_style,
                       linewidth=plot_config.line_width,
                       label=y_col, alpha=0.7)

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
    Main application window for CSV/TSV AutoPlot with Veusz embedded API.

    Supports multiple files with different column structures.
    Metadata stored in page notes (not displayed on plot).
    ALL loaded files are plotted (file selection only affects preview).
    Creates overlaid plot + individual plots per file.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("CSV/TSV AutoPlot - Overlaid + Individual Plots")
        self.setGeometry(100, 100, 1400, 950)

        self.selected_files = []
        self.loaded_data = {}  # Dict[filename, (DataFrame, metadata)]

        self.csv_config = CSVProcessingConfig()
        self.plot_config = PlotConfig()

        self.veusz_plotter = None

        self.setup_ui()

        self.log_message("CSV/TSV AutoPlot initialized")
        self.log_message("Features: Overlaid plot + Individual file plots")
        self.log_message("All datasets tagged with file names")

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
        self.file_list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.file_list_widget.itemSelectionChanged.connect(self.on_file_selection_changed)

        left_layout.addWidget(QLabel("Loaded Files (select for preview only):"))
        left_layout.addWidget(self.file_list_widget)

        preview_note = QLabel("ℹ️ Overlaid plot + individual plots for each file")
        preview_note.setStyleSheet("QLabel { color: #0066cc; font-weight: bold; }")
        left_layout.addWidget(preview_note)

        # Right side: Configuration
        right_layout = QVBoxLayout()

        # Delimiter configuration
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

        delimiter_form.addRow("Presets:", QWidget())

        right_layout.addWidget(delimiter_group)

        # Data Start Line configuration
        data_line_group = QGroupBox("Data Start Line Configuration")
        data_line_form = QFormLayout(data_line_group)

        self.data_start_line_spinbox = QSpinBox()
        self.data_start_line_spinbox.setMinimum(1)
        self.data_start_line_spinbox.setMaximum(1000)
        self.data_start_line_spinbox.setValue(1)
        self.data_start_line_spinbox.valueChanged.connect(self.on_data_start_line_changed)

        data_line_form.addRow("Data Start Line:", self.data_start_line_spinbox)
        data_line_form.addRow(QLabel("(Line where column headers begin)"))

        right_layout.addWidget(data_line_group)

        # Logging configuration
        logging_group = QGroupBox("Logging Options")
        logging_layout = QVBoxLayout(logging_group)

        self.enable_logging_checkbox = QCheckBox("Enable logging to file")
        self.enable_logging_checkbox.setChecked(False)
        self.enable_logging_checkbox.stateChanged.connect(self.on_logging_changed)
        logging_layout.addWidget(self.enable_logging_checkbox)

        logging_info = QLabel("(Logs missing columns to text file)")
        logging_info.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        logging_layout.addWidget(logging_info)

        right_layout.addWidget(logging_group)

        # Add preset layout
        preset_widget = QWidget()
        preset_widget.setLayout(preset_layout)
        right_layout.insertWidget(1, preset_widget)

        right_layout.addStretch()

        file_layout.addLayout(left_layout, 1)
        file_layout.addLayout(right_layout, 0)

        main_layout.addWidget(file_group)

        # ====== PREVIEW AREA ======
        preview_group = QGroupBox("File Preview (Selected File Only)")
        preview_layout = QVBoxLayout(preview_group)

        # Metadata preview
        preview_layout.addWidget(QLabel("Metadata:"))
        self.metadata_preview = QTextEdit()
        self.metadata_preview.setReadOnly(True)
        self.metadata_preview.setMaximumHeight(120)
        self.metadata_preview.setStyleSheet("QTextEdit { background-color: #f0f0f0; font-family: monospace; }")
        preview_layout.addWidget(self.metadata_preview)

        # Data preview
        preview_layout.addWidget(QLabel("Data Preview (first 10 rows):"))
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.data_preview_table)

        main_layout.addWidget(preview_group)

        # ====== COLUMN SELECTION ======
        column_group = QGroupBox("Column Selection for Plotting")
        column_layout = QVBoxLayout(column_group)

        selection_form = QFormLayout()

        self.x_column_combo = QComboBox()
        self.x_column_combo.currentTextChanged.connect(self.on_plot_config_changed)
        selection_form.addRow("X-Axis Column:", self.x_column_combo)

        y_label = QLabel("Y-Axis Columns (select multiple):")
        self.y_columns_list = QListWidget()
        self.y_columns_list.setSelectionMode(QListWidget.MultiSelection)
        self.y_columns_list.itemSelectionChanged.connect(self.on_plot_config_changed)
        self.y_columns_list.setMaximumHeight(100)

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

        # ====== MATPLOTLIB PREVIEW ======
        self.preview_canvas = PreviewCanvas(width=8, height=2.5)
        main_layout.addWidget(QLabel("Matplotlib Preview:"))
        main_layout.addWidget(self.preview_canvas)

        # ====== ACTION BUTTONS ======
        button_group = QGroupBox("")
        button_layout = QHBoxLayout(button_group)

        self.generate_btn = QPushButton("Generate Plots (Overlaid + Individual)")
        self.generate_btn.clicked.connect(self.generate_plots)
        self.generate_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; background-color: #4CAF50; color: white; }")

        self.save_veusz_btn = QPushButton("Save Veusz File (.vszh5)")
        self.save_veusz_btn.clicked.connect(self.save_veusz_file)
        self.save_veusz_btn.setEnabled(False)

        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.save_veusz_btn)
        button_layout.addStretch()

        main_layout.addWidget(button_group)

        # ====== STATUS LOG ======
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        status_layout.addWidget(self.status_text)

        main_layout.addWidget(status_group)

    # ========================================================================
    # FILE MANAGEMENT METHODS
    # ========================================================================

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
        if delimiter == '\\t':
            delimiter = '\t'

        self.csv_config.data_start_line = self.data_start_line_spinbox.value()

        for file_path in self.selected_files:
            filename = os.path.basename(file_path)

            if filename not in self.loaded_data:
                df, msg, metadata = load_csv_file(file_path, delimiter, self.csv_config)
                if df is not None:
                    self.loaded_data[filename] = (df, metadata)
                    self.log_message(msg)
                else:
                    self.log_message(msg)

        self.update_file_list()
        self.update_column_selectors_from_all_files()

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
        self.update_column_selectors_from_all_files()
        self.preview_canvas.clear_plot()
        self.metadata_preview.clear()
        self.data_preview_table.clear()
        self.log_message("File list cleared")

    def on_file_selection_changed(self):
        """Handle file selection change."""
        selected_items = self.file_list_widget.selectedItems()
        if selected_items:
            filename = selected_items[0].text()
            self.display_data_preview(filename)
            self.display_metadata_preview(filename)
            self.log_message(f"Previewing: {filename}")

    def auto_detect_delimiter(self):
        """Auto-detect delimiter from first loaded file."""
        if not self.loaded_data:
            QMessageBox.warning(self, "No Files Loaded", "Please load files first")
            return

        first_filename = list(self.loaded_data.keys())[0]
        file_path = self._get_file_path(first_filename)

        if file_path:
            start_line = self.data_start_line_spinbox.value()
            detected_delim = detect_delimiter(file_path, start_line=start_line)
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
        self.loaded_data.clear()
        self.load_files()

    def on_data_start_line_changed(self):
        """Handle data start line change."""
        self.csv_config.data_start_line = self.data_start_line_spinbox.value()
        self.loaded_data.clear()
        self.load_files()
        self.log_message(f"Data start line set to: {self.csv_config.data_start_line}")

    def on_logging_changed(self):
        """Handle logging checkbox change."""
        self.csv_config.enable_logging = self.enable_logging_checkbox.isChecked()
        status = "enabled" if self.csv_config.enable_logging else "disabled"
        self.log_message(f"Logging {status}")

    def _get_file_path(self, filename: str) -> str:
        """Get full file path from filename."""
        for file_path in self.selected_files:
            if os.path.basename(file_path) == filename:
                return file_path
        return ""

    def update_column_selectors_from_all_files(self):
        """Update column selectors with union of all columns from all loaded files."""
        self.x_column_combo.clear()
        self.y_columns_list.clear()

        if not self.loaded_data:
            return

        all_columns = set()
        for df, _ in self.loaded_data.values():
            all_columns.update(df.columns.tolist())

        sorted_columns = sorted(list(all_columns))

        self.x_column_combo.addItems(sorted_columns)
        self.y_columns_list.addItems(sorted_columns)

        if len(sorted_columns) > 0:
            self.x_column_combo.setCurrentIndex(0)

        self.log_message(f"Found {len(sorted_columns)} unique columns across all files")

    def display_metadata_preview(self, filename: str):
        """Display metadata lines in preview pane."""
        if filename not in self.loaded_data:
            return

        _, metadata = self.loaded_data[filename]

        if metadata:
            preview_lines = metadata[:10]
            self.metadata_preview.setText("\n".join(preview_lines))
            if len(metadata) > 10:
                self.metadata_preview.append(f"\n... ({len(metadata) - 10} more lines)")
        else:
            self.metadata_preview.setText("(No metadata)")

    def display_data_preview(self, filename: str):
        """Display a preview of the loaded data in the table."""
        if filename not in self.loaded_data:
            return

        df, _ = self.loaded_data[filename]
        preview_df = df.head(10)

        self.data_preview_table.setRowCount(preview_df.shape[0])
        self.data_preview_table.setColumnCount(preview_df.shape[1])
        self.data_preview_table.setHorizontalHeaderLabels(preview_df.columns)

        for i, row in enumerate(preview_df.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_preview_table.setItem(i, j, item)

        self.data_preview_table.resizeColumnsToContents()

    def on_plot_config_changed(self):
        """Handle plot configuration change."""
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            if self.loaded_data:
                first_filename = list(self.loaded_data.keys())[0]
            else:
                return
        else:
            first_filename = selected_items[0].text()

        if first_filename not in self.loaded_data:
            return

        df, _ = self.loaded_data[first_filename]

        x_column = self.x_column_combo.currentText()
        y_columns = [item.text() for item in self.y_columns_list.selectedItems()]

        if not x_column or not y_columns:
            return

        self.plot_config.title = self.title_input.text()
        self.plot_config.x_label = self.x_label_input.text()
        self.plot_config.y_label = self.y_label_input.text()
        self.plot_config.show_legend = self.show_legend_checkbox.isChecked()

        try:
            self.preview_canvas.plot_data(df, x_column, y_columns, self.plot_config)
        except Exception as e:
            self.log_message(f"Error updating preview: {str(e)}")

    def generate_plots(self):
        """Generate plots in Veusz from ALL loaded files."""
        if not self.loaded_data:
            QMessageBox.warning(self, "No Data", "Please load CSV/TSV files first")
            return

        try:
            x_column = self.x_column_combo.currentText()
            y_columns = [item.text() for item in self.y_columns_list.selectedItems()]

            if not x_column or not y_columns:
                QMessageBox.warning(self, "No Columns Selected",
                                  "Please select X and Y columns")
                return

            logger = None
            if self.csv_config.enable_logging:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = os.path.join(script_dir, f"column_matching_log_{timestamp}.txt")
                logger = ColumnLogger(log_file_path)
                self.log_message(f"Logging enabled: {log_file_path}")

            self.veusz_plotter = VeuszPlotter(
                plot_title=self.plot_config.title,
                dataset_name="MultiFileData"
            )

            success = self.veusz_plotter.create_multi_file_plot(
                self.loaded_data, x_column, y_columns,
                self.plot_config,
                logger
            )

            if logger:
                logger.write_log()
                self.log_message(f"✓ Log file written: {log_file_path}")

            if success:
                self.save_veusz_btn.setEnabled(True)
                num_files = len(self.loaded_data)
                self.log_message(f"✓ Created overlaid plot + {num_files} individual plots")
                self.log_message("✓ All datasets tagged with file names")
                self.log_message("✓ Metadata stored in page notes")
                QMessageBox.information(self, "Success",
                                      f"Generated plots from {num_files} file(s):\n\n" +
                                      "• Overlaid plot (all files)\n" +
                                      f"• {num_files} individual plots (one per file)\n\n" +
                                      "Navigate between pages in Veusz window.\n" +
                                      "All datasets are tagged with file names.\n" +
                                      "Metadata is in page notes (Edit → Page Properties → Notes).")
            else:
                QMessageBox.critical(self, "Error", "Failed to create plot in Veusz")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating plots: {str(e)}")
            self.log_message(f"✗ Plot generation error: {str(e)}")
            traceback.print_exc()

    def save_veusz_file(self):
        """Save the current Veusz plot to HDF5 .vszh5 file."""
        if self.veusz_plotter is None or self.veusz_plotter.doc is None:
            QMessageBox.warning(self, "No Plot", "Please generate a plot first")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Veusz File",
            "",
            "Veusz HDF5 Files (*.vszh5);;All Files (*)"
        )

        if not file_path:
            return

        try:
            success = self.veusz_plotter.save_project(file_path)

            if success:
                if not file_path.endswith('.vszh5'):
                    file_path = os.path.splitext(file_path)[0] + '.vszh5'

                self.log_message(f"✓ Veusz file saved: {file_path}")
                QMessageBox.information(self, "Success",
                                      f"Veusz file saved successfully:\n\n{file_path}\n\n" +
                                      "File format: HDF5 (.vszh5)\n" +
                                      "Contains overlaid plot + individual plots\n" +
                                      "All datasets tagged with file names")
            else:
                QMessageBox.critical(self, "Error", "Failed to save Veusz file")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving Veusz file: {str(e)}")
            self.log_message(f"✗ Save error: {str(e)}")

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
