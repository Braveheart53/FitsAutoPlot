"""
CSV/TSV AutoPlot V6 - Automatic data visualization with Veusz Embedded API.

This script provides a complete GUI application for loading CSV, TSV, and other
delimited text files, configuring custom delimiters, and generating publication-
quality plots using Veusz's embedded API (separate window with toolbar).

VERSION 6 (2026-02-03) - SEPARATE PAGES FOR INDIVIDUAL PLOTS:
- Individual plots now on separate pages (not grid/subgrid)
- Total pages = Number_of_files + 1 (overlay page + one page per file)
- Fixed legend/key implementation (boolean False, not string "False")
- Page notes stored in page.notes.val (not separate label widget)
- Each file uses its own independent X-axis data

VERSION 5 features retained:
- X-axis column shows only COMMON columns across ALL files (intersection)
- Y-axis columns show ALL columns from ALL files (union)
- Plot button turns RED/disabled when X column not in all files
- Plot button turns GREEN/enabled when X column exists in all files
- RED warning message in GUI when X column missing from files
- Missing Y columns skipped per file (logged to file)
- Full support for files with different numbers of columns
- Dataset naming: [filename]_[column_name] for each file
- Veusz API corrections and style mappings
- Logging checkbox and log file creation
- "Save to Veusz" button for .vszh5 export

Author: Based on William W. Wallace's framework
Last Updated: 2026-02-03
Python Version: 3.8+

Dependencies:
- PyQt5/PySide6 (GUI framework)
- pandas (data loading and manipulation)
- numpy (numerical computing)
- veusz (plot generation)

Installation:
pip install pandas numpy veusz pyside6

Usage:
python CSV_TSV_AutoPlot_V6.py
"""

import datetime
import multiprocessing
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

# Data Processing
import numpy as np
import pandas as pd

try:
    import scipy.stats as stats
except ImportError:
    stats = None

# Veusz for plots
try:
    import veusz.embed as vz
except ImportError:
    print("WARNING: Veusz not installed. Plot generation will not work.")
    print("Install with: pip install veusz")
    vz = None

# Qt Framework
if getattr(sys, 'frozen', False):
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QMessageBox, QTextEdit,
        QCheckBox, QGroupBox, QListWidget, QLineEdit, QComboBox,
        QFormLayout, QTableWidget, QTableWidgetItem, QSpinBox
    )
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt
else:
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
    header_line: int = 0  # Line number where column headers exist (0-indexed)
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
    line_style: str = "-"  # Matplotlib-style notation
    line_width: float = 2.0
    marker_style: str = "o"  # Matplotlib-style notation
    marker_size: float = 6.0


# ============================================================================
# CSV/TSV LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def detect_delimiter(file_path: str, sample_size: int = 5) -> str:
    """
    Attempt to auto-detect the delimiter in a delimited text file.

    Parameters:
        file_path (str): Path to the delimited text file.
        sample_size (int): Number of lines to sample for detection.

    Returns:
        str: Detected delimiter character.
    """
    delimiters = [',', '\t', ';', '|']
    delimiter_counts = {d: 0 for d in delimiters}

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(sample_size):
                line = f.readline()
                if not line:
                    break
                if not line.strip() or line.startswith('#'):
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


def read_file_metadata(file_path: str, header_line: int) -> List[str]:
    """
    Read metadata lines from file (lines before header).

    Parameters:
        file_path (str): Path to the file.
        header_line (int): 0-indexed line number where headers exist.

    Returns:
        List[str]: List of metadata lines.
    """
    metadata_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i in range(header_line):
                line = f.readline()
                if line:
                    metadata_lines.append(line.rstrip('\n\r'))
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")

    return metadata_lines


def load_csv_file(file_path: str, delimiter: str = ',',
                  config: CSVProcessingConfig = None) -> Tuple[Optional[pd.DataFrame], str, List[str]]:
    """
    Load a CSV/TSV file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV/TSV file.
        delimiter (str): Delimiter character or string.
        config (CSVProcessingConfig): Processing configuration including header_line.

    Returns:
        Tuple[DataFrame or None, str, List[str]]: Loaded DataFrame, status message, and metadata lines.
    """
    if config is None:
        config = CSVProcessingConfig()

    try:
        # Read metadata lines before header
        metadata_lines = read_file_metadata(file_path, config.header_line)

        # Calculate skiprows: skip all lines before header_line
        skiprows = list(range(config.header_line)
                        ) if config.header_line > 0 else None

        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=config.encoding,
            skiprows=skiprows,
            skip_blank_lines=config.skip_empty_lines,
            engine='python'
        )
        status_msg = f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns (header at line {config.header_line + 1})"
        return df, status_msg, metadata_lines
    except Exception as e:
        error_msg = f"Error loading {file_path}: {str(e)}"
        return None, error_msg, []


def get_common_columns(data_dict: Dict[str, pd.DataFrame]) -> Set[str]:
    """
    Get columns that are common to ALL loaded DataFrames (intersection).

    Parameters:
        data_dict: Dictionary mapping filenames to DataFrames.

    Returns:
        Set[str]: Set of column names present in all DataFrames.
    """
    if not data_dict:
        return set()

    # Start with columns from first DataFrame
    common_cols = set(next(iter(data_dict.values())).columns)

    # Intersect with columns from all other DataFrames
    for df in data_dict.values():
        common_cols = common_cols.intersection(set(df.columns))

    return common_cols


def get_all_columns(data_dict: Dict[str, pd.DataFrame]) -> Set[str]:
    """
    Get all unique columns from ALL loaded DataFrames (union).

    Parameters:
        data_dict: Dictionary mapping filenames to DataFrames.

    Returns:
        Set[str]: Set of all unique column names across all DataFrames.
    """
    all_cols = set()

    for df in data_dict.values():
        all_cols = all_cols.union(set(df.columns))

    return all_cols


# ============================================================================
# VEUSZ PLOTTING CLASS
# ============================================================================

class VeuszPlotter:
    """
    Handles plot generation in Veusz for CSV/TSV data.

    This class wraps the Veusz embedding API to create publication-quality
    plots from pandas DataFrames loaded from CSV/TSV files.
    """

    # FIXED: Mapping from matplotlib line styles to Veusz line styles
    LINE_STYLE_MAP = {
        '-': 'solid',
        '--': 'dashed',
        '-.': 'dashdot',
        ':': 'dotted',
        'solid': 'solid',
        'dashed': 'dashed',
        'dashdot': 'dashdot',
        'dotted': 'dotted'
    }

    # FIXED: Mapping from matplotlib markers to Veusz markers
    MARKER_STYLE_MAP = {
        'o': 'circle',
        's': 'square',
        '^': 'triangle',
        'v': 'triangledown',
        'd': 'diamond',
        '+': 'plus',
        'x': 'cross',
        '*': 'star',
        'circle': 'circle',
        'square': 'square',
        'triangle': 'triangle',
        'diamond': 'diamond'
    }

    def __init__(self, plot_title: str = "CSV Data Plot"):
        """Initialize Veusz plotter."""
        self.plot_title = plot_title
        self.doc = None
        self.log_file = None
        self.enable_logging = False

        try:
            if vz is not None:
                self.doc = vz.Embedded(plot_title, hidden=False)
                self.doc.EnableToolbar()
        except Exception as e:
            print(f"Warning: Could not initialize Veusz: {e}")

    def set_logging(self, enable: bool, log_path: str = None):
        """
        Enable or disable logging.

        Parameters:
            enable (bool): Enable logging.
            log_path (str): Path to log file.
        """
        self.enable_logging = enable
        if enable and log_path:
            try:
                self.log_file = open(log_path, 'w')
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"CSV/TSV AutoPlot V6 Log - {timestamp}\n")
                self.log_file.write("="*60 + "\n\n")
            except Exception as e:
                print(f"Error opening log file: {e}")
                self.log_file = None

    def log_message(self, message: str):
        """Write message to log file."""
        if self.enable_logging and self.log_file:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_file.write(f"[{timestamp}] {message}\n")
            self.log_file.flush()

    def create_overlay_and_individual_plots(
        self,
        data_dict: Dict[str, pd.DataFrame],
        metadata_dict: Dict[str, List[str]],
        x_column: str,
        y_columns: List[str],
        plot_config: PlotConfig = None
    ) -> bool:
        """
        Create overlay plot (all files on one page) and individual plots (one page per file).
        Each file uses its own independent X-axis data.

        V6 CHANGE: Individual plots are now on SEPARATE PAGES, not in a grid.
        Total pages = Number_of_files + 1 (overlay + individual pages)

        Parameters:
            data_dict: Dictionary mapping filenames to DataFrames
            metadata_dict: Dictionary mapping filenames to metadata lines
            x_column: Name of column to use for X axis (must exist in all files)
            y_columns: Names of columns to use for Y axes (matched by name per file)
            plot_config: Plot configuration

        Returns:
            bool: True if successful, False otherwise
        """
        if self.doc is None:
            return False

        if plot_config is None:
            plot_config = PlotConfig()

        try:
            # Prepare metadata text (used for all pages)
            notes_text = "File Metadata:\n" + "="*60 + "\n\n"
            for filename, metadata_lines in metadata_dict.items():
                notes_text += f"File: {filename}\n"
                notes_text += "-"*60 + "\n"
                if metadata_lines:
                    for line in metadata_lines:
                        notes_text += f"{line}\n"
                else:
                    notes_text += "(No metadata lines before header)\n"
                notes_text += "\n"

            # Color palette for plots
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown',
                      'cyan', 'magenta', 'yellow', 'black']

            # ========== CREATE OVERLAY PAGE (all files on one page) ==========
            overlay_page = self.doc.Root.Add('page', name='OverlayPlot')
            overlay_page.notes.val = notes_text  # Store metadata in page notes

            overlay_graph = overlay_page.Add('graph', name='OverlayGraph')

            plot_idx = 0
            for filename, df in data_dict.items():
                base_filename = os.path.splitext(filename)[0]

                # Check if X column exists (should always exist due to validation)
                if x_column not in df.columns:
                    msg = f"WARNING: X column '{x_column}' not found in {filename}"
                    self.log_message(msg)
                    print(msg)
                    continue

                # Add X data with filename tag (independent for each file)
                x_data = df[x_column].dropna().values.tolist()
                dataset_x_name = f'{base_filename}_{x_column}'
                self.doc.SetData(dataset_x_name, x_data)
                self.doc.TagDatasets(base_filename, [dataset_x_name])

                # Add Y data for each selected column
                for y_col in y_columns:
                    if y_col not in df.columns:
                        msg = f"WARNING: Y column '{y_col}' not found in {filename} - skipping"
                        self.log_message(msg)
                        print(msg)
                        continue

                    y_data = df[y_col].dropna().values.tolist()
                    dataset_y_name = f'{base_filename}_{y_col}'
                    self.doc.SetData(dataset_y_name, y_data)
                    self.doc.TagDatasets(base_filename, [dataset_y_name])

                    # Create XY plot on overlay (each trace uses its file's own X data)
                    xy_plot = overlay_graph.Add('xy',
                                                name=f'Overlay_{base_filename}_{y_col}',
                                                xData=dataset_x_name,  # File-specific X data
                                                yData=dataset_y_name)

                    # Convert styles
                    veusz_line_style = self.LINE_STYLE_MAP.get(
                        plot_config.line_style, 'solid')
                    veusz_marker = self.MARKER_STYLE_MAP.get(
                        plot_config.marker_style, 'circle')

                    # Configure appearance
                    with self._wrap_widget(xy_plot) as plot:
                        plot.PlotLine.color.val = colors[plot_idx % len(
                            colors)]
                        plot.PlotLine.style.val = veusz_line_style
                        plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                        plot.marker.val = veusz_marker
                        plot.markerSize.val = f'{plot_config.marker_size}pt'
                        plot.key.val = f'{base_filename}_{y_col}'

                    plot_idx += 1

            # Configure overlay graph
            with self._wrap_widget(overlay_graph) as g:
                g.Add('label', name='plotTitle')
                g.plotTitle.Text.size.val = '12pt'
                g.plotTitle.label.val = f"Overlay: {plot_config.title}"
                g.plotTitle.alignHorz.val = 'centre'
                g.plotTitle.yPos.val = 1.05
                g.plotTitle.xPos.val = 0.5
                g.topMargin.val = '1cm'

                g.x.label.val = plot_config.x_label
                g.y.label.val = plot_config.y_label
                g.x.log.val = (plot_config.x_scale == 'log')
                g.y.log.val = (plot_config.y_scale == 'log')
                g.x.GridLines.hide.val = not plot_config.show_grid
                g.y.GridLines.hide.val = not plot_config.show_grid

                if plot_config.show_legend:
                    key = g.Add('key')
                    key.hide.val = False
                    key.horzPosn.val = 'right'
                    key.vertPosn.val = 'top'

            # ========== CREATE INDIVIDUAL PAGES (one page per file) ==========
            for filename, df in data_dict.items():
                base_filename = os.path.splitext(filename)[0]

                # Create separate page for this file
                file_page = self.doc.Root.Add(
                    'page', name=f'Page_{base_filename}')
                file_page.notes.val = notes_text  # Store metadata in page notes

                # Create graph for this file
                file_graph = file_page.Add(
                    'graph', name=f'Graph_{base_filename}')

                # Check if X column exists
                if x_column not in df.columns:
                    continue

                dataset_x_name = f'{base_filename}_{x_column}'

                plot_idx = 0
                for y_col in y_columns:
                    if y_col not in df.columns:
                        continue

                    dataset_y_name = f'{base_filename}_{y_col}'

                    # Create XY plot (uses this file's own X data)
                    xy_plot = file_graph.Add('xy',
                                             name=f'{base_filename}_{y_col}',
                                             xData=dataset_x_name,  # File-specific X data
                                             yData=dataset_y_name)

                    # Convert styles
                    veusz_line_style = self.LINE_STYLE_MAP.get(
                        plot_config.line_style, 'solid')
                    veusz_marker = self.MARKER_STYLE_MAP.get(
                        plot_config.marker_style, 'circle')

                    # Configure appearance
                    with self._wrap_widget(xy_plot) as plot:
                        plot.PlotLine.color.val = colors[plot_idx % len(
                            colors)]
                        plot.PlotLine.style.val = veusz_line_style
                        plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                        plot.marker.val = veusz_marker
                        plot.markerSize.val = f'{plot_config.marker_size}pt'
                        plot.key.val = y_col

                    plot_idx += 1

                # Configure individual graph
                with self._wrap_widget(file_graph) as g:
                    g.Add('label', name='plotTitle')
                    g.plotTitle.Text.size.val = '12pt'
                    g.plotTitle.label.val = f"{base_filename}"
                    g.plotTitle.alignHorz.val = 'centre'
                    g.plotTitle.yPos.val = 1.05
                    g.plotTitle.xPos.val = 0.5
                    g.topMargin.val = '1cm'

                    g.x.label.val = plot_config.x_label
                    g.y.label.val = plot_config.y_label
                    g.x.log.val = (plot_config.x_scale == 'log')
                    g.y.log.val = (plot_config.y_scale == 'log')
                    g.x.GridLines.hide.val = not plot_config.show_grid
                    g.y.GridLines.hide.val = not plot_config.show_grid

                    if plot_config.show_legend:
                        key = g.Add('key')
                        key.hide.val = False  # FIXED: Boolean False, not string "False"
                        key.horzPosn.val = 'right'
                        key.vertPosn.val = 'top'

            num_files = len(data_dict)
            total_pages = 1 + num_files
            self.log_message(
                f"Created {total_pages} pages: 1 overlay + {num_files} individual file pages")

            return True

        except Exception as e:
            msg = f"Error creating plots: {e}"
            self.log_message(msg)
            print(msg)
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
        Save Veusz project to HDF5 file (.vszh5).

        Parameters:
            file_path (str): Path where to save the Veusz project.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.doc is None:
            return False

        try:
            # Ensure .vszh5 extension
            if not file_path.endswith('.vszh5'):
                file_path = file_path.rsplit('.', 1)[0] + '.vszh5'

            self.doc.Save(file_path, 'hdf5')
            msg = f"Saved Veusz project: {file_path}"
            self.log_message(msg)
            return True
        except Exception as e:
            msg = f"Error saving Veusz project: {e}"
            self.log_message(msg)
            print(msg)
            return False

    def export_plot(self, file_path: str, format_type: str = 'png') -> bool:
        """Export plot to image file."""
        if self.doc is None:
            return False

        try:
            pages = [c for c in self.doc.Root.children if c.typename == 'page']
            if pages:
                page = pages[0]
                self.doc.Export(file_path, format=format_type, page=page.name)
                return True
        except Exception as e:
            print(f"Error exporting plot: {e}")
            return False

    def open_gui(self, file_path: str = None):
        """Open the Veusz GUI window."""
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

    def close_log(self):
        """Close log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class CSVAutoPlotMainWindow(QMainWindow):
    """Main application window for CSV/TSV AutoPlot V6."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("CSV/TSV AutoPlot V6 - Data Visualization Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Data storage
        self.selected_files = []
        self.loaded_data = {}  # filename -> DataFrame
        self.metadata_dict = {}  # filename -> metadata lines

        # Configuration
        self.csv_config = CSVProcessingConfig()
        self.plot_config = PlotConfig()

        # Veusz plotter instance
        self.plotter = None

        # Setup UI
        self.setup_ui()
        self.log_message("CSV/TSV AutoPlot V6 initialized successfully")
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
        self.file_list_widget.itemSelectionChanged.connect(
            self.on_file_selection_changed)
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

        # Data start line (header row line number)
        self.data_start_line_spinbox = QSpinBox()
        self.data_start_line_spinbox.setMinimum(1)
        self.data_start_line_spinbox.setMaximum(10000)
        self.data_start_line_spinbox.setValue(
            1)  # Default: line 1 (0-indexed as 0)
        self.data_start_line_spinbox.setMaximumWidth(100)
        self.data_start_line_spinbox.setToolTip(
            "Line number where column headers exist (1-indexed)")
        self.data_start_line_spinbox.valueChanged.connect(
            self.on_data_start_line_changed)
        delimiter_form.addRow("Header Line Number:",
                              self.data_start_line_spinbox)

        # Logging checkbox
        self.enable_logging_checkbox = QCheckBox("Enable Logging")
        self.enable_logging_checkbox.setChecked(True)
        self.enable_logging_checkbox.setToolTip(
            "Log missing data warnings to file")
        delimiter_form.addRow("", self.enable_logging_checkbox)

        # Preset buttons
        preset_layout = QHBoxLayout()
        for name, delim in [("Comma", ","), ("Tab", "\t"), ("Semicolon", ";"), ("Pipe", "|")]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, d=delim: self.set_delimiter(d))
            btn.setMaximumWidth(80)
            preset_layout.addWidget(btn)
        preset_layout.addStretch()

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
        self.x_column_combo.currentTextChanged.connect(
            self.on_plot_config_changed)
        selection_form.addRow(
            "X-Axis Column (common to all files):", self.x_column_combo)

        # X-axis validation warning label
        self.x_axis_warning_label = QLabel("")
        self.x_axis_warning_label.setStyleSheet(
            "color: red; font-weight: bold;")
        self.x_axis_warning_label.setWordWrap(True)
        selection_form.addRow("", self.x_axis_warning_label)

        y_label = QLabel("Y-Axis Columns (select multiple - all available):")
        self.y_columns_list = QListWidget()
        self.y_columns_list.setSelectionMode(
            self.y_columns_list.SelectionMode.MultiSelection
        )
        self.y_columns_list.itemSelectionChanged.connect(
            self.on_plot_config_changed)
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
        self.show_legend_checkbox.stateChanged.connect(
            self.on_plot_config_changed)
        plot_layout.addRow(self.show_legend_checkbox)

        main_layout.addWidget(plot_group)

        # ====== ACTION BUTTONS ======
        button_group = QGroupBox("")
        button_layout = QHBoxLayout(button_group)

        self.generate_btn = QPushButton("Generate Plots in Veusz")
        self.generate_btn.clicked.connect(self.generate_plots)
        self.generate_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")

        save_veusz_btn = QPushButton("Save to Veusz (.vszh5)")
        save_veusz_btn.clicked.connect(self.save_to_veusz)
        save_veusz_btn.setToolTip("Save Veusz project in HDF5 format")

        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(save_veusz_btn)
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
        file_dialog.setNameFilter(
            "Delimited Files (*.csv *.tsv *.txt);;All Files (*)")
        file_dialog.setWindowTitle("Select CSV/TSV Files")

        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self.load_files()

    def load_files(self):
        """Load selected files into memory."""
        delimiter = self.delimiter_input.text() or ','

        # Update config with current header line setting (convert 1-indexed to 0-indexed)
        self.csv_config.header_line = self.data_start_line_spinbox.value() - 1

        for file_path in self.selected_files:
            if file_path not in [os.path.join(d, f) for d, f in
                                 [os.path.split(self._get_file_path(name))
                                 for name in self.loaded_data.keys()]]:
                df, msg, metadata_lines = load_csv_file(
                    file_path, delimiter, self.csv_config)
                if df is not None:
                    filename = os.path.basename(file_path)
                    self.loaded_data[filename] = df
                    self.metadata_dict[filename] = metadata_lines
                    self.log_message(msg)
                else:
                    self.log_message(msg)

        self.update_file_list()
        self.update_column_selectors()

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
        self.metadata_dict.clear()
        self.update_file_list()
        self.update_column_selectors()
        self.log_message("File list cleared")

    def on_file_selection_changed(self):
        """Handle file selection change."""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            filename = current_item.text()
            self.display_data_preview(filename)
            self.log_message(f"Selected: {filename}")

    def update_column_selectors(self):
        """
        Update column selection dropdowns based on ALL loaded files.
        X-axis: Shows only COMMON columns (intersection).
        Y-axis: Shows ALL columns (union).
        """
        self.x_column_combo.clear()
        self.y_columns_list.clear()

        if not self.loaded_data:
            return

        # Get common columns for X-axis (intersection)
        common_cols = get_common_columns(self.loaded_data)
        common_cols_sorted = sorted(list(common_cols))

        # Get all columns for Y-axis (union)
        all_cols = get_all_columns(self.loaded_data)
        all_cols_sorted = sorted(list(all_cols))

        # Populate X-axis combo with common columns only
        if common_cols_sorted:
            self.x_column_combo.addItems(common_cols_sorted)
            self.x_column_combo.setCurrentIndex(0)
            self.log_message(
                f"X-axis options: {len(common_cols_sorted)} common columns")
        else:
            self.log_message(
                "WARNING: No common columns found across all files!")

        # Populate Y-axis list with all columns
        if all_cols_sorted:
            self.y_columns_list.addItems(all_cols_sorted)
            self.log_message(
                f"Y-axis options: {len(all_cols_sorted)} total columns")

        # Validate X-axis selection
        self.validate_x_axis_selection()

    def validate_x_axis_selection(self):
        """
        Validate that selected X-axis column exists in ALL loaded files.
        Update button state and warning message accordingly.
        """
        x_column = self.x_column_combo.currentText()

        if not x_column or not self.loaded_data:
            self.generate_btn.setEnabled(False)
            self.generate_btn.setStyleSheet(
                "background-color: gray; color: white; font-weight: bold; padding: 8px;")
            self.x_axis_warning_label.setText("")
            return

        # Check if X column exists in all files
        missing_files = []
        for filename, df in self.loaded_data.items():
            if x_column not in df.columns:
                missing_files.append(filename)

        if missing_files:
            # X column NOT in all files - RED button, disabled
            self.generate_btn.setEnabled(False)
            self.generate_btn.setStyleSheet(
                "background-color: #DC143C; color: white; font-weight: bold; padding: 8px;")

            # Display RED warning message in all caps
            warning_msg = f"WARNING: NOT ALL LOADED FILES CONTAIN THE SELECTED X AXIS COLUMN '{x_column}'"
            self.x_axis_warning_label.setText(warning_msg)
            self.log_message(f"X-axis validation FAILED: {warning_msg}")
            self.log_message(f"Missing from: {', '.join(missing_files)}")
        else:
            # X column exists in all files - GREEN button, enabled
            self.generate_btn.setEnabled(True)
            self.generate_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
            self.x_axis_warning_label.setText("")
            self.log_message(
                f"X-axis validation PASSED: '{x_column}' exists in all {len(self.loaded_data)} files")

    def display_data_preview(self, filename: str):
        """Display a preview of the loaded data in the table."""
        if filename not in self.loaded_data:
            return

        df = self.loaded_data[filename]
        preview_df = df.head(10)

        self.data_preview_table.setRowCount(preview_df.shape[0])
        self.data_preview_table.setColumnCount(preview_df.shape[1])
        self.data_preview_table.setHorizontalHeaderLabels(preview_df.columns)

        for i, row in enumerate(preview_df.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_preview_table.setItem(i, j, item)

        self.data_preview_table.resizeColumnsToContents()

    def auto_detect_delimiter(self):
        """Auto-detect delimiter from currently selected file."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No File Selected",
                                "Please select a file first")
            return

        filename = current_item.text()
        file_path = self._get_file_path(filename)

        if file_path:
            detected_delim = detect_delimiter(file_path)
            self.delimiter_input.setText(detected_delim)
            self.log_message(
                f"Auto-detected delimiter: {repr(detected_delim)}")

    def set_delimiter(self, delimiter: str):
        """Set the delimiter and reload files."""
        self.delimiter_input.setText(delimiter)
        self.loaded_data.clear()
        self.metadata_dict.clear()
        self.load_files()
        self.log_message(f"Delimiter set to: {repr(delimiter)}")

    def on_delimiter_changed(self):
        """Handle delimiter change."""
        self.loaded_data.clear()
        self.metadata_dict.clear()
        self.load_files()

    def on_data_start_line_changed(self):
        """Handle data start line change."""
        self.loaded_data.clear()
        self.metadata_dict.clear()
        self.load_files()
        self.log_message(
            f"Header line set to: {self.data_start_line_spinbox.value()}")

    def on_plot_config_changed(self):
        """Handle plot configuration change."""
        # Update config values
        self.plot_config.title = self.title_input.text()
        self.plot_config.x_label = self.x_label_input.text()
        self.plot_config.y_label = self.y_label_input.text()
        self.plot_config.show_legend = self.show_legend_checkbox.isChecked()

        # Validate X-axis selection whenever config changes
        self.validate_x_axis_selection()

    def generate_plots(self):
        """Generate plots in Veusz."""
        if not self.loaded_data:
            QMessageBox.warning(
                self, "No Data", "Please load CSV/TSV files first")
            return

        x_column = self.x_column_combo.currentText()
        y_columns = [item.text()
                     for item in self.y_columns_list.selectedItems()]

        if not x_column or not y_columns:
            QMessageBox.warning(self, "No Columns Selected",
                                "Please select X and Y columns")
            return

        # Final validation check
        missing_files = []
        for filename, df in self.loaded_data.items():
            if x_column not in df.columns:
                missing_files.append(filename)

        if missing_files:
            QMessageBox.critical(self, "Invalid X-Axis Selection",
                                 f"X-axis column '{x_column}' not found in the following files:\n" +
                                 "\n".join(missing_files))
            return

        try:
            # Create plotter instance
            self.plotter = VeuszPlotter(plot_title=self.plot_config.title)

            # Setup logging if enabled
            if self.enable_logging_checkbox.isChecked():
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = f"CSV_AutoPlot_V6_Log_{timestamp}.txt"
                self.plotter.set_logging(True, log_path)
                self.log_message(f"Logging enabled: {log_path}")

            # Generate overlay and individual plots
            success = self.plotter.create_overlay_and_individual_plots(
                self.loaded_data,
                self.metadata_dict,
                x_column,
                y_columns,
                self.plot_config
            )

            if success:
                num_files = len(self.loaded_data)
                total_pages = 1 + num_files
                self.log_message("Plots created successfully in Veusz")
                QMessageBox.information(self, "Success",
                                        f"Overlay plot and individual plots created successfully!\n"
                                        f"Total pages: {total_pages} (1 overlay + {num_files} individual file pages)\n"
                                        f"Each file uses its own independent X-axis data.\n"
                                        "Use 'Save to Veusz' button to save the project.")
            else:
                QMessageBox.critical(
                    self, "Error", "Failed to create plots in Veusz")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Error generating plots: {str(e)}")
            self.log_message(f"Plot generation error: {str(e)}")
            traceback.print_exc()

    def save_to_veusz(self):
        """Save current Veusz project to .vszh5 file."""
        if self.plotter is None or self.plotter.doc is None:
            QMessageBox.warning(self, "No Plot",
                                "Please generate plots first before saving")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Veusz Project",
            "", "Veusz HDF5 (*.vszh5);;All Files (*)"
        )

        if file_path:
            try:
                success = self.plotter.save_project(file_path)
                if success:
                    self.log_message(f"Saved Veusz project: {file_path}")
                    QMessageBox.information(self, "Success",
                                            f"Veusz project saved successfully:\n{file_path}")

                    # Ask to open in Veusz
                    reply = QMessageBox.question(
                        self, "Open in Veusz",
                        "Would you like to open the file in Veusz?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        self.plotter.open_gui(file_path)
                else:
                    QMessageBox.critical(
                        self, "Error", "Failed to save Veusz project")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving: {str(e)}")
                self.log_message(f"Save error: {str(e)}")

    def log_message(self, message: str):
        """Add message to status log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.plotter:
            self.plotter.close_log()
        event.accept()


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
