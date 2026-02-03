"""
CSV/TSV AutoPlot - Automatic data visualization with Embedded Veusz Window.

This script provides a complete GUI application for loading CSV, TSV, and other
delimited text files, configuring custom delimiters, and generating publication-
quality plots using an EMBEDDED Veusz window with toolbar.

NEW FEATURES IN THIS VERSION:
- Embedded Veusz window directly in GUI
- Full Veusz toolbar access
- Save Veusz files button (HDF5 format)
- Support for metadata/header lines
- Data start line configuration
- Dual preview panes (metadata + data)

Key Features:
- Support for arbitrary delimiter characters/strings (comma, tab, semicolon, etc.)
- Automatic delimiter detection
- Handle files with metadata headers
- Interactive preview of loaded data
- Multi-column selection for X and Y axes
- Embedded Veusz plot window with toolbar
- Save plots as .vsz (Veusz HDF5 format)
- Configurable plot styles and formatting
- Comprehensive error handling and logging

Typical Workflow:
1. Launch the application
2. Browse and select CSV/TSV files
3. Configure delimiter (auto-detect or manual)
4. Set data start line (where actual data begins)
5. Preview metadata and data
6. Select columns for plotting
7. Configure plot options (title, labels, legend, etc.)
8. Generate plots in embedded Veusz window
9. Save Veusz file for later use

Author: Based on William W. Wallace's framework
Last Updated: 2026-02-03
Python Version: 3.8+

Dependencies:
- PyQt5/PySide6 (GUI framework)
- pandas (data loading and manipulation)
- numpy (numerical computing)
- matplotlib (preview plots)
- veusz (final plot generation with embedding)
- scipy (optional, for advanced analysis)

Installation:
    pip install pandas numpy matplotlib veusz pyside6

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
    data_start_line: int = 1  # NEW: Line where data begins
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

        status_msg = f"Successfully loaded {os.path.basename(file_path)}: {df.shape[0]} rows, {df.shape[1]} columns"
        if metadata_lines:
            status_msg += f" (skipped {len(metadata_lines)} metadata lines)"

        return df, status_msg, metadata_lines

    except Exception as e:
        error_msg = f"Error loading {file_path}: {str(e)}"
        return None, error_msg, []

# ============================================================================
# VEUSZ PLOTTING CLASS WITH EMBEDDING SUPPORT
# ============================================================================

class VeuszPlotter:
    """
    Handles plot generation in Veusz for CSV/TSV data with embedding support.

    This class wraps the Veusz embedding API to create publication-quality
    plots from pandas DataFrames and embeds the Veusz window in the GUI.
    """

    def __init__(self, plot_title: str = "CSV Data Plot",
                 dataset_name: str = "CSVDataset"):
        """
        Initialize Veusz plotter for embedding.

        Parameters:
            plot_title (str): Title for the plot window.
            dataset_name (str): Base name for datasets in Veusz.
        """
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.doc = None
        self.veusz_widget = None

        try:
            if vz is not None:
                # Create embedded document (hidden initially until makeWindow called)
                self.doc = vz.Embedded(plot_title, hidden=True)
                self.doc.EnableToolbar()
        except Exception as e:
            print(f"Warning: Could not initialize Veusz: {e}")

    def get_embed_widget(self):
        """
        Get Qt widget for embedding in main window.

        Returns:
            QWidget: Veusz window widget for embedding.
        """
        if self.doc is None:
            return None

        try:
            if self.veusz_widget is None:
                self.veusz_widget = self.doc.makeWindow()
            return self.veusz_widget
        except Exception as e:
            print(f"Error creating Veusz widget: {e}")
            traceback.print_exc()
            return None

    def create_xy_plot(self, df: pd.DataFrame, x_column: str, y_columns: List[str],
                      plot_config: PlotConfig = None, filename_tag: str = "",
                      metadata: List[str] = None) -> bool:
        """
        Create an XY plot in Veusz from DataFrame columns.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            x_column (str): Name of column to use for X axis.
            y_columns (List[str]): Names of columns to use for Y axes.
            plot_config (PlotConfig, optional): Plot configuration.
            filename_tag (str, optional): Filename tag for plot identification.
            metadata (List[str], optional): Metadata lines to include in plot.

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

            # Add metadata as text annotation if provided
            if metadata and len(metadata) > 0:
                metadata_text = "\n".join(metadata[:15])  # Limit to first 15 lines
                text_widget = page.Add('label', name='Metadata')
                text_widget.label.val = metadata_text
                text_widget.xPos.val = 0.05
                text_widget.yPos.val = 0.95
                text_widget.Text.size.val = '8pt'
                text_widget.Text.color.val = 'gray'

            # Add data to Veusz
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
                xy_plot.PlotLine.color.val = colors[idx % len(colors)]
                xy_plot.PlotLine.style.val = plot_config.line_style
                xy_plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                xy_plot.marker.val = plot_config.marker_style
                xy_plot.markerSize.val = f'{plot_config.marker_size}pt'

            # Configure graph
            graph.title.val = plot_config.title
            graph.xlabel.val = plot_config.x_label
            graph.ylabel.val = plot_config.y_label
            graph.xLog.val = (plot_config.x_scale == 'log')
            graph.yLog.val = (plot_config.y_scale == 'log')
            graph.showLegend.val = plot_config.show_legend
            if plot_config.show_legend:
                graph.legend.pos.val = plot_config.legend_position

            return True

        except Exception as e:
            print(f"Error creating XY plot: {e}")
            traceback.print_exc()
            return False

    def save_project(self, file_path: str) -> bool:
        """
        Save Veusz project to file in HDF5 format (.vsz).

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

        x_data = pd.to_numeric(df[x_column], errors='coerce').dropna()

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for idx, y_col in enumerate(y_columns):
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
    Main application window for CSV/TSV AutoPlot with embedded Veusz.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("CSV/TSV AutoPlot - Embedded Veusz Window")
        self.setGeometry(100, 100, 1600, 1000)

        self.selected_files = []
        self.loaded_data = {}  # Dict[filename, (DataFrame, metadata)]

        self.csv_config = CSVProcessingConfig()
        self.plot_config = PlotConfig()

        self.veusz_plotter = None

        self.setup_ui()

        self.log_message("CSV/TSV AutoPlot with Embedded Veusz initialized")
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

        # Right side: Delimiter and Data Start Line configuration
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

        # Add preset layout
        preset_widget = QWidget()
        preset_widget.setLayout(preset_layout)
        right_layout.insertWidget(1, preset_widget)

        right_layout.addStretch()

        file_layout.addLayout(left_layout, 1)
        file_layout.addLayout(right_layout, 0)

        main_layout.addWidget(file_group)

        # ====== DUAL PREVIEW AREA (Metadata + Data) ======
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Metadata preview
        preview_layout.addWidget(QLabel("Notes/Metadata (lines before data start):"))
        self.metadata_preview = QTextEdit()
        self.metadata_preview.setReadOnly(True)
        self.metadata_preview.setMaximumHeight(150)
        self.metadata_preview.setStyleSheet("QTextEdit { background-color: #f0f0f0; font-family: monospace; }")
        preview_layout.addWidget(self.metadata_preview)

        # Data preview
        preview_layout.addWidget(QLabel("Data Preview (first 13 rows):"))
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setMaximumHeight(250)
        preview_layout.addWidget(self.data_preview_table)

        main_layout.addWidget(preview_group)

        # ====== COLUMN SELECTION FOR PLOTTING ======
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

        # ====== MATPLOTLIB PREVIEW ======
        self.preview_canvas = PreviewCanvas(width=8, height=3)
        main_layout.addWidget(QLabel("Matplotlib Preview:"))
        main_layout.addWidget(self.preview_canvas)

        # ====== VEUSZ EMBEDDED WINDOW AREA ======
        veusz_group = QGroupBox("Veusz Plot Window (Embedded with Toolbar)")
        veusz_layout = QVBoxLayout(veusz_group)

        # Container for Veusz widget
        self.veusz_container = QWidget()
        self.veusz_container_layout = QVBoxLayout(self.veusz_container)
        self.veusz_container_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder label
        self.veusz_placeholder = QLabel("📊 Generate plots to see Veusz window here with full toolbar")
        self.veusz_placeholder.setAlignment(Qt.AlignCenter)
        self.veusz_placeholder.setMinimumHeight(400)
        font = QFont()
        font.setPointSize(14)
        self.veusz_placeholder.setFont(font)
        self.veusz_placeholder.setStyleSheet(
            "QLabel { background-color: #f8f8f8; border: 3px dashed #999; color: #666; }"
        )
        self.veusz_container_layout.addWidget(self.veusz_placeholder)

        veusz_layout.addWidget(self.veusz_container)

        # Veusz control buttons
        veusz_button_layout = QHBoxLayout()

        self.generate_veusz_btn = QPushButton("Generate Plots in Embedded Veusz Window")
        self.generate_veusz_btn.clicked.connect(self.generate_plots)
        self.generate_veusz_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")

        self.save_veusz_btn = QPushButton("Save Veusz File (HDF5 .vsz)")
        self.save_veusz_btn.clicked.connect(self.save_veusz_file)
        self.save_veusz_btn.setEnabled(False)

        self.clear_veusz_btn = QPushButton("Clear Veusz Window")
        self.clear_veusz_btn.clicked.connect(self.clear_veusz_window)
        self.clear_veusz_btn.setEnabled(False)

        veusz_button_layout.addWidget(self.generate_veusz_btn)
        veusz_button_layout.addWidget(self.save_veusz_btn)
        veusz_button_layout.addWidget(self.clear_veusz_btn)
        veusz_button_layout.addStretch()

        veusz_layout.addLayout(veusz_button_layout)

        main_layout.addWidget(veusz_group)

        # ====== STATUS/LOGGING AREA ======
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
        # Handle tab escape
        if delimiter == '\t':
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
        self.metadata_preview.clear()
        self.data_preview_table.clear()
        self.log_message("File list cleared")

    def on_file_selection_changed(self):
        """Handle file selection change."""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            filename = current_item.text()
            self.update_column_selectors(filename)
            self.display_data_preview(filename)
            self.display_metadata_preview(filename)
            self.log_message(f"Selected: {filename}")

    # ========================================================================
    # DELIMITER AND DATA START LINE METHODS
    # ========================================================================

    def auto_detect_delimiter(self):
        """Auto-detect delimiter from currently selected file."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No File Selected", "Please select a file first")
            return

        filename = current_item.text()
        file_path = self._get_file_path(filename)

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

    def _get_file_path(self, filename: str) -> str:
        """Get full file path from filename."""
        for file_path in self.selected_files:
            if os.path.basename(file_path) == filename:
                return file_path
        return ""

    # ========================================================================
    # PREVIEW METHODS
    # ========================================================================

    def update_column_selectors(self, filename: str = None):
        """Update column selection dropdowns."""
        self.x_column_combo.clear()
        self.y_columns_list.clear()

        if filename and filename in self.loaded_data:
            df, _ = self.loaded_data[filename]
            columns = df.columns.tolist()
            self.x_column_combo.addItems(columns)
            self.y_columns_list.addItems(columns)

            if len(columns) > 0:
                self.x_column_combo.setCurrentIndex(0)

    def display_metadata_preview(self, filename: str):
        """Display metadata lines in preview pane."""
        if filename not in self.loaded_data:
            return

        _, metadata = self.loaded_data[filename]

        if metadata:
            # Show first 13 lines of metadata
            preview_lines = metadata[:13]
            self.metadata_preview.setText("\n".join(preview_lines))
            if len(metadata) > 13:
                self.metadata_preview.append(f"\n... ({len(metadata) - 13} more lines)")
        else:
            self.metadata_preview.setText("(No metadata - data starts at line 1)")

    def display_data_preview(self, filename: str):
        """Display a preview of the loaded data in the table."""
        if filename not in self.loaded_data:
            return

        df, _ = self.loaded_data[filename]

        # Show first 13 rows
        preview_df = df.head(13)

        self.data_preview_table.setRowCount(preview_df.shape[0])
        self.data_preview_table.setColumnCount(preview_df.shape[1])
        self.data_preview_table.setHorizontalHeaderLabels(preview_df.columns)

        for i, row in enumerate(preview_df.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_preview_table.setItem(i, j, item)

        self.data_preview_table.resizeColumnsToContents()

    # ========================================================================
    # PLOT CONFIGURATION AND PREVIEW
    # ========================================================================

    def on_plot_config_changed(self):
        """Handle plot configuration change."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            return

        filename = current_item.text()
        if filename not in self.loaded_data:
            return

        df, _ = self.loaded_data[filename]

        x_column = self.x_column_combo.currentText()
        y_columns = [item.text() for item in self.y_columns_list.selectedItems()]

        if not x_column or not y_columns:
            return

        # Update plot config
        self.plot_config.title = self.title_input.text()
        self.plot_config.x_label = self.x_label_input.text()
        self.plot_config.y_label = self.y_label_input.text()
        self.plot_config.show_legend = self.show_legend_checkbox.isChecked()

        # Update matplotlib preview
        try:
            self.preview_canvas.plot_data(df, x_column, y_columns, self.plot_config)
        except Exception as e:
            self.log_message(f"Error updating preview: {str(e)}")

    # ========================================================================
    # VEUSZ PLOT GENERATION AND EMBEDDING
    # ========================================================================

    def generate_plots(self):
        """Generate plots in embedded Veusz window."""
        if not self.loaded_data:
            QMessageBox.warning(self, "No Data", "Please load CSV/TSV files first")
            return

        try:
            current_item = self.file_list_widget.currentItem()
            if not current_item:
                QMessageBox.warning(self, "No File Selected", "Please select a file")
                return

            filename = current_item.text()
            df, metadata = self.loaded_data[filename]

            x_column = self.x_column_combo.currentText()
            y_columns = [item.text() for item in self.y_columns_list.selectedItems()]

            if not x_column or not y_columns:
                QMessageBox.warning(self, "No Columns Selected",
                                  "Please select X and Y columns")
                return

            # Create Veusz plotter if not exists
            if self.veusz_plotter is None:
                base_filename = os.path.splitext(filename)[0]
                self.veusz_plotter = VeuszPlotter(
                    plot_title=self.plot_config.title,
                    dataset_name=f"{base_filename}_Data"
                )

            # Create XY plot with metadata
            success = self.veusz_plotter.create_xy_plot(
                df, x_column, y_columns,
                self.plot_config,
                filename_tag=filename,
                metadata=metadata
            )

            if success:
                # Get Veusz widget and embed it
                veusz_widget = self.veusz_plotter.get_embed_widget()

                if veusz_widget:
                    # Remove placeholder if it exists
                    if self.veusz_placeholder:
                        self.veusz_container_layout.removeWidget(self.veusz_placeholder)
                        self.veusz_placeholder.deleteLater()
                        self.veusz_placeholder = None

                    # Add Veusz widget to container
                    self.veusz_container_layout.addWidget(veusz_widget)

                    # Enable save/clear buttons
                    self.save_veusz_btn.setEnabled(True)
                    self.clear_veusz_btn.setEnabled(True)

                    self.log_message("✓ Plot generated successfully in embedded Veusz window")
                    QMessageBox.information(self, "Success",
                                          "Plot generated in Veusz window below.\n\n" +
                                          "Use the Veusz toolbar to zoom, pan, and customize the plot.")
                else:
                    QMessageBox.critical(self, "Error", "Failed to create Veusz widget")
            else:
                QMessageBox.critical(self, "Error", "Failed to create plot in Veusz")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating plots: {str(e)}")
            self.log_message(f"✗ Plot generation error: {str(e)}")
            traceback.print_exc()

    def save_veusz_file(self):
        """Save the current Veusz plot to HDF5 file."""
        if self.veusz_plotter is None or self.veusz_plotter.doc is None:
            QMessageBox.warning(self, "No Plot", "Please generate a plot first")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Veusz File",
            "",
            "Veusz HDF5 Files (*.vsz);;All Files (*)"
        )

        if not file_path:
            return

        # Ensure .vsz extension
        if not file_path.lower().endswith('.vsz'):
            file_path += '.vsz'

        try:
            success = self.veusz_plotter.save_project(file_path)

            if success:
                self.log_message(f"✓ Veusz file saved: {file_path}")
                QMessageBox.information(self, "Success",
                                      f"Veusz file saved successfully:\n\n{file_path}\n\n" +
                                      "You can open this file later in Veusz.")
            else:
                QMessageBox.critical(self, "Error", "Failed to save Veusz file")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving Veusz file: {str(e)}")
            self.log_message(f"✗ Save error: {str(e)}")

    def clear_veusz_window(self):
        """Clear the embedded Veusz window."""
        if self.veusz_plotter is None:
            return

        try:
            # Get current widget and remove it
            if self.veusz_plotter.veusz_widget:
                self.veusz_container_layout.removeWidget(self.veusz_plotter.veusz_widget)
                self.veusz_plotter.veusz_widget.deleteLater()
                self.veusz_plotter.veusz_widget = None

            # Reset plotter
            self.veusz_plotter = None

            # Restore placeholder
            self.veusz_placeholder = QLabel("📊 Generate plots to see Veusz window here with full toolbar")
            self.veusz_placeholder.setAlignment(Qt.AlignCenter)
            self.veusz_placeholder.setMinimumHeight(400)
            font = QFont()
            font.setPointSize(14)
            self.veusz_placeholder.setFont(font)
            self.veusz_placeholder.setStyleSheet(
                "QLabel { background-color: #f8f8f8; border: 3px dashed #999; color: #666; }"
            )
            self.veusz_container_layout.addWidget(self.veusz_placeholder)

            # Disable buttons
            self.save_veusz_btn.setEnabled(False)
            self.clear_veusz_btn.setEnabled(False)

            self.log_message("✓ Veusz window cleared")

        except Exception as e:
            self.log_message(f"✗ Error clearing Veusz window: {str(e)}")

    # ========================================================================
    # LOGGING
    # ========================================================================

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
