"""
CSV/TSV AutoPlot - Automatic data visualization with Veusz Embedded API.

This script provides a complete GUI application for loading CSV, TSV, and other
delimited text files, configuring custom delimiters, and generating publication-
quality plots using Veusz's embedded API (separate window with toolbar).

FIXED VERSION (2026-02-03):
- Corrected Veusz API calls in VeuszPlotter.create_xy_plot() method
- Line style mapping: matplotlib -> Veusz format
- Graph properties: Using correct Veusz API syntax
- Matplotlib preview canvas REMOVED (Veusz-only plotting)
- Added data start line field for header row specification

Key Features:
- Support for arbitrary delimiter characters/strings (comma, tab, semicolon, etc.)
- Automatic delimiter detection
- Configurable data start line (header row line number)
- Interactive data table preview
- Multi-column selection for X and Y axes
- Veusz plot window with full toolbar (separate window)
- Save plots as .vszh5 (Veusz HDF5 format)
- Configurable plot styles and formatting
- Comprehensive error handling and logging

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
python CSV_TSV_AutoPlot.py
"""

import datetime
import multiprocessing
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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


def load_csv_file(file_path: str, delimiter: str = ',',
                  config: CSVProcessingConfig = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV/TSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV/TSV file.
        delimiter (str): Delimiter character or string.
        config (CSVProcessingConfig): Processing configuration including header_line.
    
    Returns:
        Tuple[DataFrame or None, str]: Loaded DataFrame and status message.
    """
    if config is None:
        config = CSVProcessingConfig()
    
    try:
        # Calculate skiprows: skip all lines before header_line
        skiprows = list(range(config.header_line)) if config.header_line > 0 else None
        
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=config.encoding,
            skiprows=skiprows,
            skip_blank_lines=config.skip_empty_lines,
            engine='python'
        )
        status_msg = f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns (header at line {config.header_line + 1})"
        return df, status_msg
    except Exception as e:
        error_msg = f"Error loading {file_path}: {str(e)}"
        return None, error_msg


def load_multiple_csv_files(file_paths: List[str], delimiter: str = ',',
                            config: CSVProcessingConfig = None) -> Dict[str, pd.DataFrame]:
    """Load multiple CSV/TSV files."""
    results = {}
    for file_path in file_paths:
        df, status_msg = load_csv_file(file_path, delimiter, config)
        if df is not None:
            filename = os.path.basename(file_path)
            results[filename] = df
    return results


def infer_numeric_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """Infer which columns in a DataFrame are numeric."""
    numeric_status = {}
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='coerce')
            non_null_count = df[col].notna().sum()
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
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
    
    def __init__(self, plot_title: str = "CSV Data Plot",
                 dataset_name: str = "CSVDataset"):
        """Initialize Veusz plotter."""
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
            df: Input DataFrame
            x_column: Name of column to use for X axis
            y_columns: Names of columns to use for Y axes
            plot_config: Plot configuration
        
        Returns:
            bool: True if successful, False otherwise
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
                
                # FIXED: Convert matplotlib styles to Veusz styles
                veusz_line_style = self.LINE_STYLE_MAP.get(plot_config.line_style, 'solid')
                veusz_marker = self.MARKER_STYLE_MAP.get(plot_config.marker_style, 'circle')
                
                # Configure appearance using correct Veusz API
                with self._wrap_widget(xy_plot) as plot:
                    plot.PlotLine.color.val = colors[idx % len(colors)]
                    plot.PlotLine.style.val = veusz_line_style  # FIXED: Use mapped style
                    plot.PlotLine.width.val = f'{plot_config.line_width}pt'
                    plot.marker.val = veusz_marker  # FIXED: Use mapped marker
                    plot.markerSize.val = f'{plot_config.marker_size}pt'
            
            # FIXED: Configure graph using correct Veusz API (from ATR_AutoPlot.py)
            with self._wrap_widget(graph) as g:
                # Add title as a label widget (not a direct property)
                g.Add('label', name='plotTitle')
                g.plotTitle.Text.size.val = '12pt'
                g.plotTitle.label.val = plot_config.title
                g.plotTitle.alignHorz.val = 'centre'
                g.plotTitle.yPos.val = 1.05
                g.plotTitle.xPos.val = 0.5
                g.topMargin.val = '1cm'
                
                # FIXED: Access axis labels through axis objects
                g.x.label.val = plot_config.x_label
                g.y.label.val = plot_config.y_label
                
                # FIXED: Access log scale through axis objects
                g.x.log.val = (plot_config.x_scale == 'log')
                g.y.log.val = (plot_config.y_scale == 'log')
                
                # FIXED: Grid lines
                g.x.GridLines.hide.val = not plot_config.show_grid
                g.y.GridLines.hide.val = not plot_config.show_grid
                
                # FIXED: Legend (create as widget if needed)
                if plot_config.show_legend:
                    key = g.Add('key')
                    key.hide.val = False
                    # Map matplotlib position to Veusz position
                    position_map = {
                        'best': 'auto',
                        'upper right': 'tr',
                        'upper left': 'tl',
                        'lower right': 'br',
                        'lower left': 'bl',
                        'right': 'r',
                        'center': 'c'
                    }
                    key.horzPosn.val = position_map.get(plot_config.legend_position, 'tr')
            
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
        """Save Veusz project to file."""
        if self.doc is None:
            return False
        
        try:
            self.doc.Save(file_path)
            return True
        except Exception as e:
            print(f"Error saving Veusz project: {e}")
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


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class CSVAutoPlotMainWindow(QMainWindow):
    """Main application window for CSV/TSV AutoPlot."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("CSV/TSV AutoPlot - Data Visualization Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.selected_files = []
        self.loaded_data = {}
        
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
        
        # Data start line (header row line number)
        self.data_start_line_spinbox = QSpinBox()
        self.data_start_line_spinbox.setMinimum(1)
        self.data_start_line_spinbox.setMaximum(10000)
        self.data_start_line_spinbox.setValue(1)  # Default: line 1 (0-indexed as 0)
        self.data_start_line_spinbox.setMaximumWidth(100)
        self.data_start_line_spinbox.setToolTip("Line number where column headers exist (1-indexed)")
        self.data_start_line_spinbox.valueChanged.connect(self.on_data_start_line_changed)
        delimiter_form.addRow("Header Line Number:", self.data_start_line_spinbox)
        
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
        self.x_column_combo.currentTextChanged.connect(self.on_plot_config_changed)
        selection_form.addRow("X-Axis Column:", self.x_column_combo)
        
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
        
        # ====== ACTION BUTTONS ======
        button_group = QGroupBox("")
        button_layout = QHBoxLayout(button_group)
        
        generate_btn = QPushButton("Generate Plots in Veusz")
        generate_btn.clicked.connect(self.generate_plots)
        
        button_layout.addWidget(generate_btn)
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
        
        # Update config with current header line setting (convert 1-indexed to 0-indexed)
        self.csv_config.header_line = self.data_start_line_spinbox.value() - 1
        
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
            
            if len(columns) > 0:
                self.x_column_combo.setCurrentIndex(0)
    
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
        self.loaded_data.clear()
        self.load_files()
    
    def on_data_start_line_changed(self):
        """Handle data start line change."""
        self.loaded_data.clear()
        self.load_files()
        self.log_message(f"Header line set to: {self.data_start_line_spinbox.value()}")
    
    def on_plot_config_changed(self):
        """Handle plot configuration change."""
        # Update config values
        self.plot_config.title = self.title_input.text()
        self.plot_config.x_label = self.x_label_input.text()
        self.plot_config.y_label = self.y_label_input.text()
        self.plot_config.show_legend = self.show_legend_checkbox.isChecked()
    
    def generate_plots(self):
        """Generate plots in Veusz."""
        if not self.loaded_data:
            QMessageBox.warning(self, "No Data", "Please load CSV/TSV files first")
            return
        
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
            
            base_filename = os.path.splitext(filename)[0]
            plotter = VeuszPlotter(
                plot_title=self.plot_config.title,
                dataset_name=f"{base_filename}_Data"
            )
            
            success = plotter.create_xy_plot(df, x_column, y_columns, self.plot_config)
            
            if success:
                project_path = os.path.join(output_dir, f"{base_filename}_plot.vszh5")
                plotter.save_project(project_path)
                self.log_message(f"Plot created successfully: {project_path}")
                
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
