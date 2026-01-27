# -*- coding: utf-8 -*-

"""
=============================================================================
Enhanced R&S FSW ASCII Plotter with Batch Overlay/Average Save, Multiprocessing, and GPU Support

Author: William W. Wallace (Enhanced)
=============================================================================
"""

# Core/GUI Imports
from dataclasses import dataclass
from operator import itemgetter
from collections import defaultdict
import os
import re
import sys
import subprocess
import math
import threading
import datetime
import numpy as np
import gc
from functools import partial

# QtPy or PySide6 selection
if getattr(sys, 'frozen', False):
    os.environ['QT_API'] = 'pyside6'
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
        QRadioButton, QButtonGroup, QGroupBox, QTextEdit, QProgressBar,
        QCheckBox, QSpinBox, QListWidget, QLineEdit
    )
else:
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
        QRadioButton, QButtonGroup, QGroupBox, QTextEdit, QProgressBar,
        QCheckBox, QSpinBox, QListWidget, QLineEdit
    )

from fastest_ascii_import import fastest_file_parser as fparser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing as mp

# GPU
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pyopencl as cl

    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

import veusz.embed as embed


# --- Core Config Data Classes ---

@dataclass
class ProcessingConfig:
    enable_multiprocessing: bool = False
    enable_gpu_processing: bool = False
    use_opencl: bool = True
    num_processes: int = cpu_count()
    max_workers: int = cpu_count()
    chunk_size: int = 1000
    plot_mode: int = 0  # 0 = All, 1 = Overlay/Average, 2 = Average Only
    enable_auto_save: bool = False
    plots_per_file: int = 1000
    base_batch_filename: str = 'RnS_Plots_Batch'


@dataclass
class plotDescInfo:
    xAxis_label: str
    yAxis_label: str
    graph_notes: str
    graph_title: str
    base_name: str
    first_plot: bool


# --- GPU Processing ---

class GPUProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.gpu_available = False
        self.context = None
        self.queue = None
        if config.enable_gpu_processing:
            self._initialize_gpu()

    def _initialize_gpu(self):
        if self.config.use_opencl and PYOPENCL_AVAILABLE:
            self._initialize_opencl()
        elif CUPY_AVAILABLE:
            self._initialize_cupy()

    def _initialize_opencl(self):
        try:
            platforms = cl.get_platforms()
            if platforms:
                platform = platforms[0]
                devices = platform.get_devices(cl.device_type.GPU)
                if not devices:
                    devices = platform.get_devices(cl.device_type.CPU)
                if devices:
                    self.context = cl.Context(devices=[devices[0]])
                    self.queue = cl.CommandQueue(self.context)
                    self.gpu_available = True
        except Exception:
            pass

    def _initialize_cupy(self):
        try:
            cp.cuda.Device(0).use()
            self.gpu_available = True
        except Exception:
            pass

    def process_array_gpu(self, data_array):
        if not self.gpu_available:
            return data_array
        try:
            if self.config.use_opencl and self.context:
                return self._process_opencl(data_array)
            elif CUPY_AVAILABLE:
                return self._process_cupy(data_array)
        except Exception:
            return data_array
        return data_array

    def _process_opencl(self, data_array):
        mf = cl.mem_flags
        data_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=data_array.astype(np.float32))
        kernel_source = """
        __kernel void process_data(__global float* data) {
            int gid = get_global_id(0);
            data[gid] = data[gid] * 1.0f;
        }
        """
        program = cl.Program(self.context, kernel_source).build()
        kernel = program.process_data
        kernel(self.queue, (len(data_array),), None, data_buffer)
        result = np.empty_like(data_array, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, data_buffer)
        self.queue.finish()
        return result.astype(data_array.dtype)

    def _process_cupy(self, data_array):
        gpu_array = cp.asarray(data_array)
        return cp.asnumpy(gpu_array * 1.0)


# --- Multiprocessing Worker ---

def process_file_worker(file_info):
    filename, search_strings, sft_lines, config = file_info
    try:
        data_returned = fparser(filename, line_targets=sft_lines, string_patterns=search_strings)
        if config.enable_gpu_processing:
            gpu_processor = GPUProcessor(config)
            for key, data_match in data_returned['data_matches'].items():
                if 'extracted_value' in data_match:
                    data_array = np.array(data_match['extracted_value'])
                    processed_array = gpu_processor.process_array_gpu(data_array)
                    data_returned['data_matches'][key]['extracted_value'] = processed_array.tolist()
        return {
            'filename': filename,
            'data': data_returned,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'filename': filename,
            'data': None,
            'success': False,
            'error': str(e)
        }


def extract_with_regex(inputText: str, delim: str = ';'):
    esc = re.escape(delim)
    pattern = rf"{esc}(.*?){esc}"
    return re.findall(pattern, inputText)


# --- Qt GUI Classes ---

class FileProcessingThread(QThread):
    progress_updated = Signal(int)
    processing_finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, file_list, config, search_strings, sft_lines):
        super().__init__()
        self.file_list = file_list
        self.config = config
        self.search_strings = search_strings
        self.sft_lines = sft_lines

    def run(self):
        try:
            if self.config.enable_multiprocessing and len(self.file_list) > 1:
                results = self._process_files_parallel()
            else:
                results = self._process_files_sequential()
            self.processing_finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_files_parallel(self):
        file_info_list = [
            (filename, self.search_strings, self.sft_lines, self.config)
            for filename in self.file_list
        ]
        results = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(process_file_worker, file_info): file_info[0]
                for file_info in file_info_list
            }
            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1
                progress = int((completed / len(self.file_list)) * 100)
                self.progress_updated.emit(progress)
        return results

    def _process_files_sequential(self):
        results = []
        for i, filename in enumerate(self.file_list):
            file_info = (filename, self.search_strings, self.sft_lines, self.config)
            result = process_file_worker(file_info)
            results.append(result)
            progress = int(((i + 1) / len(self.file_list)) * 100)
            self.progress_updated.emit(progress)
        return results


class EnhancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced R&S SFT File Plotter")
        self.setGeometry(100, 100, 850, 700)
        self.config = ProcessingConfig()
        self.vzplot = VZPlotRnS(self.config)
        self._setup_ui()
        self.selected_files = []

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- File Selection ---
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)
        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.clicked.connect(self._browse_files)
        browse_layout.addWidget(self.browse_button)
        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        browse_layout.addWidget(self.clear_button)
        file_layout.addLayout(browse_layout)
        main_layout.addWidget(file_group)

        # --- Processing Options ---
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        self.enable_mp_checkbox = QCheckBox("Enable Multiprocessing")
        self.enable_mp_checkbox.setChecked(True)
        self.enable_mp_checkbox.stateChanged.connect(self._update_mp_config)
        options_layout.addWidget(self.enable_mp_checkbox)
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU Cores:"))
        self.cpu_spinbox = QSpinBox()
        self.cpu_spinbox.setMinimum(1)
        self.cpu_spinbox.setMaximum(cpu_count())
        self.cpu_spinbox.setValue(self.config.num_processes)
        self.cpu_spinbox.valueChanged.connect(self._update_cpu_config)
        cpu_layout.addWidget(self.cpu_spinbox)
        options_layout.addLayout(cpu_layout)
        self.enable_gpu_checkbox = QCheckBox("Enable GPU Processing")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.stateChanged.connect(self._update_gpu_config)
        options_layout.addWidget(self.enable_gpu_checkbox)
        self.use_opencl_checkbox = QCheckBox("Prefer OpenCL (Cross-platform)")
        self.use_opencl_checkbox.setChecked(self.config.use_opencl)
        self.use_opencl_checkbox.stateChanged.connect(self._update_opencl_config)
        options_layout.addWidget(self.use_opencl_checkbox)
        main_layout.addWidget(options_group)

        # --- Plot Mode (Radio Buttons) ---
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout(plot_group)
        self.plot_mode_group = QButtonGroup()
        self.all_plots_radio = QRadioButton("All plots (original behavior)")
        self.all_plots_radio.setChecked(True)
        self.plot_mode_group.addButton(self.all_plots_radio, 0)
        plot_layout.addWidget(self.all_plots_radio)
        self.overlay_avg_radio = QRadioButton("Overlay + Batch Average only (no individuals)")
        self.plot_mode_group.addButton(self.overlay_avg_radio, 1)
        plot_layout.addWidget(self.overlay_avg_radio)
        self.global_avg_radio = QRadioButton("Global Average only")
        self.plot_mode_group.addButton(self.global_avg_radio, 2)
        plot_layout.addWidget(self.global_avg_radio)
        self.plot_mode_group.buttonClicked.connect(self._update_plot_mode)
        main_layout.addWidget(plot_group)

        # --- Auto-Save Option ---
        autosave_group = QGroupBox("Auto-Save Options")
        autosave_layout = QVBoxLayout(autosave_group)
        self.enable_autosave_checkbox = QCheckBox("Enable automatic batch saving")
        self.enable_autosave_checkbox.stateChanged.connect(self._update_autosave_config)
        autosave_layout.addWidget(self.enable_autosave_checkbox)
        plots_layout = QHBoxLayout()
        plots_layout.addWidget(QLabel("Plots per file:"))
        self.plots_per_file_edit = QLineEdit("1000")
        self.plots_per_file_edit.setEnabled(False)
        self.plots_per_file_edit.textChanged.connect(self._update_plots_per_file)
        plots_layout.addWidget(self.plots_per_file_edit)
        autosave_layout.addLayout(plots_layout)
        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Base filename:"))
        self.base_filename_edit = QLineEdit("RnS_Plots_Batch")
        self.base_filename_edit.setEnabled(False)
        self.base_filename_edit.textChanged.connect(self._update_base_filename)
        filename_layout.addWidget(self.base_filename_edit)
        autosave_layout.addLayout(filename_layout)
        main_layout.addWidget(autosave_group)

        # --- Progress/Status ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        main_layout.addWidget(self.status_text)

        button_layout = QHBoxLayout()
        self.plot_button = QPushButton("Process and Plot")
        self.plot_button.clicked.connect(self._process_and_plot)
        button_layout.addWidget(self.plot_button)
        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self._save_project)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)

    def _browse_files(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("R&S SFT Files (*.sft)")
        file_dialog.setWindowTitle("Select SFT Files")
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            self.selected_files.extend(selected_files)
            self._update_file_list()
            self._log_message(f"Selected {len(selected_files)} files")

    def _clear_files(self):
        self.selected_files.clear()
        self._update_file_list()
        self._log_message("File list cleared")

    def _update_file_list(self):
        self.file_list_widget.clear()
        for file_path in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(file_path))

    def _update_mp_config(self, state):
        self.config.enable_multiprocessing = state == Qt.Checked
        self._log_message(f"Multiprocessing: {'Enabled' if self.config.enable_multiprocessing else 'Disabled'}")

    def _update_cpu_config(self, value):
        self.config.num_processes = value
        self.config.max_workers = value
        self._log_message(f"CPU cores set to: {value}")

    def _update_gpu_config(self, state):
        self.config.enable_gpu_processing = state == Qt.Checked
        self._log_message(f"GPU processing: {'Enabled' if self.config.enable_gpu_processing else 'Disabled'}")

    def _update_opencl_config(self, state):
        self.config.use_opencl = state == Qt.Checked
        self._log_message(f"OpenCL preference: {'Enabled' if self.config.use_opencl else 'Disabled'}")

    def _update_plot_mode(self, button):
        self.config.plot_mode = self.plot_mode_group.id(button)
        modes = ["All", "Overlay+BatchAverage", "GlobalAverage"]
        self._log_message(f"Plot mode: {modes[self.config.plot_mode]}")

    def _update_autosave_config(self, state):
        self.config.enable_auto_save = state == Qt.Checked
        self.plots_per_file_edit.setEnabled(self.config.enable_auto_save)
        self.base_filename_edit.setEnabled(self.config.enable_auto_save)
        self.vzplot.auto_save_enabled = self.config.enable_auto_save
        self._log_message(f"Auto-save: {'Enabled' if self.config.enable_auto_save else 'Disabled'}")

    def _update_plots_per_file(self, text):
        try:
            value = int(text)
            self.config.plots_per_file = max(1, value)
            self.vzplot.plots_per_file = self.config.plots_per_file
        except Exception:
            self.config.plots_per_file = 1000

    def _update_base_filename(self, text):
        self.config.base_batch_filename = text if text else "RnS_Plots_Batch"
        self.vzplot.base_batch_filename = self.config.base_batch_filename

    def _log_message(self, message):
        self.status_text.append(f"[{self._get_timestamp()}] {message}")

    def _get_timestamp(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _process_and_plot(self):
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "Please select SFT files first.")
            return
        self.plot_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.processing_thread = FileProcessingThread(
            self.selected_files,
            self.config,
            self.vzplot.searchData_strings,
            self.vzplot.sft_lines
        )
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self._on_processing_finished)
        self.processing_thread.error_occurred.connect(self._on_processing_error)
        self.processing_thread.start()
        self._log_message("Processing started...")

    def _on_processing_finished(self, results):
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        self._log_message(f"Processing completed: {len(successful_results)} successful, {len(failed_results)} failed")
        if failed_results:
            error_msg = "\n".join([f"{r['filename']}: {r['error']}" for r in failed_results])
            QMessageBox.warning(self, "Processing Errors", f"Some files failed to process:\n{error_msg}")
        if successful_results:
            # Distribute hits to the plot and save machinery, respecting batch logic
            self.vzplot.processed_batch_results = successful_results
            self.vzplot.processed_global_results = successful_results
            self.vzplot._create_plots_and_batches()
            self.save_button.setEnabled(True)

    def _on_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.plot_button.setEnabled(True)
        self._log_message(f"Processing error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)

    def _save_project(self):
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Save Veusz Project", "",
                                                   "Veusz High Precision Files (*.vszh5)")
        if save_path:
            try:
                self.vzplot.save(save_path)
                self._log_message(f"Project saved: {save_path}")
                reply = QMessageBox.question(
                    self, "Open in Veusz",
                    "Would you like to open the file in Veusz?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    VZPlotRnS.open_veusz_gui(save_path)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project: {e}")


# --- Enhanced Plotter with Batch Overlay/Average Only ---

class VZPlotRnS:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)
        self.searchData_strings = {
            'version': 'VERSION', 'type': 'TYPE',
            'mode': 'MODE', 'center freq': 'CENTER FREQ',
            'freq offset': 'FREQ OFFSET', 'span': 'SPAN',
            'x-axis': 'X-AXIS', 'start': 'START',
            'stop': 'STOP', 'stop_2': 'STOP',
            'ref level': 'REF LEVEL', 'level offset': 'LEVEL OFFSET',
            'ref position': 'REF POSITION', 'y-axis': 'Y-AXIS',
            'level range': 'LEVEL RANGE', 'rf att': 'RF ATT',
            'rbw': 'RBW', 'vbw': 'VBW', 'swt': 'SWT', 'trace mode': 'TRACE MODE',
            'detector': 'DETECTOR', 'sweep count': "SWEEP COUNT", 'trace': 'TRACE',
            'x-unit': 'X-UNIT', 'y-unit': 'Y-UNIT', 'preamplifier': 'PREAMPLIFIER',
            'transducer': 'TRANSDUCER', 'values': 'VALUES', 'section': 'SECTION'
        }
        self.sft_lines = [1, 2, 3] + list(range(5, 58, 2))
        self.plotInfo = plotDescInfo(
            xAxis_label='Frequency (Hz)',
            yAxis_label='Uncalibrated (dBm)',
            graph_notes=None,
            graph_title='Title',
            base_name=None,
            first_plot=True
        )
        self._datasets_by_base = defaultdict(list)
        self.base_batch_filename = config.base_batch_filename
        self.processed_batch_results = []
        self.processed_global_results = []
        self.plots_per_file = config.plots_per_file
        self.auto_save_enabled = config.enable_auto_save

    def _setup_auto_save_path(self):
        if not hasattr(self, 'base_save_path') or self.base_save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"RnS_AutoSave_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            self.base_save_path = os.path.join(save_dir, self.base_batch_filename)
        return self.base_save_path

    def save(self, filename: str):
        filename_root = os.path.splitext(filename)[0]
        filename_hp = filename_root + '.vszh5'
        file_split = os.path.split(filename)
        filename_vsz = (file_split[0] + '/Beware_oldVersion/' + os.path.splitext(file_split[1])[0] + '_BEWARE.vsz')
        self.doc.Save(filename_hp, mode='hdf5')
        os.makedirs(file_split[0] + '/Beware_oldVersion/', exist_ok=True)
        self.doc.Save(filename_vsz, mode='vsz')

    def _create_plots_and_batches(self):
        # 1. Divide results into batches
        batches = []
        batch_size = self.plots_per_file if self.auto_save_enabled else len(self.processed_batch_results)
        for i in range(0, len(self.processed_batch_results), batch_size):
            batches.append(self.processed_batch_results[i:i + batch_size])

        # 2. Compute the global average from all files, only if any global average option is enabled
        if self.config.plot_mode in [2, 1]:
            self._create_global_average(self.processed_global_results)

        # 3. For each batch, create new Veusz document
        for batch_index, batch_results in enumerate(batches):
            batch_doc = embed.Embedded('Batch_{:03d}'.format(batch_index + 1))
            batch_doc.EnableToolbar(enable=True)
            datasets_for_batch = []
            batch_basenames = []
            # --- Process each file in batch
            for result in batch_results:
                filename = result['filename']
                data_returned = result['data']
                base_name = os.path.splitext(os.path.basename(filename))[0]
                batch_basenames.append(base_name)
                # Only add the appropriate overlays
                batch_datasets = self._process_file_data_for_batch(batch_doc, base_name, data_returned)
                datasets_for_batch.extend(batch_datasets)
            # Add overlay if mode==1
            if self.config.plot_mode == 1 and datasets_for_batch:
                self._create_overlay_and_batch_average(batch_doc, batch_basenames, datasets_for_batch)

            # Save batch file
            batch_file = self._setup_auto_save_path() + "_{:03d}.vszh5".format(batch_index + 1)
            batch_doc.Save(batch_file, mode='hdf5')
        # Only for global average mode (mode==2), save global average file
        if self.config.plot_mode == 2:
            global_file = self._setup_auto_save_path() + "_global_avg.vszh5"
            self.global_doc.Save(global_file, mode='hdf5')

    def _process_file_data_for_batch(self, batch_doc, base_name, data_returned):
        datasets_for_batch = []
        # Extract section data
        data_sections = dict(filter(
            lambda item: 'section' in item[0], data_returned['pattern_matches'].items()))
        data_y_values = list(map(itemgetter('extracted_value'), data_returned['data_matches'].values()))
        data_line_numbers = list(map(itemgetter('line_number'), data_returned['data_matches'].values()))
        data_section_line_numbers = list(map(itemgetter('line_number'), data_sections.values()))
        data_section_content = list(map(itemgetter('content'), data_sections.values()))
        data_fields = data_returned['pattern_matches']
        num_pts = extract_with_regex(data_fields['values']['extracted_value'])
        num_pts = int(num_pts[0])
        freq_start = float(extract_with_regex(data_fields['start']['extracted_value'])[0])
        freq_stop = float(extract_with_regex(data_fields['stop_2']['extracted_value'])[0])
        freq_range = np.linspace(freq_start, freq_stop, num=num_pts, endpoint=True, dtype=np.float64)
        data_header = data_returned['line_data']
        data_notes = '\n'.join(data_header.values())
        data_notes = os.path.split(base_name)[1] + '\n\n' + data_notes

        # Only if All mode (0) or Overlay/Average (1), add x_data and y datasets
        if self.config.plot_mode in [0, 1]:
            for index, label in enumerate(data_section_content):
                dataset_name = base_name + "_" + label
                if index == 0:
                    x_data_name = base_name + '_freq'
                    batch_doc.SetData(name=x_data_name, val=freq_range)
                    datasets_for_batch.append(x_data_name)
                batch_doc.SetData(name=dataset_name, val=data_y_values[index])
                datasets_for_batch.append(dataset_name)
        return datasets_for_batch

    def _create_overlay_and_batch_average(self, batch_doc, batch_basenames, batch_datasets):
        # Overlay for selected batch
        batch_doc.Root.Add('page', name='Overlay')
        grid = batch_doc.Root.Overlay.Add('grid', columns=2)
        graph_all = grid.Add('graph', name='OverlayGraph')
        graph_all.Add('label', name='plotTitle')
        graph_all.plotTitle.label.val = 'Overlay of Batch'
        graph_all.x.label.val = self.plotInfo.xAxis_label
        graph_all.y.label.val = self.plotInfo.yAxis_label
        # Plot overlays (overlay only, no individuals)
        for ds in batch_datasets:
            if ds.endswith('_freq'):
                continue
            overlay_xy = graph_all.Add('xy', name=ds)
            overlay_xy.yData.val = ds
            overlay_xy.xData.val = ds.split('_')[0] + '_freq'
            overlay_xy.nanHandling = 'break-on'
            overlay_xy.marker.val = 'circle'
            overlay_xy.markerSize.val = '2pt'
            overlay_xy.MarkerLine.color.val = 'transparent'
            overlay_xy.MarkerFill.color.val = 'auto'
            overlay_xy.MarkerFill.transparency.val = 80
            overlay_xy.MarkerFill.style.val = 'solid'
            overlay_xy.FillBelow.transparency.val = 90
            overlay_xy.FillBelow.style.val = 'solid'
            overlay_xy.FillBelow.fillto.val = 'bottom'
            overlay_xy.FillBelow.color.val = 'darkgreen'
            overlay_xy.FillBelow.hide.val = True
            overlay_xy.PlotLine.color.val = 'auto'
        # Batch average: create and save average for batch
        self._create_average_for_batch(batch_doc, batch_datasets)

    def _create_average_for_batch(self, batch_doc, batch_datasets):
        # Extract batch datasets that are not freq
        candidates = [ds for ds in batch_datasets if 'freq' not in ds.lower()]
        if len(candidates) < 2:
            return
        arrays = [np.asarray(batch_doc.GetData(ds)[0]) for ds in candidates]
        linear_stack = np.vstack([10.0 ** (a / 10.0) for a in arrays])
        avg_lin = np.mean(linear_stack, axis=0)
        avg_db = 10.0 * np.log10(avg_lin)
        batch_doc.SetData(name='batch_avg_lin', val=avg_lin)
        batch_doc.SetData(name='batch_avg_dB', val=avg_db)
        batch_doc.TagDatasets('BatchAvg_dB', ['batch_avg_dB'])
        batch_doc.TagDatasets('BatchAvg_Linear', ['batch_avg_lin'])
        # Plot batch average
        batch_doc.Root.Add('page', name='BatchAverage')
        grid = batch_doc.Root.BatchAverage.Add('grid', columns=2)
        graph = grid.Add('graph', name='BatchAvgGraph')
        graph.Add('label', name='plotTitle')
        graph.plotTitle.label.val = 'Batch Average'
        graph.x.label.val = self.plotInfo.xAxis_label
        graph.y.label.val = self.plotInfo.yAxis_label
        xy = graph.Add('xy', name='batch_avg_dB')
        xy.yData.val = 'batch_avg_dB'
        xy.xData.val = batch_datasets[0].split('_')[0] + '_freq'
        xy.nanHandling = 'break-on'
        xy.marker.val = 'circle'
        xy.markerSize.val = '2pt'
        xy.MarkerLine.color.val = 'transparent'
        xy.MarkerFill.color.val = 'foreground'
        xy.MarkerFill.transparency.val = 80
        xy.MarkerFill.style.val = 'solid'
        xy.FillBelow.transparency.val = 90
        xy.FillBelow.style.val = 'solid'
        xy.FillBelow.fillto.val = 'bottom'
        xy.FillBelow.color.val = 'darkgreen'
        xy.FillBelow.hide.val = False
        xy.PlotLine.color.val = 'red'

    def _create_global_average(self, global_results):
        # Create single Veusz document for the global average
        self.global_doc = embed.Embedded('GlobalAverage')
        self.global_doc.EnableToolbar(enable=True)
        all_candidates = []
        base_names = []
        for result in global_results:
            filename = result['filename']
            data_returned = result['data']
            base_name = os.path.splitext(os.path.basename(filename))[0]
            base_names.append(base_name)
            # Extract values
            data_sections = dict(filter(
                lambda item: 'section' in item[0], data_returned['pattern_matches'].items()))
            data_y_values = list(map(itemgetter('extracted_value'), data_returned['data_matches'].values()))
            data_section_content = list(map(itemgetter('content'), data_sections.values()))
            # assemble per file datasets
            for index, label in enumerate(data_section_content):
                dsname = base_name + "_" + label
                self.global_doc.SetData(name=dsname, val=data_y_values[index])
                all_candidates.append(dsname)
        if len(all_candidates) < 2:
            return
        arrays = [np.asarray(self.global_doc.GetData(ds)[0]) for ds in all_candidates]
        linear_stack = np.vstack([10.0 ** (a / 10.0) for a in arrays])
        avg_lin = np.mean(linear_stack, axis=0)
        avg_db = 10.0 * np.log10(avg_lin)
        self.global_doc.SetData(name='global_avg_lin', val=avg_lin)
        self.global_doc.SetData(name='global_avg_dB', val=avg_db)
        self.global_doc.TagDatasets('GlobalAvg_dB', ['global_avg_dB'])
        self.global_doc.TagDatasets('GlobalAvg_Linear', ['global_avg_lin'])
        self.global_doc.Root.Add('page', name='GlobalAverage')
        grid = self.global_doc.Root.GlobalAverage.Add('grid', columns=2)
        graph = grid.Add('graph', name='GlobalAvgGraph')
        graph.Add('label', name='plotTitle')
        graph.plotTitle.label.val = 'Global Average'
        graph.x.label.val = self.plotInfo.xAxis_label
        graph.y.label.val = self.plotInfo.yAxis_label
        xy = graph.Add('xy', name='global_avg_dB')
        xy.yData.val = 'global_avg_dB'
        # Use first available freq
        xy.xData.val = all_candidates[0].split('_')[0] + '_freq'
        xy.nanHandling = 'break-on'
        xy.marker.val = 'circle'
        xy.markerSize.val = '2pt'
        xy.MarkerLine.color.val = 'transparent'
        xy.MarkerFill.color.val = 'foreground'
        xy.MarkerFill.transparency.val = 80
        xy.MarkerFill.style.val = 'solid'
        xy.FillBelow.transparency.val = 90
        xy.FillBelow.style.val = 'solid'
        xy.FillBelow.fillto.val = 'bottom'
        xy.FillBelow.color.val = 'darkgreen'
        xy.FillBelow.hide.val = False
        xy.PlotLine.color.val = 'red'

    @staticmethod
    def open_veusz_gui(filename: str):
        if sys.platform.startswith('win'):
            veusz_exe = os.path.join(sys.prefix, 'Scripts', 'veusz.exe')
        else:
            veusz_exe = os.path.join(sys.prefix, 'bin', 'veusz')
        if not os.path.exists(veusz_exe): return
        try:
            subprocess.Popen([veusz_exe, filename])
        except Exception:
            pass


# --- App Launcher ---

def main():
    mp.set_start_method('spawn', force=True)
    app = QApplication(sys.argv)
    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
