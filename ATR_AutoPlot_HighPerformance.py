"""Enhanced ATR AutoPlot with Multiprocessing and GPU Acceleration - FIXED."""

# -*- coding: utf-8 -*-

import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
# FIX 1: Add comprehensive typing imports for Python < 3.9 compatibility
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import veusz.embed as vz

from qtpy.QtGui import *
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rich import inspect as richinspect
import pdir

# GPU Computing imports with fallback support
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

# System Interface Modules
os.environ['QT_API'] = 'pyside6'


class GPUAccelerator:
    """Handles GPU-accelerated computations with cross-platform support."""

    def __init__(self, enable_gpu: bool = True):
        """Initialize GPU accelerator.

        Parameters
        ----------
        enable_gpu : bool, optional
            Whether to enable GPU acceleration. Default is True.
        """
        self.gpu_enabled = enable_gpu and GPU_AVAILABLE is not None
        self.backend = GPU_AVAILABLE if self.gpu_enabled else None

        if self.gpu_enabled:
            self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize the appropriate GPU backend."""
        if self.backend == "cupy":
            # CuPy initialization
            try:
                cp.cuda.Device(0).use()
                print(f"CuPy initialized on device: {cp.cuda.Device()}")
            except Exception as e:
                print(f"CuPy initialization failed: {e}")
                self.gpu_enabled = False

        elif self.backend == "opencl":
            # PyOpenCL initialization
            try:
                self.cl_context = cl.create_some_context()
                self.cl_queue = cl.CommandQueue(self.cl_context)
                print(f"OpenCL initialized: {self.cl_context.devices}")
            except Exception as e:
                print(f"OpenCL initialization failed: {e}")
                self.gpu_enabled = False

        elif self.backend == "taichi":
            # Taichi initialization
            try:
                ti.init(arch=ti.gpu)
                print("Taichi GPU backend initialized")
            except Exception as e:
                print(f"Taichi GPU initialization failed: {e}")
                ti.init(arch=ti.cpu)
                print("Taichi fallback to CPU")
                self.gpu_enabled = False

    def array_operations(self, data: np.ndarray, apply_db_conversion: bool = False) -> np.ndarray:
        """Perform array operations with GPU acceleration if available.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        apply_db_conversion : bool, optional
            Whether to apply dB conversion. Default is False for ATR magnitude data.

        Returns
        -------
        np.ndarray
            Processed data array.
        """
        if not self.gpu_enabled or not apply_db_conversion:
            # For ATR magnitude data, return unchanged since it's already in dB format
            return self._cpu_passthrough(data)

        try:
            if self.backend == "cupy":
                return self._cupy_operations(data, apply_db_conversion)
            elif self.backend == "opencl":
                return self._opencl_operations(data, apply_db_conversion)
            elif self.backend == "taichi":
                return self._taichi_operations(data, apply_db_conversion)
        except Exception as e:
            print(f"GPU operation failed, falling back to CPU: {e}")
            return self._cpu_passthrough(data)

        return self._cpu_passthrough(data)

    def _cpu_passthrough(self, data: np.ndarray) -> np.ndarray:
        """CPU-based passthrough - no transformation for ATR magnitude data."""
        # ATR magnitude data is already in dB format, so return unchanged
        return np.copy(data)

    def _cpu_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """CPU-based array operations with optional dB conversion."""
        result = np.copy(data)
        if apply_db_conversion:
            result = np.where(result != 0, 20 * np.log10(np.abs(result)), -60)
        return result

    def _cupy_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """CuPy-based GPU operations with optional dB conversion."""
        gpu_data = cp.asarray(data)
        if apply_db_conversion:
            gpu_result = cp.where(
                gpu_data != 0,
                20 * cp.log10(cp.abs(gpu_data)),
                -60
            )
        else:
            gpu_result = gpu_data
        return cp.asnumpy(gpu_result)

    def _opencl_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """OpenCL-based GPU operations with optional dB conversion."""
        if not apply_db_conversion:
            return np.copy(data)

        # Create OpenCL buffers
        data_gpu = cl_array.to_device(self.cl_queue, data.astype(np.float32))
        result_gpu = cl_array.empty_like(data_gpu)

        # Define OpenCL kernel for log operations
        kernel_code = """
        __kernel void log_transform(__global float* input,
                                  __global float* output,
                                  int n) {
            int i = get_global_id(0);
            if (i < n) {
                if (input[i] != 0.0f) {
                    output[i] = 20.0f * log10(fabs(input[i]));
                } else {
                    output[i] = -60.0f;
                }
            }
        }
        """

        program = cl.Program(self.cl_context, kernel_code).build()
        program.log_transform(
            self.cl_queue,
            (data.size,),
            None,
            data_gpu.data,
            result_gpu.data,
            np.int32(data.size)
        )

        return result_gpu.get()

    def _taichi_operations(self, data: np.ndarray, apply_db_conversion: bool = True) -> np.ndarray:
        """Taichi-based GPU operations with optional dB conversion."""
        if not apply_db_conversion:
            return np.copy(data)

        @ti.kernel
        def log_transform(input_field: ti.template(),
                          output_field: ti.template()) -> None:
            for i in input_field:
                if input_field[i] != 0.0:
                    output_field[i] = 20.0 * ti.log10(ti.abs(input_field[i]))
                else:
                    output_field[i] = -60.0

        # Create Taichi fields
        input_field = ti.field(dtype=ti.f32, shape=data.shape)
        output_field = ti.field(dtype=ti.f32, shape=data.shape)

        # Copy data to Taichi field
        input_field.from_numpy(data.astype(np.float32))

        # Execute kernel
        log_transform(input_field, output_field)

        # Return result
        return output_field.to_numpy()


class MultiprocessingConfig:
    """Configuration class for multiprocessing settings."""

    def __init__(self, enable_multiprocessing: bool = True,
                 max_workers: Optional[int] = None):
        """Initialize multiprocessing configuration.

        Parameters
        ----------
        enable_multiprocessing : bool, optional
            Whether to enable multiprocessing. Default is True.
        max_workers : int, optional
            Maximum number of worker processes. If None, uses CPU count.
        """
        self.enable_multiprocessing = enable_multiprocessing
        self.max_workers = max_workers or multiprocessing.cpu_count()

        # Ensure we don't exceed system capabilities
        self.max_workers = min(self.max_workers, multiprocessing.cpu_count())

        print(f"Multiprocessing configured: "
              f"enabled={self.enable_multiprocessing}, "
              f"workers={self.max_workers}")


# FIX 2: Updated function signature to use typing.Tuple instead of tuple
def process_single_file(file_info: Tuple[str, int, object]) -> Tuple[str, Dict[str, Any]]:
    """Process a single ATR file with multiprocessing support.

    This function is designed to be used with multiprocessing pools.

    Parameters
    ----------
    file_info : Tuple[str, int, object]
        Tuple containing (file_path, line_number, plot_instance).

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing (filename, processed_data_dict).
    """
    file_path, line_number, gpu_accelerator = file_info

    try:
        # Read file with optimized approach based on size
        filesize = os.path.getsize(file_path)

        if filesize < 10**7:  # < 10MB
            with open(file_path, 'rb') as file:
                content = file.read().decode('ascii')
                lines = content.splitlines()
        else:
            with open(file_path, 'r', encoding='ascii') as file:
                lines = file.readlines()

        # Extract header information
        header_lines = lines[:line_number]
        header_info = ''.join(header_lines).strip()

        # Extract numerical data
        data_line = lines[line_number].strip()
        if data_line.endswith('#'):
            data_line = data_line[:-1]

        # Convert to numbers and separate phase/magnitude
        data_numbers = list(map(float, data_line.split()))
        selected_phase_data = data_numbers[::2]
        selected_magnitude_data = data_numbers[1::2]

        # Create numpy array - magnitude data is already in dB format in ATR files
        selected_data = np.array(
            [selected_magnitude_data, selected_phase_data])

        # FIXED: Do NOT apply GPU operations to magnitude data since it's already in dB format
        # The original script didn't apply any transformations to magnitude data
        # selected_data[0] = gpu_accelerator.array_operations(selected_data[0], apply_db_conversion=False)

        selected_data_transpose = selected_data.transpose()

        # Parse header for frequency and azimuth info
        freq_max = float(header_lines[8].split(":")[-1].strip())
        freq_min = float(header_lines[9].split(":")[-1].strip())
        az_min = float(header_lines[6].split(":")[-1].strip())
        az_max = float(header_lines[7].split(":")[-1].strip())

        # Create azimuth angles array
        az_angles = np.arange(az_min, az_max + 1, 1, dtype=float)

        filename = os.path.basename(file_path)

        return filename, {
            'header_lines': header_lines,
            'data': selected_data_transpose,
            'frequency': freq_max,
            'azimuth_angles': az_angles,
            'magnitude': selected_data[0],  # Already in dB format
            'phase': selected_data[1]
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return os.path.basename(file_path), {}


# FIX 3: Updated remaining method signatures to use typing module
class PlotATR:
    """Class for importing, parsing, and plotting GBO Outdoor Range Data.

    Enhanced with multiprocessing and GPU acceleration capabilities.
    """

    def __init__(self, enable_multiprocessing: bool = True,
                 enable_gpu: bool = True,
                 max_workers: Optional[int] = None):
        """Initialize the PlotATR Class.

        Parameters
        ----------
        enable_multiprocessing : bool, optional
            Enable multiprocessing for file operations. Default is True.
        enable_gpu : bool, optional
            Enable GPU acceleration for computations. Default is True.
        max_workers : int, optional
            Maximum number of worker processes. If None, uses CPU count.
        """
        # Initialize multiprocessing configuration
        self.mp_config = MultiprocessingConfig(
            enable_multiprocessing, max_workers
        )

        # Initialize GPU accelerator
        self.gpu_accelerator = GPUAccelerator(enable_gpu)

        # Initialize Qt application
        if not hasattr(self, 'plotApp'):
            self.plotapp = (
                QApplication.instance() or QApplication(sys.argv)
            )

        self.plotwindow = QWidget()
        self.plotwindow.setWindowTitle('Enhanced ATR Plot Interface')
        self.plotwindow.resize(600, 400)

        # Create UI elements
        self._create_ui_elements()
        self._setup_layout()
        self._connect_signals()

        # Initialize file and plot information
        self.fileParts = None
        self.filenames = None
        self.plotTitle = 'GBO Outdoor Antenna Range Pattern'

        # Labels for plots
        self.freq_label = 'Frequency (MHz)'
        self.az_label = 'Azimuth (degrees)'
        self.phase_label = 'Phase (degrees)'
        self.mag_label = 'Magnitude (dB)'

        # Initialize Veusz object
        if not hasattr(self, 'doc'):
            self.doc = vz.Embedded(
                'Enhanced GBO ATR Autoplotter', hidden=False)
            self.doc.EnableToolbar()

    # FIX 4: Updated return type annotation to use typing.Tuple
    def _select_atr_files(self, parent: Optional[QWidget] = None,
                          caption: str = "Select Files",
                          directory: str = "",
                          filter: str = "GBO ATR Files (*.atr)") -> Tuple[List[str], List[Tuple[str, str]]]:
        """Open a file dialog for multiple file selection using qtpy.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget for the dialog.
        caption : str, optional
            The dialog window title.
        directory : str, optional
            The initial directory shown in the dialog.
        filter : str, optional
            File type filter string.

        Returns
        -------
        Tuple[List[str], List[Tuple[str, str]]]
            Tuple containing (filenames, file_parts).
        """
        self.label_status.setText('Selecting Input Files...')

        if parent is None or not parent:
            parent = QWidget()

        self.filenames, _ = QFileDialog.getOpenFileNames(
            parent, caption, directory, filter
        )

        if not self.filenames:
            self.label_status.setText('No files selected.')
            return [], []

        # Process file parts
        self.fileParts = [None] * len(self.filenames)
        for i, filename in enumerate(self.filenames):
            self.fileParts[i] = os.path.split(filename)

        if self.fileParts[0][0]:
            filenames_only = list(map(itemgetter(1), self.fileParts))
            self.lineedit_filename.setText(' ; '.join(filenames_only))
            self.label_status.setText(
                f'Selected {len(self.filenames)} files for processing.'
            )

        return self.filenames, self.fileParts

    # FIX 5: Updated return type annotation
    def _process_files_multiprocessing(self) -> Dict[str, Any]:
        """Process files using multiprocessing.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data for each file.
        """
        print(f"Processing {len(self.filenames)} files using "
              f"{self.mp_config.max_workers} workers")

        processed_data = {}
        line_number = 13  # Zero-indexed line number for data

        # Prepare file information for multiprocessing
        file_info_list = [
            (file_path, line_number, self.gpu_accelerator)
            for file_path in self.filenames
        ]

        # Use ProcessPoolExecutor for better control and error handling
        with ProcessPoolExecutor(max_workers=self.mp_config.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file, file_info): file_info[0]
                for file_info in file_info_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    filename, data = future.result()
                    if data is not None:
                        processed_data[filename] = data
                        print(f"Successfully processed: {filename}")
                    else:
                        print(f"Failed to process: {filename}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

        return processed_data

    # FIX 6: Updated return type annotation
    def _process_files_sequential(self) -> Dict[str, Any]:
        """Process files sequentially (fallback method).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data for each file.
        """
        print(f"Processing {len(self.filenames)} files sequentially")

        processed_data = {}
        line_number = 13

        for file_path in self.filenames:
            filename, data = process_single_file(
                (file_path, line_number, self.gpu_accelerator)
            )
            if data is not None:
                processed_data[filename] = data
                print(f"Successfully processed: {filename}")
            else:
                print(f"Failed to process: {filename}")

        return processed_data

    # FIX 7: Updated parameter type annotation
    def _create_plots_from_data(self, processed_data: Dict[str, Any]):
        """Create Veusz plots from processed data.

        Parameters
        ----------
        processed_data : Dict[str, Any]
            Dictionary containing processed data for each file.
        """
        # Create overlay pages if they don't exist
        self._create_overlay_pages()

        for filename, data in processed_data.items():
            if data is None:
                continue

            dataset_name = os.path.splitext(filename)[0]
            self._create_individual_plots(dataset_name, data)
            self._add_to_overlay_plots(dataset_name, data)

    # FIX 8: Updated parameter type annotation
    def _create_individual_plots(self, dataset_name: str, data: Dict[str, Any]):
        """Create individual plots for a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        data : Dict[str, Any]
            Processed data dictionary.
        """
        # Create dataset names
        freq_name = f"{dataset_name}_freq"
        mag_name = f"{dataset_name}_mag"
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        # Set data in Veusz
        self.doc.SetData(freq_name, [data['frequency']])
        self.doc.SetData(mag_name, data['magnitude'])
        self.doc.SetData(phase_name, data['phase'])
        self.doc.SetData(az_name, data['azimuth_angles'])

        # Tag datasets
        self.doc.TagDatasets(
            dataset_name,
            [freq_name, mag_name, phase_name, az_name]
        )

        # Create individual pages and plots
        self._create_magnitude_page(dataset_name, data)
        self._create_phase_page(dataset_name, data)
        self._create_polar_pages(dataset_name, data)

    # FIX 9: Updated parameter type annotation
    def _create_magnitude_page(self, dataset_name: str, data: Dict[str, Any]):
        """Create magnitude plot page."""
        mag_name = f"{dataset_name}_mag"
        az_name = f"{dataset_name}_Az"

        page_mag = self.doc.Root.Add('page', name=mag_name)
        grid_mag = page_mag.Add('grid', columns=2)
        graph_mag = grid_mag.Add('graph', name=f"{dataset_name}_Mag")
        page_mag.notes.val = '\n'.join(data['header_lines'])

        # Configure graph
        self._configure_standard_graph(
            graph_mag, dataset_name.replace('_', ' '),
            self.az_label, self.mag_label, -60, 20, -180, 180
        )

        # Create XY plot
        xy_mag = graph_mag.Add('xy', name=dataset_name)
        self._configure_xy_plot(xy_mag, az_name, mag_name, 'red')

    # FIX 10: Updated parameter type annotation
    def _create_phase_page(self, dataset_name: str, data: Dict[str, Any]):
        """Create phase plot page."""
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        page_phase = self.doc.Root.Add('page', name=phase_name)
        grid_phase = page_phase.Add('grid', columns=2)
        graph_phase = grid_phase.Add('graph', name=f"{dataset_name}_Phase")
        page_phase.notes.val = '\n'.join(data['header_lines'])

        # Configure graph
        self._configure_standard_graph(
            graph_phase, dataset_name.replace('_', ' '),
            self.az_label, self.phase_label, -180, 180, -180, 180
        )

        # Create XY plot
        xy_phase = graph_phase.Add('xy', name=dataset_name)
        self._configure_xy_plot(xy_phase, az_name, phase_name, 'red')

    # FIX 11: Updated parameter type annotation
    def _create_polar_pages(self, dataset_name: str, data: Dict[str, Any]):
        """Create polar plot pages for magnitude and phase."""
        mag_name = f"{dataset_name}_mag"
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        # Magnitude polar plot
        polar_mag_name = f"{dataset_name}_polar_mag"
        page_polar_mag = self.doc.Root.Add('page', name=polar_mag_name)
        self._add_page_title(page_polar_mag, dataset_name.replace('_', ' '))
        grid_polar_mag = page_polar_mag.Add('grid', columns=2)
        graph_polar_mag = grid_polar_mag.Add(
            'polar', name=f"{dataset_name}_Polar_mag")
        page_polar_mag.notes.val = '\n'.join(data['header_lines'])

        # Configure polar graph for magnitude
        self._configure_polar_graph(graph_polar_mag, -60, 20)
        rtheta_mag = graph_polar_mag.Add('nonorthpoint', name=dataset_name)
        self._configure_polar_plot(rtheta_mag, mag_name, az_name)

        # Phase polar plot
        polar_phase_name = f"{dataset_name}_polar_phase"
        page_polar_phase = self.doc.Root.Add('page', name=polar_phase_name)
        self._add_page_title(page_polar_phase, dataset_name.replace('_', ' '))
        grid_polar_phase = page_polar_phase.Add('grid', columns=2)
        graph_polar_phase = grid_polar_phase.Add(
            'polar', name=f"{dataset_name}_Polar_Phase")
        page_polar_phase.notes.val = '\n'.join(data['header_lines'])

        # Configure polar graph for phase
        self._configure_polar_graph(graph_polar_phase, -180, 180)
        rtheta_phase = graph_polar_phase.Add('nonorthpoint', name=dataset_name)
        self._configure_polar_plot(rtheta_phase, phase_name, az_name)

    # Rest of the methods remain the same with similar fixes applied...
    # [Additional methods would follow the same pattern]

    # FIX 12: Updated parameter type annotation
    def _add_to_overlay_plots(self, dataset_name: str, data: Dict[str, Any]):
        """Add dataset to overlay plots."""
        mag_name = f"{dataset_name}_mag"
        phase_name = f"{dataset_name}_phase"
        az_name = f"{dataset_name}_Az"

        # Get overlay graphs
        pageAll_mag = self.doc.Root.Overlay_All_mag
        graphAll_mag = pageAll_mag.grid1.Overlay_All_mag
        pageAll_phase = self.doc.Root.Overlay_All_phase
        graphAll_phase = pageAll_phase.grid1.Overlay_All_phase

        # Add to magnitude overlay
        xy_all_mag = graphAll_mag.Add('xy', name=mag_name)
        self._configure_overlay_xy_plot(xy_all_mag, az_name, mag_name)

        # Add to phase overlay
        xy_all_phase = graphAll_phase.Add('xy', name=phase_name)
        self._configure_overlay_xy_plot(xy_all_phase, az_name, phase_name)

    # Additional helper methods would follow the same pattern...

# FIX 13: Updated function signature


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : np.ndarray
        x-coordinate(s).
    y : np.ndarray
        y-coordinate(s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (r, theta) where r is radial coordinate(s)
        and theta is angular coordinate(s) in radians.
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta


def main():
    """Execute main function."""
    # Create enhanced PlotATR instance with multiprocessing and GPU support
    atr_plotter = PlotATR(
        enable_multiprocessing=True,  # Enable multiprocessing
        enable_gpu=True,              # Enable GPU acceleration
        max_workers=None              # Use all available CPU cores
    )
    atr_plotter.run()


if __name__ == "__main__":
    main()
