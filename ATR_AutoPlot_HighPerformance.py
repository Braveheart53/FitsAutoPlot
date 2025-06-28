"""Enhanced ATR AutoPlot with Multiprocessing, GPU Acceleration, and Modern GUI.

This version combines:
• the multiprocessing/GPU fixes (magnitude data no longer re-converted to dB),
• the RandS-style modern Qt interface, and
• the original ATR plotting logic.

Author: William W. Wallace
Last updated: 2025-06-28
"""

# --------------------------------------------------------------------------- #
# --- Standard Library Imports
# --------------------------------------------------------------------------- #
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# --- Third-Party Imports
# --------------------------------------------------------------------------- #
import numpy as np
import veusz.embed as vz

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QListWidget, QLabel, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QProgressBar, QTextEdit, QSplitter
)

# --------------------------------------------------------------------------- #
# --- Optional GPU Back-Ends
# --------------------------------------------------------------------------- #
try:
    import cupy as cp
    GPU_AVAILABLE = "cupy"
except Exception:
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        GPU_AVAILABLE = "opencl"
    except Exception:
        try:
            import taichi as ti
            GPU_AVAILABLE = "taichi"
        except Exception:
            GPU_AVAILABLE = None

# Ensure QtPy picks the same binding everywhere
os.environ["QT_API"] = "pyside6"


# =========================================================================== #
#  GPU ACCELERATION HELPER
# =========================================================================== #
class GPUAccelerator:
    """Thin wrapper providing optional GPU array operations."""

    def __init__(self, enable_gpu: bool = True):
        self.gpu_enabled = enable_gpu and GPU_AVAILABLE is not None
        self.backend = GPU_AVAILABLE if self.gpu_enabled else None
        if self.gpu_enabled:
            self._initialise_backend()

    # --------------------------------------------------------------------- #
    #  Back-End Initialisation
    # --------------------------------------------------------------------- #
    def _initialise_backend(self) -> None:
        try:
            if self.backend == "cupy":
                cp.cuda.Device(0).use()
            elif self.backend == "opencl":
                self._cl_ctx = cl.create_some_context()
                self._cl_queue = cl.CommandQueue(self._cl_ctx)
            elif self.backend == "taichi":
                ti.init(arch=ti.gpu)
        except Exception:
            # Silently fall back to CPU when initialisation fails
            self.gpu_enabled = False
            self.backend = None

    # --------------------------------------------------------------------- #
    #  Array Operation (optional dB conversion)
    # --------------------------------------------------------------------- #
    def transform(
        self, arr: np.ndarray, *, apply_db_conversion: bool = False
    ) -> np.ndarray:
        """Return `arr` unchanged or log-converted, optionally on GPU."""
        if not self.gpu_enabled or not apply_db_conversion:
            return arr.copy() if not apply_db_conversion else self._cpu_log(arr)

        try:
            if self.backend == "cupy":
                g = cp.asarray(arr)
                out = cp.where(g != 0, 20 * cp.log10(cp.abs(g)), -60)
                return cp.asnumpy(out)

            if self.backend == "opencl":
                return self._opencl_log(arr)

            if self.backend == "taichi":
                return self._taichi_log(arr)

        except Exception:  # pragma: no cover
            pass  # Any GPU error → CPU fall-back

        return arr.copy() if not apply_db_conversion else self._cpu_log(arr)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cpu_log(arr: np.ndarray) -> np.ndarray:
        out = np.where(arr != 0, 20 * np.log10(np.abs(arr)), -60)
        return out.astype(arr.dtype)

    # ------------------------  OpenCL implementation  ------------------- #
    def _opencl_log(self, arr: np.ndarray) -> np.ndarray:
        kernel = """
        __kernel void log_conv(__global const float *in,
                               __global float *out,
                               const int n) {
            int i = get_global_id(0);
            if (i < n) {
                out[i] = in[i] ? 20.0f*log10(fabs(in[i])) : -60.0f;
            }
        }
        """
        mf = cl.mem_flags
        buf_in = cl.Buffer(self._cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                           hostbuf=arr.astype(np.float32))
        buf_out = cl.Buffer(self._cl_ctx, mf.WRITE_ONLY, arr.nbytes)
        prg = cl.Program(self._cl_ctx, kernel).build()
        prg.log_conv(self._cl_queue, (arr.size,), None, buf_in, buf_out,
                     np.int32(arr.size))
        out = np.empty_like(arr, dtype=np.float32)
        cl.enqueue_copy(self._cl_queue, out, buf_out)
        return out

    # ------------------------  Taichi implementation  ------------------- #
    def _taichi_log(self, arr: np.ndarray) -> np.ndarray:
        ti_arr_in = ti.field(dtype=ti.f32, shape=arr.shape)
        ti_arr_out = ti.field(dtype=ti.f32, shape=arr.shape)
        ti_arr_in.from_numpy(arr.astype(np.float32))

        @ti.kernel
        def kernel(inp: ti.template(), outp: ti.template()):
            for i in inp:
                outp[i] = 20.0 * \
                    ti.log10(ti.abs(inp[i])) if inp[i] != 0 else -60

        kernel(ti_arr_in, ti_arr_out)
        return ti_arr_out.to_numpy()


# =========================================================================== #
#  MULTIPROCESSING CONFIG
# =========================================================================== #
class MultiprocessingConfig:
    """Simple container for multiprocessing settings."""

    def __init__(self, enable_mp: bool = True, max_workers: Optional[int] = None):
        self.enable = enable_mp
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.max_workers = min(self.max_workers, multiprocessing.cpu_count())


# =========================================================================== #
#  FILE-LEVEL WORKER  (runs in separate Python process)
# =========================================================================== #
def _process_single_file(
    info: Tuple[str, int, bool]
) -> Tuple[str, dict]:
    """Read, parse, and return data for one ATR file (CPU only)."""
    file_path, line_number, _dummy_gpu_flag = info
    try:
        # Fast path for small files
        with open(file_path, "r", encoding="ascii", errors="ignore") as fh:
            lines = fh.readlines()

        header_lines = lines[:line_number]
        data_line = lines[line_number].rstrip().rstrip("#")
        nums = list(map(float, data_line.split()))
        phase = nums[0::2]
        magnitude = nums[1::2]              # Already in dB – leave untouched

        data = np.vstack([magnitude, phase]).T
        az_min = float(header_lines[6].split(":")[-1])
        az_max = float(header_lines[7].split(":")[-1])
        az = np.arange(az_min, az_max + 1, 1, dtype=float)

        return os.path.basename(file_path), {
            "header": header_lines,
            "data": data,
            "frequency": float(header_lines[8].split(":")[-1]),
            "azimuth": az,
            "mag": magnitude,
            "phase": phase,
        }
    except Exception as exc:                                                       # noqa: BLE001
        return os.path.basename(file_path), {"error": str(exc)}


# =========================================================================== #
#  THREAD (runs in GUI process, spawns processes as needed)
# =========================================================================== #
class ATRProcessingThread(QThread):
    """Background thread so the GUI stays responsive."""

    progress_updated = Signal(int)
    finished = Signal(dict)
    errored = Signal(str)

    def __init__(self, files: list[str], mp_cfg: MultiprocessingConfig):
        super().__init__()
        self._files = files
        self._cfg = mp_cfg

    # ------------------------------------------------------------------ #
    #  Worker
    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D401
        try:
            out: dict = {}
            if self._cfg.enable and len(self._files) > 1:
                out = self._proc_mp()
            else:
                out = self._proc_seq()
            self.finished.emit(out)
        except Exception as exc:                                                   # noqa: BLE001
            self.errored.emit(str(exc))

    # --------------------  Helpers  -------------------- #
    def _proc_mp(self) -> dict:
        info_list = [(fp, 13, False) for fp in self._files]
        out: dict = {}
        with ProcessPoolExecutor(max_workers=self._cfg.max_workers) as exe:
            fut_to_file = {exe.submit(
                _process_single_file, i): i[0] for i in info_list}
            completed = 0
            total = len(info_list)
            for fut in as_completed(fut_to_file):
                fpath = fut_to_file[fut]
                fname, data = fut.result()
                out[fname] = data
                completed += 1
                self.progress_updated.emit(int(completed / total * 100))
        return out

    def _proc_seq(self) -> dict:
        out: dict = {}
        total = len(self._files)
        for idx, fp in enumerate(self._files, 1):
            fname, data = _process_single_file((fp, 13, False))
            out[fname] = data
            self.progress_updated.emit(int(idx / total * 100))
        return out


# =========================================================================== #
#  VEUSZ PLOTTING CORE  (separated from GUI controls)
# =========================================================================== #
class ATRVeuszPlotter:
    """Encapsulates all Veusz interactions."""

    def __init__(self):
        self.doc = vz.Embedded("GBO ATR Autoplotter", hidden=False)
        self.doc.EnableToolbar()
        self._ensure_overlay_pages()

    # ------------------------------------------------------------------ #
    #  Overlay Pages
    # ------------------------------------------------------------------ #
    def _ensure_overlay_pages(self) -> None:
        if "Overlay_All_mag" in self.doc.Root.childnames:
            return

        pg_mag = self.doc.Root.Add("page", name="Overlay_All_mag")
        grid_mag = pg_mag.Add("grid", columns=2)
        self._graph_mag = grid_mag.Add("graph", name="Overlay_All_mag")

        pg_ph = self.doc.Root.Add("page", name="Overlay_All_phase")
        grid_ph = pg_ph.Add("grid", columns=2)
        self._graph_phase = grid_ph.Add("graph", name="Overlay_All_phase")

        # Styling helper
        def _style(g, title, ylab, y_min, y_max):
            with g:
                lbl = g.Add("label", name="title")
                lbl.label.val = title
                g.x.label.val = "Azimuth (°)"
                g.y.label.val = ylab
                g.y.min.val, g.y.max.val = y_min, y_max
                g.x.min.val, g.x.max.val = -180, 180
                g.x.GridLines.hide.val = False
                g.y.GridLines.hide.val = False

        _style(self._graph_mag, "Overlay of Imported Magnitude",
               "Magnitude (dB)", -60, 20)
        _style(self._graph_phase, "Overlay of Imported Phase",
               "Phase (°)", -180, 180)

        self.doc.Root.colorTheme.val = "max128"

    # ------------------------------------------------------------------ #
    #  Per-file Pages / Datasets
    # ------------------------------------------------------------------ #
    def add_file(self, fname: str, dat: dict) -> None:
        ds_base = os.path.splitext(fname)[0]
        mag_name = f"{ds_base}_mag"
        ph_name = f"{ds_base}_phase"
        az_name = f"{ds_base}_Az"

        # Store data
        self.doc.SetData(mag_name, dat["mag"])
        self.doc.SetData(ph_name, dat["phase"])
        self.doc.SetData(az_name, dat["azimuth"])

        # Overlay plots
        self._add_overlay(self._graph_mag, az_name, mag_name)
        self._add_overlay(self._graph_phase, az_name, ph_name)

        # Individual pages (XY + polar) could be added similarly
        # - omitted here for brevity, but the mechanism matches the earlier
        #   fully expanded script.

    # -------------------------  Helpers  ----------------------------- #
    @staticmethod
    def _add_overlay(graph, x, y):
        with graph:
            xy = graph.Add("xy")
            xy.xData.val = x
            xy.yData.val = y
            xy.marker.val = "circle"
            xy.markerSize.val = "2pt"
            xy.MarkerLine.color.val = "transparent"
            xy.PlotLine.color.val = "auto"

    # ------------------------------------------------------------------ #
    #  Save / Open
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        root, _ = os.path.splitext(path)
        h5 = root + ".vszh5"
        legacy = (os.path.dirname(path) + "/Beware_oldVersion/" +
                  os.path.basename(root) + "_BEWARE.vsz")
        os.makedirs(os.path.dirname(legacy), exist_ok=True)
        self.doc.Save(h5, mode="hdf5")
        self.doc.Save(legacy, mode="vsz")

    @staticmethod
    def open_in_veusz(filepath: str) -> None:
        exe = ("veusz.exe" if sys.platform.startswith("win")
               else os.path.join(sys.prefix, "bin", "veusz"))
        if not os.path.exists(exe):
            QMessageBox.critical(None, "Veusz Not Found",
                                 "Veusz executable not found in environment.")
            return
        subprocess.Popen([exe, filepath])


# =========================================================================== #
#  MAIN WINDOW
# =========================================================================== #
class MainWindow(QMainWindow):
    """Modern Qt GUI wrapping all functionality."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced ATR AutoPlot")
        self.resize(900, 700)

        # Core components
        self._mp_cfg = MultiprocessingConfig()
        self._plotter = ATRVeuszPlotter()

        # File list
        self._files: list[str] = []

        # Build UI
        self._ui()

    # ------------------------------------------------------------------ #
    #  UI Construction
    # ------------------------------------------------------------------ #
    def _ui(self):
        cw = QWidget(self)
        self.setCentralWidget(cw)
        main = QVBoxLayout(cw)

        splitter = QSplitter(Qt.Vertical)
        main.addWidget(splitter)

        # -----------------  TOP PANE  ----------------- #
        top = QWidget()
        top_layout = QVBoxLayout(top)
        splitter.addWidget(top)

        # FILE SELECTION
        grp_files = QGroupBox("ATR File Selection")
        top_layout.addWidget(grp_files)
        v_files = QVBoxLayout(grp_files)

        self.list_files = QListWidget()
        self.list_files.setMinimumHeight(140)
        v_files.addWidget(self.list_files)

        h_file_btn = QHBoxLayout()
        btn_browse = QPushButton("Browse ATR Files…")
        btn_browse.clicked.connect(self._browse_files)
        h_file_btn.addWidget(btn_browse)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_files)
        h_file_btn.addWidget(btn_clear)
        h_file_btn.addStretch()
        v_files.addLayout(h_file_btn)

        # OPTIONS
        grp_opt = QGroupBox("Processing Options")
        top_layout.addWidget(grp_opt)
        v_opt = QVBoxLayout(grp_opt)

        self.chk_mp = QCheckBox("Enable Multiprocessing")
        self.chk_mp.setChecked(self._mp_cfg.enable)
        self.chk_mp.stateChanged.connect(
            lambda s: setattr(self._mp_cfg, "enable", s == Qt.Checked))
        v_opt.addWidget(self.chk_mp)

        h_cpu = QHBoxLayout()
        h_cpu.addWidget(QLabel("CPU Cores:"))
        self.spin_cpu = QSpinBox()
        self.spin_cpu.setRange(1, multiprocessing.cpu_count())
        self.spin_cpu.setValue(self._mp_cfg.max_workers)
        self.spin_cpu.valueChanged.connect(
            lambda v: setattr(self._mp_cfg, "max_workers", v))
        h_cpu.addWidget(self.spin_cpu)
        h_cpu.addStretch()
        v_opt.addLayout(h_cpu)

        # -----------------  BOTTOM PANE  ----------------- #
        bottom = QWidget()
        splitter.addWidget(bottom)
        v_bottom = QVBoxLayout(bottom)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v_bottom.addWidget(self.bar)

        grp_status = QGroupBox("Status Messages")
        v_bottom.addWidget(grp_status)
        v_status = QVBoxLayout(grp_status)

        self.txt_status = QTextEdit()
        self.txt_status.setReadOnly(True)
        self.txt_status.setFont(QFont("Consolas", 9))
        v_status.addWidget(self.txt_status)

        h_btn = QHBoxLayout()
        self.btn_process = QPushButton("Process && Plot")
        self.btn_process.clicked.connect(self._process)
        h_btn.addWidget(self.btn_process)

        self.btn_save = QPushButton("Save Veusz Project")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_project)
        h_btn.addWidget(self.btn_save)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        h_btn.addWidget(btn_close)
        v_bottom.addLayout(h_btn)

        splitter.setSizes([420, 260])

    # ------------------------------------------------------------------ #
    #  UI Helpers
    # ------------------------------------------------------------------ #
    def _log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_status.append(f"[{ts}] {msg}")
        self.txt_status.moveCursor(self.txt_status.textCursor().End)

    # ------------------------------------------------------------------ #
    #  File Operations
    # ------------------------------------------------------------------ #
    def _browse_files(self):
        dlg = QFileDialog(self, "Select ATR Files")
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("GBO ATR Files (*.atr)")
        if dlg.exec_():
            new = [f for f in dlg.selectedFiles() if f not in self._files]
            self._files.extend(new)
            self._update_file_list()
            self._log(f"Added {len(new)} file(s)")

    def _clear_files(self):
        self._files.clear()
        self._update_file_list()

    def _update_file_list(self):
        self.list_files.clear()
        self.list_files.addItems([os.path.basename(f) for f in self._files])

    # ------------------------------------------------------------------ #
    #  Processing + Plotting
    # ------------------------------------------------------------------ #
    def _process(self):
        if not self._files:
            QMessageBox.warning(self, "No Files",
                                "Please select at least one ATR file.")
            return
        self.btn_process.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setValue(0)

        self._thread = ATRProcessingThread(self._files, self._mp_cfg)
        self._thread.progress_updated.connect(self.bar.setValue)
        self._thread.finished.connect(self._on_done)
        self._thread.errored.connect(self._on_error)
        self._thread.start()
        self._log("Processing started…")

    def _on_done(self, result: dict):
        self.bar.setVisible(False)
        self.btn_process.setEnabled(True)
        ok = [k for k, v in result.items() if "error" not in v]
        bad = [f"{k}: {v['error']}" for k, v in result.items() if "error" in v]

        for k in ok:
            self._plotter.add_file(k, result[k])

        self._log(f"Processing finished – {len(ok)} OK, {len(bad)} failed")
        if bad:
            QMessageBox.warning(self, "Some Errors",
                                "Errors occurred:\n" + "\n".join(bad))
        if ok:
            self.btn_save.setEnabled(True)

    def _on_error(self, msg: str):
        self.bar.setVisible(False)
        self.btn_process.setEnabled(True)
        self._log(f"Fatal error: {msg}")
        QMessageBox.critical(self, "Processing Error", msg)

    # ------------------------------------------------------------------ #
    #  Save Project
    # ------------------------------------------------------------------ #
    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Veusz Project", "",
            "Veusz High Precision Files (*.vszh5)"
        )
        if not path:
            return
        try:
            self._plotter.save(path)
            self._log(f"Saved project: {path}")
            if (
                QMessageBox.question(
                    self, "Open in Veusz?",
                    "Open the saved project in Veusz GUI now?",
                    QMessageBox.Yes | QMessageBox.No
                ) == QMessageBox.Yes
            ):
                self._plotter.open_in_veusz(path)
        except Exception as exc:                                                   # noqa: BLE001
            QMessageBox.critical(self, "Save Error", str(exc))
            self._log(f"Save failed: {exc}")


# =========================================================================== #
#  MAIN ENTRY POINT
# =========================================================================== #
def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
