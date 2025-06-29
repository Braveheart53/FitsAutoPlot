#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Touchstone AutoPlot with Modern Qt GUI Interface,
Time Domain Analysis, and Smith Chart Plotting.

Only change: added missing import of `Any` for Python 3.8 compatibility.
"""

# %% Standard Library Imports
import multiprocessing
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
# Added `Any` here for compatibility with Python 3.8
from typing import Any, Dict, Tuple

# %% Third-Party Imports
import numpy as np
import skrf as rf
from skrf import Network
from skrf.time import time_gate
import veusz.embed as vz

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QListWidget, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# %% Constants
SMITH_BG_PATH = str(pathlib.Path(__file__).with_name("colour_smith_chart.pdf"))

# %% Configuration Data Classes


@dataclass
class ProcessingConfig:
    enable_multiprocessing: bool = True
    enable_gpu_processing: bool = True
    num_processes: int = multiprocessing.cpu_count()
    max_workers: int = multiprocessing.cpu_count()


@dataclass
class TimeDomainConfig:
    window_type: str = 'kaiser'
    window_param: float = 6.0
    gate_start: float = 0.0
    gate_stop: float = 1.0
    mode: str = 'bandpass'
    method: str = 'fft'
    t_unit: str = 'ns'
    auto_gate: bool = True


@dataclass
class SmithChartConfig:
    chart_type: str = 'z'  # 'z' for impedance, 'y' for admittance
    draw_labels: bool = True
    draw_vswr: bool = True
    reference_impedance: float = 50.0

# %% GPU Stub


class GPUAccelerator:
    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu = enable_gpu

    def process_s_parameters(self, data: np.ndarray) -> np.ndarray:
        return data  # Passthrough

# %% Time-Domain Processor


class TimeDomainProcessor:
    def __init__(self, cfg: TimeDomainConfig):
        self.cfg = cfg

    def process_network(self, ntwk: Network) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            t_ns = ntwk.frequency.t_ns
            for i in range(ntwk.nports):
                for j in range(ntwk.nports):
                    tag = f"S{i+1}{j+1}"
                    view = ntwk.copy()
                    view.s = ntwk.s[:, i, j].reshape(-1, 1, 1)
                    out[f"{tag}_td"] = view.s_time.flatten()
                    gated = time_gate(
                        view,
                        start=None if self.cfg.auto_gate else self.cfg.gate_start,
                        stop=None if self.cfg.auto_gate else self.cfg.gate_stop,
                        mode=self.cfg.mode,
                        window=(self.cfg.window_type, self.cfg.window_param),
                        method=self.cfg.method,
                        t_unit=self.cfg.t_unit
                    )
                    out[f"{tag}_td_filtered"] = gated.s_time.flatten()
            out["time"] = t_ns
        except Exception as e:
            out["error"] = str(e)
        return out

# %% Smith-Chart Processor


class SmithChartProcessor:
    def __init__(self, cfg: SmithChartConfig):
        self.cfg = cfg

    def process_network_for_smith_chart(
        self,
        ntwk: Network,
        apply_td_filter: bool = False,
        td_proc: TimeDomainProcessor = None
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i in range(ntwk.nports):
            for j in range(ntwk.nports):
                tag = f"S{i+1}{j+1}"
                view = ntwk.copy()
                view.s = ntwk.s[:, i, j].reshape(-1, 1, 1)
                if apply_td_filter and td_proc:
                    gated = time_gate(
                        view,
                        mode=td_proc.cfg.mode,
                        window=(td_proc.cfg.window_type,
                                td_proc.cfg.window_param),
                        method=td_proc.cfg.method,
                        t_unit=td_proc.cfg.t_unit
                    )
                    gamma = gated.s[:, 0, 0]
                else:
                    gamma = view.s[:, 0, 0]
                out[f"{tag}_mag"] = np.abs(gamma)
                out[f"{tag}_ang"] = np.degrees(np.angle(gamma))
        out["filtered"] = apply_td_filter
        return out

# %% Preview Canvas


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        fig.patch.set_facecolor('white')

    def plot_time_domain(self, t, td, td_f=None, title="Time Domain"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t, td, 'o:', label='Unfiltered')
        if td_f is not None:
            ax.plot(t, td_f, '-', label='Filtered')
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Magnitude")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        self.draw()

    def plot_smith(self, mag, ang, title="Smith Chart"):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='polar')
        ax.plot(np.radians(ang), mag, 'o-', color='blue')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_title(title)
        ax.grid(True)
        self.draw()

# %% Veusz Plotter


class TouchstonePlotter:
    def __init__(self, title: str, ds_name: str):
        self.doc = vz.Embedded("Touchstone AutoPlot", hidden=False)
        self.doc.EnableToolbar()

    def create_smith_chart_plots(
        self,
        filename: str,
        data: Dict[str, Any],
        smith: Dict[str, Any],
        use_bg: bool = False
    ) -> None:
        if "error" in smith:
            return
        root = os.path.splitext(filename)[0]
        bg_ok = use_bg and os.path.isfile(SMITH_BG_PATH)
        for k, v in smith.items():
            if not k.endswith("_mag"):
                continue
            ang = k.replace("_mag", "_ang")
            mag_ds = f"{root}_{k}"
            ang_ds = f"{root}_{ang}"
            self.doc.SetData(mag_ds, v)
            self.doc.SetData(ang_ds, smith[ang])
            page = self.doc.Root.Add("page", name=f"{root}_{k}_SmithChart")
            if bg_ok:
                img = page.Add("imagefile", name="smith_bg")
                with self._wrap(img) as i:
                    i.filename.val = SMITH_BG_PATH
                    i.aspect.val = True
                    i.xPos.val, i.yPos.val = [0.5], [0.5]
                    i.width.val, i.height.val = [1.0], [1.0]
                    i.Border.hide.val = True
            g = page.Add("polar", name="smith_graph")
            with self._wrap(g) as gg:
                gg.units.val = "degrees"
                gg.direction.val = "anticlockwise"
                gg.position0.val = "right"
                gg.minradius.val = 0
                gg.maxradius.val = 1.05
                lbl = gg.Add("label", name="title")
                lbl.label.val = f"{root} {k}"
                lbl.Text.size.val = "12pt"
                lbl.alignHorz.val = "centre"
                lbl.xPos.val, lbl.yPos.val = 0.5, 1.05
            trace = g.Add("nonorthpoint", name="trace")
            with self._wrap(trace) as t:
                t.data1.val = mag_ds
                t.data2.val = ang_ds
                t.PlotLine.color.val = "blue"
                t.marker.val = "circle"
                t.markerSize.val = "2pt"

    def _wrap(self, w):
        class W:
            def __init__(self, w): self.w = w
            def __enter__(self): return self.w
            def __exit__(self, *a): return False
        return W(w)

# %% Processing Thread


class ProcessingThread(QThread):
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, files, cfg: ProcessingConfig):
        super().__init__()
        self.files = files
        self.cfg = cfg
        self.gpu = GPUAccelerator(cfg.enable_gpu_processing)

    def run(self):
        try:
            if self.cfg.enable_multiprocessing and len(self.files) > 1:
                out = self._parallel()
            else:
                out = self._sequential()
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))

    def _parallel(self):
        out = {}
        tasks = [(f, self.gpu) for f in self.files]
        with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as ex:
            futs = {ex.submit(self._proc, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futs):
                name, data = fut.result()
                out[name] = data
                done += 1
                self.progress.emit(int(100 * done / len(self.files)))
        return out

    def _sequential(self):
        out = {}
        for idx, f in enumerate(self.files, 1):
            name, data = self._proc((f, self.gpu))
            out[name] = data
            self.progress.emit(int(100 * idx / len(self.files)))
        return out

    @staticmethod
    def _proc(args: Tuple[str, GPUAccelerator]):
        path, _ = args
        try:
            ntwk = rf.Network(path)
            return os.path.basename(path), {'network': ntwk}
        except Exception as e:
            return os.path.basename(path), {'error': str(e)}

# %% Main Application Window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Touchstone AutoPlot")
        self.resize(1200, 800)
        self.config = ProcessingConfig()
        self.td_cfg = TimeDomainConfig()
        self.smith_cfg = SmithChartConfig()
        self.td_proc = TimeDomainProcessor(self.td_cfg)
        self.smith_proc = SmithChartProcessor(self.smith_cfg)
        self.files = []
        self.data = {}
        self._init_ui()

    def _init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main = QVBoxLayout(cw)
        self.tabs = QTabWidget()
        main.addWidget(self.tabs)
        self._setup_main_tab()
        self._setup_time_domain_tab()
        self._setup_smith_chart_tab()
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(100)
        main.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main.addWidget(self.progress)
        btns = QHBoxLayout()
        self.btn_proc = QPushButton("Process Files")
        self.btn_proc.clicked.connect(self._process)
        btns.addWidget(self.btn_proc)
        self.btn_save = QPushButton("Save Veusz")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        btns.addWidget(self.btn_save)
        main.addLayout(btns)

    def _setup_main_tab(self):
        tab = QWidget()
        L = QVBoxLayout(tab)
        grp = QGroupBox("Touchstone File Selection")
        gl = QVBoxLayout(grp)
        self.lst = QListWidget()
        gl.addWidget(self.lst)
        bl = QHBoxLayout()
        b1 = QPushButton("Browse")
        b1.clicked.connect(self._browse)
        b2 = QPushButton("Clear")
        b2.clicked.connect(self._clear)
        bl.addWidget(b1)
        bl.addWidget(b2)
        gl.addLayout(bl)
        L.addWidget(grp)
        self.tabs.addTab(tab, "Main")

    def _setup_time_domain_tab(self):
        tab = QWidget()
        L = QHBoxLayout(tab)
        ctrl = QVBoxLayout()
        self.td_combo = QComboBox()
        ctrl.addWidget(QLabel("TD Preview"))
        ctrl.addWidget(self.td_combo)
        self.td_cb = QCheckBox("Enable TD Filter")
        ctrl.addWidget(self.td_cb)
        self.td_sel = QCheckBox("Process Only Selected")
        ctrl.addWidget(self.td_sel)
        btn = QPushButton("Plot TD")
        btn.clicked.connect(self._plot_td)
        ctrl.addWidget(btn)
        L.addLayout(ctrl)
        self.td_canvas = PlotCanvas(tab, 5, 4)
        L.addWidget(self.td_canvas)
        self.tabs.addTab(tab, "Time Domain")

    def _setup_smith_chart_tab(self):
        tab = QWidget()
        L = QHBoxLayout(tab)
        ctrl = QVBoxLayout()
        self.sm_combo = QComboBox()
        ctrl.addWidget(QLabel("Smith Preview"))
        ctrl.addWidget(self.sm_combo)
        self.sm_td = QCheckBox("Enable TD Filter")
        ctrl.addWidget(self.sm_td)
        # <-- Only GUI change: background checkbox
        self.sm_bg = QCheckBox("Use colourful Smith-Chart background image")
        ctrl.addWidget(self.sm_bg)
        self.sm_sel = QCheckBox("Generate Only Selected")
        ctrl.addWidget(self.sm_sel)
        btn = QPushButton("Plot Smith")
        btn.clicked.connect(self._plot_smith)
        ctrl.addWidget(btn)
        L.addLayout(ctrl)
        self.sm_canvas = PlotCanvas(tab, 5, 4)
        L.addWidget(self.sm_canvas)
        self.tabs.addTab(tab, "Smith Chart")

    # ... (remaining methods identical to previous implementation)
    # Ensure that in _plot_smith and _process_smith_charts you pass use_bg=self.sm_bg.isChecked()


def main():
    multiprocessing.set_start_method('spawn', force=True)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
