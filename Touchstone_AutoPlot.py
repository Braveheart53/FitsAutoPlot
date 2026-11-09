#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
Touchstone_AutoPlot.py

Batch Touchstone (.s1p ... .sNp) processor with a Qt GUI:

    * magnitude (dB) and wrapped / unwrapped phase of every S-parameter,
    * time-domain (band-pass) response and time gating with scikit-rf
      (``Network.windowed`` + ``skrf.time.time_gate``), auto or manual gate,
      band-pass or band-stop, gated response shown in the time domain and
      back in the frequency domain on the measured frequency axis,
    * Smith charts (scikit-rf) with PNG / SVG / PDF / interactive HTML
      export; combined PDF with a file > parameter bookmark hierarchy;
      Smith chart pages in the Veusz project (0.7.0),
    * native Veusz project (.vszh5) written with h5py through
      ``_autoplot_native.NativeDocument`` - Veusz is NOT needed to create
      the file, only (optionally) to view it.

Created on 2025-06-29

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later (GNU General Public License v3.0 or later, see LICENSE)
Compatible with: Python 3.8+, qtpy 2.x (PySide6 preferred), scikit-rf,
                 numpy, scipy, matplotlib, h5py
Optional: cupy (GPU time-domain transform), pypdf or PyPDF2 (bookmarked PDF),
          mpld3 (interactive HTML Smith charts), Veusz (viewer only)

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.7.0: Smith charts in the Veusz project (writer 0.0.26)
Date: 2026-11-09
    Added:
      * "Smith chart pages in the Veusz project" (Smith settings, on by
        default): Prepare Smith Charts also adds a page "Smith_<file>" per
        file with its reflection parameters (S11, S22, ...) at the reference
        impedance, drawn with ordinary Veusz widgets by vszh5_writer 0.0.26
        (aspect-locked graph, resistance / reactance or conductance /
        susceptance grid, labels and VSWR circles as chosen, legend beside
        the chart).  Graph "<file>_Smith_Graph"; traces named after their
        Im(Gamma) dataset "<file>_<Sii>_gamma_im" (Re in "_re", frequency in
        "_freq_Hz").  The Plots tab draws these pages as interactive Smith
        charts (f, Gamma, Z, VSWR at the cursor).
      * Writer 0.0.26 / _autoplot_native 0.4.0 / _autoplot_pyqtgraph 0.3.0.

# %%%% 0.6.1: Names as in 0.4.x, traces named after their data, writer 0.0.25
Date: 2026-11-09
    Changed:
      * Graph names of the time-gated pages are those of the 0.4.x script
        again: "<file>_<Sij>_TG_Graph" (time domain) and
        "<file>_<Sij>_TG_Freq_Graph" (gated frequency response); 0.5.0 had
        shortened them to _TD / _FD.  Every trace is named after its y
        dataset ("<file>_S11_mag_dB", "<file>_S21_td_gated_dB", ...)
        instead of "S11_mag", "S21_td" ..., so names are unique and say
        what they draw.
      * Page notes: "Plot title", "Source: <file>", the Touchstone header
        (comment and option lines) and a summary line (ports, points,
        frequency span, z0), built with vszh5_writer.header_notes.
      * Writer 0.0.25 / _autoplot_native 0.3.0.

# %%%% 0.6.0: pyqtgraph previews and Plots tab, suite theme, writer 0.0.24
Date: 2026-11-05
    Changed:
      * Time-domain preview drawn with pyqtgraph (shared
        _autoplot_pyqtgraph.TracePlot): time response (ungated / gated, dB)
        with the gate as a shaded region, and the measured / gated
        frequency response below, both with values at the cursor.  In
        Manual gate mode the region can be dragged; the gate centre and
        span boxes follow.
      * Smith chart preview drawn with pyqtgraph (SmithPlot): impedance or
        admittance grid, optional labels and VSWR circles (scikit-rf's 1.5,
        2, 3, 5), readout of f, Gamma, Z and VSWR under the cursor.
        Matplotlib (Agg, no Qt backend) is used only for the Smith chart
        files (PNG / SVG / PDF with bookmarks / mpld3 HTML).  Without
        pyqtgraph the matplotlib previews are used as before.
      * New "Plots" tab: every page of the Veusz project (frequency and
        time-gated pages) in pyqtgraph.
      * Theme: the suite-wide light / dark setting shared with the launcher
        and the other tools (_suite_gui), instead of a Touchstone-only
        setting; the previews follow the theme.
      * The .vszh5 is written by vszh5_writer 0.0.24 through
        _autoplot_native 0.2.0 (atomic save, gzip for large arrays).
      * Open in Veusz and the new File > Export PDF / PNG (Veusz) use the
        suite helpers.

# %%%% 0.5.0: scikit-rf time domain and gating, native .vszh5, defect fixes
Date: 2026-11-05
    Fixed:
      * Every check box was read with ``state == Qt.CheckState.Checked``.
        PySide6 emits an int (2) and the comparison is always False, so
        ticking "Unwrap Phase" turned unwrapping OFF, and ticking/unticking
        the time / frequency plot options left the "Generate Time-Gated
        Plots" button disabled.  All options now use ``toggled(bool)``.
      * Smith chart export referenced an undefined ``output_path_normalized``
        (NameError), so every PNG / SVG / PDF / HTML export failed.
      * Header reader stored the whole data block as page notes when the
        numeric lines start with spaces (common).  It now stops at the
        first numeric line.
      * Gated frequency response was plotted against ``fftfreq`` (a
        baseband axis starting at 0 Hz and wrapping negative) instead of the
        measured frequency axis; it now comes from ``skrf.time.time_gate``
        on the original axis, beside the ungated trace.
      * Unwrapped phase was clipped to +/-360 deg; magnitude to -80..20 dB.
        Both axes are now Auto.
      * Reference impedance box was not connected; it now renormalizes the
        Smith chart networks.
      * Invalid colour name 'mediumcyan'; duplicate file names from
        different folders overwrote each other.
      * QT_API is set before any Qt import, and qtpy is imported before the
        matplotlib Qt backend, so both always use the same binding.
    Changed:
      * Time domain: ``Network.windowed().s_time`` convention (band-pass,
        t = 0 centred) via a vectorized NumPy / optional CuPy transform that
        is checked to match scikit-rf exactly; time gating with
        ``skrf.time.time_gate`` per S-parameter (it gates one-ports).
        Gate modes: Auto (peak, set span), Auto (peak and span; falls back
        to the set span when scikit-rf cannot find two peaks), Manual.
        Mode band-pass / band-stop is now applied (was ignored).  Separate
        gate-shape window.  Non-uniform frequency grids (including rounding
        in the file) are re-gridded to a uniform grid first.
      * Veusz output is native .vszh5 (NativeDocument); no embedded Veusz
        window.  Falls back to veusz.embed only if _autoplot_native.py is
        missing.  Time-domain traces are in dB.  Legends added.
      * Batch time gating and Smith export run in a worker thread with a
        process pool; the GUI stays responsive.
      * Menu bar: File, View (Light Mode / Dark Mode), Help > About (NRAO).
        Tighter main-tab layout (less blank space).
      * Combined Smith PDF: two-level bookmarks (file > S-parameter), pypdf
        preferred, PyPDF2 accepted.
    Removed:
      * Unused ``create_time_domain_plots``, ``apply_gpu_acceleration`` and
        ``create_short_temp_dir``; Taichi / PyOpenCL probing (no FFT use).

# %%%% 0.4.0: "V4" - Windows long-path support, Python 3.12
Date: 2026-02-03
    Extended-length (\\\\?\\) paths for reading, path-length validation.
    (V1 - V3 and dated snapshots are in Archive/.)
=============================================================================
"""
# %% Import Modules
from __future__ import annotations

import datetime
import gc
import html
import inspect
import multiprocessing
import os
import platform
import re
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Prefer PySide6; an explicit QT_API from the user wins.  Must precede every
# Qt import (qtpy and matplotlib both read it).
os.environ.setdefault("QT_API", "pyside6")

# %%% Scientific
import numpy as np
from scipy.signal import get_window
import skrf
from skrf import Frequency, Network
from skrf import time as skrf_time

# %%% Qt (qtpy first so matplotlib's Qt backend picks the same binding)
if getattr(sys, "frozen", False):
    from PySide6.QtCore import Qt, QThread, Signal, qVersion
    from PySide6.QtGui import QAction, QActionGroup, QKeySequence
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
        QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
        QTabWidget, QTextEdit, QVBoxLayout, QWidget)
    QT_BINDING = "PySide6"
else:
    import qtpy
    from qtpy.QtCore import Qt, QThread, Signal, qVersion
    from qtpy.QtGui import QKeySequence
    try:
        from qtpy.QtGui import QAction, QActionGroup
    except ImportError:  # pragma: no cover - qtpy 1.x
        from qtpy.QtWidgets import QAction, QActionGroup
    from qtpy.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
        QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
        QTabWidget, QTextEdit, QVBoxLayout, QWidget)
    QT_BINDING = qtpy.API_NAME

# %%% Plotting
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# %%% Interactive previews and the Plots tab (pyqtgraph); suite theme / About
import _suite_gui as suite
try:
    import _autoplot_pyqtgraph as autoplot_pg
    autoplot_pg.configure(antialias=True, wheel_needs_ctrl=True)
    PYQTGRAPH_ERROR = ""
except Exception as _exc:  # pragma: no cover - pyqtgraph missing
    autoplot_pg = None
    PYQTGRAPH_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

# %%% Optional modules
try:
    import mpld3
    MPLD3_AVAILABLE = True
except ImportError:
    mpld3 = None
    MPLD3_AVAILABLE = False

try:  # pypdf is the maintained successor of PyPDF2
    from pypdf import PdfReader, PdfWriter
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        PdfReader = PdfWriter = None
        PDF_LIBRARY = None

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()      # raises when no usable GPU
    GPU_AVAILABLE = "cupy"
except Exception:
    cp = None
    GPU_AVAILABLE = None

# %%% Veusz document: native h5py writer (preferred) or embedded Veusz
try:
    from _autoplot_native import NativeDocument
    VEUSZ_BACKEND = "native"
except ImportError:  # pragma: no cover - script copied without the module
    NativeDocument = None
    try:
        import veusz.embed as vz
        VEUSZ_BACKEND = "embedded"
    except ImportError:
        vz = None
        VEUSZ_BACKEND = None


# %% Application identity
APP_NAME = "Touchstone AutoPlot"
__version__ = "0.7.0"
__author__ = "William W. Wallace"
__email__ = "wwallace@nrao.edu"
__company__ = "NRAO / Green Bank Observatory (GBO)"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (C) 2025-2026 William W. Wallace"
ORGANIZATION_NAME = "NRAO-GBO"
FACILITY_NOTE = ("The National Radio Astronomy Observatory and the Green Bank "
                 "Observatory are facilities of the U.S. National Science "
                 "Foundation operated under cooperative agreement by "
                 "Associated Universities, Inc.")
GPL_NOTICE = (
    "This program is free software: you can redistribute it and/or modify it "
    "under the terms of the GNU General Public License as published by the "
    "Free Software Foundation, either version 3 of the License, or (at your "
    "option) any later version. This program is distributed in the hope that "
    "it will be useful, but WITHOUT ANY WARRANTY; without even the implied "
    "warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the "
    "GNU General Public License for more details: "
    "<a href='https://www.gnu.org/licenses/'>https://www.gnu.org/licenses/</a>.")

DB_FLOOR = 1e-12            # |x| floor before 20*log10 (-240 dB)
TD_DISPLAY_RANGE_DB = 120.0  # time-domain y axis spans peak - 120 dB .. Auto
UNIFORM_RTOL = 1e-9         # steps within this are taken as exactly uniform
NONUNIFORM_NOTE_RTOL = 1e-3  # above this the re-gridding is reported
CPU_COUNT = multiprocessing.cpu_count()


def make_process_pool(max_workers: int) -> ProcessPoolExecutor:
    """
    Process pool with the 'spawn' start method on every platform.

    'fork' (the Linux default before Python 3.14) can deadlock when the
    parent has Qt / matplotlib threads running; 'spawn' is what Windows uses
    anyway, so behaviour is identical everywhere.
    """
    return ProcessPoolExecutor(max_workers=max(1, int(max_workers)),
                               mp_context=multiprocessing.get_context("spawn"))


# %% Windows long-path support (from 0.4.0)
MAX_PATH_WINDOWS = 260
PATH_WARNING_THRESHOLD = 200
EXTENDED_PATH_PREFIX = "\\\\?\\"


def is_windows() -> bool:
    """True on Windows."""
    return platform.system() == "Windows"


def get_extended_path(path: Union[str, Path]) -> Path:
    """
    Convert a path to Windows extended-length form (\\\\?\\) if needed.

    Extended paths may be up to 32,767 characters.  Used for READING input
    files with Python's own I/O; output written through HDF5 uses the plain
    path (see ``TouchstonePlotter.save``).
    """
    path_obj = Path(path).resolve()
    if not is_windows():
        return path_obj
    path_str = str(path_obj)
    if path_str.startswith(EXTENDED_PATH_PREFIX):
        return path_obj
    if path_str.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + path_str[2:])
    if path_obj.is_absolute():
        return Path(EXTENDED_PATH_PREFIX + path_str)
    return path_obj


def normalize_path(path: Union[str, Path]) -> Path:
    """Resolved path; extended-length form on Windows."""
    return get_extended_path(path) if is_windows() else Path(path).resolve()


def validate_path_length(path: Union[str, Path],
                         operation: str = "operation") -> Tuple[bool, str]:
    """
    Check a path against the classic Windows MAX_PATH limit.

    Returns
    -------
    (bool, str)
        (usable, message); message is empty when nothing needs saying.
    """
    path_str = str(path)
    if path_str.startswith(EXTENDED_PATH_PREFIX):
        return True, ""
    length = len(path_str)
    if is_windows() and length >= MAX_PATH_WINDOWS:
        return False, (
            "ERROR: Path too long for %s (%d/%d chars).\nUse shorter folder "
            "names or enable Windows long paths (HKLM\\SYSTEM\\"
            "CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled=1).\n"
            "Path: %s" % (operation, length, MAX_PATH_WINDOWS, path_str))
    if is_windows() and length >= PATH_WARNING_THRESHOLD:
        return True, ("Warning: path approaching the limit (%d/%d chars) for "
                      "%s." % (length, MAX_PATH_WINDOWS, operation))
    return True, ""


def truncate_filename(filename: str, max_length: int = 50) -> str:
    """Shorten a file name to ``max_length`` keeping the extension."""
    if len(filename) <= max_length:
        return filename
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
        available = max_length - len(ext) - 4
        if available > 0:
            return "%s...%s" % (name[:available], ext)
        return filename[:max_length]
    return filename[:max_length - 3] + "..."


# %% Configuration classes
@dataclass
class ProcessingConfig:
    """File processing options."""

    enable_multiprocessing: bool = True
    enable_gpu_processing: bool = GPU_AVAILABLE is not None
    num_processes: int = CPU_COUNT
    max_workers: int = CPU_COUNT
    unwrap_phase: bool = False
    use_extended_paths: bool = is_windows()
    validate_path_lengths: bool = True


GATING_MODES = {
    "Auto Gate (peak, set span)": "auto_peak",
    "Auto Gate (peak and span)": "auto_full",
    "Manual Gate": "manual",
}
WINDOW_TYPES = ("kaiser", "hamming", "hann", "blackman", "boxcar")


@dataclass
class TimeDomainConfig:
    """
    Time-domain transform and gate settings.

    ``window_*`` is the frequency-domain window applied before the inverse
    FFT (display and ``fft_window`` of the gate); ``gate_window_*`` is the
    shape of the time gate itself.  Times are in ns.

    ``gate_use_fft_window`` passes the transform window to scikit-rf's gate as
    ``fft_window``.  scikit-rf divides the gated spectrum by that window
    afterwards, which magnifies the gate's truncation error at the band
    edges (where the window is small) and is undefined for windows that
    reach zero there (hann, blackman).  Off by default: on the example
    S-parameter files the band-edge error was lower without it (S21, 1 ns
    gate: 2.8 dB vs 6.8 dB) and the mid-band error equal or lower.
    """

    window_type: str = "kaiser"
    window_param: float = 6.0
    gate_window_type: str = "kaiser"
    gate_window_param: float = 6.0
    gate_center: float = 0.0
    gate_span: float = 1.0
    gate_use_fft_window: bool = False
    mode: str = "bandpass"
    gating_type: str = "auto_peak"
    plot_time_domain: bool = True
    plot_frequency_domain: bool = True


@dataclass
class SmithChartConfig:
    """Smith chart options."""

    chart_type: str = "z"
    draw_labels: bool = True
    draw_vswr: bool = True
    reference_impedance: float = 50.0


# %% Colour map (valid SVG / matplotlib names; Veusz accepts the same)
def get_sparam_color(param_name: str) -> str:
    """
    Consistent colour for an S-parameter across every plot.

    Parameters
    ----------
    param_name : str
        S-parameter name, e.g. "S11", "S21".

    Returns
    -------
    str
        Colour name; 'auto' for parameters outside the table.
    """
    color_map = {
        'S11': 'blue', 'S22': 'red', 'S33': 'green', 'S44': 'magenta',
        'S55': 'orange', 'S66': 'brown', 'S77': 'pink', 'S88': 'grey',
        'S21': 'darkblue', 'S31': 'darkgreen', 'S41': 'darkmagenta',
        'S51': 'darkorange', 'S61': 'darkred', 'S71': 'darkslategray',
        'S81': 'darkviolet',
        'S12': 'cyan', 'S32': 'lightgreen', 'S42': 'lightseagreen',
        'S52': 'goldenrod', 'S62': 'lightcoral', 'S72': 'cadetblue',
        'S82': 'lightslategray',
        'S13': 'indigo', 'S23': 'turquoise', 'S43': 'maroon',
        'S53': 'mediumblue', 'S63': 'darkcyan', 'S73': 'mediumpurple',
        'S83': 'mediumseagreen',
        'S14': 'navy', 'S24': 'olive', 'S34': 'orchid', 'S54': 'sienna',
        'S64': 'peru', 'S74': 'plum', 'S84': 'steelblue',
        'S15': 'purple', 'S25': 'rosybrown', 'S35': 'royalblue',
        'S45': 'saddlebrown', 'S65': 'salmon', 'S75': 'sandybrown',
        'S85': 'seagreen',
    }
    return color_map.get(param_name, 'auto')


# %% Small helpers
def sanitize_name(text: str) -> str:
    """Veusz-safe dataset / widget / tag stem (letters, digits, '_')."""
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    return stem or "dataset"


def s_param_names(nports: int) -> List[Tuple[int, int, str]]:
    """[(i, j, 'S<i+1><j+1>'), ...] in row-major order."""
    return [(i, j, "S%d%d" % (i + 1, j + 1))
            for i in range(nports) for j in range(nports)]


def to_db(values: np.ndarray) -> np.ndarray:
    """20*log10|x| with a floor so silence never becomes -inf; inf/NaN -> NaN."""
    magnitude = np.abs(values)
    db = 20.0 * np.log10(np.maximum(magnitude, DB_FLOOR))
    return np.where(np.isfinite(magnitude), db, np.nan)


def td_axis_floor(*traces: np.ndarray) -> float:
    """
    Lower y limit (dB) for time-domain plots: peak minus TD_DISPLAY_RANGE_DB,
    rounded down to 10 dB.  Outside the gate the gated trace is exactly zero,
    so an Auto axis would otherwise run down to the -240 dB floor.
    """
    peak = max(float(np.nanmax(to_db(t))) for t in traces)
    return float(np.floor((peak - TD_DISPLAY_RANGE_DB) / 10.0) * 10.0)


_NUMERIC_START = re.compile(r"^[+\-.]?\d")


def read_touchstone_header(path: Union[str, Path], max_lines: int = 500) -> List[str]:
    """
    Comment ('!') and option ('#') lines, plus any keyword lines, up to the
    first numeric data line.

    The 0.4.0 reader tested ``line[0].isdigit()`` and therefore read the
    whole file when data lines start with spaces.
    """
    header = []
    with open(path, "r", encoding="ascii", errors="ignore") as fh:
        for count, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            if _NUMERIC_START.match(stripped) or count >= max_lines:
                break
            header.append(stripped)
    return header


def ensure_uniform_network(network: Network) -> Tuple[Network, str]:
    """
    Return a network on an exactly uniform frequency grid.

    The FFT behind the time domain assumes equal steps.  Touchstone files
    often print frequencies with few digits, so steps differ slightly; such
    grids (and genuinely non-uniform ones) are linearly re-gridded with the
    same start, stop and point count.

    Returns
    -------
    (Network, str)
        Network to use and a note ('' unless the grid was clearly
        non-uniform).
    """
    f = network.frequency.f
    if len(f) < 3:
        raise ValueError("At least 3 frequency points are needed for a "
                         "time-domain transform (got %d)." % len(f))
    steps = np.diff(f)
    median = float(np.median(steps))
    if median <= 0:
        raise ValueError("Frequencies must increase.")
    deviation = float(np.max(np.abs(steps - median)) / median)
    if deviation <= UNIFORM_RTOL:
        return network, ""
    uniform = Frequency(f[0], f[-1], len(f), unit="hz")
    regridded = network.interpolate(uniform, kind="linear")
    note = ""
    if deviation > NONUNIFORM_NOTE_RTOL:
        note = ("non-uniform frequency steps (max deviation %.2g of the "
                "step); re-gridded linearly to %d uniform points"
                % (deviation, len(f)))
    return regridded, note


# %% Time-domain engine
def resolve_window(window_type: str, param: float) -> Union[str, Tuple[str, float]]:
    """scipy.signal.get_window specification for a GUI window choice."""
    if window_type == "kaiser":
        return ("kaiser", float(param))
    return window_type


def time_domain_vectorized(s: np.ndarray, f: np.ndarray, window: Any,
                           use_gpu: bool = False) -> np.ndarray:
    """
    Windowed band-pass time-domain response of every S-parameter at once.

    Identical to scikit-rf's ``Network.windowed(window).s_time``:
    the window is normalized to keep sum|S| per parameter, and
    ``fftshift(ifft(ifftshift(S)))`` puts t = 0 at the centre sample.

    Parameters
    ----------
    s : ndarray, shape (N, P, P)
        S-parameters on a uniform grid.
    f : ndarray, shape (N,)
        Frequencies, Hz (only f[0] is used, to choose a DC-centred window).
    window : str or tuple
        scipy window specification.
    use_gpu : bool
        Use CuPy for the transform when available.

    Returns
    -------
    ndarray, shape (N, P, P), complex
    """
    n = s.shape[0]
    if f[0] == 0:
        win = get_window(window, 2 * n)[n:]
    else:
        win = get_window(window, n)
    xp = cp if (use_gpu and cp is not None) else np
    s_dev = xp.asarray(s)
    windowed = s_dev * xp.asarray(win)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = xp.sum(xp.abs(s_dev), axis=0) / xp.sum(xp.abs(windowed), axis=0)
    norm = xp.where(xp.isfinite(norm), norm, 1.0)
    windowed = windowed * norm
    td = xp.fft.fftshift(xp.fft.ifft(xp.fft.ifftshift(windowed, axes=0),
                                     axis=0), axes=0)
    return cp.asnumpy(td) if xp is not np else td


def one_port(network: Network, i: int, j: int) -> Network:
    """S_ij of ``network`` as a one-port (scikit-rf gates one-ports only)."""
    return Network(frequency=network.frequency,
                   s=network.s[:, i, j].reshape(-1, 1, 1),
                   z0=network.z0[:, j].reshape(-1, 1),
                   name="%s_S%d%d" % (network.name or "net", i + 1, j + 1))


_TIME_GATE_PARAMS = set(inspect.signature(skrf_time.time_gate).parameters)


def call_time_gate(network_1port: Network, center_ns: Optional[float],
                   span_ns: Optional[float], mode: str, gate_window: Any,
                   fft_window: Any) -> Network:
    """
    ``skrf.time.time_gate`` with only the arguments this scikit-rf supports.

    Times are given in ns (``t_unit='ns'``); 'fft' (band-pass) method.
    """
    kwargs = {"center": center_ns, "span": span_ns, "mode": mode,
              "window": gate_window}
    if "t_unit" in _TIME_GATE_PARAMS:
        kwargs["t_unit"] = "ns"
    if "method" in _TIME_GATE_PARAMS:
        kwargs["method"] = "fft"
    if "fft_window" in _TIME_GATE_PARAMS:
        kwargs["fft_window"] = fft_window
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return skrf_time.time_gate(network_1port, **kwargs)


def compute_time_domain(network: Network, td_config: Dict[str, Any],
                        use_gpu: bool = False,
                        params: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Time-domain response and time gating of every (or selected) S-parameter.

    Parameters
    ----------
    network : Network
        Measured network.
    td_config : dict
        ``asdict(TimeDomainConfig)``.
    use_gpu : bool
        CuPy for the un-gated transform.
    params : sequence of str, optional
        Only these parameters (e.g. ["S11"]); default all.

    Returns
    -------
    dict
        ``frequency_hz``, ``time_ns``, ``notes`` (list), ``error`` (None or
        text) and ``params``: {name: {"td", "td_gated", "s", "s_gated",
        "gate_center_ns", "gate_span_ns", "note"}}.
    """
    result: Dict[str, Any] = {"params": {}, "notes": [], "error": None}
    try:
        net, note = ensure_uniform_network(network)
        if note:
            result["notes"].append(note)
        f = net.frequency.f
        t_ns = net.frequency.t_ns
        fft_window = resolve_window(td_config["window_type"],
                                    td_config["window_param"])
        gate_window = resolve_window(td_config["gate_window_type"],
                                     td_config["gate_window_param"])
        gate_fft_window = None
        if td_config.get("gate_use_fft_window"):
            edges = get_window(fft_window, len(f))[[0, -1]]
            if float(np.min(edges)) > 1e-3:
                gate_fft_window = fft_window
            else:
                result["notes"].append(
                    "%s window is zero at the band edges; no window used "
                    "inside the gate" % td_config["window_type"])
        td_all = time_domain_vectorized(net.s, f, fft_window, use_gpu=use_gpu)
        result["frequency_hz"] = f
        result["time_ns"] = t_ns
        gating = td_config["gating_type"]
        for i, j, name in s_param_names(net.nports):
            if params and name not in params:
                continue
            td = td_all[:, i, j]
            op = one_port(net, i, j)
            span = float(td_config["gate_span"])
            entry_note = ""
            if gating == "manual":
                center = float(td_config["gate_center"])
            else:
                center = float(t_ns[int(np.argmax(np.abs(td)))])
                if gating == "auto_full":
                    try:
                        span = float(skrf_time.detect_span(op, t_unit="ns"))
                    except Exception as exc:  # e.g. "Couldnt find 2 peaks"
                        entry_note = ("auto span not found (%s); used the set "
                                      "span %.4g ns" % (exc, span))
            gated = call_time_gate(op, center, span, td_config["mode"],
                                   gate_window, gate_fft_window)
            td_gated = time_domain_vectorized(gated.s, f, fft_window,
                                              use_gpu=use_gpu)[:, 0, 0]
            result["params"][name] = {
                "td": td, "td_gated": td_gated,
                "s": net.s[:, i, j], "s_gated": gated.s[:, 0, 0],
                "gate_center_ns": center, "gate_span_ns": span,
                "note": entry_note,
            }
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    return result


def _td_worker(job: Tuple[str, Network, Dict[str, Any], bool]) -> Tuple[str, Dict[str, Any]]:
    """Process-pool entry point: (key, network, config, gpu) -> (key, result)."""
    key, network, td_config, use_gpu = job
    return key, compute_time_domain(network, td_config, use_gpu=use_gpu)


def run_time_domain_batch(items: List[Tuple[str, Network]],
                          td_config: Dict[str, Any], use_gpu: bool,
                          max_workers: int, use_processes: bool,
                          progress: Callable[[int], None]) -> Dict[str, Dict[str, Any]]:
    """
    Time domain + gating for many files.

    Files run in a process pool (one file per task) unless the GPU is used
    (the GPU already vectorizes over parameters; several processes would
    only contend for it) or there is a single file.
    """
    results: Dict[str, Dict[str, Any]] = {}
    jobs = [(key, net, td_config, use_gpu) for key, net in items]
    total = max(len(jobs), 1)
    if use_processes and not use_gpu and len(jobs) > 1 and max_workers > 1:
        with make_process_pool(min(max_workers, len(jobs))) as pool:
            futures = [pool.submit(_td_worker, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                key, res = future.result()
                results[key] = res
                progress(int(100 * done / total))
    else:
        for done, job in enumerate(jobs, start=1):
            key, res = _td_worker(job)
            results[key] = res
            progress(int(100 * done / total))
    return results


class TimeDomainProcessor:
    """Thin object wrapper over ``compute_time_domain`` (kept for 0.4.0 users)."""

    def __init__(self, config: Optional[TimeDomainConfig] = None):
        """Store the time-domain configuration."""
        self.config = config or TimeDomainConfig()

    def process_network(self, network: Network, use_gpu: bool = False,
                        params: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Time domain and gating of ``network`` with the current settings."""
        return compute_time_domain(network, asdict(self.config),
                                   use_gpu=use_gpu, params=params)


# %% Smith charts
def make_smith_figure(network_1port: Network, title: str, chart_type: str = "z",
                      draw_labels: bool = True, draw_vswr: bool = True,
                      figure: Optional[Figure] = None) -> Figure:
    """
    Smith chart of a one-port with scikit-rf.

    A plain ``Figure`` (no pyplot state) so it is safe in worker processes.
    """
    fig = figure or Figure(figsize=(8, 8))
    if figure is None:
        FigureCanvasAgg(fig)
    fig.clear()
    ax = fig.add_subplot(111)
    try:
        network_1port.plot_s_smith(ax=ax, chart_type=chart_type,
                                   draw_labels=draw_labels,
                                   draw_vswr=draw_vswr or None,
                                   show_legend=True)
        kind = "Impedance" if chart_type == "z" else "Admittance"
        ax.set_title("%s - Smith Chart (%s)" % (title, kind),
                     fontsize=12, fontweight="bold")
    except Exception as exc:  # fall back to the complex plane
        ax.clear()
        s = network_1port.s[:, 0, 0]
        ax.plot(s.real, s.imag, marker="o", markersize=2)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("%s - complex plane (Smith chart failed: %s)" % (title, exc))
    return fig


def export_smith_worker(job: Tuple[Network, str, str, str, str, bool, bool]) -> Tuple[str, bool, str]:
    """
    Process-pool entry point that writes one Smith chart file.

    Parameters
    ----------
    job : tuple
        (network_1port, title, chart_type, output_path, fmt, draw_labels,
        draw_vswr); fmt is png, svg, pdf or html.

    Returns
    -------
    (str, bool, str)
        (output_path, success, message)
    """
    network, title, chart_type, output_path, fmt, labels, vswr = job
    try:
        fig = make_smith_figure(network, title, chart_type, labels, vswr)
        fmt = fmt.lower()
        if fmt == "html":
            if not MPLD3_AVAILABLE:
                output_path = os.path.splitext(output_path)[0] + ".png"
                fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
                return output_path, True, "mpld3 missing: wrote PNG instead"
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(mpld3.fig_to_html(fig))
        elif fmt in ("png", "svg", "pdf"):
            kwargs = {"dpi": 150} if fmt == "png" else {}
            fig.savefig(output_path, format=fmt, bbox_inches="tight", **kwargs)
        else:
            return output_path, False, "unknown format %s" % fmt
        return output_path, True, ""
    except Exception as exc:
        return output_path, False, "%s: %s" % (type(exc).__name__, exc)


class SmithChartPlottermpld3:
    """Smith chart export helpers (name kept from 0.4.0)."""

    def __init__(self, reference_impedance: float = 50.0):
        """Store the reference impedance."""
        self.reference_impedance = reference_impedance

    @staticmethod
    def export_smith_chart(network: Network, param_name: str, chart_type: str,
                           output_path: str, export_format: str,
                           draw_labels: bool = True, draw_vswr: bool = True) -> bool:
        """Write one Smith chart; True on success."""
        _path, ok, msg = export_smith_worker(
            (network, param_name, chart_type, output_path, export_format,
             draw_labels, draw_vswr))
        if not ok:
            print("Smith chart export failed (%s): %s" % (output_path, msg))
        return ok

    @staticmethod
    def create_pdf_with_bookmarks(entries: List[Tuple[str, str, str]],
                                  output_path: str) -> bool:
        """
        Merge single-page PDFs with a two-level outline.

        Parameters
        ----------
        entries : list of (pdf_path, group_title, item_title)
            Pages in order; consecutive entries with the same group share a
            parent bookmark (file > S-parameter).
        output_path : str
            Combined PDF.
        """
        if PdfWriter is None:
            raise RuntimeError("Install pypdf (or PyPDF2) for combined PDFs.")
        writer = PdfWriter()
        parents: Dict[str, Any] = {}
        page_index = 0
        for pdf_path, group, item in entries:
            reader = PdfReader(pdf_path)
            first = page_index
            for page in reader.pages:
                writer.add_page(page)
                page_index += 1
            if group not in parents:
                parents[group] = writer.add_outline_item(group, first)
            writer.add_outline_item(item, first, parent=parents[group])
        try:
            writer.page_mode = "/UseOutlines"
        except Exception:
            pass
        with open(output_path, "wb") as fh:
            writer.write(fh)
        return True


# %% Veusz project builder
class TouchstonePlotter:
    """
    Builds the Veusz project for the processed Touchstone files.

    Uses ``_autoplot_native.NativeDocument`` (h5py, no Veusz needed); the
    widget / dataset calls are the Veusz embedding API, so ``veusz.embed``
    works as a fallback.
    """

    def __init__(self, plot_title: str = "Touchstone S-Parameter Analysis",
                 dataset_name: str = "Touchstone_Dataset",
                 unwrap_phase: bool = False):
        """
        Parameters
        ----------
        plot_title : str
            Stored in each page's notes.
        dataset_name : str
            Extra tag applied to every dataset of this run.
        unwrap_phase : bool
            Unwrap phase with numpy.unwrap (O(N), the cheapest method).
        """
        self.plot_title = plot_title
        self.dataset_name = dataset_name
        self.run_tag = sanitize_name(dataset_name)
        self.unwrap_phase = unwrap_phase
        self._smith_grids: Dict[str, Any] = {}     # 0.7.0: grid datasets written once
        if VEUSZ_BACKEND == "native":
            self.doc = NativeDocument("Touchstone_AutoPlot")
        elif VEUSZ_BACKEND == "embedded":
            self.doc = vz.Embedded("Touchstone_AutoPlot")
        else:
            raise RuntimeError("Neither _autoplot_native.py nor Veusz is "
                               "available; keep _autoplot_native.py next to "
                               "this script.")

    # ----- widget helpers ---------------------------------------------------
    def _tag(self, file_tag: str, names: Sequence[str]):
        """Tag datasets with the file tag and the run tag."""
        self.doc.TagDatasets(file_tag, list(names))
        if self.run_tag and self.run_tag != file_tag:
            self.doc.TagDatasets(self.run_tag, list(names))

    def _remove_page(self, name: str):
        """Remove an existing page so regenerating does not duplicate it."""
        try:
            if VEUSZ_BACKEND == "native":
                self.doc.Root.Remove(name)
            else:
                getattr(self.doc.Root, name).Remove()
        except Exception:
            pass

    @staticmethod
    def _configure_standard_graph(graph, title: str, x_label: str, y_label: str,
                                  y_min: Union[float, str] = "Auto",
                                  y_max: Union[float, str] = "Auto"):
        """Title label, axis labels, grid lines and y range of a graph."""
        graph.Add('label', name='plotTitle')
        graph.topMargin.val = '1cm'
        graph.plotTitle.Text.size.val = '12pt'
        graph.plotTitle.label.val = title
        graph.plotTitle.alignHorz.val = 'centre'
        graph.plotTitle.yPos.val = 1.05
        graph.plotTitle.xPos.val = 0.5
        graph.x.label.val = x_label
        graph.y.label.val = y_label
        graph.x.GridLines.hide.val = False
        graph.y.GridLines.hide.val = False
        graph.x.MinorGridLines.hide.val = False
        graph.y.MinorGridLines.hide.val = False
        graph.y.min.val = y_min
        graph.y.max.val = y_max
        graph.Add('key', name='key')

    @staticmethod
    def _configure_xy_plot(xy, x_data: str, y_data: str, color: str,
                           key: str = "", transparency: int = 0,
                           style: str = "solid", marker: str = "circle"):
        """Line and markers in one colour."""
        xy.xData.val = x_data
        xy.yData.val = y_data
        xy.PlotLine.color.val = color
        xy.PlotLine.width.val = '2pt' if style == "solid" else '1.5pt'
        xy.PlotLine.style.val = style
        xy.PlotLine.transparency.val = transparency
        xy.marker.val = marker
        xy.markerSize.val = '2pt'
        xy.MarkerFill.color.val = color
        xy.MarkerFill.transparency.val = transparency
        xy.MarkerLine.color.val = color
        xy.MarkerLine.width.val = '1pt'
        if key:
            xy.key.val = key

    # ----- pages -------------------------------------------------------------
    def create_plots_from_data(self, key: str, data: Dict[str, Any]):
        """Frequency-domain page (magnitude and phase) of one file."""
        network = data['network']
        base = sanitize_name(key)
        freq_name = "%s_freq_GHz" % base
        self.doc.SetData(freq_name, network.frequency.f / 1e9)
        page_name = "SParam_%s" % base
        self._remove_page(page_name)
        page = self.doc.Root.Add('page', name=page_name)
        page.width.val = '30cm'          # two graphs side by side
        page.height.val = '15cm'
        page.notes.val = self._notes(data)
        grid = page.Add('grid', columns=2)
        graph_mag = grid.Add('graph', name="%s_Mag" % base)
        graph_phase = grid.Add('graph', name="%s_Phase" % base)
        self._configure_standard_graph(graph_mag, "%s - Magnitude" % key,
                                       'Frequency (GHz)', 'Magnitude (dB)')
        unwrap = self.unwrap_phase
        self._configure_standard_graph(
            graph_phase, "%s - Phase%s" % (key, " (unwrapped)" if unwrap else ""),
            'Frequency (GHz)', 'Phase (deg)',
            "Auto" if unwrap else -180.0, "Auto" if unwrap else 180.0)
        names = [freq_name]
        for i, j, param in s_param_names(network.nports):
            s = network.s[:, i, j]
            mag_name = "%s_%s_mag_dB" % (base, param)
            phase_name = "%s_%s_phase_deg%s" % (base, param,
                                                "_unwrapped" if unwrap else "")
            phase = np.angle(s)
            if unwrap:
                phase = np.unwrap(phase)          # O(N), cheapest exact method
            self.doc.SetData(mag_name, to_db(s))
            self.doc.SetData(phase_name, np.degrees(phase))
            names += [mag_name, phase_name]
            color = get_sparam_color(param)
            self._configure_xy_plot(graph_mag.Add('xy', name=mag_name),
                                    freq_name, mag_name, color, key=param)
            self._configure_xy_plot(graph_phase.Add('xy', name=phase_name),
                                    freq_name, phase_name, color, key=param)
        self._tag(base, names)

    def create_time_gated_plots(self, key: str, data: Dict[str, Any],
                                td_result: Dict[str, Any],
                                plot_time_domain: bool = True,
                                plot_frequency_domain: bool = True):
        """One page per S-parameter: time domain and/or gated frequency response."""
        if td_result.get('error'):
            return
        base = sanitize_name(key)
        time_name = "%s_time_ns" % base
        freq_name = "%s_tg_freq_GHz" % base
        self.doc.SetData(time_name, td_result['time_ns'])
        self.doc.SetData(freq_name, td_result['frequency_hz'] / 1e9)
        names = [time_name, freq_name]
        for param, entry in td_result['params'].items():
            page_name = "%s_%s_TimeGated" % (base, param)
            self._remove_page(page_name)
            page = self.doc.Root.Add('page', name=page_name)
            page.notes.val = "%s\nGate: %s, centre %.6g ns, span %.6g ns%s" % (
                self._notes(data), td_result.get('mode', ''),
                entry['gate_center_ns'], entry['gate_span_ns'],
                ("\n" + entry['note']) if entry['note'] else "")
            columns = 2 if (plot_time_domain and plot_frequency_domain) else 1
            page.width.val = '30cm' if columns == 2 else '20cm'
            page.height.val = '15cm'
            grid = page.Add('grid', columns=columns)
            color = get_sparam_color(param)
            if plot_time_domain:
                td_name = "%s_%s_td_dB" % (base, param)
                tdg_name = "%s_%s_td_gated_dB" % (base, param)
                self.doc.SetData(td_name, to_db(entry['td']))
                self.doc.SetData(tdg_name, to_db(entry['td_gated']))
                names += [td_name, tdg_name]
                graph = grid.Add('graph', name="%s_%s_TG_Graph" % (base, param))
                self._configure_standard_graph(
                    graph, "%s - %s time domain (gate %.4g +/- %.4g ns)" % (
                        key, param, entry['gate_center_ns'],
                        entry['gate_span_ns'] / 2.0),
                    'Time (ns)', 'Magnitude (dB)',
                    td_axis_floor(entry['td'], entry['td_gated']), "Auto")
                self._configure_xy_plot(graph.Add('xy', name=td_name),
                                        time_name, td_name, 'grey',
                                        key="%s ungated" % param,
                                        style="dotted", marker="none")
                self._configure_xy_plot(graph.Add('xy', name=tdg_name),
                                        time_name, tdg_name, color,
                                        key="%s gated" % param, transparency=30)
            if plot_frequency_domain:
                fd_name = "%s_%s_fd_dB" % (base, param)
                fdg_name = "%s_%s_fd_gated_dB" % (base, param)
                self.doc.SetData(fd_name, to_db(entry['s']))
                self.doc.SetData(fdg_name, to_db(entry['s_gated']))
                names += [fd_name, fdg_name]
                graph = grid.Add('graph', name="%s_%s_TG_Freq_Graph" % (base, param))
                self._configure_standard_graph(
                    graph, "%s - %s gated frequency response" % (key, param),
                    'Frequency (GHz)', 'Magnitude (dB)')
                self._configure_xy_plot(graph.Add('xy', name=fd_name),
                                        freq_name, fd_name, 'grey',
                                        key="%s measured" % param,
                                        style="dotted", marker="none")
                self._configure_xy_plot(graph.Add('xy', name=fdg_name),
                                        freq_name, fdg_name, color,
                                        key="%s gated" % param, transparency=30)
        self._tag(base, names)

    def create_smith_page(self, key: str, data: Dict[str, Any], networks: Dict[str, Network],
                          chart: "SmithChartConfig") -> str:
        """
        Smith chart page of one file in the Veusz project (0.7.0).

        Parameters
        ----------
        key : str
            File key (as for the other pages).
        data : dict
            Processed file (header, path, network) for the page notes.
        networks : dict
            {parameter: one-port Network at the chart's reference impedance},
            normally the reflection parameters S11, S22, ...
        chart : SmithChartConfig
            Chart type (z / y), labels, VSWR circles and reference impedance.

        Returns
        -------
        str
            Page name "Smith_<file>"; the graph is "<file>_Smith_Graph" and
            each trace is named after its Im(Gamma) dataset
            "<file>_<Sii>_gamma_im" (Re in "_re", frequency in "_freq_Hz").
            The grid (vszh5_writer.smith_grid) is written once per chart
            style and shared by every Smith page of the project.
        """
        import vszh5_writer as vw
        from _autoplot_native import add_smith_page
        base = sanitize_name(key)
        prefix = "smith_%s%s%s" % (chart.chart_type, "_vswr" if chart.draw_vswr else "",
                                   "" if chart.draw_labels else "_nolabels")
        grid = self._smith_grids.get(prefix)
        if grid is None:
            grid = vw.smith_grid(prefix, chart.chart_type, labels=chart.draw_labels,
                                 vswr=vw.SMITH_VSWR if chart.draw_vswr else ())
            grid.write(self.doc)
            self._smith_grids[prefix] = grid
        traces, names = [], list(grid.dataset_names())
        for param, op in networks.items():
            stem = "%s_%s_gamma" % (base, param)
            for name, values in vw.smith_trace_datasets(stem, op.s[:, 0, 0], op.f).items():
                self.doc.SetData(name, values)
                names.append(name)
            colour = get_sparam_color(param)
            traces.append(vw.smith_trace(stem, param, colour="" if colour == "auto" else colour,
                                         width="1.5pt"))
        page_name = "Smith_%s" % base
        self._remove_page(page_name)
        kind = "admittance" if chart.chart_type == "y" else "impedance"
        add_smith_page(self.doc.Root, page_name, grid, traces,
                       title="%s - Smith chart (%s, Z0 = %g ohm)" % (key, kind,
                                                                    chart.reference_impedance),
                       notes="%s\nSmith chart: %s, renormalized to Z0 = %g ohm" % (
                           self._notes(data), ", ".join(networks), chart.reference_impedance),
                       graph_name="%s_Smith_Graph" % base,
                       graph_notes="Z0 = %g ohm" % chart.reference_impedance,
                       key=True)
        self._tag(base, names)
        return page_name

    def _notes(self, data: Dict[str, Any]) -> str:
        """Page notes: run title, source file and the Touchstone header
        (comment and option lines), via vszh5_writer.header_notes."""
        from vszh5_writer import header_notes
        network = data.get('network')
        extra = []
        if network is not None:
            extra.append("%d-port, %d points, %.6g - %.6g GHz, z0 %s ohm" % (
                network.nports, len(network.f), network.f[0] / 1e9, network.f[-1] / 1e9,
                ", ".join("%g" % z for z in np.unique(np.real(network.z0)))))
        return header_notes(list(data.get('header_info', [])), source=data.get('file_path', ''),
                            title="Plot title: %s" % self.plot_title, extra=extra)

    def save(self, filename: str) -> str:
        """
        Save the project as .vszh5 (atomic for the native writer).

        The plain (non-\\\\?\\) path is used: HDF5 is not guaranteed to
        accept extended-length paths.
        """
        target = str(Path(filename).resolve())
        ok, msg = validate_path_length(target, "Veusz project save")
        if not ok:
            raise OSError(msg)
        if msg:
            print(msg)
        if VEUSZ_BACKEND == "native":
            return self.doc.save(target)
        self.doc.Save(target, mode='hdf5')
        return target


# %% Previews: pyqtgraph (matplotlib Qt canvas only when pyqtgraph is missing)
def _matplotlib_canvas(parent, width, height):
    """Fallback matplotlib Qt canvas (imported only when needed)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    fig = Figure(figsize=(width, height), dpi=100, tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setParent(parent)
    return canvas, fig


class TouchstonePlotCanvas(QWidget):
    """
    Time-domain and Smith chart previews.

    pyqtgraph (shared _autoplot_pyqtgraph plots, light / dark) when it is
    installed, otherwise the matplotlib canvas of 0.5.0.  ``gate_dragged``
    (callable(centre_ns, span_ns)) is called when the user drags the gate
    region of a manual gate.
    """

    def __init__(self, parent=None, width=8, height=6, dpi=100, dark=False):
        super().__init__(parent)
        self.dark = bool(dark)
        self.gate_dragged: Optional[Callable[[float, float], None]] = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._plots: List[Any] = []
        self.fig = None
        self._canvas = None
        if autoplot_pg is None:
            self._canvas, self.fig = _matplotlib_canvas(self, width, height)
            self._layout.addWidget(self._canvas)

    # ----- helpers -----------------------------------------------------------
    def _clear(self) -> None:
        for plot in self._plots:
            try:
                plot.dispose()
            except Exception:
                pass
            plot.setParent(None)
            plot.deleteLater()
        self._plots = []

    def _add(self, plot) -> None:
        self._layout.addWidget(plot, 1)
        self._plots.append(plot)
        plot.hovered.connect(lambda x, y, p=plot: p.set_cursor(x, y))
        plot.unhovered.connect(lambda p=plot: p.set_cursor(None))

    def apply_theme(self, dark: bool) -> None:
        self.dark = bool(dark)
        for plot in self._plots:
            plot.apply_theme(self.dark)

    # ----- Smith chart --------------------------------------------------------
    def plot_smith_chart(self, network, title="Smith Chart", chart_type="z",
                         draw_labels=True, draw_vswr=True):
        """Smith chart of a one-port."""
        if self.fig is not None:
            make_smith_figure(network, title, chart_type, draw_labels, draw_vswr,
                              figure=self.fig)
            self._canvas.draw()
            return
        self._clear()
        z0 = float(np.real(np.ravel(network.z0)[0])) if np.size(network.z0) else 50.0
        plot = autoplot_pg.SmithPlot(
            "%s (%s)" % (title, "impedance" if chart_type == "z" else "admittance"),
            chart_type=chart_type, z0=z0, dark=self.dark,
            draw_labels=draw_labels, draw_vswr=draw_vswr)
        plot.add_trace(title.split(" - ")[-1], network.s[:, 0, 0], network.f)
        self._add(plot)

    # ----- time domain -------------------------------------------------------
    def plot_time_domain(self, time_ns, td, td_gated, freq_hz=None, s=None,
                         s_gated=None, gate=None, title="Time Domain", movable=False):
        """Time domain (dB) with the gate shaded; gated frequency response below."""
        if self.fig is not None:
            self._plot_time_domain_mpl(time_ns, td, td_gated, freq_hz, s, s_gated, gate, title)
            return
        self._clear()
        top = autoplot_pg.TracePlot(title, "Time (ns)", "Magnitude (dB)", False, dark=self.dark)
        top.add_trace("ungated", time_ns, to_db(td), "#9399b2", "dotted")
        top.add_trace("gated", time_ns, to_db(td_gated), "#f28e2b")
        floor = td_axis_floor(td, td_gated)
        peak = float(np.nanmax(to_db(np.concatenate([np.ravel(td), np.ravel(td_gated)]))))
        top.plot.setYRange(floor, peak + 5.0, padding=0.02)
        if gate is not None:
            centre, span = gate
            top.set_region(centre - span / 2.0, centre + span / 2.0, movable=movable,
                           on_change=self._region_moved if movable else None)
        top._set_header(None)
        self._add(top)
        if freq_hz is not None and s is not None:
            bottom = autoplot_pg.TracePlot("Frequency response", "Frequency (GHz)",
                                           "Magnitude (dB)", False, dark=self.dark)
            bottom.add_trace("measured", freq_hz / 1e9, to_db(s), "#9399b2", "dotted")
            bottom.add_trace("gated", freq_hz / 1e9, to_db(s_gated), "#4e79a7")
            bottom._set_header(None)
            self._add(bottom)

    def _region_moved(self, start: float, stop: float) -> None:
        if self.gate_dragged is not None and stop > start:
            self.gate_dragged(0.5 * (start + stop), stop - start)

    def _plot_time_domain_mpl(self, time_ns, td, td_gated, freq_hz, s, s_gated, gate, title):
        """0.5.0 matplotlib preview (pyqtgraph missing)."""
        self.fig.clear()
        two = freq_hz is not None and s is not None
        ax = self.fig.add_subplot(211 if two else 111)
        ax.plot(time_ns, to_db(td), ':', color='grey', label="ungated")
        ax.plot(time_ns, to_db(td_gated), '-', linewidth=1.5, alpha=0.8, label="gated")
        if gate is not None:
            center, span = gate
            ax.axvspan(center - span / 2.0, center + span / 2.0, color='orange',
                       alpha=0.15, label="gate")
        ax.set_ylim(bottom=td_axis_floor(td, td_gated))
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        if two:
            ax2 = self.fig.add_subplot(212)
            ax2.plot(freq_hz / 1e9, to_db(s), ':', color='grey', label="measured")
            ax2.plot(freq_hz / 1e9, to_db(s_gated), '-', linewidth=1.5, alpha=0.8,
                     label="gated")
            ax2.set_xlabel("Frequency (GHz)")
            ax2.set_ylabel("Magnitude (dB)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="lower left", fontsize=8)
        self._canvas.draw()


# %% File loading
def process_single_touchstone_file(file_info: Tuple[str, Any]) -> Tuple[str, Optional[dict]]:
    """
    Read one Touchstone file (process-pool entry point).

    Returns
    -------
    (str, dict or None)
        (file path as given, data dict or None on failure).  The data dict
        holds 'network', 'frequency', 'file_path', 'header_info' and
        'error'.
    """
    filepath, _unused = file_info
    try:
        normalized = str(normalize_path(filepath))
        ok, msg = validate_path_length(normalized, "file processing")
        if not ok:
            return filepath, {"error": msg}
        network = Network(normalized)
        try:
            header = read_touchstone_header(normalized)
        except Exception as exc:
            header = ["(header not read: %s)" % exc]
        return filepath, {'network': network, 'frequency': network.frequency.f,
                          'file_path': str(filepath), 'header_info': header,
                          'error': None}
    except Exception as exc:
        return filepath, {"error": "%s: %s" % (type(exc).__name__, exc)}


def unique_keys(paths: Sequence[str]) -> Dict[str, str]:
    """Map each path to a unique display key (file name, then ' (2)', ...)."""
    keys: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for path in paths:
        name = os.path.basename(path)
        count = used.get(name, 0) + 1
        used[name] = count
        keys[path] = name if count == 1 else "%s (%d)" % (name, count)
    return keys


class TouchstoneProcessingThread(QThread):
    """Reads the selected files (process pool for several files)."""

    progress_updated = Signal(int)
    processing_finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, file_list: List[str], config: ProcessingConfig):
        """Store the file list and options."""
        super().__init__()
        self.file_list = list(dict.fromkeys(file_list))   # drop repeated paths
        self.config = config

    def run(self):
        """Read every file; emit {key: data or {'error': ...}} in list order."""
        try:
            raw: Dict[str, dict] = {}
            total = max(len(self.file_list), 1)
            if self.config.enable_multiprocessing and len(self.file_list) > 1:
                workers = max(1, min(self.config.max_workers, len(self.file_list)))
                with make_process_pool(workers) as pool:
                    futures = [pool.submit(process_single_touchstone_file, (f, None))
                               for f in self.file_list]
                    for done, future in enumerate(as_completed(futures), start=1):
                        path, data = future.result()
                        raw[path] = data
                        self.progress_updated.emit(int(100 * done / total))
            else:
                for done, path in enumerate(self.file_list, start=1):
                    _p, data = process_single_touchstone_file((path, None))
                    raw[path] = data
                    self.progress_updated.emit(int(100 * done / total))
            keys = unique_keys(self.file_list)
            self.processing_finished.emit({keys[p]: raw[p] for p in self.file_list})
        except Exception as exc:
            self.error_occurred.emit("%s: %s" % (type(exc).__name__, exc))


class FunctionWorker(QThread):
    """Run ``func(progress_callback)`` off the GUI thread."""

    progress_updated = Signal(int)
    finished_ok = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, func: Callable[[Callable[[int], None]], Any]):
        """Store the callable; it receives a progress callback (0-100)."""
        super().__init__()
        self.func = func

    def run(self):
        """Call the function; report its result or error."""
        try:
            self.finished_ok.emit(self.func(self.progress_updated.emit))
        except Exception as exc:
            self.error_occurred.emit("%s: %s" % (type(exc).__name__, exc))


# %% Theme: _suite_gui (suite-wide light / dark, shared with the launcher)


def _module_version(name: str) -> str:
    """Installed version of a module, or 'not installed'."""
    try:
        module = sys.modules.get(name) or __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return "not installed"


def build_about_content() -> Dict[str, str]:
    """Rich text and runtime details for Help > About."""
    esc = html.escape
    mail = esc(__email__, quote=True)
    text = (
        "<h3>%s</h3><p>Version <b>%s</b> (Semantic Versioning 2.0.0)</p>"
        "<p>Touchstone S-parameter plotting: magnitude/phase, time domain and "
        "time gating (scikit-rf), Smith charts, native Veusz .vszh5 "
        "projects.</p><table cellspacing='4'>"
        "<tr><td><b>Author</b></td><td>%s</td></tr>"
        "<tr><td><b>Company</b></td><td>%s</td></tr>"
        "<tr><td><b>Email</b></td><td><a href='mailto:%s'>%s</a></td></tr>"
        "<tr><td><b>Licence</b></td><td>GNU General Public License v3.0 or "
        "later (%s)</td></tr><tr><td><b>Copyright</b></td><td>%s</td></tr>"
        "</table><p><small>%s</small></p>" % (
            esc(APP_NAME), esc(__version__), esc(__author__), esc(__company__),
            mail, mail, esc(__license__), esc(__copyright__),
            esc(FACILITY_NOTE)))
    runtime = "\n".join([
        "%s %s" % (APP_NAME, __version__),
        "Python     : %s (%s)" % (platform.python_version(), sys.executable),
        "Qt         : %s, Qt %s" % (QT_BINDING, qVersion()),
        "scikit-rf  : %s" % skrf.__version__,
        "numpy      : %s" % np.__version__,
        "scipy      : %s" % _module_version("scipy"),
        "matplotlib : %s" % matplotlib.__version__,
        "h5py       : %s" % _module_version("h5py"),
        "Veusz out  : %s" % (VEUSZ_BACKEND or "unavailable"),
        "GPU        : %s" % (GPU_AVAILABLE or "none (CPU)"),
        "PDF merge  : %s" % (PDF_LIBRARY or "not installed"),
        "mpld3      : %s" % ("available" if MPLD3_AVAILABLE else "not installed"),
        "CPU cores  : %d" % CPU_COUNT,
        "Platform   : %s" % platform.platform(),
    ])
    return {"text": text, "notice": GPL_NOTICE, "runtime": runtime}


# %% Main window
class TouchstoneMainWindow(QMainWindow):
    """Main window: processing, time domain and Smith chart tabs."""

    def __init__(self):
        """Build the window and restore the theme."""
        super().__init__()
        self.setWindowTitle("%s v%s" % (APP_NAME, __version__))
        self.resize(1400, 900)
        self.config = ProcessingConfig()
        self.td_config = TimeDomainConfig()
        self.smith_config = SmithChartConfig()
        self.td_processor = TimeDomainProcessor(self.td_config)
        self.touchstone_plotter: Optional[TouchstonePlotter] = None
        self.selected_files: List[str] = []
        self.processed_data: Dict[str, dict] = {}
        self.td_results: Dict[str, dict] = {}
        self.smith_networks: Dict[str, dict] = {}
        self._worker: Optional[QThread] = None
        self.theme_mode = suite.saved_theme()     # shared with the launcher and tools

        self._build_menu()
        self.setup_ui()
        self._apply_theme(self.theme_mode, persist=False)
        self._log_message("%s %s ready" % (APP_NAME, __version__))
        self._log_message("Veusz output: %s | GPU: %s | CPU cores: %d | PDF: %s"
                          " | mpld3: %s" % (
                              VEUSZ_BACKEND or "unavailable",
                              GPU_AVAILABLE or "none", CPU_COUNT,
                              PDF_LIBRARY or "not installed",
                              "yes" if MPLD3_AVAILABLE else "no"))

    # ----- menus ------------------------------------------------------------
    def _build_menu(self):
        """File, View (theme) and Help menus."""
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        for text, slot, shortcut in (
                ("&Open Touchstone Files...", self._browse_files, "Ctrl+O"),
                ("&Clear File List", self._clear_files, ""),
                ("&Save Veusz Project (.vszh5)...", self._save_project, "Ctrl+S"),
                ("Open Last Project in &Veusz", self._open_last_in_veusz, ""),
                ("&Export Last Project to PDF / PNG (Veusz)", self._export_pages, "")):
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        act = QAction("E&xit", self)
        act.setShortcut(QKeySequence("Ctrl+Q"))
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        view_menu = bar.addMenu("&View")
        group = QActionGroup(self)
        self.act_light = QAction("&Light Mode", self, checkable=True)
        self.act_dark = QAction("&Dark Mode", self, checkable=True)
        for action, mode in ((self.act_light, "light"), (self.act_dark, "dark")):
            group.addAction(action)
            action.triggered.connect(lambda checked=False, m=mode: self._apply_theme(m))
            view_menu.addAction(action)

        help_menu = bar.addMenu("&Help")
        act = QAction("&About %s..." % APP_NAME, self)
        act.triggered.connect(self.show_about)
        help_menu.addAction(act)
        act = QAction("About &Qt", self)
        act.triggered.connect(QApplication.aboutQt)
        help_menu.addAction(act)

    def _apply_theme(self, mode: str, persist: bool = True):
        """Suite light / dark theme (stored for every tool); previews follow."""
        mode = suite.apply_theme(mode, persist=persist)
        self.theme_mode = mode
        self.act_light.setChecked(mode == "light")
        self.act_dark.setChecked(mode == "dark")
        for widget in (getattr(self, "td_plot_canvas", None),
                       getattr(self, "smith_plot_canvas", None),
                       getattr(self, "plot_panel", None)):
            if widget is not None:
                widget.apply_theme(mode == "dark")

    def show_about(self):
        """Help > About."""
        content = build_about_content()
        box = QMessageBox(self)
        box.setWindowTitle("About %s" % APP_NAME)
        box.setIcon(QMessageBox.Icon.Information)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(content["text"])
        box.setInformativeText(content["notice"])
        box.setDetailedText(content["runtime"])
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        box.exec()

    # ----- layout -------------------------------------------------------------
    def setup_ui(self):
        """Tabs, progress bar, log and the action buttons."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, stretch=1)
        self.setup_main_tab()
        self.setup_time_domain_tab()
        self.setup_smith_chart_tab()
        widget, self.plot_panel = suite.make_plot_panel(self.theme_mode == "dark")
        self.tab_widget.addTab(widget, "Plots")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(130)
        main_layout.addWidget(self.status_text)

        row = QHBoxLayout()
        self.process_button = QPushButton("Process Touchstone Files")
        self.process_button.clicked.connect(self._process_files)
        self.save_button = QPushButton("Save Veusz Project (.vszh5)")
        self.save_button.clicked.connect(self._save_project)
        self.save_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        for button in (self.process_button, self.save_button, self.close_button):
            row.addWidget(button)
        main_layout.addLayout(row)

    def setup_main_tab(self):
        """File list, processing options and plot configuration."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Main Processing")
        layout = QVBoxLayout(tab)

        file_group = QGroupBox("Touchstone File Selection")
        file_layout = QVBoxLayout(file_group)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumHeight(150)
        file_layout.addWidget(self.file_list_widget)
        row = QHBoxLayout()
        self.browse_button = QPushButton("Browse Touchstone Files")
        self.browse_button.clicked.connect(self._browse_files)
        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self._clear_files)
        row.addWidget(self.browse_button)
        row.addWidget(self.clear_button)
        file_layout.addLayout(row)
        layout.addWidget(file_group, stretch=1)

        side = QHBoxLayout()
        options_group = QGroupBox("Processing Options")
        options = QFormLayout(options_group)
        self.enable_mp_checkbox = QCheckBox("Enable multiprocessing")
        self.enable_mp_checkbox.setChecked(self.config.enable_multiprocessing)
        self.enable_mp_checkbox.toggled.connect(self._update_mp_config)
        options.addRow(self.enable_mp_checkbox)
        self.cpu_spinbox = QSpinBox()
        self.cpu_spinbox.setRange(1, CPU_COUNT)
        self.cpu_spinbox.setValue(self.config.num_processes)
        self.cpu_spinbox.valueChanged.connect(self._update_cpu_config)
        options.addRow("CPU cores:", self.cpu_spinbox)
        self.enable_gpu_checkbox = QCheckBox(
            "Enable GPU (CuPy) for the time-domain transform")
        self.enable_gpu_checkbox.setChecked(self.config.enable_gpu_processing)
        self.enable_gpu_checkbox.setEnabled(GPU_AVAILABLE is not None)
        if GPU_AVAILABLE is None:
            self.enable_gpu_checkbox.setToolTip("CuPy with a usable GPU was not found.")
        self.enable_gpu_checkbox.toggled.connect(self._update_gpu_config)
        options.addRow(self.enable_gpu_checkbox)
        side.addWidget(options_group)

        plot_group = QGroupBox("Plot Configuration")
        plot = QFormLayout(plot_group)
        self.plot_title_edit = QLineEdit("Touchstone S-Parameter Analysis")
        plot.addRow("Plot title (page notes):", self.plot_title_edit)
        self.dataset_name_edit = QLineEdit("Touchstone_Dataset")
        plot.addRow("Dataset tag:", self.dataset_name_edit)
        self.unwrap_phase_checkbox = QCheckBox("Unwrap phase (numpy.unwrap)")
        self.unwrap_phase_checkbox.setChecked(self.config.unwrap_phase)
        self.unwrap_phase_checkbox.toggled.connect(self._update_unwrap_phase)
        plot.addRow(self.unwrap_phase_checkbox)
        side.addWidget(plot_group)
        layout.addLayout(side)

    def setup_time_domain_tab(self):
        """Window, gate and output controls beside the preview."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Time Domain Analysis")
        layout = QHBoxLayout(tab)
        controls = QWidget()
        controls.setMaximumWidth(400)
        cl = QVBoxLayout(controls)

        sel_group = QGroupBox("Preview")
        sel = QFormLayout(sel_group)
        self.td_file_combo = QComboBox()
        self.td_file_combo.currentTextChanged.connect(self._on_td_file_changed)
        sel.addRow("File:", self.td_file_combo)
        self.td_param_combo = QComboBox()
        self.td_param_combo.currentTextChanged.connect(self._update_td_preview)
        sel.addRow("Parameter:", self.td_param_combo)
        cl.addWidget(sel_group)

        win_group = QGroupBox("Transform Window (frequency domain)")
        win = QFormLayout(win_group)
        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(WINDOW_TYPES)
        self.window_type_combo.setCurrentText(self.td_config.window_type)
        self.window_type_combo.currentTextChanged.connect(self._update_window_config)
        win.addRow("Window:", self.window_type_combo)
        self.window_param_spin = QDoubleSpinBox()
        self.window_param_spin.setRange(0.0, 20.0)
        self.window_param_spin.setValue(self.td_config.window_param)
        self.window_param_spin.valueChanged.connect(self._update_window_param)
        win.addRow("Kaiser beta:", self.window_param_spin)
        cl.addWidget(win_group)

        gate_group = QGroupBox("Time Gate")
        gate = QFormLayout(gate_group)
        self.gating_type_combo = QComboBox()
        self.gating_type_combo.addItems(list(GATING_MODES))
        self.gating_type_combo.currentTextChanged.connect(self._update_gating_type)
        gate.addRow("Gating mode:", self.gating_type_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["bandpass", "bandstop"])
        self.mode_combo.setCurrentText(self.td_config.mode)
        self.mode_combo.currentTextChanged.connect(self._update_mode)
        gate.addRow("Gate type:", self.mode_combo)
        self.gate_center_spin = QDoubleSpinBox()
        self.gate_center_spin.setRange(-10000.0, 10000.0)
        self.gate_center_spin.setDecimals(4)
        self.gate_center_spin.setSuffix(" ns")
        self.gate_center_spin.setValue(self.td_config.gate_center)
        self.gate_center_spin.valueChanged.connect(self._update_gate_center)
        gate.addRow("Gate centre (manual):", self.gate_center_spin)
        self.gate_span_spin = QDoubleSpinBox()
        self.gate_span_spin.setRange(0.001, 10000.0)
        self.gate_span_spin.setDecimals(4)
        self.gate_span_spin.setSuffix(" ns")
        self.gate_span_spin.setValue(self.td_config.gate_span)
        self.gate_span_spin.valueChanged.connect(self._update_gate_span)
        gate.addRow("Gate span:", self.gate_span_spin)
        self.gate_window_combo = QComboBox()
        self.gate_window_combo.addItems(WINDOW_TYPES)
        self.gate_window_combo.setCurrentText(self.td_config.gate_window_type)
        self.gate_window_combo.currentTextChanged.connect(self._update_gate_window)
        gate.addRow("Gate shape:", self.gate_window_combo)
        self.gate_window_param_spin = QDoubleSpinBox()
        self.gate_window_param_spin.setRange(0.0, 20.0)
        self.gate_window_param_spin.setValue(self.td_config.gate_window_param)
        self.gate_window_param_spin.valueChanged.connect(self._update_gate_window_param)
        gate.addRow("Gate Kaiser beta:", self.gate_window_param_spin)
        self.gate_fft_window_checkbox = QCheckBox(
            "Apply transform window inside the gate")
        self.gate_fft_window_checkbox.setToolTip(
            "scikit-rf fft_window. The gated spectrum is divided by this window "
            "afterwards, which raises the band-edge error; not possible with "
            "hann or blackman (zero at the edges).")
        self.gate_fft_window_checkbox.setChecked(self.td_config.gate_use_fft_window)
        self.gate_fft_window_checkbox.toggled.connect(self._update_gate_fft_window)
        gate.addRow(self.gate_fft_window_checkbox)
        self.gate_used_label = QLabel("Gate used: -")
        self.gate_used_label.setWordWrap(True)
        gate.addRow(self.gate_used_label)
        cl.addWidget(gate_group)
        self._update_gate_controls()

        out_group = QGroupBox("Veusz Output")
        out = QVBoxLayout(out_group)
        self.plot_time_domain_checkbox = QCheckBox("Time domain (ungated and gated, dB)")
        self.plot_time_domain_checkbox.setChecked(self.td_config.plot_time_domain)
        self.plot_time_domain_checkbox.toggled.connect(self._update_plot_time_domain)
        out.addWidget(self.plot_time_domain_checkbox)
        self.plot_frequency_domain_checkbox = QCheckBox(
            "Frequency domain (measured and gated, dB)")
        self.plot_frequency_domain_checkbox.setChecked(self.td_config.plot_frequency_domain)
        self.plot_frequency_domain_checkbox.toggled.connect(self._update_plot_frequency_domain)
        out.addWidget(self.plot_frequency_domain_checkbox)
        cl.addWidget(out_group)

        self.td_timegated_button = QPushButton("Generate Time-Gated Plots in Veusz")
        self.td_timegated_button.clicked.connect(self._process_time_gated_plots)
        self.td_timegated_button.setEnabled(False)
        cl.addWidget(self.td_timegated_button)
        cl.addStretch()
        layout.addWidget(controls)

        right = QVBoxLayout()
        self.td_plot_canvas = TouchstonePlotCanvas(tab, width=8, height=6,
                                                   dark=self.theme_mode == "dark")
        self.td_plot_canvas.gate_dragged = self._gate_dragged
        right.addWidget(self.td_plot_canvas)
        layout.addLayout(right, stretch=1)

    def setup_smith_chart_tab(self):
        """Smith chart settings, export and preview."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "Smith Chart Analysis")
        layout = QHBoxLayout(tab)
        controls = QWidget()
        controls.setMaximumWidth(400)
        cl = QVBoxLayout(controls)

        sel_group = QGroupBox("Preview")
        sel = QFormLayout(sel_group)
        self.smith_file_combo = QComboBox()
        self.smith_file_combo.currentTextChanged.connect(self._on_smith_file_changed)
        sel.addRow("File:", self.smith_file_combo)
        self.smith_param_combo = QComboBox()
        self.smith_param_combo.currentTextChanged.connect(self._update_smith_preview)
        sel.addRow("Parameter:", self.smith_param_combo)
        cl.addWidget(sel_group)

        smith_group = QGroupBox("Smith Chart Settings")
        sm = QFormLayout(smith_group)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["z (Impedance)", "y (Admittance)"])
        self.chart_type_combo.currentTextChanged.connect(self._update_chart_type)
        sm.addRow("Chart type:", self.chart_type_combo)
        self.ref_impedance_spin = QDoubleSpinBox()
        self.ref_impedance_spin.setRange(1.0, 1000.0)
        self.ref_impedance_spin.setValue(self.smith_config.reference_impedance)
        self.ref_impedance_spin.setSuffix(" Ω")
        self.ref_impedance_spin.valueChanged.connect(self._update_ref_impedance)
        sm.addRow("Reference impedance:", self.ref_impedance_spin)
        self.draw_labels_checkbox = QCheckBox("Draw labels")
        self.draw_labels_checkbox.setChecked(self.smith_config.draw_labels)
        self.draw_labels_checkbox.toggled.connect(self._update_draw_labels)
        sm.addRow(self.draw_labels_checkbox)
        self.draw_vswr_checkbox = QCheckBox("Draw VSWR circles")
        self.draw_vswr_checkbox.setChecked(self.smith_config.draw_vswr)
        self.draw_vswr_checkbox.toggled.connect(self._update_draw_vswr)
        sm.addRow(self.draw_vswr_checkbox)
        self.smith_pages_checkbox = QCheckBox("Smith chart pages in the Veusz project")
        self.smith_pages_checkbox.setToolTip(
            "Prepare also adds one Smith chart page per file (reflection parameters "
            "S11, S22, ...) to the .vszh5 project, drawn with Veusz widgets.")
        self.smith_pages_checkbox.setChecked(True)
        sm.addRow(self.smith_pages_checkbox)
        cl.addWidget(smith_group)

        export_group = QGroupBox("Export")
        ex = QFormLayout(export_group)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG", "SVG", "PDF", "HTML (Interactive)"])
        ex.addRow("Format:", self.export_format_combo)
        self.pdf_bookmarks_checkbox = QCheckBox("Combined PDF with bookmarks")
        self.pdf_bookmarks_checkbox.setChecked(True)
        self.pdf_bookmarks_checkbox.setEnabled(PdfWriter is not None)
        ex.addRow(self.pdf_bookmarks_checkbox)
        cl.addWidget(export_group)

        self.smith_process_button = QPushButton("Prepare Smith Charts (all files)")
        self.smith_process_button.clicked.connect(self._process_smith_charts_mpld3)
        self.smith_process_button.setEnabled(False)
        cl.addWidget(self.smith_process_button)
        self.smith_export_button = QPushButton("Export Smith Charts")
        self.smith_export_button.clicked.connect(self._export_smith_charts)
        self.smith_export_button.setEnabled(False)
        cl.addWidget(self.smith_export_button)
        cl.addStretch()
        layout.addWidget(controls)

        right = QVBoxLayout()
        self.smith_plot_canvas = TouchstonePlotCanvas(tab, width=8, height=8,
                                                      dark=self.theme_mode == "dark")
        right.addWidget(self.smith_plot_canvas)
        layout.addLayout(right, stretch=1)

    # ----- option callbacks (toggled(bool): fixes the 0.4.0 state bug) -------------
    def _update_mp_config(self, checked: bool):
        self.config.enable_multiprocessing = bool(checked)
        self._log_message("Multiprocessing: %s" % ("on" if checked else "off"))

    def _update_cpu_config(self, value: int):
        self.config.num_processes = self.config.max_workers = int(value)
        self._log_message("CPU cores set to %d" % value)

    def _update_gpu_config(self, checked: bool):
        self.config.enable_gpu_processing = bool(checked)
        self._log_message("GPU transform: %s" % ("on" if checked else "off"))

    def _update_unwrap_phase(self, checked: bool):
        self.config.unwrap_phase = bool(checked)
        self._log_message("Phase unwrapping (numpy.unwrap): %s - applies at the "
                          "next Process" % ("ON" if checked else "OFF"))

    def _update_window_config(self, window_type: str):
        self.td_config.window_type = window_type
        self.window_param_spin.setEnabled(window_type == "kaiser")
        self._update_td_preview()

    def _update_window_param(self, value: float):
        self.td_config.window_param = float(value)
        self._update_td_preview()

    def _update_gate_window(self, window_type: str):
        self.td_config.gate_window_type = window_type
        self.gate_window_param_spin.setEnabled(window_type == "kaiser")
        self._update_td_preview()

    def _update_gate_window_param(self, value: float):
        self.td_config.gate_window_param = float(value)
        self._update_td_preview()

    def _update_gate_fft_window(self, checked: bool):
        self.td_config.gate_use_fft_window = bool(checked)
        self._update_td_preview()

    def _update_mode(self, mode: str):
        self.td_config.mode = mode
        self._update_td_preview()

    def _update_gating_type(self, text: str):
        self.td_config.gating_type = GATING_MODES.get(text, "auto_peak")
        self._update_gate_controls()
        self._log_message("Gating mode: %s" % text)
        self._update_td_preview()

    def _update_gate_controls(self):
        """Centre is manual-only; span is ignored by the full auto mode."""
        self.gate_center_spin.setEnabled(self.td_config.gating_type == "manual")
        self.gate_span_spin.setEnabled(True)

    def _update_gate_center(self, value: float):
        self.td_config.gate_center = float(value)
        self._update_td_preview()

    def _update_gate_span(self, value: float):
        self.td_config.gate_span = float(value)
        self._update_td_preview()

    def _update_plot_time_domain(self, checked: bool):
        self.td_config.plot_time_domain = bool(checked)
        self._update_timegated_button_state()

    def _update_plot_frequency_domain(self, checked: bool):
        self.td_config.plot_frequency_domain = bool(checked)
        self._update_timegated_button_state()

    def _update_timegated_button_state(self):
        """Enabled when files are processed and at least one output is ticked."""
        any_output = self.td_config.plot_time_domain or self.td_config.plot_frequency_domain
        self.td_timegated_button.setEnabled(bool(self.processed_data) and any_output
                                            and self._worker is None)

    def _update_chart_type(self, text: str):
        self.smith_config.chart_type = "z" if "Impedance" in text else "y"
        self._update_smith_preview()

    def _update_ref_impedance(self, value: float):
        self.smith_config.reference_impedance = float(value)
        self._update_smith_preview()

    def _update_draw_labels(self, checked: bool):
        self.smith_config.draw_labels = bool(checked)
        self._update_smith_preview()

    def _update_draw_vswr(self, checked: bool):
        self.smith_config.draw_vswr = bool(checked)
        self._update_smith_preview()

    # ----- file handling -----------------------------------------------------------
    def _browse_files(self):
        """Add Touchstone files to the list."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Touchstone Files", "",
            "Touchstone Files (*.s*p *.S*P *.ts);;All Files (*)")
        if files:
            new = [f for f in files if f not in self.selected_files]
            self.selected_files.extend(new)
            self._update_file_list()
            self._log_message("Added %d file(s)" % len(new))

    def _clear_files(self):
        """Forget files, results and the Veusz document; release memory."""
        self.selected_files.clear()
        self.processed_data.clear()
        self.td_results.clear()
        self.smith_networks.clear()
        self.touchstone_plotter = None
        self._update_file_list()
        self._refresh_combos()
        self.save_button.setEnabled(False)
        self.smith_process_button.setEnabled(False)
        self.smith_export_button.setEnabled(False)
        self._update_timegated_button_state()
        gc.collect()
        self._log_message("File list cleared (memory released)")

    def _update_file_list(self):
        self.file_list_widget.clear()
        for path in self.selected_files:
            self.file_list_widget.addItem(path)

    def _refresh_combos(self):
        """Fill the preview file combos from the processed data."""
        names = list(self.processed_data)
        for combo in (self.td_file_combo, self.smith_file_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)
        self._on_td_file_changed(self.td_file_combo.currentText())
        self._on_smith_file_changed(self.smith_file_combo.currentText())

    @staticmethod
    def _fill_param_combo(combo: QComboBox, network: Optional[Network]):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if network is not None:
            names = [name for _i, _j, name in s_param_names(network.nports)]
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_td_file_changed(self, key: str):
        data = self.processed_data.get(key)
        self._fill_param_combo(self.td_param_combo, data['network'] if data else None)
        self._update_td_preview()

    def _on_smith_file_changed(self, key: str):
        data = self.processed_data.get(key)
        self._fill_param_combo(self.smith_param_combo, data['network'] if data else None)
        self._update_smith_preview()

    # ----- previews ---------------------------------------------------------------
    def _update_td_preview(self, *_args):
        """Time domain and gate of the selected file and parameter."""
        key = self.td_file_combo.currentText()
        param = self.td_param_combo.currentText()
        data = self.processed_data.get(key)
        if not data or not param:
            return
        result = self.td_processor.process_network(
            data['network'], use_gpu=self.config.enable_gpu_processing,
            params=[param])
        if result['error']:
            self._log_message("Time-domain preview error: %s" % result['error'])
            return
        entry = result['params'][param]
        self.gate_used_label.setText(
            "Gate used: centre %.4f ns, span %.4f ns%s" % (
                entry['gate_center_ns'], entry['gate_span_ns'],
                ("\n" + entry['note']) if entry['note'] else ""))
        self.td_plot_canvas.plot_time_domain(
            result['time_ns'], entry['td'], entry['td_gated'],
            result['frequency_hz'], entry['s'], entry['s_gated'],
            gate=(entry['gate_center_ns'], entry['gate_span_ns']),
            title="%s - %s" % (key, param),
            movable=self.td_config.gating_type == "manual")
        for note in result['notes']:
            self._log_message("%s: %s" % (key, note))

    def _smith_network(self, key: str, param: str) -> Optional[Network]:
        """S-parameter of a file as a one-port at the reference impedance."""
        data = self.processed_data.get(key)
        if not data:
            return None
        network = data['network']
        names = {name: (i, j) for i, j, name in s_param_names(network.nports)}
        if param not in names:
            return None
        op = one_port(network, *names[param])
        z_ref = self.smith_config.reference_impedance
        if not np.allclose(op.z0, z_ref):
            op = op.copy()
            op.renormalize(z_ref)
        return op

    def _update_smith_preview(self, *_args):
        """Smith chart of the selected file and parameter."""
        key = self.smith_file_combo.currentText()
        param = self.smith_param_combo.currentText()
        op = self._smith_network(key, param) if key and param else None
        if op is None:
            return
        try:
            self.smith_plot_canvas.plot_smith_chart(
                op, title="%s - %s" % (key, param),
                chart_type=self.smith_config.chart_type,
                draw_labels=self.smith_config.draw_labels,
                draw_vswr=self.smith_config.draw_vswr)
        except Exception as exc:
            self._log_message("Smith chart preview error: %s" % exc)

    # ----- processing ----------------------------------------------------------------
    def _set_busy(self, busy: bool, text: str = ""):
        """Disable the action buttons while a worker runs."""
        for button in (self.process_button, self.td_timegated_button,
                       self.smith_process_button, self.smith_export_button,
                       self.save_button):
            button.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        self.progress_bar.setValue(0)
        if text:
            self._log_message(text)
        if not busy:
            has = bool(self.processed_data)
            self.save_button.setEnabled(has and self.touchstone_plotter is not None)
            self.smith_process_button.setEnabled(has)
            self.smith_export_button.setEnabled(bool(self.smith_networks))
            self._update_timegated_button_state()

    def _process_files(self):
        """Read the selected files in a worker thread."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select Touchstone files first.")
            return
        try:
            self.touchstone_plotter = TouchstonePlotter(
                plot_title=self.plot_title_edit.text(),
                dataset_name=self.dataset_name_edit.text(),
                unwrap_phase=self.config.unwrap_phase)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Veusz Output Unavailable", str(exc))
            return
        thread = TouchstoneProcessingThread(self.selected_files, self.config)
        thread.progress_updated.connect(self.progress_bar.setValue)
        thread.processing_finished.connect(self._on_processing_finished)
        thread.error_occurred.connect(self._on_processing_error)
        self._worker = thread
        self._set_busy(True, "Reading %d Touchstone file(s)..." % len(self.selected_files))
        thread.start()

    def _on_processing_finished(self, results: Dict[str, dict]):
        """Keep good files, report failures, build the frequency pages."""
        self._worker = None
        good = {k: v for k, v in results.items() if v and not v.get('error')}
        bad = {k: v for k, v in results.items() if not v or v.get('error')}
        self.processed_data = good
        self.td_results.clear()
        self.smith_networks.clear()
        for key, data in bad.items():
            self._log_message("FAILED %s: %s" % (key, (data or {}).get('error')))
        for key, data in good.items():
            try:
                self.touchstone_plotter.create_plots_from_data(key, data)
            except Exception as exc:
                self._log_message("Plot error %s: %s" % (key, exc))
        phase = "unwrapped" if self.config.unwrap_phase else "wrapped"
        self._log_message("Processed %d file(s), %d failed; frequency pages built "
                          "(phase %s)" % (len(good), len(bad), phase))
        self._refresh_combos()
        self._set_busy(False)
        self._show_plots()
        if bad:
            QMessageBox.warning(self, "Processing Warnings",
                                "%d file(s) failed. See the log for details." % len(bad))

    def _on_processing_error(self, message: str):
        self._worker = None
        self._set_busy(False)
        self._log_message("Processing error: %s" % message)
        QMessageBox.critical(self, "Processing Error", message)

    def _process_time_gated_plots(self):
        """Time domain + gating of every file (worker thread), then Veusz pages."""
        if not self.processed_data or self.touchstone_plotter is None:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return
        cfg = asdict(self.td_config)
        items = [(k, d['network']) for k, d in self.processed_data.items()]
        use_gpu = self.config.enable_gpu_processing and GPU_AVAILABLE is not None
        workers = self.config.max_workers
        use_proc = self.config.enable_multiprocessing

        def job(progress):
            return run_time_domain_batch(items, cfg, use_gpu, workers, use_proc, progress)

        worker = FunctionWorker(job)
        worker.progress_updated.connect(self.progress_bar.setValue)
        worker.finished_ok.connect(self._on_time_gated_finished)
        worker.error_occurred.connect(self._on_worker_error)
        self._worker = worker
        self._set_busy(True, "Time domain and gating of %d file(s)..." % len(items))
        worker.start()

    def _on_time_gated_finished(self, results: Dict[str, dict]):
        """Add the time-gated pages to the Veusz project."""
        self._worker = None
        made = 0
        mode = self.td_config.mode
        for key, result in results.items():
            if result.get('error'):
                self._log_message("Time domain FAILED %s: %s" % (key, result['error']))
                continue
            result['mode'] = mode
            self.td_results[key] = result
            for note in result['notes']:
                self._log_message("%s: %s" % (key, note))
            try:
                self.touchstone_plotter.create_time_gated_plots(
                    key, self.processed_data[key], result,
                    plot_time_domain=self.td_config.plot_time_domain,
                    plot_frequency_domain=self.td_config.plot_frequency_domain)
                made += 1
            except Exception as exc:
                self._log_message("Time-gated plot error %s: %s" % (key, exc))
        self._set_busy(False)
        outputs = [n for n, on in (("time domain", self.td_config.plot_time_domain),
                                   ("gated frequency domain",
                                    self.td_config.plot_frequency_domain)) if on]
        self._log_message("Time-gated pages for %d file(s): %s. Save the project to "
                          "keep them." % (made, ", ".join(outputs)))
        self._show_plots()

    def _on_worker_error(self, message: str):
        self._worker = None
        self._set_busy(False)
        self._log_message("Error: %s" % message)
        QMessageBox.critical(self, "Error", message)

    # ----- Smith charts ------------------------------------------------------------------
    def _process_smith_charts_mpld3(self):
        """One-port networks of every parameter of every file."""
        if not self.processed_data:
            QMessageBox.warning(self, "No Data", "Please process Touchstone files first.")
            return
        self.smith_networks.clear()
        for key, data in self.processed_data.items():
            for _i, _j, param in s_param_names(data['network'].nports):
                op = self._smith_network(key, param)
                if op is not None:
                    self.smith_networks["%s_%s" % (key, param)] = {
                        'network': op, 'param_name': param, 'filename': key}
        self.smith_export_button.setEnabled(bool(self.smith_networks))
        self._update_smith_preview()
        self._log_message("Prepared %d Smith charts at %.4g ohm. Click Export to "
                          "save them." % (len(self.smith_networks),
                                          self.smith_config.reference_impedance))
        if self.smith_pages_checkbox.isChecked() and self.touchstone_plotter is not None:
            self._add_smith_pages()

    def _add_smith_pages(self):
        """Smith chart pages (reflection parameters) of every file in the project (0.7.0)."""
        pages = 0
        for key, data in self.processed_data.items():
            reflections = {}
            for i, j, param in s_param_names(data['network'].nports):
                if i == j:
                    op = self._smith_network(key, param)
                    if op is not None:
                        reflections[param] = op
            if not reflections:
                continue
            try:
                self.touchstone_plotter.create_smith_page(key, data, reflections,
                                                          self.smith_config)
                pages += 1
            except Exception as exc:
                self._log_message("Smith page error %s: %s" % (key, exc))
        self._log_message("Smith chart pages added to the project: %d (save the project "
                          "to keep them)" % pages)
        self._show_plots()

    def _export_smith_charts(self):
        """Write the charts (process pool) and optionally merge a bookmarked PDF."""
        if not self.smith_networks:
            QMessageBox.warning(self, "No Charts", "Please prepare Smith charts first.")
            return
        choice = self.export_format_combo.currentText()
        fmt = "html" if "HTML" in choice else choice.lower()
        combined = fmt == "pdf" and self.pdf_bookmarks_checkbox.isChecked()
        if combined:
            pdf_output, _ = QFileDialog.getSaveFileName(
                self, "Save Combined Smith Chart PDF", "SmithCharts_Combined.pdf",
                "PDF Files (*.pdf)")
            if not pdf_output:
                return
            out_dir = tempfile.mkdtemp(prefix="smith_")
        else:
            out_dir = QFileDialog.getExistingDirectory(self, "Folder for Smith charts")
            if not out_dir:
                return
            pdf_output = ""
        ext = {"png": ".png", "svg": ".svg", "pdf": ".pdf", "html": ".html"}[fmt]
        chart = self.smith_config
        jobs, order = [], []
        for index, (key, item) in enumerate(self.smith_networks.items()):
            stem = sanitize_name(item['filename'])   # unique per file key
            name = "%04d_%s_%s%s" % (index, stem, item['param_name'], ext) if combined \
                else "%s_%s%s" % (stem, item['param_name'], ext)
            path = os.path.join(out_dir, name)
            jobs.append((item['network'], "%s - %s" % (item['filename'], item['param_name']),
                         chart.chart_type, path, fmt, chart.draw_labels, chart.draw_vswr))
            order.append((path, item['filename'], item['param_name']))
        workers = self.config.max_workers if self.config.enable_multiprocessing else 1

        def job(progress):
            done: Dict[str, Tuple[bool, str]] = {}
            total = max(len(jobs), 1)
            if workers > 1 and len(jobs) > 1:
                with make_process_pool(min(workers, len(jobs))) as pool:
                    futures = [pool.submit(export_smith_worker, j) for j in jobs]
                    for n, fut in enumerate(as_completed(futures), start=1):
                        path, ok, msg = fut.result()
                        done[path] = (ok, msg)
                        progress(int(90 * n / total))
            else:
                for n, j in enumerate(jobs, start=1):
                    path, ok, msg = export_smith_worker(j)
                    done[path] = (ok, msg)
                    progress(int(90 * n / total))
            merged = ""
            if combined:
                entries = [(p, f, s) for p, f, s in order if done.get(p, (False,))[0]]
                if entries:
                    SmithChartPlottermpld3.create_pdf_with_bookmarks(entries, pdf_output)
                    merged = pdf_output
                shutil.rmtree(out_dir, ignore_errors=True)
            progress(100)
            return {"done": done, "merged": merged, "out_dir": out_dir}

        worker = FunctionWorker(job)
        worker.progress_updated.connect(self.progress_bar.setValue)
        worker.finished_ok.connect(self._on_smith_export_finished)
        worker.error_occurred.connect(self._on_worker_error)
        self._worker = worker
        self._set_busy(True, "Exporting %d Smith chart(s) as %s..." % (len(jobs), fmt.upper()))
        worker.start()

    def _on_smith_export_finished(self, report: Dict[str, Any]):
        self._worker = None
        self._set_busy(False)
        done = report["done"]
        good = sum(1 for ok, _m in done.values() if ok)
        for path, (ok, msg) in done.items():
            if not ok or msg:
                self._log_message("%s %s: %s" % ("FAILED" if not ok else "Note",
                                                 os.path.basename(path), msg))
        where = report["merged"] or report["out_dir"]
        self._log_message("Smith charts exported: %d of %d -> %s" % (good, len(done), where))
        QMessageBox.information(self, "Export Complete",
                                "Exported %d of %d Smith charts to:\n%s" % (good, len(done), where))

    # ----- save -----------------------------------------------------------------------------
    def _save_project(self):
        """Save the .vszh5 project; optionally open it in Veusz."""
        if self.touchstone_plotter is None or not self.processed_data:
            QMessageBox.warning(self, "Nothing to Save", "Please process Touchstone files first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Touchstone Veusz Project", "",
                                              "Veusz HDF5 Project (*.vszh5)")
        if not path:
            return
        try:
            saved = self.touchstone_plotter.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", "Failed to save project:\n%s" % exc)
            return
        self._log_message("Project saved: %s" % saved)
        self.last_saved = saved
        reply = QMessageBox.question(self, "Open in Veusz",
                                     "Open the saved project in Veusz?",
                                     QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._open_in_veusz(saved)

    def _open_in_veusz(self, filename: str):
        """External Veusz viewer (VEUSZ_EXECUTABLE, PATH, or python -m veusz)."""
        if suite.open_in_veusz(self, filename):
            self._log_message("Opening in Veusz: %s" % filename)

    def _gate_dragged(self, centre_ns: float, span_ns: float):
        """Manual gate dragged in the preview: update the boxes (and preview)."""
        for spin, value in ((self.gate_center_spin, centre_ns), (self.gate_span_spin, span_ns)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.td_config.gate_center = float(self.gate_center_spin.value())
        self.td_config.gate_span = float(self.gate_span_spin.value())
        self._update_td_preview()

    def _show_plots(self):
        """Refresh the Plots tab from the project (native backend)."""
        if self.plot_panel is None or self.touchstone_plotter is None:
            return
        doc = getattr(self.touchstone_plotter, "doc", None)
        if doc is None or not hasattr(doc, "datasets"):
            return
        try:
            pages = self.plot_panel.set_document(doc)
            self._log_message("Plots tab: %d page(s)" % pages)
        except Exception as exc:
            self._log_message("Plots tab could not draw the project: %s" % exc)

    def _open_last_in_veusz(self):
        if not getattr(self, "last_saved", None):
            QMessageBox.information(self, APP_NAME, "Save a project first.")
            return
        self._open_in_veusz(self.last_saved)

    def _export_pages(self):
        if not getattr(self, "last_saved", None):
            QMessageBox.information(self, APP_NAME, "Save a project first.")
            return
        suite.export_with_veusz(self, self.last_saved, self._log_message)

    def _log_message(self, message: str):
        """Time-stamped line in the log pane."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_text.append("[%s] %s" % (stamp, message))

    def closeEvent(self, event):  # noqa: N802 - Qt API
        """Wait for a running worker so the process pool shuts down cleanly."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        if getattr(self, "plot_panel", None) is not None:
            self.plot_panel.clear()
        super().closeEvent(event)


# %% Entry point
def main() -> int:
    """Start the GUI."""
    multiprocessing.freeze_support()
    print("%s %s | Python %s | %s | long paths: %s" % (
        APP_NAME, __version__, platform.python_version(), platform.system(),
        "extended (\\\\?\\) for reading" if is_windows() else "n/a"))
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    window = TouchstoneMainWindow()
    window.show()
    run = getattr(app, "exec", None) or getattr(app, "exec_")
    return run()


if __name__ == "__main__":
    sys.exit(main())
