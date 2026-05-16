# -*- coding: utf-8 -*-
"""
=============================================================================
_autoplot_common.py
-----------------------------------------------------------------------------
# %% Header Info

Shared helpers for FITS_AutoPlot and Franks_AutoPlot:

    * qtpy abstraction layer GUI building blocks (replacement for the older
      FITS_AutoPlot.py minimal dialog) modelled after Touchstone_AutoPlot.py.
    * Dark / Light theme palettes, selectable from a textual View menu.
    * Memory monitor that spills numpy arrays to disk via np.memmap (the
      most efficient zero-copy RAM->disk path) when an RSS threshold is
      crossed.
    * Thread pool helper with a script-level MAX_THREADS knob.
    * Veusz embedded helper that is compatible with both Veusz 3.4 and 4.1,
      including the saving of .vszh5 (HDF5) project files in both versions.

The module is written for Python 3.8 and avoids any syntax that was
introduced in 3.9+ (no walrus inside f-strings, no PEP 604 unions, no dict
union operators, no parenthesised context managers).

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Initial implementation of shared infrastructure
Date: 2026-05-16
# %%%% 0.0.2: Added mjd_to_datestr() helper for MJD -> YYYY-MM-DD_HH:MM:SS
#             string conversion used by all four AutoPlot modules.
Date: 2026-05-16
# %%%% 0.0.3: NaN-preserving dataset emission policy. Numeric datasets keep
#             NaN floats (Veusz natively supports NaN in numeric datasets;
#             plots simply skip NaN samples). Text datasets cannot carry a
#             true NaN, so non-finite MJD inputs to mjd_to_datestr() now
#             yield the explicit sentinel string ``"NaN"`` (length-preserving)
#             rather than an empty string -- this prevents NaN rows from
#             being silently dropped or confused with a missing token.
Date: 2026-05-16
# %%%% 0.0.4: Added register_nrao_fits_units() and suppress_fits_unit_warnings()
#             helpers.  NRAO 1PPS-delta FITS files carry non-standard unit
#             strings (``'none'`` on CHANNELA/CHANNELB, ``'NanoSeconds'`` on
#             DELTAT) that astropy.io.fits / QTable.read flag with a noisy
#             UnitsWarning -- and astropy.table additionally warns that the
#             text columns are kept as MaskedColumn because the unit cannot
#             be converted to a Quantity.  The new helpers register these
#             unit aliases with astropy.units and provide a context manager
#             that filters the residual harmless warnings so a 900-file
#             batch run no longer floods the log.
Date: 2026-05-16
# %%%% 0.0.5: Added Spyder-style cell markers (``# %% TITLE`` for top-level
#             sections, ``# %%% TITLE`` for nested sections) on the
#             existing dashed banner blocks so the file is navigable in
#             Spyder's Outline / cell navigator.  Pure cosmetic change --
#             no runtime behaviour modified.
Date: 2026-05-16
# %%%% 0.0.6: AutoPlotMainWindow now provides three labelled progress bars
#             (read / parse / per-column) plus helper methods
#             ``show_progress_bars()``, ``hide_progress_bars()``,
#             ``begin_column_progress()`` and ``tick_column_progress()``.
#             Subclasses use these to drive a multi-stage progress UI.
Date: 2026-05-16
# %%%%% Function Descriptions
        make_dark_palette/make_light_palette/apply_theme: textual menu theme
            switching for dark vs light mode (View menu).
        MemoryAwareCache: thread-safe data cache that spills large numpy
            arrays to disk via numpy.memmap (fastest zero-copy RAM->disk).
        MemoryMonitor: background thread that polls process RSS and fires a
            callback on high-water transitions.
        run_in_threadpool: ThreadPoolExecutor helper used by every AutoPlot
            module; respects the per-script MAX_THREADS knob.
        open_embedded/save_vszh5: Veusz embedded helpers that work on both
            Veusz 3.4 and Veusz 4.1 (.vszh5 HDF5 output).
        AutoPlotMainWindow: qtpy-based QMainWindow base class providing the
            Touchstone_AutoPlot.py-style file list / options / log / button
            row layout, plus the View menu dark/light toggle.
        open_maybe_gzipped/safe_dsname: small I/O helpers used by callers.
# %%%%% Variable Descriptions
        MemoryMonitorConfig.rss_high_water_mb: RSS threshold for spilling.
        MemoryMonitorConfig.array_min_bytes: arrays smaller than this stay
            in RAM regardless of RSS pressure.
        MemoryMonitorConfig.cache_dir: directory used for memmap backing
            files (created in $TMP by default).
# %%%%% More Info
        numpy.memmap was chosen as the RAM->disk transport because it is
        the most efficient option available to pure Python/numpy: the file
        backs an mmap'd region and writes go through the kernel page cache
        without any user-space buffering or pickling overhead.
=============================================================================
"""
# %% Imports
from __future__ import annotations

# %%% IMPORTS - Standard library
import gc
import os
import sys
import tempfile
import threading
import time
import uuid
import gzip
import shutil
import warnings
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# %%% IMPORTS - Scientific
import numpy as np

# ---------------------------------------------------------------------------
# %%% Qt imports via qtpy (with frozen-bundle fallback to PySide6, matching the
# pattern used elsewhere in the FitsAutoPlot repo).
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    from PySide6.QtCore import Qt, QThread, Signal, QObject  # noqa: F401
    from PySide6.QtGui import QPalette, QColor, QAction      # noqa: F401
    from PySide6.QtWidgets import (                          # noqa: F401
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QMessageBox, QListWidget,
        QGroupBox, QCheckBox, QSpinBox, QComboBox, QLineEdit, QTextEdit,
        QProgressBar, QFormLayout, QMenuBar, QMenu, QStatusBar,
        QTabWidget, QRadioButton, QButtonGroup,
    )
else:
    from qtpy.QtCore import Qt, QThread, Signal, QObject      # noqa: F401
    from qtpy.QtGui import QPalette, QColor                   # noqa: F401
    try:
        # qtpy maps QAction onto QtGui on Qt6 and QtWidgets on Qt5.
        from qtpy.QtGui import QAction                        # noqa: F401
    except ImportError:  # pragma: no cover -- Qt5
        from qtpy.QtWidgets import QAction                    # type: ignore  # noqa: F401
    from qtpy.QtWidgets import (                              # noqa: F401
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QMessageBox, QListWidget,
        QGroupBox, QCheckBox, QSpinBox, QComboBox, QLineEdit, QTextEdit,
        QProgressBar, QFormLayout, QMenuBar, QMenu, QStatusBar,
        QTabWidget, QRadioButton, QButtonGroup,
    )

# psutil is preferred for memory monitoring, but we degrade gracefully
try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# %% THEME PALETTES
# ---------------------------------------------------------------------------
def make_dark_palette() -> "QPalette":
    """Return a Fusion-style dark palette."""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(53, 53, 53))
    p.setColor(QPalette.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.Base, QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    p.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    p.setColor(QPalette.Text, QColor(220, 220, 220))
    p.setColor(QPalette.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.BrightText, QColor(255, 0, 0))
    p.setColor(QPalette.Link, QColor(42, 130, 218))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    return p


def make_light_palette() -> "QPalette":
    """Return a default light palette (style standard)."""
    return QApplication.style().standardPalette()


def apply_theme(app: "QApplication", mode: str) -> None:
    """Apply 'dark' or 'light' palette to the running QApplication."""
    if mode == "dark":
        app.setStyle("Fusion")
        app.setPalette(make_dark_palette())
    else:
        app.setStyle("Fusion")
        app.setPalette(make_light_palette())


# ---------------------------------------------------------------------------
# %% MEMORY MONITOR + MEMMAP CACHE
# ---------------------------------------------------------------------------
@dataclass
class MemoryMonitorConfig:
    """Configuration knobs for the memory monitor."""
    rss_high_water_mb: int = 1024            # spill arrays once RSS exceeds this
    array_min_bytes: int = 8 * 1024 * 1024    # only spill arrays >= 8 MiB
    cache_dir: Optional[str] = None
    poll_interval_sec: float = 0.5


class MemoryAwareCache:
    """
    Memory-aware data cache that spills large numpy arrays to disk via
    ``numpy.memmap``.

    ``numpy.memmap`` is the most efficient RAM-to-disk transfer method
    available from pure numpy because the bytes are written through the
    operating system page cache without going through Python-level buffering
    -- effectively a zero-copy mmap.  The on-disk file is created with the
    array's native dtype and shape so it can be re-mapped read-only without
    any conversion.

    The class is thread-safe and tracks process RSS.  Once RSS exceeds the
    configured high-water mark, any subsequent ``store`` calls that exceed
    the per-array threshold are written straight to a memmap.
    """

    def __init__(self, config: Optional[MemoryMonitorConfig] = None) -> None:
        self.cfg = config or MemoryMonitorConfig()
        self._cache: Dict[str, Any] = {}
        self._spilled: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.cache_dir = self.cfg.cache_dir or tempfile.mkdtemp(prefix="autoplot_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ----- introspection ----------------------------------------------------
    def rss_mb(self) -> float:
        if _PSUTIL_AVAILABLE:
            try:
                return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def over_high_water(self) -> bool:
        return self.rss_mb() >= self.cfg.rss_high_water_mb

    # ----- public api -------------------------------------------------------
    def store(self, key: str, value: Any) -> Any:
        """
        Store ``value`` under ``key``.  If ``value`` is a sufficiently large
        numpy array and we are above the RSS high-water mark, write it to a
        memmap on disk and return a read-only memmap view.  Otherwise keep
        it in RAM.
        """
        with self._lock:
            if isinstance(value, np.ndarray) and value.nbytes >= self.cfg.array_min_bytes \
                    and self.over_high_water():
                value = self._spill_to_disk(key, value)
            self._cache[key] = value
            return value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._cache.get(key, default)

    def items(self):
        with self._lock:
            return list(self._cache.items())

    def keys(self):
        with self._lock:
            return list(self._cache.keys())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            for path in list(self._spilled.values()):
                try:
                    os.remove(path)
                except OSError:
                    pass
            self._spilled.clear()
            gc.collect()

    def cleanup(self) -> None:
        """Remove the on-disk cache directory completely."""
        self.clear()
        try:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        except Exception:
            pass

    # ----- internals --------------------------------------------------------
    def _spill_to_disk(self, key: str, arr: np.ndarray) -> np.ndarray:
        # numpy.memmap performs an in-kernel copy via mmap which is the
        # fastest user-mode-visible RAM->disk path on Linux/Windows/macOS.
        safe_key = "".join(c if (c.isalnum() or c in "._-") else "_" for c in key)
        path = os.path.join(self.cache_dir, "%s_%s.npy" % (safe_key, uuid.uuid4().hex[:8]))
        # We use np.save+np.load(mmap_mode='r') so dtype/shape are preserved.
        np.save(path, arr, allow_pickle=False)
        self._spilled[key] = path
        mm = np.load(path, mmap_mode="r")
        # drop original RAM reference
        del arr
        gc.collect()
        return mm


class MemoryMonitor(threading.Thread):
    """
    Background thread that publishes a callback whenever process RSS crosses
    the high-water mark.  The callback is called with the current RSS in
    MiB.
    """

    def __init__(self, cache: MemoryAwareCache,
                 callback: Optional[Callable[[float], None]] = None) -> None:
        super().__init__(daemon=True)
        self.cache = cache
        self.callback = callback
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        prev_over = False
        while not self._stop.is_set():
            if self.callback is not None:
                try:
                    rss = self.cache.rss_mb()
                    over = self.cache.over_high_water()
                    if over and not prev_over:
                        self.callback(rss)
                    prev_over = over
                except Exception:
                    pass
            time.sleep(self.cache.cfg.poll_interval_sec)


# ---------------------------------------------------------------------------
# %% THREAD POOL HELPER
# ---------------------------------------------------------------------------
def run_in_threadpool(work_items: List[Tuple[str, Callable[..., Any], Tuple[Any, ...]]],
                      max_workers: int,
                      progress_cb: Optional[Callable[[int, int, str], None]] = None
                      ) -> Dict[str, Any]:
    """
    Submit ``work_items`` -- ``(key, fn, args)`` triplets -- to a
    ``ThreadPoolExecutor`` and collect results keyed by ``key``.

    Threads are preferred over processes for FITS I/O because:
      * astropy + numpy release the GIL on the I/O and array creation paths,
      * the Veusz embedded document must be touched from a single thread, so
        the parent process keeps that handle and only the heavy data
        decoding runs concurrently.
    """
    max_workers = max(1, int(max_workers))
    results: Dict[str, Any] = {}
    total = len(work_items)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fn, *args): key for key, fn, args in work_items}
        for fut in as_completed(future_map):
            key = future_map[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:
                results[key] = exc
            done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, total, key)
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# %% VEUSZ EMBED COMPAT
# ---------------------------------------------------------------------------
def open_embedded(name: str = "AutoPlot"):
    """
    Return a ``veusz.embed.Embedded`` instance (the on-screen embedded
    document) that works on Veusz 3.4 and 4.1.

    Veusz 3.4 's Embedded.__init__ accepts ``(name, ...)``; 4.1 keeps the
    same positional API.  We additionally enable the toolbar.
    """
    import veusz.embed as vz_embed  # local import so the module loads w/o veusz
    doc = vz_embed.Embedded(name)
    try:
        doc.EnableToolbar(enable=True)
    except TypeError:
        doc.EnableToolbar()
    return doc


def save_vszh5(doc, filename: str) -> str:
    """
    Save a Veusz embedded document as an HDF5 (.vszh5) project file in a
    way that is compatible with both Veusz 3.4 and 4.1.

    Veusz 3.4 introduced ``Save`` with an explicit ``mode='hdf5'`` keyword
    and started keying off the filename extension; 4.1 dropped the mode
    keyword and now keys solely off the extension.  We try both, in order,
    so the same code path works on either release.
    """
    if not filename.lower().endswith(".vszh5"):
        filename = os.path.splitext(filename)[0] + ".vszh5"
    try:
        doc.Save(filename, mode="hdf5")  # Veusz >= 3.4
    except TypeError:
        doc.Save(filename)               # Veusz >= 4.1 (extension based)
    return filename


# ---------------------------------------------------------------------------
# %% COMMON GUI BUILDING BLOCKS
# ---------------------------------------------------------------------------
class AutoPlotMainWindow(QMainWindow):
    """
    Base main window mirroring the layout used by Touchstone_AutoPlot.py:

        * top: File / View(theme) / Help text menu bar
        * centre: file list, processing options, log area
        * bottom: progress bar and Process / Save / Close buttons

    Subclasses provide ``_process_files`` and ``_save_project``.
    """

    def __init__(self, title: str, default_mode: str = "dark") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1100, 800)
        self.selected_files: List[str] = []
        self._theme_mode = default_mode
        self._build_menu()
        self._build_central_widget()
        self._build_statusbar()
        apply_theme(QApplication.instance(), self._theme_mode)

    # ----- chrome -----------------------------------------------------------
    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        act_open = QAction("&Open Files...", self)
        act_open.triggered.connect(self._browse_files)
        file_menu.addAction(act_open)
        act_clear = QAction("&Clear List", self)
        act_clear.triggered.connect(self._clear_files)
        file_menu.addAction(act_clear)
        file_menu.addSeparator()
        act_save = QAction("&Save Project (.vszh5)...", self)
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("E&xit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = bar.addMenu("&View")
        self.act_dark = QAction("&Dark Mode", self, checkable=True)
        self.act_light = QAction("&Light Mode", self, checkable=True)
        self.act_dark.setChecked(self._theme_mode == "dark")
        self.act_light.setChecked(self._theme_mode == "light")
        self.act_dark.triggered.connect(lambda: self._set_theme("dark"))
        self.act_light.triggered.connect(lambda: self._set_theme("light"))
        view_menu.addAction(self.act_dark)
        view_menu.addAction(self.act_light)

        help_menu = bar.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.mem_label = QLabel("RSS: -- MiB")
        sb.addPermanentWidget(self.mem_label)

    def _build_central_widget(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # File selection
        fgroup = QGroupBox("Input File Selection")
        flay = QVBoxLayout(fgroup)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumHeight(180)
        flay.addWidget(self.file_list_widget)
        row = QHBoxLayout()
        self.browse_button = QPushButton("Browse Files...")
        self.browse_button.clicked.connect(self._browse_files)
        row.addWidget(self.browse_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_files)
        row.addWidget(self.clear_button)
        flay.addLayout(row)
        root.addWidget(fgroup)

        # Options (subclass populates options_layout)
        ogroup = QGroupBox("Processing Options")
        self.options_layout = QFormLayout(ogroup)
        root.addWidget(ogroup)
        self._populate_options(self.options_layout)

        # Log area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(150)
        root.addWidget(self.status_text)

        # Progress bars (read + parse/push + per-column)
        # %% Progress bars
        # ``progress_bar``        : file-level read/parse progress (worker thread)
        # ``parse_progress_bar``  : file-level Veusz-push progress  (GUI thread)
        # ``column_progress_bar`` : per-column progress within the file currently
        #                           being pushed                  (GUI thread)
        self.progress_label = QLabel("Reading files:")
        self.progress_label.setVisible(False)
        root.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.parse_progress_label = QLabel("Parsing / pushing to Veusz:")
        self.parse_progress_label.setVisible(False)
        root.addWidget(self.parse_progress_label)
        self.parse_progress_bar = QProgressBar()
        self.parse_progress_bar.setVisible(False)
        root.addWidget(self.parse_progress_bar)

        self.column_progress_label = QLabel("Current file - columns:")
        self.column_progress_label.setVisible(False)
        root.addWidget(self.column_progress_label)
        self.column_progress_bar = QProgressBar()
        self.column_progress_bar.setVisible(False)
        root.addWidget(self.column_progress_bar)

        bbox = QHBoxLayout()
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self._process_files)
        bbox.addWidget(self.process_button)
        self.save_button = QPushButton("Save Veusz Project (.vszh5)")
        self.save_button.clicked.connect(self._save_project)
        self.save_button.setEnabled(False)
        bbox.addWidget(self.save_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        bbox.addWidget(self.close_button)
        root.addLayout(bbox)

    # subclasses override
    def _populate_options(self, form: QFormLayout) -> None:
        pass

    # ----- theme handling ---------------------------------------------------
    def _set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        apply_theme(QApplication.instance(), mode)
        self.act_dark.setChecked(mode == "dark")
        self.act_light.setChecked(mode == "light")
        self.log("Theme set to %s mode" % mode)

    # ----- file list --------------------------------------------------------
    def _browse_files(self) -> None:
        # subclasses set self._file_filter
        flt = getattr(self, "_file_filter", "All Files (*.*)")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select files", "", flt
        )
        if files:
            for f in files:
                if f not in self.selected_files:
                    self.selected_files.append(f)
            self._refresh_file_list()
            self.log("Selected %d new file(s); %d total." %
                     (len(files), len(self.selected_files)))

    def _clear_files(self) -> None:
        self.selected_files.clear()
        self._refresh_file_list()
        self.log("File list cleared.")

    def _refresh_file_list(self) -> None:
        self.file_list_widget.clear()
        for p in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(p))

    # ----- log / status -----------------------------------------------------
    def log(self, msg: str) -> None:
        self.status_text.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
        try:
            self.status_text.ensureCursorVisible()
        except Exception:
            pass

    def update_mem_label(self, rss_mb: float) -> None:
        self.mem_label.setText("RSS: %.1f MiB" % rss_mb)

    # ----- progress helpers ------------------------------------------------
    def show_progress_bars(self, n_files: int) -> None:
        """Make all three progress bars visible and size the file-level ones to
        ``n_files``.  The column bar is sized lazily as each file is pushed."""
        for w in (self.progress_label, self.progress_bar,
                  self.parse_progress_label, self.parse_progress_bar,
                  self.column_progress_label, self.column_progress_bar):
            w.setVisible(True)
        self.progress_bar.setRange(0, max(1, n_files))
        self.progress_bar.setValue(0)
        self.parse_progress_bar.setRange(0, max(1, n_files))
        self.parse_progress_bar.setValue(0)
        self.column_progress_bar.setRange(0, 1)
        self.column_progress_bar.setValue(0)

    def hide_progress_bars(self) -> None:
        for w in (self.progress_label, self.progress_bar,
                  self.parse_progress_label, self.parse_progress_bar,
                  self.column_progress_label, self.column_progress_bar):
            w.setVisible(False)

    def begin_column_progress(self, file_label: str, n_cols: int) -> None:
        """Reset the column bar at the start of pushing one file."""
        self.column_progress_label.setText("Current file - columns: %s" % file_label)
        self.column_progress_bar.setRange(0, max(1, n_cols))
        self.column_progress_bar.setValue(0)

    def tick_column_progress(self, done: int) -> None:
        self.column_progress_bar.setValue(done)

    # ----- abstract hooks ---------------------------------------------------
    def _process_files(self) -> None:
        raise NotImplementedError

    def _save_project(self) -> None:
        raise NotImplementedError

    def _show_about(self) -> None:
        QMessageBox.information(
            self, "About",
            "AutoPlot suite\n"
            "Veusz embedded GUI (qtpy abstraction)\n"
            "Author: William W. Wallace\n"
            "Compatible with Veusz 3.4 and 4.1."
        )


# ---------------------------------------------------------------------------
# %% I/O helpers shared by both apps
# ---------------------------------------------------------------------------
def open_maybe_gzipped(path: str):
    """Return a binary file handle for ``path``; transparently gunzip *.gz."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


# ---------------------------------------------------------------------------
# %% MJD -> date string conversion
# ---------------------------------------------------------------------------
# Modified Julian Date epoch: MJD 0.0 == 1858-11-17 00:00:00 UTC.
# JD = MJD + 2400000.5  -- we implement the conversion directly without
# astropy.time so that the helper can run in light-weight contexts
# (Veusz plugin call paths, etc.) without paying the astropy import cost.
MJD_EPOCH_JD_OFFSET = 2400000.5


def mjd_to_datestr(mjd_arr: "np.ndarray",
                   fmt: str = "%Y-%m-%d_%H:%M:%S") -> "np.ndarray":
    """
    Convert an array of Modified Julian Dates (UTC, double precision) into
    an array of formatted timestamp strings.

    Parameters
    ----------
    mjd_arr : array-like of float
        MJD values.  NaN/Inf and non-finite entries are NEVER dropped --
        the output array is always the same length as ``mjd_arr``.  Veusz
        text datasets cannot carry a true NaN, so non-finite MJDs become
        the sentinel string ``"NaN"`` (rather than "") to make the missing
        sample explicit and visually distinguishable.
    fmt : str, optional
        ``strftime`` format string; defaults to ``YYYY-MM-DD_HH:MM:SS``.

    Returns
    -------
    numpy.ndarray of dtype ``<U`` (Unicode) with one string per input MJD.

    Notes
    -----
    Implementation uses Fliegel & Van Flandern's integer Julian-day-number
    algorithm so it is vectorisable, deterministic and tz-free (UTC).
    """
    a = np.asarray(mjd_arr, dtype=float)
    out = np.empty(a.shape, dtype=object)
    finite = np.isfinite(a)
    # Replace non-finite slots with 0 for the vectorised math; we mask them
    # back to the sentinel "NaN" string at the end.  This avoids
    # RuntimeWarnings from the int64 cast when NaN/Inf values are present,
    # and crucially preserves array length so callers never "drop" rows.
    a_safe = np.where(finite, a, 0.0)
    # integer part of JD (Julian Day Number) + fractional day
    jd = a_safe + MJD_EPOCH_JD_OFFSET
    # Split into integer JDN and fractional day-of-day
    jdn = np.floor(jd + 0.5).astype(np.int64)
    frac = (jd + 0.5) - jdn   # in [0, 1)
    # Fliegel & Van Flandern algorithm
    L = jdn + 68569
    N = (4 * L) // 146097
    L = L - (146097 * N + 3) // 4
    I = (4000 * (L + 1)) // 1461001
    L = L - (1461 * I) // 4 + 31
    J = (80 * L) // 2447
    day = L - (2447 * J) // 80
    L = J // 11
    month = J + 2 - 12 * L
    year = 100 * (N - 49) + I + L
    # time of day
    seconds_of_day = frac * 86400.0
    hh = np.floor(seconds_of_day / 3600.0).astype(np.int64)
    mm = np.floor((seconds_of_day - hh * 3600.0) / 60.0).astype(np.int64)
    ss = seconds_of_day - hh * 3600.0 - mm * 60.0
    # Compose strings (fast path: avoid datetime allocation per element)
    import datetime as _dt
    n = a.size
    flat_out = np.empty(n, dtype=object)
    a_f = a.reshape(-1)
    fin_f = finite.reshape(-1)
    yr = year.reshape(-1)
    mo = month.reshape(-1)
    dy = day.reshape(-1)
    hr = hh.reshape(-1)
    mi = mm.reshape(-1)
    se = ss.reshape(-1)
    for k in range(n):
        if not fin_f[k]:
            # NaN-preserving sentinel: Veusz text datasets have no native
            # NaN, but we must keep array length consistent with the
            # numeric companion so downstream sorts/joins line up.
            flat_out[k] = "NaN"
            continue
        # Re-use strftime so callers can request any format they like.
        try:
            # round seconds to int for display while preserving the float
            # microsecond if the format requests it.
            sec_int = int(round(float(se[k])))
            # handle wrap-around if rounding nudged seconds to 60
            extra_min = sec_int // 60
            sec_int = sec_int % 60
            mi_eff = int(mi[k]) + extra_min
            extra_hr = mi_eff // 60
            mi_eff = mi_eff % 60
            hr_eff = int(hr[k]) + extra_hr
            dt = _dt.datetime(int(yr[k]), int(mo[k]), int(dy[k]),
                              hr_eff % 24, mi_eff, sec_int)
            # If hour wrapped, advance the date by one day
            if hr_eff >= 24:
                dt = dt + _dt.timedelta(days=hr_eff // 24)
            flat_out[k] = dt.strftime(fmt)
        except Exception:
            # Defensive: bad calendar values from a corrupt MJD also map
            # to the NaN sentinel (still length-preserving).
            flat_out[k] = "NaN"
    out = flat_out.reshape(a.shape)
    return np.asarray(out.tolist())


# ---------------------------------------------------------------------------
# %% NRAO FITS unit-warning helpers
# ---------------------------------------------------------------------------
# NRAO 1PPS-delta FITS files use non-standard FITS unit strings:
#   * CHANNELA / CHANNELB carry unit='none'  (text columns, no real unit)
#   * DELTAT  carries unit='NanoSeconds'    (should be FITS 'ns')
# astropy.units flags both as UnitsWarning, and astropy.table follows up
# with a "kept as MaskedColumn...convert to Quantity failed" warning for
# the text columns.  The two helpers below register the unit aliases and
# provide a context manager that filters the residual harmless warnings.
_NRAO_UNITS_REGISTERED = False


def register_nrao_fits_units() -> None:
    """
    Register the non-standard NRAO FITS unit aliases with astropy.units so
    that ``QTable.read`` / ``Table.read`` no longer emit a UnitsWarning.

    Idempotent -- safe to call from every worker thread, and from every
    plugin invocation.  Falls back silently if astropy is not importable
    (this module is also used by code paths that never touch FITS).
    """
    global _NRAO_UNITS_REGISTERED
    if _NRAO_UNITS_REGISTERED:
        return
    try:
        from astropy import units as u
    except Exception:
        # astropy is not available in this process; nothing to do.
        _NRAO_UNITS_REGISTERED = True
        return
    new_units = []
    # 'none' -- placeholder unit used on text columns.  Map it to a custom
    # dimensionless unit so the FITS unit parser accepts it.
    try:
        u.Unit("none")
    except Exception:
        try:
            none_unit = u.def_unit(
                "none",
                represents=u.dimensionless_unscaled,
                doc="NRAO placeholder unit on text columns (treated as "
                    "dimensionless).",
            )
            new_units.append(none_unit)
        except Exception:
            pass
    # 'NanoSeconds' -- should be FITS 'ns'.  Define as an explicit alias
    # for u.nanosecond so callers can still get proper time arithmetic.
    try:
        u.Unit("NanoSeconds")
    except Exception:
        try:
            ns_unit = u.def_unit(
                ["NanoSeconds", "nanoseconds", "NanoSecond", "nanosecond_alias"],
                represents=u.ns,
                doc="NRAO-style alias for the SI nanosecond.",
            )
            new_units.append(ns_unit)
        except Exception:
            pass
    if new_units:
        try:
            u.add_enabled_units(new_units)
        except Exception:
            pass
    _NRAO_UNITS_REGISTERED = True


@contextlib.contextmanager
def suppress_fits_unit_warnings():
    """
    Context manager that filters the harmless FITS unit-related warnings
    emitted by ``astropy.io.fits`` and ``astropy.table`` when reading
    NRAO 1PPS-delta files.  Real problems (corrupt headers, bad casts,
    etc.) still propagate because the filter is narrow.

    Use this around ``QTable.read``/``Table.read``/``fits.open`` calls in
    worker threads to keep the log readable across hundreds of files.
    """
    with warnings.catch_warnings():
        # UnitsWarning -- emitted when a non-standard unit string is parsed.
        try:
            from astropy.units import UnitsWarning  # type: ignore
            warnings.simplefilter("ignore", UnitsWarning)
        except Exception:
            pass
        # AstropyUserWarning -- emitted by astropy.table for the
        # "kept as MaskedColumn ... attempt to convert it to Quantity failed"
        # message.  Filter only this specific message text so other
        # AstropyUserWarning instances still surface.
        try:
            from astropy.utils.exceptions import AstropyUserWarning  # type: ignore
            warnings.filterwarnings(
                "ignore",
                message=r".*kept as a? ?MaskedColumn.*",
                category=AstropyUserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*has a unit but is kept.*",
                category=AstropyUserWarning,
            )
        except Exception:
            pass
        # As a final safety net, hide the generic "did not parse as fits unit"
        # message regardless of which category astropy raises it under.
        warnings.filterwarnings(
            "ignore",
            message=r".*did not parse as fits unit.*",
        )
        yield


def safe_dsname(name: str) -> str:
    """Sanitize an arbitrary string for use as a Veusz dataset name."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if s and s[0].isdigit():
        s = "ds_" + s
    return s or "dataset"
