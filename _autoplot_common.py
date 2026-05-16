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

# %%%% 0.0.14: Datetime display via xy.labels (text dataset) instead of
# Date: 2026-05-16
#              SetDataDateTime.  Real-host testing on Veusz 3.4 showed
#              every SetDataDateTime call raises ``unsupported operand
#              type(s) for -: 'float' and 'datetime.datetime'`` from
#              INSIDE Veusz internals (the epoch-subtraction path of
#              SetDataDateTime), making all dt-duplicate pages render
#              empty.  The pivot keeps the existing numeric ``__sorted``
#              seconds dataset as the x-axis binding (which works) and
#              instead annotates each xy trace with a parallel TEXT
#              dataset bound to ``xy.labels.val``: roughly
#              ``DEFAULT_DATETIME_LABEL_ANCHORS`` (default 10) evenly
#              spaced date strings of the form
#              ``YYYY-MM-DD HH:MM:SS``, with ``""`` at every other
#              index so Veusz's renderer draws nothing there.  New
#              helpers in this module:
#                * build_sparse_datestr_dataset(doc, name, mjd_array,
#                  n_anchors=10, fmt=..., log_cb=None) -- pushes a
#                  same-length text dataset with date strings only at
#                  ``n_anchors`` evenly spaced indices and ``""``
#                  everywhere else.  Default rendering on the dt pages.
#                * build_full_datestr_dataset(doc, name, mjd_array,
#                  fmt=..., log_cb=None) -- pushes a same-length text
#                  dataset with the date string at EVERY index (one
#                  label per data point).  Opt-in via the GUI checkbox
#                  because long traces become visually crowded.
#                * style_xy_datetime_labels(xy, angle=45, size='6pt',
#                  posnVert='centre', posnHorz='left') -- applies
#                  rotated compact label formatting to the xy widget's
#                  Label sub-group.
#                * DEFAULT_DATETIME_LABEL_ANCHORS / ANGLE / SIZE / FMT
#                  module constants for the default look.
#              Both pipelines now build BOTH sparse and full datasets
#              for every dt-eligible trace and bind ``xy.labels.val``
#              to one or the other based on the new GUI flag
#              ``datetime_full_labels`` (default False -> sparse).
#              The user can toggle the project between sparse and full
#              after the fact by editing the xy widget's Labels setting
#              in Veusz; both datasets stay in the document.
#              set_datetime_dataset() is RETAINED but documented as
#              deprecated in its docstring -- callers that explicitly
#              know they are running on Veusz 4.1+ can still use it.
#              FITS_AutoPlot and Franks_AutoPlot now wire every dt
#              site through the labels helpers instead of
#              SetDataDateTime, so the dt-duplicate pages render
#              correctly on both Veusz 3.4 and 4.1.
#              Perf: mjd_to_datestr() now has a vectorised fast path
#              for the default ``%Y-%m-%d_%H:%M:%S`` format that
#              composes the output with np.char string ops and skips
#              the per-element datetime/strftime allocation entirely
#              (~2-3x faster on 10k+ point traces, which matters for
#              the new full-labels rendering).  Any other strftime
#              format string still uses the per-element fallback so
#              callers retain full strftime support.  GPU + threadpool
#              audit re-confirmed: gpu_argsort is wired at every
#              FITS/Franks sort site (per-file, per-tag, overlay) and
#              run_in_threadpool's 2x cpu_count over-subscription
#              stays appropriate for the IO-bound FITS read path.

# %%%% 0.0.13: Identity-stable trace styling + datetime-duplicate hardening.
# Date: 2026-05-16
#                * apply_trace_style(xy, identity_key, vary_style=False)
#                  assigns each trace a deterministic colour (md5 over the
#                  identity key, mapped into TRACE_COLOR_PALETTE) and a
#                  line style.  Line is shown by default with a 1pt width;
#                  markers shrink to 1pt so dense traces remain legible.
#                * Identity key is (column_name, tag_tuple) so the SAME
#                  trace gets the SAME (colour, line-style) on every page
#                  -- per-file, datetime-duplicate, and unit-overlay --
#                  across both FITS and Franks pipelines.
#                * vary_style activates only when a graph carries more
#                  than TRACE_STYLE_VARY_THRESHOLD (16) traces, at which
#                  point the line style cycles through TRACE_LINE_STYLES
#                  (solid / dashed / dotted / dash-dot) keyed by the same
#                  stable hash.  Small graphs stay fully solid.
#                * set_datetime_dataset(doc, name, secs, log_cb) coerces
#                  every value to a plain Python float, replaces NaN/inf
#                  with 0.0, and logs '+datetime dataset <name>' on
#                  success or 'SetDataDateTime FAILED for <name>' on
#                  failure.  Wraps every SetDataDateTime call across the
#                  FITS + Franks pipelines so real-Veusz no longer
#                  silently drops dt-duplicate pages when the input array
#                  contains NumPy float64 / NaN.

# %%%% 0.0.12: Combined-in-time overlay semantics.  The unit-overlay
# Date: 2026-05-16
#               builders now stitch every file's samples for a given
#               (unit, column) into a single time-sorted xy trace -- one
#               line per column instead of N lines per file.  The
#               concatenated x is also emitted as a Veusz date-time
#               dataset for the datetime-duplicate page.
#               Fix: AutoPlotMainWindow.log() previously assumed
#               ``self.status_text`` already existed, but subclass
#               ``_populate_options()`` runs during _build_central_widget
#               BEFORE that widget is created and legitimately calls
#               log() (e.g. "GPU backend: ...").  log() now buffers any
#               pre-widget messages and flushes them as soon as
#               status_text is constructed, eliminating the
#               AttributeError seen at startup.
#               Fix: ``open_in_veusz_app()`` previously launched
#               ``python -m veusz`` with closed stdio under
#               DETACHED_PROCESS, which silently exited on Windows for
#               some Veusz wheels.  The launcher now tries the
#               pip-installed ``veusz``/``veusz.exe`` console script
#               first, then ``python -m veusz``, then
#               ``python -m veusz.veusz_main``, then an inline
#               ``from veusz.veusz_main import run`` call.  Each
#               candidate is polled for ~700 ms before declaring
#               success and the strategy that worked is logged.
#               stdin/stdout/stderr are redirected to DEVNULL (instead
#               of closed) so Veusz's startup banner does not abort
#               the child, and CREATE_NO_WINDOW suppresses the brief
#               console flash.

# %%%% 0.0.11: Added datetime-duplicate plot support.  Plots whose x axis
# Date: 2026-05-16
#              is a Modified Julian Date column (or any column known to be
#              MJD-valued via the ``SORTED_KEY_HINT`` convention) can now
#              optionally be duplicated with a parallel x axis that uses a
#              proper Veusz date-time dataset and a YYYY-MM-DD HH:MM:SS
#              tick label format.
#                * ``MJD_VEUSZ_EPOCH_MJD`` = 54832.0 (MJD of 2009-01-01)
#                  -- Veusz stores date-time datasets as seconds since
#                  2009-01-01 UTC, see veusz/utils/dates.py.
#                * ``mjd_to_veusz_seconds(mjd_arr)`` returns a 1-D float64
#                  numpy array of Veusz date-time seconds; NaN-preserving
#                  (non-finite MJD -> NaN seconds so the datetime dataset
#                  keeps the same length and Veusz simply skips that
#                  sample on plotting).
#                * ``style_datetime_x_axis(axis, rotate_deg, fmt,
#                  major_ticks_target)`` applies a date format string,
#                  rotates tick labels, and sets a target number of major
#                  ticks so the cadence is readable for typical batches.
#                  The default 45-degree rotation keeps a 19-character
#                  ``YYYY-MM-DD HH:MM:SS`` label legible without major
#                  visual overlap.  Compatible with both Veusz 3.4 and 4.1
#                  (the TickLabels.format / TickLabels.rotate /
#                  MajorTicks.number settings are stable since Veusz 1.x).

# %%%% 0.0.10: Optional GPU acceleration via CuPy.  Added
# Date: 2026-05-16
#              ``is_gpu_available()``, ``enable_gpu(flag)`` (process-wide
#              toggle), ``set_gpu_argsort_threshold(n)``, and
#              ``gpu_argsort(arr, force=None)`` which returns the same
#              ``np.argsort`` index array but uses CuPy under the hood
#              when the array is large enough (default >= 200 000
#              samples; tuneable via the setter) and CuPy is importable.
#              CuPy is a soft dependency -- import failure is silent and
#              the helpers always fall through to NumPy.  Used by the
#              per-file argsort step in push_to_veusz()/
#              push_franks_to_veusz() (the dominant O(N log N) cost when
#              N is large).  CuPy's radix-sort on the GPU is typically
#              2.7-10x faster than NumPy's SIMD sort once the array is
#              big enough to amortise the host<->device transfer (see
#              https://gist.github.com/magnium/cf96160d248a79f9463439695a7748e8).
#              Small-array workloads continue to use NumPy on the CPU.
#              GUI surfaces gained a 'Use GPU acceleration (CuPy)'
#              checkbox that is disabled and tooltipped when CuPy is not
#              importable on the host -- there is no install or
#              configuration burden when CuPy is absent.

# %%%% 0.0.9: Added broken-axis helpers used by FITS_AutoPlot and
# Date: 2026-05-16
#             Franks_AutoPlot to generate plots that handle large time
#             gaps without dead-space:
#               * detect_time_breaks(x, k_factor=10, absolute_gap=0)
#                 returns a list of (start, end) gap pairs.  Auto
#                 threshold is ``K * median(|diff(x)|)`` (default K=10);
#                 a positive ``absolute_gap`` overrides this in units of
#                 ``x`` (seconds for time, days for MJD, etc.).
#               * break_pairs_to_breakpoints(pairs) flattens to the
#                 [s1,e1,s2,e2,...] FloatList format Veusz's
#                 axis-broken widget expects in its ``breakPoints``
#                 setting.
#               * make_broken_x_axis(graph, pairs, label, show_gridlines)
#                 removes the default 'x' axis of a graph and replaces
#                 it with an ``axis-broken`` widget whose breakPoints
#                 are set.  No-op if pairs is empty (caller keeps the
#                 plain axis).  Veusz 3.4 and 4.1 compatible (the
#                 widget and the breakPoints setting are stable since
#                 Veusz 1.17, 2014).
# %%%%% Function Descriptions

# %%%% 0.0.8: Added ``open_in_veusz_app(filename)`` helper that launches the
# Date: 2026-05-16
#             full standalone Veusz GUI in the current Python environment
#             via ``subprocess.Popen([sys.executable, '-m', 'veusz', fn])``
#             so the freshly-saved .vszh5 project can be inspected without
#             leaving the user's active venv / conda env.
#             AutoPlotMainWindow gained an 'Open in Veusz...' button in the
#             bottom button row (greyed out until ``mark_project_saved()``
#             is called with a valid path).  Subclasses call
#             ``self.mark_project_saved(fn)`` from their _save_project()
#             override after save_vszh5() returns successfully.
#             Also: parallelization audit.  ``MAX_THREADS`` defaults were
#             bumped from ``cpu_count`` to ``cpu_count * 2`` in
#             FITS_AutoPlot.py and Franks_AutoPlot.py because the file-read
#             stage is I/O bound (GIL released in numpy + astropy) and
#             oversubscription measurably helps on slow filesystems.
#             The Veusz push phase is kept serial intentionally:
#             ``veusz.embed.Embedded`` is documented as single-threaded --
#             all document operations must come from one thread.

# %%%% 0.0.6: AutoPlotMainWindow now provides three labelled progress bars
# Date: 2026-05-16
#             (read / parse / per-column) plus helper methods
#             ``show_progress_bars()``, ``hide_progress_bars()``,
#             ``begin_column_progress()`` and ``tick_column_progress()``.
#             Subclasses use these to drive a multi-stage progress UI.

# %%%% 0.0.5: Added Spyder-style cell markers (``# %% TITLE`` for top-level
# Date: 2026-05-16
#             sections, ``# %%% TITLE`` for nested sections) on the
#             existing dashed banner blocks so the file is navigable in
#             Spyder's Outline / cell navigator.  Pure cosmetic change --
#             no runtime behaviour modified.

# %%%% 0.0.4: Added register_nrao_fits_units() and suppress_fits_unit_warnings()
# Date: 2026-05-16
#             helpers.  NRAO 1PPS-delta FITS files carry non-standard unit
#             strings (``'none'`` on CHANNELA/CHANNELB, ``'NanoSeconds'`` on
#             DELTAT) that astropy.io.fits / QTable.read flag with a noisy
#             UnitsWarning -- and astropy.table additionally warns that the
#             text columns are kept as MaskedColumn because the unit cannot
#             be converted to a Quantity.  The new helpers register these
#             unit aliases with astropy.units and provide a context manager
#             that filters the residual harmless warnings so a 900-file
#             batch run no longer floods the log.

# %%%% 0.0.3: NaN-preserving dataset emission policy. Numeric datasets keep
# Date: 2026-05-16
#             NaN floats (Veusz natively supports NaN in numeric datasets;
#             plots simply skip NaN samples). Text datasets cannot carry a
#             true NaN, so non-finite MJD inputs to mjd_to_datestr() now
#             yield the explicit sentinel string ``"NaN"`` (length-preserving)
#             rather than an empty string -- this prevents NaN rows from
#             being silently dropped or confused with a missing token.

# %%%% 0.0.2: Added mjd_to_datestr() helper for MJD -> YYYY-MM-DD_HH:MM:SS
# Date: 2026-05-16
#             string conversion used by all four AutoPlot modules.

# %%%% 0.0.1: Initial implementation of shared infrastructure
# Date: 2026-05-16

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
        open_in_veusz_app: launch the full Veusz GUI in the current Python
            env and load the given .vszh5 file.
        detect_time_breaks/break_pairs_to_breakpoints/make_broken_x_axis:
            time-gap detection and Veusz axis-broken widget helpers.
        build_sparse_datestr_dataset/build_full_datestr_dataset/
        style_xy_datetime_labels: v0.0.14 sparse and full date-string
            text datasets bound to ``xy.labels.val`` (the per-point
            label property on the xy widget) so dt pages render readable
            ``YYYY-MM-DD HH:MM:SS`` annotations without going through
            the buggy ``SetDataDateTime`` path on Veusz 3.4.
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
import subprocess
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
        QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
        QComboBox, QLineEdit, QTextEdit,
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
        QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
        QComboBox, QLineEdit, QTextEdit,
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


# %%% Launch full Veusz GUI in current Python env
# ---------------------------------------------------------------------------
def open_in_veusz_app(filename, python_exe=None):
    """
    Launch the *full* Veusz GUI in the current Python environment and load
    ``filename`` (typically a .vszh5 project just written by save_vszh5).

    The standalone Veusz application is invoked as a Python module so it
    uses the exact interpreter that is running this AutoPlot session --
    that guarantees the GUI sees the same numpy / astropy / qtpy stack and
    the same site-packages-installed Veusz that produced the file.

    Parameters
    ----------
    filename : str
        Path to the .vszh5 project to open.
    python_exe : Optional[str]
        Override the interpreter; defaults to ``sys.executable``.

    Returns
    -------
    subprocess.Popen
        Handle to the spawned Veusz process.  The caller is not required
        to wait on it -- the new process is detached so the parent (the
        AutoPlot GUI) can be closed without killing the Veusz session.
    """
    # v0.0.12: launching the full Veusz GUI is fiddly across Windows /
    # Linux + Anaconda / pip installs.  ``python -m veusz`` works on
    # most modern wheels but several older site-packages layouts expose
    # the GUI ONLY via the ``veusz`` / ``veusz.exe`` console script that
    # pip generates in the env's Scripts/ (or bin/) directory.  Even
    # when ``-m veusz`` is importable, on Windows it can silently exit
    # under DETACHED_PROCESS because stdio handles are closed while
    # Veusz tries to write a startup banner.  We try strategies in
    # order, return the first one that survives a brief sanity poll,
    # and raise a descriptive RuntimeError if every strategy dies.
    py = python_exe or sys.executable
    py_dir = os.path.dirname(py)
    is_win = (os.name == "nt")

    # Common detach kwargs.  On Windows, pipe stdio to DEVNULL instead of
    # closing the handles -- Veusz writes a banner at startup and the
    # process dies if its stdout is unwritable.  CREATE_NO_WINDOW also
    # hides the transient console flash you get with python.exe.
    base_kwargs = {"close_fds": True}
    try:
        base_kwargs["stdin"] = subprocess.DEVNULL
        base_kwargs["stdout"] = subprocess.DEVNULL
        base_kwargs["stderr"] = subprocess.DEVNULL
    except Exception:
        pass
    if is_win:
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        base_kwargs["creationflags"] = (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        )
    else:
        base_kwargs["start_new_session"] = True

    strategies = []
    # 1. ``veusz`` console script next to the active interpreter (most
    #    reliable on Anaconda envs because it is generated by pip and
    #    points back at this exact python).
    script_name = "veusz.exe" if is_win else "veusz"
    candidate_dirs = []
    if is_win:
        candidate_dirs.append(os.path.join(py_dir, "Scripts"))
        candidate_dirs.append(py_dir)
    else:
        candidate_dirs.append(os.path.join(py_dir, ".."))
        candidate_dirs.append(os.path.join(py_dir, "..", "bin"))
        candidate_dirs.append(py_dir)
    seen_paths = set()
    for d in candidate_dirs:
        try:
            cand = os.path.normpath(os.path.join(d, script_name))
        except Exception:
            continue
        if cand in seen_paths:
            continue
        seen_paths.add(cand)
        if os.path.isfile(cand):
            strategies.append(("script", [cand, filename]))
    # 2. ``python -m veusz`` -- the standard module entry.
    strategies.append(("module-veusz", [py, "-m", "veusz", filename]))
    # 3. ``python -m veusz.veusz_main`` -- older Veusz layouts (3.4 era).
    strategies.append(("module-veusz_main",
                       [py, "-m", "veusz.veusz_main", filename]))
    # 4. ``python -c "from veusz.veusz_main import run; run()"`` -- last
    #    ditch: import the entry directly so we get a real ImportError
    #    if Veusz isn't installed, instead of a silent exit.
    inline_code = (
        "import sys; from veusz.veusz_main import run; "
        "sys.argv=['veusz', %r]; run()"
    ) % filename
    strategies.append(("inline-run", [py, "-c", inline_code]))

    last_exc = None
    for label, cmd in strategies:
        try:
            proc = subprocess.Popen(cmd, **base_kwargs)
        except FileNotFoundError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            continue
        # Give the child ~700 ms to crash (e.g. ImportError on a stale
        # entry point).  If it survives, declare success.
        try:
            import time as _t
            _t.sleep(0.7)
        except Exception:
            pass
        rc = proc.poll()
        if rc is None:
            # Still running -- attach the launch label for debugging.
            try:
                setattr(proc, "_autoplot_launch_strategy", label)
            except Exception:
                pass
            return proc
        else:
            last_exc = RuntimeError(
                "strategy %r exited immediately with code %s" % (label, rc)
            )
            continue

    raise RuntimeError(
        "Could not launch the full Veusz GUI.  Tried: %s.  Last error: %s.  "
        "Verify ``%s -m veusz`` works in a terminal, or that the ``veusz`` "
        "executable is on PATH in this environment."
        % (", ".join(s[0] for s in strategies), last_exc, py)
    )


# ---------------------------------------------------------------------------
# %% BROKEN-AXIS HELPERS
# ---------------------------------------------------------------------------
# %%% detect_time_breaks
def detect_time_breaks(x, k_factor=10.0, absolute_gap=0.0):
    # type: (np.ndarray, float, float) -> List[Tuple[float, float]]
    """
    Detect large gaps in a 1-D monotonic-ish array of time samples and return
    a list of (start, end) pairs suitable for Veusz's ``axis-broken`` widget.

    A 'gap' is any consecutive pair where ``x[i+1] - x[i]`` exceeds the
    chosen threshold.  Two threshold sources are supported, with the
    absolute value winning when given:

        * ``absolute_gap > 0`` -- treat any Δt > absolute_gap as a break.
          Units must match ``x`` (e.g. seconds if x is seconds, days if x
          is MJD).  Pass 0.0 (or negative) to disable.
        * Auto: gap > ``k_factor * median(|diff(x)|)``.  This is robust to
          uniform sampling rates that differ between files and avoids
          surprising the user when the typical Δt changes by orders of
          magnitude across loaded files.

    NaN samples are removed from ``x`` before gap detection (NaN-NaN diffs
    are NaN which would otherwise propagate).  Non-monotonic input is
    tolerated: we first ``np.sort`` the input copy so the gap analysis is
    well-defined.  The returned pairs are also produced in sorted-x space,
    which is what Veusz wants for ``breakPoints``.

    Returns
    -------
    list of (gap_start, gap_end) tuples in the units of ``x``.  Empty list
    if no break is detected, or if there are fewer than 3 finite samples.
    """
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return []
    arr = np.sort(arr)
    d = np.diff(arr)
    if d.size == 0:
        return []
    # Absolute threshold wins if the user gave a positive value.
    if absolute_gap and absolute_gap > 0.0:
        thresh = float(absolute_gap)
    else:
        med = float(np.median(np.abs(d[d > 0])) if np.any(d > 0) else 0.0)
        if med <= 0.0:
            return []
        thresh = float(k_factor) * med
    mask = d > thresh
    if not np.any(mask):
        return []
    starts = arr[:-1][mask]
    stops = arr[1:][mask]
    # Shrink the pair very slightly so the break does not eat the first /
    # last sample on either side of the gap when Veusz computes ticks.
    eps = 1e-9 * max(1.0, float(np.nanmax(np.abs(arr))))
    pairs = []  # type: List[Tuple[float, float]]
    for s, e in zip(starts, stops):
        s2 = float(s) + eps
        e2 = float(e) - eps
        if e2 > s2:
            pairs.append((s2, e2))
    return pairs


# %%% break_pairs_to_breakpoints
def break_pairs_to_breakpoints(pairs):
    # type: (List[Tuple[float, float]]) -> List[float]
    """Flatten a list of (start, end) tuples to Veusz's flat-list format
    [s1, e1, s2, e2, ...] as expected by the ``breakPoints`` setting of the
    ``axis-broken`` widget."""
    out = []  # type: List[float]
    for s, e in pairs:
        out.append(float(s))
        out.append(float(e))
    return out


# %%% make_broken_x_axis
def make_broken_x_axis(graph, break_pairs, label="", show_gridlines=True):
    """
    Add an ``axis-broken`` widget named ``x`` to ``graph`` with the given
    list of (start, end) break pairs, after removing the default ``x``
    axis that ``graph`` already carries.  Returns the new axis widget.

    The function is idempotent: if a widget named ``x`` already exists on
    ``graph`` it is removed first.  If ``break_pairs`` is empty the
    function does nothing and returns None (the caller should keep the
    default plain axis in that case).

    Compatible with both Veusz 3.4 and 4.1 because the ``axis-broken``
    widget and its ``breakPoints`` FloatList setting have been stable
    since Veusz 1.17 (2014).
    """
    if not break_pairs:
        return None
    # Remove the default 'x' axis first.  Both Veusz versions accept
    # Remove() on a child widget by name via the parent's API.
    try:
        graph.Remove("x")
    except Exception:
        # Fall back: it may have been removed already, or the API differs.
        try:
            for child in list(getattr(graph, "children", [])):
                if getattr(child, "name", None) == "x":
                    try:
                        child.Remove()
                    except Exception:
                        pass
        except Exception:
            pass
    ax = graph.Add("axis-broken", name="x")
    try:
        ax.direction.val = "horizontal"
    except Exception:
        pass
    try:
        ax.breakPoints.val = break_pairs_to_breakpoints(break_pairs)
    except Exception:
        pass
    if label:
        try:
            ax.label.val = str(label)
        except Exception:
            pass
    if show_gridlines:
        try:
            ax.GridLines.hide.val = False
        except Exception:
            pass
    return ax


# ---------------------------------------------------------------------------
# %% DATETIME-DUPLICATE PLOT HELPERS (v0.0.11)
# ---------------------------------------------------------------------------
#
# Veusz stores date-time datasets internally as a float64 array of seconds
# since 2009-01-01 00:00:00 UTC (see veusz/utils/dates.py, ``offsetdate``).
# Modified Julian Date 54832.0 corresponds to that same epoch.  Therefore
# the conversion from MJD to a Veusz date-time scalar is simply:
#
#     veusz_seconds = (mjd - 54832.0) * 86400.0
#
# When a date-time dataset is bound to an xy widget's ``xData``, Veusz
# automatically formats the tick labels using its ``%VDx`` strftime tokens
# instead of the default ``%Vg`` numeric formatter.  The default tick label
# rotation is 0; we set it to 45 degrees so a 19-character
# ``YYYY-MM-DD HH:MM:SS`` label fits without overlapping its neighbours at
# typical sampling cadences.
#
# These helpers are pure-numpy (no astropy / no datetime allocations) so
# they are cheap enough to call inside the plugin path on every file.

# MJD that corresponds to Veusz's internal date-time zero (2009-01-01 UTC).
MJD_VEUSZ_EPOCH_MJD = 54832.0
# Default datetime tick-label format string used everywhere we render a
# duplicate-with-datetime-axis page.  %VDx tokens are Veusz's strftime-ish
# tokens, defined in the manual under "Axis numeric scales".  Result:
#     2024-03-15 14:30:00
DEFAULT_DATETIME_TICK_FORMAT = "%VDY-%VDm-%VDd %VDH:%VDM:%VDS"
# Default tick label rotation -- 45 degrees keeps a 19-character label
# readable at typical cadences.  Veusz accepts integer or stringified
# integer degrees ("45" or 45 both work).
DEFAULT_DATETIME_TICK_ROTATE_DEG = 45
# Target number of major ticks along the x axis.  Veusz auto-picks the
# spacing to satisfy roughly this many major ticks; we keep it modest so
# the rotated date strings have room to breathe.
DEFAULT_DATETIME_MAJOR_TICKS_TARGET = 8


def mjd_to_veusz_seconds(mjd_arr):
    # type: (np.ndarray) -> np.ndarray
    """
    Convert an array of Modified Julian Dates (UTC) to the float-seconds
    units Veusz uses internally for date-time datasets.  Non-finite MJD
    inputs become NaN seconds, which Veusz silently skips on plots while
    preserving the dataset length (so a date-time dataset stays aligned
    with its companion numeric dataset row-for-row).
    """
    a = np.asarray(mjd_arr, dtype=float).ravel()
    finite = np.isfinite(a)
    out = np.full(a.shape, np.nan, dtype=float)
    # vectorised: (mjd - epoch) * 86400 seconds/day
    out[finite] = (a[finite] - MJD_VEUSZ_EPOCH_MJD) * 86400.0
    return out


def _coerce_axis_setting_to_str(value):
    """Some Veusz axis settings (rotate, in particular) are documented as
    accepting either an int or its string form.  Older Veusz versions
    sometimes reject the int -- the safe move is to pass a string."""
    if isinstance(value, str):
        return value
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def style_datetime_x_axis(axis,
                          rotate_deg=DEFAULT_DATETIME_TICK_ROTATE_DEG,
                          fmt=DEFAULT_DATETIME_TICK_FORMAT,
                          major_ticks_target=DEFAULT_DATETIME_MAJOR_TICKS_TARGET,
                          label=""):
    """
    Apply the standard datetime tick-label format and angled rotation to
    an axis widget.  Works on both plain ``axis`` and ``axis-broken``
    widgets because both share the same TickLabels / MajorTicks setting
    groups.

    Each ``try/except`` is independent so that the most-supported
    settings (format, rotate) still land even if a less-supported one
    (MajorTicks/number) is renamed on an exotic Veusz build.

    Compatible with both Veusz 3.4 and 4.1 (these setting groups exist
    unchanged since Veusz 1.x).
    """
    if axis is None:
        return axis
    # Date format string for tick labels.
    try:
        axis.TickLabels.format.val = str(fmt)
    except Exception:
        pass
    # Angled rotation so a 19-char date+time string fits.
    try:
        axis.TickLabels.rotate.val = _coerce_axis_setting_to_str(rotate_deg)
    except Exception:
        # Some Veusz builds expect a numeric value; try that too.
        try:
            axis.TickLabels.rotate.val = int(round(float(rotate_deg)))
        except Exception:
            pass
    # Target tick density.
    try:
        axis.MajorTicks.number.val = int(major_ticks_target)
    except Exception:
        pass
    # Axis label, if requested.
    if label:
        try:
            axis.label.val = str(label)
        except Exception:
            pass
    # Force gridlines on so the date-aware plot stays readable when many
    # ticks land close together.
    try:
        axis.GridLines.hide.val = False
    except Exception:
        pass
    return axis


# ---------------------------------------------------------------------------
# %% GPU ACCELERATION HELPERS (CuPy, optional)
# ============================================================================
#
# Design notes:
#   * CuPy is a SOFT dependency.  We try to import it once at module load,
#     and every public helper degrades to NumPy when the import fails or
#     when the user toggles the global flag off.
#   * The GPU is only used when the input array is big enough to amortise
#     the host<->device transfer.  The default 200 000 element threshold
#     was picked from public benchmarks where CuPy first overtakes NumPy
#     sort/argsort on a mid-range consumer GPU (see the 0.0.10 revision
#     note above).  The threshold is settable from the GUI.
#   * Only the argsort step is GPU-accelerated.  detect_time_breaks does
#     a tiny sort + diff that is overwhelmed by H2D transfer; we leave
#     it on the CPU.  Image-HDU dataset transfers are dominated by the
#     embedded-Veusz round-trip and gain nothing from GPU.

try:                       # CuPy is OPTIONAL.
    import cupy as _cp     # type: ignore
    _HAS_CUPY = True
except Exception:          # pragma: no cover -- depends on host
    _cp = None
    _HAS_CUPY = False

# Process-wide enable flag (driven by the GUI checkbox).  Off by default.
_GPU_ENABLED = False
# Minimum array size at which we attempt the GPU sort.  Tuneable.
_GPU_ARGSORT_THRESHOLD = 200_000


def is_gpu_available() -> bool:
    """True iff CuPy imported successfully AND at least one CUDA device is
    visible to it.  Cheap: we cache the runtime-device check after the
    first call so the GUI can poll this on every refresh."""
    if not _HAS_CUPY:
        return False
    cached = getattr(is_gpu_available, "_cache", None)
    if cached is not None:
        return bool(cached)
    try:
        n = int(_cp.cuda.runtime.getDeviceCount())
        ok = n > 0
    except Exception:
        ok = False
    setattr(is_gpu_available, "_cache", ok)
    return ok


def gpu_backend_name() -> str:
    """Short string describing the GPU backend status for the GUI/log."""
    if not _HAS_CUPY:
        return "CuPy not installed -- CPU only"
    if not is_gpu_available():
        return "CuPy installed but no CUDA device -- CPU only"
    try:
        dev = _cp.cuda.Device(0)
        try:
            # CuPy 11+/12+
            props = _cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"GPU")
            if isinstance(name, bytes):
                name = name.decode("ascii", "replace")
        except Exception:
            name = "CUDA device 0"
        return "CuPy ready: %s" % name
    except Exception:
        return "CuPy ready"


def enable_gpu(flag: bool) -> None:
    """Toggle the process-wide GPU acceleration flag.  Safe to call even
    when CuPy is not installed (it will simply have no effect)."""
    global _GPU_ENABLED
    _GPU_ENABLED = bool(flag) and is_gpu_available()


def gpu_enabled() -> bool:
    """Return the current global GPU-enabled state."""
    return bool(_GPU_ENABLED)


def set_gpu_argsort_threshold(n: int) -> None:
    """Set the minimum array size (samples) at which the optional CuPy
    path is used.  Smaller arrays fall through to NumPy."""
    global _GPU_ARGSORT_THRESHOLD
    try:
        _GPU_ARGSORT_THRESHOLD = max(0, int(n))
    except Exception:
        pass


def get_gpu_argsort_threshold() -> int:
    return int(_GPU_ARGSORT_THRESHOLD)


def gpu_argsort(arr, force=None):
    """
    Return ``np.argsort(arr)`` (ascending, int64) with optional CuPy
    acceleration.

    Parameters
    ----------
    arr : array_like
        1-D numeric array.
    force : None or bool
        * None (default) -- use GPU iff ``gpu_enabled()`` is True AND the
          array is large enough AND CuPy is available.
        * True -- attempt GPU regardless of size (still falls back to
          NumPy if anything fails).
        * False -- always use NumPy (useful for inner loops where the
          caller already decided).

    The function is type-stable: it always returns a NumPy ``int64``
    array, so callers using ``arr[idx]`` work unchanged whether CuPy was
    used or not.
    """
    a = np.asarray(arr)
    n = a.size
    use_gpu = False
    if force is True:
        use_gpu = _HAS_CUPY and is_gpu_available()
    elif force is False:
        use_gpu = False
    else:
        use_gpu = (gpu_enabled()
                   and n >= _GPU_ARGSORT_THRESHOLD
                   and _HAS_CUPY
                   and is_gpu_available())
    if not use_gpu:
        return np.argsort(a, kind="stable")
    try:
        # Transfer -> sort -> bring index array back.
        g = _cp.asarray(a)
        gi = _cp.argsort(g)
        idx = _cp.asnumpy(gi).astype(np.int64, copy=False)
        # Free GPU memory promptly so subsequent files do not OOM the
        # device.
        del g
        del gi
        try:
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return idx
    except Exception:
        # Any failure -- fall back to NumPy silently.  The GUI logs a
        # one-time warning on first failure.
        if not getattr(gpu_argsort, "_warned", False):
            setattr(gpu_argsort, "_warned", True)
            import warnings
            warnings.warn("gpu_argsort: CuPy path failed -- using NumPy. "
                          "Future calls will continue silently.",
                          RuntimeWarning, stacklevel=2)
        return np.argsort(a, kind="stable")


# ============================================================================
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
        # Path of the most recently saved .vszh5 project (None until
        # mark_project_saved() is called by a subclass _save_project()).
        self._last_saved_path: Optional[str] = None
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
        # v0.0.12: flush any messages buffered by log() during the
        # subclass _populate_options() call above, which ran before
        # status_text existed.
        pending = getattr(self, "_pending_log", None)
        if pending:
            for ln in pending:
                try:
                    self.status_text.append(ln)
                except Exception:
                    pass
            self._pending_log = []

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
        # %% Open-in-Veusz button (enabled after a successful save)
        self.open_veusz_button = QPushButton("Open in Veusz...")
        self.open_veusz_button.setToolTip(
            "Launch the full Veusz GUI in the current Python environment "
            "and load the most recently saved project."
        )
        self.open_veusz_button.clicked.connect(self._open_last_in_veusz)
        self.open_veusz_button.setEnabled(False)
        bbox.addWidget(self.open_veusz_button)
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
        # v0.0.12: subclass _populate_options() runs before the log widget
        # exists, so any early log() calls (e.g. "GPU backend: ...") would
        # AttributeError on self.status_text.  Buffer messages emitted
        # before the widget exists, then flush them once the widget is
        # built.  Also guard ensureCursorVisible() against a stray None.
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        widget = getattr(self, "status_text", None)
        if widget is None:
            buf = getattr(self, "_pending_log", None)
            if buf is None:
                buf = []
                self._pending_log = buf
            buf.append(line)
            return
        # Drain any pre-widget buffer first so messages appear in order.
        pending = getattr(self, "_pending_log", None)
        if pending:
            for ln in pending:
                try:
                    widget.append(ln)
                except Exception:
                    pass
            self._pending_log = []
        try:
            widget.append(line)
        except Exception:
            pass
        try:
            widget.ensureCursorVisible()
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

    # ----- saved-project tracking / Open-in-Veusz --------------------------
    def mark_project_saved(self, filename: str) -> None:
        """Record the path of a freshly saved .vszh5 project and enable the
        'Open in Veusz...' button.  Subclasses should call this from their
        ``_save_project`` after ``save_vszh5`` returns successfully."""
        if filename:
            self._last_saved_path = filename
            self.open_veusz_button.setEnabled(True)
            self.open_veusz_button.setToolTip(
                "Open %s in the full Veusz GUI (current Python env)." % filename
            )

    def _open_last_in_veusz(self) -> None:
        """Slot for the 'Open in Veusz...' button.  Spawns a detached
        full-Veusz GUI on the last saved project using the active interpreter.
        """
        fn = self._last_saved_path
        if not fn:
            QMessageBox.information(
                self, "No saved project",
                "Save a Veusz project first, then this button will open it."
            )
            return
        if not os.path.exists(fn):
            QMessageBox.warning(
                self, "File missing",
                "The previously saved project no longer exists:\n\n%s" % fn
            )
            return
        try:
            proc = open_in_veusz_app(fn)
            strat = getattr(proc, "_autoplot_launch_strategy", "unknown")
            self.log("Launched Veusz (PID %s, strategy=%s) on %s" %
                     (getattr(proc, "pid", "?"), strat, fn))
        except FileNotFoundError as exc:
            self.log("Open in Veusz failed: %s" % exc)
            QMessageBox.critical(
                self, "Open in Veusz failed",
                "Could not launch Veusz with the current Python interpreter:\n\n%s\n\n"
                "Make sure the ``veusz`` package is installed in this environment "
                "(``python -m pip install veusz``)." % exc
            )
        except Exception as exc:
            self.log("Open in Veusz failed: %s" % exc)
            QMessageBox.critical(self, "Open in Veusz failed", str(exc))

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
    # Compose strings.
    #
    # v0.0.14 perf path: for the most common format string
    # "%Y-%m-%d_%H:%M:%S" (used by the new full-labels rendering on
    # 10k+ point traces), build the output array with vectorised NumPy
    # string operations and skip the per-element datetime/strftime
    # allocation entirely.  For any other ``fmt`` we fall back to the
    # general per-element path so callers retain full strftime support.
    import datetime as _dt
    n = a.size
    a_f = a.reshape(-1)
    fin_f = finite.reshape(-1)
    yr = year.reshape(-1)
    mo = month.reshape(-1)
    dy = day.reshape(-1)
    hr = hh.reshape(-1)
    mi = mm.reshape(-1)
    se = ss.reshape(-1)

    # ---- normalise seconds wrap-around (vectorised) --------------------
    # ``ss`` can round up to 60 after int-rounding, which then propagates
    # to minutes/hours/days.  Do this in one shot so both the fast and
    # slow paths see consistent calendar values.
    sec_int_v = np.rint(se).astype(np.int64)
    extra_min_v = sec_int_v // 60
    sec_int_v = sec_int_v % 60
    mi_eff_v = mi.astype(np.int64) + extra_min_v
    extra_hr_v = mi_eff_v // 60
    mi_eff_v = mi_eff_v % 60
    hr_eff_v = hr.astype(np.int64) + extra_hr_v
    extra_day_v = hr_eff_v // 24
    hr_eff_v = hr_eff_v % 24
    # Apply any whole-day carry to the calendar date.  np.datetime64
    # arithmetic handles month/year roll-over correctly, which is rare
    # but possible right at midnight UTC after seconds round-up.
    if np.any(extra_day_v != 0):
        try:
            base_dates = (np.asarray(yr, dtype="datetime64[Y]")
                          + (mo.astype(np.int64) - 1).astype("timedelta64[M]")
                          + (dy.astype(np.int64) - 1).astype("timedelta64[D]")
                          + extra_day_v.astype("timedelta64[D]"))
            yr_eff = base_dates.astype("datetime64[Y]").astype(np.int64) + 1970
            mo_eff = (base_dates.astype("datetime64[M]").astype(np.int64) % 12) + 1
            dy_eff = ((base_dates - base_dates.astype("datetime64[M]"))
                      .astype("timedelta64[D]").astype(np.int64) + 1)
        except Exception:
            # If the datetime64 path fails for any pathological calendar
            # value, fall back to the per-element loop below.
            yr_eff = yr.astype(np.int64)
            mo_eff = mo.astype(np.int64)
            dy_eff = dy.astype(np.int64)
    else:
        yr_eff = yr.astype(np.int64)
        mo_eff = mo.astype(np.int64)
        dy_eff = dy.astype(np.int64)

    flat_out = np.empty(n, dtype=object)

    if fmt == "%Y-%m-%d_%H:%M:%S":
        # ---- vectorised fast path -------------------------------------
        # np.char.zfill expects a string array, so cast once and pad.
        def _z(arr, w):
            return np.char.zfill(
                np.asarray(arr, dtype=np.int64).astype("U%d" % (w + 1)),
                w,
            )
        y4 = _z(yr_eff, 4)
        m2 = _z(mo_eff, 2)
        d2 = _z(dy_eff, 2)
        h2 = _z(hr_eff_v, 2)
        n2 = _z(mi_eff_v, 2)
        s2 = _z(sec_int_v, 2)
        # Compose "YYYY-MM-DD_HH:MM:SS" without any Python-level loop.
        joined = np.char.add(np.char.add(np.char.add(y4, "-"), m2), "-")
        joined = np.char.add(np.char.add(joined, d2), "_")
        joined = np.char.add(np.char.add(np.char.add(joined, h2), ":"), n2)
        joined = np.char.add(np.char.add(joined, ":"), s2)
        # NaN-preserving sentinel for non-finite MJDs.
        joined = np.where(fin_f, joined, "NaN")
        flat_out[:] = joined.tolist()
    else:
        # ---- general per-element fallback (any strftime fmt) ----------
        for k in range(n):
            if not fin_f[k]:
                # NaN-preserving sentinel: Veusz text datasets have no
                # native NaN, but we must keep array length consistent
                # with the numeric companion so downstream sorts/joins
                # line up.
                flat_out[k] = "NaN"
                continue
            try:
                dt = _dt.datetime(int(yr_eff[k]), int(mo_eff[k]),
                                  int(dy_eff[k]),
                                  int(hr_eff_v[k]), int(mi_eff_v[k]),
                                  int(sec_int_v[k]))
                flat_out[k] = dt.strftime(fmt)
            except Exception:
                # Defensive: bad calendar values from a corrupt MJD also
                # map to the NaN sentinel (still length-preserving).
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


def set_datetime_dataset(doc, name, secs, log_cb=None):
    """Push a Veusz datetime dataset, coercing values to plain floats.

    Veusz' embedded ``SetDataDateTime`` is strict on some builds: it
    refuses NumPy scalars (``np.float64``) and silently truncates lists
    containing NaN.  This wrapper converts the iterable to a plain
    Python ``list[float]`` with NaN replaced by ``0.0`` before pushing,
    and logs both success AND failure verbosely so the duplicate-page
    path becomes diagnosable from the log panel.

    Returns True on success, False on failure.

    .. deprecated:: 0.0.14
        ``SetDataDateTime`` raises ``unsupported operand type(s) for
        -: 'float' and 'datetime.datetime'`` inside Veusz internals on
        Veusz 3.4 (NRAO's installed version), making every dt-axis
        plot fail.  The dt-page code paths now use
        :func:`add_datetime_anchor_labels` instead, which annotates
        the existing numeric seconds-x-axis with sparse Veusz
        ``label`` widgets carrying YYYY-MM-DD HH:MM:SS text.  This
        helper is retained for callers that explicitly want the
        datetime-dataset behaviour on Veusz 4.1+.
    """
    try:
        import math
        clean = []
        bad = 0
        for v in secs:
            try:
                f = float(v)
            except Exception:
                bad += 1
                f = 0.0
            if not math.isfinite(f):
                bad += 1
                f = 0.0
            clean.append(f)
        if log_cb and bad:
            log_cb("    note: %d non-finite/non-numeric "
                   "datetime sample(s) replaced with 0.0 in %s"
                   % (bad, name))
        doc.SetDataDateTime(name, clean)
        if log_cb:
            log_cb("    +datetime dataset %s (%d samples)"
                   % (name, len(clean)))
        return True
    except Exception as exc:
        if log_cb:
            log_cb("    SetDataDateTime FAILED for %s: %s" % (name, exc))
        return False


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


# ============================================================================
# %%% TRACE-STYLE PALETTE (v0.0.13)
# ----------------------------------------------------------------------------
# Identity-stable color + line-style assignment.  The plotting helpers in
# FITS_AutoPlot.py and Franks_AutoPlot.py call ``apply_trace_style`` for
# every xy widget so that ``(column, tag_tuple)`` always maps to the
# same (color, line-style) pair across every plot in the project (per-
# file pages, datetime-duplicate pages, and cross-file overlays).
#
# Rules:
#   * Color is primary -- a 16-color palette indexed by a stable hash of
#     the identity key (column, tag_tuple).
#   * Line is shown by default.
#   * Line style is fixed ``solid`` while a graph has <= 16 traces.
#     Once the trace count on a graph exceeds 16, callers pass
#     ``vary_style=True`` and the style cycles
#     solid -> dashed -> dotted -> dash-dot, also keyed by the same
#     stable hash so the same identity gets the same style everywhere
#     it appears.
#   * Markers stay visible at 1pt so dense traces still have anchors.
# ============================================================================
TRACE_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
]
TRACE_LINE_STYLES = ["solid", "dashed", "dotted", "dash-dot"]

# Style threshold: when the number of traces on a graph reaches this
# value, callers should pass ``vary_style=True`` so the line-style
# cycle activates.  Below the threshold every trace stays solid.
TRACE_STYLE_VARY_THRESHOLD = len(TRACE_COLOR_PALETTE)  # 16


def _stable_index(key, modulus):
    """Deterministic non-negative index in ``[0, modulus)``.

    Uses Python's ``hash`` on the canonical string form of the key.
    Python's per-process hash randomisation does NOT apply to str hashes
    when the same interpreter session is used end-to-end, BUT to be
    fully reproducible across processes we md5 the string form.
    """
    import hashlib
    if isinstance(key, (list, tuple)):
        skey = "|".join("" if k is None else str(k) for k in key)
    else:
        skey = "" if key is None else str(key)
    h = hashlib.md5(skey.encode("utf-8", "replace")).digest()
    # 4 bytes -> unsigned int
    n = (h[0] << 24) | (h[1] << 16) | (h[2] << 8) | h[3]
    return n % max(1, int(modulus))


def trace_style_for(identity_key, vary_style=False):
    """Return ``(color_hex, line_style_str)`` for an identity key.

    Parameters
    ----------
    identity_key : hashable
        Recommended: ``(column_name, tag_tuple)``.  ``tag_tuple`` may be
        ``(None,)`` for untagged HDUs.
    vary_style : bool
        When ``False`` (default), line style is always ``"solid"``.
        When ``True``, the style cycles through ``TRACE_LINE_STYLES``
        keyed by the same stable hash, so the same identity gets the
        same style on every graph that varies styles.
    """
    color = TRACE_COLOR_PALETTE[
        _stable_index(identity_key, len(TRACE_COLOR_PALETTE))
    ]
    if vary_style:
        style = TRACE_LINE_STYLES[
            _stable_index(identity_key, len(TRACE_LINE_STYLES))
        ]
    else:
        style = "solid"
    return color, style


def apply_trace_style(xy, identity_key, vary_style=False,
                      show_line=True, marker="circle",
                      marker_size="1pt", line_width="1pt"):
    """Apply identity-stable color + line style + markers to a Veusz xy.

    All attribute writes are wrapped in try/except so older Veusz
    versions that lack a property simply ignore that one tweak.
    """
    color, style = trace_style_for(identity_key, vary_style=vary_style)
    try:
        xy.marker.val = marker
    except Exception:
        pass
    try:
        xy.markerSize.val = marker_size
    except Exception:
        pass
    try:
        xy.MarkerFill.color.val = color
    except Exception:
        pass
    try:
        xy.MarkerLine.color.val = color
    except Exception:
        pass
    try:
        xy.PlotLine.hide.val = not bool(show_line)
    except Exception:
        pass
    try:
        xy.PlotLine.color.val = color
    except Exception:
        pass
    try:
        xy.PlotLine.style.val = style
    except Exception:
        pass
    try:
        xy.PlotLine.width.val = line_width
    except Exception:
        pass
    return color, style


# ============================================================================
# %%% SPARSE DATETIME LABELS (v0.0.14)
# ----------------------------------------------------------------------------
# Replacement strategy for the dt duplicate pages.  Instead of pushing a
# Veusz datetime dataset via ``SetDataDateTime`` (which fails inside Veusz
# internals on the user's Veusz 3.4 with
# ``unsupported operand type(s) for -: 'float' and 'datetime.datetime'``),
# we keep the existing numeric ``__sorted`` seconds x dataset for every
# trace and attach a parallel TEXT dataset to the xy widget's ``labels``
# property.  Veusz's xy widget renders one label per data point from the
# text dataset; empty strings render as no-ops, so a SPARSE labels array
# (date string at ~10 evenly spaced indices, ``""`` everywhere else) gives
# us readable date annotations on the trace without obscuring the data.
#
# Why this works everywhere:
#   * ``xy.labels`` and ``xy.Label`` (formatting group) have been part of
#     Veusz's xy/point widget since well before 3.x, so the same code
#     path runs on Veusz 3.4 and 4.1 unchanged.
#   * ``SetDataText`` is the boring text-dataset API and does not touch
#     the buggy SetDataDateTime epoch-subtraction path.
#   * Length-preserving NaN handling stays consistent with ``mjd_to_datestr``
#     (non-finite samples become ``""`` so they render nothing).
# ============================================================================
DEFAULT_DATETIME_LABEL_ANCHORS = 10
DEFAULT_DATETIME_LABEL_ANGLE = 45
DEFAULT_DATETIME_LABEL_SIZE = "6pt"
DEFAULT_DATETIME_LABEL_FMT = "%Y-%m-%d %H:%M:%S"


def build_sparse_datestr_dataset(doc, name, mjd_array,
                                 n_anchors=DEFAULT_DATETIME_LABEL_ANCHORS,
                                 fmt=DEFAULT_DATETIME_LABEL_FMT,
                                 log_cb=None):
    """Push a same-length sparse date-string text dataset to Veusz.

    The returned dataset has the same length as ``mjd_array``; every
    element is ``""`` except at ``n_anchors`` evenly spaced finite
    indices, where the element is a ``YYYY-MM-DD HH:MM:SS`` formatted
    string converted from the corresponding MJD via
    :func:`mjd_to_datestr`.  Bind it to ``xy.labels.val`` to render
    sparse date annotations along an xy trace.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        Active embedded Veusz document.
    name : str
        Veusz dataset name to create (use ``safe_dsname`` upstream).
    mjd_array : array-like of float
        Modified Julian Date values, same length as the trace's y data
        (so labels align row-for-row with markers).  Non-finite entries
        are skipped when picking anchor indices.
    n_anchors : int, optional
        Approximate number of date-string annotations to scatter along
        the trace.  Default 10.  Clamped to ``[1, len(finite indices)]``.
    fmt : str, optional
        ``strftime`` format string.  Default ``"%Y-%m-%d %H:%M:%S"``.
    log_cb : callable, optional
        ``log_cb(str)`` for human-readable progress messages.

    Returns
    -------
    str or None
        ``name`` on success (the dataset is now in ``doc``), ``None``
        on any failure (e.g. empty input, SetDataText raised).

    Notes
    -----
    * Veusz text datasets cannot carry a true NaN; we use ``""`` as the
      missing-value sentinel because the xy.labels renderer draws empty
      strings as nothing.  This keeps the labels array length-aligned
      with the trace's x/y datasets without producing spurious glyphs.
    * Both Veusz 3.4 and 4.1 accept a plain ``list[str]`` here, so we
      never have to touch ``SetDataDateTime``.
    """
    try:
        arr = np.asarray(mjd_array, dtype=float)
    except Exception as exc:
        if log_cb:
            log_cb("    sparse-datestr build failed for %s: %s"
                   % (name, exc))
        return None
    n = int(arr.size)
    if n == 0:
        if log_cb:
            log_cb("    sparse-datestr skipped for %s: empty input" % name)
        return None
    finite_mask = np.isfinite(arr)
    finite_idx = np.flatnonzero(finite_mask)
    if finite_idx.size == 0:
        if log_cb:
            log_cb("    sparse-datestr skipped for %s: no finite MJDs"
                   % name)
        return None
    # Pick up to n_anchors evenly spaced indices from the FINITE subset
    # so anchors always land on real data points.
    k = max(1, int(n_anchors))
    k = min(k, int(finite_idx.size))
    # ``np.linspace`` with endpoints included so the first and last
    # anchors fall at the trace endpoints when possible.
    if k == 1:
        anchor_positions = np.array([finite_idx[finite_idx.size // 2]],
                                    dtype=np.int64)
    else:
        sel = np.linspace(0, finite_idx.size - 1, k).round().astype(np.int64)
        # unique in case rounding produced duplicates at small sizes
        sel = np.unique(sel)
        anchor_positions = finite_idx[sel]
    # Convert only the anchor MJDs to strings; everywhere else stays "".
    try:
        anchor_strs = mjd_to_datestr(arr[anchor_positions], fmt=fmt)
    except Exception as exc:
        if log_cb:
            log_cb("    sparse-datestr conversion failed for %s: %s"
                   % (name, exc))
        return None
    labels = [""] * n
    for pos, s in zip(anchor_positions.tolist(),
                      np.asarray(anchor_strs).tolist()):
        # NaN-sentinel handling: mjd_to_datestr returns "NaN" for non-
        # finite MJDs (defensive fallback), which would render as the
        # literal string "NaN" on the plot.  Map it back to "" so the
        # missing sample is silently skipped.
        if s is None or s == "NaN":
            labels[pos] = ""
        else:
            labels[pos] = str(s)
    try:
        doc.SetDataText(name, labels)
    except Exception as exc:
        if log_cb:
            log_cb("    SetDataText failed for %s: %s" % (name, exc))
        return None
    if log_cb:
        log_cb("    +sparse-datestr labels %s (%d anchors / %d points)"
               % (name, int(anchor_positions.size), n))
    return name


def build_full_datestr_dataset(doc, name, mjd_array,
                               fmt=DEFAULT_DATETIME_LABEL_FMT,
                               log_cb=None):
    """Push a same-length per-point date-string text dataset to Veusz.

    Each element of the returned dataset is the
    ``YYYY-MM-DD HH:MM:SS`` formatted string for the corresponding MJD
    (one label per data point).  Non-finite MJDs map to ``""`` so they
    render as nothing and never produce a spurious ``"NaN"`` glyph on
    the trace.

    Bind it to ``xy.labels.val`` when the user opts in via the GUI
    ``datetime_full_labels`` checkbox.  Note that for long traces
    (thousands of samples) every label is rendered, which can be
    visually crowded -- the default rendering on dt pages remains the
    sparse variant; this helper is here so the dataset is available in
    the document and the user can switch to full labels by repointing
    ``xy.labels`` in Veusz without rerunning the pipeline.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        Active embedded Veusz document.
    name : str
        Veusz dataset name to create (use ``safe_dsname`` upstream).
    mjd_array : array-like of float
        Modified Julian Date values, same length as the trace's y data.
    fmt : str, optional
        ``strftime`` format string.  Default ``"%Y-%m-%d %H:%M:%S"``.
    log_cb : callable, optional
        ``log_cb(str)`` for human-readable progress messages.

    Returns
    -------
    str or None
        ``name`` on success, ``None`` on any failure.
    """
    try:
        arr = np.asarray(mjd_array, dtype=float)
    except Exception as exc:
        if log_cb:
            log_cb("    full-datestr build failed for %s: %s"
                   % (name, exc))
        return None
    n = int(arr.size)
    if n == 0:
        if log_cb:
            log_cb("    full-datestr skipped for %s: empty input" % name)
        return None
    try:
        strs = mjd_to_datestr(arr, fmt=fmt)
    except Exception as exc:
        if log_cb:
            log_cb("    full-datestr conversion failed for %s: %s"
                   % (name, exc))
        return None
    # mjd_to_datestr returns the sentinel "NaN" for non-finite MJDs; we
    # convert those to "" so the label simply doesn't render on the
    # plot (no literal NaN glyph cluttering the trace).
    labels = []
    for s in np.asarray(strs).tolist():
        if s is None or s == "NaN":
            labels.append("")
        else:
            labels.append(str(s))
    try:
        doc.SetDataText(name, labels)
    except Exception as exc:
        if log_cb:
            log_cb("    SetDataText failed for %s: %s" % (name, exc))
        return None
    if log_cb:
        log_cb("    +full-datestr labels %s (%d labels)" % (name, n))
    return name


def style_xy_datetime_labels(xy,
                             angle=DEFAULT_DATETIME_LABEL_ANGLE,
                             size=DEFAULT_DATETIME_LABEL_SIZE,
                             posnVert="centre", posnHorz="left",
                             color="auto"):
    """Apply rotated, compact text formatting to an xy widget's labels.

    Use after binding ``xy.labels.val`` so the YYYY-MM-DD HH:MM:SS
    strings stay legible without obscuring the trace.  All writes are
    guarded so older Veusz versions that lack a sub-setting silently
    skip that one tweak.
    """
    if xy is None:
        return xy
    try:
        xy.Label.angle.val = float(angle)
    except Exception:
        try:
            xy.Label.angle.val = int(round(float(angle)))
        except Exception:
            pass
    try:
        xy.Label.size.val = str(size)
    except Exception:
        pass
    try:
        xy.Label.posnVert.val = str(posnVert)
    except Exception:
        pass
    try:
        xy.Label.posnHorz.val = str(posnHorz)
    except Exception:
        pass
    if color and color != "auto":
        try:
            xy.Label.color.val = str(color)
        except Exception:
            pass
    return xy

