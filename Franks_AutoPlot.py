# -*- coding: utf-8 -*-
"""
=============================================================================
Franks_AutoPlot.py
-----------------------------------------------------------------------------
# %% Header Info

GUI-driven batch processor for the "FranksProcessed" ASCII time-series data
files distributed alongside the NRAO 1PPS sync FITS bundle.  Two file
flavours are recognised automatically:

    * ``YYYY_MM.t00new`` - GB Timing Data (medians over 60 min), columns:
          MJD  date  time  Site1Hz  TAC  MicroSem  CNS_GPS1  MSM-GPS
          CNS_GPS  CNS2018  GBT_VLBA  RA  GBTRtn

    * ``time_gbt.dat``   - long-term GBT timing log, columns:
          MJD  EECO-REF  NIST-REF  NS  DATE  [COMMENTS]

Both flavours are read as numpy column arrays, pushed into a Veusz embedded
document as one dataset per column tagged with the file's base name (plus
``raw`` and ``sorted`` tags), and plotted column-vs-MJD on dedicated pages.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Initial implementation, FranksProcessed file support
Date: 2026-05-16
# %%%% 0.0.2: Added "Generate MJD->date strings" checkbox.  Whenever the
#             parsed file contains an MJD column, ticking the option emits
#             two additional Veusz text datasets (raw + sorted) of
#             YYYY-MM-DD_HH:MM:SS strings tagged with the file's base name
#             plus ``datestr``.
Date: 2026-05-16
# %%%% 0.0.3: NaN-preserving emission policy.  When a Franks token is
#             missing or non-numeric the parser now records ``NaN`` in the
#             numeric column (length-preserving) AND keeps the text token
#             ("" for fully missing) in the text companion -- rows are
#             never dropped.  Veusz numeric datasets carry the NaN through;
#             non-finite MJDs in the optional date-string text datasets
#             become the sentinel string ``"NaN"`` so all per-row arrays
#             retain the same length and stay index-aligned.
Date: 2026-05-16
# %%%% 0.0.5: _on_done() now calls QApplication.processEvents() between
#             every per-file push so the GUI stays responsive (log pane
#             scrolls live, buttons remain clickable) during a long
#             batch insert -- previously the window could appear stuck
#             when processing many .t00new files in one batch.
#             (No 0.0.4: aligned with FITS_AutoPlot.py version stream.)
# %%%% 0.0.6: Added Spyder IDE cell markers (# %% / # %%%) at all major
#             section banners and import subsections so the file can be
#             navigated and run cell-by-cell in Spyder's Outline view.
#             Cosmetic only -- no behavior change.
Date: 2026-05-16
# %%%% 0.0.7: Added two new progress bars in the GUI -- a 'Parsing/pushing'
#             file-level bar that ticks once per file as push_franks_to_veusz()
#             finishes that file, and a 'Current file - columns' bar that
#             ticks per source column.  push_franks_to_veusz() gained a
#             ``column_cb(done, total)`` parameter.  No skip-images knob
#             here (Franks files have no image HDUs).
Date: 2026-05-16
# %%%% 0.0.8: Parallelization audit + Open-in-Veusz button.
# Date: 2026-05-16
# Date: 2026-05-16
# %%%% 0.0.12: Combined-in-time overlay semantics.
#              build_unit_overlay_pages_franks now collapses each shared
#              column onto a single time-sorted xy trace stitched across
#              every file (legend names the column, e.g. 'DELTAT').  The
#              datetime-axis duplicate page mirrors the concatenation
#              with the Veusz-seconds version of the same x.
Date: 2026-05-16
# %%%% 0.0.11: Duplicate plots with a date-time x axis.  A new GUI
#              checkbox 'Duplicate plots with datetime X axis' makes
#              push_franks_to_veusz emit a companion Veusz date-time
#              dataset built from the file's MJD column (MJD->Veusz
#              seconds via the 2009-01-01 UTC epoch).  A clone of the
#              per-file page is appended right after the original page
#              with the same graphs, but the xy widgets reference the
#              date-time dataset and the x axis is formatted as
#              YYYY-MM-DD HH:MM:SS rotated 45 deg with ~8 major ticks.
#              build_unit_overlay_pages_franks gained a matching
#              ``datetime_duplicate`` kwarg; the cross-file overlay is
#              cloned the same way.  Helpers ``mjd_to_veusz_seconds``
#              and ``style_datetime_x_axis`` live in _autoplot_common.
Date: 2026-05-16
# %%%% 0.0.10: Optional GPU acceleration for the per-file argsort step.
#              When the 'Use GPU acceleration (CuPy)' checkbox is
#              checked and CuPy is importable on the host, the large
#              MJD-sort permutation is dispatched to ``gpu_argsort``
#              from ``_autoplot_common`` -- typically 2.7-10x faster
#              than NumPy on consumer GPUs once N exceeds ~200k
#              samples.  Smaller files still use NumPy.  CuPy is a soft
#              dependency: the checkbox is disabled and tooltipped when
#              CuPy is absent.
# %%%% 0.0.9: Per-plot broken-x-axis on time gaps + column-name overlay
#             pages.  push_franks_to_veusz() now takes new keyword args
#             ``plot_individual`` (default True), ``gap_k`` (default 10.0),
#             and ``gap_absolute`` (default 0.0); when ``plot_individual``
#             is False the per-file page is skipped (datasets are still
#             pushed) so the user can produce overlay-only projects.  When
#             time-axis breaks are detected, the per-file grid graphs are
#             given a native ``axis-broken`` X axis.  Added
#             ``build_unit_overlay_pages_franks`` which post-builds one
#             page per shared column name across the loaded batch (Franks
#             files have no unit annotations, so grouping is by column
#             name).  GUI gained a ``Gap K (× median Δt)`` spin, an
#             ``Absolute gap`` override spin (0 = auto), and a
#             ``Combined plots only`` checkbox that suppresses per-file
#             pages but keeps the overlays.
#               * MAX_THREADS default bumped from ``os.cpu_count() or 4``
#                 to ``(os.cpu_count() or 4) * 2``.  parse_franks_file()'s
#                 work is overwhelmingly I/O bound (file read) and bounded
#                 string -> float conversion; both release the GIL through
#                 numpy.  Oversubscription helps on slow storage.
#               * parse_franks_file() vectorized: previously a Python double
#                 loop with ``try: float(tok) except ValueError: NaN`` per
#                 cell.  Now tokenizes once into a rectangular numpy str
#                 matrix (right-padded for short rows) and converts each
#                 column with ``np.frompyfunc(_to_float, 1, 1).astype(float)``
#                 -- NaN-on-failure semantics preserved, ~10-30x faster on
#                 long time_gbt logs because the Python interpreter loop is
#                 collapsed into a numpy ufunc loop.
#               * 'Open in Veusz...' button inherited from base class
#                 AutoPlotMainWindow.  ``_save_project`` now calls
#                 ``self.mark_project_saved(written)`` to enable it so the
#                 user can launch the full Veusz GUI in the current Python
#                 env on the freshly saved project.
Date: 2026-05-16
# %%%%% Function Descriptions
        main: build QApplication and open the AutoPlot main window.
        FranksAutoPlotWindow: qtpy main window with the Touchstone-style
            file list / options / log / progress layout and the View menu
            dark/light theme toggle.
        FranksBatchWorker: QThread that parses selected files in parallel.
        parse_franks_file: format-autodetecting reader.  Returns a dict of
            column-shaped numpy arrays plus the sort key.
        push_franks_to_veusz: push column arrays into the embedded Veusz
            document, tagged with filename + raw/sorted, and create one
            scatter plot per numeric column versus MJD.
# %%%%% Variable Descriptions
        MAX_THREADS: top-of-file thread-pool size knob.
        DEFAULT_RSS_HIGH_WATER_MB: RSS threshold for memmap spill.
        T00_COLUMNS / TIMEDAT_COLUMNS: hardcoded column lists for the two
            file flavours (used when the on-file header is malformed).
# %%%%% More Info
        Because Veusz cannot natively import these custom ASCII formats
        (mixed date + numeric columns), we always import the raw text via
        Python (numpy.loadtxt) and produce derived "sorted" datasets in
        Veusz; both the raw and sorted variants are kept so the user can
        compare them, and both are tagged accordingly.
=============================================================================
"""
# %% Imports
from __future__ import annotations

# ============================================================================
# %%% IMPORTS - Standard library
# ============================================================================
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# %%% IMPORTS - Scientific
# ============================================================================
import numpy as np

# ============================================================================
# %%% IMPORTS - Veusz embedded
# ============================================================================
import veusz.embed as vz_embed

# ============================================================================
# %%% IMPORTS - shared GUI helpers
# ============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _autoplot_common import (   # noqa: E402
    Qt, QThread, Signal,
    QApplication, QFileDialog, QFormLayout, QLabel, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit,
    AutoPlotMainWindow,
    MemoryAwareCache, MemoryMonitor, MemoryMonitorConfig,
    open_embedded, save_vszh5, run_in_threadpool, safe_dsname,
    mjd_to_datestr,
    detect_time_breaks, make_broken_x_axis,
    gpu_argsort, is_gpu_available, enable_gpu, gpu_backend_name,
    # v0.0.11: datetime-duplicate helpers
    mjd_to_veusz_seconds, style_datetime_x_axis,
    set_datetime_dataset,
    MJD_VEUSZ_EPOCH_MJD,
    DEFAULT_DATETIME_TICK_FORMAT, DEFAULT_DATETIME_TICK_ROTATE_DEG,
    DEFAULT_DATETIME_MAJOR_TICKS_TARGET,
)

# v0.0.11: Franks files always use MJD as their sort key, so the upper-cased
# sort-key column name is *always* in this set.  We keep the set explicit so
# the conditional reads the same as FITS_AutoPlot, and so future Franks
# flavours (e.g. one that drops MJD in favour of JD) only need to extend it.
MJD_LIKE_SORT_KEYS = ("MJD", "DMJD")
JD_LIKE_SORT_KEYS = ("JD",)

# ============================================================================
# %% SCRIPT-LEVEL KNOBS
# ============================================================================
# I/O-bound default: ASCII reads + np.genfromtxt both release the GIL on
# the heavy paths, so we get a real speedup from oversubscribing the CPU.
MAX_THREADS = max(1, (os.cpu_count() or 4) * 2)
DEFAULT_RSS_HIGH_WATER_MB = 1024

T00_COLUMNS = ["MJD", "date", "time", "Site1Hz", "TAC", "MicroSem",
               "CNS_GPS1", "MSM_GPS", "CNS_GPS", "CNS2018",
               "GBT_VLBA", "RA", "GBTRtn"]

TIMEDAT_COLUMNS = ["MJD", "EECO_REF", "NIST_REF", "NS", "DATE"]


# ============================================================================
# %% FILE PARSER
# ============================================================================
def _detect_flavour(path: str) -> str:
    name = os.path.basename(path).lower()
    if name.endswith(".t00new") or ".t00new." in name:
        return "t00new"
    if "time_gbt" in name:
        return "time_gbt"
    # try sniff
    with open(path, "rt", errors="replace") as fh:
        head = fh.readline()
    if "GB Timing Data" in head:
        return "t00new"
    return "time_gbt"


def _safe_columns(raw_names: List[str]) -> List[str]:
    out = []
    for n in raw_names:
        n2 = re.sub(r"[^A-Za-z0-9_]", "_", n.strip())
        if not n2:
            n2 = "col%d" % len(out)
        if n2[0].isdigit():
            n2 = "c_" + n2
        out.append(n2)
    return out


def parse_franks_file(path: str, cache: MemoryAwareCache) -> Dict[str, Any]:
    """
    Read one Franks-style ASCII file into a dict of column arrays.

    Returns
    -------
    {
      'flavour'      : 't00new' or 'time_gbt'
      'columns'      : OrderedDict-like {colname: np.ndarray}
      'sort_key'     : 'MJD'
      'header_lines' : list[str]
      'base_name'    : str
    }
    """
    flavour = _detect_flavour(path)
    base = os.path.basename(path)
    base = re.sub(r"\.(t00new|dat)(\..*)?$", "", base)
    base = safe_dsname(base)

    header_lines: List[str] = []
    columns: Dict[str, np.ndarray] = {}

    with open(path, "rt", errors="replace") as fh:
        lines = fh.readlines()

    # find data start
    if flavour == "t00new":
        # line[0]=TITLE, line[1]=column names, line[2+]=data
        header_lines = [lines[0].rstrip()] if lines else []
        colnames_line = lines[1] if len(lines) > 1 else ""
        names = _safe_columns(colnames_line.split())
        if len(names) < 13:
            names = T00_COLUMNS[:]
        data_lines = lines[2:]
    else:  # time_gbt
        # line[0]=column names, line[1]=separator '====', line[2+]=data
        header_lines = [lines[0].rstrip()] if lines else []
        colnames_line = lines[0] if lines else ""
        names = _safe_columns(colnames_line.split())
        # Drop the COMMENTS column from the parsing target since most rows
        # have no comment text and the header has 6 names with often only
        # 5 numeric tokens per row.
        names = names[:5] if len(names) >= 5 else names
        data_lines = lines[2:]

    # Parse rows: keep MJD as float, parse numeric columns where possible,
    # keep text columns (dates / times) as strings.
    #
    # Performance note: the previous implementation used a Python double-
    # loop with per-token try/except float() calls, which dominates run
    # time on long time_gbt logs (tens of MB).  We now (a) filter and
    # tokenize once into a rectangular ``str`` ndarray and (b) convert
    # each column to float with a single vectorized call, so the hot
    # path lives almost entirely inside numpy C code.
    #
    # NaN preservation: numeric columns keep their NaN floats verbatim --
    # we never strip rows.  Veusz numeric datasets natively support NaN
    # (samples are skipped at plot time only).  Text columns also keep
    # every row (missing tokens recorded as "" by the rectangularization
    # step below).
    ncols = len(names)
    rows: List[List[str]] = []
    for ln in data_lines:
        s = ln.rstrip()
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        # Handle the embedded "==" separator (time_gbt) that contains '='
        if "====" in s:
            continue
        toks = s.split()
        # Don't run past the names list (extra COMMENTS tokens in time_gbt);
        # right-pad short rows with "" so the matrix stays rectangular.
        if len(toks) >= ncols:
            rows.append(toks[:ncols])
        else:
            rows.append(toks + [""] * (ncols - len(toks)))

    if rows:
        # str-dtype 2D array: shape (nrows, ncols), no copy beyond the
        # initial fromiter materialisation.
        tok_matrix = np.asarray(rows, dtype=str)
    else:
        tok_matrix = np.empty((0, ncols), dtype=str)

    def _to_float_vec(col_strs: np.ndarray) -> np.ndarray:
        """Vectorized str->float with NaN on parse failure.  Uses a single
        numpy.frompyfunc round-trip instead of a Python for-loop."""
        if col_strs.size == 0:
            return np.empty(0, dtype=float)
        def _f(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan
        # frompyfunc keeps the call dispatch in C; the per-element float()
        # is still Python but the loop overhead and exception machinery
        # cost less than the previous per-column .append + try/except.
        return np.frompyfunc(_f, 1, 1)(col_strs).astype(float)

    for i, n in enumerate(names):
        col_strs = tok_matrix[:, i] if tok_matrix.size else np.empty(0, dtype=str)
        arr_num = _to_float_vec(col_strs)
        finite_frac = np.isfinite(arr_num).mean() if arr_num.size else 0.0
        if finite_frac > 0.5:
            columns[n] = cache.store("%s.%s" % (base, n), arr_num)
        else:
            columns[n] = cache.store("%s.%s.txt" % (base, n), col_strs.copy())

    return {
        "flavour": flavour,
        "columns": columns,
        "sort_key": "MJD" if "MJD" in columns else None,
        "header_lines": header_lines,
        "base_name": base,
    }


# ============================================================================
# %% VEUSZ INTEGRATION
# ============================================================================
def push_franks_to_veusz(doc, data: Dict[str, Any], log_cb=None,
                         emit_datestr: bool = False,
                         column_cb=None,
                         plot_individual: bool = True,
                         gap_k: float = 10.0,
                         gap_absolute: float = 0.0,
                         datetime_duplicate: bool = False) -> None:
    """Push parsed Franks columns into the running embedded Veusz document.

    ``column_cb(done, total)`` -- optional, ticked once per source column
    successfully pushed (raw+sorted counted as one) so callers can drive a
    per-file column progress bar from the GUI thread.

    Parameters (v0.0.9)
    -------------------
    plot_individual : bool
        If False, the per-file plot page is **not** built (datasets are
        still pushed so the cross-file overlay post-pass can use them).
    gap_k : float
        K factor for ``detect_time_breaks`` (threshold = K * median(Δt)).
    gap_absolute : float
        Absolute gap in MJD units; when > 0 overrides ``gap_k``.
    """
    base = data["base_name"]
    cols = data["columns"]
    sort_key = data["sort_key"]

    raw_names: List[str] = []
    sorted_names: List[str] = []

    # Column-bar plan: one tick per source column plus (optionally) one for
    # the datestr emission step.
    _col_total = len(cols) + (1 if emit_datestr and sort_key is not None
                              and sort_key in cols else 0)
    _col_done = 0

    def _tick(n: int = 1) -> None:
        nonlocal _col_done
        _col_done += n
        if column_cb is not None:
            try:
                column_cb(_col_done, _col_total)
            except Exception:
                pass

    # Compute sort permutation once
    if sort_key is not None and sort_key in cols:
        # v0.0.10: optional GPU-accelerated argsort (CuPy) -- falls through
        # to NumPy when disabled, when CuPy is absent, or when the array
        # is too small to amortise the host<->device transfer.
        sort_idx = gpu_argsort(np.asarray(cols[sort_key]))
    else:
        sort_idx = None

    for cname, arr in cols.items():
        arr = np.asarray(arr)
        raw_name = "%s__%s__raw" % (base, safe_dsname(cname))
        sorted_name = "%s__%s__sorted" % (base, safe_dsname(cname))
        try:
            if arr.dtype.kind in ("U", "S", "O"):
                values = [v.decode("ascii", "replace") if isinstance(v, bytes) else str(v)
                          for v in arr]
                doc.SetDataText(raw_name, values)
                if sort_idx is not None:
                    doc.SetDataText(sorted_name, [values[i] for i in sort_idx])
                else:
                    doc.SetDataText(sorted_name, values)
            else:
                # NaN-preserving: Veusz handles NaN natively in numeric
                # datasets, so we push the float array through unchanged.
                fa = np.ascontiguousarray(arr, dtype=float)
                doc.SetData(raw_name, fa)
                if sort_idx is not None:
                    doc.SetData(sorted_name, np.ascontiguousarray(fa[sort_idx]))
                else:
                    doc.SetData(sorted_name, fa)
        except Exception as exc:
            if log_cb:
                log_cb("  SetData failed for %s: %s" % (cname, exc))
            continue
        raw_names.append(raw_name)
        sorted_names.append(sorted_name)
        _tick()

    # Tag everything
    try:
        doc.TagDatasets(base, raw_names + sorted_names)
        doc.TagDatasets("raw", raw_names)
        doc.TagDatasets("sorted", sorted_names)
    except Exception:
        pass

    # ---- v0.0.11: Optional datetime companion dataset --------------------
    # When the user asked for datetime-duplicate plots, build a single
    # Veusz date-time dataset from the file's MJD-like sort key.  The
    # dataset shares its sort permutation with the sorted y datasets, so
    # xy widgets line up row-for-row.
    datetime_x_name = None
    if datetime_duplicate and sort_key is not None and sort_key in cols:
        upper_sk = sort_key.upper()
        arr_mjd = None
        if upper_sk in MJD_LIKE_SORT_KEYS:
            try:
                arr_mjd = np.asarray(cols[sort_key], dtype=float)
            except Exception:
                arr_mjd = None
        elif upper_sk in JD_LIKE_SORT_KEYS:
            try:
                arr_mjd = (
                    np.asarray(cols[sort_key], dtype=float) - 2400000.5
                )
            except Exception:
                arr_mjd = None
        if arr_mjd is not None:
            if sort_idx is not None:
                try:
                    arr_mjd = arr_mjd[sort_idx]
                except Exception:
                    pass
            secs = mjd_to_veusz_seconds(arr_mjd)
            dt_name = "%s__%s__dt" % (base, safe_dsname(sort_key))
            if set_datetime_dataset(doc, dt_name, secs, log_cb=log_cb):
                datetime_x_name = dt_name
                try:
                    doc.TagDatasets(base, [dt_name])
                    doc.TagDatasets("sorted", [dt_name])
                except Exception:
                    pass
        elif log_cb:
            log_cb("  Datetime duplicate skipped: %s epoch unknown"
                   % sort_key)

    # ---- Optional MJD -> date-string text datasets -----------------------
    if emit_datestr and sort_key is not None and sort_key in cols:
        arr_mjd = np.asarray(cols[sort_key])
        if arr_mjd.dtype.kind in ("f", "i", "u"):
            try:
                date_strings = mjd_to_datestr(arr_mjd.astype(float))
                raw_ds = "%s__%s__datestr" % (base, safe_dsname(sort_key))
                sorted_ds = "%s__%s__datestr_sorted" % (base, safe_dsname(sort_key))
                doc.SetDataText(raw_ds, list(date_strings))
                if sort_idx is not None:
                    doc.SetDataText(sorted_ds,
                                    list(np.asarray(date_strings)[sort_idx]))
                else:
                    doc.SetDataText(sorted_ds, list(date_strings))
                try:
                    doc.TagDatasets(base, [raw_ds, sorted_ds])
                    doc.TagDatasets("datestr", [raw_ds, sorted_ds])
                except Exception:
                    pass
                if log_cb:
                    log_cb("  Date-string datasets created (2)")
                _tick()
            except Exception as exc:
                if log_cb:
                    log_cb("  MJD->date conversion failed: %s" % exc)

    # Build pages: one page named after the file, one graph per numeric col vs MJD
    # ``plot_individual`` (v0.0.9) lets the caller suppress per-file pages
    # while still pushing the underlying datasets (overlays still work).
    if not plot_individual:
        if log_cb:
            log_cb("  Per-file plot page skipped (combined plots only).")
        return

    page = doc.Root.Add("page", name=safe_dsname(base))
    try:
        page.notes.val = "\n".join(data.get("header_lines", []))
    except Exception:
        pass

    grid = page.Add("grid", columns=2)
    x_ds = "%s__%s__sorted" % (base, safe_dsname(sort_key)) if sort_key else None

    # Compute time-axis breaks once per file (all graphs on this page
    # share the same X axis dataset).
    break_pairs = []
    if sort_key is not None and sort_key in cols:
        try:
            x_arr = np.asarray(cols[sort_key], dtype=float)
            break_pairs = detect_time_breaks(
                x_arr, k_factor=gap_k, absolute_gap=gap_absolute
            )
        except Exception:
            break_pairs = []

    for cname in cols.keys():
        if cname == sort_key:
            continue
        arr = np.asarray(cols[cname])
        if arr.dtype.kind not in ("f", "i", "u"):
            # skip text columns in plotting
            continue
        ds = "%s__%s__sorted" % (base, safe_dsname(cname))
        graph = grid.Add("graph", name=safe_dsname("g_%s" % cname))
        # Install a broken X axis if time gaps were found; otherwise label
        # the default X axis as before.
        if break_pairs:
            try:
                make_broken_x_axis(graph, break_pairs,
                                   label=sort_key or "index",
                                   show_gridlines=True)
            except Exception:
                try:
                    graph.x.label.val = sort_key or "index"
                    graph.x.GridLines.hide.val = False
                except Exception:
                    pass
        else:
            try:
                graph.x.label.val = sort_key or "index"
                graph.x.GridLines.hide.val = False
            except Exception:
                pass
        try:
            graph.y.label.val = cname
            graph.y.GridLines.hide.val = False
        except Exception:
            pass
        xy = graph.Add("xy", name=safe_dsname("xy_%s" % cname))
        if x_ds is not None:
            xy.xData.val = x_ds
        xy.yData.val = ds
        try:
            xy.marker.val = "circle"
            xy.markerSize.val = "2pt"
            xy.PlotLine.hide.val = True
        except Exception:
            pass

    if log_cb:
        log_cb("  Page built with %d graphs" %
               sum(1 for c in cols if c != sort_key
                   and np.asarray(cols[c]).dtype.kind in ("f", "i", "u")))

    # ---- v0.0.11: datetime-duplicate per-file page -----------------------
    # When a datetime companion dataset was emitted above, clone the
    # just-built page with the same graphs but bind xy.xData to the
    # date-time dataset and style the x axis for YYYY-MM-DD HH:MM:SS
    # labels rotated 45 deg.  This page lands right after the original
    # page in the document tree.
    if datetime_x_name is not None:
        page_dt = doc.Root.Add(
            "page", name=safe_dsname("%s_dt" % base)
        )
        try:
            page_dt.notes.val = "\n".join(
                data.get("header_lines", [])
                + ["", "Datetime-axis duplicate (v0.0.11)."]
            )
        except Exception:
            pass
        grid_dt = page_dt.Add("grid", columns=2)
        for cname in cols.keys():
            if cname == sort_key:
                continue
            arr = np.asarray(cols[cname])
            if arr.dtype.kind not in ("f", "i", "u"):
                continue
            ds = "%s__%s__sorted" % (base, safe_dsname(cname))
            graph = grid_dt.Add(
                "graph", name=safe_dsname("g_%s_dt" % cname)
            )
            if break_pairs:
                try:
                    ax = make_broken_x_axis(
                        graph, break_pairs,
                        label=sort_key or "index",
                        show_gridlines=True,
                    )
                    style_datetime_x_axis(ax, label=sort_key or "index")
                except Exception:
                    style_datetime_x_axis(
                        graph.x, label=sort_key or "index"
                    )
            else:
                style_datetime_x_axis(graph.x, label=sort_key or "index")
            try:
                graph.y.label.val = cname
                graph.y.GridLines.hide.val = False
            except Exception:
                pass
            xy = graph.Add("xy", name=safe_dsname("xy_%s_dt" % cname))
            xy.xData.val = datetime_x_name
            xy.yData.val = ds
            try:
                xy.marker.val = "circle"
                xy.markerSize.val = "2pt"
                xy.PlotLine.hide.val = True
            except Exception:
                pass
        if log_cb:
            log_cb("  Datetime-duplicate page built (%s_dt)." % base)


# ============================================================================
# %% UNIT-OVERLAY POST-PASS (Franks groups by COLUMN NAME, no units)
# ============================================================================
_OVERLAY_COLORS = [
    "blue", "red", "green", "darkorange", "purple", "saddlebrown",
    "deeppink", "olive", "teal", "navy", "darkred", "darkgreen",
    "magenta", "black", "darkcyan", "goldenrod",
]


def build_unit_overlay_pages_franks(doc, file_records,
                                     gap_k=10.0, gap_absolute=0.0,
                                     log_cb=None,
                                     datetime_duplicate=False):
    """Build one overlay page per **shared column name** across the loaded
    batch.  Franks files carry no unit annotations, so all files that
    expose a column with the same name (e.g. ``DELTAT``) are overlaid on a
    single page.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        The active embedded Veusz document.
    file_records : list of dict
        One entry per successfully pushed file, each containing keys
        ``base``, ``columns`` (dict colname -> array), and ``sort_key``
        (column name, typically ``'MJD'``), produced by the GUI
        accumulator.
    gap_k, gap_absolute : float
        Forwarded to ``detect_time_breaks``.
    log_cb : callable or None
        Optional ``log_cb(msg)`` for status messages.
    """
    if not file_records:
        return
    # v0.0.12: Combined-in-time overlay semantics.
    # ---------------------------------------------------------------
    # Old behaviour (v0.0.9 -- v0.0.11): one xy trace per (file, column)
    # -- 12 DELTAT files = 12 DELTAT traces on the overlay page.
    # New behaviour (v0.0.12): for each column name, concatenate every
    # contributing file's x and y arrays into a single time-sorted
    # series and emit ONE xy trace per column.  Datasets are pushed
    # as ``OverlayCat__{col}__x__sorted`` / ``__y__sorted`` (and
    # ``__x__dt`` for the optional datetime page).
    # ---------------------------------------------------------------
    # group: col_name -> list of {base, x, y, mjd_or_none}
    by_col = {}
    x_samples_by_col = {}
    for rec in file_records:
        base = rec.get("base")
        cols = rec.get("columns") or {}
        sort_key = rec.get("sort_key")
        if not base or sort_key is None or sort_key not in cols:
            continue
        try:
            x_arr = np.asarray(cols[sort_key], dtype=float)
        except Exception:
            continue
        mjd_arr = None
        sk_upper = sort_key.upper()
        if datetime_duplicate and sk_upper in ("MJD", "DMJD", "JD"):
            try:
                if sk_upper == "JD":
                    mjd_arr = x_arr - 2400000.5
                else:
                    mjd_arr = x_arr.copy()
            except Exception:
                mjd_arr = None
        for cname, arr in cols.items():
            if cname == sort_key:
                continue
            arr = np.asarray(arr)
            if arr.dtype.kind not in ("f", "i", "u"):
                continue
            try:
                y_arr = np.asarray(arr, dtype=float)
            except Exception:
                continue
            if y_arr.shape != x_arr.shape:
                continue
            by_col.setdefault(cname, []).append({
                "base": base,
                "x": x_arr,
                "y": y_arr,
                "mjd": mjd_arr,
            })
            x_samples_by_col.setdefault(cname, []).append(x_arr)

    if not by_col:
        if log_cb:
            log_cb("  No numeric columns suitable for column-overlay pages.")
        return

    for cname, members in by_col.items():
        page_name = safe_dsname("Overlay_%s" % cname)
        page = doc.Root.Add("page", name=page_name)
        try:
            page.notes.val = (
                "Combined-in-time overlay of column '%s' across the "
                "loaded batch (v0.0.12: single time-sorted trace)."
                % cname
            )
        except Exception:
            pass
        graph = page.Add("graph", name="g_overlay")
        try:
            graph.y.label.val = cname
            graph.y.GridLines.hide.val = False
        except Exception:
            pass
        # Combined x across files for break detection.
        x_pieces = [a for a in x_samples_by_col.get(cname, []) if a.size]
        x_all = (np.concatenate(x_pieces)
                 if x_pieces else np.empty(0, dtype=float))
        break_pairs = detect_time_breaks(
            x_all, k_factor=gap_k, absolute_gap=gap_absolute
        )
        if break_pairs:
            make_broken_x_axis(graph, break_pairs,
                               label="MJD", show_gridlines=True)
        else:
            try:
                graph.x.label.val = "MJD"
                graph.x.GridLines.hide.val = False
            except Exception:
                pass
        try:
            key = graph.Add("key", name="key1")
            key.Border.hide.val = False
        except Exception:
            pass

        # Concatenate + time-sort this column across all files.
        try:
            xs = np.concatenate([m["x"] for m in members])
            ys = np.concatenate([m["y"] for m in members])
        except Exception:
            continue
        if xs.size == 0:
            continue
        # v0.0.12: stable sort by time; uses GPU when the array is large
        # enough (gpu_argsort falls back to numpy.argsort otherwise).
        try:
            order = gpu_argsort(xs)
        except Exception:
            try:
                order = np.argsort(xs, kind="mergesort")
            except Exception:
                order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = ys[order]
        c_safe = safe_dsname(cname)
        x_name = "OverlayCat__%s__x__sorted" % c_safe
        y_name = "OverlayCat__%s__y__sorted" % c_safe
        try:
            doc.SetData(x_name, xs_s)
            doc.SetData(y_name, ys_s)
        except Exception as _exc:
            if log_cb:
                log_cb("  Overlay '%s' SetData failed: %s" % (cname, _exc))
            continue
        xy = graph.Add("xy", name=safe_dsname("xy_%s" % cname))
        xy.xData.val = x_name
        xy.yData.val = y_name
        try:
            xy.key.val = cname
        except Exception:
            pass
        try:
            xy.marker.val = "circle"
            xy.markerSize.val = "2pt"
            xy.PlotLine.hide.val = True
            colour = _OVERLAY_COLORS[0]
            xy.MarkerFill.color.val = colour
            xy.MarkerLine.color.val = colour
        except Exception:
            pass
        if log_cb:
            log_cb("  Overlay page '%s': 1 trace (time-combined across "
                   "%d files)." % (page_name, len(members)))

        # Datetime companion: only when EVERY file has an MJD array.
        dt_name = None
        if datetime_duplicate and all(
            m.get("mjd") is not None for m in members
        ):
            try:
                mjds = np.concatenate(
                    [np.asarray(m["mjd"], dtype=float) for m in members]
                )
                mjds_s = mjds[order]
                dt_secs = mjd_to_veusz_seconds(mjds_s)
                dt_name = "OverlayCat__%s__x__dt" % c_safe
                if not set_datetime_dataset(
                    doc, dt_name, dt_secs, log_cb=log_cb
                ):
                    dt_name = None
            except Exception as _exc:
                if log_cb:
                    log_cb("  Overlay '%s' datetime build failed: %s"
                           % (cname, _exc))
                dt_name = None

        # ---- v0.0.11/0.0.12: datetime-duplicate overlay page ----------
        if datetime_duplicate and dt_name:
            page_dt = doc.Root.Add(
                "page", name=safe_dsname("%s_dt" % page_name)
            )
            try:
                page_dt.notes.val = (
                    "Datetime-axis duplicate of '%s' "
                    "(v0.0.12 combined-in-time)."
                    % page_name
                )
            except Exception:
                pass
            graph_dt = page_dt.Add("graph", name="g_overlay_dt")
            try:
                graph_dt.y.label.val = cname
                graph_dt.y.GridLines.hide.val = False
            except Exception:
                pass
            if break_pairs:
                try:
                    ax = make_broken_x_axis(
                        graph_dt, break_pairs,
                        label="MJD", show_gridlines=True,
                    )
                    style_datetime_x_axis(ax, label="MJD")
                except Exception:
                    style_datetime_x_axis(graph_dt.x, label="MJD")
            else:
                style_datetime_x_axis(graph_dt.x, label="MJD")
            try:
                key_dt = graph_dt.Add("key", name="key1_dt")
                key_dt.Border.hide.val = False
            except Exception:
                pass
            xy_dt = graph_dt.Add(
                "xy", name=safe_dsname("xy_%s_dt" % cname)
            )
            xy_dt.xData.val = dt_name
            xy_dt.yData.val = y_name
            try:
                xy_dt.key.val = cname
            except Exception:
                pass
            try:
                xy_dt.marker.val = "circle"
                xy_dt.markerSize.val = "2pt"
                xy_dt.PlotLine.hide.val = True
                colour = _OVERLAY_COLORS[0]
                xy_dt.MarkerFill.color.val = colour
                xy_dt.MarkerLine.color.val = colour
            except Exception:
                pass
            if log_cb:
                log_cb("  Datetime-overlay page '%s_dt': 1 trace."
                       % page_name)


# ============================================================================
# %% WORKER THREAD
# ============================================================================
class FranksBatchWorker(QThread):
    """Background QThread that parses Franks files in parallel."""

    progress = Signal(int, int, str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, files: List[str], max_threads: int,
                 cache: MemoryAwareCache, parent=None) -> None:
        super().__init__(parent)
        self.files = files
        self.max_threads = max_threads
        self.cache = cache

    def run(self) -> None:
        try:
            work = [(p, parse_franks_file, (p, self.cache)) for p in self.files]
            results = run_in_threadpool(
                work, max_workers=self.max_threads,
                progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit("%s\n%s" % (exc, traceback.format_exc()))


# ============================================================================
# %% MAIN WINDOW
# ============================================================================
class FranksAutoPlotWindow(AutoPlotMainWindow):
    """Top-level window for the Franks AutoPlot script."""

    def __init__(self) -> None:
        super().__init__("FranksProcessed AutoPlot - Veusz Embedded GUI",
                         default_mode="dark")
        self._file_filter = (
            "Franks files (*.t00new *.dat *.bak* *.OrigBack* *.back);;All files (*)"
        )
        self.cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=DEFAULT_RSS_HIGH_WATER_MB
        ))
        self.monitor = MemoryMonitor(
            self.cache,
            callback=lambda rss: self.log(
                "RSS crossed high-water mark (%.1f MiB) -- spilling to disk." % rss
            )
        )
        self.monitor.start()
        self.veusz_doc = None
        self.worker = None
        self.log("MAX_THREADS = %d" % MAX_THREADS)

    # ----- options form ----------------------------------------------------
    def _populate_options(self, form: QFormLayout) -> None:
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, max(1, (os.cpu_count() or 4) * 2))
        self.thread_spin.setValue(MAX_THREADS)
        form.addRow(QLabel("Worker threads:"), self.thread_spin)

        self.rss_spin = QSpinBox()
        self.rss_spin.setRange(128, 65536)
        self.rss_spin.setSingleStep(128)
        self.rss_spin.setValue(DEFAULT_RSS_HIGH_WATER_MB)
        form.addRow(QLabel("RSS spill threshold (MiB):"), self.rss_spin)

        self.show_embed_cb = QCheckBox("Show embedded Veusz preview while building")
        self.show_embed_cb.setChecked(True)
        form.addRow(self.show_embed_cb)

        self.datestr_cb = QCheckBox(
            "Generate MJD -> date strings (YYYY-MM-DD_HH:MM:SS) datasets"
        )
        self.datestr_cb.setChecked(False)
        form.addRow(self.datestr_cb)

        # --- Broken-axis / overlay controls (v0.0.9) ----------------------
        self.gap_k_spin = QDoubleSpinBox()
        self.gap_k_spin.setRange(1.0, 1000.0)
        self.gap_k_spin.setSingleStep(1.0)
        self.gap_k_spin.setDecimals(2)
        self.gap_k_spin.setValue(10.0)
        form.addRow(QLabel("Gap K (× median Δt):"), self.gap_k_spin)

        self.gap_abs_spin = QDoubleSpinBox()
        self.gap_abs_spin.setRange(0.0, 1e12)
        self.gap_abs_spin.setDecimals(6)
        self.gap_abs_spin.setSingleStep(1.0)
        self.gap_abs_spin.setValue(0.0)
        form.addRow(QLabel("Absolute gap (MJD units; 0=auto):"),
                    self.gap_abs_spin)

        self.combined_only_cb = QCheckBox(
            "Combined (overlay) plots only -- skip per-file pages"
        )
        self.combined_only_cb.setChecked(False)
        form.addRow(self.combined_only_cb)

        # --- Datetime-duplicate plots (v0.0.11) ---------------------------
        self.datetime_dup_cb = QCheckBox(
            "Duplicate plots with datetime X axis (YYYY-MM-DD HH:MM:SS)"
        )
        self.datetime_dup_cb.setChecked(False)
        form.addRow(self.datetime_dup_cb)

        # --- GPU acceleration (CuPy, optional) (v0.0.10) ------------------
        self.gpu_cb = QCheckBox("Use GPU acceleration (CuPy) for large sorts")
        self.gpu_cb.setChecked(False)
        _gpu_ok = is_gpu_available()
        self.gpu_cb.setEnabled(_gpu_ok)
        self.gpu_cb.setToolTip(gpu_backend_name())
        form.addRow(self.gpu_cb)
        self.log("GPU backend: %s" % gpu_backend_name())

    # ----- run -------------------------------------------------------------
    def _process_files(self) -> None:
        if not self.selected_files:
            QMessageBox.information(self, "No files",
                                    "Select one or more Franks data files first.")
            return
        self.cache.cfg.rss_high_water_mb = int(self.rss_spin.value())

        if self.veusz_doc is None and self.show_embed_cb.isChecked():
            self.veusz_doc = open_embedded("Franks_AutoPlot")
        elif self.veusz_doc is None:
            self.veusz_doc = vz_embed.Embedded("Franks_AutoPlot", hidden=True)

        self.process_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.show_progress_bars(len(self.selected_files))

        self.worker = FranksBatchWorker(
            self.selected_files, int(self.thread_spin.value()),
            self.cache, parent=self
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.log("Starting batch: %d file(s), threads=%d" %
                 (len(self.selected_files), int(self.thread_spin.value())))
        self.worker.start()

    def _on_progress(self, done: int, total: int, key: str) -> None:
        self.progress_bar.setValue(done)
        self.log("  parsed [%d/%d] %s" % (done, total, os.path.basename(key)))
        self.update_mem_label(self.cache.rss_mb())

    def _on_failed(self, msg: str) -> None:
        self.log("Batch failed: %s" % msg)
        QMessageBox.critical(self, "Batch failed", msg)
        self.process_button.setEnabled(True)
        self.hide_progress_bars()

    def _on_done(self, results: Dict[str, Any]) -> None:
        emit_datestr = bool(self.datestr_cb.isChecked())
        # v0.0.9 broken-axis / overlay knobs
        gap_k = float(self.gap_k_spin.value())
        gap_absolute = float(self.gap_abs_spin.value())
        plot_individual = not bool(self.combined_only_cb.isChecked())
        # v0.0.11: datetime-duplicate toggle
        datetime_duplicate = bool(self.datetime_dup_cb.isChecked())
        # v0.0.10: drive the process-wide GPU flag from the checkbox
        enable_gpu(self.gpu_cb.isChecked() and self.gpu_cb.isEnabled())
        # Keep the GUI responsive across many files: pump the Qt event
        # loop between every push so the log pane scrolls live and the
        # window doesn't appear "stuck" during a long batch insert.
        app = QApplication.instance()
        total = len(results)
        def _col_cb(done: int, total_ops: int) -> None:
            if self.column_progress_bar.maximum() != max(1, total_ops):
                self.column_progress_bar.setRange(0, max(1, total_ops))
            self.column_progress_bar.setValue(done)
            if app is not None:
                app.processEvents()
        file_records = []  # accumulator for the overlay post-pass
        for idx, (path, data) in enumerate(results.items(), start=1):
            if isinstance(data, Exception):
                self.log("  ERROR processing %s: %s" % (path, data))
                self.parse_progress_bar.setValue(idx)
                if app is not None:
                    app.processEvents()
                continue
            self.log("Inserting datasets [%d/%d] %s"
                     % (idx, total, os.path.basename(path)))
            n_cols = len(data.get("columns") or {})
            self.begin_column_progress(os.path.basename(path), max(1, n_cols))
            try:
                push_franks_to_veusz(self.veusz_doc, data, log_cb=self.log,
                                     emit_datestr=emit_datestr,
                                     column_cb=_col_cb,
                                     plot_individual=plot_individual,
                                     gap_k=gap_k,
                                     gap_absolute=gap_absolute,
                                     datetime_duplicate=datetime_duplicate)
            except Exception as exc:
                self.log("  push failed: %s" % exc)
            else:
                file_records.append({
                    "base": data.get("base_name") or
                            safe_dsname(os.path.basename(path)),
                    "columns": data.get("columns") or {},
                    "sort_key": data.get("sort_key"),
                })
            self.parse_progress_bar.setValue(idx)
            if app is not None:
                app.processEvents()
        # Build cross-file column-name overlay pages.
        if file_records:
            try:
                build_unit_overlay_pages_franks(
                    self.veusz_doc, file_records,
                    gap_k=gap_k, gap_absolute=gap_absolute,
                    log_cb=self.log,
                    datetime_duplicate=datetime_duplicate,
                )
            except Exception as exc:
                self.log("  build_unit_overlay_pages_franks failed: %s"
                         % exc)
        self.log("Batch complete.")
        self.hide_progress_bars()
        self.process_button.setEnabled(True)
        self.save_button.setEnabled(True)

    # ----- save ------------------------------------------------------------
    def _save_project(self) -> None:
        if self.veusz_doc is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Process some files first.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Veusz Project", "", "Veusz HDF5 Projects (*.vszh5)"
        )
        if not fn:
            return
        try:
            written = save_vszh5(self.veusz_doc, fn)
            self.log("Saved %s" % written)
            self.mark_project_saved(written)
            QMessageBox.information(self, "Saved", "Wrote %s" % written)
        except Exception as exc:
            self.log("Save failed: %s" % exc)
            QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event) -> None:
        try:
            self.monitor.stop()
        except Exception:
            pass
        try:
            if self.veusz_doc is not None:
                self.veusz_doc.Close()
        except Exception:
            pass
        try:
            self.cache.cleanup()
        except Exception:
            pass
        super().closeEvent(event)


# ============================================================================
# %% ENTRY POINT
# ============================================================================
def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = FranksAutoPlotWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
