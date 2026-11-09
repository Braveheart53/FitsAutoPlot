# -*- coding: utf-8 -*-
"""
=============================================================================
Franks_AutoPlot.py
-----------------------------------------------------------------------------
# %% Header Info
--------

GUI-driven batch processor for the "FranksProcessed" ASCII time-series data
files distributed alongside the NRAO 1PPS sync FITS bundle.  Two file
flavours are recognised automatically:

    * ``YYYY_MM.t00new`` - GB Timing Data (medians over 60 min), columns:
          MJD  date  time  Site1Hz  TAC  MicroSem  CNS_GPS1  MSM-GPS
          CNS_GPS  CNS2018  GBT_VLBA  RA  GBTRtn

    * ``time_gbt.dat``   - long-term GBT timing log, columns:
          MJD  EECO-REF  NIST-REF  NS  DATE  [COMMENTS]

Both flavours are read as numpy column arrays, stored in a native project
model as one dataset per column tagged with the file's base name (plus
``raw`` and ``sorted`` tags), and plotted column-vs-MJD on dedicated pages.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.4.0: Small/large gap controls; local ISO 8601 on every page.
Date: 2026-11-09
#              * Gaps (GUI, shared with FITS AutoPlot 0.4.0): small gap - the
#                line breaks where dt > K x median dt (or a fixed time);
#                large gap - the x axis breaks where dt > n x the small gap
#                (default n = 100) or a fixed time.
#              * Every page draws its MJD axis as a Veusz date-time axis in
#                local ISO 8601; the duplicate _dt/_dt_labels pages are now
#                "Legacy" (off by default).
#              * Pages in file-name order, not thread completion order.
#              * The memory monitor calls a bound method (_rss_crossed, held
#                weakly) instead of a lambda that kept the window alive from
#                the monitor thread (_autoplot_common 0.4.0).
#              * Docstrings for every function and method; author contact
#                lines in the header.
#              * Tooltips (pop-up help) on every button, box and field that
#                had none (threads, RSS threshold, date strings, overlay-only).
# %%%%% Function Descriptions
        _detect_flavour: Return 't00new' or 'time_gbt' for a Franks data file.
        _safe_columns: Turn header names into Veusz-safe dataset names.
        parse_franks_file: Read one Franks-style ASCII file into a dict of column arrays.
        push_franks_to_veusz: Add parsed Franks columns to a native in-memory project model.
        build_unit_overlay_pages_franks: Build one overlay page per **shared column name** across the loaded batch.
        FranksBatchWorker (class): Background QThread that parses Franks files in parallel.
        FranksAutoPlotWindow (class): Top-level window for the Franks AutoPlot script.
        main: Run the Franks AutoPlot window; returns the Qt exit code.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP_NAME = 'Franks AutoPlot'
        APP_VERSION = "0.4.0"
        EMIT_TEXT_X_DATASETS: 0.3.0: per-point date TEXT datasets (the v0.0.15 text-x variant) are no : longer written; the dt_labels pages use Veusz date-time datasets on a : l...
        _THIS_DIR: folder of this file
        MJD_LIKE_SORT_KEYS: v0.0.11: Franks files always use MJD as their sort key, so the upper-cased sort-key column name is *always* in this set.  We keep the set explicit...
        JD_LIKE_SORT_KEYS: other sort-key names treated like MJD
        MAX_THREADS: I/O-bound default: ASCII reads + np.genfromtxt both release the GIL on the heavy paths, so we get a real speedup from oversubscribing the CPU.
        DEFAULT_RSS_HIGH_WATER_MB = 1024
        T00_COLUMNS: column names of a .t00new file (used when its header is malformed)
        TIMEDAT_COLUMNS: column names of a time_gbt.dat file
        _OVERLAY_COLORS: overlay trace colours
# %%%%% More Info
    Run:  python Franks_AutoPlot.py   (GUI; View menu: light/dark)
    Franks timing files (.t00new, time_gbt.dat and backups) are ASCII with
    mixed date and numeric columns, so they are parsed with numpy (not a Veusz
    importer); raw and MJD-sorted datasets are both kept and tagged.  Time axes
    and gaps as in FITS AutoPlot.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.3.0: Date-times as Veusz date-time datasets, local ISO 8601 axes.
Date: 2026-11-09
#              * The time stamps of every dt_labels page (per file, per HDU /
#                tag and the unit overlays) are stored as Veusz DATE
#                datasets in UTC (were plain numeric seconds) plus a
#                "<name>_local" date dataset on the clock of the computer
#                doing the processing (DST-aware; VSZH5_DISPLAY_TZ
#                overrides).  The pages draw the local one; the axis ticks
#                read 2026-11-09T08:39:00-05:00 (ISO 8601, 24 h, with the
#                UTC offset) and the axis label names the offset.  Broken
#                axes use the same local clock.
#              * No per-point date text: the text-x datasets are no longer
#                written (EMIT_TEXT_X_DATASETS = True restores them) and the
#                older labelled _dt page is now off by default ("Legacy:
#                per-point date text labels page"); its label datasets are
#                only built when that page is requested.
#              * Page names and notes are unchanged (file headers stay in
#                the page notes).
#              * Writer 0.0.25 / _autoplot_native 0.3.0 / _autoplot_common
#                0.3.0.
#              * FIX: the overlay dt_labels break detection called
#                detect_time_breaks_unit_aware without importing it (a
#                NameError swallowed by its try block, so overlay date pages
#                never got a broken axis).
# %%%% 0.2.1: Writer 0.0.24 and the suite-wide theme.
Date: 2026-11-05
#              No code change in this file: the .vszh5 is now written by
#              vszh5_writer 0.0.24 (through _autoplot_native 0.2.0, atomic
#              save) and the window opens in the light / dark theme shared
#              with the launcher (_autoplot_common 0.3.0).
# %%%% 0.2.0: Interactive pyqtgraph Plots tab (AB monitor style), About.
Date: 2026-11-05
#              After a batch the new "Plots" tab shows every page of the
#              in-memory project with pyqtgraph (_autoplot_pyqtgraph 0.1.0,
#              modelled on AB Power Meter Monitor 1.7.0): linked x axes,
#              shared crosshair, legend column with values at the cursor,
#              Ctrl+wheel zoom, CSV/PNG/SVG export, MJD as UTC dates, gap
#              breaks, one page in memory at a time.  Saving the .vszh5
#              project and Open in Veusz are unchanged.
#
# %%%% 0.1.0: Native h5py project generation, no Veusz runtime dependency.
Date: 2026-10-11
#              Uses _autoplot_native.NativeDocument and shared atomic saving.
#              Retains data, tag, gap, overlay, NaN and datetime semantics.
#              External Veusz viewing is optional and only after saving.
#
# %%%% 0.0.19: Wire string gap parser + hide-keys checkbox through GUI.
Date: 2026-05-16
#              Same GUI additions as FITS_AutoPlot 0.0.19:
#                * ``self.gap_abs_str`` QLineEdit overrides the hours
#                  spinbox when non-empty (parsed via
#                  ``parse_gap_string`` from ``_autoplot_common``).
#                * ``self.hide_keys_cb`` QCheckBox plumbed via new
#                  ``hide_keys`` kwarg through ``push_franks_to_veusz``
#                  and ``build_unit_overlay_pages_franks``; every
#                  ``key/key_dt/key_dtL.Border.hide.val`` is now
#                  ``bool(hide_keys)``.
#              Default behaviour unchanged when both controls are left
#              at their defaults (blank string + unchecked).
#
# %%%% 0.0.18: Unit-aware time-break detection (manual gap units fix).
Date: 2026-05-16
#              Mirrors the FITS_AutoPlot v0.0.18 fix.  All break-
#              detection call sites in Franks_AutoPlot now route
#              through ``detect_time_breaks_unit_aware(x, col_name,
#              k_factor=gap_k, absolute_gap_days=gap_absolute)`` from
#              _autoplot_common.  Franks always uses ``MJD`` as the
#              sort key (day-scale, factor = 1.0) so behavior is
#              unchanged in practice; the call sites stay in sync
#              with FITS_AutoPlot for future-proofing against
#              column-name changes.  New imports:
#              ``detect_time_breaks_unit_aware``,
#              ``column_unit_factor_from_day``.  Revision history
#              kept in DESCENDING semantic-version order.
#
# %%%% 0.0.17: Minimized .vszh5 save (GUI checkboxes).
Date: 2026-05-16
#              Two new checkboxes on the Franks GUI:
#                * "Minimized Veusz File" -- saved .vszh5 contains ONLY
#                  the datasets directly referenced by widgets in the
#                  document.  Backed by ``save_vszh5_minimized`` in
#                  _autoplot_common (walks Root.WalkWidgets(), evicts
#                  unreferenced datasets, saves, restores in-memory).
#                * "Generate Full Veusz file" -- sub-checkbox, only
#                  enabled when the parent is checked.  When BOTH are
#                  checked, save writes the minimized file AND a
#                  parallel ``<name>_full.vszh5`` (legacy behaviour).
#              The FITS-side v0.0.17 sentinel channel-tag filter does
#              NOT apply to Franks because the Franks overlay is one
#              trace per page (no per-row channel tagging).
#              Revision history kept in DESCENDING semantic-version
#              order.
#
# %%%% 0.0.16: dt_labels page upgraded to mode='datetime' (true date ticks).
Date: 2026-05-16
#              * Mirror of FITS_AutoPlot v0.0.16 for Franks records.
#                The v0.0.15 dt_labels page used a text-x dataset bound
#                to xData with axis ``mode='labels'``.  Functional, but
#                the axis was uniform-sample-spaced (no proper date
#                tick cadence) and there was no broken-axis support.
#              * v0.0.16 emits a NEW numeric Veusz-datetime-seconds
#                dataset alongside the text-x dataset:
#                ``..__datelabels_dtnum``.  Computed from MJD by
#                ``mjd_to_veusz_seconds`` (NaN-preserving, vectorised).
#                The dt_labels page now binds xData to this numeric
#                dataset and sets the x axis to ``mode='datetime'``
#                via ``configure_axis_datetime_mode``.  Veusz draws
#                proper date ticks (DateTicks on Axis / AxisBroken).
#              * Broken-axis parity: when the seconds-axis dt page
#                has a broken x-axis, the dt_labels page now gets one
#                too.  MJD ``break_pairs`` are mapped to Veusz seconds
#                by ``mjd_break_pairs_to_dtsec`` and installed via
#                ``make_broken_x_axis`` BEFORE configure_datetime_mode
#                (AxisBroken subclasses Axis so it inherits the mode).
#              * v0.0.15 text-x dataset (``..__datelabels_textx``) is
#                still emitted as a back-compat shim.  Per-file and
#                overlay dt_labels pages prefer dtnum and fall back to
#                textx only when dtnum failed to build.
#              * Plugin fields/GUI unchanged in 0.0.16; the plugin
#                header is bumped only to keep versions in lock-step.
#              * GUI: "Absolute gap (MJD units; 0=auto)" spinbox is
#                renamed "Manual gap (hours; 0=auto)" and now expects
#                HOURS as input.  Widget value is divided by 24 before
#                being passed to detect_time_breaks so MJD-axis time
#                gaps are entered in a familiar unit.  Same change is
#                mirrored in the plugin dialog.
#
# %%%% 0.0.15: Density-pct date labels + text-x dt_labels page variant.
Date: 2026-05-16
#              * v0.0.14's "sparse vs full" boolean is generalised to an
#                integer percentage 0..100 (``datetime_label_density_pct``).
#                0 = no labels, 100 = one per finite point, 10 (default)
#                approximates the old sparse behaviour.  Legacy
#                ``datetime_full_labels`` is still accepted as a back-
#                compat shim (True -> 100, False -> 10).
#              * NEW dt page variant: ``<base>_dt_labels`` and the
#                overlay counterpart ``Overlay_<col>_dt_labels``.
#                xData is a per-point TEXT dataset and the x axis is
#                set to ``mode='labels'`` -- Veusz renders the strings
#                on the axis itself.  Sample spacing is uniform (one
#                tick per point), NOT proportional to elapsed time.
#              * GUI: v0.0.14 "Use full per-point date labels" checkbox
#                is replaced by a 0..100 percentage spinbox plus two
#                checkboxes (numeric-x dt page; text-x dt_labels page).
#                Same controls land in the plugin dialog FieldInt +
#                FieldBool fields.
#              * GPU + parallelization re-audit: the new text-x page
#                reuses the same GPU-sorted permutation as the numeric-x
#                page (gpu_argsort already applied at the single sort
#                site per record).  No new sort hotspots; mjd_to_datestr
#                fast path (v0.0.14) remains vectorised for the
#                canonical format.  Worker-thread parallelization of
#                file reads is untouched.
#              * Revision history blocks kept in DESCENDING semantic-
#                version order across every shared file.
#
# %%%% 0.0.14: Datetime via xy.labels per-point text labels.
Date: 2026-05-16
#                * The v0.0.11..0.0.13 dt-duplicate pages used a Veusz
#                  date-time x axis driven by SetDataDateTime.  On Veusz
#                  3.4 (current NRAO production) that path fires hundreds
#                  of internal exceptions inside Veusz's date code.  The
#                  v0.0.14 dt-duplicate pages keep the SAME numeric
#                  seconds x dataset as the seconds page, and instead
#                  bind a TEXT dataset to ``xy.labels.val`` so every
#                  data point carries its own "YYYY-MM-DD HH:MM:SS"
#                  label rendered in place.  No SetDataDateTime calls.
#                * Two label datasets are built per trace:
#                    - sparse (~10 evenly spaced anchors; the rest are
#                      empty strings so they render as nothing)
#                    - full   (one label per data point)
#                  Both are always pushed; the new GUI checkbox
#                  "Use full per-point date labels" toggles which one
#                  is bound by default.
#                * Per-trace style sub-group ``xy.Label`` is set to
#                  angle=45deg, size=6pt, posnVert=centre, posnHorz=left
#                  by style_xy_datetime_labels().
#                * dt-page x axis stays NUMERIC -- style_datetime_x_axis
#                  is no longer called on the dt-page x axis.

# %%%% 0.0.13: Identity-stable trace styling + datetime-duplicate hardening.
Date: 2026-05-16
#                * Every xy widget (per-file, datetime-duplicate, and
#                  cross-file column-overlay) is now styled by
#                  apply_trace_style() with identity key
#                  (column_name, (None,)).  The same column keeps the
#                  SAME colour and line style on every page it appears on
#                  -- the seconds page, the datetime-duplicate page, and
#                  the cross-file overlay -- so it is easy to follow a
#                  single column (e.g. DELTAT) through the document.
#                * Line is shown by default with a 1pt width; markers
#                  shrink to 1pt.  vary_style is False everywhere because
#                  every Franks graph carries exactly one trace.
#                * SetDataDateTime calls now go through
#                  set_datetime_dataset(), which coerces every value to a
#                  plain Python float (NaN/inf -> 0.0) and logs the
#                  outcome.  Plugged in at the single-file dt site and
#                  the overlay dt site.

# %%%% 0.0.12: Combined-in-time overlay semantics.
Date: 2026-05-16
#              build_unit_overlay_pages_franks now collapses each shared
#              column onto a single time-sorted xy trace stitched across
#              every file (legend names the column, e.g. 'DELTAT').  The
#              datetime-axis duplicate page mirrors the concatenation
#              with the Veusz-seconds version of the same x.

# %%%% 0.0.11: Duplicate plots with a date-time x axis.  A new GUI
Date: 2026-05-16
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

# %%%% 0.0.10: Optional GPU acceleration for the per-file argsort step.
Date: 2026-05-16
#              When the 'Use GPU acceleration (CuPy)' checkbox is
#              checked and CuPy is importable on the host, the large
#              MJD-sort permutation is dispatched to ``gpu_argsort``
#              from ``_autoplot_common`` -- typically 2.7-10x faster
#              than NumPy on consumer GPUs once N exceeds ~200k
#              samples.  Smaller files still use NumPy.  CuPy is a soft
#              dependency: the checkbox is disabled and tooltipped when
#              CuPy is absent.

# %%%% 0.0.9: Per-plot broken-x-axis on time gaps + column-name overlay
Date: 2026-05-16
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

# %%%% 0.0.8: Parallelization audit + Open-in-Veusz button.
Date: 2026-05-16

# %%%% 0.0.7: Added two new progress bars in the GUI -- a 'Parsing/pushing'
Date: 2026-05-16
#             file-level bar that ticks once per file as push_franks_to_veusz()
#             finishes that file, and a 'Current file - columns' bar that
#             ticks per source column.  push_franks_to_veusz() gained a
#             ``column_cb(done, total)`` parameter.  No skip-images knob
#             here (Franks files have no image HDUs).

# %%%% 0.0.6: Added Spyder IDE cell markers (# %% / # %%%) at all major
Date: 2026-05-16
#             section banners and import subsections so the file can be
#             navigated and run cell-by-cell in Spyder's Outline view.
#             Cosmetic only -- no behavior change.

# %%%% 0.0.5: _on_done() now calls QApplication.processEvents() between
Date: 2026-05-16
#             every per-file push so the GUI stays responsive (log pane
#             scrolls live, buttons remain clickable) during a long
#             batch insert -- previously the window could appear stuck
#             when processing many .t00new files in one batch.
#             (No 0.0.4: aligned with FITS_AutoPlot.py version stream.)

# %%%% 0.0.3: NaN-preserving emission policy.  When a Franks token is
Date: 2026-05-16
#             missing or non-numeric the parser now records ``NaN`` in the
#             numeric column (length-preserving) AND keeps the text token
#             ("" for fully missing) in the text companion -- rows are
#             never dropped.  Veusz numeric datasets carry the NaN through;
#             non-finite MJDs in the optional date-string text datasets
#             become the sentinel string ``"NaN"`` so all per-row arrays
#             retain the same length and stay index-aligned.

# %%%% 0.0.2: Added "Generate MJD->date strings" checkbox.  Whenever the
Date: 2026-05-16
#             parsed file contains an MJD column, ticking the option emits
#             two additional Veusz text datasets (raw + sorted) of
#             YYYY-MM-DD_HH:MM:SS strings tagged with the file's base name
#             plus ``datestr``.

# %%%% 0.0.1: Initial implementation, FranksProcessed file support
Date: 2026-05-16

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
# %%% IMPORTS - Native project model
# ============================================================================
from _autoplot_native import NativeDocument

APP_NAME = 'Franks AutoPlot'
APP_VERSION = "0.4.0"
#: 0.3.0: per-point date TEXT datasets (the v0.0.15 text-x variant) are no
#: longer written; the dt_labels pages use Veusz date-time datasets on a
#: local ISO-8601 axis.  True writes them again (back-compat only).
EMIT_TEXT_X_DATASETS = False
__version__ = APP_VERSION

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
    detect_time_breaks, detect_time_breaks_unit_aware, make_broken_x_axis,
    gpu_argsort, is_gpu_available, enable_gpu, gpu_backend_name,
    # v0.0.11: datetime-duplicate helpers (legacy SetDataDateTime path,
    # retained only for backward-compatibility -- v0.0.14 no longer calls
    # them on dt-pages).
    mjd_to_veusz_seconds, style_datetime_x_axis,
    set_datetime_dataset,
    # v0.0.13: identity-stable trace styling
    apply_trace_style, TRACE_STYLE_VARY_THRESHOLD,
    MJD_VEUSZ_EPOCH_MJD,
    DEFAULT_DATETIME_TICK_FORMAT, DEFAULT_DATETIME_TICK_ROTATE_DEG,
    DEFAULT_DATETIME_MAJOR_TICKS_TARGET,
    # v0.0.14: per-point datetime labels via xy.labels.val
    build_sparse_datestr_dataset, build_full_datestr_dataset,
    build_density_datestr_dataset, build_textx_dataset,
    configure_axis_labels_mode,
    style_xy_datetime_labels,
    # v0.0.16: dt_labels via mode='datetime' + broken-axis parity
    build_dtnum_dataset, configure_axis_datetime_mode,
    mjd_break_pairs_to_dtsec,
    # v0.0.17: minimized .vszh5 save (Franks overlay is single-trace per
    # page so the sentinel channel-tag filter from FITS does not apply).
    save_vszh5_minimized,
    # v0.0.19: string gap parser ('5m3d2h' style)
    parse_gap_string,
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
    """Return 't00new' or 'time_gbt' for a Franks data file.

    The file name decides first (``*.t00new`` or containing "time_gbt");
    otherwise a first line containing "GB Timing Data" means t00new, and
    anything else is read as time_gbt.
    """
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
    """Turn header names into Veusz-safe dataset names.

    Characters other than letters, digits and '_' become '_'; an empty name
    becomes ``col<n>`` and a name starting with a digit gets a ``c_`` prefix.
    """
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
            """float(x), or NaN when x is not a number."""
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
                         datetime_duplicate: bool = False,
                         datetime_full_labels=None,
                         datetime_label_density_pct: int = 10,
                         datetime_emit_numeric_dt: bool = False,
                         datetime_emit_text_dt: bool = True,
                         hide_keys: bool = False) -> None:
    """Add parsed Franks columns to a native in-memory project model.

    v0.0.19: ``hide_keys`` -- when True every key (legend) widget added
    to the per-file page is created with its Border hidden.

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
    datetime_label_density_pct : int
        v0.0.15. Percentage of finite points labelled on the numeric-x
        dt page (0..100). Supersedes the v0.0.14 boolean.
    datetime_emit_numeric_dt : bool
        v0.0.15. Emit the v0.0.14-lineage numeric-x dt page.
    datetime_emit_text_dt : bool
        v0.0.15. Emit the new v0.0.15 text-x dt_labels page (xData is
        a per-point text dataset, axis mode='labels').
    datetime_full_labels : bool or None
        Back-compat shim: True -> density_pct=100, False -> density_pct=10.
    """
    # ---- v0.0.15 back-compat shim ---------------------------------
    if datetime_full_labels is not None:
        try:
            datetime_label_density_pct = (
                100 if bool(datetime_full_labels) else 10
            )
        except Exception:
            pass
    try:
        datetime_label_density_pct = max(0, min(100,
            int(round(float(datetime_label_density_pct)))))
    except Exception:
        datetime_label_density_pct = 10
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
        """Count ``n`` finished dataset operations and report them to
        ``column_cb(done, total)``; callback errors are ignored.
        """
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

    # ---- v0.0.15: Optional per-point datetime label datasets -------------
    # Replaces the v0.0.11..0.0.13 SetDataDateTime path.  When the user
    # asks for datetime-duplicate plots we build TWO text datasets from
    # the file's MJD-like sort key (already in sort permutation order
    # via ``sort_idx``):
    #   * density -- density_pct % of finite points labelled, rest empty
    #                (bound to ``xy.labels.val`` on the numeric-x page)
    #   * textx   -- one YYYY-MM-DD HH:MM:SS string per data point
    #                (bound to ``xy.xData.val`` on the text-x page;
    #                axis is set to mode='labels')
    # Both are pushed; per-page gates decide which dt page(s) to render.
    # The numeric seconds x dataset is reused from the seconds page.
    datelabels_density_name = None
    datelabels_textx_name = None
    # v0.0.16: numeric Veusz-datetime-seconds dataset (drives dt_labels
    # page in mode='datetime') and the sorted MJD array used for
    # broken-axis parity on the dt_labels page.
    datelabels_dtnum_name = None
    datelabels_mjd_sorted = None
    dt_total_points = 0              # for axis tick scaling on text-x
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
            dt_total_points = int(arr_mjd.size)
            density_name = "%s__%s__datelabels_density" % (
                base, safe_dsname(sort_key)
            )
            textx_name = "%s__%s__datelabels_textx" % (
                base, safe_dsname(sort_key)
            )
            # v0.0.16: numeric Veusz-datetime-seconds dataset name.
            dtnum_name = "%s__%s__datelabels_dtnum" % (
                base, safe_dsname(sort_key)
            )
            datelabels_density_name = None if not datetime_emit_numeric_dt else build_density_datestr_dataset(
                doc, density_name, arr_mjd,
                density_pct=datetime_label_density_pct,
                log_cb=log_cb,
            )
            if datetime_emit_text_dt:
                datelabels_textx_name = None if not EMIT_TEXT_X_DATASETS else build_textx_dataset(
                    doc, textx_name, arr_mjd, log_cb=log_cb,
                )
                # v0.0.16: emit numeric dtsec dataset that drives the
                # dt_labels page (mode='datetime').
                if build_dtnum_dataset(
                    doc, dtnum_name, arr_mjd, log_cb=log_cb,
                ):
                    datelabels_dtnum_name = dtnum_name
            # v0.0.16: stash the sorted MJD array for broken-axis
            # parity computation on the dt_labels page.
            try:
                datelabels_mjd_sorted = np.asarray(
                    arr_mjd, dtype=float
                )
            except Exception:
                datelabels_mjd_sorted = None
            # Tag all datasets so they group nicely in Veusz's Data
            # Browser alongside the file's other sorted datasets.
            for _ln in (datelabels_density_name, datelabels_textx_name,
                        datelabels_dtnum_name):
                if _ln:
                    try:
                        doc.TagDatasets(base, [_ln])
                        doc.TagDatasets("sorted", [_ln])
                    except Exception:
                        pass
            # Note: legacy v0.0.14 dataset names
            # (``..__datelabels_sparse`` / ``..__datelabels_full``) are
            # not re-emitted here.  External code that read them must
            # be updated to use ``..__datelabels_density`` (numeric-x)
            # or ``..__datelabels_textx`` (text-x).  The v0.0.15 GUI
            # plumbs the same controls so the user-facing experience
            # is preserved.
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
        # v0.0.13: Franks has no channel tags, so identity is
        # (col_name, (None,)); 1 trace per graph -> vary_style False.
        apply_trace_style(
            xy, identity_key=(cname, (None,)), vary_style=False,
        )

    if log_cb:
        log_cb("  Page built with %d graphs" %
               sum(1 for c in cols if c != sort_key
                   and np.asarray(cols[c]).dtype.kind in ("f", "i", "u")))

    # ---- v0.0.15: datetime-duplicate per-file pages ----------------------
    # Two pages may be emitted, gated independently:
    #   numeric-x dt page (v0.0.14 lineage) -- xData = seconds dataset;
    #     per-point xy.labels carry density-thinned date strings.
    #   text-x dt_labels page (v0.0.15) -- xData = per-point text
    #     dataset; axis mode='labels' renders strings on the axis itself.
    # ---- numeric-x dt page (v0.0.14 lineage) -----------------------------
    if datetime_emit_numeric_dt and datelabels_density_name is not None:
        page_dt = doc.Root.Add(
            "page", name=safe_dsname("%s_dt" % base)
        )
        try:
            page_dt.notes.val = "\n".join(
                data.get("header_lines", [])
                + ["",
                   "Datetime-axis duplicate (v0.0.15 numeric-x; "
                   "per-point xy.labels at %d%% density)."
                   % datetime_label_density_pct]
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
                    make_broken_x_axis(
                        graph, break_pairs,
                        label=sort_key or "index",
                        show_gridlines=True,
                    )
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
            xy = graph.Add("xy", name=safe_dsname("xy_%s_dt" % cname))
            if x_ds is not None:
                xy.xData.val = x_ds
            xy.yData.val = ds
            try:
                xy.labels.val = datelabels_density_name
            except Exception:
                pass
            style_xy_datetime_labels(xy)
            apply_trace_style(
                xy, identity_key=(cname, (None,)), vary_style=False,
            )
        if log_cb:
            log_cb("  Datetime-duplicate page built (%s_dt, %d%% density "
                   "labels)." % (base, datetime_label_density_pct))

    # ---- dt_labels page (v0.0.16 mode='datetime') -----------------------
    # v0.0.16 swap: bind xData to the numeric dtsec dataset and set the
    # axis to mode='datetime' for true date ticks; install broken x-axis
    # when the seconds-axis dt page has one (MJD->dtsec mapping).
    if datetime_emit_text_dt and (datelabels_dtnum_name is not None
                                  or datelabels_textx_name is not None):
        page_dtL = doc.Root.Add(
            "page", name=safe_dsname("%s_dt_labels" % base)
        )
        try:
            page_dtL.notes.val = "\n".join(
                data.get("header_lines", [])
                + ["",
                   "Datetime-axis duplicate (v0.0.16 numeric dtsec + "
                   "axis mode='datetime'; broken-axis parity with dt "
                   "page)."]
            )
        except Exception:
            pass
        grid_dtL = page_dtL.Add("grid", columns=2)
        # v0.0.16: derive dtsec break pairs from the SAME MJD break
        # pairs the dt page uses.  break_pairs above was computed from
        # the seconds-page sort-key array which IS MJD-derived for
        # MJD-like sort keys (or MJD-shifted JD), so the mapping is
        # direct.
        try:
            dt_break_pairs = mjd_break_pairs_to_dtsec(break_pairs)
        except Exception:
            dt_break_pairs = []
        x_ds_dtl = datelabels_dtnum_name or datelabels_textx_name
        for cname in cols.keys():
            if cname == sort_key:
                continue
            arr = np.asarray(cols[cname])
            if arr.dtype.kind not in ("f", "i", "u"):
                continue
            ds = "%s__%s__sorted" % (base, safe_dsname(cname))
            graph = grid_dtL.Add(
                "graph", name=safe_dsname("g_%s_dtL" % cname)
            )
            try:
                graph.y.label.val = cname
                graph.y.GridLines.hide.val = False
            except Exception:
                pass
            # Install broken x-axis FIRST (when needed); AxisBroken
            # subclasses Axis so it inherits the mode property.
            if dt_break_pairs:
                try:
                    make_broken_x_axis(
                        graph, dt_break_pairs,
                        label=sort_key or "index",
                        show_gridlines=True,
                    )
                except Exception:
                    pass
            try:
                configure_axis_datetime_mode(
                    graph.x, label=sort_key or "index",
                )
            except Exception:
                pass
            xy = graph.Add("xy", name=safe_dsname("xy_%s_dtL" % cname))
            xy.xData.val = x_ds_dtl       # numeric dtsec
            xy.yData.val = ds
            apply_trace_style(
                xy, identity_key=(cname, (None,)), vary_style=False,
            )
        if log_cb:
            log_cb("  Datetime-labels page built (%s_dt_labels, numeric "
                   "dtsec, mode='datetime')." % base)


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
                                     datetime_duplicate=False,
                                     datetime_full_labels=None,
                                     datetime_label_density_pct: int = 10,
                                     datetime_emit_numeric_dt: bool = False,
                                     datetime_emit_text_dt: bool = True,
                                     hide_keys: bool = False):
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
    datetime_label_density_pct : int
        v0.0.15. Density (0..100) of date-string anchors on the
        numeric-x dt overlay.
    datetime_emit_numeric_dt : bool
        v0.0.15. Emit the v0.0.14-lineage numeric-x dt overlay.
    datetime_emit_text_dt : bool
        v0.0.15. Emit the text-x dt_labels overlay (uniform spacing).
    datetime_full_labels : bool or None
        Back-compat shim (True -> 100, False -> 10).
    """
    # ---- v0.0.15 back-compat shim ---------------------------------
    if datetime_full_labels is not None:
        try:
            datetime_label_density_pct = (
                100 if bool(datetime_full_labels) else 10
            )
        except Exception:
            pass
    try:
        datetime_label_density_pct = max(0, min(100,
            int(round(float(datetime_label_density_pct)))))
    except Exception:
        datetime_label_density_pct = 10
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
            key.hide.val = bool(hide_keys)
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
        # v0.0.13: overlay carries 1 trace per page (time-combined
        # across all member files), identity = (col, (None,)).
        apply_trace_style(
            xy, identity_key=(cname, (None,)), vary_style=False,
        )
        if log_cb:
            log_cb("  Overlay page '%s': 1 trace (time-combined across "
                   "%d files)." % (page_name, len(members)))

        # v0.0.15/0.0.16: datetime companion datasets built from the
        # concatenated, time-sorted MJD array.  Only proceeds when
        # EVERY contributing file has an MJD column.
        # v0.0.17: NO channel-tag sentinel filter here (Franks overlay is
        # one trace per page; the FITS sentinel-tuple case does not
        # apply).
        dt_density_name = None
        dt_textx_name = None
        dt_dtnum_name = None        # v0.0.16: numeric dtsec dataset
        dt_mjds_sorted = None       # v0.0.16: for broken-axis parity
        dt_total = 0
        if datetime_duplicate and all(
            m.get("mjd") is not None for m in members
        ):
            try:
                mjds = np.concatenate(
                    [np.asarray(m["mjd"], dtype=float) for m in members]
                )
                mjds_s = mjds[order]
                dt_total = int(mjds_s.size)
                dt_density_name = None if not datetime_emit_numeric_dt else build_density_datestr_dataset(
                    doc,
                    "OverlayCat__%s__datelabels_density" % c_safe,
                    mjds_s,
                    density_pct=datetime_label_density_pct,
                    log_cb=log_cb,
                )
                if datetime_emit_text_dt:
                    dt_textx_name = None if not EMIT_TEXT_X_DATASETS else build_textx_dataset(
                        doc,
                        "OverlayCat__%s__datelabels_textx" % c_safe,
                        mjds_s, log_cb=log_cb,
                    )
                    # v0.0.16: numeric dtsec dataset for mode='datetime'.
                    dtnum_candidate = (
                        "OverlayCat__%s__datelabels_dtnum" % c_safe
                    )
                    if build_dtnum_dataset(
                        doc, dtnum_candidate, mjds_s, log_cb=log_cb,
                    ):
                        dt_dtnum_name = dtnum_candidate
                # v0.0.16: keep the sorted MJD array for break detection.
                dt_mjds_sorted = np.asarray(mjds_s, dtype=float)
            except Exception as _exc:
                if log_cb:
                    log_cb("  Overlay '%s' datetime build failed: %s"
                           % (cname, _exc))
                dt_density_name = None
                dt_textx_name = None
                dt_dtnum_name = None
                dt_mjds_sorted = None

        # ---- v0.0.15: numeric-x dt overlay page ----------------
        if (datetime_duplicate and datetime_emit_numeric_dt
                and dt_density_name):
            page_dt = doc.Root.Add(
                "page", name=safe_dsname("%s_dt" % page_name)
            )
            try:
                page_dt.notes.val = (
                    "Datetime-axis duplicate of '%s' "
                    "(v0.0.15 numeric-x; per-point xy.labels at "
                    "%d%% density)."
                    % (page_name, datetime_label_density_pct)
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
                    make_broken_x_axis(
                        graph_dt, break_pairs,
                        label="MJD", show_gridlines=True,
                    )
                except Exception:
                    try:
                        graph_dt.x.label.val = "MJD"
                        graph_dt.x.GridLines.hide.val = False
                    except Exception:
                        pass
            else:
                try:
                    graph_dt.x.label.val = "MJD"
                    graph_dt.x.GridLines.hide.val = False
                except Exception:
                    pass
            try:
                key_dt = graph_dt.Add("key", name="key1_dt")
                key_dt.hide.val = bool(hide_keys)
                key_dt.Border.hide.val = False
            except Exception:
                pass
            xy_dt = graph_dt.Add(
                "xy", name=safe_dsname("xy_%s_dt" % cname)
            )
            xy_dt.xData.val = x_name              # numeric seconds (overlay dataset)
            xy_dt.yData.val = y_name
            try:
                xy_dt.labels.val = dt_density_name
            except Exception:
                pass
            style_xy_datetime_labels(xy_dt)
            try:
                xy_dt.key.val = cname
            except Exception:
                pass
            apply_trace_style(
                xy_dt, identity_key=(cname, (None,)), vary_style=False,
            )
            if log_cb:
                log_cb("  Datetime-overlay page '%s_dt': 1 trace (%d%% "
                       "density labels)."
                       % (page_name, datetime_label_density_pct))

        # ---- v0.0.16: dt_labels overlay page (mode='datetime') ----
        if (datetime_duplicate and datetime_emit_text_dt
                and (dt_dtnum_name or dt_textx_name)):
            page_dtL = doc.Root.Add(
                "page", name=safe_dsname("%s_dt_labels" % page_name)
            )
            try:
                page_dtL.notes.val = (
                    "Datetime-axis duplicate of '%s' "
                    "(v0.0.16 numeric dtsec + axis mode='datetime'; "
                    "broken-axis parity)."
                    % page_name
                )
            except Exception:
                pass
            graph_dtL = page_dtL.Add("graph", name="g_overlay_dtL")
            try:
                graph_dtL.y.label.val = cname
                graph_dtL.y.GridLines.hide.val = False
            except Exception:
                pass
            # v0.0.16: detect breaks in MJD space from the same sorted
            # MJD array used for the labels datasets, then map MJD
            # gap pairs to Veusz seconds for the broken-axis call.
            dt_break_pairs_ov = []
            try:
                if dt_mjds_sorted is not None and dt_mjds_sorted.size:
                    # v0.0.18: unit-aware helper for parity (MJD,
                    # factor=1.0 -> behavior unchanged).
                    mjd_breaks = detect_time_breaks_unit_aware(
                        dt_mjds_sorted, "MJD",
                        k_factor=gap_k,
                        absolute_gap_days=gap_absolute,
                    )
                    dt_break_pairs_ov = mjd_break_pairs_to_dtsec(
                        mjd_breaks
                    )
            except Exception:
                dt_break_pairs_ov = []
            if dt_break_pairs_ov:
                try:
                    make_broken_x_axis(
                        graph_dtL, dt_break_pairs_ov,
                        label="MJD", show_gridlines=True,
                    )
                except Exception:
                    pass
            try:
                configure_axis_datetime_mode(
                    graph_dtL.x, label="MJD",
                )
            except Exception:
                pass
            try:
                key_dtL = graph_dtL.Add("key", name="key1_dtL")
                key_dtL.hide.val = bool(hide_keys)
                key_dtL.Border.hide.val = False
            except Exception:
                pass
            xy_dtL = graph_dtL.Add(
                "xy", name=safe_dsname("xy_%s_dtL" % cname)
            )
            xy_dtL.xData.val = dt_dtnum_name or dt_textx_name
            xy_dtL.yData.val = y_name
            try:
                xy_dtL.key.val = cname
            except Exception:
                pass
            apply_trace_style(
                xy_dtL, identity_key=(cname, (None,)), vary_style=False,
            )
            if log_cb:
                log_cb("  Datetime-overlay page '%s_dt_labels': 1 trace "
                       "(numeric dtsec, mode='datetime')." % page_name)


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
        """Worker thread for one batch: ``files`` parsed on ``max_threads`` threads
        into ``cache``.
        """
        super().__init__(parent)
        self.files = files
        self.max_threads = max_threads
        self.cache = cache

    def run(self) -> None:
        """Parse every file in a thread pool; emit ``progress(done, total, path)``
        per file, then ``finished_ok(results)`` ({path: parse result or
        exception}) or ``failed(message with traceback)``.
        """
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
        """Main window: shared AutoPlot layout plus the Franks options, a memory
        monitor and an empty project (created on the first batch).
        """
        super().__init__("%s %s - Native HDF5 Export" % (APP_NAME, APP_VERSION),
                         default_mode="dark")
        self.app_name, self.app_version = APP_NAME, APP_VERSION
        self._file_filter = (
            "Franks files (*.t00new *.dat *.bak* *.OrigBack* *.back);;All files (*)"
        )
        self.cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=DEFAULT_RSS_HIGH_WATER_MB
        ))
        # 0.4.0: a bound method (held weakly by the monitor thread); log() is
        # thread-safe.  The old lambda kept the window alive from that thread.
        self.monitor = MemoryMonitor(self.cache, callback=self._rss_crossed)
        self.monitor.start()
        self.veusz_doc = None
        self.worker = None
        self.log("MAX_THREADS = %d" % MAX_THREADS)

    # ----- options form ----------------------------------------------------
    def _rss_crossed(self, rss: float) -> None:
        """Memory monitor callback (monitor thread): log the RSS crossing."""
        self.log("RSS crossed high-water mark (%.1f MiB) -- spilling to disk." % rss)

    def _populate_options(self, form: QFormLayout) -> None:
        """Build the Franks processing options: worker threads, RSS spill
        threshold, Open-in-Veusz, MJD date strings, the small/large gap controls
        (add_gap_controls), hidden legend keys, combined (overlay) plots only, the
        legacy duplicate date-time pages, minimized/full project output and GPU
        (CuPy) sorting.
        """
        self.thread_spin = QSpinBox()
        self.thread_spin.setToolTip("Number of files read at the same time (threads).")
        self.thread_spin.setRange(1, max(1, (os.cpu_count() or 4) * 2))
        self.thread_spin.setValue(MAX_THREADS)
        form.addRow(QLabel("Worker threads:"), self.thread_spin)

        self.rss_spin = QSpinBox()
        self.rss_spin.setToolTip(
            "Resident memory (MiB) above which large arrays are moved from "
            "RAM to temporary files on disk (numpy.memmap) - lower it on a "
            "small computer.")
        self.rss_spin.setRange(128, 65536)
        self.rss_spin.setSingleStep(128)
        self.rss_spin.setValue(DEFAULT_RSS_HIGH_WATER_MB)
        form.addRow(QLabel("RSS spill threshold (MiB):"), self.rss_spin)

        self.show_embed_cb = QCheckBox("Native build: use Open in Veusz after saving")
        self.show_embed_cb.setToolTip(("Not available in the native build: use 'Open in Veusz' after"
            ' saving (Veusz is only needed to view the project).'))
        self.show_embed_cb.setChecked(False)
        self.show_embed_cb.setEnabled(False)
        form.addRow(self.show_embed_cb)

        self.datestr_cb = QCheckBox(
            "Generate MJD -> date strings (YYYY-MM-DD_HH:MM:SS) datasets"
        )
        self.datestr_cb.setToolTip(("Also store each MJD column as text 'YYYY-MM-DD_HH:MM:SS' "
            '(UTC) - only needed by old Veusz documents; the time axes '
            'are date-times without it.'))
        self.datestr_cb.setChecked(False)
        form.addRow(self.datestr_cb)

        # --- Gap controls (0.4.0, shared with FITS AutoPlot) ---------------
        # Line break where dt > K x median(dt); axis break only where
        # dt > n x K x median(dt) (n = 0: one continuous time axis).
        self.add_gap_controls(form)

        # v0.0.19: hide legend keys on every page.
        self.hide_keys_cb = QCheckBox("Hide legend keys on all pages")
        self.hide_keys_cb.setChecked(False)
        self.hide_keys_cb.setToolTip(
            "When checked, every key (legend) widget added to a graph "
            "is created with Border/hide set, suppressing the legend "
            "frame."
        )
        form.addRow(self.hide_keys_cb)

        self.combined_only_cb = QCheckBox(
            "Combined (overlay) plots only -- skip per-file pages"
        )
        self.combined_only_cb.setToolTip(
            "Build only the overlay pages (all files on one graph per unit)"
            " and no page per file - much faster to open for hundreds of "
            "files.")
        self.combined_only_cb.setChecked(False)
        form.addRow(self.combined_only_cb)

        # --- Datetime-duplicate plots (v0.0.11) ---------------------------
        self.datetime_dup_cb = QCheckBox(
            "Legacy: also build duplicate _dt / _dt_labels pages"
        )
        self.datetime_dup_cb.setChecked(False)
        self.datetime_dup_cb.setToolTip(
            "0.4.0: every page already plots its MJD axis as a Veusz "
            "date-time axis in local ISO 8601; the duplicate pages are only "
            "needed for old workflows.")
        form.addRow(self.datetime_dup_cb)

        # --- v0.0.15: date-label density + dt page variants ---------------
        # The v0.0.14 "full vs sparse" boolean is generalised to an
        # integer percentage 0..100 that controls how many evenly
        # spaced anchor points get a YYYY-MM-DD HH:MM:SS label on the
        # numeric-x dt page.  Two independent dt page variants are
        # available; each can be turned on or off independently.
        self.label_density_spin = QSpinBox()
        self.label_density_spin.setRange(0, 100)
        self.label_density_spin.setValue(10)
        self.label_density_spin.setSuffix(" %")
        self.label_density_spin.setToolTip(
            "Percentage of finite data points that get a date-string "
            "label.  0 = no labels, 100 = one label per point.  "
            "Anchors are evenly spaced.  Applies to the numeric-x dt "
            "page (xy.labels) and scales the tick count on the text-x "
            "dt_labels page (axis mode='labels')."
        )
        form.addRow(
            QLabel("Date-label density on dt pages:"),
            self.label_density_spin,
        )
        self.emit_numeric_dt_cb = QCheckBox(
            "Legacy: per-point date text labels page (_dt)"
        )
        self.emit_numeric_dt_cb.setChecked(False)
        self.emit_numeric_dt_cb.setToolTip(
            "Off by default (0.3.0). When checked, also builds the older _dt page: "
            "the sort-key x axis with density-thinned date text labels "
            "on the points (xy.labels).  The dt_labels page below is the "
            "date-time version and needs no labels."
        )
        form.addRow(self.emit_numeric_dt_cb)
        self.emit_text_dt_cb = QCheckBox(
            "Date-time dt_labels page (Veusz date-time axis, local ISO 8601)"
        )
        self.emit_text_dt_cb.setChecked(True)
        self.emit_text_dt_cb.setToolTip(
            "When checked, builds the dt_labels pages: the time stamps are "
            "stored as Veusz date-time datasets (UTC) and drawn on a "
            "date-time axis in the local time of this computer, ticks "
            "like 2026-11-09T08:39:00-05:00 (24 h, with the UTC offset); "
            "spacing is proportional to elapsed time and gaps break the axis."
        )
        form.addRow(self.emit_text_dt_cb)

        # --- Minimized Veusz save (v0.0.17) -------------------------------
        # When "Minimized Veusz File" is checked, the saved .vszh5 contains
        # ONLY the datasets that the plot widget tree actually references.
        # The nested "Generate Full Veusz file" sub-checkbox is enabled
        # only when the parent is on; when also checked, a second file
        # with the legacy full dataset list is written alongside (suffix
        # "_full").
        self.min_vesz_cb = QCheckBox("Minimized Veusz File")
        self.min_vesz_cb.setChecked(False)
        self.min_vesz_cb.setToolTip(
            "When checked, the saved .vszh5 contains only the datasets "
            "directly referenced by widgets in the document.  Unreferenced "
            "datasets (intermediate label / density / dtnum byproducts that "
            "no widget consumes) are evicted from the file.  The in-memory "
            "document is unchanged -- datasets are restored after save."
        )
        form.addRow(self.min_vesz_cb)
        self.full_vesz_cb = QCheckBox("Generate Full Veusz file")
        self.full_vesz_cb.setChecked(False)
        self.full_vesz_cb.setEnabled(False)
        self.full_vesz_cb.setToolTip(
            "Only available when 'Minimized Veusz File' is checked.  When "
            "BOTH are checked, the save action writes the minimized file "
            "AND a parallel '_full.vszh5' that includes every dataset "
            "(legacy v0.0.16 save behaviour)."
        )
        form.addRow(self.full_vesz_cb)
        self.min_vesz_cb.toggled.connect(self.full_vesz_cb.setEnabled)
        self.min_vesz_cb.toggled.connect(
            lambda on: self.full_vesz_cb.setChecked(False) if not on else None
        )

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
        """Start a batch: create the project if needed and run FranksBatchWorker
        on the selected files (buttons disabled, progress bars shown).
        """
        if not self.selected_files:
            QMessageBox.information(self, "No files",
                                    "Select one or more Franks data files first.")
            return
        self.cache.cfg.rss_high_water_mb = int(self.rss_spin.value())

        if self.veusz_doc is None:
            self.veusz_doc = NativeDocument("Franks_AutoPlot")

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
        """Advance the parsing progress bar, log the file and refresh the RSS.
        """
        self.progress_bar.setValue(done)
        self.log("  parsed [%d/%d] %s" % (done, total, os.path.basename(key)))
        self.update_mem_label(self.cache.rss_mb())

    def _on_failed(self, msg: str) -> None:
        """Report a failed batch, re-enable Process and hide the progress bars.
        """
        self.log("Batch failed: %s" % msg)
        QMessageBox.critical(self, "Batch failed", msg)
        self.process_button.setEnabled(True)
        self.hide_progress_bars()

    def _on_done(self, results: Dict[str, Any]) -> None:
        """Turn the batch results into project pages.

        Files are taken in file-name order; each is pushed with
        push_franks_to_veusz (per-file pages), then the overlay pages are built.
        The gap settings of the GUI become the project's TimeLayout; errors are
        logged.
        """
        emit_datestr = bool(self.datestr_cb.isChecked())
        # v0.0.9 broken-axis / overlay knobs
        gap_k = float(self.gap_k_spin.value())
        # 0.4.0: manual gap in days for the page builders; every time axis
        # is laid out by the document's TimeLayout (line breaks at
        # dt > K x median(dt), axis breaks at n x that, local ISO 8601).
        gap_absolute = self.gap_manual_days()
        self.veusz_doc.time_layout = self.time_layout()
        self.log(self.veusz_doc.time_layout.describe())
        # v0.0.19: hide-legend-keys checkbox
        hide_keys = bool(self.hide_keys_cb.isChecked())
        plot_individual = not bool(self.combined_only_cb.isChecked())
        # v0.0.11: datetime-duplicate toggle
        datetime_duplicate = bool(self.datetime_dup_cb.isChecked())
        # v0.0.15: density-pct (0..100) + two dt page gates.
        datetime_label_density_pct = int(self.label_density_spin.value())
        datetime_emit_numeric_dt = bool(self.emit_numeric_dt_cb.isChecked())
        datetime_emit_text_dt = bool(self.emit_text_dt_cb.isChecked())
        # v0.0.10: drive the process-wide GPU flag from the checkbox
        enable_gpu(self.gpu_cb.isChecked() and self.gpu_cb.isEnabled())
        # Keep the GUI responsive across many files: pump the Qt event
        # loop between every push so the log pane scrolls live and the
        # window doesn't appear "stuck" during a long batch insert.
        app = QApplication.instance()
        total = len(results)
        def _col_cb(done: int, total_ops: int) -> None:
            """Per-file column progress: resize the bar to ``total_ops`` when it
            changes, set it to ``done`` and keep the GUI responsive.
            """
            if self.column_progress_bar.maximum() != max(1, total_ops):
                self.column_progress_bar.setRange(0, max(1, total_ops))
            self.column_progress_bar.setValue(done)
            if app is not None:
                app.processEvents()
        file_records = []  # accumulator for the overlay post-pass
        # 0.4.0: pages in file-name order, not thread completion order.
        ordered = sorted(results.items(), key=lambda item: (os.path.basename(str(item[0])), str(item[0])))
        for idx, (path, data) in enumerate(ordered, start=1):
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
                                     datetime_duplicate=datetime_duplicate,
                                     datetime_label_density_pct=datetime_label_density_pct,
                                     datetime_emit_numeric_dt=datetime_emit_numeric_dt,
                                     datetime_emit_text_dt=datetime_emit_text_dt,
                                     hide_keys=hide_keys)
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
                    datetime_label_density_pct=datetime_label_density_pct,
                    datetime_emit_numeric_dt=datetime_emit_numeric_dt,
                    datetime_emit_text_dt=datetime_emit_text_dt,
                    hide_keys=hide_keys,
                )
            except Exception as exc:
                self.log("  build_unit_overlay_pages_franks failed: %s"
                         % exc)
        self.log("Batch complete.")
        self.hide_progress_bars()
        self.process_button.setEnabled(True)
        self.save_button.setEnabled(True)
        # v0.2.0: interactive pyqtgraph view of the in-memory project
        self.show_plots(self.veusz_doc)

    # ----- save ------------------------------------------------------------
    def _save_project(self) -> None:
        """Ask for a file name and save the project with the current gap settings.

        With "Minimized Veusz File" the minimized project is written (and, if
        also ticked, the full one as ``<name>_full.vszh5``); otherwise the full
        project.  The first file written is offered to "Open in Veusz".
        """
        if self.veusz_doc is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Process some files first.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Veusz Project", "", "Veusz HDF5 Projects (*.vszh5)"
        )
        if not fn:
            return
        # 0.4.0: lay out the time axes with the gap settings shown now.
        self.veusz_doc.time_layout = self.time_layout()
        # v0.0.17: branch on Minimized Veusz File checkbox.
        want_min = bool(self.min_vesz_cb.isChecked())
        want_full = bool(self.full_vesz_cb.isChecked()) and want_min
        try:
            written_paths = []
            if want_min:
                primary = save_vszh5_minimized(
                    self.veusz_doc, fn, log_cb=self.log,
                )
                self.log("Saved minimized %s" % primary)
                written_paths.append(primary)
                if want_full:
                    base, _ext = os.path.splitext(primary)
                    full_path = base + "_full.vszh5"
                    written_full = save_vszh5(self.veusz_doc, full_path)
                    self.log("Saved full %s" % written_full)
                    written_paths.append(written_full)
            else:
                written = save_vszh5(self.veusz_doc, fn)
                self.log("Saved %s" % written)
                written_paths.append(written)
            self.mark_project_saved(written_paths[0])
            QMessageBox.information(
                self, "Saved", "Wrote:\n%s" % "\n".join(written_paths),
            )
        except Exception as exc:
            self.log("Save failed: %s" % exc)
            QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event) -> None:
        """Stop the memory monitor, close the project and delete the cache's
        spill files before the window closes.
        """
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
    """Run the Franks AutoPlot window; returns the Qt exit code."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = FranksAutoPlotWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
