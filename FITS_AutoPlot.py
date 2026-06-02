# -*- coding: utf-8 -*-
"""
=============================================================================
FITS_AutoPlot.py
-----------------------------------------------------------------------------
# %% Header Info

Replacement for the older FITS_AutoPlot.py in the FitsAutoPlot repo.
GUI-driven batch processor for NRAO Green Bank FITS files (OnePpsDeltas
samplers and any other FITS BinTable / image HDUs) that produces a Veusz
3.4 / 4.1 compatible ``.vszh5`` project file with:

    * column-oriented sorted datasets (one Veusz dataset per FITS column),
    * tagged datasets (filename tag + raw/sorted tag),
    * appropriate plots per HDU (scatter for tables, image for image HDUs),
    * a live embedded Veusz preview while the file is built.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.19: Wire string gap parser + hide-keys checkbox through GUI.
# Date: 2026-05-16
#              GUI additions:
#                * New ``self.gap_abs_str`` QLineEdit (placeholder
#                  "e.g. 5m3d2h (overrides spinbox)") sits below the
#                  existing hours spinbox.  When non-empty its parsed
#                  value (via ``parse_gap_string`` from
#                  ``_autoplot_common``) overrides the spinbox-derived
#                  ``gap_absolute`` -- backward compatible.
#                * New ``self.hide_keys_cb`` QCheckBox ("Hide legend
#                  keys on all pages").  The flag is plumbed via a new
#                  ``hide_keys`` kwarg through ``push_to_veusz`` ->
#                  ``_build_pages`` and ``build_unit_overlay_pages``;
#                  every key widget's ``Border.hide.val`` is now driven
#                  by ``hide_keys`` instead of being unconditionally
#                  False.
#              Default behaviour unchanged when both controls are left
#              at their defaults (blank string + unchecked).
#
# %%%% 0.0.18: Unit-aware time-break detection (manual gap units fix).
# Date: 2026-05-16
#              The GUI "Manual gap (hours)" spinbox is converted to
#              MJD-days in submit_handler (``gap_absolute = hours /
#              24``).  Pre-v0.0.18 that day-scaled threshold was
#              passed directly to ``detect_time_breaks(x, ...)`` even
#              when ``x`` was in seconds-of-day units (sort_key
#              column == TIME / TIMESTAMP), which made the threshold
#              microscopic compared to second-scale diffs and
#              manufactured false broken-X axes on the per-file and
#              seconds-overlay pages.
#
#                * All three break-detection call sites in
#                  ``_build_pages`` (tagged-HDU page, untagged-HDU
#                  page, unit-overlay seconds page) now call
#                  ``detect_time_breaks_unit_aware(x, col_name,
#                  k_factor=gap_k, absolute_gap_days=gap_absolute)``
#                  which converts the day-scale threshold to the
#                  column's native units (1.0 for MJD/DMJD/JD,
#                  86400.0 for TIME/TIMESTAMP/SECONDS-style).
#                * The dt_labels overlay page (MJD-array break
#                  detection) also routes through the unit-aware
#                  helper with col_name="MJD" for parity; behavior
#                  is unchanged since the conversion factor is 1.0.
#                * Overlay path now tracks ``sort_col_name_by_unit``
#                  (first observed sort-key column per unit) so the
#                  combined-x break detection knows what units it is
#                  working with.
#                * New imports: ``detect_time_breaks_unit_aware``,
#                  ``column_unit_factor_from_day``.
#                * Revision history kept in DESCENDING semantic-
#                  version order.
#
# %%%% 0.0.17: Minimized .vszh5 save + sentinel-tag dt overlay filter.
# Date: 2026-05-16
#              New GUI checkboxes + correctness fix on the dt overlay
#              pages:
#                * Two new checkboxes on the FITS GUI:
#                    - "Minimized Veusz File" -- saved .vszh5 contains
#                      ONLY the datasets directly referenced by widgets.
#                      Implementation lives in ``_autoplot_common.
#                      save_vszh5_minimized``: walks Root.WalkWidgets(),
#                      collects xData/yData/labels/scalePoints/...
#                      references, evicts unreferenced datasets via
#                      RemoveData, calls the standard save_vszh5, then
#                      restores the evicted datasets in memory so the
#                      ``doc`` is unchanged post-save.
#                    - "Generate Full Veusz file" -- sub-checkbox, only
#                      enabled when the parent is checked.  When BOTH
#                      are checked, save writes the minimized file AND
#                      a parallel ``<name>_full.vszh5`` carrying every
#                      dataset (legacy v0.0.16 behaviour).
#                * FIX: the FITS dt overlay pages now skip placeholder
#                  channel-tag tuples like ``("dataset", "dataset")``.
#                  These catch-all groupings (the FITS tag-column parser
#                  emits them when every row of a tag column is the same
#                  literal placeholder string) carried degenerate MJD
#                  coverage that manufactured spurious broken-axis
#                  breaks in the combined-MJD break detection used by
#                  the dt overlay pages.  The new predicate
#                  ``is_sentinel_tag_tuple(tup)`` (in _autoplot_common)
#                  gates dt emission; real channel-pair tuples like
#                  ("A", "B") pass through unaffected.  Seconds-overlay
#                  traces for sentinel tuples are still drawn -- they're
#                  harmless on a unitless seconds axis.
#                * Revision history kept in DESCENDING semantic-version
#                  order.
#
# %%%% 0.0.16: dt_labels page upgraded to mode='datetime' (true date ticks).
# Date: 2026-05-16
#                * The v0.0.15 dt_labels page used a per-point text
#                  dataset bound to xData with axis mode='labels'.  That
#                  works for compact "strip" views but the axis is
#                  uniform-sample-spaced and gives no proper date tick
#                  cadence (and no broken-axis support).
#                * v0.0.16 emits a NEW numeric dataset alongside the
#                  text-x dataset: ``__datelabels_dtnum``.  Values are
#                  Veusz datetime seconds (seconds since 2009-01-01
#                  UTC), computed from MJD by
#                  ``mjd_to_veusz_seconds`` (NaN-preserving, vectorised).
#                  The dt_labels page now binds xData to this numeric
#                  dataset and sets the axis to ``mode='datetime'``
#                  via the new helper ``configure_axis_datetime_mode``.
#                  Veusz renders proper date ticks (DateTicks class on
#                  Axis / AxisBroken).
#                * Broken-axis parity: when the seconds-axis (dt) page
#                  gets a broken x-axis for time gaps, the dt_labels
#                  page now gets one too.  The MJD break_pairs computed
#                  for break detection are mapped to Veusz seconds by
#                  the new helper ``mjd_break_pairs_to_dtsec`` and
#                  installed via ``make_broken_x_axis`` BEFORE
#                  configure_axis_datetime_mode (AxisBroken extends
#                  Axis so it inherits the mode property).
#                * v0.0.15 text-x dataset (``__datelabels_textx``) is
#                  still emitted for back-compat; the dt_labels render
#                  prefers dtnum and falls back to textx only if dtnum
#                  failed to build (e.g. all-NaN MJDs).
#                * Tagged HDU, untagged HDU, and overlay dt_labels
#                  pages all share this pattern.  Overlay break_pairs
#                  are computed from concatenated MJDs across every
#                  eligible (hdu, col, tag) trace on the page.
#                * Plugin fields/GUI unchanged in 0.0.16 (no new
#                  user-facing controls); plugin headers bumped to
#                  0.0.16 only to keep versions in lock-step.
#                * Why not SetDataDateTime? On Veusz 3.4 that path
#                  fires hundreds of internal exceptions in
#                  DatasetDateTime construction; axis-side mode
#                  ='datetime' on a plain numeric dataset is the
#                  documented, working alternative.
#                * GUI: "Absolute gap (units of x; 0=auto)" spinbox is
#                  renamed "Manual gap (hours; 0=auto)" and now expects
#                  HOURS as input.  The widget value is divided by 24
#                  before being passed to detect_time_breaks so MJD-axis
#                  time gaps are specified in a familiar unit.  Same
#                  change is mirrored in the plugin dialog.
#
# %%%% 0.0.15: Density-pct date labels + text-x dt_labels page variant.
# Date: 2026-05-16
#                * v0.0.14 dropped the SetDataDateTime path entirely.
#                  v0.0.15 generalises its "sparse vs full" boolean into
#                  an integer percentage 0..100 (``datetime_label_density_pct``)
#                  -- 0 = no labels, 100 = one label per finite point,
#                  10 (default) approximates the old sparse behaviour.
#                  The legacy ``datetime_full_labels`` boolean is still
#                  accepted as a back-compat shim (True -> 100,
#                  False -> 10).
#                * NEW dt page variant: ``<base>_<hdu>_dt_labels`` and
#                  the overlay counterpart ``Overlay_<unit>_dt_labels``.
#                  These pages use a per-point TEXT dataset as xData and
#                  set the x axis to ``mode='labels'`` so the axis itself
#                  renders the date strings.  Sample spacing is uniform
#                  (one tick per point), NOT proportional to elapsed
#                  time -- gaps in time disappear, which is exactly what
#                  the user wanted for visually compact dt views.
#                * GUI: the v0.0.14 "Use full per-point date labels"
#                  checkbox is replaced by a 0..100 percentage spinbox
#                  plus two checkboxes ("Emit numeric-x dt page" and
#                  "Emit text-x dt_labels page").  Each variant can be
#                  toggled independently.  Same controls land in the
#                  plugin dialog FieldInt + FieldBool fields.
#                * GPU + parallelization re-audit: the new text-x page
#                  reuses the same GPU-sorted permutation as the
#                  numeric-x page (gpu_argsort already applied at the
#                  single sort site per record).  No new sort hotspots
#                  were introduced; mjd_to_datestr fast path (v0.0.14)
#                  remains the dominant per-element cost on huge traces
#                  and is still vectorised for the canonical format.
#                  Worker-thread parallelization of file reads is
#                  untouched.
#                * Revision history blocks are kept in DESCENDING
#                  semantic-version order across every shared file.
#
# %%%% 0.0.14: Datetime via xy.labels per-point text labels.
# Date: 2026-05-16
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
#                  is bound by default.  Users can swap variants in the
#                  Veusz GUI without rebuilding the document.
#                * Per-trace style sub-group ``xy.Label`` is set to
#                  angle=45deg, size=6pt, posnVert=centre, posnHorz=left
#                  by style_xy_datetime_labels(), so the labels visually
#                  mimic rotated datetime tick labels.
#                * dt-page x axis stays NUMERIC -- style_datetime_x_axis
#                  is no longer called on the dt-page x axis.  The MJD
#                  -> seconds conversion is still done so the
#                  x-ordering on the dt page matches the seconds page.

# %%%% 0.0.13: Identity-stable trace styling + datetime-duplicate hardening.
# Date: 2026-05-16
#                * Every xy widget (tagged per-file, untagged per-file,
#                  datetime-duplicate, and unit-overlay) is now styled by
#                  apply_trace_style() with identity key (column_name,
#                  tag_tuple).  The same (column, channel-pair) trace
#                  keeps the SAME colour and line style on every page it
#                  appears on -- so a given DELTAT trace for
#                  (CHA1, CHB2) looks identical on the per-file page, on
#                  the datetime-duplicate page, and on the cross-file
#                  unit-overlay -- making it easy to follow a single
#                  channel across the full document.
#                * Line is shown by default with a 1pt width; markers
#                  shrink to 1pt.  Line style cycles only when a graph
#                  carries more than 16 traces, so the typical NRAO
#                  1PPS case (a handful of channel pairs) stays fully
#                  solid.
#                * SetDataDateTime calls now go through
#                  set_datetime_dataset(), which coerces every value to
#                  a plain Python float (NaN/inf -> 0.0) and logs the
#                  outcome.  Plugged in at every dt emission site:
#                  per-tag dt, untagged dt fallback, and overlay dt.

# %%%% 0.0.12: Combined-in-time overlay semantics.  build_unit_overlay_pages
# Date: 2026-05-16
#              now collapses each (unit, column) onto a single time-sorted
#              xy trace stitched across every file that contributes that
#              column.  The legend names the column (e.g. 'DELTAT'); the
#              file boundaries disappear into the time series.  The
#              datetime-axis duplicate overlay does the same with the
#              concatenated time array converted to Veusz seconds.
#              Channel-tag row model: text columns named in
#              TAG_COLUMN_HINTS (e.g. CHANNELA, CHANNELB) are treated as
#              row-tags that identify which channel-pair each row of
#              numeric data belongs to.  They are NOT plotted as series.
#              Instead each numeric column (e.g. DELTAT) is split into
#              one trace per unique tag-tuple (e.g. ('CHA1','CHB2'))
#              both on per-file pages and on cross-file overlay pages.

# %%%% 0.0.11: Duplicate plots with a date-time x axis.  A new GUI
# Date: 2026-05-16
#              checkbox ("Duplicate plots with datetime X axis") asks
#              every per-file page and every cross-file unit-overlay
#              page to be cloned with a parallel Veusz date-time x
#              dataset (seconds since 2009-01-01 UTC).  The clones use
#              the ``YYYY-MM-DD HH:MM:SS`` tick label format (Veusz
#              %VDx tokens) rotated 45 degrees, so the cadence sits at
#              a readable density.  The original numeric (MJD) pages
#              are unchanged.
#                * push_to_veusz() and _build_pages() take the new
#                  ``datetime_duplicate`` kwarg.  The duplicate-pages
#                  pass runs after the original page so the project
#                  tree shows them side-by-side.
#                * build_unit_overlay_pages() also gained
#                  ``datetime_duplicate``; the cross-file overlay is
#                  cloned once per unit using the same dt datasets.
#                * The date-time companion dataset is emitted with the
#                  suffix ``__dt`` (sorted by the same sort key); it is
#                  built only when the sort key is an MJD-flavoured
#                  column (DMJD, MJD, JD-with-2400000.5-correction).
#                  TIME / TIMESTAMP columns are skipped with a log note
#                  because their epoch is ambiguous.

# %%%% 0.0.10: Optional GPU acceleration for the per-file argsort step.
# Date: 2026-05-16
#              When the 'Use GPU acceleration (CuPy)' checkbox is
#              checked and CuPy is importable on the host (Windows or
#              Linux), large time-axis sorts are dispatched to
#              ``gpu_argsort`` from ``_autoplot_common`` -- typically
#              2.7-10x faster than NumPy on consumer GPUs once N exceeds
#              ~200k samples.  Smaller files still use NumPy.  CuPy is a
#              soft dependency: the checkbox is disabled and tooltipped
#              when CuPy is absent.

# %%%% 0.0.9: Per-plot broken-x-axis on time gaps + unit-overlay pages.
# Date: 2026-05-16
#             FITSProcessor.read() now also captures per-column unit
#             strings (qt[col].unit if set, else TUNIT header), exposed
#             via data['units'][(hdu_name, col_name)].  push_to_veusz()
#             and _build_pages() take new keyword args ``plot_individual``
#             (default True), ``gap_k`` (default 10.0), and
#             ``gap_absolute`` (default 0.0); per-file pages now install a
#             native ``axis-broken`` X axis whenever ``detect_time_breaks``
#             returns non-empty pairs.  Added ``build_unit_overlay_pages``
#             which post-builds one page per distinct unit string with a
#             single graph overlaying every (file, column) matching that
#             unit -- broken X axis is computed from the combined x sample
#             distribution across all contributing files.  GUI gained a
#             ``Gap K (× median Δt)`` spin, an ``Absolute gap`` override
#             spin (0 = auto), and a ``Combined plots only`` checkbox that
#             suppresses per-file pages but keeps the overlays.
#               * MAX_THREADS default bumped from ``os.cpu_count() or 4``
#                 to ``(os.cpu_count() or 4) * 2``.  The per-file read path
#                 (astropy.io.fits + numpy memmap) is I/O bound and releases
#                 the GIL through both libraries; oversubscribing the worker
#                 pool gives a measurable speedup on slow / network storage.
#                 The Veusz-push phase remains single-threaded by design --
#                 ``veusz.embed.Embedded`` is documented as not thread-safe.
#               * 'Open in Veusz...' button inherited from the base class
#                 AutoPlotMainWindow is wired up by calling
#                 ``self.mark_project_saved(written)`` after save_vszh5()
#                 returns; this enables the button and lets the user launch
#                 the full standalone Veusz GUI in the current Python env
#                 (``python -m veusz <fn>``) without leaving the AutoPlot
#                 session.
# %%%%% Function Descriptions

# %%%% 0.0.8: Parallelization audit + Open-in-Veusz button.
# Date: 2026-05-16

# %%%% 0.0.7: Added two new progress bars in the GUI -- a 'Parsing/pushing'
# Date: 2026-05-16
#             file-level bar that ticks once per file as push_to_veusz()
#             finishes that file (separate from the original 'Reading'
#             worker-thread bar), and a 'Current file - columns' bar that
#             ticks per Veusz dataset (raw, sorted, image, datestr) as the
#             file's columns are poured into the document. push_to_veusz()
#             gained a ``column_cb(done, total)`` parameter and counts its
#             own operations so callers don't have to.
#             Also added a 'Skip image HDUs' checkbox to the options form
#             (and a ``skip_images`` field on FITSProcessor and a kwarg on
#             push_to_veusz) that completely bypasses the image-HDU read,
#             push and page-build paths.  Useful for NRAO 1PPS-delta
#             files where image HDUs are unused -- measurable speedup on
#             large batches.  (No 0.0.6: aligned with sibling plugin
#             version stream.)

# %%%% 0.0.5: Explicit early-exit in push_to_veusz() and _build_pages()
# Date: 2026-05-16
#             when ``data['images']`` is empty (the normal case for NRAO
#             1PPS-delta files, whose only HDU is OnePpsDeltas). Adds an
#             explicit log line so it's obvious image processing was
#             skipped on purpose rather than hanging.  Also calls
#             QApplication.processEvents() between every per-file push
#             inside _on_worker_done() so a 900-file batch insert no
#             longer makes the GUI look frozen.

# %%%% 0.0.4: Register the NRAO non-standard FITS unit aliases (``'none'``
# Date: 2026-05-16
#             on CHANNELA/CHANNELB, ``'NanoSeconds'`` on DELTAT) at module
#             load via register_nrao_fits_units(), and wrap fits.open /
#             QTable.read in suppress_fits_unit_warnings() so a 900-file
#             batch no longer floods the log with harmless UnitsWarning
#             and "kept as MaskedColumn" messages.

# %%%% 0.0.3: NaN values in numeric columns are now explicitly preserved as
# Date: 2026-05-16
#             NaN floats in Veusz numeric datasets (Veusz handles NaN
#             natively; samples are skipped only at plot time, not dropped
#             from the dataset).  The corresponding date-string text
#             datasets use the sentinel ``"NaN"`` for non-finite MJDs so
#             array lengths stay aligned with their numeric companions.

# %%%% 0.0.2: Added optional MJD -> YYYY-MM-DD_HH:MM:SS string conversion.
# Date: 2026-05-16
#             A new "Generate MJD->date strings" checkbox appears under the
#             Processing Options form.  When ticked, every numeric column
#             whose name matches an MJD/JD/TIME hint produces an additional
#             Veusz text dataset suffixed ``__datestr`` (also tagged with
#             the file's base name plus ``datestr``).

# %%%% 0.0.1: Initial port of FITS_AutoPlot to the Touchstone-style GUI
# Date: 2026-05-16

        main: build QApplication, open AutoPlot main window, run event loop.
        FITSAutoPlotWindow: qtpy main window subclassing AutoPlotMainWindow;
            adds the FITS-specific processing-options form (import backend
            choice, thread count, RSS high-water mark) and orchestrates the
            batch run through a worker QThread.
        FITSBatchWorker: QThread that runs the parallel batch read using
            ``run_in_threadpool``; emits per-file progress signals.
        FITSProcessor: per-file reader.  Supports both Veusz native FITS
            import and astropy fallback; transparently handles ``.fits`` and
            ``.fits.gz`` files; produces a normalised dict of column-shaped
            arrays keyed by ``(hdu_name, column_name)`` or ``(hdu_name,
            "IMAGE")`` for image HDUs.
        push_to_veusz: install one file's worth of column datasets into the
            running embedded Veusz document, with raw/sorted tagging and
            per-HDU plot creation.
# %%%%% Variable Descriptions
        MAX_THREADS: top-of-file knob for the worker thread pool size.
        DEFAULT_RSS_HIGH_WATER_MB: RSS threshold for memmap spill.
        SORTED_KEY_HINT: ordered list of preferred sort keys (DMJD first).
# %%%%% More Info
        FITS files from NRAO Green Bank carry a single BinTableHDU named
        ``OnePpsDeltas`` with columns DMJD (1D), CHANNELA (8A), CHANNELB
        (8A), DELTAT (J).  We expose every column as its own Veusz dataset
        plus an additional "sorted" pass (sorted by DMJD) so derived plots
        match the convention used in the NRAO 1PPS documentation.  The
        sorted datasets are produced from the raw arrays inside Python
        because Veusz's internal expressions cannot reorder a structured
        BinTable on import; the raw datasets are still imported so the
        user can pivot back to the unsorted representation if desired.
=============================================================================
"""
# %% Imports
from __future__ import annotations

# ============================================================================
# %%% IMPORTS - Standard library
# ============================================================================
import gzip
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# %%% IMPORTS - Scientific
# ============================================================================
import numpy as np
from astropy.io import fits
from astropy.table import QTable

# ============================================================================
# %%% IMPORTS - Veusz embedded (deferred to runtime in workers, OK at module
#           level here because the standalone GUI is also a Veusz client)
# ============================================================================
import veusz.embed as vz_embed

# ============================================================================
# %%% IMPORTS - shared GUI / threading / cache helpers
# ============================================================================
# Allow this file to be launched directly even if _autoplot_common.py lives
# in the same folder but the folder is not on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _autoplot_common import (   # noqa: E402
    Qt, QThread, Signal,
    QApplication, QFileDialog, QFormLayout, QLabel, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit,
    AutoPlotMainWindow,
    MemoryAwareCache, MemoryMonitor, MemoryMonitorConfig,
    apply_theme, open_embedded, save_vszh5,
    run_in_threadpool, open_maybe_gzipped, safe_dsname,
    gpu_argsort, is_gpu_available, enable_gpu, gpu_backend_name,
    set_gpu_argsort_threshold,
    mjd_to_datestr,
    register_nrao_fits_units, suppress_fits_unit_warnings,
    detect_time_breaks, make_broken_x_axis,
    # v0.0.18: unit-aware break detection (manual gap units fix)
    detect_time_breaks_unit_aware, column_unit_factor_from_day,
    mjd_to_veusz_seconds, style_datetime_x_axis,
    set_datetime_dataset,
    # v0.0.13: identity-stable trace styling
    apply_trace_style, TRACE_STYLE_VARY_THRESHOLD,
    # v0.0.14: per-point datetime labels (bypasses SetDataDateTime bug)
    build_sparse_datestr_dataset, build_full_datestr_dataset,
    build_density_datestr_dataset, build_textx_dataset,
    style_xy_datetime_labels, configure_axis_labels_mode,
    # v0.0.16: dt_labels via mode='datetime' + broken-axis parity
    build_dtnum_dataset, configure_axis_datetime_mode,
    mjd_break_pairs_to_dtsec,
    # v0.0.17: minimized .vszh5 save + sentinel channel-tag dt overlay filter
    save_vszh5_minimized, is_sentinel_tag_tuple,
    # v0.0.19: string gap parser ('5m3d2h' style)
    parse_gap_string,
    MJD_VEUSZ_EPOCH_MJD,
    DEFAULT_DATETIME_TICK_FORMAT, DEFAULT_DATETIME_TICK_ROTATE_DEG,
    DEFAULT_DATETIME_MAJOR_TICKS_TARGET,
)

# v0.0.11: list of FITS sort-key column names that we treat as MJD-valued
# for the purposes of building a Veusz date-time companion dataset.  JD is
# included separately because it needs a -2400000.5 offset before the
# standard MJD->Veusz-seconds conversion.
MJD_LIKE_SORT_KEYS = ("DMJD", "MJD")
JD_LIKE_SORT_KEYS = ("JD",)

# Register the NRAO non-standard FITS unit aliases ('none', 'NanoSeconds')
# once at module load so QTable.read no longer emits UnitsWarning when
# batch-processing 1PPS-delta files.
register_nrao_fits_units()

# ============================================================================
# %% SCRIPT-LEVEL KNOBS  (kept at the top per spec)
# ============================================================================
# I/O-bound default: FITS reads spend most wall-clock time in disk I/O
# (memmap + numpy array creation both release the GIL), so we get a real
# speedup from oversubscribing the CPU.  The GUI spin box caps at
# cpu_count*2; the default now matches that cap.
MAX_THREADS = max(1, (os.cpu_count() or 4) * 2)   # used by ThreadPoolExecutor
DEFAULT_RSS_HIGH_WATER_MB = 1024              # spill threshold per process
DEFAULT_BACKEND = "both"                      # 'veusz' | 'astropy' | 'both'
SORTED_KEY_HINT = ["DMJD", "MJD", "TIME", "TIMESTAMP", "JD"]
ALLOWED_EXT = (".fits", ".fit", ".fits.gz", ".fit.gz")
# v0.0.12: Column names that should be treated as ROW-TAGS rather than
# their own plottable series.  NRAO 1PPS-delta files use CHANNELA /
# CHANNELB to identify which sampler-pair each row's DELTAT belongs to.
# Any text-typed column whose name (case-insensitive) appears here is
# combined with the others -- in declaration order -- into a per-row
# label tuple, and the actual numeric measurement columns (e.g. DELTAT)
# are then split per unique label tuple into one trace each.
TAG_COLUMN_HINTS = ["CHANNELA", "CHANNELB", "CHANNEL", "SAMPLER", "PORT"]


# ============================================================================
# %% PER-FILE PROCESSOR
# ============================================================================
class FITSProcessor:
    """
    Read a single FITS file (compressed or not) and return a normalised dict
    of column-shaped numpy arrays plus minimal metadata.

    Two backends are supported:

        * ``astropy``  -- uses ``astropy.io.fits``; works for any HDU.
        * ``veusz``    -- uses ``veusz.embed.Embedded.ImportFileFITS`` which
                          imports each column as its own dataset.  We mirror
                          the resulting column data back into the same dict
                          structure so the caller can attach raw/sorted
                          datasets uniformly.

    The two backends are not redundant: ``veusz`` keeps the link to the
    original FITS file (if requested) so the project file follows updates
    to the source; ``astropy`` is the fallback path for image HDUs and for
    NRAO files where Veusz's importer skips header-only HDUs.

    Per the spec, when Veusz's internal import does not yield the desired
    column representation we additionally produce the column arrays
    ourselves via astropy and tag them as 'sorted'.
    """

    def __init__(self, backend: str, cache: MemoryAwareCache,
                 linked: bool = False, skip_images: bool = False) -> None:
        if backend not in ("veusz", "astropy", "both"):
            raise ValueError("backend must be 'veusz', 'astropy' or 'both'")
        self.backend = backend
        self.cache = cache
        self.linked = linked
        # When True, skip the image-HDU read/store path entirely.  Useful
        # for NRAO 1PPS-delta files where the image HDUs are not used and
        # the user just wants the OnePpsDeltas BinTableHDU columns.
        self.skip_images = bool(skip_images)

    # ------------------------------------------------------------------
    def _open(self, path: str):
        # astropy.io.fits transparently opens .gz, but we still need a
        # local copy on disk for Veusz's importer if the user picked the
        # veusz backend on a gzipped file (Veusz cannot open .fits.gz).
        if path.lower().endswith(".gz"):
            tmp = tempfile.NamedTemporaryFile(prefix="autoplot_fits_",
                                              suffix=".fits", delete=False)
            tmp.close()
            with gzip.open(path, "rb") as src, open(tmp.name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return tmp.name, True
        return path, False

    # ------------------------------------------------------------------
    def read(self, path: str) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:

            'columns'     : {(hdu_name, col_name) : np.ndarray}  -- 1D table cols
            'images'      : {hdu_name             : np.ndarray}  -- nD images
            'sort_key'    : (hdu_name, col_name) or None
            'header'      : list[str]  -- pretty-printed PRIMARY header lines
            'fits_for_vz' : path to a non-gzipped FITS file usable by Veusz
            'tmp_uncompressed' : bool
        """
        local_path, did_decompress = self._open(path)
        base = os.path.splitext(os.path.basename(path))[0]
        if base.endswith(".fits"):
            base = base[:-5]
        out = {
            "columns": {},
            "images": {},
            # (hdu_name, col_name) -> unit string ("" if no TUNITn declared).
            # Used by FITSAutoPlotWindow to group columns into unit-overlay
            # pages (one extra page per distinct unit string).
            "units": {},
            "sort_key": None,
            # v0.0.12: row-tagging structures populated below per HDU.
            #   tag_columns:  {hname: [tag_col_name, ...]}  declaration order
            #   tag_groups:   {hname: {tag_tuple: int64 row index array}}
            # Numeric measurement columns are SPLIT per tag_tuple downstream.
            "tag_columns": {},
            "tag_groups": {},
            "header": [],
            "fits_for_vz": local_path,
            "tmp_uncompressed": did_decompress,
            "base_name": base,
        }
        with suppress_fits_unit_warnings(), \
                fits.open(local_path, memmap=True) as hdul:
            # Collect a compact header dump from PRIMARY
            try:
                out["header"] = [
                    "%s = %r" % (k, hdul[0].header[k])
                    for k in list(hdul[0].header.keys())[:20]
                ]
            except Exception:
                pass

            sort_key: Optional[Tuple[str, str]] = None
            for i, hdu in enumerate(hdul):
                hname = hdu.name or "HDU%d" % i
                if hdu.data is None:
                    continue
                if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
                    # suppress_fits_unit_warnings() above already covers
                    # QTable.read; this nested call is harmless and ensures
                    # the filter is active even if a subclass changes the
                    # outer context (defensive).
                    with suppress_fits_unit_warnings():
                        qt = QTable.read(hdu)
                    for col in qt.colnames:
                        try:
                            data = np.asarray(qt[col])
                        except Exception:
                            data = np.asarray(hdu.data[col])
                        # store via cache (will memmap-spill if needed)
                        key = (hname, col)
                        out["columns"][key] = self.cache.store(
                            "%s.%s.%s" % (base, hname, col), data
                        )
                        # Capture the unit string for this column when astropy
                        # exposes one (Quantity columns have ``.unit``); fall
                        # back to the bare ``TUNITn`` header keyword.  An empty
                        # string means 'no declared unit' and groups all such
                        # columns into a single 'dimensionless' overlay page.
                        unit_str = ""
                        try:
                            u = getattr(qt[col], "unit", None)
                            if u is not None:
                                unit_str = str(u)
                        except Exception:
                            pass
                        if not unit_str:
                            try:
                                # TUNITn keyword (1-based column index)
                                col_idx = qt.colnames.index(col) + 1
                                unit_str = str(
                                    hdu.header.get("TUNIT%d" % col_idx, "") or ""
                                ).strip()
                            except Exception:
                                unit_str = ""
                        out["units"][key] = unit_str
                        if sort_key is None:
                            for hint in SORTED_KEY_HINT:
                                if col.upper() == hint:
                                    sort_key = (hname, col)
                                    break
                    # v0.0.12: identify ROW-TAG columns in this HDU.  A
                    # tag column is any column whose name (uppercased) is
                    # in TAG_COLUMN_HINTS and whose dtype is text-like
                    # (S/U/O).  Build a per-row tag tuple and group row
                    # indices by that tuple.
                    tag_cols = []
                    for col in qt.colnames:
                        if col.upper() not in [
                            t.upper() for t in TAG_COLUMN_HINTS
                        ]:
                            continue
                        try:
                            tarr = np.asarray(out["columns"][(hname, col)])
                        except Exception:
                            continue
                        if tarr.dtype.kind not in ("S", "U", "O"):
                            continue
                        tag_cols.append(col)
                    if tag_cols:
                        out["tag_columns"][hname] = list(tag_cols)
                        # Build a row-aligned 2D list of decoded strings.
                        decoded = []
                        n_rows = None
                        for col in tag_cols:
                            tarr = np.asarray(out["columns"][(hname, col)])
                            vals = [
                                v.decode("ascii", "replace").strip()
                                if isinstance(v, bytes)
                                else str(v).strip()
                                for v in tarr
                            ]
                            if n_rows is None:
                                n_rows = len(vals)
                            elif len(vals) != n_rows:
                                # Length mismatch -- abandon tag grouping
                                # for this HDU rather than emit broken data.
                                decoded = []
                                break
                            decoded.append(vals)
                        groups = {}
                        if decoded and n_rows:
                            for row_idx in range(n_rows):
                                tup = tuple(
                                    decoded[ci][row_idx]
                                    for ci in range(len(decoded))
                                )
                                groups.setdefault(tup, []).append(row_idx)
                            # Materialise as int64 arrays for fast fancy
                            # indexing later.
                            groups = {
                                k: np.asarray(v, dtype=np.int64)
                                for k, v in groups.items()
                            }
                        out["tag_groups"][hname] = groups
                else:
                    # Image / primary array HDU.  Honor the user-requested
                    # skip_images flag: when set, we don't even read the
                    # array off disk -- a big win for large image-bearing
                    # FITS files when the user only cares about table HDUs.
                    if self.skip_images:
                        continue
                    data = np.asarray(hdu.data)
                    out["images"][hname] = self.cache.store(
                        "%s.%s.IMAGE" % (base, hname), data
                    )
            out["sort_key"] = sort_key
        return out


# ============================================================================
# %% VEUSZ INTEGRATION  (runs on the GUI thread)
# ============================================================================
def push_to_veusz(doc, file_path: str, data: Dict[str, Any],
                  backend: str, log_cb=None,
                  emit_datestr: bool = False,
                  column_cb=None,
                  skip_images: bool = False,
                  plot_individual: bool = True,
                  gap_k: float = 10.0,
                  gap_absolute: float = 0.0,
                  datetime_duplicate: bool = False,
                  datetime_full_labels=None,
                  datetime_label_density_pct: int = 10,
                  datetime_emit_numeric_dt: bool = True,
                  datetime_emit_text_dt: bool = True,
                  hide_keys: bool = False) -> None:
    """
    Push a single processed FITS file into the running embedded Veusz
    document.

    v0.0.19: ``hide_keys`` -- when True every key (legend) widget added
    by ``_build_pages`` and downstream helpers is created with its
    Border hidden so the legend does not draw.

    Strategy
    --------
    1.  If backend in ('veusz', 'both'): call ``ImportFileFITS`` so the
        raw FITS columns/images appear as Veusz datasets, tagged with the
        base filename + 'raw'.  This keeps the Veusz-internal link to the
        FITS file when ``linked=True``.
    2.  If backend in ('astropy', 'both') or the FITS file was gzipped:
        push python-level numpy column arrays as their own Veusz datasets,
        tagged with the base filename + 'sorted' (sorted by DMJD when the
        column is present, otherwise plain copies).
    3.  Build one Veusz page per HDU containing scatter plots for table
        HDUs (DELTAT vs DMJD, etc.) and image widgets for image HDUs.

    v0.0.15 parameters
    ------------------
    datetime_label_density_pct : int
        Percentage (0-100) of trace samples that get a date-string
        label.  0 = no labels, 100 = a label at every point.  Replaces
        the v0.0.14 boolean ``datetime_full_labels``.  See
        :func:`_autoplot_common.build_density_datestr_dataset`.
    datetime_emit_numeric_dt : bool
        Emit the v0.0.14 dt page (numeric seconds x + xy.labels date
        annotations).  Default True.
    datetime_emit_text_dt : bool
        Emit the v0.0.15 dt_labels page variant (xData bound to the
        text dataset + axis mode='labels').  Uniform sample spacing,
        date strings ON the axis.  Default True.
    datetime_full_labels : bool or None
        Back-compat alias for v0.0.14 callers.  None (default) means
        use ``datetime_label_density_pct``.  True -> density_pct=100,
        False -> density_pct=10 (the v0.0.14 sparse default).
    """
    # ---- v0.0.15 back-compat shim for datetime_full_labels ---------------
    # If the caller passed the old boolean (e.g. an older plugin / script),
    # translate it into an equivalent density percentage.  Once translated,
    # the rest of this function only consults
    # ``datetime_label_density_pct``.
    if datetime_full_labels is not None:
        try:
            datetime_label_density_pct = (
                100 if bool(datetime_full_labels) else 10
            )
        except Exception:
            pass
    # Clip to [0, 100] defensively.
    try:
        datetime_label_density_pct = max(0, min(100,
            int(round(float(datetime_label_density_pct)))))
    except Exception:
        datetime_label_density_pct = 10

    base = safe_dsname(data["base_name"])
    raw_names: List[str] = []
    sorted_names: List[str] = []

    # ---- column-level progress book-keeping ------------------------------
    # Pre-count the operations the caller will see ticking on the column bar:
    # one tick per Veusz native dataset added (raw), one per sorted column,
    # one per image HDU, one per datestr emission.
    _n_raw = 0  # known only after ImportFileFITS returns
    _n_sorted = len(data.get("columns") or {})
    _n_images = 0 if skip_images else len(data.get("images") or {})
    _n_datestr = 0
    if emit_datestr:
        _hints = [k.upper() for k in SORTED_KEY_HINT]
        for (_h, _c) in (data.get("columns") or {}):
            if _c.upper() in _hints:
                _n_datestr += 1
    _col_total = _n_sorted + _n_images + _n_datestr  # raw added later
    _col_done = 0

    def _tick(n: int = 1) -> None:
        # Local helper so each successful sub-push advances the GUI bar by
        # one unit.  Safe no-op when the caller didn't supply a callback.
        nonlocal _col_done
        _col_done += n
        if column_cb is not None:
            try:
                column_cb(_col_done, _col_total)
            except Exception:
                pass

    # ---- 1) Veusz native import (raw datasets) ---------------------------
    if backend in ("veusz", "both") and not data["tmp_uncompressed"]:
        try:
            existing = set(doc.GetDatasets())
            doc.ImportFileFITS(
                filename=data["fits_for_vz"],
                items=["/"],          # import every HDU/column
                namemap={},
                linked=False,         # we already point at a real file
            )
            added = [n for n in doc.GetDatasets() if n not in existing]
            # extend the column-bar total to include the raw datasets we are
            # about to rename / tag, so the bar grows rather than overflowing
            _col_total_ref = _col_total + len(added)
            if column_cb is not None:
                try:
                    column_cb(_col_done, _col_total_ref)
                except Exception:
                    pass
            for n in added:
                # rename with base prefix so multi-file runs don't collide
                new_name = "%s__%s__raw" % (base, safe_dsname(n))
                try:
                    doc.RenameDataset(n, new_name)
                except Exception:
                    new_name = n
                raw_names.append(new_name)
                _col_done += 1
                if column_cb is not None:
                    try:
                        column_cb(_col_done, _col_total_ref)
                    except Exception:
                        pass
            doc.TagDatasets(base, raw_names)
            doc.TagDatasets("raw", raw_names)
            if log_cb:
                log_cb("  Veusz native import: %d raw datasets" % len(raw_names))
        except Exception as exc:
            if log_cb:
                log_cb("  Veusz native import failed: %s" % exc)

    # ---- 2) Astropy/column-oriented sorted datasets ----------------------
    # v0.0.12: when an HDU carries row-tag columns (e.g. CHANNELA, CHANNELB),
    # we no longer emit one ``__sorted`` dataset per numeric column.  Instead
    # we split each numeric column by tag-tuple and emit one
    # ``__{tagstr}__sorted`` dataset per (column, tag-tuple).  The tag
    # columns themselves are NOT emitted as plottable datasets -- they are
    # labels, not series.  ``per_tag_sorted_names`` records what was emitted
    # so _build_pages and the overlay builder can find it.  When no tags are
    # present for an HDU we fall back to the original per-column emission.
    if backend in ("astropy", "both") or backend == "veusz" and data["tmp_uncompressed"]:
        sort_key = data["sort_key"]
        sort_idx_by_hdu: Dict[str, np.ndarray] = {}
        if sort_key is not None:
            hkey, ckey = sort_key
            # v0.0.10: gpu_argsort uses CuPy on large arrays when enabled
            # by the GUI; otherwise falls through to NumPy.  Result is a
            # NumPy int64 index array either way.  v0.0.12: kept for the
            # un-tagged fallback path only; per-tag splits re-sort their
            # own subarrays below.
            sort_idx_by_hdu[hkey] = gpu_argsort(
                np.asarray(data["columns"][sort_key])
            )

        # Helper: stringify a tag tuple into a safe dataset suffix.
        def _tag_suffix(tup):
            if not tup:
                return ""
            return "_".join(safe_dsname(t or "NA") for t in tup)

        # Track per-tag dataset names per HDU so downstream stages can
        # discover them.  Schema:
        #   {hname: {tag_tuple: {col_name: ds_name_for_that_col_and_tag,
        #                        "__x__": x_ds_name,
        #                        "__dt__": dt_ds_name_or_None}}}
        per_tag_sorted_names = {}
        # Untagged-fallback HDUs use the same shape but with key None.
        for (hname, col), arr in data["columns"].items():
            arr = np.asarray(arr)
            # ----- tagged path --------------------------------------------
            tag_groups = data.get("tag_groups", {}).get(hname) or {}
            tag_cols = data.get("tag_columns", {}).get(hname) or []
            if tag_groups:
                # Skip tag columns themselves -- they are row labels.
                if col in tag_cols:
                    _tick()
                    continue
                # Skip non-numeric columns under the tag path: they cannot
                # be split into per-tag y-series (Veusz xy widgets expect
                # numeric y).
                if arr.dtype.kind not in ("f", "i", "u"):
                    _tick()
                    continue
                hkey_safe = safe_dsname(hname)
                col_safe = safe_dsname(col)
                # The time axis (sort_key) gets one per-tag x dataset and,
                # when requested, one per-tag datetime companion.  Other
                # numeric columns get a per-tag y dataset that is sorted by
                # the tag's x permutation so x/y stay aligned row-for-row.
                is_xcol = (sort_key is not None
                           and (hname, col) == sort_key)
                # Pre-compute the time-axis sort permutations per tag once
                # per HDU (cached on the function's enclosing dict).
                if hname not in per_tag_sorted_names:
                    per_tag_sorted_names[hname] = {}
                    if sort_key is not None and sort_key[0] == hname:
                        try:
                            x_full = np.asarray(
                                data["columns"][sort_key], dtype=float
                            )
                        except Exception:
                            x_full = None
                    else:
                        x_full = None
                    # Stash the cached subarrays + their sort indices on
                    # the outer per_tag_sorted_names dict via a sentinel
                    # key so the loop below can reuse them.
                    per_tag_sorted_names[hname]["__xcache__"] = {
                        "x_full": x_full,
                        "sort_idx": {},
                    }
                xcache = per_tag_sorted_names[hname]["__xcache__"]
                x_full = xcache["x_full"]
                sort_idx_per_tag = xcache["sort_idx"]
                # Build per-tag sort index lazily and only once.
                for tup, row_idx in tag_groups.items():
                    if tup not in sort_idx_per_tag and x_full is not None:
                        try:
                            xs = x_full[row_idx]
                            order = gpu_argsort(xs)
                            sort_idx_per_tag[tup] = (row_idx, order)
                        except Exception:
                            sort_idx_per_tag[tup] = (row_idx, None)
                    elif tup not in sort_idx_per_tag:
                        sort_idx_per_tag[tup] = (row_idx, None)
                # Emit one dataset per tag for this column.
                for tup, (row_idx, order) in sort_idx_per_tag.items():
                    tag_suffix = _tag_suffix(tup)
                    try:
                        sub = np.asarray(arr[row_idx], dtype=float)
                    except Exception:
                        continue
                    if order is not None:
                        try:
                            sub = sub[order]
                        except Exception:
                            pass
                    if is_xcol:
                        ds_name = "%s__%s__%s__%s__x__sorted" % (
                            base, hkey_safe, col_safe, tag_suffix
                        )
                    else:
                        ds_name = "%s__%s__%s__%s__sorted" % (
                            base, hkey_safe, col_safe, tag_suffix
                        )
                    try:
                        doc.SetData(
                            ds_name,
                            np.ascontiguousarray(sub, dtype=float),
                        )
                    except Exception as exc:
                        if log_cb:
                            log_cb("  SetData failed for %s: %s"
                                   % (ds_name, exc))
                        continue
                    sorted_names.append(ds_name)
                    bucket = per_tag_sorted_names[hname].setdefault(
                        tup, {}
                    )
                    if is_xcol:
                        bucket["__x__"] = ds_name
                    else:
                        bucket[col] = ds_name
                _tick()
                continue
            # ----- untagged fallback path --------------------------------
            if hname in sort_idx_by_hdu:
                arr_s = arr[sort_idx_by_hdu[hname]]
            else:
                arr_s = arr
            ds_name = "%s__%s__%s__sorted" % (
                base, safe_dsname(hname), safe_dsname(col)
            )
            # SetData refuses to accept bytes columns; convert to str list
            if arr_s.dtype.kind in ("S", "U", "O"):
                arr_s = np.asarray([
                    v.decode("ascii", "replace") if isinstance(v, bytes) else str(v)
                    for v in arr_s
                ])
                try:
                    doc.SetDataText(ds_name, list(arr_s))
                except Exception:
                    # Older Veusz: drop text columns silently
                    continue
            else:
                # NaN-preserving: Veusz natively supports NaN in numeric
                # datasets, so we push the float array through unchanged.
                # Samples with NaN are simply skipped by xy plots; the row
                # is NOT removed from the dataset.
                doc.SetData(ds_name, np.ascontiguousarray(arr_s, dtype=float))
            sorted_names.append(ds_name)
            _tick()
        # Stash the per-tag map on the data dict so _build_pages and
        # the overlay builder can locate the per-tag datasets without
        # re-deriving the names.
        data["_per_tag_sorted_names"] = per_tag_sorted_names

        # Image HDU push (skip explicitly when there are no images so the
        # log makes it obvious to the user that nothing image-shaped was
        # found in this file -- NRAO 1PPS-delta files contain only the
        # OnePpsDeltas BinTableHDU, so 'images' is normally empty).
        # Also honor the user-requested skip_images flag.
        if skip_images:
            if log_cb:
                log_cb("  Image HDUs skipped (user requested).")
            images_dict = {}
        else:
            images_dict = data.get("images") or {}
        if not images_dict:
            if log_cb:
                log_cb("  No image HDUs in this file -- skipping image push.")
        else:
            for hname, img in images_dict.items():
                img = np.asarray(img)
                ds_name = "%s__%s__IMAGE" % (base, safe_dsname(hname))
                try:
                    doc.SetData2D(ds_name, np.ascontiguousarray(img, dtype=float))
                except Exception:
                    doc.SetData(
                        ds_name,
                        np.ascontiguousarray(img.ravel(), dtype=float),
                    )
                sorted_names.append(ds_name)
                _tick()

        # ---- 2a) Optional datetime label datasets (v0.0.14) ---------------
        # When the user asked for datetime-duplicate plots, build a pair
        # of TEXT datasets (sparse + full) of YYYY-MM-DD HH:MM:SS strings
        # from the file's MJD/JD-like sort key.  The dt-duplicate page
        # then keeps the SAME numeric seconds x binding as the seconds
        # page (which works) and binds ``xy.labels.val`` to one of these
        # text datasets so date annotations float along each trace.
        # This replaces the v0.0.11..0.0.13 SetDataDateTime path which
        # raises 'unsupported operand type(s) for -: float and
        # datetime.datetime' inside Veusz 3.4 internals on the user's
        # real host.
        #
        # ``datetime_x_name`` is retained as the single-untagged HDU
        # signal that downstream pages need a dt clone.  v0.0.15: when
        # tagged, the per-tag bucket carries ``__datelabels_density__``
        # (single text dataset whose density matches the user's pct
        # slider) and ``__datelabels_textx__`` (the per-point text
        # dataset that serves as xData on the new dt_labels page).
        datetime_x_name = None  # signal flag for downstream dt-page clone
        if datetime_duplicate and log_cb:
            log_cb("  Datetime-duplicate requested (sort_key=%s, "
                   "density=%d%%, emit_numeric=%s, emit_text=%s)"
                   % (sort_key, datetime_label_density_pct,
                      bool(datetime_emit_numeric_dt),
                      bool(datetime_emit_text_dt)))
        if datetime_duplicate and sort_key is None and log_cb:
            log_cb("  Datetime duplicate skipped: no sort_key for this file")
        if datetime_duplicate and sort_key is not None:
            sk_hdu, sk_col = sort_key
            upper_sk = sk_col.upper()
            arr_mjd = None
            if upper_sk in MJD_LIKE_SORT_KEYS:
                try:
                    arr_mjd = np.asarray(
                        data["columns"][sort_key], dtype=float
                    )
                except Exception:
                    arr_mjd = None
            elif upper_sk in JD_LIKE_SORT_KEYS:
                try:
                    arr_mjd = np.asarray(
                        data["columns"][sort_key], dtype=float
                    ) - 2400000.5
                except Exception:
                    arr_mjd = None
            if arr_mjd is not None:
                tag_groups_dt = (
                    data.get("tag_groups", {}).get(sk_hdu) or {}
                )
                if tag_groups_dt:
                    # Per-tag label datasets aligned with each tag's
                    # already-sorted y datasets.  We emit BOTH sparse
                    # and full variants so the user can swap between
                    # them in Veusz without re-running the pipeline.
                    xcache = (per_tag_sorted_names
                              .get(sk_hdu, {})
                              .get("__xcache__", {}))
                    sort_idx_per_tag = xcache.get("sort_idx", {})
                    n_built = 0
                    for tup, _row in tag_groups_dt.items():
                        rec = sort_idx_per_tag.get(tup)
                        if rec is None:
                            continue
                        row_idx, order = rec
                        try:
                            sub = arr_mjd[row_idx]
                            if order is not None:
                                sub = sub[order]
                            tag_suffix = "_".join(
                                safe_dsname(t or "NA") for t in tup
                            )
                            density_name = (
                                "%s__%s__%s__%s__datelabels_density" % (
                                    base, safe_dsname(sk_hdu),
                                    safe_dsname(sk_col), tag_suffix,
                                )
                            )
                            textx_name = (
                                "%s__%s__%s__%s__datelabels_textx" % (
                                    base, safe_dsname(sk_hdu),
                                    safe_dsname(sk_col), tag_suffix,
                                )
                            )
                            # v0.0.16: numeric Veusz-datetime-seconds
                            # dataset that drives the dt_labels page
                            # under axis mode='datetime' instead of the
                            # v0.0.15 text-x / labels-mode approach.
                            dtnum_name = (
                                "%s__%s__%s__%s__datelabels_dtnum" % (
                                    base, safe_dsname(sk_hdu),
                                    safe_dsname(sk_col), tag_suffix,
                                )
                            )
                            ok_density = build_density_datestr_dataset(
                                doc, density_name,
                                np.asarray(sub, dtype=float),
                                density_pct=datetime_label_density_pct,
                                log_cb=log_cb,
                            )
                            # v0.0.15: text-x dataset (per-point, no ""
                            # gaps).  Kept for back-compat -- the
                            # v0.0.16 dt_labels page does NOT bind it
                            # to xData anymore, but external code that
                            # imports build_textx_dataset still works.
                            ok_textx = None
                            if datetime_emit_text_dt:
                                ok_textx = build_textx_dataset(
                                    doc, textx_name,
                                    np.asarray(sub, dtype=float),
                                    log_cb=log_cb,
                                )
                            # v0.0.16: numeric Veusz-seconds xData for
                            # the dt_labels page (datetime-mode axis).
                            ok_dtnum = None
                            if datetime_emit_text_dt:
                                ok_dtnum = build_dtnum_dataset(
                                    doc, dtnum_name,
                                    np.asarray(sub, dtype=float),
                                    log_cb=log_cb,
                                )
                            # v0.0.16: stash the MJD-domain x source for
                            # per-tag broken-axis detection on the
                            # dt_labels page.  This is the same MJD
                            # subarray after the per-tag sort, so
                            # detect_time_breaks on it yields gap pairs
                            # in MJD space (convert via
                            # mjd_break_pairs_to_dtsec at render time).
                            mjd_for_breaks = np.asarray(sub, dtype=float)
                            if ok_density or ok_textx or ok_dtnum:
                                bucket = per_tag_sorted_names \
                                    .setdefault(sk_hdu, {}) \
                                    .setdefault(tup, {})
                                if ok_density:
                                    bucket["__datelabels_density__"] = (
                                        density_name
                                    )
                                    sorted_names.append(density_name)
                                if ok_textx:
                                    bucket["__datelabels_textx__"] = (
                                        textx_name
                                    )
                                    sorted_names.append(textx_name)
                                if ok_dtnum:
                                    bucket["__datelabels_dtnum__"] = (
                                        dtnum_name
                                    )
                                    sorted_names.append(dtnum_name)
                                    # Stash MJD source for v0.0.16
                                    # broken-axis parity on dt_labels.
                                    bucket["__datelabels_mjd__"] = (
                                        mjd_for_breaks
                                    )
                                # v0.0.14 keys kept for back-compat with
                                # any external code that read them.  We
                                # alias both to the single density ds.
                                if ok_density:
                                    bucket["__datelabels_sparse__"] = (
                                        density_name
                                    )
                                    bucket["__datelabels_full__"] = (
                                        density_name
                                    )
                                n_built += 1
                        except Exception as exc:
                            if log_cb:
                                log_cb(
                                    "  Datetime-label build failed for "
                                    "tag %s: %s" % (tup, exc)
                                )
                    if log_cb:
                        log_cb(
                            "  Datetime-label dataset pairs per tag: %d"
                            % n_built
                        )
                else:
                    # Apply the same sort permutation the sorted column
                    # got, so the per-point labels line up row-for-row
                    # with the y datasets and the numeric seconds x.
                    if sk_hdu in sort_idx_by_hdu:
                        try:
                            arr_mjd = arr_mjd[sort_idx_by_hdu[sk_hdu]]
                        except Exception:
                            pass
                    density_name = "%s__%s__%s__datelabels_density" % (
                        base, safe_dsname(sk_hdu), safe_dsname(sk_col)
                    )
                    textx_name = "%s__%s__%s__datelabels_textx" % (
                        base, safe_dsname(sk_hdu), safe_dsname(sk_col)
                    )
                    # v0.0.16: numeric Veusz-seconds dataset name.
                    dtnum_name = "%s__%s__%s__datelabels_dtnum" % (
                        base, safe_dsname(sk_hdu), safe_dsname(sk_col)
                    )
                    ok_density = build_density_datestr_dataset(
                        doc, density_name, arr_mjd,
                        density_pct=datetime_label_density_pct,
                        log_cb=log_cb,
                    )
                    ok_textx = None
                    if datetime_emit_text_dt:
                        ok_textx = build_textx_dataset(
                            doc, textx_name, arr_mjd, log_cb=log_cb,
                        )
                    # v0.0.16: numeric Veusz-seconds xData for the
                    # dt_labels page (datetime-mode axis).
                    ok_dtnum = None
                    if datetime_emit_text_dt:
                        ok_dtnum = build_dtnum_dataset(
                            doc, dtnum_name, arr_mjd, log_cb=log_cb,
                        )
                    if ok_density:
                        sorted_names.append(density_name)
                    if ok_textx:
                        sorted_names.append(textx_name)
                    if ok_dtnum:
                        sorted_names.append(dtnum_name)
                    if ok_density or ok_textx or ok_dtnum:
                        # v0.0.15: stash both names on the data record
                        # so _build_pages can clone the appropriate
                        # dt page variants.  v0.0.14 keys aliased to
                        # the density ds for back-compat.
                        # v0.0.16: stash the dtnum name + MJD source so
                        # the dt_labels page can render mode='datetime'
                        # and (if break_pairs is non-empty) install a
                        # broken axis.
                        data["_datelabels_density"] = (
                            density_name if ok_density else None
                        )
                        data["_datelabels_textx"] = (
                            textx_name if ok_textx else None
                        )
                        data["_datelabels_dtnum"] = (
                            dtnum_name if ok_dtnum else None
                        )
                        # Source MJDs for the per-trace break detection
                        # on the dt_labels page (mapped to Veusz seconds
                        # at render time).
                        try:
                            data["_datelabels_mjd"] = np.asarray(
                                arr_mjd, dtype=float
                            )
                        except Exception:
                            data["_datelabels_mjd"] = None
                        data["_datelabels_sparse"] = data["_datelabels_density"]
                        data["_datelabels_full"] = data["_datelabels_density"]
                        # Trigger the dt-page clone in _build_pages.
                        datetime_x_name = (
                            density_name if ok_density else textx_name
                        )
            elif log_cb:
                log_cb("  Datetime duplicate skipped: %s epoch unknown"
                       % sk_col)

        if sorted_names:
            doc.TagDatasets(base, sorted_names)
            doc.TagDatasets("sorted", sorted_names)
        if log_cb:
            log_cb("  Column datasets created: %d" % len(sorted_names))

    # ---- 2b) Optional MJD -> date-string text datasets -------------------
    if emit_datestr:
        datestr_names = []
        for (hname, col), arr in data["columns"].items():
            if col.upper() not in [k.upper() for k in SORTED_KEY_HINT]:
                continue
            arr = np.asarray(arr)
            if arr.dtype.kind not in ("f", "i", "u"):
                continue
            try:
                date_strings = mjd_to_datestr(arr.astype(float))
            except Exception as exc:
                if log_cb:
                    log_cb("  MJD->date conversion failed for %s.%s: %s"
                           % (hname, col, exc))
                continue
            ds_raw = "%s__%s__%s__datestr" % (base, safe_dsname(hname), safe_dsname(col))
            ds_sorted = "%s__%s__%s__datestr_sorted" % (
                base, safe_dsname(hname), safe_dsname(col)
            )
            try:
                doc.SetDataText(ds_raw, list(date_strings))
                idx = gpu_argsort(arr.astype(float))  # v0.0.10
                doc.SetDataText(ds_sorted, list(np.asarray(date_strings)[idx]))
            except Exception as exc:
                if log_cb:
                    log_cb("  SetDataText failed for %s: %s" % (ds_raw, exc))
                continue
            datestr_names.extend([ds_raw, ds_sorted])
        if datestr_names:
            try:
                doc.TagDatasets(base, datestr_names)
                doc.TagDatasets("datestr", datestr_names)
            except Exception:
                pass
            if log_cb:
                log_cb("  Date-string datasets created: %d" % len(datestr_names))

    # ---- 3) Build the per-file plot pages -------------------------------
    # When ``plot_individual`` is False, the user asked for combined
    # (unit-overlay) plots only -- skip building the per-file pages here.
    # The caller (FITSAutoPlotWindow) still builds the unit-overlay pages
    # at the end of the batch from accumulated metadata.
    if plot_individual:
        # When the user opted to skip image HDUs, pass a shallow-copied
        # data dict with an empty images map so _build_pages doesn't try
        # to render image pages for HDUs that were never read.
        # v0.0.11: forward datetime_x_name so _build_pages can clone each
        # per-HDU page against the date-time companion dataset.
        _dt_x = locals().get("datetime_x_name", None)
        if skip_images:
            _build_pages(doc, base, dict(data, images={}), sorted_names,
                         gap_k=gap_k, gap_absolute=gap_absolute,
                         datetime_x_name=_dt_x,
                         datetime_label_density_pct=datetime_label_density_pct,
                         datetime_emit_numeric_dt=datetime_emit_numeric_dt,
                         datetime_emit_text_dt=datetime_emit_text_dt,
                         hide_keys=hide_keys)
        else:
            _build_pages(doc, base, data, sorted_names,
                         gap_k=gap_k, gap_absolute=gap_absolute,
                         datetime_x_name=_dt_x,
                         datetime_label_density_pct=datetime_label_density_pct,
                         datetime_emit_numeric_dt=datetime_emit_numeric_dt,
                         datetime_emit_text_dt=datetime_emit_text_dt,
                         hide_keys=hide_keys)
    elif log_cb:
        log_cb("  Per-file plot pages skipped (combined plots only).")

    # cleanup any temporary uncompressed copy
    if data["tmp_uncompressed"]:
        try:
            os.remove(data["fits_for_vz"])
        except OSError:
            pass


def _build_pages(doc, base: str, data: Dict[str, Any],
                 sorted_names: List[str],
                 gap_k: float = 10.0,
                 gap_absolute: float = 0.0,
                 datetime_x_name=None,
                 datetime_full_labels=None,
                 datetime_label_density_pct: int = 10,
                 datetime_emit_numeric_dt: bool = True,
                 datetime_emit_text_dt: bool = True,
                 hide_keys: bool = False) -> None:
    """Create one Veusz page per HDU with the appropriate plot widgets.

    v0.0.19: ``hide_keys`` -- when True the legend border on every key
    widget is set to hidden, suppressing the legend frame.

    v0.0.15 dt-page variants
    ------------------------
    Two independent dt-page variants are now emitted per HDU when a
    sort_key with a known time epoch is available:

      * ``<base>_<hdu>_dt`` (numeric): numeric seconds x-axis,
        xy.labels carries a density-controlled date-string text
        dataset.  Gated by ``datetime_emit_numeric_dt`` (default True).
      * ``<base>_<hdu>_dt_labels`` (text-x): xData binds to the
        per-point text dataset; axis ``mode='labels'``.  Uniform sample
        spacing.  Gated by ``datetime_emit_text_dt`` (default True).

    The legacy ``datetime_full_labels`` boolean is accepted as a
    back-compat shim: ``True`` -> density_pct=100, ``False`` ->
    density_pct=10.

    v0.0.12: when an HDU has row-tag columns (CHANNELA/CHANNELB etc.), each
    numeric measurement column gets ONE graph with N xy widgets -- one trace
    per unique tag tuple -- using the per-tag x/y/dt datasets that
    push_to_veusz emitted.  The tag tuple is rendered in the legend (e.g.
    'CHA1.CHB2').  Untagged HDUs keep the original one-graph-per-column
    layout for backward compatibility.

    When the x-axis column has large gaps (per ``detect_time_breaks``), the
    plain ``x`` axis is replaced with an ``axis-broken`` widget.
    """
    # ---- v0.0.15 back-compat shim ---------------------------------------
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

    per_tag = data.get("_per_tag_sorted_names") or {}

    # Tagged HDUs are handled separately from the legacy per-column path.
    tagged_hdus = set(
        h for h, m in per_tag.items()
        if any(k not in ("__xcache__",) for k in m.keys())
    )

    # Group untagged sorted datasets by HDU for the fallback path.
    by_hdu_untagged: Dict[str, List[Tuple[str, str]]] = {}
    for (hname, col) in data["columns"].keys():
        if hname in tagged_hdus:
            continue
        ds = "%s__%s__%s__sorted" % (
            base, safe_dsname(hname), safe_dsname(col)
        )
        if ds in sorted_names:
            by_hdu_untagged.setdefault(hname, []).append((col, ds))

    sort_key = data.get("sort_key")

    # ============================================================
    # Tagged-HDU pages
    # ============================================================
    for hname in tagged_hdus:
        tag_map = {
            k: v for k, v in per_tag[hname].items()
            if k != "__xcache__"
        }
        if not tag_map:
            continue
        tag_cols_list = data.get("tag_columns", {}).get(hname) or []
        page = doc.Root.Add(
            "page", name=safe_dsname("%s_%s" % (base, hname))
        )
        try:
            page.notes.val = "\n".join(
                data.get("header", [])
                + [
                    "",
                    "v0.0.12: rows tagged by %s; one trace per tag."
                    % ", ".join(tag_cols_list),
                ]
            )
        except Exception:
            pass
        grid = page.Add("grid", columns=2)
        # x dataset name to use for the time axis: every tag bucket has
        # a __x__ key when the HDU's sort_key column was processed.  Use
        # the first available tag's x for break-pair detection's data
        # source (we still compute break_pairs from the FULL HDU x array).
        x_label = (sort_key[1] if sort_key and sort_key[0] == hname
                   else "x")
        try:
            x_full = np.asarray(
                data["columns"][sort_key], dtype=float
            ) if sort_key and sort_key[0] == hname else np.empty(0, dtype=float)
        except Exception:
            x_full = np.empty(0, dtype=float)
        # v0.0.18: route through unit-aware break detection so the
        # GUI "Manual gap (hours)" -> MJD-days threshold is converted
        # to the column's native units (seconds for TIME/TIMESTAMP,
        # days for MJD/DMJD/JD).  Pre-v0.0.18 this passed the
        # day-scaled threshold against second-scale diffs and
        # manufactured false break-axes on every per-file page.
        _sort_col_name = (
            sort_key[1] if sort_key and sort_key[0] == hname else None
        )
        break_pairs = detect_time_breaks_unit_aware(
            x_full, _sort_col_name,
            k_factor=gap_k, absolute_gap_days=gap_absolute,
        )

        # Discover the set of measurement columns by union over tag buckets.
        meas_cols = []
        seen = set()
        for tup, bucket in tag_map.items():
            for col in bucket.keys():
                if col in ("__x__", "__dt__"):
                    continue
                if col not in seen:
                    seen.add(col)
                    meas_cols.append(col)

        for col in meas_cols:
            graph = grid.Add("graph", name=safe_dsname("g_%s" % col))
            try:
                graph.y.label.val = col
                graph.y.GridLines.hide.val = False
            except Exception:
                pass
            if break_pairs:
                make_broken_x_axis(
                    graph, break_pairs,
                    label=x_label, show_gridlines=True,
                )
            else:
                try:
                    graph.x.label.val = x_label
                    graph.x.GridLines.hide.val = False
                except Exception:
                    pass
            try:
                key = graph.Add("key", name="key1")
                key.Border.hide.val = bool(hide_keys)
            except Exception:
                pass
            # v0.0.13: vary_style only when this graph has >16 traces.
            vary_style = (
                len(tag_map) > TRACE_STYLE_VARY_THRESHOLD
            )
            for i, (tup, bucket) in enumerate(sorted(tag_map.items())):
                y_ds = bucket.get(col)
                x_ds = bucket.get("__x__")
                if not y_ds or not x_ds:
                    continue
                tag_label = ".".join(t or "NA" for t in tup)
                xy = graph.Add(
                    "xy",
                    name=safe_dsname("xy_%s_%s" % (col, tag_label)),
                )
                xy.xData.val = x_ds
                xy.yData.val = y_ds
                try:
                    xy.key.val = tag_label
                except Exception:
                    pass
                # v0.0.13: identity-stable styling keyed by (col, tup).
                apply_trace_style(
                    xy, identity_key=(col, tup),
                    vary_style=vary_style,
                )

        # ---- v0.0.14/0.0.15 datetime duplicate pages (tagged) ------------
        # Emit up to TWO dt pages per HDU.  The numeric-x variant clones
        # the seconds page and binds xy.labels to a density-controlled
        # text dataset; the text-x variant binds xData to the per-point
        # text dataset and sets axis mode='labels' for axis-driven
        # tick labels.  Either or both can be disabled per call.
        has_density = any(
            "__datelabels_density__" in b for b in tag_map.values()
        )
        has_textx = any(
            "__datelabels_textx__" in b for b in tag_map.values()
        )
        # ---- numeric-x dt page (v0.0.14 lineage) ------------------------
        if has_density and datetime_emit_numeric_dt:
            page_dt = doc.Root.Add(
                "page",
                name=safe_dsname("%s_%s_dt" % (base, hname)),
            )
            try:
                page_dt.notes.val = "\n".join(
                    data.get("header", [])
                    + ["",
                       "Datetime-axis duplicate (numeric x + xy.labels, "
                       "density=%d%%)."
                       % datetime_label_density_pct]
                )
            except Exception:
                pass
            grid_dt = page_dt.Add("grid", columns=2)
            for col in meas_cols:
                graph = grid_dt.Add(
                    "graph", name=safe_dsname("g_%s_dt" % col)
                )
                try:
                    graph.y.label.val = col
                    graph.y.GridLines.hide.val = False
                except Exception:
                    pass
                if break_pairs:
                    make_broken_x_axis(
                        graph, break_pairs,
                        label=x_label, show_gridlines=True,
                    )
                else:
                    try:
                        graph.x.label.val = x_label
                        graph.x.GridLines.hide.val = False
                    except Exception:
                        pass
                try:
                    key = graph.Add("key", name="key1_dt")
                    key.Border.hide.val = bool(hide_keys)
                except Exception:
                    pass
                vary_style_dt = (
                    len(tag_map) > TRACE_STYLE_VARY_THRESHOLD
                )
                for i, (tup, bucket) in enumerate(sorted(tag_map.items())):
                    y_ds = bucket.get(col)
                    x_ds_dt = bucket.get("__x__")
                    label_ds = bucket.get("__datelabels_density__")
                    if not y_ds or not x_ds_dt or not label_ds:
                        continue
                    tag_label = ".".join(t or "NA" for t in tup)
                    xy = graph.Add(
                        "xy",
                        name=safe_dsname(
                            "xy_%s_%s_dt" % (col, tag_label)
                        ),
                    )
                    xy.xData.val = x_ds_dt   # same numeric seconds
                    xy.yData.val = y_ds
                    try:
                        xy.labels.val = label_ds
                    except Exception:
                        pass
                    style_xy_datetime_labels(xy)
                    try:
                        xy.key.val = tag_label
                    except Exception:
                        pass
                    apply_trace_style(
                        xy, identity_key=(col, tup),
                        vary_style=vary_style_dt,
                    )
        # ---- dt_labels page (v0.0.16: mode='datetime', numeric dtsec) ----
        # v0.0.16 swap: bind xData to a numeric Veusz-datetime-seconds
        # dataset and switch axis to mode='datetime' so Veusz renders
        # proper date ticks.  Broken-axis parity: the same MJD
        # break_pairs detected above are mapped to dtsec and installed
        # via make_broken_x_axis when non-empty.
        has_dtnum = any(
            "__datelabels_dtnum__" in b for b in tag_map.values()
        )
        if (has_textx or has_dtnum) and datetime_emit_text_dt:
            page_dtl = doc.Root.Add(
                "page",
                name=safe_dsname("%s_%s_dt_labels" % (base, hname)),
            )
            try:
                page_dtl.notes.val = "\n".join(
                    data.get("header", [])
                    + ["",
                       "Datetime-axis duplicate (numeric dtsec + axis "
                       "mode='datetime'; broken-axis parity with dt page)."]
                )
            except Exception:
                pass
            grid_dtl = page_dtl.Add("grid", columns=2)
            # v0.0.16: compute dtsec break pairs from the same MJD
            # break_pairs used by the numeric dt page so that any gaps
            # rendered as breaks on the seconds-axis page are also
            # rendered as breaks here.
            try:
                dt_break_pairs = mjd_break_pairs_to_dtsec(break_pairs)
            except Exception:
                dt_break_pairs = []
            for col in meas_cols:
                graph = grid_dtl.Add(
                    "graph", name=safe_dsname("g_%s_dtl" % col)
                )
                try:
                    graph.y.label.val = col
                    graph.y.GridLines.hide.val = False
                except Exception:
                    pass
                # v0.0.16: install broken x-axis FIRST (when needed) so
                # the subsequent configure_axis_datetime_mode call lands
                # on the broken-axis widget (AxisBroken extends Axis and
                # inherits the mode property).
                if dt_break_pairs:
                    try:
                        make_broken_x_axis(
                            graph, dt_break_pairs,
                            label=x_label, show_gridlines=True,
                        )
                    except Exception:
                        pass
                try:
                    configure_axis_datetime_mode(
                        graph.x, label=x_label,
                    )
                except Exception:
                    pass
                try:
                    key = graph.Add("key", name="key1_dtl")
                    key.Border.hide.val = bool(hide_keys)
                except Exception:
                    pass
                vary_style_dtl = (
                    len(tag_map) > TRACE_STYLE_VARY_THRESHOLD
                )
                for i, (tup, bucket) in enumerate(sorted(tag_map.items())):
                    y_ds = bucket.get(col)
                    # v0.0.16: prefer the new numeric dtsec dataset; fall
                    # back to the v0.0.15 text-x dataset only if dtnum
                    # was not emitted (legacy compatibility).
                    dtnum_ds = bucket.get("__datelabels_dtnum__")
                    textx_ds = bucket.get("__datelabels_textx__")
                    x_ds = dtnum_ds or textx_ds
                    if not y_ds or not x_ds:
                        continue
                    tag_label = ".".join(t or "NA" for t in tup)
                    xy = graph.Add(
                        "xy",
                        name=safe_dsname(
                            "xy_%s_%s_dtl" % (col, tag_label)
                        ),
                    )
                    xy.xData.val = x_ds
                    xy.yData.val = y_ds
                    try:
                        xy.key.val = tag_label
                    except Exception:
                        pass
                    apply_trace_style(
                        xy, identity_key=(col, tup),
                        vary_style=vary_style_dtl,
                    )

    # ============================================================
    # Untagged-HDU pages (legacy)
    # ============================================================
    for hname, cols in by_hdu_untagged.items():
        page = doc.Root.Add("page", name=safe_dsname("%s_%s" % (base, hname)))
        try:
            page.notes.val = "\n".join(data.get("header", []))
        except Exception:
            pass
        grid = page.Add("grid", columns=2)
        x_col = None
        for c, ds in cols:
            if c.upper() in [k.upper() for k in SORTED_KEY_HINT]:
                x_col = (c, ds)
                break
        if x_col is None:
            x_col = cols[0]
        x_name = x_col[1]
        try:
            x_arr = np.asarray(data["columns"][(hname, x_col[0])], dtype=float)
        except Exception:
            x_arr = np.empty(0, dtype=float)
        # v0.0.18: unit-aware break detection (see _build_pages tagged-
        # HDU site for rationale).
        break_pairs = detect_time_breaks_unit_aware(
            x_arr, x_col[0],
            k_factor=gap_k, absolute_gap_days=gap_absolute,
        )
        for c, ds in cols:
            if ds == x_name:
                continue
            graph = grid.Add("graph", name=safe_dsname("g_%s" % c))
            try:
                graph.y.label.val = c
                graph.y.GridLines.hide.val = False
            except Exception:
                pass
            if break_pairs:
                make_broken_x_axis(graph, break_pairs,
                                   label=x_col[0], show_gridlines=True)
            else:
                try:
                    graph.x.label.val = x_col[0]
                    graph.x.GridLines.hide.val = False
                except Exception:
                    pass
            xy = graph.Add("xy", name=safe_dsname("xy_%s" % c))
            xy.xData.val = x_name
            xy.yData.val = ds
            # v0.0.13: untagged graph carries exactly one trace, so
            # vary_style is always False; identity is (col, (None,)).
            apply_trace_style(
                xy, identity_key=(c, (None,)), vary_style=False,
            )
        if datetime_x_name:
            # v0.0.15: untagged datetime-duplicate pages.  Two variants
            # may be emitted, each gated by an independent boolean:
            #
            #   * numeric-x dt page (v0.0.14 lineage) -- x axis stays
            #     numeric seconds (same dataset as the seconds page);
            #     per-point xy.labels carry density-thinned date
            #     strings.  Gated by ``datetime_emit_numeric_dt``.
            #   * text-x dt_labels page (v0.0.15 new) -- xData is the
            #     per-point text dataset; axis mode='labels' renders
            #     the strings on the axis itself.  Spacing is uniform
            #     (not proportional to elapsed time).  Gated by
            #     ``datetime_emit_text_dt``.
            label_density = data.get("_datelabels_density")
            label_textx = data.get("_datelabels_textx")
            # Back-compat: if older callers populated only the legacy
            # sparse/full keys, fall back to those for the numeric page.
            if not label_density:
                label_density = (data.get("_datelabels_sparse")
                                 or data.get("_datelabels_full"))

            # ---- numeric-x dt page (v0.0.14 lineage) ----------------
            if datetime_emit_numeric_dt:
                page_dt = doc.Root.Add(
                    "page",
                    name=safe_dsname("%s_%s_dt" % (base, hname))
                )
                try:
                    page_dt.notes.val = "\n".join(
                        data.get("header", [])
                        + ["",
                           "Datetime-axis duplicate (v0.0.15 numeric-x; "
                           "per-point xy.labels at %d%% density)."
                           % datetime_label_density_pct]
                    )
                except Exception:
                    pass
                grid_dt = page_dt.Add("grid", columns=2)
                for c, ds in cols:
                    if ds == x_name:
                        continue
                    graph = grid_dt.Add(
                        "graph", name=safe_dsname("g_%s_dt" % c)
                    )
                    try:
                        graph.y.label.val = c
                        graph.y.GridLines.hide.val = False
                    except Exception:
                        pass
                    if break_pairs:
                        make_broken_x_axis(
                            graph, break_pairs,
                            label=x_col[0], show_gridlines=True,
                        )
                    else:
                        try:
                            graph.x.label.val = x_col[0]
                            graph.x.GridLines.hide.val = False
                        except Exception:
                            pass
                    xy = graph.Add("xy", name=safe_dsname("xy_%s_dt" % c))
                    xy.xData.val = x_name           # numeric seconds (same as seconds page)
                    xy.yData.val = ds
                    if label_density:
                        try:
                            xy.labels.val = label_density
                        except Exception:
                            # Older Veusz: labels property may not be a
                            # DatasetOrStr -- silently fall back.
                            pass
                        style_xy_datetime_labels(xy)
                    apply_trace_style(
                        xy, identity_key=(c, (None,)), vary_style=False,
                    )

            # ---- dt_labels page (v0.0.16 mode='datetime') -----------
            # v0.0.16 swap: prefer numeric dtsec dataset and switch axis
            # to mode='datetime' for true date ticks; install broken
            # x-axis when the seconds-axis page has one (MJD break pairs
            # mapped to Veusz seconds).
            label_dtnum = data.get("_datelabels_dtnum")
            label_mjd = data.get("_datelabels_mjd")
            if datetime_emit_text_dt and (label_dtnum or label_textx):
                page_dtL = doc.Root.Add(
                    "page",
                    name=safe_dsname("%s_%s_dt_labels" % (base, hname))
                )
                try:
                    page_dtL.notes.val = "\n".join(
                        data.get("header", [])
                        + ["",
                           "Datetime-axis duplicate (v0.0.16 numeric "
                           "dtsec + axis mode='datetime'; broken-axis "
                           "parity with dt page)."]
                    )
                except Exception:
                    pass
                grid_dtL = page_dtL.Add("grid", columns=2)
                # Compute dtsec break pairs from the MJD break pairs used
                # by the numeric-x dt page.
                try:
                    dt_break_pairs_unt = mjd_break_pairs_to_dtsec(
                        break_pairs
                    )
                except Exception:
                    dt_break_pairs_unt = []
                x_ds_dtl = label_dtnum or label_textx
                for c, ds in cols:
                    if ds == x_name:
                        continue
                    graph = grid_dtL.Add(
                        "graph", name=safe_dsname("g_%s_dtL" % c)
                    )
                    try:
                        graph.y.label.val = c
                        graph.y.GridLines.hide.val = False
                    except Exception:
                        pass
                    # Install broken x-axis (when needed) BEFORE
                    # configuring datetime mode -- AxisBroken extends
                    # Axis so it inherits the mode property.
                    if dt_break_pairs_unt:
                        try:
                            make_broken_x_axis(
                                graph, dt_break_pairs_unt,
                                label=x_col[0], show_gridlines=True,
                            )
                        except Exception:
                            pass
                    try:
                        configure_axis_datetime_mode(
                            graph.x, label=x_col[0],
                        )
                    except Exception:
                        pass
                    xy = graph.Add("xy", name=safe_dsname("xy_%s_dtL" % c))
                    xy.xData.val = x_ds_dtl       # numeric dtsec
                    xy.yData.val = ds
                    apply_trace_style(
                        xy, identity_key=(c, (None,)), vary_style=False,
                    )

    # Image pages -- explicit early-exit when no image HDUs exist in the
    # file.  This is the normal case for NRAO 1PPS-delta FITS (the only
    # table HDU is OnePpsDeltas), so we don't even enter the page-
    # creation loop.
    images_dict = data.get("images") or {}
    if not images_dict:
        return
    for hname, img in images_dict.items():
        ds_name = "%s__%s__IMAGE" % (base, safe_dsname(hname))
        page = doc.Root.Add("page", name=safe_dsname("%s_%s_img" % (base, hname)))
        graph = page.Add("graph", name="g_img")
        try:
            image_w = graph.Add("image", name="imageIn",
                                data=ds_name, colorMap="plasma")
            graph.Add("colorbar", direction="vertical",
                      name="colorbar1", widgetName="imageIn")
        except Exception:
            # Older Veusz API: omit colorbar
            graph.Add("image", name="imageIn", data=ds_name)


# ============================================================================
# %% UNIT-OVERLAY PAGES (combined plots across files)
# ============================================================================
# A palette of distinct colours used to disambiguate the (file, column)
# traces stacked on the same overlay graph.  Veusz also accepts hex codes;
# we use named colours so the saved .vszh5 stays readable.
_OVERLAY_COLORS = [
    "blue", "red", "green", "darkorange", "purple", "saddlebrown",
    "deeppink", "olive", "teal", "navy", "darkred", "darkgreen",
    "magenta", "black", "darkcyan", "goldenrod",
]


def build_unit_overlay_pages(doc, file_records,
                              gap_k=10.0, gap_absolute=0.0, log_cb=None,
                              datetime_duplicate=False,
                              datetime_full_labels=None,
                              datetime_label_density_pct: int = 10,
                              datetime_emit_numeric_dt: bool = True,
                              datetime_emit_text_dt: bool = True,
                              hide_keys: bool = False):
    """
    Build one overlay page per distinct unit string across the loaded
    batch.  Each overlay page contains a single graph that plots every
    (file, column, tag-tuple) combination whose unit string matches that
    page, against its file's x-axis (sort-key) dataset.  When the combined
    x-values across files show large gaps, the graph uses an
    ``axis-broken`` x axis (gap thresholds: K * median(Δt) auto, or
    ``gap_absolute`` if positive).

    v0.0.12 channel-tag row model
    -----------------------------
    For any HDU whose record carries a ``tag_columns`` declaration (e.g.
    ``CHANNELA``, ``CHANNELB``), each numeric measurement column is split
    by the unique tag-tuples observed across the batch.  Each unique
    tuple becomes its OWN concatenated-in-time trace on the overlay page
    -- there is no longer a single combined DELTAT trace; instead there
    is one DELTAT trace per channel-pair (e.g. CHA1.CHB2, CHA1.CHB3, ...).
    HDUs with no tag declaration fall back to the legacy single-trace
    behaviour.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        The active embedded Veusz document.
    file_records : list of dict
        One entry per successfully pushed file, each containing keys
        ``base``, ``columns`` (dict (hname,col) -> array), ``units``
        (dict (hname,col) -> unit-string), ``sort_key`` (the (hname,col)
        used as time-axis), and optionally ``tag_columns`` (dict hname ->
        [tag_col_name,...]) and ``tag_groups`` (dict hname -> dict
        tag_tuple -> int64 row-index array).
    gap_k, gap_absolute : float
        Forwarded to ``detect_time_breaks``.
    log_cb : callable or None
        Optional ``log_cb(msg)`` for status messages.
    datetime_label_density_pct : int
        v0.0.15. Density of date-string anchors for the numeric-x dt
        overlay (0..100). Supersedes the v0.0.14 boolean
        ``datetime_full_labels``.
    datetime_emit_numeric_dt : bool
        v0.0.15. Emit the v0.0.14-style numeric-x dt overlay page.
    datetime_emit_text_dt : bool
        v0.0.15. Emit the new v0.0.15 text-x dt_labels overlay page
        (xData is a per-point text dataset, axis mode='labels').
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
    if not file_records:
        return
    # v0.0.12: Combined-in-time overlay semantics + channel-tag rows.
    # ---------------------------------------------------------------
    # Group key: (unit_str, hname, col, tag_tuple)
    #   -> list of dicts: {base, x, y, mjd_or_none}
    # tag_tuple is (None,) for untagged HDUs (legacy single-series).
    # ---------------------------------------------------------------
    groups = {}
    # accumulate x samples across files per unit for gap detection on
    # the unit page (combined across all columns sharing the unit).
    x_samples_by_unit = {}
    # v0.0.18: track the sort-key column name (e.g. "DMJD", "TIME")
    # for each unit page so the overlay can route through the unit-
    # aware break detector.  We record the FIRST observed sort column
    # name per unit; mixed-column batches are rare in practice and
    # falling back to the first name is no worse than the pre-v0.0.18
    # behavior of ignoring units entirely.
    sort_col_name_by_unit = {}
    for rec in file_records:
        base = rec.get("base")
        cols = rec.get("columns") or {}
        units = rec.get("units") or {}
        sort_key = rec.get("sort_key")
        tag_columns_map = rec.get("tag_columns") or {}
        tag_groups_map = rec.get("tag_groups") or {}
        if not base or sort_key is None or sort_key not in cols:
            # without a time axis we cannot overlay against time
            continue
        x_hdu, x_col = sort_key
        try:
            x_arr_full = np.asarray(cols[sort_key], dtype=float)
        except Exception:
            continue
        # If the sort key is MJD/JD-like, convert to MJD-equivalent for
        # the datetime overlay.  JD is shifted by 2400000.5.
        mjd_arr_full = None
        sk_upper = x_col.upper()
        if datetime_duplicate and sk_upper in ("DMJD", "MJD", "JD"):
            try:
                if sk_upper == "JD":
                    mjd_arr_full = x_arr_full - 2400000.5
                else:
                    mjd_arr_full = x_arr_full.copy()
            except Exception:
                mjd_arr_full = None

        # Build set of (hname, col) names that are tag-columns -- skip
        # them as plottable series.
        tag_col_set = set()
        for h, tlist in tag_columns_map.items():
            for tcol in tlist or []:
                tag_col_set.add((h, tcol))

        for (hname, col), arr in cols.items():
            if (hname, col) == sort_key:
                continue
            if (hname, col) in tag_col_set:
                continue  # row-tag, not a series
            arr = np.asarray(arr)
            if arr.dtype.kind not in ("f", "i", "u"):
                continue  # skip text columns
            try:
                y_full = np.asarray(arr, dtype=float)
            except Exception:
                continue
            if y_full.shape != x_arr_full.shape:
                # length mismatch -- cannot align this column safely
                continue
            unit_str = units.get((hname, col), "") or ""
            unit_str = str(unit_str).strip()

            # Decide split: tagged HDU -> per tag-tuple; else single (None,)
            tg = tag_groups_map.get(hname)
            tcols = tag_columns_map.get(hname) or []
            if tg and tcols:
                # one slice per tag tuple
                for tup, row_idx in tg.items():
                    try:
                        ri = np.asarray(row_idx, dtype=np.int64)
                    except Exception:
                        continue
                    if ri.size == 0:
                        continue
                    x_slice = x_arr_full[ri]
                    y_slice = y_full[ri]
                    mjd_slice = (mjd_arr_full[ri]
                                 if mjd_arr_full is not None else None)
                    gkey = (unit_str, hname, col, tuple(tup))
                    groups.setdefault(gkey, []).append({
                        "base": base,
                        "x": x_slice,
                        "y": y_slice,
                        "mjd": mjd_slice,
                    })
                    x_samples_by_unit.setdefault(
                        unit_str, []).append(x_slice)
                    sort_col_name_by_unit.setdefault(unit_str, x_col)
            else:
                # untagged HDU -- legacy single trace
                gkey = (unit_str, hname, col, (None,))
                groups.setdefault(gkey, []).append({
                    "base": base,
                    "x": x_arr_full,
                    "y": y_full,
                    "mjd": mjd_arr_full,
                })
                x_samples_by_unit.setdefault(
                    unit_str, []).append(x_arr_full)
                sort_col_name_by_unit.setdefault(unit_str, x_col)

    if not groups:
        if log_cb:
            log_cb("  No columns suitable for unit-overlay pages.")
        return

    # Bucket groups by unit so we still emit one page per unit, with
    # one xy trace per (hdu, col, tag-tuple) on that page.
    by_unit = {}
    for (unit_str, hname, col, tup), members in groups.items():
        by_unit.setdefault(unit_str, []).append((hname, col, tup, members))

    def _tup_suffix(tup):
        if not tup or tup == (None,):
            return "all"
        return "_".join(safe_dsname(str(t) if t is not None else "NA")
                        for t in tup)

    def _tup_label(tup):
        if not tup or tup == (None,):
            return ""
        return ".".join(str(t) if t is not None else "NA" for t in tup)

    for unit_str, entries in by_unit.items():
        unit_label = unit_str if unit_str else "(dimensionless)"
        page_name = safe_dsname("Overlay_%s" % (unit_str or "none"))
        page = doc.Root.Add("page", name=page_name)
        try:
            page.notes.val = (
                "Combined-in-time overlay of all columns with unit "
                "'%s' across the loaded batch (v0.0.12: one trace per "
                "(column, channel-tag tuple), time-sorted concatenation "
                "across files)."
                % unit_label
            )
        except Exception:
            pass
        graph = page.Add("graph", name="g_overlay")
        try:
            graph.y.label.val = unit_label
            graph.y.GridLines.hide.val = False
        except Exception:
            pass
        # Compute break pairs from the combined x distribution across
        # all files contributing to this unit page so the broken axis
        # spans the whole batch.
        x_all = np.concatenate(
            [a for a in x_samples_by_unit.get(unit_str, []) if a.size]
        ) if x_samples_by_unit.get(unit_str) else np.empty(0, dtype=float)
        # v0.0.18: unit-aware break detection so the day-scaled
        # manual-gap threshold is converted to the seconds-axis
        # units when the sort key is TIME/TIMESTAMP.  The MJD-keyed
        # dt_labels overlay (below) keeps its own MJD-array detection.
        _ov_sort_col = sort_col_name_by_unit.get(unit_str)
        break_pairs = detect_time_breaks_unit_aware(
            x_all, _ov_sort_col,
            k_factor=gap_k, absolute_gap_days=gap_absolute,
        )
        if break_pairs:
            make_broken_x_axis(graph, break_pairs,
                               label="time", show_gridlines=True)
        else:
            try:
                graph.x.label.val = "time"
                graph.x.GridLines.hide.val = False
            except Exception:
                pass
        try:
            key = graph.Add("key", name="key1")
            key.Border.hide.val = bool(hide_keys)
        except Exception:
            pass

        # Track which (hname, col, tup) entries produced datetime-
        # eligible concatenations (every contributing file had an
        # MJD-like sort key).  Used for the dt duplicate page.
        dt_eligible = []  # list of (hname, col, tup, dt_x_name, y_name, idx)
        u_safe = safe_dsname(unit_str or "none")
        # v0.0.13: vary_style only when the overlay graph would exceed
        # the color palette and need additional disambiguation.
        vary_style_overlay = (
            len(entries) > TRACE_STYLE_VARY_THRESHOLD
        )
        for i, (hname, col, tup, members) in enumerate(entries):
            # Concatenate and time-sort this (unit, hdu, col, tup) group.
            try:
                xs = np.concatenate([m["x"] for m in members])
                ys = np.concatenate([m["y"] for m in members])
            except Exception:
                continue
            if xs.size == 0:
                continue
            # v0.0.12: GPU-aware stable sort.  gpu_argsort routes to CuPy
            # when enabled AND xs is large enough, else falls back to
            # np.argsort(kind='mergesort').  NaNs end up at the tail.
            try:
                order = gpu_argsort(xs)
            except Exception:
                try:
                    order = np.argsort(xs, kind="mergesort")
                except Exception:
                    order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = ys[order]
            h_safe = safe_dsname(hname)
            c_safe = safe_dsname(col)
            t_safe = _tup_suffix(tup)
            x_name = "OverlayCat__%s__%s__%s__%s__x__sorted" % (
                u_safe, h_safe, c_safe, t_safe
            )
            y_name = "OverlayCat__%s__%s__%s__%s__y__sorted" % (
                u_safe, h_safe, c_safe, t_safe
            )
            try:
                doc.SetData(x_name, xs_s)
                doc.SetData(y_name, ys_s)
            except Exception as _exc:
                if log_cb:
                    log_cb("  Overlay '%s.%s [%s]' SetData failed: %s"
                           % (hname, col, _tup_label(tup), _exc))
                continue
            xy_name = safe_dsname("xy_%s_%s_%s" % (hname, col, t_safe))
            xy = graph.Add("xy", name=xy_name)
            xy.xData.val = x_name
            xy.yData.val = y_name
            try:
                tlab = _tup_label(tup)
                if tlab:
                    xy.key.val = "%s.%s [%s]" % (hname, col, tlab)
                else:
                    xy.key.val = "%s.%s" % (hname, col)
            except Exception:
                pass
            # v0.0.13: identity = (col, tup); this trace will keep the
            # same color/style on the seconds overlay, the dt overlay,
            # and every per-file page that contains the same (col, tup).
            apply_trace_style(
                xy, identity_key=(col, tup),
                vary_style=vary_style_overlay,
            )

            # v0.0.14: datetime companion -- only if EVERY member has an
            # MJD array.  Replaces v0.0.11..0.0.13 SetDataDateTime path
            # (which fails on Veusz 3.4 internals) with TEXT-LABEL
            # datasets bound to xy.labels.val on the dt-overlay page.
            # We build BOTH sparse and full label datasets every time so
            # the GUI checkbox can toggle the default at render time
            # without rebuilding the document.  The numeric seconds x is
            # REUSED from the seconds-page dataset (``x_name``) so the
            # dt-overlay page uses the same x dataset as the seconds
            # overlay; only the labels are added.
            # v0.0.17: skip dt-overlay emission for sentinel channel-tag
            # tuples (e.g. tup == ("dataset", "dataset")) -- placeholder
            # catch-all groupings whose per-file MJD coverage is
            # degenerate and would manufacture spurious broken-axis
            # breaks in the combined-MJD break detection below.  Real
            # channel-pair tuples like ("A", "B") pass through.  The
            # seconds-overlay trace above is still drawn for sentinel
            # tuples -- harmless on a unitless seconds axis.
            if datetime_duplicate and all(
                m.get("mjd") is not None for m in members
            ) and not is_sentinel_tag_tuple(tup):
                try:
                    mjds = np.concatenate(
                        [np.asarray(m["mjd"], dtype=float)
                         for m in members]
                    )
                    mjds_s = mjds[order]
                    # v0.0.15: build density-thinned label dataset for
                    # the numeric-x dt overlay AND a per-point text
                    # dataset for the text-x dt_labels overlay.  Both
                    # are emitted unconditionally on each trace; the
                    # per-page gates decide which pages are rendered.
                    density_name = (
                        "OverlayCat__%s__%s__%s__%s__datelabels_density"
                        % (u_safe, h_safe, c_safe, t_safe)
                    )
                    textx_name = (
                        "OverlayCat__%s__%s__%s__%s__datelabels_textx"
                        % (u_safe, h_safe, c_safe, t_safe)
                    )
                    # v0.0.16: numeric Veusz-datetime-seconds dataset
                    # alongside the text-x dataset so the dt_labels
                    # overlay can render with axis mode='datetime'.
                    dtnum_name = (
                        "OverlayCat__%s__%s__%s__%s__datelabels_dtnum"
                        % (u_safe, h_safe, c_safe, t_safe)
                    )
                    ok_density = build_density_datestr_dataset(
                        doc, density_name, mjds_s,
                        density_pct=datetime_label_density_pct,
                        log_cb=log_cb,
                    )
                    ok_textx = False
                    ok_dtnum = False
                    if datetime_emit_text_dt:
                        ok_textx = build_textx_dataset(
                            doc, textx_name, mjds_s, log_cb=log_cb,
                        )
                        # v0.0.16: emit numeric dtsec dataset
                        ok_dtnum = build_dtnum_dataset(
                            doc, dtnum_name, mjds_s, log_cb=log_cb,
                        )
                    if ok_density or ok_textx or ok_dtnum:
                        dt_eligible.append((
                            hname, col, tup,
                            (density_name if ok_density else None),
                            (textx_name if ok_textx else None),
                            x_name, y_name, i,
                            int(mjds_s.size),
                            (dtnum_name if ok_dtnum else None),
                            np.asarray(mjds_s, dtype=float),
                        ))
                except Exception as _exc:
                    if log_cb:
                        log_cb("  Overlay '%s.%s [%s]' datetime build "
                               "failed: %s"
                               % (hname, col, _tup_label(tup), _exc))

        if log_cb:
            log_cb("  Overlay page '%s': %d trace(s) (time-combined, "
                   "per channel-tag)"
                   % (unit_label, len(entries)))

        # ---- v0.0.15: datetime-duplicate overlay pages ----------------
        # Two pages may be emitted, gated independently:
        #   numeric-x dt overlay (v0.0.14 lineage) -- xData is the
        #     numeric seconds dataset reused from the seconds overlay;
        #     per-point xy.labels carry density-thinned date strings.
        #   text-x dt_labels overlay (v0.0.15) -- xData is the per-point
        #     text dataset; axis mode='labels' renders strings on the
        #     axis itself (uniform sample spacing).
        if datetime_duplicate and dt_eligible:
            # ---- numeric-x dt overlay page ---------------------------
            if datetime_emit_numeric_dt:
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
                    graph_dt.y.label.val = unit_label
                    graph_dt.y.GridLines.hide.val = False
                except Exception:
                    pass
                if break_pairs:
                    make_broken_x_axis(
                        graph_dt, break_pairs,
                        label="time", show_gridlines=True,
                    )
                else:
                    try:
                        graph_dt.x.label.val = "time"
                        graph_dt.x.GridLines.hide.val = False
                    except Exception:
                        pass
                try:
                    key_dt = graph_dt.Add("key", name="key1_dt")
                    key_dt.Border.hide.val = bool(hide_keys)
                except Exception:
                    pass
                vary_style_overlay_dt = (
                    len(dt_eligible) > TRACE_STYLE_VARY_THRESHOLD
                )
                for (hname, col, tup, density_name, textx_name,
                     xnum_name, y_name, idx, _npts,
                     _dtnum_name, _mjds_arr) in dt_eligible:
                    t_safe = _tup_suffix(tup)
                    xy_name = safe_dsname(
                        "xy_%s_%s_%s_dt" % (hname, col, t_safe)
                    )
                    xy = graph_dt.Add("xy", name=xy_name)
                    xy.xData.val = xnum_name        # numeric seconds (seconds-overlay dataset)
                    xy.yData.val = y_name
                    if density_name:
                        try:
                            xy.labels.val = density_name
                        except Exception:
                            pass
                        style_xy_datetime_labels(xy)
                    try:
                        tlab = _tup_label(tup)
                        if tlab:
                            xy.key.val = "%s.%s [%s]" % (hname, col, tlab)
                        else:
                            xy.key.val = "%s.%s" % (hname, col)
                    except Exception:
                        pass
                    apply_trace_style(
                        xy, identity_key=(col, tup),
                        vary_style=vary_style_overlay_dt,
                    )
                if log_cb:
                    log_cb("  Datetime-overlay page '%s_dt': %d trace(s) "
                           "(%d%% density labels)"
                           % (page_name, len(dt_eligible),
                              datetime_label_density_pct))

            # ---- dt_labels overlay page (v0.0.16 mode='datetime') ----
            # v0.0.16 swap: bind xData to per-trace numeric Veusz-
            # datetime-seconds dataset; axis mode='datetime'; broken-
            # axis parity derived from concatenated MJD break detection
            # across all eligible overlay traces.
            if datetime_emit_text_dt and any(
                e[9] or e[4] for e in dt_eligible
            ):
                page_dtL = doc.Root.Add(
                    "page", name=safe_dsname("%s_dt_labels" % page_name)
                )
                try:
                    page_dtL.notes.val = (
                        "Datetime-axis duplicate of '%s' "
                        "(v0.0.16 numeric dtsec + axis "
                        "mode='datetime'; broken-axis parity)."
                        % page_name
                    )
                except Exception:
                    pass
                graph_dtL = page_dtL.Add("graph", name="g_overlay_dtL")
                try:
                    graph_dtL.y.label.val = unit_label
                    graph_dtL.y.GridLines.hide.val = False
                except Exception:
                    pass
                # v0.0.16: compute combined-MJD break pairs across all
                # eligible overlay traces, then map MJD->dtsec for the
                # broken-axis pairs.
                dt_break_pairs_ov = []
                try:
                    mjd_arrays = [
                        e[10] for e in dt_eligible
                        if e[10] is not None and e[10].size
                    ]
                    if mjd_arrays:
                        mjd_all = np.concatenate(mjd_arrays)
                        # v0.0.18: route through unit-aware helper for
                        # parity with the seconds-overlay path.  mjd_all
                        # is already day-scaled (factor = 1.0) so the
                        # behavior is unchanged, but the call site no
                        # longer assumes implicit MJD-day units.
                        mjd_breaks = detect_time_breaks_unit_aware(
                            mjd_all, "MJD",
                            k_factor=gap_k,
                            absolute_gap_days=gap_absolute,
                        )
                        dt_break_pairs_ov = mjd_break_pairs_to_dtsec(
                            mjd_breaks
                        )
                except Exception:
                    dt_break_pairs_ov = []
                # Install broken x-axis BEFORE datetime-mode configure
                # so the AxisBroken widget receives the mode setting.
                if dt_break_pairs_ov:
                    try:
                        make_broken_x_axis(
                            graph_dtL, dt_break_pairs_ov,
                            label="time", show_gridlines=True,
                        )
                    except Exception:
                        pass
                try:
                    configure_axis_datetime_mode(
                        graph_dtL.x, label="time",
                    )
                except Exception:
                    pass
                try:
                    key_dtL = graph_dtL.Add("key", name="key1_dtL")
                    key_dtL.Border.hide.val = bool(hide_keys)
                except Exception:
                    pass
                vary_style_overlay_dtL = (
                    len(dt_eligible) > TRACE_STYLE_VARY_THRESHOLD
                )
                for (hname, col, tup, density_name, textx_name,
                     xnum_name, y_name, idx, _npts,
                     dtnum_name, _mjds_arr) in dt_eligible:
                    # v0.0.16: prefer numeric dtsec dataset; fall back
                    # to text-x for legacy emission paths.
                    x_ds_dtL = dtnum_name or textx_name
                    if not x_ds_dtL:
                        continue
                    t_safe = _tup_suffix(tup)
                    xy_name = safe_dsname(
                        "xy_%s_%s_%s_dtL" % (hname, col, t_safe)
                    )
                    xy = graph_dtL.Add("xy", name=xy_name)
                    xy.xData.val = x_ds_dtL
                    xy.yData.val = y_name
                    try:
                        tlab = _tup_label(tup)
                        if tlab:
                            xy.key.val = "%s.%s [%s]" % (hname, col, tlab)
                        else:
                            xy.key.val = "%s.%s" % (hname, col)
                    except Exception:
                        pass
                    apply_trace_style(
                        xy, identity_key=(col, tup),
                        vary_style=vary_style_overlay_dtL,
                    )
                if log_cb:
                    log_cb("  Datetime-overlay page '%s_dt_labels': %d "
                           "trace(s) (numeric dtsec, mode='datetime')"
                           % (page_name,
                              sum(1 for e in dt_eligible
                                  if (e[9] or e[4]))))


# ============================================================================
# %% WORKER THREAD (parallel read)
# ============================================================================
class FITSBatchWorker(QThread):
    """Background QThread that reads a batch of FITS files concurrently."""

    progress = Signal(int, int, str)   # done, total, last_key
    log = Signal(str)
    finished_ok = Signal(dict)         # {path: read_dict}
    failed = Signal(str)

    def __init__(self, files: List[str], backend: str, max_threads: int,
                 cache: MemoryAwareCache, parent=None,
                 skip_images: bool = False) -> None:
        super().__init__(parent)
        self.files = files
        self.backend = backend
        self.max_threads = max_threads
        self.cache = cache
        self.skip_images = bool(skip_images)

    def run(self) -> None:
        try:
            # Outer suppressor: in addition to the inner wraps inside
            # FITSProcessor.read(), this keeps the worker thread free of
            # the noisy NRAO FITS UnitsWarnings during the whole batch.
            with suppress_fits_unit_warnings():
                proc = FITSProcessor(self.backend, self.cache,
                                     skip_images=self.skip_images)
                work = [(p, proc.read, (p,)) for p in self.files]
                results = run_in_threadpool(
                    work,
                    max_workers=self.max_threads,
                    progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
                )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit("%s\n%s" % (exc, traceback.format_exc()))


# ============================================================================
# %% MAIN WINDOW
# ============================================================================
class FITSAutoPlotWindow(AutoPlotMainWindow):
    """Top-level window for the FITS AutoPlot script."""

    def __init__(self) -> None:
        super().__init__("FITS AutoPlot - Veusz Embedded GUI", default_mode="dark")
        self._file_filter = (
            "FITS files (*.fits *.fit *.fits.gz *.fit.gz);;All files (*)"
        )
        self.cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=DEFAULT_RSS_HIGH_WATER_MB,
        ))
        self.monitor = MemoryMonitor(
            self.cache,
            callback=lambda rss: self.log("RSS crossed high-water mark "
                                          "(%.1f MiB) -- spilling to disk." % rss),
        )
        self.monitor.start()
        self.veusz_doc = None
        self.worker = None
        self.log("MAX_THREADS = %d" % MAX_THREADS)

    # ----- options form ----------------------------------------------------
    def _populate_options(self, form: QFormLayout) -> None:
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["both (Veusz + astropy)",
                                     "veusz (native FITS import)",
                                     "astropy"])
        form.addRow(QLabel("Import backend:"), self.backend_combo)

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

        # Speed knob: skip image HDUs entirely.  NRAO 1PPS-delta files
        # have no useful image HDUs, so this can speed up processing
        # measurably for large batches.
        self.skip_images_cb = QCheckBox(
            "Skip image HDUs (faster -- recommended for NRAO 1PPS files)"
        )
        self.skip_images_cb.setChecked(False)
        form.addRow(self.skip_images_cb)

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
        # v0.0.16: user-facing absolute-gap input is in HOURS for the
        # primary MJD-axis use case (1 hour = 1/24 MJD-day).  The value
        # is converted to days when calling detect_time_breaks.  Set 0
        # to fall back to the K * median(dt) heuristic.
        form.addRow(QLabel("Manual gap (hours; 0=auto):"),
                    self.gap_abs_spin)

        # v0.0.19: free-form gap-string input.  Accepts "5m3d2h"-style
        # text (m=months, d=days, h=hours; case-insensitive; whitespace
        # tolerated).  When non-empty, OVERRIDES the spinbox above.
        self.gap_abs_str = QLineEdit()
        self.gap_abs_str.setPlaceholderText("e.g. 5m3d2h  (overrides spinbox)")
        self.gap_abs_str.setToolTip(
            "Optional gap string.  Units: m = months (~30.44 days), "
            "d = days, h = hours.  Examples: '3d', '120h', '5m 3d 2h'.  "
            "Leave blank to use the numeric spinbox above.  Bare number "
            "is interpreted as hours."
        )
        form.addRow(QLabel("Manual gap (string; overrides):"),
                    self.gap_abs_str)

        # v0.0.19: hide legend keys on every page when ticked.
        self.hide_keys_cb = QCheckBox("Hide legend keys on all pages")
        self.hide_keys_cb.setChecked(False)
        self.hide_keys_cb.setToolTip(
            "When checked, every key (legend) widget added to a graph "
            "is created with Border/hide set, suppressing the legend "
            "frame.  Useful for cleaner screenshots / publication plots."
        )
        form.addRow(self.hide_keys_cb)

        self.combined_only_cb = QCheckBox(
            "Combined (overlay) plots only -- skip per-file pages"
        )
        self.combined_only_cb.setChecked(False)
        form.addRow(self.combined_only_cb)

        # --- Datetime-duplicate plots (v0.0.11) ---------------------------
        # When ticked, every per-file page and every cross-file overlay
        # page is duplicated against a parallel Veusz date-time x axis
        # (tick labels: YYYY-MM-DD HH:MM:SS, rotated 45 deg).
        self.datetime_dup_cb = QCheckBox(
            "Duplicate plots with datetime X axis (YYYY-MM-DD HH:MM:SS)"
        )
        self.datetime_dup_cb.setChecked(False)
        form.addRow(self.datetime_dup_cb)

        # --- v0.0.15: date-label density + dt page variants ----------------
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
            "label.  0 = no labels, 100 = one label per point. "
            "Anchors are evenly spaced.  Applies to the numeric-x dt "
            "page (xy.labels) and scales the tick count on the text-x "
            "dt_labels page (axis mode='labels')."
        )
        form.addRow(
            QLabel("Date-label density on dt pages:"),
            self.label_density_spin,
        )
        self.emit_numeric_dt_cb = QCheckBox(
            "Emit numeric-x dt page (v0.0.14 lineage)"
        )
        self.emit_numeric_dt_cb.setChecked(True)
        self.emit_numeric_dt_cb.setToolTip(
            "When checked, builds the v0.0.14-style dt page: xData is "
            "the numeric seconds dataset (same as the seconds page) and "
            "per-point xy.labels carry density-thinned YYYY-MM-DD "
            "HH:MM:SS strings.  Sample spacing is proportional to "
            "elapsed time."
        )
        form.addRow(self.emit_numeric_dt_cb)
        self.emit_text_dt_cb = QCheckBox(
            "Emit text-x dt_labels page (v0.0.15 new)"
        )
        self.emit_text_dt_cb.setChecked(True)
        self.emit_text_dt_cb.setToolTip(
            "When checked, builds the v0.0.15 dt_labels page: xData is "
            "a per-point text dataset and the x axis is set to "
            "mode='labels'.  The axis itself renders the date strings. "
            "Sample spacing is uniform (one tick per point), NOT "
            "proportional to elapsed time -- gaps in time disappear."
        )
        form.addRow(self.emit_text_dt_cb)

        # --- Minimized Veusz save (v0.0.17) -------------------------------
        # When "Minimized Veusz File" is checked, the saved .vszh5 contains
        # ONLY the datasets that the plot widget tree actually references
        # (xData / yData / labels / scalePoints / ...).  The nested
        # "Generate Full Veusz file" sub-checkbox is enabled only when the
        # parent is on; when also checked, a second file with the legacy
        # full dataset list is written alongside (suffix "_full").
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
        # Gate the sub-checkbox on the parent's state.
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
        if not self.selected_files:
            QMessageBox.information(self, "No files", "Select one or more FITS files first.")
            return
        backend_idx = self.backend_combo.currentIndex()
        backend = ["both", "veusz", "astropy"][backend_idx]
        self.cache.cfg.rss_high_water_mb = int(self.rss_spin.value())

        if self.veusz_doc is None and self.show_embed_cb.isChecked():
            self.veusz_doc = open_embedded("FITS_AutoPlot")
        elif self.veusz_doc is None:
            # invisible doc: still required to build the project
            self.veusz_doc = vz_embed.Embedded("FITS_AutoPlot", hidden=True)

        self.process_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.show_progress_bars(len(self.selected_files))

        self.worker = FITSBatchWorker(
            self.selected_files, backend, int(self.thread_spin.value()),
            self.cache, parent=self,
            skip_images=bool(self.skip_images_cb.isChecked()),
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_failed)
        self.log("Starting batch: %d file(s), backend=%s, threads=%d" %
                 (len(self.selected_files), backend, int(self.thread_spin.value())))
        self.worker.start()

    def _on_worker_progress(self, done: int, total: int, key: str) -> None:
        self.progress_bar.setValue(done)
        self.log("  read [%d/%d] %s" % (done, total, os.path.basename(key)))
        self.update_mem_label(self.cache.rss_mb())

    def _on_worker_failed(self, msg: str) -> None:
        self.log("Batch failed: %s" % msg)
        QMessageBox.critical(self, "Batch failed", msg)
        self.process_button.setEnabled(True)
        self.hide_progress_bars()

    def _on_worker_done(self, results: Dict[str, Any]) -> None:
        backend_idx = self.backend_combo.currentIndex()
        backend = ["both", "veusz", "astropy"][backend_idx]
        emit_datestr = bool(self.datestr_cb.isChecked())
        skip_images = bool(self.skip_images_cb.isChecked())
        # v0.0.9 broken-axis / overlay knobs
        gap_k = float(self.gap_k_spin.value())
        # v0.0.16: spinbox is now in HOURS -- convert to MJD-days for
        # detect_time_breaks (which compares against the x-array units).
        gap_absolute_hours = float(self.gap_abs_spin.value())
        gap_absolute = gap_absolute_hours / 24.0
        # v0.0.19: if the user typed a free-form gap string, that takes
        # precedence (already in MJD-days).
        gap_string_text = ""
        try:
            gap_string_text = self.gap_abs_str.text().strip()
        except Exception:
            gap_string_text = ""
        if gap_string_text:
            gap_absolute = parse_gap_string(gap_string_text,
                                            log_cb=self.log)
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
        # Keep the GUI responsive across hundreds of files: pump the Qt
        # event loop between every push so the log pane scrolls live and
        # the window doesn't appear "stuck" during a long batch insert.
        app = QApplication.instance()
        total = len(results)
        # Build a column-progress callback that updates the per-file bar.
        # Defined here so it captures the GUI handles, then passed into
        # push_to_veusz() which counts column-level operations.
        def _col_cb(done: int, total_ops: int) -> None:
            if self.column_progress_bar.maximum() != max(1, total_ops):
                self.column_progress_bar.setRange(0, max(1, total_ops))
            self.column_progress_bar.setValue(done)
            if app is not None:
                app.processEvents()
        file_records = []  # accumulator for the unit-overlay post-pass
        for idx, (path, data) in enumerate(results.items(), start=1):
            if isinstance(data, Exception):
                self.log("  ERROR processing %s: %s" % (path, data))
                self.parse_progress_bar.setValue(idx)
                if app is not None:
                    app.processEvents()
                continue
            self.log("Inserting datasets [%d/%d] %s"
                     % (idx, total, os.path.basename(path)))
            # Pre-size the column bar from what the read pass produced.
            n_cols = len(data.get("columns") or {})
            n_imgs = 0 if skip_images else len(data.get("images") or {})
            self.begin_column_progress(os.path.basename(path),
                                       max(1, n_cols + n_imgs))
            try:
                push_to_veusz(self.veusz_doc, path, data, backend,
                              log_cb=self.log, emit_datestr=emit_datestr,
                              column_cb=_col_cb,
                              skip_images=skip_images,
                              plot_individual=plot_individual,
                              gap_k=gap_k,
                              gap_absolute=gap_absolute,
                              datetime_duplicate=datetime_duplicate,
                              datetime_label_density_pct=datetime_label_density_pct,
                              datetime_emit_numeric_dt=datetime_emit_numeric_dt,
                              datetime_emit_text_dt=datetime_emit_text_dt,
                              hide_keys=hide_keys)
            except Exception as exc:
                self.log("  push_to_veusz failed for %s: %s" % (path, exc))
            else:
                # Record this file for the overlay post-pass.
                # v0.0.12 channel-tag: also carry tag_columns/tag_groups so
                # build_unit_overlay_pages can split each numeric column
                # into one trace per unique tag-tuple.
                file_records.append({
                    "base": safe_dsname(data.get("base_name") or
                                        os.path.basename(path)),
                    "columns": data.get("columns") or {},
                    "units": data.get("units") or {},
                    "sort_key": data.get("sort_key"),
                    "tag_columns": data.get("tag_columns") or {},
                    "tag_groups": data.get("tag_groups") or {},
                })
            self.parse_progress_bar.setValue(idx)
            if app is not None:
                app.processEvents()
        # Build cross-file unit-overlay pages.
        if file_records:
            try:
                build_unit_overlay_pages(
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
                self.log("  build_unit_overlay_pages failed: %s" % exc)
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
            # Mark the primary (first-written) project for the Open-in-Veusz button.
            self.mark_project_saved(written_paths[0])
            QMessageBox.information(
                self, "Saved", "Wrote:\n%s" % "\n".join(written_paths),
            )
        except Exception as exc:
            self.log("Save failed: %s" % exc)
            QMessageBox.critical(self, "Save failed", str(exc))

    # ----- shutdown --------------------------------------------------------
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
    win = FITSAutoPlotWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
