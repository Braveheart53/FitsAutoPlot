# -*- coding: utf-8 -*-
"""
=============================================================================
FITS_AutoPlot_plugin.py
-----------------------------------------------------------------------------
# %% Header Info
--------

Veusz Tools plugin equivalent of FITS_AutoPlot.py.  Drop this file into
your Veusz plugins folder (or load it via
``Edit > Preferences > Plugins > Add``) and the entry

    Tools -> NRAO / FitsAutoPlot -> FITS AutoPlot (Batch)

will appear.  Selecting it opens the same Touchstone-style qtpy GUI as the
standalone script and exports a separate native .vszh5 project using h5py.
The current Veusz document is left untouched.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.3.1: Progress throttle; GPU checkbox for NVIDIA or AMD.
Date: 2026-11-09
#              * Column progress at most every 50 ms (ProgressThrottle).
#              * GPU checkbox and tooltip as FITS AutoPlot 0.4.1; the log
#                reports GPU / CPU sorts per batch.
#              * Reading stays in threads: a process pool would re-run
#                Veusz's own main module in every 'spawn' worker.
# %%%%% Function Descriptions
        _PluginBatchDialog (class): Modal version of the AutoPlot main window for in-plugin use.
        FITSAutoPlotPlugin (class): Veusz menu entry for native FITS project export, not host-document editing.
# %%%%% Variable Descriptions
    Define all utilized variables
        _HOST_PLUGIN_PATH: Make sibling files importable when Veusz loads the plugin from a path Veusz 3.x executes plugin text inside Document.loadPlugins(), where __file__...
        _THIS_DIR: folder of this plugin file (the Veusz host's plugin path when loaded by Veusz)
# %%%%% More Info
    Veusz: Tools -> NRAO / FitsAutoPlot -> FITS AutoPlot (Batch).  The dialog
    has the standalone program's options; the result is written as a separate
    native .vszh5 project and the host Veusz document is not changed.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.3.0: Gap rule and local ISO 8601 on every page (standalone 0.4.0).
Date: 2026-11-09
#              * Small gap: a trace breaks where its step dt > K x median(dt)
#                (or a fixed time).  Large gap: the x axis breaks where no
#                trace has data for dt > n x the small-gap threshold (new
#                "Large gap n", default 100; field gap_axis_n) or a fixed time
#                (new "Large gap fixed", field gap_axis_string).
#              * Every page plots MJD as a Veusz date-time axis in local ISO
#                8601 (the exported NativeDocument is laid out on save).
#              * Pages in time order, not thread completion order.
#              * "Duplicate plots" relabelled "Legacy: also build duplicate
#                _dt / _dt_labels pages".
#              * FIX (Python 3.12 crash): the memory monitor callback is the
#                dialog's bound method rss_crossed (held weakly); the lambda
#                it replaced let the monitor thread drop the last reference
#                and destroy the dialog outside the GUI thread.
#                append_log() is thread-safe (queued to the GUI thread).
#              * Docstrings for every function and method; author contact
#                lines in the header.
#              * Tooltips (pop-up help) on every button, box and field that
#                had none; 16 widgets.
# %%%%% Function Descriptions
        _PluginBatchDialog (class): Modal version of the AutoPlot main window for in-plugin use.
        FITSAutoPlotPlugin (class): Veusz menu entry for native FITS project export, not host-document editing.
# %%%%% Variable Descriptions
    Define all utilized variables
        _HOST_PLUGIN_PATH: Make sibling files importable when Veusz loads the plugin from a path Veusz 3.x executes plugin text inside Document.loadPlugins(), where __file__...
        _THIS_DIR: folder of this plugin file (the Veusz host's plugin path when loaded by Veusz)
# %%%%% More Info
    Veusz: Tools -> NRAO / FitsAutoPlot -> FITS AutoPlot (Batch).  The dialog
    has the standalone program's options; the result is written as a separate
    native .vszh5 project and the host Veusz document is not changed.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.2.0: Dialog-only theme; local ISO date-time pages by default.
Date: 2026-11-09
#              * The legacy labelled _dt page is off by default
#                (datetime_emit_numeric_dt field and check box); the
#                dt_labels pages are Veusz date-time axes in local ISO 8601
#                time (see the standalone program 0.3.0).
#              * The Dark / Light choice styles only this dialog (below).
#              (2026-11-05) The Dark / Light choice styles only this dialog
#              (_suite_gui.dialog_stylesheet).  It used to set the palette of
#              the whole application, which re-themed the Veusz window the
#              plugin runs in.
# %%%% 0.1.0: Native project export plugin; leave the host document untouched.
Date: 2026-10-11
#              Veusz plugin API is used only for host menu integration.
#              Output uses the same h5py writer as the standalone script.
#              Adds output_file, minimized and also_full plugin fields.
#              Can also launch the standalone GUI without a Veusz install.
#
# %%%% 0.0.19: Add gap_string + hide_keys plugin fields; plumb through.
Date: 2026-05-16
#              Two new plugin fields:
#                * ``gap_string`` (FieldText, default "") -- optional
#                  free-form gap string (e.g. "5m3d2h").  When non-empty
#                  it OVERRIDES the existing ``gap_absolute`` (hours)
#                  field by being parsed via ``parse_gap_string`` from
#                  ``_autoplot_common``.
#                * ``hide_keys`` (FieldBool, default False) -- when True
#                  every key widget added by ``push_to_veusz`` and
#                  ``build_unit_overlay_pages`` has its Border hidden.
#              The batch dialog gained matching ``self.gap_abs_str``
#              QLineEdit and ``self.hide_keys_cb`` QCheckBox.  Both
#              fields are forwarded as the new ``hide_keys`` kwarg on
#              push_to_veusz / build_unit_overlay_pages calls.
#
# %%%% 0.0.18: Unit-aware time-break detection (no field changes).
Date: 2026-05-16
#              Plugin-side version bump only.  The unit-aware break-
#              detection fix lives entirely in _autoplot_common
#              (``detect_time_breaks_unit_aware``) and the engine
#              entry points in FITS_AutoPlot, which the plugin already
#              calls with the same ``gap_absolute`` (MJD-days) value.
#              No plugin field schema changes; the existing
#              ``gap_absolute`` saved-field (hours) continues to be
#              divided by 24.0 in the plugin submit handler and
#              passed through to the engine as MJD-days, where the
#              engine now converts to column units automatically.
#              Revision history kept in DESCENDING semantic-version
#              order.
#
# %%%% 0.0.17: Sentinel channel-tag dt-overlay filter (no field changes).
Date: 2026-05-16
#              Plugin inherits the v0.0.17 FITS-side fix via
#              ``build_unit_overlay_pages``: dt overlay emission now
#              skips placeholder channel-tag tuples like
#              ``("dataset", "dataset")`` that previously manufactured
#              spurious broken-axis breaks on the combined-MJD detector.
#              No new plugin fields; the minimized-save UI lives on the
#              standalone GUI only (the plugin operates inside an open
#              Veusz document and the user saves it via Veusz itself).
#              Revision history kept in DESCENDING semantic-version
#              order.
#
# %%%% 0.0.16: dt_labels page mode='datetime' + broken-axis parity.
Date: 2026-05-16
#                * No plugin-field changes vs. v0.0.15.  The plugin
#                  surface is unchanged; the engine (FITS_AutoPlot.py
#                  + _autoplot_common.py) does all the new work:
#                  emits a numeric Veusz-datetime-seconds dataset
#                  alongside the text-x dataset, binds the dt_labels
#                  page to it, sets the x axis to mode='datetime',
#                  and adds broken-axis parity with the seconds-axis
#                  dt page.
#                * Header bumped to 0.0.16 only so the plugin and
#                  engine version strings stay in lock-step in the
#                  Veusz Tools menu.
#                * GUI: "Absolute gap (units of x; 0=auto)" spinbox is
#                  renamed "Manual gap (hours; 0=auto)" and the value
#                  is divided by 24 before being passed to the engine,
#                  so MJD-axis time gaps are entered in hours.
#
# %%%% 0.0.15: Density-pct date labels + text-x dt_labels page variant.
Date: 2026-05-16
#                * Plugin-side equivalent of the FITS_AutoPlot.py
#                  v0.0.15 change.  The legacy ``datetime_full_labels``
#                  boolean field is REPLACED by:
#                    - ``datetime_label_density_pct`` (FieldInt 0..100,
#                      default 10): controls how many evenly-spaced
#                      anchor labels are emitted on the numeric-x dt
#                      page.  0 = no labels, 100 = one label per finite
#                      data point.
#                    - ``datetime_emit_numeric_dt`` (FieldBool,
#                      default True): toggle the v0.0.14 numeric-x dt
#                      page (``<base>_<hdu>_dt`` / ``Overlay_<unit>_dt``).
#                    - ``datetime_emit_text_dt`` (FieldBool,
#                      default True): toggle the NEW v0.0.15 text-x
#                      dt_labels page (``<base>_<hdu>_dt_labels`` /
#                      ``Overlay_<unit>_dt_labels``) which uses a per-
#                      point text dataset as xData with axis
#                      ``mode='labels'``.  Sample spacing is uniform.
#                * Dialog UI: the v0.0.14 ``full_labels_cb`` checkbox is
#                  replaced by a QSpinBox 0..100 (with " %" suffix) and
#                  two QCheckBoxes for the two dt page variants.
#                * The old ``datetime_full_labels`` field is still
#                  accepted in saved-doc plugin invocations as a back-
#                  compat shim handled by FITS_AutoPlot.push_to_veusz
#                  and build_unit_overlay_pages (True -> 100,
#                  False -> 10).
#                * GPU + parallelization re-audit: no new sort sites
#                  introduced by the plugin path; worker pool and
#                  GPU-sort wiring are inherited unchanged from
#                  FITS_AutoPlot.
#                * Revision history kept in DESCENDING semantic-
#                  version order.
#
# %%%% 0.0.14: Datetime via xy.labels per-point text labels.
Date: 2026-05-16
#                * Mirrors FITS_AutoPlot.py v0.0.14: the dt-duplicate
#                  pages now bind a per-point text dataset to
#                  xy.labels.val (rendering one date string per data
#                  point) instead of using a Veusz datetime x axis
#                  (which is broken on Veusz 3.4 internals).
#                * Adds plugin field ``datetime_full_labels`` and a
#                  matching dialog checkbox; threads the flag through
#                  ``push_to_veusz`` and ``build_unit_overlay_pages``.
#                * No SetDataDateTime calls in the dt path.

# %%%% 0.0.13: Inherits identity-stable trace styling + datetime-duplicate
Date: 2026-05-16
#              hardening from FITS_AutoPlot.py.  No plugin-side code
#              changes are required: push_to_veusz, build_unit_overlay_
#              pages, and the page builders are shared with the
#              standalone GUI, so the new apply_trace_style() wiring and
#              set_datetime_dataset() coercion both take effect inside
#              the plugin entry point automatically.  Header bumped for
#              version visibility in the Veusz Tools menu.

# %%%% 0.0.12: Inherits the combined-in-time overlay semantics from
Date: 2026-05-16
#              FITS_AutoPlot.py v0.0.12 -- plugin code unchanged because
#              build_unit_overlay_pages does the heavy lifting.
#              Channel-tag row model: the plugin's file_records
#              accumulator now also forwards ``tag_columns`` and
#              ``tag_groups`` from FITSProcessor.read so the overlay
#              builder can split each numeric column into one trace
#              per unique (CHANNELA, CHANNELB) tuple.

# %%%% 0.0.11: Plugin-side support for datetime-duplicate plots (mirrors
Date: 2026-05-16
#              FITS_AutoPlot.py v0.0.11).  _PluginBatchDialog gained a
#              'Duplicate plots with datetime X axis (YYYY-MM-DD HH:MM:SS)'
#              checkbox; ``apply()`` threads the resulting
#              ``datetime_duplicate`` boolean through every push_to_veusz()
#              call and through ``build_unit_overlay_pages``.

# %%%% 0.0.10: Plugin-side support for the optional GPU acceleration
Date: 2026-05-16
#              (mirrors FITS_AutoPlot.py v0.0.10).  _PluginBatchDialog
#              now exposes a 'Use GPU acceleration (CuPy)' checkbox that
#              is disabled when CuPy is not importable on the host;
#              ``apply()`` toggles the process-wide flag via
#              ``enable_gpu()`` before pushing.

# %%%% 0.0.9: Plugin-side support for broken-axis + unit-overlay (mirrors
Date: 2026-05-16
#             FITS_AutoPlot.py v0.0.9).  _PluginBatchDialog now exposes
#             ``Gap K (× median Δt)``, ``Absolute gap``, and
#             ``Combined plots only`` controls; ``apply()`` threads
#             ``plot_individual``, ``gap_k``, ``gap_absolute`` through
#             every ``push_to_veusz()`` call, accumulates ``file_records``
#             across the batch, and post-builds unit-overlay pages by
#             calling ``FITS_AutoPlot.build_unit_overlay_pages`` on the
#             host ``interface`` doc.

# %%%% 0.0.8: Version-bumped in lockstep with the rest of the AutoPlot
Date: 2026-05-16
#             suite (FITS_AutoPlot.py / Franks_AutoPlot.py / _autoplot_common.py)
#             which gained 'Open in Veusz...' buttons + a parallelization
#             audit (MAX_THREADS doubled in standalone tools).
#             No 'Open in Veusz' button is added to this plugin dialog: the
#             plugin already runs *inside* the live Veusz application, so
#             the freshly pushed datasets are immediately visible in the
#             host document -- there is nothing external to launch.
#             The plugin transparently picks up the MAX_THREADS bump and
#             the vectorized Franks parser through its imports from the
#             standalone modules; no code changes are required here.

# %%%% 0.0.7: Added two new progress bars to the plugin dialog -- a
Date: 2026-05-16
#             'Parsing/pushing' file-level bar and a 'Current file -
#             columns' per-file column bar that ticks per Veusz dataset
#             as push_to_veusz() pours that file into the active document.
#             Also added a 'Skip image HDUs' checkbox to the modal dialog
#             and a new ``skip_images`` boolean field, threaded through
#             FITSProcessor() and push_to_veusz() so the plugin honours
#             the same speed knob as the standalone window.

# %%%% 0.0.6: Added Spyder-style cell markers (``# %% TITLE`` / ``# %%%``)
Date: 2026-05-16
#             on the existing dashed banner blocks so the file is
#             navigable in Spyder's Outline / cell navigator.  Pure
#             cosmetic change -- no runtime behaviour modified.

# %%%% 0.0.5: Inherits the empty-images early-exit from FITS_AutoPlot.py
Date: 2026-05-16
#             0.0.5: push_to_veusz() and _build_pages() now skip the image-
#             push and image-page-creation loops explicitly when
#             ``data['images']`` is empty (the normal case for NRAO
#             OnePpsDeltas-only files), with an explicit log line so the
#             user knows the skip was intentional and the plugin is not
#             hung.

# %%%% 0.0.4: FITS-unit-warning suppression.  The plugin already inherits
Date: 2026-05-16
#             register_nrao_fits_units() and the suppress_fits_unit_warnings()
#             context manager via FITSProcessor (imported from
#             FITS_AutoPlot.py).  As a belt-and-suspenders measure we
#             explicitly call register_nrao_fits_units() at plugin module
#             load and wrap the entire apply() FITS-reading loop in the
#             suppression context manager so the Veusz log stays clean
#             during batch runs of 1PPS-delta files.

# %%%% 0.0.3: NaN-preserving emission policy (inherits push_to_veusz from
Date: 2026-05-16
#             FITS_AutoPlot.py): numeric NaN floats are kept verbatim in
#             Veusz numeric datasets; non-finite MJDs become the sentinel
#             string ``"NaN"`` in the date-string text datasets so row
#             counts always match their numeric companions.

# %%%% 0.0.2: Added "Generate MJD->date strings" checkbox to the modal
Date: 2026-05-16
#             dialog and an ``emit_datestr`` boolean field so the same
#             option is available from the Veusz Tools menu.

# %%%% 0.0.1: Initial plugin equivalent of FITS_AutoPlot.py
Date: 2026-05-16


"""
# %% Imports
from __future__ import annotations

__version__ = "0.3.1"          # Semantic Versioning: see the Revisions entries above

# %%% IMPORTS - Standard library

import os
import sys
import threading
import traceback
from typing import Any, Dict, List

# Make sibling files importable when Veusz loads the plugin from a path
# Veusz 3.x executes plugin text inside Document.loadPlugins(), where
# __file__ belongs to Veusz and the selected filename is a local "plugin".
_HOST_PLUGIN_PATH = None
_module_file = globals().get("__file__")
if not _module_file or globals().get("__name__", "").startswith("veusz."):
    # 3.6/4.1 use exec(text, {}), which also omits __file__. Inspect only
    # the loader's filename, never any document or dataset content.
    _frame = sys._getframe()
    while _frame is not None:
        if _frame.f_code.co_name == "loadPlugins":
            _candidate = _frame.f_locals.get("plugin")
            if isinstance(_candidate, str) and os.path.isfile(_candidate):
                _HOST_PLUGIN_PATH = _candidate
                break
        _frame = _frame.f_back
    del _frame
if not (_HOST_PLUGIN_PATH or _module_file):
    raise RuntimeError("Cannot locate plugin file; import this file as a module")
_THIS_DIR = os.path.dirname(os.path.abspath(_HOST_PLUGIN_PATH or _module_file))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# %%% IMPORTS - Veusz plugin API
try:
    import veusz.plugins as vzp  # menu registration only, never file generation
except ImportError:
    vzp = None

# %%% IMPORTS - shared AutoPlot infrastructure
# Reuse the standalone implementation
from FITS_AutoPlot import (                       # noqa: E402
    MAX_THREADS, DEFAULT_RSS_HIGH_WATER_MB,
    FITSProcessor, push_to_veusz,
)
# v0.0.9: pull in the overlay post-pass helper from the shared module.
from FITS_AutoPlot import (                       # noqa: E402
    build_unit_overlay_pages,
)
from _autoplot_common import (                    # noqa: E402
    QApplication, QFileDialog, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QFormLayout,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QProgressBar, QTextEdit, QLineEdit,
    MemoryAwareCache, MemoryMonitorConfig, MemoryMonitor,
    Signal, QUEUED_CONNECTION, ProgressThrottle,
    run_in_threadpool, safe_dsname,
    is_gpu_available, enable_gpu, gpu_backend_name,
    gpu_enabled, gpu_tooltip, gpu_usage_text, reset_gpu_usage,
    register_nrao_fits_units, suppress_fits_unit_warnings,
    # v0.0.19: string gap parser
    parse_gap_string, GAP_STRING_HELP,
    begin_native_plugin_export, finish_native_plugin_export,
)

# Idempotent: ensures the NRAO custom FITS unit aliases ('none',
# 'NanoSeconds') are registered with astropy.units even if this plugin
# happens to be loaded before FITS_AutoPlot.py would otherwise run its
# module-level registration call.
register_nrao_fits_units()
# qtpy's QDialog
try:
    from qtpy.QtWidgets import QDialog
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QDialog


# ============================================================================
# %% A modal qtpy dialog mirroring the standalone GUI layout
# ============================================================================
class _PluginBatchDialog(QDialog):
    """Modal version of the AutoPlot main window for in-plugin use."""

    # 0.3.0: append_log() from another thread is delivered through this.
    _log_from_thread = Signal(str)

    def __init__(self, parent=None, default_mode: str = "dark") -> None:
        """Modal batch dialog shown inside Veusz: file list, the same options as
        the standalone program (threads, RSS threshold, small/large gaps, overlay
        and output choices), progress bars and a log.  Only this dialog is themed
        (``default_mode``); the Veusz host keeps its own look.
        """
        super().__init__(parent)
        self._log_from_thread.connect(self.append_log, QUEUED_CONNECTION)
        self.setWindowTitle("FITS AutoPlot - Veusz Plugin")
        self.resize(900, 700)
        self.selected_files: List[str] = []
        self._theme_mode = default_mode

        root = QVBoxLayout(self)

        # Theme toggle row (textual menu would require a QMainWindow)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.setToolTip("Colours of this dialog only (Veusz keeps its own look).")
        self.theme_combo.addItems(["Dark Mode", "Light Mode"])
        self.theme_combo.setCurrentIndex(0 if default_mode == "dark" else 1)
        self.theme_combo.currentIndexChanged.connect(self._theme_changed)
        trow.addWidget(self.theme_combo)
        trow.addStretch()
        root.addLayout(trow)

        # File selection
        root.addWidget(QLabel("Input files:"))
        self.file_list = QListWidget()
        self.file_list.setToolTip("Files to process; add them with Browse, remove all with Clear.")
        self.file_list.setMinimumHeight(160)
        root.addWidget(self.file_list)
        brow = QHBoxLayout()
        b1 = QPushButton("Browse...")
        b1.setToolTip("Add files to the list (several can be selected at once).")
        b1.clicked.connect(self._browse)
        brow.addWidget(b1)
        b2 = QPushButton("Clear")
        b2.setToolTip("Remove every file from the list (the files are not touched).")
        b2.clicked.connect(self._clear)
        brow.addWidget(b2)
        root.addLayout(brow)

        # Options
        form = QFormLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.setToolTip(
            "Reader: astropy. The result is a native .vszh5 file (the old "
            "'veusz' and 'both' choices are no longer needed).")
        self.backend_combo.addItems(["astropy (native HDF5 output)"])
        form.addRow("Import backend:", self.backend_combo)
        self.thread_spin = QSpinBox()
        self.thread_spin.setToolTip("Number of files read at the same time (threads).")
        self.thread_spin.setRange(1, max(1, (os.cpu_count() or 4) * 2))
        self.thread_spin.setValue(MAX_THREADS)
        form.addRow("Worker threads:", self.thread_spin)
        self.rss_spin = QSpinBox()
        self.rss_spin.setToolTip(
            "Resident memory (MiB) above which large arrays are moved from "
            "RAM to temporary files on disk (numpy.memmap) - lower it on a "
            "small computer.")
        self.rss_spin.setRange(128, 65536)
        self.rss_spin.setSingleStep(128)
        self.rss_spin.setValue(DEFAULT_RSS_HIGH_WATER_MB)
        form.addRow("RSS spill threshold (MiB):", self.rss_spin)
        root.addLayout(form)

        self.datestr_cb = QCheckBox(
            "Generate MJD -> date strings (YYYY-MM-DD_HH:MM:SS) datasets"
        )
        self.datestr_cb.setToolTip(("Also store each MJD column as text 'YYYY-MM-DD_HH:MM:SS' "
            '(UTC) - only needed by old Veusz documents; the time axes '
            'are date-times without it.'))
        self.datestr_cb.setChecked(False)
        root.addWidget(self.datestr_cb)

        self.skip_images_cb = QCheckBox(
            "Skip image HDUs (faster -- recommended for NRAO 1PPS files)"
        )
        self.skip_images_cb.setToolTip(
            "Do not read image HDUs. GBT sampler2log and 1PPS files have "
            "none, so this only saves time.")
        self.skip_images_cb.setChecked(False)
        root.addWidget(self.skip_images_cb)

        # --- Broken-axis / overlay controls (v0.0.9) ----------------------
        form2 = QFormLayout()
        self.gap_k_spin = QDoubleSpinBox()
        self.gap_k_spin.setRange(1.0, 1000.0)
        self.gap_k_spin.setSingleStep(1.0)
        self.gap_k_spin.setDecimals(2)
        self.gap_k_spin.setValue(10.0)
        self.gap_k_spin.setToolTip(
            "Line break: a trace stops and restarts where its sample step "
            "dt > K x median(dt) of that trace; the time axis stays one axis.")
        form2.addRow("Small gap K (line break: dt > K × median dt):", self.gap_k_spin)
        # 0.3.0: large gaps may also cut the axis (n x K x median dt).
        self.gap_axis_spin = QDoubleSpinBox()
        self.gap_axis_spin.setRange(0.0, 1.0e6)
        self.gap_axis_spin.setSingleStep(10.0)
        self.gap_axis_spin.setDecimals(1)
        self.gap_axis_spin.setValue(100.0)
        self.gap_axis_spin.setSpecialValueText("0 = no axis breaks")
        self.gap_axis_spin.setToolTip(
            "Large gap: the x axis is cut only where no trace has data for "
            "dt > n x the small-gap threshold; 0 = never.")
        form2.addRow("Large gap n (axis break: dt > n × small gap):", self.gap_axis_spin)
        self.gap_axis_str = QLineEdit()
        self.gap_axis_str.setPlaceholderText("e.g. 12h, 3d, 1m  (overrides n)")
        self.gap_axis_str.setToolTip(GAP_STRING_HELP)
        form2.addRow("Large gap fixed (text):", self.gap_axis_str)

        self.gap_abs_spin = QDoubleSpinBox()
        self.gap_abs_spin.setToolTip(
            "Small gap as a fixed time in hours: the line breaks where two "
            "samples are further apart. 0 = use K x median step.")
        self.gap_abs_spin.setRange(0.0, 1e12)
        self.gap_abs_spin.setDecimals(6)
        self.gap_abs_spin.setSingleStep(1.0)
        self.gap_abs_spin.setValue(0.0)
        # v0.0.16: spinbox is in HOURS; converted to MJD-days when read.
        form2.addRow("Small gap fixed (hours; 0 = K × median dt):", self.gap_abs_spin)

        # v0.0.19: free-form gap-string input (overrides spinbox when set).
        self.gap_abs_str = QLineEdit()
        self.gap_abs_str.setPlaceholderText("e.g. 5m3d2h  (overrides spinbox)")
        self.gap_abs_str.setToolTip(
            "Optional gap string.  Units: m = months (~30.44 days), "
            "d = days, h = hours.  Examples: '3d', '120h', '5m 3d 2h'."
        )
        form2.addRow("Small gap fixed (text; overrides hours):", self.gap_abs_str)
        root.addLayout(form2)

        # v0.0.19: hide legend keys on every page.
        self.hide_keys_cb = QCheckBox("Hide legend keys on all pages")
        self.hide_keys_cb.setToolTip(
            "Hide the legend (key) on every page; trace names stay in the "
            "data.")
        self.hide_keys_cb.setChecked(False)
        root.addWidget(self.hide_keys_cb)

        self.combined_only_cb = QCheckBox(
            "Combined (overlay) plots only -- skip per-file pages"
        )
        self.combined_only_cb.setToolTip(
            "Build only the overlay pages (all files on one graph per unit)"
            " and no page per file - much faster to open for hundreds of "
            "files.")
        self.combined_only_cb.setChecked(False)
        root.addWidget(self.combined_only_cb)

        # --- Datetime-duplicate plots (v0.0.11) ---------------------------
        self.datetime_dup_cb = QCheckBox(
            "Legacy: also build duplicate _dt / _dt_labels pages"
        )
        self.datetime_dup_cb.setToolTip(
            "Old style: also build the duplicate _dt / _dt_labels pages. "
            "Every page already has a local ISO 8601 date-time axis, so "
            "this is off by default.")
        self.datetime_dup_cb.setChecked(False)
        root.addWidget(self.datetime_dup_cb)

        # --- v0.0.15: density-pct + dt page variant toggles ----------------
        # Replaces the v0.0.14 binary full_labels_cb with a 0..100
        # percentage spinbox and two checkboxes for the two dt page
        # variants (numeric-x and text-x).
        dens_row = QHBoxLayout()
        dens_row.addWidget(QLabel("Date-label density on dt pages:"))
        self.label_density_spin = QSpinBox()
        self.label_density_spin.setRange(0, 100)
        self.label_density_spin.setValue(10)
        self.label_density_spin.setSuffix(" %")
        self.label_density_spin.setToolTip(
            "Fraction of finite points that get a YYYY-MM-DD HH:MM:SS "
            "label on the numeric-x dt page.  0 = no labels, "
            "100 = one label per finite point.  10 (default) matches "
            "the v0.0.14 sparse behaviour."
        )
        dens_row.addWidget(self.label_density_spin)
        dens_row.addStretch(1)
        root.addLayout(dens_row)
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
        root.addWidget(self.emit_numeric_dt_cb)
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
        root.addWidget(self.emit_text_dt_cb)

        # --- GPU acceleration (NVIDIA or AMD, optional) (v0.0.10; 0.3.1) ---
        self.gpu_cb = QCheckBox("Sort large arrays on the GPU (NVIDIA or AMD)")
        self.gpu_cb.setChecked(False)
        self.gpu_cb.setEnabled(is_gpu_available())
        self.gpu_cb.setToolTip(gpu_tooltip())
        root.addWidget(self.gpu_cb)

        self.log = QTextEdit()
        self.log.setToolTip("Progress and messages of this session, with time stamps.")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        root.addWidget(self.log)
        # Reading-stage bar (file-level, ticked from worker threads)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        # Parse / push-stage bar (file-level, ticked on the GUI thread)
        self.parse_progress = QProgressBar()
        self.parse_progress.setVisible(False)
        root.addWidget(self.parse_progress)
        # Per-column bar for the currently-pushing file (GUI thread)
        self.column_progress = QProgressBar()
        self.column_progress.setVisible(False)
        root.addWidget(self.column_progress)

        # Action buttons
        arow = QHBoxLayout()
        self.process_btn = QPushButton("Process Files")
        self.process_btn.setToolTip(
            "Close the dialog and process the listed files into a new "
            "native .vszh5 project (the open Veusz document is not "
            "changed).")
        self.process_btn.clicked.connect(self.accept)   # close dialog -> plugin reads selection
        arow.addWidget(self.process_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Close the dialog without processing.")
        self.cancel_btn.clicked.connect(self.reject)
        arow.addWidget(self.cancel_btn)
        root.addLayout(arow)

        self._style_dialog(default_mode)

    # ----- helpers ---------------------------------------------------------
    def _theme_changed(self, idx: int) -> None:
        """Theme combo box handler: 0 = dark, 1 = light."""
        mode = "dark" if idx == 0 else "light"
        self._theme_mode = mode
        self._style_dialog(mode)

    def _style_dialog(self, mode: str) -> None:
        """Theme this dialog only (the Veusz host keeps its own look)."""
        try:
            from _suite_gui import dialog_stylesheet
            self.setStyleSheet(dialog_stylesheet(mode))
        except Exception:  # pragma: no cover - _suite_gui missing
            pass

    def _browse(self) -> None:
        """Ask for FITS files and add the new ones to the list."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS files", "",
            "FITS files (*.fits *.fit *.fits.gz *.fit.gz);;All files (*)"
        )
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
        self.file_list.clear()
        for p in self.selected_files:
            self.file_list.addItem(os.path.basename(p))

    def _clear(self) -> None:
        """Empty the file list."""
        self.selected_files.clear()
        self.file_list.clear()

    def append_log(self, msg: str) -> None:
        """Append ``msg`` to the dialog's log pane (safe from any thread: off
        the GUI thread the text is queued to it, 0.3.0)."""
        if threading.current_thread() is not threading.main_thread():
            self._log_from_thread.emit(msg)
            return
        self.log.append(msg)

    def rss_crossed(self, rss: float) -> None:
        """Memory monitor callback (monitor thread): log the RSS crossing."""
        self.append_log("RSS over high-water mark (%.1f MiB) -- spilling." % rss)


# ============================================================================
# %% THE PLUGIN
# ============================================================================
class FITSAutoPlotPlugin(vzp.ToolsPlugin if vzp is not None else object):
    """Veusz menu entry for native FITS project export, not host-document editing."""

    menu = ("NRAO / FitsAutoPlot", "FITS AutoPlot (Batch)")
    name = "FITS AutoPlot"
    description_short = "Batch import NRAO FITS files (column oriented)"
    description_full = (
        "Open a Touchstone-style GUI to select one or more FITS files "
        "(compressed .gz allowed), choose threading and RSS spill threshold, "
        "then export a native .vszh5 file through h5py. The current Veusz "
        "document is not modified."
    )

    def __init__(self) -> None:
        """Declare the plugin fields shown by Veusz (output file, threads, RSS
        threshold, small/large gap settings, theme, ...).  Raises RuntimeError
        outside the Veusz plugin host - run FITS_AutoPlot.py there instead.
        """
        if vzp is None:
            raise RuntimeError("Use FITS_AutoPlot.py outside the Veusz plugin host")
        self.fields = [
            vzp.FieldText("output_file", descr="Native .vszh5 output (blank: ask)",
                          default=""),
            vzp.FieldBool("minimized", descr="Save only plot-used datasets",
                          default=True),
            vzp.FieldBool("also_full", descr="Also save a full companion project",
                          default=False),
            vzp.FieldText("default_theme",
                          descr="Default theme (dark/light)",
                          default="dark"),
            vzp.FieldInt("max_threads",
                         descr="Maximum worker threads",
                         default=MAX_THREADS, minval=1, maxval=128),
            vzp.FieldInt("rss_mb",
                         descr="RSS spill threshold (MiB)",
                         default=DEFAULT_RSS_HIGH_WATER_MB,
                         minval=128, maxval=65536),
            vzp.FieldText("preselected_files",
                          descr="Optional: ;-separated FITS file paths "
                                "(leave blank to pick interactively)",
                          default=""),
            vzp.FieldBool("emit_datestr",
                          descr="Also create MJD -> date-string datasets "
                                "(YYYY-MM-DD_HH:MM:SS)",
                          default=False),
            vzp.FieldBool("skip_images",
                          descr="Skip image HDUs (faster -- recommended "
                                "for NRAO 1PPS files)",
                          default=False),
            # v0.0.11: pre-seed the datetime-duplicate checkbox from a
            # plugin field so saved-document plugin invocations can opt in
            # without having to re-tick the dialog checkbox manually.
            vzp.FieldBool("datetime_duplicate",
                          descr="Duplicate plots with datetime X axis "
                                "(YYYY-MM-DD HH:MM:SS)",
                          default=False),
            # v0.0.15: density-pct + dt page variant toggles.
            # Replaces the v0.0.14 ``datetime_full_labels`` boolean.
            # The old field name is still accepted by
            # FITS_AutoPlot.push_to_veusz / build_unit_overlay_pages
            # as a back-compat shim (True -> 100, False -> 10).
            vzp.FieldInt("datetime_label_density_pct",
                         descr="Date-label density on dt pages "
                               "(0=no labels, 100=every finite point, "
                               "10 default approximates v0.0.14 sparse)",
                         default=10, minval=0, maxval=100),
            vzp.FieldBool("datetime_emit_numeric_dt",
                          descr="Legacy _dt page with per-point date text labels (off by default)",
                          default=False),
            vzp.FieldBool("datetime_emit_text_dt",
                          descr="dt_labels page: Veusz date-time axis in local ISO 8601 time",
                          default=True),
            # v0.0.19: free-form gap string + hide-keys.  ``gap_string``
            # overrides ``gap_absolute`` (hours) when non-empty.
            vzp.FieldText("gap_string",
                          descr="Optional gap string (overrides gap_absolute). "
                                "e.g. '5m3d2h' (m=months, d=days, h=hours; "
                                "case-insensitive; whitespace OK).",
                          default=""),
            vzp.FieldBool("hide_keys",
                          descr="Hide legend keys on every page",
                          default=False),
            vzp.FieldFloat("gap_axis_n",
                           descr="Large gap n (x-axis break: dt > n x small gap; 0 = none)",
                           default=100.0),
            vzp.FieldText("gap_axis_string",
                          descr="Large gap fixed, e.g. 12h or 3d (overrides n)",
                          default=""),
        ]

    # ------------------------------------------------------------------
    def apply(self, interface, fields):
        """Plugin entry point.  ``interface`` is the document command API."""
        try:
            return self._apply(interface, fields)
        except Exception as exc:
            raise vzp.ToolsPluginException(
                "FITS AutoPlot plugin failed: %s\n%s" %
                (exc, traceback.format_exc())
            )

    def _apply(self, interface, fields):
        """Run the batch dialog with the field values as defaults, then process
        the chosen files into a separate NativeDocument with the dialog's
        TimeLayout and write it as a native .vszh5 project
        (finish_native_plugin_export).  The host Veusz document is not changed.
        """
        app = QApplication.instance() or QApplication(sys.argv)
        mode = "dark" if (fields.get("default_theme") or "dark").lower().startswith("d") else "light"
        dlg = _PluginBatchDialog(default_mode=mode)

        # Honour pre-seeded file list
        pre = (fields.get("preselected_files") or "").strip()
        if pre:
            for p in [x.strip() for x in pre.split(";") if x.strip()]:
                if os.path.isfile(p):
                    dlg.selected_files.append(p)
            for p in dlg.selected_files:
                dlg.file_list.addItem(os.path.basename(p))
        dlg.thread_spin.setValue(int(fields.get("max_threads") or MAX_THREADS))
        dlg.rss_spin.setValue(int(fields.get("rss_mb") or DEFAULT_RSS_HIGH_WATER_MB))
        dlg.datestr_cb.setChecked(bool(fields.get("emit_datestr") or False))
        dlg.skip_images_cb.setChecked(bool(fields.get("skip_images") or False))
        # v0.0.9 pre-seed knobs (optional)
        try:
            dlg.gap_k_spin.setValue(float(fields.get("gap_k") or 10.0))
        except Exception:
            pass
        try:
            dlg.gap_abs_spin.setValue(float(fields.get("gap_absolute") or 0.0))
        except Exception:
            pass
        try:                                     # 0.3.0: axis-break multiple n
            _n = fields.get("gap_axis_n")
            dlg.gap_axis_spin.setValue(100.0 if _n is None else float(_n))
            dlg.gap_axis_str.setText(str(fields.get("gap_axis_string") or ""))
        except Exception:
            pass
        dlg.combined_only_cb.setChecked(bool(fields.get("combined_only") or False))
        # v0.0.11: pre-seed the datetime-duplicate checkbox
        dlg.datetime_dup_cb.setChecked(
            bool(fields.get("datetime_duplicate") or False)
        )
        # v0.0.15: pre-seed density spin + dt page-variant checkboxes.
        # The old ``datetime_full_labels`` field (v0.0.14) is still
        # honoured here: True -> 100 %, False -> 10 %.  Explicit
        # ``datetime_label_density_pct`` wins if both are set.
        _legacy_full = fields.get("datetime_full_labels")
        if "datetime_label_density_pct" in fields:
            _pct = int(fields.get("datetime_label_density_pct") or 0)
        elif _legacy_full is True:
            _pct = 100
        elif _legacy_full is False:
            _pct = 10
        else:
            _pct = 10
        _pct = max(0, min(100, _pct))
        dlg.label_density_spin.setValue(_pct)
        dlg.emit_numeric_dt_cb.setChecked(
            bool(fields.get("datetime_emit_numeric_dt", False))
        )
        dlg.emit_text_dt_cb.setChecked(
            bool(fields.get("datetime_emit_text_dt", True))
        )
        # v0.0.19: pre-seed gap-string + hide-keys.
        try:
            _gs = fields.get("gap_string")
            if _gs is not None:
                dlg.gap_abs_str.setText(str(_gs))
        except Exception:
            pass
        try:
            dlg.hide_keys_cb.setChecked(bool(fields.get("hide_keys") or False))
        except Exception:
            pass
        # v0.0.10: optional GPU pre-seed
        dlg.gpu_cb.setChecked(bool(fields.get("use_gpu") or False)
                              and dlg.gpu_cb.isEnabled())

        if dlg.exec_() != QDialog.Accepted:
            return
        if not dlg.selected_files:
            QMessageBox.information(None, "FITS AutoPlot",
                                    "No files selected -- aborting.")
            return

        # Deliberately replace the host API with a native in-memory model.
        interface, output_file = begin_native_plugin_export(fields)
        if interface is None:
            return
        backend = "astropy"
        cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=int(dlg.rss_spin.value())
        ))
        # 0.3.0: a bound method of the dialog (held weakly by the monitor
        # thread; append_log is thread-safe).  The old lambda let the monitor
        # thread drop the last reference and destroy the dialog off the GUI
        # thread, which crashed PySide6 later.
        mon = MemoryMonitor(cache, callback=dlg.rss_crossed)
        mon.start()

        skip_images = bool(dlg.skip_images_cb.isChecked())
        proc = FITSProcessor(backend, cache, skip_images=skip_images)
        # Sequentially run-in-threadpool inside this plugin call: we use
        # threads (not processes) so the plugin remains within the Veusz
        # interpreter and can write to ``interface`` directly afterwards.
        work = [(p, proc.read, (p,)) for p in dlg.selected_files]
        n_files = len(work)
        dlg.progress.setVisible(True)
        dlg.progress.setRange(0, n_files)
        dlg.parse_progress.setVisible(True)
        dlg.parse_progress.setRange(0, n_files)
        dlg.parse_progress.setValue(0)
        dlg.column_progress.setVisible(True)
        dlg.column_progress.setRange(0, 1)
        dlg.column_progress.setValue(0)
        progress_state = {"done": 0}

        def _cb(done, total, key):
            """Reader progress: advance the bar, log the file (read) and keep the
            dialog responsive.
            """
            progress_state["done"] = done
            dlg.progress.setValue(done)
            dlg.append_log("read [%d/%d] %s" % (done, total, os.path.basename(key)))
            app.processEvents()

        col_throttle = ProgressThrottle()
        def _col_cb(done, total_ops):
            """Per-file column progress: resize the bar to ``total_ops`` when it
            changes and set it to ``done``.
            """
            if not col_throttle.due(done, total_ops, force_end=False):    # 0.3.1: <= 20 updates/s
                return
            if dlg.column_progress.maximum() != max(1, total_ops):
                dlg.column_progress.setRange(0, max(1, total_ops))
            dlg.column_progress.setValue(done)
            app.processEvents()

        # Belt-and-suspenders: suppress NRAO FITS unit warnings around the
        # entire batch read + push.  FITSProcessor already wraps fits.open
        # / QTable.read internally, but this outer context ensures that any
        # downstream astropy code paths (e.g. push_to_veusz fallbacks) also
        # stay quiet during the Veusz batch run.
        # v0.0.9: thread broken-axis / overlay knobs through every push
        gap_k = float(dlg.gap_k_spin.value())
        # v0.0.16: spinbox is in HOURS -- convert to MJD-days.
        gap_absolute_hours = float(dlg.gap_abs_spin.value())
        gap_absolute = gap_absolute_hours / 24.0
        # v0.0.19: free-form gap string overrides the spinbox when set.
        gap_string_text = ""
        try:
            gap_string_text = dlg.gap_abs_str.text().strip()
        except Exception:
            gap_string_text = ""
        if gap_string_text:
            gap_absolute = parse_gap_string(gap_string_text,
                                            log_cb=dlg.append_log)
        # v0.0.19: hide-legend-keys checkbox
        hide_keys = bool(dlg.hide_keys_cb.isChecked())
        plot_individual = not bool(dlg.combined_only_cb.isChecked())
        # v0.0.11: datetime-duplicate toggle
        datetime_duplicate = bool(dlg.datetime_dup_cb.isChecked())
        # v0.0.15: density-pct + dt page-variant toggles.
        datetime_label_density_pct = int(dlg.label_density_spin.value())
        datetime_emit_numeric_dt = bool(dlg.emit_numeric_dt_cb.isChecked())
        datetime_emit_text_dt = bool(dlg.emit_text_dt_cb.isChecked())
        # v0.0.10: drive the process-wide GPU flag from the checkbox
        enable_gpu(dlg.gpu_cb.isChecked() and dlg.gpu_cb.isEnabled())
        dlg.append_log("GPU backend: %s" % gpu_backend_name())
        reset_gpu_usage()                     # 0.3.1: count this batch's GPU / CPU calls
        file_records = []  # accumulator for the unit-overlay post-pass
        with suppress_fits_unit_warnings():
            results = run_in_threadpool(work,
                                        max_workers=int(dlg.thread_spin.value()),
                                        progress_cb=_cb)
            emit_datestr = bool(dlg.datestr_cb.isChecked())
            dlg.append_log("Building native project datasets and pages...")
            # 0.3.0: pages in time order, not thread completion order.
            from FITS_AutoPlot import _result_order
            for idx, (path, data) in enumerate(sorted(results.items(), key=_result_order), start=1):
                if isinstance(data, Exception):
                    dlg.append_log("  ERROR %s: %s" % (path, data))
                    dlg.parse_progress.setValue(idx)
                    app.processEvents()
                    continue
                # Pre-size the column bar from this file's read-pass output.
                n_cols = len(data.get("columns") or {})
                n_imgs = 0 if skip_images else len(data.get("images") or {})
                dlg.column_progress.setRange(0, max(1, n_cols + n_imgs))
                dlg.column_progress.setValue(0)
                try:
                    push_to_veusz(interface, path, data, backend,
                                  log_cb=dlg.append_log,
                                  emit_datestr=emit_datestr,
                                  column_cb=_col_cb,
                                  skip_images=skip_images,
                                  plot_individual=plot_individual,
                                  gap_k=gap_k,
                                  gap_absolute=gap_absolute,
                                  datetime_duplicate=datetime_duplicate,
                                  datetime_label_density_pct=
                                      datetime_label_density_pct,
                                  datetime_emit_numeric_dt=
                                      datetime_emit_numeric_dt,
                                  datetime_emit_text_dt=
                                      datetime_emit_text_dt,
                                  hide_keys=hide_keys)
                except Exception as exc:
                    dlg.append_log("  push_to_veusz failed for %s: %s"
                                   % (path, exc))
                else:
                    # v0.0.12 channel-tag: carry tag_columns/tag_groups so
                    # build_unit_overlay_pages can split each numeric
                    # column into one trace per unique tag-tuple.
                    file_records.append({
                        "base": safe_dsname(data.get("base_name") or
                                            os.path.basename(path)),
                        "columns": data.get("columns") or {},
                        "units": data.get("units") or {},
                        "sort_key": data.get("sort_key"),
                        "tag_columns": data.get("tag_columns") or {},
                        "tag_groups": data.get("tag_groups") or {},
                    })
                dlg.parse_progress.setValue(idx)
                app.processEvents()
            # Build cross-file unit-overlay pages on the host document.
            if file_records:
                try:
                    build_unit_overlay_pages(
                        interface, file_records,
                        gap_k=gap_k, gap_absolute=gap_absolute,
                        log_cb=dlg.append_log,
                        datetime_duplicate=datetime_duplicate,
                        datetime_label_density_pct=
                            datetime_label_density_pct,
                        datetime_emit_numeric_dt=
                            datetime_emit_numeric_dt,
                        datetime_emit_text_dt=
                            datetime_emit_text_dt,
                        hide_keys=hide_keys,
                    )
                except Exception as exc:
                    dlg.append_log(
                        "  build_unit_overlay_pages failed: %s" % exc
                    )
        dlg.progress.setVisible(False)
        dlg.parse_progress.setVisible(False)
        dlg.column_progress.setVisible(False)
        mon.stop()
        cache.cleanup()
        # 0.3.0: every time axis in local ISO 8601; line breaks where
        # dt > K x median(dt), axis breaks where dt > n x K x median(dt).
        from _autoplot_native import TimeLayout
        _axis_text = dlg.gap_axis_str.text().strip()
        interface.time_layout = TimeLayout(
            k=gap_k, n=float(dlg.gap_axis_spin.value()), manual_s=gap_absolute * 86400.0,
            axis_manual_s=(parse_gap_string(_axis_text, log_cb=dlg.append_log) * 86400.0
                           if _axis_text else 0.0))
        dlg.append_log(interface.time_layout.describe())
        finish_native_plugin_export(interface, output_file, fields, dlg.append_log)
        if gpu_enabled():
            dlg.append_log("GPU use: %s" % gpu_usage_text())
        dlg.append_log("Done.")


# ============================================================================
# %% REGISTER WITH VEUSZ
# ============================================================================
if _HOST_PLUGIN_PATH:
    # Import in a real module namespace so class methods retain their globals.
    # The temporary classes from Veusz's exec scope are deliberately discarded.
    import importlib.util as _import_util
    _spec = _import_util.spec_from_file_location(
        "_autoplot_registered_fits", _HOST_PLUGIN_PATH)
    _module = _import_util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
elif vzp is not None:
    vzp.toolspluginregistry.append(FITSAutoPlotPlugin)

if __name__ == "__main__":
    from FITS_AutoPlot import main
    sys.exit(main())
