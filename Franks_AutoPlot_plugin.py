# -*- coding: utf-8 -*-
"""
=============================================================================
Franks_AutoPlot_plugin.py
-----------------------------------------------------------------------------
# %% Header Info

Veusz Tools plugin equivalent of Franks_AutoPlot.py.  Adds the menu entry

    Tools -> NRAO / FitsAutoPlot -> Franks AutoPlot (Batch)

to the active Veusz instance.  Invoking it opens a Touchstone-style qtpy
dialog identical in shape to the standalone Franks_AutoPlot.py window
(file list, thread count, RSS spill threshold, theme toggle, log area,
progress bar) and exports a separate native .vszh5 project using h5py.
The current Veusz document is left untouched.

# %%% Author Information
@author: William W. Wallace

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.1.0: Native project export plugin; leave the host document untouched.
# Date: 2026-10-11
#              Veusz plugin API is used only for host menu integration.
#              Output uses the same h5py writer as the standalone script.
#              Adds output_file, minimized and also_full plugin fields.
#              Can also launch the standalone GUI without a Veusz install.
#
# %%%% 0.0.19: Add gap_string + hide_keys plugin fields; plumb through.
# Date: 2026-05-16
#              Same two new fields as FITS_AutoPlot_plugin 0.0.19:
#                * ``gap_string`` (FieldText, default "") -- overrides
#                  ``gap_absolute`` (hours) when non-empty; parsed via
#                  ``parse_gap_string``.  Tokens: m=months, d=days,
#                  h=hours; case-insensitive; whitespace tolerated.
#                * ``hide_keys`` (FieldBool, default False) -- forwarded
#                  as the ``hide_keys`` kwarg to
#                  ``push_franks_to_veusz`` and
#                  ``build_unit_overlay_pages_franks`` so every key
#                  widget's Border is hidden when True.
#              The batch dialog gained matching
#              ``self.gap_abs_str`` QLineEdit + ``self.hide_keys_cb``
#              QCheckBox.
#
# %%%% 0.0.18: Unit-aware time-break detection (no plugin changes).
# Date: 2026-05-16
#              Plugin-side version bump only.  The unit-aware break
#              detection fix is implemented in _autoplot_common
#              (``detect_time_breaks_unit_aware``) and in the
#              Franks_AutoPlot engine, which the plugin already calls
#              with ``gap_absolute`` (MJD-days).  Franks always uses
#              ``MJD`` as the sort key (day-scale, factor = 1.0), so
#              the behavior in Franks is unchanged in practice -- the
#              version bump tracks the lockstep release schedule.
#              Revision history kept in DESCENDING semantic-version
#              order.
#
# %%%% 0.0.17: No plugin-side changes (engine bump only).
# Date: 2026-05-16
#              The standalone Franks_AutoPlot GUI gained the v0.0.17
#              minimized-Veusz-save checkboxes.  This plugin operates
#              inside an open Veusz document (saves are user-driven via
#              Veusz itself) so it does NOT receive those fields.
#              The FITS-side sentinel channel-tag dt-overlay filter does
#              not apply to Franks (single-trace overlay).
#              Revision history kept in DESCENDING semantic-version
#              order.
#
# %%%% 0.0.16: dt_labels page mode='datetime' + broken-axis parity.
# Date: 2026-05-16
#              * No plugin-field changes vs. v0.0.15.  The plugin
#                surface is unchanged; the engine (Franks_AutoPlot.py
#                + _autoplot_common.py) does all the new work: emits
#                a numeric Veusz-datetime-seconds dataset alongside
#                the text-x dataset, binds the dt_labels page to it,
#                sets the x axis to mode='datetime', and adds
#                broken-axis parity with the seconds-axis dt page.
#              * Header bumped to 0.0.16 only so the plugin and
#                engine version strings stay in lock-step in the
#                Veusz Tools menu.
#              * GUI: "Absolute gap (MJD units; 0=auto)" spinbox is
#                renamed "Manual gap (hours; 0=auto)" and the value
#                is divided by 24 before being passed to the engine,
#                so MJD-axis time gaps are entered in hours.
#
# %%%% 0.0.15: Density-pct date labels + text-x dt_labels page variant.
# Date: 2026-05-16
#                * Plugin-side equivalent of the Franks_AutoPlot.py
#                  v0.0.15 change.  The legacy ``datetime_full_labels``
#                  boolean field is REPLACED by:
#                    - ``datetime_label_density_pct`` (FieldInt 0..100,
#                      default 10): controls how many evenly-spaced
#                      anchor labels are emitted on the numeric-x dt
#                      page.  0 = no labels, 100 = one label per finite
#                      data point.
#                    - ``datetime_emit_numeric_dt`` (FieldBool,
#                      default True): toggle the v0.0.14 numeric-x dt
#                      page (``<base>_dt`` / ``Overlay_<col>_dt``).
#                    - ``datetime_emit_text_dt`` (FieldBool,
#                      default True): toggle the NEW v0.0.15 text-x
#                      dt_labels page (``<base>_dt_labels`` /
#                      ``Overlay_<col>_dt_labels``) which uses a per-
#                      point text dataset as xData with axis
#                      ``mode='labels'``.  Sample spacing is uniform.
#                * Dialog UI: the v0.0.14 ``full_labels_cb`` checkbox is
#                  replaced by a QSpinBox 0..100 (with " %" suffix) and
#                  two QCheckBoxes for the two dt page variants.
#                * The old ``datetime_full_labels`` field is still
#                  accepted in saved-doc plugin invocations as a back-
#                  compat shim handled by Franks_AutoPlot.
#                  push_franks_to_veusz and
#                  build_unit_overlay_pages_franks (True -> 100,
#                  False -> 10).
#                * GPU + parallelization re-audit: no new sort sites
#                  introduced by the plugin path; worker pool and
#                  GPU-sort wiring are inherited unchanged from
#                  Franks_AutoPlot.
#                * Revision history kept in DESCENDING semantic-
#                  version order.
#
# %%%% 0.0.14: Datetime via xy.labels per-point text labels.
# Date: 2026-05-16
#                * Mirrors Franks_AutoPlot.py v0.0.14: the dt-duplicate
#                  pages now bind a per-point text dataset to
#                  xy.labels.val (rendering one date string per data
#                  point) instead of using a Veusz datetime x axis
#                  (which is broken on Veusz 3.4 internals).
#                * Adds plugin field ``datetime_full_labels`` and a
#                  matching dialog checkbox; threads the flag through
#                  ``push_franks_to_veusz`` and
#                  ``build_unit_overlay_pages_franks``.
#                * No SetDataDateTime calls in the dt path.

# %%%% 0.0.13: Inherits identity-stable trace styling + datetime-duplicate
# Date: 2026-05-16
#              hardening from Franks_AutoPlot.py.  No plugin-side code
#              changes are required: push_franks_to_veusz,
#              build_unit_overlay_pages_franks, and the page builders are
#              shared with the standalone GUI, so the new
#              apply_trace_style() wiring and set_datetime_dataset()
#              coercion both take effect inside the plugin entry point
#              automatically.  Header bumped for version visibility in
#              the Veusz Tools menu.

# %%%% 0.0.12: Inherits combined-in-time overlay semantics from
# Date: 2026-05-16
#              Franks_AutoPlot.py v0.0.12 -- plugin unchanged because
#              build_unit_overlay_pages_franks does the work.

# %%%% 0.0.11: Plugin-side support for datetime-duplicate plots (mirrors
# Date: 2026-05-16
#              Franks_AutoPlot.py v0.0.11).  _PluginBatchDialog gained a
#              'Duplicate plots with datetime X axis (YYYY-MM-DD HH:MM:SS)'
#              checkbox; ``apply()`` threads the resulting
#              ``datetime_duplicate`` boolean through every
#              push_franks_to_veusz() call and through
#              build_unit_overlay_pages_franks.

# %%%% 0.0.10: Plugin-side support for the optional GPU acceleration
# Date: 2026-05-16
#              (mirrors Franks_AutoPlot.py v0.0.10).  _PluginBatchDialog
#              now exposes a 'Use GPU acceleration (CuPy)' checkbox
#              that is disabled when CuPy is not importable; ``apply()``
#              toggles the process-wide flag via ``enable_gpu()`` before
#              pushing.

# %%%% 0.0.9: Plugin-side support for broken-axis + column-name overlay
# Date: 2026-05-16
#             (mirrors Franks_AutoPlot.py v0.0.9).  _PluginBatchDialog
#             now exposes ``Gap K (× median Δt)``, ``Absolute gap``, and
#             ``Combined plots only`` controls; ``apply()`` threads
#             ``plot_individual``, ``gap_k``, ``gap_absolute`` through
#             every ``push_franks_to_veusz()`` call, accumulates
#             ``file_records`` across the batch, and post-builds
#             column-name overlay pages by calling
#             ``Franks_AutoPlot.build_unit_overlay_pages_franks`` on the
#             host ``interface`` doc.

# %%%% 0.0.8: Version-bumped in lockstep with the rest of the AutoPlot
# Date: 2026-05-16
#             suite which gained 'Open in Veusz...' buttons + a
#             parallelization audit (MAX_THREADS doubled; parse_franks_file()
#             vectorized).
#             No 'Open in Veusz' button is added to this plugin dialog: the
#             plugin already runs *inside* the live Veusz application, so
#             pushed datasets are immediately visible in the host document.
#             The plugin transparently picks up the MAX_THREADS bump and
#             the vectorized parse_franks_file() through its imports from
#             Franks_AutoPlot.py; no code changes are required here.
# %%%%% Function Descriptions

# %%%% 0.0.7: Added two new progress bars to the plugin dialog -- a
# Date: 2026-05-16
#             'Parsing/pushing' file-level bar and a 'Current file -
#             columns' bar ticked per source column as push_franks_to_veusz()
#             pours that file into the active document.  No skip-images
#             knob here (Franks files have no image HDUs).

# %%%% 0.0.6: Added Spyder IDE cell markers (# %% / # %%%) at all major
# Date: 2026-05-16
#             section banners and import subsections so the file can be
#             navigated and run cell-by-cell in Spyder's Outline view.
#             Cosmetic only -- no behavior change.

# %%%% 0.0.5: The plugin's apply() loop already calls processEvents()
# Date: 2026-05-16
#             between every push, so the GUI-responsiveness fix shipped
#             in Franks_AutoPlot.py 0.0.5 is inherited automatically.
#             Version bumped here to keep the four files version-aligned.
#             (No 0.0.4: aligned with the FITS_AutoPlot.py version stream.)

# %%%% 0.0.3: NaN-preserving emission policy (inherits push_franks_to_veusz
# Date: 2026-05-16
#             from Franks_AutoPlot.py): rows containing missing or non-
#             numeric tokens are never dropped -- numeric NaN floats pass
#             through Veusz's native NaN-aware numeric datasets, and the
#             optional date-string text datasets use the sentinel string
#             ``"NaN"`` for non-finite MJDs so all per-row arrays stay the
#             same length and remain index-aligned.

# %%%% 0.0.2: Added "Generate MJD->date strings" checkbox to the modal
# Date: 2026-05-16
#             dialog and an ``emit_datestr`` boolean field so the same
#             option is available from the Veusz Tools menu.

# %%%% 0.0.1: Initial plugin equivalent of Franks_AutoPlot.py
# Date: 2026-05-16

        FranksAutoPlotPlugin: Veusz ToolsPlugin subclass providing the menu
            entry, fields (max_threads, rss_mb, default_theme, preseed) and
            the apply() entry point.
        _PluginBatchDialog: modal qtpy dialog mirroring the standalone
            main window without a menu bar (plugin dialogs cannot host a
            QMainWindow menu, so dark/light is offered as a combo box).
# %%%%% Variable Descriptions
        MAX_THREADS / DEFAULT_RSS_HIGH_WATER_MB: imported from
            Franks_AutoPlot.py so all four deliverables share the same
            top-of-file knobs.
# %%%%% More Info
        Because the Franks data files are pure ASCII, no equivalent of
        Veusz's native FITS importer is offered; the plugin always parses
        with numpy and produces both raw and sorted Veusz datasets, both
        tagged with the file's base name plus 'raw' / 'sorted'.
=============================================================================
"""
# %% Imports
from __future__ import annotations

# %%% IMPORTS - Standard library
import os
import sys
import traceback
from typing import Any, Dict, List

# Veusz 3.x's exec loader supplies its selected filename as local "plugin".
_HOST_PLUGIN_PATH = None
_module_file = globals().get("__file__")
if not _module_file or globals().get("__name__", "").startswith("veusz."):
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

# %%% IMPORTS - Sibling modules / shared GUI helpers
from Franks_AutoPlot import (                       # noqa: E402
    MAX_THREADS, DEFAULT_RSS_HIGH_WATER_MB,
    parse_franks_file, push_franks_to_veusz,
    build_unit_overlay_pages_franks,
)
from _autoplot_common import (                      # noqa: E402
    QApplication, QFileDialog, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QFormLayout,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QProgressBar, QTextEdit, QLineEdit,
    apply_theme, MemoryAwareCache, MemoryMonitorConfig, MemoryMonitor,
    run_in_threadpool, safe_dsname, mjd_to_datestr,
    is_gpu_available, enable_gpu, gpu_backend_name,
    # v0.0.19: string gap parser
    parse_gap_string,
    begin_native_plugin_export, finish_native_plugin_export,
)
try:
    from qtpy.QtWidgets import QDialog
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QDialog


# ============================================================================
# %% MODAL DIALOG (mirror of the standalone GUI)
# ============================================================================
class _PluginBatchDialog(QDialog):
    """Modal qtpy dialog used by the plugin entry point."""

    def __init__(self, parent=None, default_mode: str = "dark") -> None:
        super().__init__(parent)
        self.setWindowTitle("Franks AutoPlot - Veusz Plugin")
        self.resize(900, 700)
        self.selected_files: List[str] = []
        self._theme_mode = default_mode

        root = QVBoxLayout(self)

        trow = QHBoxLayout()
        trow.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Mode", "Light Mode"])
        self.theme_combo.setCurrentIndex(0 if default_mode == "dark" else 1)
        self.theme_combo.currentIndexChanged.connect(self._theme_changed)
        trow.addWidget(self.theme_combo)
        trow.addStretch()
        root.addLayout(trow)

        root.addWidget(QLabel("Input files:"))
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(160)
        root.addWidget(self.file_list)
        brow = QHBoxLayout()
        b1 = QPushButton("Browse...")
        b1.clicked.connect(self._browse)
        brow.addWidget(b1)
        b2 = QPushButton("Clear")
        b2.clicked.connect(self._clear)
        brow.addWidget(b2)
        root.addLayout(brow)

        form = QFormLayout()
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, max(1, (os.cpu_count() or 4) * 2))
        self.thread_spin.setValue(MAX_THREADS)
        form.addRow("Worker threads:", self.thread_spin)
        self.rss_spin = QSpinBox()
        self.rss_spin.setRange(128, 65536)
        self.rss_spin.setSingleStep(128)
        self.rss_spin.setValue(DEFAULT_RSS_HIGH_WATER_MB)
        form.addRow("RSS spill threshold (MiB):", self.rss_spin)
        root.addLayout(form)

        self.datestr_cb = QCheckBox(
            "Generate MJD -> date strings (YYYY-MM-DD_HH:MM:SS) datasets"
        )
        self.datestr_cb.setChecked(False)
        root.addWidget(self.datestr_cb)

        # --- Broken-axis / overlay controls (v0.0.9) ----------------------
        form2 = QFormLayout()
        self.gap_k_spin = QDoubleSpinBox()
        self.gap_k_spin.setRange(1.0, 1000.0)
        self.gap_k_spin.setSingleStep(1.0)
        self.gap_k_spin.setDecimals(2)
        self.gap_k_spin.setValue(10.0)
        form2.addRow("Gap K (× median Δt):", self.gap_k_spin)

        self.gap_abs_spin = QDoubleSpinBox()
        self.gap_abs_spin.setRange(0.0, 1e12)
        self.gap_abs_spin.setDecimals(6)
        self.gap_abs_spin.setSingleStep(1.0)
        self.gap_abs_spin.setValue(0.0)
        # v0.0.16: spinbox is in HOURS; converted to MJD-days when read.
        form2.addRow("Manual gap (hours; 0=auto):", self.gap_abs_spin)

        # v0.0.19: free-form gap-string input (overrides spinbox when set).
        self.gap_abs_str = QLineEdit()
        self.gap_abs_str.setPlaceholderText("e.g. 5m3d2h  (overrides spinbox)")
        self.gap_abs_str.setToolTip(
            "Optional gap string.  Units: m = months (~30.44 days), "
            "d = days, h = hours.  Examples: '3d', '120h', '5m 3d 2h'."
        )
        form2.addRow("Manual gap (string; overrides):", self.gap_abs_str)
        root.addLayout(form2)

        # v0.0.19: hide legend keys on every page.
        self.hide_keys_cb = QCheckBox("Hide legend keys on all pages")
        self.hide_keys_cb.setChecked(False)
        root.addWidget(self.hide_keys_cb)

        self.combined_only_cb = QCheckBox(
            "Combined (overlay) plots only -- skip per-file pages"
        )
        self.combined_only_cb.setChecked(False)
        root.addWidget(self.combined_only_cb)

        # --- Datetime-duplicate plots (v0.0.11) ---------------------------
        # v0.0.11: datetime-duplicate toggle.
        self.datetime_dup_cb = QCheckBox(
            "Duplicate plots with datetime X axis (YYYY-MM-DD HH:MM:SS)"
        )
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
            "Emit numeric-x dt page (v0.0.14 lineage)"
        )
        self.emit_numeric_dt_cb.setChecked(True)
        self.emit_numeric_dt_cb.setToolTip(
            "When checked, the per-file '<base>_dt' page (and the "
            "matching 'Overlay_<col>_dt' overlay) is emitted with a "
            "numeric seconds x axis plus density-controlled text "
            "labels."
        )
        root.addWidget(self.emit_numeric_dt_cb)
        self.emit_text_dt_cb = QCheckBox(
            "Emit text-x dt_labels page (v0.0.15 new)"
        )
        self.emit_text_dt_cb.setChecked(True)
        self.emit_text_dt_cb.setToolTip(
            "When checked, the per-file '<base>_dt_labels' page "
            "(and the matching 'Overlay_<col>_dt_labels' overlay) is "
            "emitted with a per-point text dataset as xData and the x "
            "axis set to mode='labels'.  Sample spacing is uniform."
        )
        root.addWidget(self.emit_text_dt_cb)

        # --- GPU acceleration (CuPy, optional) (v0.0.10) ------------------
        self.gpu_cb = QCheckBox("Use GPU acceleration (CuPy) for large sorts")
        self.gpu_cb.setChecked(False)
        self.gpu_cb.setEnabled(is_gpu_available())
        self.gpu_cb.setToolTip(gpu_backend_name())
        root.addWidget(self.gpu_cb)

        self.log = QTextEdit()
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

        arow = QHBoxLayout()
        self.process_btn = QPushButton("Process Files")
        self.process_btn.clicked.connect(self.accept)
        arow.addWidget(self.process_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        arow.addWidget(self.cancel_btn)
        root.addLayout(arow)

        apply_theme(QApplication.instance(), default_mode)

    def _theme_changed(self, idx: int) -> None:
        mode = "dark" if idx == 0 else "light"
        self._theme_mode = mode
        apply_theme(QApplication.instance(), mode)

    def _browse(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Franks data files", "",
            "Franks files (*.t00new *.dat *.bak* *.OrigBack* *.back);;All files (*)"
        )
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
        self.file_list.clear()
        for p in self.selected_files:
            self.file_list.addItem(os.path.basename(p))

    def _clear(self) -> None:
        self.selected_files.clear()
        self.file_list.clear()

    def append_log(self, msg: str) -> None:
        self.log.append(msg)


# ============================================================================
# %% THE PLUGIN
# ============================================================================
class FranksAutoPlotPlugin(vzp.ToolsPlugin if vzp is not None else object):
    """Veusz menu entry for native ASCII project export, not host-document editing."""

    menu = ("NRAO / FitsAutoPlot", "Franks AutoPlot (Batch)")
    name = "Franks AutoPlot"
    description_short = "Batch import FranksProcessed ASCII files (column oriented)"
    description_full = (
        "Open a Touchstone-style GUI to select one or more FranksProcessed "
        "files (*.t00new, time_gbt.dat, and their .bak variants), choose "
        "threading and RSS spill threshold, then export a native .vszh5 "
        "file through h5py. The current Veusz document is not modified."
    )

    def __init__(self) -> None:
        if vzp is None:
            raise RuntimeError("Use Franks_AutoPlot.py outside the Veusz plugin host")
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
                          descr="Optional: ;-separated file paths "
                                "(leave blank to pick interactively)",
                          default=""),
            vzp.FieldBool("emit_datestr",
                          descr="Also create MJD -> date-string datasets "
                                "(YYYY-MM-DD_HH:MM:SS)",
                          default=False),
            # v0.0.15: density-pct + dt page variant toggles.
            # Replaces the v0.0.14 ``datetime_full_labels`` boolean.
            # The old field name is still accepted by
            # Franks_AutoPlot.push_franks_to_veusz /
            # build_unit_overlay_pages_franks as a back-compat shim
            # (True -> 100, False -> 10).
            vzp.FieldInt("datetime_label_density_pct",
                         descr="Date-label density on dt pages "
                               "(0=no labels, 100=every finite point, "
                               "10 default approximates v0.0.14 sparse)",
                         default=10, minval=0, maxval=100),
            vzp.FieldBool("datetime_emit_numeric_dt",
                          descr="Emit numeric-x dt page "
                                "(<base>_dt, Overlay_<col>_dt) "
                                "with density-controlled date labels",
                          default=True),
            vzp.FieldBool("datetime_emit_text_dt",
                          descr="Emit text-x dt_labels page "
                                "(<base>_dt_labels, "
                                "Overlay_<col>_dt_labels) using a "
                                "per-point text dataset as xData",
                          default=True),
            # v0.0.11: pre-seed the datetime-duplicate checkbox
            vzp.FieldBool("datetime_duplicate",
                          descr="Duplicate plots with datetime X axis "
                                "(YYYY-MM-DD HH:MM:SS)",
                          default=False),
            # v0.0.19: free-form gap string + hide-keys.
            vzp.FieldText("gap_string",
                          descr="Optional gap string (overrides gap_absolute). "
                                "e.g. '5m3d2h' (m=months, d=days, h=hours; "
                                "case-insensitive; whitespace OK).",
                          default=""),
            vzp.FieldBool("hide_keys",
                          descr="Hide legend keys on every page",
                          default=False),
        ]

    # ------------------------------------------------------------------
    def apply(self, interface, fields):
        try:
            return self._apply(interface, fields)
        except Exception as exc:
            raise vzp.ToolsPluginException(
                "Franks AutoPlot plugin failed: %s\n%s" %
                (exc, traceback.format_exc())
            )

    def _apply(self, interface, fields):
        app = QApplication.instance() or QApplication(sys.argv)
        mode = "dark" if (fields.get("default_theme") or "dark").lower().startswith("d") else "light"
        dlg = _PluginBatchDialog(default_mode=mode)

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
        # v0.0.9 pre-seed knobs (optional)
        try:
            dlg.gap_k_spin.setValue(float(fields.get("gap_k") or 10.0))
        except Exception:
            pass
        try:
            dlg.gap_abs_spin.setValue(float(fields.get("gap_absolute") or 0.0))
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
            bool(fields.get("datetime_emit_numeric_dt", True))
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
            QMessageBox.information(None, "Franks AutoPlot",
                                    "No files selected -- aborting.")
            return

        interface, output_file = begin_native_plugin_export(fields)
        if interface is None:
            return
        cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=int(dlg.rss_spin.value())
        ))
        mon = MemoryMonitor(cache, callback=lambda r: dlg.append_log(
            "RSS over high-water mark (%.1f MiB) -- spilling." % r))
        mon.start()

        work = [(p, parse_franks_file, (p, cache)) for p in dlg.selected_files]
        n_files = len(work)
        dlg.progress.setVisible(True)
        dlg.progress.setRange(0, n_files)
        dlg.parse_progress.setVisible(True)
        dlg.parse_progress.setRange(0, n_files)
        dlg.parse_progress.setValue(0)
        dlg.column_progress.setVisible(True)
        dlg.column_progress.setRange(0, 1)
        dlg.column_progress.setValue(0)

        def _cb(done, total, key):
            dlg.progress.setValue(done)
            dlg.append_log("parsed [%d/%d] %s" % (done, total, os.path.basename(key)))
            app.processEvents()

        def _col_cb(done, total_ops):
            if dlg.column_progress.maximum() != max(1, total_ops):
                dlg.column_progress.setRange(0, max(1, total_ops))
            dlg.column_progress.setValue(done)
            app.processEvents()

        # v0.0.9: thread broken-axis / overlay knobs through every push
        gap_k = float(dlg.gap_k_spin.value())
        # v0.0.16: spinbox is in HOURS -- convert to MJD-days.
        gap_absolute_hours = float(dlg.gap_abs_spin.value())
        gap_absolute = gap_absolute_hours / 24.0
        # v0.0.19: free-form gap string overrides spinbox when set.
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
        file_records = []  # accumulator for the overlay post-pass
        results = run_in_threadpool(work,
                                    max_workers=int(dlg.thread_spin.value()),
                                    progress_cb=_cb)
        emit_datestr = bool(dlg.datestr_cb.isChecked())
        dlg.append_log("Building native project datasets and pages...")
        for idx, (path, data) in enumerate(results.items(), start=1):
            if isinstance(data, Exception):
                dlg.append_log("  ERROR %s: %s" % (path, data))
                dlg.parse_progress.setValue(idx)
                app.processEvents()
                continue
            n_cols = len(data.get("columns") or {})
            dlg.column_progress.setRange(0, max(1, n_cols))
            dlg.column_progress.setValue(0)
            try:
                push_franks_to_veusz(interface, data,
                                     log_cb=dlg.append_log,
                                     emit_datestr=emit_datestr,
                                     column_cb=_col_cb,
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
                dlg.append_log("  push failed for %s: %s" % (path, exc))
            else:
                file_records.append({
                    "base": data.get("base_name") or
                            safe_dsname(os.path.basename(path)),
                    "columns": data.get("columns") or {},
                    "sort_key": data.get("sort_key"),
                })
            dlg.parse_progress.setValue(idx)
            app.processEvents()
        # Build cross-file column-name overlay pages on the host document.
        if file_records:
            try:
                build_unit_overlay_pages_franks(
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
                    "  build_unit_overlay_pages_franks failed: %s" % exc
                )
        dlg.progress.setVisible(False)
        dlg.parse_progress.setVisible(False)
        dlg.column_progress.setVisible(False)
        mon.stop()
        cache.cleanup()
        finish_native_plugin_export(interface, output_file, fields, dlg.append_log)
        dlg.append_log("Done.")


# ============================================================================
# %% REGISTER
# ============================================================================
if _HOST_PLUGIN_PATH:
    import importlib.util as _import_util
    _spec = _import_util.spec_from_file_location(
        "_autoplot_registered_franks", _HOST_PLUGIN_PATH)
    _module = _import_util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
elif vzp is not None:
    vzp.toolspluginregistry.append(FranksAutoPlotPlugin)

if __name__ == "__main__":
    from Franks_AutoPlot import main
    sys.exit(main())
