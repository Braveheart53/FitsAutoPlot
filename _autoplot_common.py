# -*- coding: utf-8 -*-
"""
=============================================================================
_autoplot_common.py
-----------------------------------------------------------------------------
# %% Header Info
--------

Shared helpers for FITS_AutoPlot and Franks_AutoPlot:

    * qtpy abstraction layer GUI building blocks (replacement for the older
      FITS_AutoPlot.py minimal dialog) modelled after Touchstone_AutoPlot.py.
    * Dark / Light theme palettes, selectable from a textual View menu.
    * Memory monitor that spills numpy arrays to disk via np.memmap (the
      most efficient zero-copy RAM->disk path) when an RSS threshold is
      crossed.
    * Thread pool helper with a script-level MAX_THREADS knob.
    * Native .vszh5 HDF5 export targeting Veusz 3.4 and 4.1, with an
      optional external viewer launch after saving.

The module is written for Python 3.8 and avoids any syntax that was
introduced in 3.9+ (no walrus inside f-strings, no PEP 604 unions, no dict
union operators, no parenthesised context managers).

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.4.1: GPU for NVIDIA or AMD through _autoplot_gpu; progress throttle.
Date: 2026-11-09
#              * GPU helpers (is_gpu_available, gpu_backend_name, enable_gpu,
#                gpu_argsort ...) now use _autoplot_gpu: CuPy on CUDA
#                (NVIDIA), CuPy on ROCm (AMD, Linux) or PyOpenCL (any
#                OpenCL GPU, AMD on Windows), NumPy otherwise.  Same API,
#                same results (the GPU order equals np.argsort(kind=
#                "stable")).  CuPy is no longer imported with this module,
#                so programs and their spawn workers start faster.
#              * NEW gpu_tooltip(), gpu_usage_text(), reset_gpu_usage():
#                the GPU checkbox tooltips say which device was found and
#                from which size it is used; the log says how many sorts
#                ran on the GPU.
#              * NEW ProgressThrottle: progress bars and per-column log
#                lines at most every 50 ms (the 30-day batch spent seconds
#                repainting the bars for 40 000 columns).
# %%%%% Function Descriptions
        make_dark_palette: Return a Fusion-style dark palette.
        make_light_palette: Return a default light palette (style standard).
        apply_theme: Apply the suite's 'dark' or 'light' palette to the running QApplication (standalone windows only - plugins style their own dialog instead).
        suite_theme: Theme stored for the whole suite (launcher and tools).
        store_suite_theme: Save ``mode`` ('dark' or 'light') as the suite-wide theme.
        MemoryMonitorConfig (class): Configuration knobs for the memory monitor.
        MemoryAwareCache (class): Memory-aware data cache that spills large numpy arrays to disk via ``numpy.memmap``.
        MemoryMonitor (class): Background thread that publishes a callback whenever process RSS crosses the high-water mark.
        run_in_threadpool: Submit ``work_items`` -- ``(key, fn, args)`` triplets -- to a ``ThreadPoolExecutor`` and collect results keyed by ``key``.
        open_embedded: Legacy function name; return a native model, never launch Veusz.
        begin_native_plugin_export: Pick an output file before parsing; never modify the host document.
        finish_native_plugin_export: Same HDF5 writer for both plugin entry points and standalone GUIs.
        save_vszh5: Atomically write the native model with h5py; no Veusz runtime needed.
        _walk_widget_tree: Recursive sibling of ``Root.WalkWidgets``.
        _collect_referenced_dataset_names_via_walk: Original (v0.0.17) WalkWidgets-based traversal.
        _collect_referenced_dataset_names_via_text: v0.0.19 authoritative collector: dump the document via Veusz's own text serialization and regex out every ``Set('xData', 'name')``-style binding.
        _collect_referenced_dataset_names: v0.0.19: combine WalkWidgets-based traversal AND a document-text scan, returning the UNION of both whitelists.
        save_vszh5_minimized: Save exactly the referenced native datasets without deleting anything.
        _norm_tag_token: Lower-cased, stripped string form of a tag-tuple element.
        is_sentinel_tag_tuple: Return True iff ``tup`` is a placeholder/catch-all channel-tag tuple that should be EXCLUDED from dt-overlay datetime emission.
        is_delta_column: Deprecated stub kept for v0.0.17 back-compat.
        open_in_veusz_app: Launch an optional external Veusz GUI and load ``filename`` (typically a .vszh5 project just written by save_vszh5).
        detect_time_breaks: Detect large gaps in a 1-D monotonic-ish array of time samples and return a list of (start, end) pairs suitable for Veusz's ``axis-broken`` widget.
        parse_gap_string: Parse a human-readable gap string into MJD-days.
        column_unit_factor_from_day: Return the multiplier that converts a day-scale threshold into the units of the column named ``col_name``.
        detect_time_breaks_unit_aware: Unit-aware wrapper around ``detect_time_breaks``.
        break_pairs_to_breakpoints: Flatten a list of (start, end) tuples to Veusz's flat-list format [s1, e1, s2, e2, ...] as expected by the ``breakPoints`` setting of the ``axis-br...
        make_broken_x_axis: Add an ``axis-broken`` widget named ``x`` to ``graph`` with the given list of (start, end) break pairs, after removing the default ``x`` axis that...
        mjd_to_veusz_seconds: Convert an array of Modified Julian Dates (UTC) to the float-seconds units Veusz uses internally for date-time datasets.
        _coerce_axis_setting_to_str: Some Veusz axis settings (rotate, in particular) are documented as accepting either an int or its string form.
        style_datetime_x_axis: Apply the standard datetime tick-label format and angled rotation to an axis widget.
        is_gpu_available: True when a GPU back end (CuPy CUDA/ROCm or PyOpenCL) found a device.
        gpu_backend_name: Short string describing the GPU back end (or why there is none) for the GUI/log.
        enable_gpu: Toggle the process-wide GPU acceleration flag.
        gpu_enabled: Return the current global GPU-enabled state.
        set_gpu_argsort_threshold: Set the minimum array size (samples) at which the GPU sort is used.
        get_gpu_argsort_threshold: Minimum array size (samples) for the GPU argsort path.
        gpu_tooltip: Tooltip of a GPU checkbox: the device found (or why none), the size threshold of ``operation`` and the install hints for NVIDIA and AMD.
        gpu_usage_text: GPU and CPU calls of each GPU-capable operation since reset_gpu_usage().
        reset_gpu_usage: Start counting GPU and CPU calls again (at the start of a batch).
        gpu_argsort: Return ``np.argsort(arr, kind="stable")`` (ascending, int64) with optional GPU acceleration (NVIDIA or AMD; see _autoplot_gpu).
        ProgressThrottle (class): Rate limit for progress-bar updates (0.4.1).
        AutoPlotMainWindow (class): Base main window mirroring the layout used by Touchstone_AutoPlot.py:
        open_maybe_gzipped: Return a binary file handle for ``path``; transparently gunzip *.gz.
        mjd_to_datestr: Convert an array of Modified Julian Dates (UTC, double precision) into an array of formatted timestamp strings.
        register_nrao_fits_units: Register the non-standard NRAO FITS unit aliases with astropy.units so that ``QTable.read`` / ``Table.read`` no longer emit a UnitsWarning.
        suppress_fits_unit_warnings: Context manager that filters the harmless FITS unit-related warnings emitted by ``astropy.io.fits`` and ``astropy.table`` when reading NRAO 1PPS-de...
        set_datetime_dataset: Push a Veusz datetime dataset, coercing values to plain floats.
        safe_dsname: Sanitize an arbitrary string for use as a Veusz dataset name.
        _stable_index: Deterministic non-negative index in ``[0, modulus)``.
        trace_style_for: Return ``(color_hex, line_style_str)`` for an identity key.
        apply_trace_style: Apply identity-stable color + line style + markers to a Veusz xy.
        build_sparse_datestr_dataset: Push a same-length sparse date-string text dataset to Veusz.
        build_full_datestr_dataset: Push a same-length per-point date-string text dataset to Veusz.
        style_xy_datetime_labels: Apply rotated, compact text formatting to an xy widget's labels.
        build_density_datestr_dataset: Push a same-length date-string text dataset with continuous density.
        build_textx_dataset: Push a same-length per-point date-string text dataset for use as xData.
        configure_axis_labels_mode: Switch a Veusz axis to ``mode='labels'`` with sensible defaults.
        build_dtnum_dataset: Push a numeric Veusz-datetime-seconds dataset for use as xData.
        configure_axis_datetime_mode: Switch a Veusz x axis to ``mode='datetime'`` and style its ticks.
        mjd_break_pairs_to_dtsec: Convert MJD (start, end) gap pairs to Veusz datetime-seconds pairs.
# %%%%% Variable Descriptions
    Define all utilized variables
        _DATASET_REF_SETTINGS: v0.0.19: trimmed the tuple to entries that *actually* hold dataset names.  ``posn`` is a position rectangle, ``key`` is a legend-label STRING -- ne...
        _DOC_DATASET_REF_REGEX: v0.0.19: regex used to extract dataset references from the document's own saved-text representation.  This is the AUTHORITATIVE source of truth --...
        SENTINEL_TAG_TOKENS: The user can extend the sentinel set in place; the predicate is case-insensitive and trims whitespace before matching.
        DELTA_COLUMN_NAME_PATTERNS: Deprecated column-name predicate -- retained as a no-op for any external script that may have imported it.  v0.0.17 superseded this with the tuple-...
        DAY_SCALE_COLUMN_NAMES: Unknown column names default to 1.0 (i.e. assume MJD-day units), which is the safe choice when the column appears to be a generic time axis: a too-...
        SECOND_SCALE_COLUMN_NAMES: column names whose values are seconds (time axes)
        _GAP_TOKEN_REGEX: Accepts strings like ``"5m3d2h"`` (5 months, 3 days, 2 hours), with or without whitespace between tokens, case-insensitive.  A bare number (no unit...
        GAP_STRING_HELP: (0.4.0) Tool tip of the gap text boxes.
        _DAYS_PER_MONTH: Average Gregorian month length (365.25 / 12) -- intentionally the same value used by other scientific tools (e.g. pandas) so 1m + 1m + ... = ~year.
        _DAYS_PER_HOUR: 1 / 24 (gap text parsing)
        _DAYS_PER_MINUTE: 0.4.0
        _DAYS_PER_SECOND: 0.4.0
        MJD_VEUSZ_EPOCH_MJD: MJD that corresponds to Veusz's internal date-time zero (2009-01-01 UTC).
        DEFAULT_DATETIME_TICK_FORMAT: 0.3.0: ISO 8601
        LOCAL_TIME_AXIS_LABEL: 0.3.0: label of a local date-time axis (the offset is appended on save).
        DEFAULT_DATETIME_TICK_ROTATE_DEG: Default tick label rotation -- 45 degrees keeps a 19-character label readable at typical cadences.  Veusz accepts integer or stringified integer de...
        DEFAULT_DATETIME_MAJOR_TICKS_TARGET: Target number of major ticks along the x axis.  Veusz auto-picks the spacing to satisfy roughly this many major ticks; we keep it modest so the rot...
        QUEUED_CONNECTION: queued signal connection, valid for PySide6, PyQt5 and PyQt6
        MJD_EPOCH_JD_OFFSET: Modified Julian Date epoch: MJD 0.0 == 1858-11-17 00:00:00 UTC. JD = MJD + 2400000.5  -- we implement the conversion directly without astropy.time...
        _NRAO_UNITS_REGISTERED: NRAO 1PPS-delta FITS files use non-standard FITS unit strings: * CHANNELA / CHANNELB carry unit='none'  (text columns, no real unit) * DELTAT  carr...
        TRACE_COLOR_PALETTE: trace colours, cycled
        TRACE_LINE_STYLES: line styles cycled after the colours
        TRACE_STYLE_VARY_THRESHOLD: 16
        DEFAULT_DATETIME_LABEL_ANCHORS: legacy date-label pages: labels per trace
        DEFAULT_DATETIME_LABEL_ANGLE: legacy date-label pages: label angle (degrees)
        DEFAULT_DATETIME_LABEL_SIZE: legacy date-label pages: label font size
        DEFAULT_DATETIME_LABEL_FMT: legacy date-label pages: strftime format
        PYQTGRAPH_ERROR = None
        _PSUTIL_AVAILABLE = True
# %%%%% More Info
    Shared by FITS and Franks AutoPlot and their plugins.  numpy.memmap is
    the RAM -> disk transport of MemoryAwareCache (no pickling, the kernel page
    cache does the writing).  The memory monitor runs in its own thread: its
    callback must not touch Qt widgets (log() is safe).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.4.0: Gap controls shared by FITS and Franks AutoPlot.
Date: 2026-11-09
#              * AutoPlotMainWindow.add_gap_controls(form): "Small gap K"
#                (line break where dt > K x median dt), "Small gap fixed"
#                (hours box or text), "Large gap n" (x-axis break where
#                dt > n x the small gap; default 100, 0 = none), "Large gap
#                fixed" (text, overrides n); time_layout() returns the
#                _autoplot_native.TimeLayout; gap_manual_days(),
#                gap_axis_days().
#              * parse_gap_string: also "s" (seconds) and "min" (minutes);
#                GAP_STRING_HELP.
#              * FIX (Python 3.12 crash) MemoryMonitor: a bound-method
#                callback is held weakly; the thread waits on its stop event
#                (stop() wakes and joins it); the event is no longer called
#                "_stop", which shadowed threading.Thread._stop() and broke
#                join().  Callers passed lambdas holding the window/dialog,
#                so the monitor thread could drop the last reference and
#                destroy Qt widgets outside the GUI thread - PySide6 6.12
#                then crashed at random (segmentation fault in a later Qt
#                timer or gc.collect(), 3 runs in 4 of the unit tests).
#              * AutoPlotMainWindow.log() is thread-safe: off the main
#                thread the text is queued to it (_log_from_thread signal,
#                QUEUED_CONNECTION).  The RSS callback used to append to the
#                log widget from the monitor thread.
#              * Docstrings for every function, class and method; header
#                template (author contact, Function/Variable Descriptions).
#              * Tooltips (pop-up help) on every button, box and field that
#                had none (file list, Browse, Clear, log, Process, Save, Close).
# %%%%% Function Descriptions
        make_dark_palette: Return a Fusion-style dark palette.
        make_light_palette: Return a default light palette (style standard).
        apply_theme: Apply the suite's 'dark' or 'light' palette to the running QApplication (standalone windows only - plugins style their own dialog instead).
        suite_theme: Theme stored for the whole suite (launcher and tools).
        store_suite_theme: Save ``mode`` ('dark' or 'light') as the suite-wide theme.
        MemoryMonitorConfig (class): Configuration knobs for the memory monitor.
        MemoryAwareCache (class): Memory-aware data cache that spills large numpy arrays to disk via ``numpy.memmap``.
        MemoryMonitor (class): Background thread that publishes a callback whenever process RSS crosses the high-water mark.
        run_in_threadpool: Submit ``work_items`` -- ``(key, fn, args)`` triplets -- to a ``ThreadPoolExecutor`` and collect results keyed by ``key``.
        open_embedded: Legacy function name; return a native model, never launch Veusz.
        begin_native_plugin_export: Pick an output file before parsing; never modify the host document.
        finish_native_plugin_export: Same HDF5 writer for both plugin entry points and standalone GUIs.
        save_vszh5: Atomically write the native model with h5py; no Veusz runtime needed.
        _walk_widget_tree: Recursive sibling of ``Root.WalkWidgets``.
        _collect_referenced_dataset_names_via_walk: Original (v0.0.17) WalkWidgets-based traversal.
        _collect_referenced_dataset_names_via_text: v0.0.19 authoritative collector: dump the document via Veusz's own text serialization and regex out every ``Set('xData', 'name')``-style binding.
        _collect_referenced_dataset_names: v0.0.19: combine WalkWidgets-based traversal AND a document-text scan, returning the UNION of both whitelists.
        save_vszh5_minimized: Save exactly the referenced native datasets without deleting anything.
        _norm_tag_token: Lower-cased, stripped string form of a tag-tuple element.
        is_sentinel_tag_tuple: Return True iff ``tup`` is a placeholder/catch-all channel-tag tuple that should be EXCLUDED from dt-overlay datetime emission.
        is_delta_column: Deprecated stub kept for v0.0.17 back-compat.
        open_in_veusz_app: Launch an optional external Veusz GUI and load ``filename`` (typically a .vszh5 project just written by save_vszh5).
        detect_time_breaks: Detect large gaps in a 1-D monotonic-ish array of time samples and return a list of (start, end) pairs suitable for Veusz's ``axis-broken`` widget.
        parse_gap_string: Parse a human-readable gap string into MJD-days.
        column_unit_factor_from_day: Return the multiplier that converts a day-scale threshold into the units of the column named ``col_name``.
        detect_time_breaks_unit_aware: Unit-aware wrapper around ``detect_time_breaks``.
        break_pairs_to_breakpoints: Flatten a list of (start, end) tuples to Veusz's flat-list format [s1, e1, s2, e2, ...] as expected by the ``breakPoints`` setting of the ``axis-br...
        make_broken_x_axis: Add an ``axis-broken`` widget named ``x`` to ``graph`` with the given list of (start, end) break pairs, after removing the default ``x`` axis that...
        mjd_to_veusz_seconds: Convert an array of Modified Julian Dates (UTC) to the float-seconds units Veusz uses internally for date-time datasets.
        _coerce_axis_setting_to_str: Some Veusz axis settings (rotate, in particular) are documented as accepting either an int or its string form.
        style_datetime_x_axis: Apply the standard datetime tick-label format and angled rotation to an axis widget.
        is_gpu_available: True iff CuPy imported successfully AND at least one CUDA device is visible to it.
        gpu_backend_name: Short string describing the GPU backend status for the GUI/log.
        enable_gpu: Toggle the process-wide GPU acceleration flag.
        gpu_enabled: Return the current global GPU-enabled state.
        set_gpu_argsort_threshold: Set the minimum array size (samples) at which the optional CuPy path is used.
        get_gpu_argsort_threshold: Minimum array size (samples) for the optional CuPy argsort path.
        gpu_argsort: Return ``np.argsort(arr)`` (ascending, int64) with optional CuPy acceleration.
        AutoPlotMainWindow (class): Base main window mirroring the layout used by Touchstone_AutoPlot.py:
        open_maybe_gzipped: Return a binary file handle for ``path``; transparently gunzip *.gz.
        mjd_to_datestr: Convert an array of Modified Julian Dates (UTC, double precision) into an array of formatted timestamp strings.
        register_nrao_fits_units: Register the non-standard NRAO FITS unit aliases with astropy.units so that ``QTable.read`` / ``Table.read`` no longer emit a UnitsWarning.
        suppress_fits_unit_warnings: Context manager that filters the harmless FITS unit-related warnings emitted by ``astropy.io.fits`` and ``astropy.table`` when reading NRAO 1PPS-de...
        set_datetime_dataset: Push a Veusz datetime dataset, coercing values to plain floats.
        safe_dsname: Sanitize an arbitrary string for use as a Veusz dataset name.
        _stable_index: Deterministic non-negative index in ``[0, modulus)``.
        trace_style_for: Return ``(color_hex, line_style_str)`` for an identity key.
        apply_trace_style: Apply identity-stable color + line style + markers to a Veusz xy.
        build_sparse_datestr_dataset: Push a same-length sparse date-string text dataset to Veusz.
        build_full_datestr_dataset: Push a same-length per-point date-string text dataset to Veusz.
        style_xy_datetime_labels: Apply rotated, compact text formatting to an xy widget's labels.
        build_density_datestr_dataset: Push a same-length date-string text dataset with continuous density.
        build_textx_dataset: Push a same-length per-point date-string text dataset for use as xData.
        configure_axis_labels_mode: Switch a Veusz axis to ``mode='labels'`` with sensible defaults.
        build_dtnum_dataset: Push a numeric Veusz-datetime-seconds dataset for use as xData.
        configure_axis_datetime_mode: Switch a Veusz x axis to ``mode='datetime'`` and style its ticks.
        mjd_break_pairs_to_dtsec: Convert MJD (start, end) gap pairs to Veusz datetime-seconds pairs.
# %%%%% Variable Descriptions
    Define all utilized variables
        _DATASET_REF_SETTINGS: v0.0.19: trimmed the tuple to entries that *actually* hold dataset names.  ``posn`` is a position rectangle, ``key`` is a legend-label STRING -- ne...
        _DOC_DATASET_REF_REGEX: v0.0.19: regex used to extract dataset references from the document's own saved-text representation.  This is the AUTHORITATIVE source of truth --...
        SENTINEL_TAG_TOKENS: The user can extend the sentinel set in place; the predicate is case-insensitive and trims whitespace before matching.
        DELTA_COLUMN_NAME_PATTERNS: Deprecated column-name predicate -- retained as a no-op for any external script that may have imported it.  v0.0.17 superseded this with the tuple-...
        DAY_SCALE_COLUMN_NAMES: Unknown column names default to 1.0 (i.e. assume MJD-day units), which is the safe choice when the column appears to be a generic time axis: a too-...
        SECOND_SCALE_COLUMN_NAMES: column names whose values are seconds (time axes)
        _GAP_TOKEN_REGEX: Accepts strings like ``"5m3d2h"`` (5 months, 3 days, 2 hours), with or without whitespace between tokens, case-insensitive.  A bare number (no unit...
        GAP_STRING_HELP: (0.4.0) Tool tip of the gap text boxes.
        _DAYS_PER_MONTH: Average Gregorian month length (365.25 / 12) -- intentionally the same value used by other scientific tools (e.g. pandas) so 1m + 1m + ... = ~year.
        _DAYS_PER_HOUR: 1 / 24 (gap text parsing)
        _DAYS_PER_MINUTE: 0.4.0
        _DAYS_PER_SECOND: 0.4.0
        MJD_VEUSZ_EPOCH_MJD: MJD that corresponds to Veusz's internal date-time zero (2009-01-01 UTC).
        DEFAULT_DATETIME_TICK_FORMAT: 0.3.0: ISO 8601
        LOCAL_TIME_AXIS_LABEL: 0.3.0: label of a local date-time axis (the offset is appended on save).
        DEFAULT_DATETIME_TICK_ROTATE_DEG: Default tick label rotation -- 45 degrees keeps a 19-character label readable at typical cadences.  Veusz accepts integer or stringified integer de...
        DEFAULT_DATETIME_MAJOR_TICKS_TARGET: Target number of major ticks along the x axis.  Veusz auto-picks the spacing to satisfy roughly this many major ticks; we keep it modest so the rot...
        _GPU_ENABLED: Process-wide enable flag (driven by the GUI checkbox).  Off by default.
        _GPU_ARGSORT_THRESHOLD: Minimum array size at which we attempt the GPU sort.  Tuneable.
        QUEUED_CONNECTION: queued signal connection, valid for PySide6, PyQt5 and PyQt6
        MJD_EPOCH_JD_OFFSET: Modified Julian Date epoch: MJD 0.0 == 1858-11-17 00:00:00 UTC. JD = MJD + 2400000.5  -- we implement the conversion directly without astropy.time...
        _NRAO_UNITS_REGISTERED: NRAO 1PPS-delta FITS files use non-standard FITS unit strings: * CHANNELA / CHANNELB carry unit='none'  (text columns, no real unit) * DELTAT  carr...
        TRACE_COLOR_PALETTE: trace colours, cycled
        TRACE_LINE_STYLES: line styles cycled after the colours
        TRACE_STYLE_VARY_THRESHOLD: 16
        DEFAULT_DATETIME_LABEL_ANCHORS: legacy date-label pages: labels per trace
        DEFAULT_DATETIME_LABEL_ANGLE: legacy date-label pages: label angle (degrees)
        DEFAULT_DATETIME_LABEL_SIZE: legacy date-label pages: label font size
        DEFAULT_DATETIME_LABEL_FMT: legacy date-label pages: strftime format
        PYQTGRAPH_ERROR = None
        _PSUTIL_AVAILABLE = True
        _HAS_CUPY = True
# %%%%% More Info
    Shared by FITS and Franks AutoPlot and their plugins.  numpy.memmap is
    the RAM -> disk transport of MemoryAwareCache (no pickling, the kernel page
    cache does the writing).  The memory monitor runs in its own thread: its
    callback must not touch Qt widgets (log() is safe).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.3.0: Suite theme; Veusz date-times in local ISO 8601.
Date: 2026-11-09
#              Date-time axes (dt_labels pages and their overlays):
#                * build_dtnum_dataset() stores the time stamps as a Veusz
#                  DATE dataset (seconds since 2009-01-01 UTC; it was a plain
#                  numeric dataset) and a "<name>_local" date dataset on the
#                  clock of the computer doing the processing (DST-aware;
#                  VSZH5_DISPLAY_TZ overrides), via
#                  NativeDocument.SetDataLocalTime (writer 0.0.25).
#                * configure_axis_datetime_mode()/style_datetime_x_axis()
#                  with the default format mark the axis as a local-time
#                  axis: when the document is saved the traces on it draw
#                  the local dataset and the ticks read ISO 8601, 24 h, with
#                  the offset, e.g. 2026-11-09T08:39:00-05:00, and the label
#                  "Local time (ISO 8601, UTC-05:00)".  Tick labels stay at
#                  45 degrees.  DEFAULT_DATETIME_TICK_FORMAT is now the ISO
#                  form with a "T".
#                * mjd_break_pairs_to_dtsec() returns the broken-axis gaps
#                  on the same local clock (they were UTC while the axis now
#                  shows local time).
#              Theme (2026-11-05):
#              The window starts in the theme stored by the launcher and the
#              other tools (_suite_gui: QSettings "NRAO-GBO" / "GBO
#              Electronics Scientific Plotting Suite", key "theme") instead
#              of a fixed dark start, and View > Light / Dark stores the
#              choice for every tool.  apply_theme() uses the suite palette
#              (the launcher's colours).  The Veusz plugins no longer call it
#              (it re-themed the whole Veusz program); they style only their
#              own dialog (_suite_gui.dialog_stylesheet).
#
# %%%% 0.2.0: pyqtgraph "Plots" tab, Help > About (NRAO).
Date: 2026-11-05
#              AutoPlotMainWindow now has two tabs: "Processing" (file list,
#              options, log - unchanged) and "Plots", an interactive
#              pyqtgraph view of the in-memory project built by
#              _autoplot_pyqtgraph.ProjectPlotPanel (modelled on the AB
#              Power Meter Monitor 1.7.0 live plots).  show_plots(doc) fills
#              it after processing; the View theme also re-colours it.
#              Without pyqtgraph the tab says how to install it; saving and
#              Open in Veusz are unchanged.  Help > About shows the tool's
#              name and version, author, NRAO / GBO, e-mail, licence and the
#              library versions (Show Details).
#
# %%%% 0.1.0: Shared native HDF5 save and plugin export workflow.
Date: 2026-10-11
#              No Veusz imports or embedded process for document generation.
#              Minimized saves select model references without removing data.
#              Shared native writer is in _autoplot_native.py.
#
# %%%% 0.0.19: Minimized save union fix + string gap parser + key-hide.
Date: 2026-05-16
#              Three independent fixes/features bundled together:
#
#              (1) MINIMIZED-SAVE NO-OP FIX.  Inspection of a user-
#                  attached v0.0.18 minimized .vszh5 showed the
#                  minimized file was actually LARGER than the
#                  non-minimized companion (1869 datasets in both,
#                  13.86 MB vs 13.65 MB).  Only ~162 dataset names
#                  appear in the document XML's ``Set('xData'/'yData'
#                  /...)`` bindings, yet the v0.0.17 collector reported
#                  enough references that nothing was evicted.
#
#                  Root cause: ``Root.WalkWidgets`` on Veusz 3.6.x can
#                  silently return widget nodes whose ``.path`` resolves
#                  to entries that ``doc.Get('<path>/xData')`` accepts
#                  for many incidental settings, polluting the
#                  whitelist.  The trimmed ``_DATASET_REF_SETTINGS``
#                  helps but does not eliminate the issue.
#
#                  Fix: ``_collect_referenced_dataset_names`` now
#                  takes the UNION of two independently-derived
#                  whitelists:
#                    * ``_collect_referenced_dataset_names_via_walk``
#                      (the v0.0.17 widget traversal, retained as a
#                      fallback / cross-check).
#                    * ``_collect_referenced_dataset_names_via_text``
#                      which serializes the document via
#                      ``doc.Action('SaveToString')`` (with fallbacks
#                      to ``GetSaveString`` and direct method calls)
#                      and regex-extracts every ``Set('xData', 'name')``
#                      binding -- exactly the bindings Veusz writes to
#                      disk.  This is the authoritative reference set;
#                      anything not in it cannot possibly be loaded
#                      back by Veusz on reopen.
#                  The trimmed ``_DATASET_REF_SETTINGS`` removes
#                  ``posn`` and ``key`` (position rectangles and legend
#                  label STRINGS -- never dataset references).
#
#              (2) STRING GAP PARSER.  ``parse_gap_string(text)`` accepts
#                  human-readable strings like ``"5m3d2h"`` (5 months,
#                  3 days, 2 hours), with or without whitespace, case-
#                  insensitive.  Units: ``m`` = months (30.4375 days),
#                  ``d`` = days, ``h`` = hours.  A plain number (no
#                  units) is interpreted as hours, preserving the
#                  pre-0.0.19 GUI behaviour where the spinbox value was
#                  always hours.  Returns MJD-days so the engine
#                  signature (``absolute_gap_days``) does not change.
#                  Empty/unparseable input returns 0.0 (= disable
#                  absolute threshold).
#
#              (3) HIDE-KEY OPTION.  New plumbing parameter
#                  ``hide_keys`` flows from the GUI / plugin handlers
#                  through to ``_build_pages`` and
#                  ``build_unit_overlay_pages``; when True, every key
#                  widget added to a graph is created with
#                  ``Border/hide`` set so the legend frame does not
#                  draw.  Default False (pre-0.0.19 behaviour).
#
# %%%% 0.0.18: Unit-aware time-break detection (manual gap units fix).
Date: 2026-05-16
#              Fixes a unit-mismatch bug where the GUI "Manual gap
#              (hours)" spinbox was converted to MJD-days but compared
#              directly against ``np.diff(x)`` even when ``x`` was not
#              in days.  For FITS files whose sort_key column is
#              ``TIME`` / ``TIMESTAMP`` (seconds-of-day, 0-86400 range)
#              the day-scale threshold became microscopic compared to
#              the second-scale diffs and every diff exceeded it,
#              manufacturing a broken X axis on the per-file pages and
#              the seconds-axis overlay even when no real gap existed.
#              The dt_labels overlay was unaffected because its break
#              detection already ran in MJD-space.
#
#                * NEW: ``column_unit_factor_from_day(col_name)`` --
#                  returns 1.0 for day-scale time columns (MJD, DMJD,
#                  JD) and 86400.0 for second-scale time columns
#                  (TIME, TIMESTAMP, SECONDS, UT_SECONDS, ELAPSED,
#                  ...).  Case-insensitive, whitespace-trimmed.
#                  Unknown names default to 1.0 (the safer choice).
#                  Backed by two module-level constants
#                  ``DAY_SCALE_COLUMN_NAMES`` and
#                  ``SECOND_SCALE_COLUMN_NAMES``.
#                * NEW: ``detect_time_breaks_unit_aware(x, col_name,
#                  k_factor, absolute_gap_days)`` -- thin wrapper that
#                  multiplies the day-scaled threshold by the column
#                  factor before calling ``detect_time_breaks``.  All
#                  per-file and overlay break-detection call sites in
#                  FITS_AutoPlot and Franks_AutoPlot now route through
#                  this wrapper.  The K-factor auto path is unchanged
#                  (its threshold is derived from ``median(|diff(x)|)``
#                  in column units, so no conversion is needed).
#                * No behavior change for Franks_AutoPlot in practice
#                  because its sort_key is hard-coded to ``MJD``
#                  (factor = 1.0), but the call sites still moved to
#                  the unit-aware helper so future column-name changes
#                  are handled automatically.
#                * Revision history kept in DESCENDING semantic-version
#                  order.
#
# %%%% 0.0.17: Minimized .vszh5 save + sentinel-tag dt overlay filter.
Date: 2026-05-16
#              Two user-facing additions and one correctness fix:
#
#                * NEW: ``save_vszh5_minimized(doc, filename, log_cb)``.
#                  Walks every widget in the document via
#                  ``Root.WalkWidgets()`` and collects every dataset name
#                  referenced as ``xData`` / ``yData`` / ``labels`` /
#                  ``scalePoints`` / ``yDataLow`` / ``yDataHigh`` /
#                  ``xDataLow`` / ``xDataHigh``.  Datasets not in that
#                  whitelist are removed via ``doc.RemoveData(name)``
#                  prior to saving and re-injected from a value cache
#                  immediately after the save returns -- so the in-memory
#                  ``doc`` is unchanged from the caller's perspective.
#                  The resulting .vszh5 contains ONLY the datasets that
#                  the plot definitions actually consume, which is what
#                  the new "Minimized Veusz File" GUI checkbox enables.
#                * NEW: ``GUI_MINIMIZED_VESZ`` field group plumbing.
#                  Both GUIs gained a "Minimized Veusz File" checkbox
#                  and a nested, dependent "Generate Full Veusz file"
#                  sub-checkbox (only enabled when the parent is on).
#                  When both are checked, the save path writes BOTH the
#                  minimized file (``<name>.vszh5``) AND the legacy full
#                  file (``<name>_full.vszh5``).  When only the parent is
#                  checked, only the minimized file is written.  When
#                  neither is checked, the legacy full file is written
#                  as before.
#                * FIX: sentinel channel-tag tuples (e.g. ``("dataset",
#                  "dataset")``) are now EXCLUDED from the FITS
#                  datetime-overlay pages.  These are placeholder
#                  catch-all groupings emitted when the per-row tag
#                  columns carry a literal placeholder string (the FITS
#                  reader still groups them, but the resulting trace
#                  does not correspond to a real channel pair).  Their
#                  per-file MJD coverage is degenerate, which when fed
#                  into the combined-MJD break detection used by the dt
#                  overlay pages manufactured spurious broken-axis
#                  breaks.  New module-level constant
#                  ``SENTINEL_TAG_TOKENS`` lists the placeholder tokens
#                  (case-insensitive, whitespace-trimmed);
#                  ``is_sentinel_tag_tuple(tup)`` is the predicate the
#                  FITS overlay uses to gate dt emission.  Real channel
#                  tuples like ``("A", "B")`` or ``("chA", "chB")`` pass
#                  through.  The seconds-overlay trace is still drawn
#                  for sentinel tuples -- harmless on a unitless
#                  seconds axis.  Franks_AutoPlot is unaffected by this
#                  fix (its overlay is one trace per page).  The older
#                  v0.0.17-draft column-name predicate ``is_delta_column``
#                  is kept as a no-op stub for back-compat.
#                * Revision history kept in DESCENDING semantic-version
#                  order.
#
# %%%% 0.0.16: dt_labels page upgraded to mode='datetime' (proper date ticks).
Date: 2026-05-16
#              v0.0.15 made the dt_labels page bind xData to a per-point
#              TEXT dataset and set axis mode='labels' so Veusz pulled
#              tick strings from the plotter.  That works but yields
#              UNIFORM sample spacing (one tick per index 0..N-1) and
#              uneven tick collisions on long traces.  v0.0.16 swaps
#              that for a true datetime axis:
#
#                * NEW: ``build_dtnum_dataset(doc, name, mjd_array)`` --
#                  pushes a numeric Veusz-datetime-seconds dataset (the
#                  internal numeric format Veusz uses for its native
#                  datetime axis: seconds since 2009-01-01 UTC).  We do
#                  NOT use ``SetDataDateTime`` here because the v0.0.13
#                  audit showed that path fires hundreds of internal
#                  exceptions on Veusz 3.4 -- a plain numeric dataset
#                  with the axis itself flipped to mode='datetime'
#                  is just as date-aware on the axis side and avoids
#                  Veusz 3.4's broken DatasetDateTime code path.
#                * NEW: ``configure_axis_datetime_mode(axis, ...)`` --
#                  sets ``axis.mode.val = 'datetime'`` and applies the
#                  shared ``style_datetime_x_axis`` formatting (date
#                  tick label format, rotation, gridlines).  Works on
#                  both ``axis`` and ``axis-broken`` because the v0.0.9
#                  ``style_datetime_x_axis`` already does.
#                * NEW: ``mjd_break_pairs_to_dtsec(pairs)`` -- converts
#                  a list of MJD (start, end) gap pairs (the output of
#                  ``detect_time_breaks`` on an MJD x array) into Veusz
#                  datetime seconds so the same pairs can drive an
#                  ``axis-broken`` widget on the dt_labels page when
#                  one was installed on the numeric-seconds dt page.
#                  The numeric-seconds dt page detects breaks in the
#                  SECONDS x array, not the MJD array, so it has its
#                  own (already-correct) break pairs; the dt_labels
#                  page detects breaks in the MJD array (which is what
#                  the AutoPlot scripts hold) and converts to Veusz
#                  seconds via this helper.
#                * ``build_textx_dataset`` and ``configure_axis_labels_mode``
#                  are retained as deprecated back-compat shims so any
#                  external scripts that imported them keep working.
#                * GUI/plugin field surface is UNCHANGED from v0.0.15:
#                  the same density spinbox + two emit checkboxes drive
#                  both dt variants; only the under-the-hood encoding
#                  of the dt_labels page changes.
#                * Broken-axis parity: when the numeric-x dt page gets
#                  a broken axis on a file/overlay, the dt_labels page
#                  for the same file/overlay now ALSO gets a broken
#                  axis, computed from the same gap thresholds applied
#                  to the MJD x array.  Previously the dt_labels page
#                  always rendered a continuous axis.
#                * GPU + parallelization re-audit: ``build_dtnum_dataset``
#                  is O(N) per trace (one vectorised
#                  ``mjd_to_veusz_seconds`` call); no new sort sites.
#                * Revision history kept in DESCENDING semantic-version
#                  order across every shared file.
#
# %%%% 0.0.15: Continuous datetime-label density + text-x dt-page variant.
Date: 2026-05-16
#              Two GUI/plugin changes layered on top of v0.0.14:
#                * The boolean ``datetime_full_labels`` checkbox is
#                  REPLACED by an integer percentage
#                  ``datetime_label_density_pct`` in [0, 100] (default
#                  10).  Anchor count per trace is
#                  ``round(pct/100 * len(trace))`` so 0 -> no labels,
#                  100 -> a label on every point, and values in between
#                  give a smooth sweep.  Internally we generalised the
#                  sparse helper: ``build_density_datestr_dataset()``
#                  takes a density pct and computes anchors with the
#                  existing evenly-spaced-on-finite-indices algorithm.
#                  ``build_sparse_datestr_dataset`` and
#                  ``build_full_datestr_dataset`` remain as v0.0.14
#                  shims for older callers (they now both delegate to
#                  the density helper internally).
#                * NEW dt-page variant emitted alongside the existing
#                  numeric-seconds dt page: a "dt_labels" page where the
#                  xy widget's ``xData`` binds to the same text dataset
#                  used for labels and the x axis is set to
#                  ``mode='labels'``.  Veusz's xy widget exposes
#                  ``getAxisLabels()`` which, when xData is a text
#                  dataset, returns ``(strings, arange(N))`` -- the
#                  axis then plots samples at integer positions 0..N-1
#                  with each tick labeled by the corresponding date
#                  string.  Uniform sample spacing (NOT proportional to
#                  elapsed time) but the date string is now ON the axis
#                  rather than as an annotation next to the point.  The
#                  user can compare both representations side-by-side.
#                  Helpers:
#                    - build_textx_dataset(doc, name, mjd_array,
#                      fmt=DEFAULT_DATETIME_LABEL_FMT, log_cb=None)
#                      pushes a same-length per-point text dataset
#                      (no "" gaps because it has to serve as the x
#                      data).  Non-finite MJDs map to "NaN" so length
#                      is preserved -- Veusz draws no point at that
#                      index because the y dataset is NaN there.
#                    - configure_axis_labels_mode(axis_widget,
#                      angle=45, size='6pt', max_ticks=20) sets
#                      axis.mode='labels', applies rotation, and caps
#                      MajorTicks.number so dense traces do not spam
#                      every integer position with a tick.  The
#                      max_ticks cap is derived from
#                      ``datetime_label_density_pct`` * N so the same
#                      density slider controls BOTH dt variants.
#                * Two GUI checkboxes added in both standalone GUIs and
#                  both plugins:
#                    - "Emit dt page (numeric x + sparse labels)"
#                      (default ON, controls the v0.0.14 dt page).
#                    - "Emit dt_labels page (text x, mode=labels)"
#                      (default ON, controls the new variant).
#                * datetime_label_density_pct, datetime_emit_numeric_dt,
#                  datetime_emit_text_dt flow through every call site
#                  (push_to_veusz / push_franks_to_veusz / build_unit_
#                  overlay_pages / build_unit_overlay_pages_franks) and
#                  both plugin dialogs.
#                * GPU + parallelization re-audited with the new code
#                  paths active.  ``gpu_argsort`` still wraps every
#                  sort site (per-file, per-tag, overlay) including
#                  the new dt_labels page (which reuses the same sort
#                  permutation as the numeric dt page since the y data
#                  is identical).  ``run_in_threadpool`` is unchanged;
#                  the new dt_labels page emission is a thin Veusz
#                  Add()/SetData call per trace and adds no measurable
#                  hot path, so no extra parallelization is warranted.

# %%%% 0.0.14: Datetime display via xy.labels (text dataset) instead of
Date: 2026-05-16
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
Date: 2026-05-16
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
Date: 2026-05-16
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
Date: 2026-05-16
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
Date: 2026-05-16
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
Date: 2026-05-16
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

# %%%% 0.0.8: Added ``open_in_veusz_app(filename)`` helper that launches the
Date: 2026-05-16
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
Date: 2026-05-16
#             (read / parse / per-column) plus helper methods
#             ``show_progress_bars()``, ``hide_progress_bars()``,
#             ``begin_column_progress()`` and ``tick_column_progress()``.
#             Subclasses use these to drive a multi-stage progress UI.

# %%%% 0.0.5: Added Spyder-style cell markers (``# %% TITLE`` for top-level
Date: 2026-05-16
#             sections, ``# %%% TITLE`` for nested sections) on the
#             existing dashed banner blocks so the file is navigable in
#             Spyder's Outline / cell navigator.  Pure cosmetic change --
#             no runtime behaviour modified.

# %%%% 0.0.4: Added register_nrao_fits_units() and suppress_fits_unit_warnings()
Date: 2026-05-16
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
Date: 2026-05-16
#             NaN floats (Veusz natively supports NaN in numeric datasets;
#             plots simply skip NaN samples). Text datasets cannot carry a
#             true NaN, so non-finite MJD inputs to mjd_to_datestr() now
#             yield the explicit sentinel string ``"NaN"`` (length-preserving)
#             rather than an empty string -- this prevents NaN rows from
#             being silently dropped or confused with a missing token.

# %%%% 0.0.2: Added mjd_to_datestr() helper for MJD -> YYYY-MM-DD_HH:MM:SS
Date: 2026-05-16
#             string conversion used by all four AutoPlot modules.

# %%%% 0.0.1: Initial implementation of shared infrastructure
Date: 2026-05-16

"""
# %% Imports
from __future__ import annotations

__version__ = "0.4.1"          # Semantic Versioning: see the Revisions entries above

# %%% IMPORTS - Standard library

import gc
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import weakref
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
from _autoplot_native import NativeDocument

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

# pyqtgraph "Plots" tab (optional: the GUIs still work and save without it)
try:
    import _autoplot_pyqtgraph as autoplot_pg
    PYQTGRAPH_ERROR = None
except Exception as _exc:  # pragma: no cover - pyqtgraph missing or broken
    autoplot_pg = None
    PYQTGRAPH_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

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
    """Apply the suite's 'dark' or 'light' palette to the running QApplication
    (standalone windows only - plugins style their own dialog instead)."""
    try:
        import _suite_gui
        app.setStyle("Fusion")
        app.setPalette(_suite_gui.build_palette("dark" if mode == "dark" else "light"))
        return
    except Exception:  # pragma: no cover - _suite_gui missing
        pass
    app.setStyle("Fusion")
    app.setPalette(make_dark_palette() if mode == "dark" else make_light_palette())


def suite_theme(default: str = "dark") -> str:
    """Theme stored for the whole suite (launcher and tools)."""
    try:
        import _suite_gui
        return _suite_gui.saved_theme(default)
    except Exception:  # pragma: no cover
        return default


def store_suite_theme(mode: str) -> None:
    """Save ``mode`` ('dark' or 'light') as the suite-wide theme.

    Delegates to _suite_gui.save_theme so the launcher and every program open
    with the same theme next time; silently does nothing when _suite_gui is
    not importable (a plugin inside Veusz, say).
    """
    try:
        import _suite_gui
        _suite_gui.save_theme(mode)
    except Exception:  # pragma: no cover
        pass


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
        """Create an empty cache with ``config`` (defaults if None).

        Spilled arrays go to ``config.cache_dir`` or a new temporary folder
        ``autoplot_cache_*``; a lock makes the cache safe to use from the reader
        threads.
        """
        self.cfg = config or MemoryMonitorConfig()
        self._cache: Dict[str, Any] = {}
        self._spilled: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.cache_dir = self.cfg.cache_dir or tempfile.mkdtemp(prefix="autoplot_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ----- introspection ----------------------------------------------------
    def rss_mb(self) -> float:
        """Resident set size of this process in MiB (0.0 without psutil)."""
        if _PSUTIL_AVAILABLE:
            try:
                return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def over_high_water(self) -> bool:
        """True when the process RSS is at or above the configured high-water mark.
        """
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
        """Cached value for ``key`` (an array or a read-only memmap), else ``default``.
        """
        with self._lock:
            return self._cache.get(key, default)

    def items(self):
        """Snapshot list of (key, value) pairs, taken under the lock."""
        with self._lock:
            return list(self._cache.items())

    def keys(self):
        """Snapshot list of the cached keys, taken under the lock."""
        with self._lock:
            return list(self._cache.keys())

    def clear(self) -> None:
        """Drop every cached entry and delete the spill files from disk.

        Call it after a batch (or let the GUI's "Flush RAM" do it) to release both
        RAM and temporary disk space.
        """
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
        """Write ``arr`` to a .npy file and cache a read-only memmap of it.

        np.save / np.load(mmap_mode='r') keeps the dtype and shape; the file name
        is the key with unsafe characters replaced plus a random suffix, so two
        keys never collide.
        """
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

    0.4.0 - thread safety:
      * The callback runs in THIS thread: it must not touch Qt widgets.
        AutoPlotMainWindow.log() and the plugin dialogs' append_log() are
        safe to call from it (they hand the text to the GUI thread).
      * A bound method is held through a weak reference, so the monitor
        never keeps a window or dialog alive.  Before, a lambda capturing
        the window/dialog meant the monitor thread could drop the last
        reference and destroy Qt widgets outside the GUI thread, which
        crashed PySide6 6.12 under Python 3.12 later, at random.
      * stop() wakes the thread at once and waits for it to finish (it
        used to sleep out its poll interval and outlive its owner).
    """

    def __init__(self, cache: MemoryAwareCache,
                 callback: Optional[Callable[[float], None]] = None) -> None:
        """Daemon thread that polls ``cache``'s RSS; ``callback(rss_mb)`` is called
        each time the RSS rises past the high-water mark.

        A bound method is stored as a weakref.WeakMethod (dropped when its
        object goes); any other callable is stored as given.
        """
        super().__init__(daemon=True)
        self.cache = cache
        if callback is not None and getattr(callback, "__self__", None) is not None \
                and getattr(callback, "__func__", None) is not None:
            self._callback_ref = weakref.WeakMethod(callback)
        else:
            self._callback_ref = (lambda cb=callback: cb)
        # Not "_stop": threading.Thread has its own _stop() method, which
        # join() calls (Python <= 3.12); shadowing it broke join().
        self._stop_event = threading.Event()

    @property
    def callback(self) -> Optional[Callable[[float], None]]:
        """The callback, or None (none given, or its object has been deleted)."""
        return self._callback_ref()

    def stop(self, timeout: float = 5.0) -> None:
        """Wake the thread, ask it to finish and wait up to ``timeout`` seconds.

        Safe to call more than once, before start() and from the monitor
        thread itself (then it does not wait).
        """
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    def run(self) -> None:
        """Poll the RSS every ``poll_interval_sec`` until stop() is called.

        The callback fires once per crossing (edge triggered), not on every poll
        while the RSS stays high; callback errors are ignored so the monitor keeps
        running.  The wait is on the stop event, so stop() ends it at once.
        """
        prev_over = False
        while not self._stop_event.is_set():
            callback = self.callback
            if callback is not None:
                try:
                    rss = self.cache.rss_mb()
                    over = self.cache.over_high_water()
                    if over and not prev_over:
                        callback(rss)
                    prev_over = over
                except Exception:
                    pass
            callback = None                      # hold no reference while waiting
            self._stop_event.wait(self.cache.cfg.poll_interval_sec)


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
      * Qt updates and native model assembly are kept on the calling thread;
        file decoding runs concurrently. The native writer no longer has
        an embedded Veusz thread-affinity constraint.
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
    """Legacy function name; return a native model, never launch Veusz."""
    return NativeDocument(name)


def begin_native_plugin_export(fields):
    """Pick an output file before parsing; never modify the host document."""
    filename = str(fields.get("output_file") or "").strip()
    if not filename:
        filename, _ = QFileDialog.getSaveFileName(
            None, "Export native Veusz project", "",
            "Veusz HDF5 Projects (*.vszh5)")
    if not filename:
        return None, None
    return NativeDocument("AutoPlot plugin export"), filename


def finish_native_plugin_export(doc, filename, fields, log_cb=None):
    """Same HDF5 writer for both plugin entry points and standalone GUIs."""
    minimized = bool(fields.get("minimized", True))
    written = doc.save(filename, minimized=minimized)
    paths = [written]
    if minimized and fields.get("also_full", False):
        paths.append(doc.save(os.path.splitext(written)[0] + "_full.vszh5"))
    if log_cb:
        for path in paths:
            log_cb("Native project saved: %s" % path)
    QMessageBox.information(None, "Native Veusz export",
                            "Wrote:\n%s\n\nThe open Veusz document was not changed."
                            % "\n".join(paths))
    return paths


def save_vszh5(doc, filename: str) -> str:
    """Atomically write the native model with h5py; no Veusz runtime needed."""
    if not filename.lower().endswith(".vszh5"):
        filename = os.path.splitext(filename)[0] + ".vszh5"
    if not isinstance(doc, NativeDocument):
        raise TypeError("Native .vszh5 output requires a NativeDocument")
    return doc.save(filename)


# %%% Minimized .vszh5 save (only plot-referenced datasets)
# ---------------------------------------------------------------------------
# Settings on Veusz widgets that hold a dataset NAME (string) the widget
# reads from at render time.  These are the 'dataset references' we want to
# preserve when writing a minimized .vszh5.  All of these are present on
# both Veusz 3.4 and 4.1; widgets that don't have a given setting simply
# raise on Get() and our caller swallows the exception.
#
# v0.0.19: trimmed the tuple to entries that *actually* hold dataset
# names.  ``posn`` is a position rectangle, ``key`` is a legend-label
# STRING -- neither is a dataset reference.  Including them in the v0.0.17
# tuple was harmless on paper (the loop's isinstance(item, str) filter
# would discard floats and label strings would not match any actual
# dataset name) but produced log noise and a slow Get() per widget.
_DATASET_REF_SETTINGS = (
    "xData", "yData",
    "labels", "scalePoints",
    "yDataLow", "yDataHigh",
    "xDataLow", "xDataHigh",
    "data",         # image / colorbar widget
)


# v0.0.19: regex used to extract dataset references from the document's
# own saved-text representation.  This is the AUTHORITATIVE source of
# truth -- the document XML/text format is exactly what Veusz writes to
# disk, so any dataset name that survives ``SaveToString`` is one that
# the document actually depends on.  Used as a fallback when the
# WalkWidgets-based traversal returns a suspiciously empty set.
_DOC_DATASET_REF_REGEX = re.compile(
    r"Set\(\s*'(?:xData|yData|labels|scalePoints|"
    r"yDataLow|yDataHigh|xDataLow|xDataHigh|data)'\s*,\s*'([^']+)'\s*\)"
)


def _walk_widget_tree(node, depth=0, max_depth=64):
    # type: (object, int, int) -> Any
    """Recursive sibling of ``Root.WalkWidgets``.  v0.0.19 fallback used
    when ``WalkWidgets`` is missing or returns nothing useful.  Yields
    every descendant widget node accessible via ``.children``.
    Compatible with Veusz 3.4 / 3.6 / 4.1: ``WidgetNode`` exposes
    ``children`` as a list-like attribute on all of those.
    """
    if depth > max_depth:
        return
    yield node
    try:
        kids = list(node.children)
    except Exception:
        return
    for c in kids:
        for n in _walk_widget_tree(c, depth + 1, max_depth):
            yield n


def _collect_referenced_dataset_names_via_walk(doc, log_cb=None):
    # type: (object, Optional[Callable[[str], None]]) -> set
    """Original (v0.0.17) WalkWidgets-based traversal.  Walks every widget
    in ``doc`` and collects the set of dataset names referenced by widget
    settings.  v0.0.19 keeps this as the FIRST attempt; the caller falls
    back to the document-text scan if the result is suspiciously small.
    """
    used = set()
    visited_widgets = 0
    # Try WalkWidgets first; fall back to a recursive descent via
    # .children when WalkWidgets is not available or returns nothing.
    walker = None
    try:
        walker = doc.Root.WalkWidgets()
        walker = list(walker)
    except Exception as exc:
        if log_cb:
            log_cb("  Minimized-save: WalkWidgets unavailable (%s); "
                   "using recursive descent." % exc)
        try:
            walker = list(_walk_widget_tree(doc.Root))
        except Exception as exc2:
            if log_cb:
                log_cb("  Minimized-save: recursive descent failed: %s"
                       % exc2)
            return used
    if not walker:
        try:
            walker = list(_walk_widget_tree(doc.Root))
        except Exception:
            walker = []
    for w in walker:
        visited_widgets += 1
        try:
            wpath = w.path
        except Exception:
            continue
        for setname in _DATASET_REF_SETTINGS:
            try:
                val = doc.Get("%s/%s" % (wpath, setname))
            except Exception:
                continue
            if val is None:
                continue
            if isinstance(val, str):
                if val:
                    used.add(val)
            else:
                try:
                    for item in val:
                        if isinstance(item, str) and item:
                            used.add(item)
                except TypeError:
                    pass
    if log_cb:
        log_cb("  Minimized-save: walk visited %d widget(s), found "
               "%d dataset reference(s)."
               % (visited_widgets, len(used)))
    return used


def _collect_referenced_dataset_names_via_text(doc, log_cb=None):
    # type: (object, Optional[Callable[[str], None]]) -> set
    """v0.0.19 authoritative collector: dump the document via Veusz's own
    text serialization and regex out every ``Set('xData', 'name')``-style
    binding.  This matches exactly what Veusz writes to disk and so
    cannot disagree with the on-disk .vszh5 about which datasets are
    referenced.  Returns an empty set if neither serialization API is
    available.

    Veusz Embedded exposes a couple of paths to the saved text:

      * ``doc.Action('SaveToString')`` -- Veusz >= 3.0 returns the
        full document script as a single string.  Available on both
        3.4 and 4.1 (and the on-disk file we see in user reports was
        produced by version 3.6.2).
      * ``doc.GetSaveString()`` / ``doc.SaveToString()`` -- older
        Embedded API on some 3.x builds.
    """
    used = set()
    text = None
    for getter in ("SaveToString", "GetSaveString"):
        if text is not None:
            break
        # Try Action(...) first (newer Embedded API)
        try:
            text = doc.Action(getter)
            if not isinstance(text, str) or not text:
                text = None
        except Exception:
            text = None
        if text is not None:
            break
        # Try as a direct method on the Embedded object
        fn = getattr(doc, getter, None)
        if callable(fn):
            try:
                text = fn()
                if not isinstance(text, str) or not text:
                    text = None
            except Exception:
                text = None
    if not text:
        if log_cb:
            log_cb("  Minimized-save: document-text dump unavailable; "
                   "skipping text-based reference scan.")
        return used
    for m in _DOC_DATASET_REF_REGEX.finditer(text):
        nm = m.group(1)
        if nm:
            used.add(nm)
    if log_cb:
        log_cb("  Minimized-save: text scan found %d dataset "
               "reference(s) in document source." % len(used))
    return used


def _collect_referenced_dataset_names(doc, log_cb=None):
    # type: (object, Optional[Callable[[str], None]]) -> set
    """v0.0.19: combine WalkWidgets-based traversal AND a document-text
    scan, returning the UNION of both whitelists.  Either source alone
    has missed references in earlier reports (the WalkWidgets path can
    return spuriously few entries on Veusz 3.6 when widget .path is not
    populated; the text scan catches everything that survives the
    document's own text serialization).  Taking the union is the
    correctness-first choice: a dataset is evicted only when BOTH
    sources agree it is unreferenced.

    Returns the set of dataset name strings to preserve.
    """
    via_walk = _collect_referenced_dataset_names_via_walk(
        doc, log_cb=log_cb,
    )
    via_text = _collect_referenced_dataset_names_via_text(
        doc, log_cb=log_cb,
    )
    used = via_walk | via_text
    if log_cb:
        only_walk = len(via_walk - via_text)
        only_text = len(via_text - via_walk)
        log_cb("  Minimized-save: %d dataset(s) referenced (walk=%d, "
               "text=%d, walk-only=%d, text-only=%d)."
               % (len(used), len(via_walk), len(via_text),
                  only_walk, only_text))
    return used


def save_vszh5_minimized(doc, filename, log_cb=None):
    # type: (object, str, Optional[Callable[[str], None]]) -> str
    """Save exactly the referenced native datasets without deleting anything.

    All supported references are direct dataset names. Missing references fail
    before writing; unsupported expressions must be materialized by the caller.
    Tags are filtered to the same retained set. Text and 2-D data stay typed.
    """
    if not isinstance(doc, NativeDocument):
        raise TypeError("Native minimized save requires a NativeDocument")
    used = doc.referenced_datasets()
    if log_cb:
        log_cb("Native minimized save: keeping %d/%d datasets; "
               "the in-memory model is not modified."
               % (len(used), len(doc.datasets)))
    return doc.save(filename, minimized=True)



# %%% Sentinel channel-tag tuple filter for dt-overlay (v0.0.17)
# ---------------------------------------------------------------------------
# Some FITS tables tag every row with a placeholder channel-tag string
# (e.g. ``"dataset"``) instead of a real channel-pair label like
# ``("A", "B")`` or ``("channel_A", "channel_B")``.  When the AutoPlot
# grouper sees such a placeholder column, the resulting tag-tuple key
# collapses to a repeated sentinel like ``("dataset", "dataset")`` and
# the trace becomes a catch-all rather than a real per-channel sub-set.
#
# These catch-all traces typically carry degenerate MJD coverage (one or
# two valid samples per file) for time-difference columns, which when
# concatenated into the combined-MJD break detection used by the dt
# overlay pages manufactures large false-positive gaps in the broken
# axis.  v0.0.17 detects them via :func:`is_sentinel_tag_tuple` and
# excludes them from dt-overlay datetime emission.  Real channel-pair
# tuples (``("A", "B")``, ``("chA", "chB")``, ...) pass through.
#
# The user can extend the sentinel set in place; the predicate is
# case-insensitive and trims whitespace before matching.
SENTINEL_TAG_TOKENS = (
    "dataset",
    "none",
    "na",
    "n/a",
    "null",
    "",
)


def _norm_tag_token(t):
    # type: (object) -> str
    """Lower-cased, stripped string form of a tag-tuple element."""
    if t is None:
        return ""
    try:
        return str(t).strip().lower()
    except Exception:
        return ""


def is_sentinel_tag_tuple(tup):
    # type: (Optional[Tuple]) -> bool
    """Return True iff ``tup`` is a placeholder/catch-all channel-tag
    tuple that should be EXCLUDED from dt-overlay datetime emission.

    Two patterns are flagged as sentinel:
        1. Every element collapses (lower+strip) to the same token AND
           that token is in :data:`SENTINEL_TAG_TOKENS`.  Catches
           ``("dataset", "dataset")``, ``("none", "None")``, ``("NA", "NA")``.
        2. ``tup`` is empty / equals ``(None,)`` AND the empty-token form
           appears in :data:`SENTINEL_TAG_TOKENS`.  This is the
           no-tagging-at-all case the grouper already treats as a single
           catch-all bucket.  Excluding it from dt overlay emission is
           usually wrong for files that lack tag columns entirely, so by
           default this case is NOT treated as sentinel -- only the
           explicit repeated-sentinel-token form is filtered.

    Examples that return True:  ``("dataset", "dataset")``,
    ``("NONE", "none")``, ``("", "")`` (only if "" is in the sentinel set).
    Examples that return False: ``(None,)``, ``("A", "B")``,
    ``("chA", "chB")``, ``("dataset", "channel_A")`` (mixed).
    """
    if not tup:
        return False
    # The no-tagging sentinel (None,) is the grouper's universal bucket;
    # treat it as non-sentinel so files without tag columns still emit
    # their dt overlay trace.
    if tup == (None,):
        return False
    tokens = [_norm_tag_token(t) for t in tup]
    if not tokens:
        return False
    first = tokens[0]
    if any(t != first for t in tokens):
        return False
    # All tokens are equal; flag only if that token is sentinel.
    sentinels_lc = tuple(s.lower() for s in SENTINEL_TAG_TOKENS)
    return first in sentinels_lc


# Deprecated column-name predicate -- retained as a no-op for any external
# script that may have imported it.  v0.0.17 superseded this with the
# tuple-based filter above (see is_sentinel_tag_tuple) because the real
# culprit was the channel-tag tuple, not the column name itself.
DELTA_COLUMN_NAME_PATTERNS = ()


def is_delta_column(col_name):
    # type: (str) -> bool
    """Deprecated stub kept for v0.0.17 back-compat.  Always returns
    ``False`` now; the real dt-overlay filter is
    :func:`is_sentinel_tag_tuple`."""
    return False


# %%% Launch full Veusz GUI in current Python env
# ---------------------------------------------------------------------------
def open_in_veusz_app(filename, python_exe=None):
    """
    Launch an optional external Veusz GUI and load
    ``filename`` (typically a .vszh5 project just written by save_vszh5).

    Prefer VEUSZ_EXECUTABLE, then an executable on PATH. Legacy environment
    entry points are fallbacks. The viewer is never used to generate files.

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
    configured = os.environ.get("VEUSZ_EXECUTABLE")
    if configured:
        strategies.append(("configured-viewer", [configured, filename]))
    executable = shutil.which("veusz")
    if executable:
        strategies.append(("path-viewer", [executable, filename]))
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


# %%% column_unit_factor_from_day
# v0.0.18: Map a sort-key column name to the multiplier that converts
# an MJD-day-scaled threshold into the column's own units.  The user-
# facing "Manual gap (hours)" spinbox is converted to MJD-days in the
# GUI (``gap_absolute_days = hours / 24``), but ``detect_time_breaks``
# compares that threshold directly against ``np.diff(x)``.  When ``x``
# is *not* in MJD-days the threshold silently becomes meaningless
# (e.g. 5.0 days compared against seconds-of-day diffs in the 0.01-
# range -> every diff is below threshold; or 0.0833 days compared
# against MJD diffs of ~1e-5 -> every diff is above threshold).  This
# helper makes the conversion explicit and table-driven so callers can
# pass either ``x`` and the column name (preferred) or pre-convert the
# threshold themselves.
#
# Recognized column names (case-insensitive):
#
#   * Day-scale (factor = 1.0):     MJD, DMJD, JD, MJDS_DAYS
#   * Second-scale (factor = 86400):TIME, TIMESTAMP, TIMESTAMP_SEC,
#                                   SECONDS, SEC, UT_SECONDS,
#                                   UTSECONDS, UT, ELAPSED, ELAPSED_S,
#                                   T_SEC, DURATION_S
#
# Unknown column names default to 1.0 (i.e. assume MJD-day units),
# which is the safe choice when the column appears to be a generic
# time axis: a too-large threshold yields zero false breaks rather
# than a wall of false breaks.
DAY_SCALE_COLUMN_NAMES = frozenset((
    "MJD", "DMJD", "JD", "MJDS_DAYS",
))
SECOND_SCALE_COLUMN_NAMES = frozenset((
    "TIME", "TIMESTAMP", "TIMESTAMP_SEC",
    "SECONDS", "SEC", "UT_SECONDS", "UTSECONDS",
    "UT", "ELAPSED", "ELAPSED_S", "T_SEC", "DURATION_S",
))


# %%% parse_gap_string
# v0.0.19: human-readable gap string parser.
#
# Accepts strings like ``"5m3d2h"`` (5 months, 3 days, 2 hours), with or
# without whitespace between tokens, case-insensitive.  A bare number
# (no unit suffix) is interpreted as hours, preserving the pre-0.0.19
# behaviour of the GUI spinbox which was always in hours.  Returns the
# total absolute gap expressed in MJD-days so callers can hand the
# value straight to ``detect_time_breaks_unit_aware`` without changing
# the engine signature.
_GAP_TOKEN_REGEX = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(min|[mMdDhHsS]?)", re.IGNORECASE
)
#: (0.4.0) Tool tip of the gap text boxes.
GAP_STRING_HELP = ("Units: s = seconds, min = minutes, h = hours, d = days, "
                   "m = months (30.4375 days).  Examples: '90s', '15min', '12h', "
                   "'3d', '1m 3d 2h'.  A bare number is hours.")

# Average Gregorian month length (365.25 / 12) -- intentionally the same
# value used by other scientific tools (e.g. pandas) so 1m + 1m + ... = ~year.
_DAYS_PER_MONTH = 30.4375
_DAYS_PER_HOUR = 1.0 / 24.0
_DAYS_PER_MINUTE = _DAYS_PER_HOUR / 60.0     # 0.4.0
_DAYS_PER_SECOND = _DAYS_PER_MINUTE / 60.0   # 0.4.0


def parse_gap_string(text, log_cb=None):
    # type: (object, Optional[Callable[[str], None]]) -> float
    """Parse a human-readable gap string into MJD-days.

    Supported syntaxes (case-insensitive, whitespace tolerated)::

        "5m3d2h"   -> 5 months + 3 days + 2 hours, in days
        "5m 3d 2h" -> same
        "3d"       -> 3 days
        "120h"     -> 5 days
        "120"      -> 120 hours = 5 days   (bare number = hours)
        "0" / ""   -> 0.0  (disables absolute threshold)
        None       -> 0.0

    On parse failure returns 0.0 and (when ``log_cb`` is provided)
    emits a warning so the GUI log shows that the entry was ignored.
    Mixed unit ordering (e.g. ``"2h5m"``) is accepted -- the parser
    sums all matched tokens.
    """
    if text is None:
        return 0.0
    try:
        s = str(text).strip()
    except Exception:
        return 0.0
    if not s:
        return 0.0
    # Strip ALL internal whitespace so "5m 3d 2h" and "5m3d2h" parse the
    # same way.  Then iterate number+unit pairs.
    compact = "".join(s.split())
    if not compact:
        return 0.0

    # Fast path: pure number (no unit suffix anywhere).
    try:
        as_float = float(compact)
    except (TypeError, ValueError):
        as_float = None
    if as_float is not None:
        # Bare number => hours (pre-0.0.19 spinbox semantics).
        return as_float * _DAYS_PER_HOUR

    total_days = 0.0
    matched_span = 0
    saw_token = False
    for m in _GAP_TOKEN_REGEX.finditer(compact):
        num_str = m.group(1)
        unit = (m.group(2) or "").lower()
        if not num_str:
            continue
        try:
            value = float(num_str)
        except (TypeError, ValueError):
            continue
        if unit == "min":                            # 0.4.0
            total_days += value * _DAYS_PER_MINUTE
        elif unit == "s":                            # 0.4.0
            total_days += value * _DAYS_PER_SECOND
        elif unit == "m":
            total_days += value * _DAYS_PER_MONTH
        elif unit == "d":
            total_days += value
        elif unit == "h":
            total_days += value * _DAYS_PER_HOUR
        else:
            # No unit on this fragment -- treat as hours (back-compat).
            total_days += value * _DAYS_PER_HOUR
        matched_span = m.end()
        saw_token = True

    if not saw_token or matched_span != len(compact):
        if log_cb:
            log_cb("  parse_gap_string: could not parse %r; "
                   "treating as 0 (no absolute gap)." % (s,))
        return 0.0
    return total_days


def column_unit_factor_from_day(col_name):
    # type: (Optional[str]) -> float
    """Return the multiplier that converts a day-scale threshold into
    the units of the column named ``col_name``.

    Returns 1.0 for day-scale time columns (MJD/DMJD/JD), 86400.0 for
    second-scale time columns (TIME/TIMESTAMP/SECONDS/...), and 1.0
    for unknown/None inputs.  Comparison is case-insensitive and
    whitespace-trimmed.
    """
    if col_name is None:
        return 1.0
    try:
        key = str(col_name).strip().upper()
    except Exception:
        return 1.0
    if not key:
        return 1.0
    if key in DAY_SCALE_COLUMN_NAMES:
        return 1.0
    if key in SECOND_SCALE_COLUMN_NAMES:
        return 86400.0
    return 1.0


# %%% detect_time_breaks_unit_aware
def detect_time_breaks_unit_aware(x, col_name,
                                  k_factor=10.0,
                                  absolute_gap_days=0.0):
    # type: (np.ndarray, Optional[str], float, float) -> List[Tuple[float, float]]
    """Unit-aware wrapper around ``detect_time_breaks``.

    The user-facing threshold is always expressed in MJD-days (the GUI
    converts the "Manual gap (hours)" spinbox via ``hours / 24``).
    When the x-array is in a different unit (e.g. seconds-of-day for
    a TIME/TIMESTAMP column) the threshold must be scaled to that
    unit before ``np.diff(x)`` is compared against it.  This wrapper
    consults ``column_unit_factor_from_day(col_name)`` to do that
    conversion automatically.

    Parameters
    ----------
    x : np.ndarray
        1-D array of time samples in the column's native units.
    col_name : str or None
        Name of the column ``x`` was read from.  Used to look up
        the unit conversion factor.  Pass ``None`` if no name is
        available -- the threshold is then assumed to be already in
        the column's units.
    k_factor : float
        Forwarded unchanged to ``detect_time_breaks``.  The auto-
        threshold is ``k_factor * median(|diff(x)|)`` which is
        already in column units, so no conversion is needed.
    absolute_gap_days : float
        User-entered threshold in MJD-days.  Pass 0.0 (or negative)
        to disable the absolute threshold and rely on the K-factor
        auto-detection.

    Returns
    -------
    list of (start, end) pairs in the column's native units.
    """
    try:
        factor = column_unit_factor_from_day(col_name)
    except Exception:
        factor = 1.0
    try:
        ag_days = float(absolute_gap_days)
    except (TypeError, ValueError):
        ag_days = 0.0
    abs_gap_col = ag_days * factor if ag_days > 0.0 else 0.0
    return detect_time_breaks(
        x, k_factor=k_factor, absolute_gap=abs_gap_col,
    )


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
DEFAULT_DATETIME_TICK_FORMAT = "%VDY-%VDm-%VDdT%VDH:%VDM:%VDS"   # 0.3.0: ISO 8601
# 0.3.0: label of a local date-time axis (the offset is appended on save).
LOCAL_TIME_AXIS_LABEL = "Local time"
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
    # 0.3.0: default format -> local ISO-8601 axis resolved on save
    # (_autoplot_native.mark_local_time_axis); a caller's own format stays.
    if fmt == DEFAULT_DATETIME_TICK_FORMAT and hasattr(axis, "local_time"):
        try:
            from _autoplot_native import mark_local_time_axis
            # The x data are local date-times now, whatever column (MJD,
            # DMJD ...) they came from, so the label says so.
            mark_local_time_axis(axis, LOCAL_TIME_AXIS_LABEL)
            label = ""                     # set with the offset by finalize()
        except Exception:
            pass
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
# %% GPU ACCELERATION HELPERS (NVIDIA or AMD, optional)
# ============================================================================
#
# Design notes (0.4.1):
#   * The work is done by _autoplot_gpu: CuPy on CUDA (NVIDIA), CuPy on ROCm
#     (AMD, Linux) or PyOpenCL (any OpenCL GPU: AMD, NVIDIA, Intel; also AMD
#     on Windows), with NumPy as the fallback.  These wrappers keep the
#     0.0.10 API (is_gpu_available, enable_gpu, gpu_argsort ...) for the
#     FITS / Franks programs and their Veusz plugins.
#   * The GPU libraries are imported on first use, not when this module is
#     imported (0.4.0 imported CuPy here, which slowed the start of every
#     program and of every 'spawn' worker process).
#   * The GPU is only used when the input array is big enough to amortise
#     the host<->device transfer: 200 000 elements by default (public
#     CuPy/NumPy sort benchmarks), settable from the GUI.  Measured
#     2026-11-09: the largest sort of a 319-file OnePpsDeltas batch had
#     28 212 elements, so real GBT batches stay on the CPU.
#   * Only the argsort step is GPU-accelerated.  detect_time_breaks does
#     a tiny sort + diff that is overwhelmed by H2D transfer; we leave
#     it on the CPU.  Image-HDU dataset transfers are dominated by the
#     project write and gain nothing from GPU.

import _autoplot_gpu


def is_gpu_available() -> bool:
    """True when a GPU back end (CuPy CUDA/ROCm or PyOpenCL) found a device.

    The first call imports the GPU libraries; the answer is cached, so the
    GUI can poll this on every refresh."""
    return _autoplot_gpu.available()


def gpu_backend_name() -> str:
    """Short string describing the GPU back end (or why there is none) for the GUI/log."""
    return _autoplot_gpu.describe()


def enable_gpu(flag: bool) -> None:
    """Toggle the process-wide GPU acceleration flag.  Safe to call even
    when no GPU library is installed (it will simply have no effect)."""
    _autoplot_gpu.enable(flag)


def gpu_enabled() -> bool:
    """Return the current global GPU-enabled state."""
    return _autoplot_gpu.enabled()


def set_gpu_argsort_threshold(n: int) -> None:
    """Set the minimum array size (samples) at which the GPU sort is used.
    Smaller arrays fall through to NumPy."""
    try:
        _autoplot_gpu.set_threshold("argsort", int(n))
    except Exception:
        pass


def get_gpu_argsort_threshold() -> int:
    """Minimum array size (samples) for the GPU argsort path."""
    return _autoplot_gpu.threshold("argsort")


def gpu_tooltip(task: str = "sorts", operation: str = "argsort") -> str:
    """Tooltip of a GPU checkbox: the device found (or why none), the size
    threshold of ``operation`` and the install hints for NVIDIA and AMD."""
    return ("%s\n\nThe GPU is used only for %s of at least %d values; smaller ones "
            "are faster with NumPy on the CPU (the largest sort of a 319-file "
            "OnePpsDeltas batch had 28 212 values, so GBT files stay on the CPU). "
            "The results are identical.\nNVIDIA: CuPy (pip install cupy-cuda12x). "
            "AMD: CuPy for ROCm (Linux, cupy-rocm-7-0) or PyOpenCL (Windows and "
            "Linux, pip install pyopencl).\nEnvironment variable AUTOPLOT_GPU = "
            "auto, cuda, rocm, opencl or off." % (
                _autoplot_gpu.describe(), task, _autoplot_gpu.threshold(operation)))


def gpu_usage_text() -> str:
    """GPU and CPU calls of each GPU-capable operation since reset_gpu_usage()."""
    return _autoplot_gpu.usage_text()


def reset_gpu_usage() -> None:
    """Start counting GPU and CPU calls again (at the start of a batch)."""
    _autoplot_gpu.reset_usage()


def gpu_argsort(arr, force=None):
    """
    Return ``np.argsort(arr, kind="stable")`` (ascending, int64) with
    optional GPU acceleration (NVIDIA or AMD; see _autoplot_gpu).

    Parameters
    ----------
    arr : array_like
        1-D numeric array.
    force : None or bool
        * None (default) -- use GPU iff ``gpu_enabled()`` is True AND the
          array is large enough AND a GPU is available.
        * True -- attempt GPU regardless of size (still falls back to
          NumPy if anything fails).
        * False -- always use NumPy (useful for inner loops where the
          caller already decided).

    The function is type-stable: it always returns a NumPy ``int64``
    array, and the GPU order is identical to NumPy's (NaN last, ties in
    input order), so callers using ``arr[idx]`` work unchanged whether the
    GPU was used or not.  A GPU failure is recorded once (see
    gpu_backend_name()) and NumPy is used from then on.
    """
    if force is True:
        return _autoplot_gpu.argsort(arr, use_gpu=True, min_elements=0)
    if force is False:
        return _autoplot_gpu.argsort(arr, use_gpu=False)
    return _autoplot_gpu.argsort(arr, use_gpu=None)


# ============================================================================
# %% COMMON GUI BUILDING BLOCKS
# ---------------------------------------------------------------------------
class ProgressThrottle(object):
    """Rate limit for progress-bar updates (0.4.1).

    ``due(done, total)`` is True at most once every ``interval`` seconds, and
    always for the last step.  The page builders report every dataset; a
    progress bar repaint plus processEvents() per dataset cost 3 s of a
    15 s page build for the 319-file 1PPS set.
    """

    def __init__(self, interval: float = 0.05) -> None:
        """Updates no closer together than ``interval`` seconds."""
        self.interval = float(interval)
        self._last = 0.0

    def due(self, done: int, total: int, force_end: bool = True) -> bool:
        """True when the bar should be updated for ``done`` of ``total``.

        With ``force_end`` the last step (done >= total) is always due; the
        per-file column bars pass False, so a batch of short files does not
        repaint twice per file.
        """
        now = time.monotonic()
        if (force_end and done >= total) or now - self._last >= self.interval:
            self._last = now
            return True
        return False


# 0.4.0: queued signal connection, spelled so that PySide6, PyQt5 and PyQt6
# (through qtpy) all accept it.
QUEUED_CONNECTION = getattr(Qt, "QueuedConnection", None) or Qt.ConnectionType.QueuedConnection


class AutoPlotMainWindow(QMainWindow):
    """
    Base main window mirroring the layout used by Touchstone_AutoPlot.py:

        * top: File / View(theme) / Help text menu bar
        * centre: file list, processing options, log area
        * bottom: progress bar and Process / Save / Close buttons

    Subclasses provide ``_process_files`` and ``_save_project``.
    """

    # 0.4.0: log() called outside the GUI thread (the memory monitor) is
    # delivered here and appended by the GUI thread (queued connection).
    _log_from_thread = Signal(str)

    def __init__(self, title: str, default_mode: str = "dark") -> None:
        """Build the shared main window: menus, status bar, tabs and log.

        ``title`` is the window title; ``default_mode`` the theme used when the
        suite has no saved theme yet.  Subclasses add their options form through
        _populate_options() and the processing through _process_files().
        """
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1100, 800)
        self.selected_files: List[str] = []
        self._theme_mode = suite_theme(default_mode)   # shared with the launcher
        # Path of the most recently saved .vszh5 project (None until
        # mark_project_saved() is called by a subclass _save_project()).
        self._last_saved_path: Optional[str] = None
        self._log_from_thread.connect(self.log, QUEUED_CONNECTION)
        self._build_menu()
        self._build_central_widget()
        self._build_statusbar()
        apply_theme(QApplication.instance(), self._theme_mode)

    # ----- chrome -----------------------------------------------------------
    def _build_menu(self) -> None:
        """Create the File (open, clear, save, exit), View (dark / light mode) and
        Help (About) menus.
        """
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
        """Status bar with the permanent RSS (memory) read-out."""
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.mem_label = QLabel("RSS: -- MiB")
        sb.addPermanentWidget(self.mem_label)

    def _build_central_widget(self) -> None:
        """Lay out the Processing tab (file list, options, progress bars) and the
        Plots tab (pyqtgraph view), with the log below both tabs.
        """
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        # v0.2.0: "Processing" tab (everything below up to the log) and a
        # "Plots" tab with the pyqtgraph view; progress bars and the
        # Process / Save / Close buttons stay under the tabs.
        self.main_tabs = QTabWidget()
        outer.addWidget(self.main_tabs, 1)
        processing_tab = QWidget()
        self.main_tabs.addTab(processing_tab, "Processing")
        root = QVBoxLayout(processing_tab)

        # File selection
        fgroup = QGroupBox("Input File Selection")
        flay = QVBoxLayout(fgroup)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setToolTip("Files to process; add them with Browse, remove all with Clear.")
        self.file_list_widget.setMinimumHeight(180)
        flay.addWidget(self.file_list_widget)
        row = QHBoxLayout()
        self.browse_button = QPushButton("Browse Files...")
        self.browse_button.setToolTip("Add files to the list (several can be selected at once).")
        self.browse_button.clicked.connect(self._browse_files)
        row.addWidget(self.browse_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Remove every file from the list (the files are not touched).")
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
        self.status_text.setToolTip("Progress and messages of this session, with time stamps.")
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

        # v0.2.0: interactive pyqtgraph plots of the processed project
        self.plot_panel = None
        if autoplot_pg is not None:
            autoplot_pg.configure(antialias=True, wheel_needs_ctrl=True)
            self.plot_panel = autoplot_pg.ProjectPlotPanel(
                dark=(self._theme_mode == "dark"))
            self.main_tabs.addTab(self.plot_panel, "Plots")
        else:
            note = QLabel(
                "Interactive plots need pyqtgraph:\n\n    mamba install pyqtgraph"
                "\n\n(%s)\n\nProcessing and saving the Veusz project work "
                "without it." % PYQTGRAPH_ERROR)
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_tabs.addTab(note, "Plots")
        root = outer

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
        self.process_button.setToolTip(
            "Read the listed files and build the pages (shown on the Plots "
            "tab).")
        self.process_button.clicked.connect(self._process_files)
        bbox.addWidget(self.process_button)
        self.save_button = QPushButton("Save Veusz Project (.vszh5)")
        self.save_button.setToolTip(
            "Write the project as a native Veusz HDF5 file (.vszh5); Veusz "
            "is not needed.")
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
        self.close_button.setToolTip("Close this window (unsaved projects are lost).")
        self.close_button.clicked.connect(self.close)
        bbox.addWidget(self.close_button)
        root.addLayout(bbox)

    # subclasses override
    def _populate_options(self, form: QFormLayout) -> None:
        """Hook: subclasses add their processing options to ``form``."""
        pass

    # ----- theme handling ---------------------------------------------------
    def _set_theme(self, mode: str) -> None:
        """Switch to the 'dark' or 'light' theme, save it for the whole suite and
        update the View menu check marks and the Plots tab colours.
        """
        self._theme_mode = mode
        apply_theme(QApplication.instance(), mode)
        store_suite_theme(mode)
        self.act_dark.setChecked(mode == "dark")
        self.act_light.setChecked(mode == "light")
        if getattr(self, "plot_panel", None) is not None:
            self.plot_panel.apply_theme(mode == "dark")
        self.log("Theme set to %s mode" % mode)

    # ----- file list --------------------------------------------------------
    def _browse_files(self) -> None:
        """Ask for input files (filter ``self._file_filter``) and add the new ones
        to the list.
        """
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
        """Empty the input file list."""
        self.selected_files.clear()
        self._refresh_file_list()
        self.log("File list cleared.")

    def _refresh_file_list(self) -> None:
        """Show the base names of the selected files in the list widget."""
        self.file_list_widget.clear()
        for p in self.selected_files:
            self.file_list_widget.addItem(os.path.basename(p))

    # ----- log / status -----------------------------------------------------
    def log(self, msg: str) -> None:
        """Append ``msg`` with an HH:MM:SS time stamp to the log pane.

        Messages logged before the log widget exists (from _populate_options, for
        example) are buffered and written first once it does; widget errors are
        ignored so logging can never stop a batch.  Safe to call from any
        thread (0.4.0): outside the main (GUI) thread the text is queued to it.
        """
        if threading.current_thread() is not threading.main_thread():
            self._log_from_thread.emit(msg)
            return
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
        """Show ``rss_mb`` in the status bar memory read-out."""
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
        """Hide all three progress bars and their labels (batch finished)."""
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
        """Set the per-file column progress bar to ``done``."""
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
                "Install a separate Veusz viewer and put it on PATH, or set "
                "VEUSZ_EXECUTABLE. Veusz is not required to create files." % exc
            )
        except Exception as exc:
            self.log("Open in Veusz failed: %s" % exc)
            QMessageBox.critical(self, "Open in Veusz failed", str(exc))

    # ----- time gaps (0.4.0) -------------------------------------------------
    def add_gap_controls(self, form) -> None:
        """
        Gap controls shared by FITS and Franks AutoPlot (0.4.0).

        Small gap - the trace line breaks, the x axis stays continuous:
            dt > K x median(dt) of that trace, or a fixed time.
        Large gap - the x axis itself breaks (Veusz broken axis) where no
        trace has data for dt > n x (small-gap threshold), or a fixed time;
        n = 0 and no fixed time: no axis breaks.
        """
        heading = QLabel("<b>Gaps</b> - small: the line breaks; large: the x axis breaks")
        form.addRow(heading)
        self.gap_k_spin = QDoubleSpinBox()
        self.gap_k_spin.setRange(1.0, 1000.0)
        self.gap_k_spin.setSingleStep(1.0)
        self.gap_k_spin.setDecimals(2)
        self.gap_k_spin.setValue(10.0)
        self.gap_k_spin.setToolTip(
            "Small gap: the trace line stops and restarts where its sample "
            "step dt > K x median(dt) of that trace; the x axis is not cut.")
        form.addRow(QLabel("Small gap K (line break: dt > K × median dt):"), self.gap_k_spin)
        self.gap_abs_spin = QDoubleSpinBox()
        self.gap_abs_spin.setRange(0.0, 1e12)
        self.gap_abs_spin.setDecimals(6)
        self.gap_abs_spin.setSingleStep(1.0)
        self.gap_abs_spin.setValue(0.0)
        self.gap_abs_spin.setSpecialValueText("0 = use K × median dt")
        self.gap_abs_spin.setToolTip("A fixed small-gap time in hours instead of K x median(dt).")
        form.addRow(QLabel("Small gap fixed (hours):"), self.gap_abs_spin)
        self.gap_abs_str = QLineEdit()
        self.gap_abs_str.setPlaceholderText("e.g. 90s, 15min, 2h  (overrides the hours box)")
        self.gap_abs_str.setToolTip(GAP_STRING_HELP)
        form.addRow(QLabel("Small gap fixed (text):"), self.gap_abs_str)
        self.gap_axis_spin = QDoubleSpinBox()
        self.gap_axis_spin.setRange(0.0, 1.0e6)
        self.gap_axis_spin.setSingleStep(10.0)
        self.gap_axis_spin.setDecimals(1)
        self.gap_axis_spin.setValue(100.0)
        self.gap_axis_spin.setSpecialValueText("0 = no axis breaks")
        self.gap_axis_spin.setToolTip(
            "Large gap: the x axis itself is cut (a Veusz broken axis) where "
            "no trace on the graph has data for dt > n x the small-gap "
            "threshold (the largest of its traces).  0 = never.")
        form.addRow(QLabel("Large gap n (axis break: dt > n × small gap):"), self.gap_axis_spin)
        self.gap_axis_str = QLineEdit()
        self.gap_axis_str.setPlaceholderText("e.g. 12h, 3d, 1m  (overrides n)")
        self.gap_axis_str.setToolTip(GAP_STRING_HELP + "  Never shorter than the small gap.")
        form.addRow(QLabel("Large gap fixed (text):"), self.gap_axis_str)

    def gap_manual_days(self) -> float:
        """The fixed small gap in days (0 = K x median dt) for the page builders."""
        text = ""
        if getattr(self, "gap_abs_str", None) is not None:
            text = self.gap_abs_str.text().strip()
        if text:
            return parse_gap_string(text, log_cb=self.log)
        hours = float(self.gap_abs_spin.value()) if getattr(self, "gap_abs_spin", None) else 0.0
        return hours * _DAYS_PER_HOUR

    def gap_axis_days(self) -> float:
        """The fixed large gap in days (0 = n x small gap)."""
        text = ""
        if getattr(self, "gap_axis_str", None) is not None:
            text = self.gap_axis_str.text().strip()
        return parse_gap_string(text, log_cb=self.log) if text else 0.0

    def time_layout(self):
        """The TimeLayout (_autoplot_native) chosen in the gap controls."""
        from _autoplot_native import TimeLayout
        k = float(self.gap_k_spin.value()) if getattr(self, "gap_k_spin", None) else 10.0
        n = float(self.gap_axis_spin.value()) if getattr(self, "gap_axis_spin", None) else 0.0
        return TimeLayout(k=k, n=n, manual_s=self.gap_manual_days() * 86400.0,
                          axis_manual_s=self.gap_axis_days() * 86400.0)

    # ----- abstract hooks ---------------------------------------------------
    def _process_files(self) -> None:
        """Hook: subclasses run the batch here (File list -> project)."""
        raise NotImplementedError

    def _save_project(self) -> None:
        """Hook: subclasses write the .vszh5 project here."""
        raise NotImplementedError

    def show_plots(self, doc) -> int:
        """Show ``doc`` (the in-memory project) in the Plots tab.

        Called by the subclasses when a batch has finished.  Returns the
        number of plottable pages (0 when pyqtgraph is unavailable).
        """
        if getattr(self, "plot_panel", None) is None:
            return 0
        try:
            pages = self.plot_panel.set_document(doc)
        except Exception as exc:  # the project is still saved normally
            self.log("Plots tab: could not draw the project (%s)" % exc)
            return 0
        self.log("Plots tab: %d page(s) ready." % pages)
        if pages:
            self.main_tabs.setCurrentWidget(self.plot_panel)
        return pages

    def _show_about(self) -> None:
        """Help > About: tool, version, author, NRAO / GBO, licence, runtime."""
        import html as _html
        import platform as _platform
        esc = _html.escape
        name = getattr(self, "app_name", None) or self.windowTitle()
        version = getattr(self, "app_version", "")
        mail = "wwallace@nrao.edu"
        text = (
            "<h3>%s</h3><p>Version <b>%s</b> (Semantic Versioning 2.0.0)</p>"
            "<p>Batch processing of GBO timing data into native Veusz "
            ".vszh5 projects, with interactive pyqtgraph plots.</p>"
            "<table cellspacing='4'>"
            "<tr><td><b>Author</b></td><td>William W. Wallace</td></tr>"
            "<tr><td><b>Company</b></td><td>NRAO / Green Bank Observatory "
            "(GBO)</td></tr>"
            "<tr><td><b>Email</b></td><td><a href='mailto:%s'>%s</a></td></tr>"
            "<tr><td><b>Licence</b></td><td>GNU General Public License v3.0 or "
            "later (GPL-3.0-or-later)</td></tr>"
            "<tr><td><b>Copyright</b></td><td>Copyright (C) 2025-2026 William "
            "W. Wallace</td></tr></table>"
            "<p><small>The National Radio Astronomy Observatory and the Green "
            "Bank Observatory are facilities of the U.S. National Science "
            "Foundation operated under cooperative agreement by Associated "
            "Universities, Inc.</small></p>" % (esc(name), esc(version), mail, mail))
        try:
            import pyqtgraph as _pg
            pg_version = _pg.__version__
        except Exception:
            pg_version = "not installed (%s)" % PYQTGRAPH_ERROR
        details = "\n".join([
            "%s %s" % (name, version),
            "Python     : %s (%s)" % (_platform.python_version(), sys.executable),
            "numpy      : %s" % np.__version__,
            "pyqtgraph  : %s" % pg_version,
            "Plots tab  : %s" % (getattr(autoplot_pg, "VERSION", "-")
                                 if autoplot_pg is not None else "off"),
            "Writer     : _autoplot_native %s (Veusz 3.4 / 4.1 format)"
            % getattr(sys.modules.get("_autoplot_native"), "VERSION", "?"),
            "Platform   : %s" % _platform.platform(),
        ])
        box = QMessageBox(self)
        box.setWindowTitle("About %s" % name)
        box.setIcon(QMessageBox.Icon.Information)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(text)
        box.setInformativeText(
            "This program is free software: you can redistribute it and/or "
            "modify it under the terms of the GNU General Public License as "
            "published by the Free Software Foundation, either version 3 of "
            "the License, or (at your option) any later version. It comes "
            "with ABSOLUTELY NO WARRANTY.")
        box.setDetailedText(details)
        box.exec()


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
            """Integers of ``arr`` as text, zero-padded to ``w`` digits."""
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

    .. note::
        v0.0.15 compatibility shim.  New callers should use
        :func:`build_density_datestr_dataset` which takes an integer
        density percentage (0..100) instead of a fixed anchor count.
        This shim converts ``n_anchors`` -> equivalent density pct and
        delegates so both code paths share the same anchor-picking
        algorithm.

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
    # Translate fixed anchor count to a density pct using the trace's
    # finite-sample count.  We need to peek at the array length up
    # front, then delegate.
    try:
        arr = np.asarray(mjd_array, dtype=float)
    except Exception as exc:
        if log_cb:
            log_cb("    sparse-datestr build failed for %s: %s"
                   % (name, exc))
        return None
    n_finite = int(np.count_nonzero(np.isfinite(arr))) if arr.size else 0
    if n_finite <= 0:
        # Delegate so the helper logs the empty-input case consistently.
        return build_density_datestr_dataset(
            doc, name, mjd_array,
            density_pct=0, fmt=fmt, log_cb=log_cb,
        )
    k = max(1, int(n_anchors))
    if k >= n_finite:
        pct = 100
    else:
        pct = int(round((k / float(n_finite)) * 100.0))
        if pct < 1:
            pct = 1
    return build_density_datestr_dataset(
        doc, name, mjd_array,
        density_pct=pct, fmt=fmt, log_cb=log_cb,
    )


def build_full_datestr_dataset(doc, name, mjd_array,
                               fmt=DEFAULT_DATETIME_LABEL_FMT,
                               log_cb=None):
    """Push a same-length per-point date-string text dataset to Veusz.

    .. note::
        v0.0.15 compatibility shim.  Equivalent to
        :func:`build_density_datestr_dataset` with ``density_pct=100``;
        delegates to that helper so both code paths share the same
        anchor-picking algorithm.

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
    # v0.0.15 shim: a label at every point = density_pct=100.  Delegate
    # so all date-string emission shares one anchor-picking algorithm.
    return build_density_datestr_dataset(
        doc, name, mjd_array,
        density_pct=100, fmt=fmt, log_cb=log_cb,
    )


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


# ============================================================================
# %%% v0.0.15 DENSITY-PCT DATE-STRING DATASET + TEXT-X DT VARIANT
# ----------------------------------------------------------------------------
# Two new helpers layered on top of the v0.0.14 sparse/full builders:
#
#   * ``build_density_datestr_dataset`` -- generalises the v0.0.14 sparse
#     builder.  Instead of a fixed n_anchors count, it accepts an integer
#     ``density_pct`` in [0, 100].  Anchor count = ``round(pct/100 * N)``
#     where N is the number of FINITE samples in the trace.  pct=0 -> no
#     anchors (empty-string labels everywhere -> nothing renders),
#     pct=100 -> a label at every finite index.  Output dataset always has
#     the same length as the input MJD array so it stays aligned with the
#     trace's x/y datasets row-for-row.
#
#   * ``build_textx_dataset`` -- pushes a same-length per-point text
#     dataset whose values are the formatted date strings (no "" gaps,
#     because this dataset is going to serve as the xy widget's xData).
#     Non-finite MJDs are emitted as ``"NaN"`` so length is preserved;
#     Veusz then plots no marker at that index because the y dataset is
#     NaN there.
#
#   * ``configure_axis_labels_mode`` -- sets the dt-page x axis to
#     ``mode='labels'``, applies rotation/size, and caps
#     ``MajorTicks.number`` to a sane subset of N so dense traces do not
#     spam a tick at every integer position.  Cap is derived from the
#     same density pct so a single GUI knob controls both dt variants.
# ============================================================================


def build_density_datestr_dataset(doc, name, mjd_array,
                                  density_pct=10,
                                  fmt=DEFAULT_DATETIME_LABEL_FMT,
                                  log_cb=None):
    """Push a same-length date-string text dataset with continuous density.

    The returned dataset has the same length as ``mjd_array``; values
    are the empty string ``""`` everywhere except at
    ``round(density_pct/100 * Nfinite)`` evenly spaced FINITE indices,
    where the value is a ``YYYY-MM-DD HH:MM:SS`` formatted string
    converted from the corresponding MJD via :func:`mjd_to_datestr`.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        Active embedded Veusz document.
    name : str
        Veusz dataset name to create.
    mjd_array : array-like of float
        Modified Julian Date values, same length as the trace's y data.
    density_pct : int, optional
        Integer percentage in [0, 100].  Default 10.
        Anchor count = ``round(density_pct/100 * Nfinite)``.
        Values are clipped: <0 -> 0 (no labels), >100 -> 100 (every
        finite point).
    fmt : str, optional
        ``strftime`` format string.  Default ``"%Y-%m-%d %H:%M:%S"``.
    log_cb : callable, optional
        ``log_cb(str)`` for human-readable progress messages.

    Returns
    -------
    str or None
        ``name`` on success, ``None`` on any failure (empty input,
        SetDataText raised, etc.).
    """
    try:
        arr = np.asarray(mjd_array, dtype=float)
    except Exception as exc:
        if log_cb:
            log_cb("    density-datestr build failed for %s: %s"
                   % (name, exc))
        return None
    n = int(arr.size)
    if n == 0:
        if log_cb:
            log_cb("    density-datestr skipped for %s: empty input" % name)
        return None
    # Clip the density percentage to [0, 100].
    try:
        pct = int(round(float(density_pct)))
    except Exception:
        pct = 10
    if pct < 0:
        pct = 0
    if pct > 100:
        pct = 100
    # Empty-density case: push an all-"" text dataset (so xy.labels.val
    # can still bind to a real text dataset, just with nothing visible).
    if pct == 0:
        try:
            doc.SetDataText(name, [""] * n)
        except Exception as exc:
            if log_cb:
                log_cb("    SetDataText failed for %s: %s" % (name, exc))
            return None
        if log_cb:
            log_cb("    +density-datestr labels %s (0%% -> no anchors, "
                   "%d points)" % (name, n))
        return name
    finite_mask = np.isfinite(arr)
    finite_idx = np.flatnonzero(finite_mask)
    if finite_idx.size == 0:
        if log_cb:
            log_cb("    density-datestr skipped for %s: no finite MJDs"
                   % name)
        return None
    # Anchor count: round(pct/100 * Nfinite), clamped to [1, Nfinite].
    k = int(round((pct / 100.0) * float(finite_idx.size)))
    if k < 1:
        k = 1
    if k > int(finite_idx.size):
        k = int(finite_idx.size)
    if k == 1:
        anchor_positions = np.array([finite_idx[finite_idx.size // 2]],
                                    dtype=np.int64)
    elif k == int(finite_idx.size):
        # Every finite sample is an anchor: avoid the linspace round-trip.
        anchor_positions = finite_idx
    else:
        sel = np.linspace(0, finite_idx.size - 1, k).round().astype(np.int64)
        sel = np.unique(sel)
        anchor_positions = finite_idx[sel]
    try:
        anchor_strs = mjd_to_datestr(arr[anchor_positions], fmt=fmt)
    except Exception as exc:
        if log_cb:
            log_cb("    density-datestr conversion failed for %s: %s"
                   % (name, exc))
        return None
    labels = [""] * n
    for pos, s in zip(anchor_positions.tolist(),
                      np.asarray(anchor_strs).tolist()):
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
        log_cb("    +density-datestr labels %s (%d%% -> %d anchors / "
               "%d points)" % (name, pct, int(anchor_positions.size), n))
    return name


def build_textx_dataset(doc, name, mjd_array,
                        fmt=DEFAULT_DATETIME_LABEL_FMT,
                        log_cb=None):
    """Push a same-length per-point date-string text dataset for use as xData.

    Unlike :func:`build_density_datestr_dataset`, every index gets a
    string (no ``""`` gaps), because this dataset will serve as the x
    data of an xy widget.  Non-finite MJDs are stored as the sentinel
    ``"NaN"`` so the dataset length stays consistent with the y
    dataset; Veusz draws no marker at that index because the y dataset
    is NaN there.

    When the xy widget's ``xData`` setting points at this dataset and
    the parent x axis is set to ``mode='labels'``, Veusz queries
    ``xy.getAxisLabels('x')`` which returns
    ``(strings, np.arange(N))``.  The axis then plots samples at
    integer positions 0..N-1 with each tick labeled by the
    corresponding date string.  Sample spacing is UNIFORM (not
    proportional to elapsed time), so this is the right helper for the
    new dt_labels page variant only.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        Active embedded Veusz document.
    name : str
        Veusz dataset name to create.
    mjd_array : array-like of float
        Modified Julian Date values, same length as the trace's y data.
    fmt : str, optional
        ``strftime`` format string.  Default ``"%Y-%m-%d %H:%M:%S"``.
    log_cb : callable, optional
        ``log_cb(str)`` for progress messages.

    Returns
    -------
    str or None
        ``name`` on success, ``None`` on any failure.
    """
    try:
        arr = np.asarray(mjd_array, dtype=float)
    except Exception as exc:
        if log_cb:
            log_cb("    textx-datestr build failed for %s: %s"
                   % (name, exc))
        return None
    n = int(arr.size)
    if n == 0:
        if log_cb:
            log_cb("    textx-datestr skipped for %s: empty input" % name)
        return None
    try:
        strs = mjd_to_datestr(arr, fmt=fmt)
    except Exception as exc:
        if log_cb:
            log_cb("    textx-datestr conversion failed for %s: %s"
                   % (name, exc))
        return None
    # Keep "NaN" sentinel here (not "") because this serves as xData --
    # an empty string at index i would make Veusz draw a blank tick at
    # integer position i but the y point at the same index would still
    # render, producing a confusing unlabeled marker.  "NaN" is more
    # informative and matches the y-side NaN convention.
    labels = [str(s) if s is not None else "NaN"
              for s in np.asarray(strs).tolist()]
    try:
        doc.SetDataText(name, labels)
    except Exception as exc:
        if log_cb:
            log_cb("    SetDataText failed for %s: %s" % (name, exc))
        return None
    if log_cb:
        log_cb("    +textx-datestr labels %s (%d labels)" % (name, n))
    return name


def configure_axis_labels_mode(axis_widget,
                               angle=DEFAULT_DATETIME_LABEL_ANGLE,
                               size=DEFAULT_DATETIME_LABEL_SIZE,
                               max_ticks=None,
                               total_points=None,
                               density_pct=None):
    """Switch a Veusz axis to ``mode='labels'`` with sensible defaults.

    Bind the parent xy widget's ``xData`` to a text dataset
    (e.g. one produced by :func:`build_textx_dataset`) BEFORE calling
    this -- Veusz pulls tick label strings from the plotter's text
    xData via ``xy.getAxisLabels``.

    Parameters
    ----------
    axis_widget : Veusz axis widget
        The dt-page x axis.
    angle : float, optional
        Tick label rotation in degrees (default 45).
    size : str, optional
        Tick label font size (default ``'6pt'``).
    max_ticks : int or None, optional
        Hard cap on ``MajorTicks.number``.  If None, derive from
        ``density_pct`` and ``total_points``:
        ``cap = max(2, round(density_pct/100 * total_points))``,
        clamped to [2, 20] for readability.
    total_points : int or None, optional
        Length of the trace; used with ``density_pct`` to derive a tick
        cap when ``max_ticks`` is None.
    density_pct : int or None, optional
        Same density slider that drives label density on the numeric dt
        page; reused here so a single knob controls both variants.

    Returns
    -------
    The axis widget (for chaining).  All attribute writes are guarded.
    """
    if axis_widget is None:
        return axis_widget
    try:
        axis_widget.mode.val = "labels"
    except Exception:
        pass
    # Tick label appearance.
    try:
        axis_widget.TickLabels.rotate.val = float(angle)
    except Exception:
        try:
            axis_widget.TickLabels.rotate.val = int(round(float(angle)))
        except Exception:
            pass
    try:
        axis_widget.TickLabels.size.val = str(size)
    except Exception:
        pass
    # Derive a sensible MajorTicks.number cap.
    cap = None
    if max_ticks is not None:
        try:
            cap = int(max_ticks)
        except Exception:
            cap = None
    elif density_pct is not None and total_points is not None:
        try:
            pct = int(round(float(density_pct)))
            n = int(total_points)
            cap = int(round((pct / 100.0) * n))
        except Exception:
            cap = None
    if cap is not None:
        if cap < 2:
            cap = 2
        if cap > 50:
            # Hard upper bound so an over-dense slider doesn't paint a
            # tick at every integer index on a 10k-point trace.
            cap = 50
        try:
            axis_widget.MajorTicks.number.val = int(cap)
        except Exception:
            pass
        try:
            # Minor ticks are mostly noise in labels mode.
            axis_widget.MinorTicks.number.val = 0
        except Exception:
            pass
    return axis_widget


# ============================================================================
# %%% v0.0.16 datetime-mode helpers for the dt_labels page
# ----------------------------------------------------------------------------
# build_dtnum_dataset / configure_axis_datetime_mode / mjd_break_pairs_to_dtsec
#
# These replace the v0.0.15 text-x / labels-mode approach for the dt_labels
# page.  Instead of binding a per-point TEXT dataset to xData and setting the
# axis to mode='labels', v0.0.16 binds a NUMERIC Veusz-datetime-seconds
# dataset to xData and sets the axis to mode='datetime'.  Sample spacing is
# then proportional to elapsed time, and the axis is responsible for tick
# formatting via Veusz's DateTicks class.
#
# We deliberately do NOT call ``SetDataDateTime`` here because the v0.0.13
# audit logged that ``DatasetDateTime`` round-trips fire hundreds of
# internal exceptions on Veusz 3.4; the plain-numeric / axis-side approach
# is just as date-aware on the axis and works identically on Veusz 3.4
# and 4.1.
# ============================================================================

def build_dtnum_dataset(doc, name, mjd_array, log_cb=None):
    """Push a numeric Veusz-datetime-seconds dataset for use as xData.

    Builds a plain ``SetData`` numeric array whose values are seconds since
    2009-01-01 UTC (Veusz's internal datetime numeric format).  Non-finite
    MJD inputs become NaN seconds so the dataset stays length-aligned with
    the companion y dataset.

    Bind this dataset to ``xy.xData.val`` and then call
    :func:`configure_axis_datetime_mode` on the parent x axis -- the axis
    will then render proper date ticks with elapsed-time-proportional
    spacing.

    Parameters
    ----------
    doc : veusz.embed.Embedded
        Active embedded Veusz document.
    name : str
        Veusz dataset name to create.
    mjd_array : array-like of float
        Modified Julian Date values, same length as the trace's y data.
    log_cb : callable, optional
        ``log_cb(str)`` for progress messages.

    Returns
    -------
    str or None
        ``name`` on success, ``None`` on any failure.
    """
    try:
        arr = np.asarray(mjd_array, dtype=float)
    except Exception as exc:
        if log_cb:
            log_cb("    dtnum build failed for %s: %s" % (name, exc))
        return None
    n = int(arr.size)
    if n == 0:
        if log_cb:
            log_cb("    dtnum skipped for %s: empty input" % name)
        return None
    try:
        secs = mjd_to_veusz_seconds(arr)
    except Exception as exc:
        if log_cb:
            log_cb("    dtnum mjd_to_veusz_seconds failed for %s: %s"
                   % (name, exc))
        return None
    try:
        # 0.3.0: a real Veusz date-time dataset (UTC) when the document
        # supports it, plus its local display copy "<name>_local".
        if hasattr(doc, "SetDataDateTime"):
            doc.SetDataDateTime(name, secs)
        else:
            doc.SetData(name, secs)
        local = None
        if hasattr(doc, "SetDataLocalTime"):
            local = doc.SetDataLocalTime(name)
    except Exception as exc:
        if log_cb:
            log_cb("    SetData failed for %s: %s" % (name, exc))
        return None
    if log_cb:
        log_cb("    +date-time %s (%d points, UTC)%s" % (
            name, n, (" and %s (local display)" % local) if local else ""))
    return name


def configure_axis_datetime_mode(axis_widget,
                                 fmt=DEFAULT_DATETIME_TICK_FORMAT,
                                 rotate_deg=DEFAULT_DATETIME_TICK_ROTATE_DEG,
                                 major_ticks_target=
                                     DEFAULT_DATETIME_MAJOR_TICKS_TARGET,
                                 label=""):
    """Switch a Veusz x axis to ``mode='datetime'`` and style its ticks.

    Use after the parent xy widget's ``xData`` has been bound to a
    numeric Veusz-datetime-seconds dataset (e.g. one produced by
    :func:`build_dtnum_dataset`).  Veusz then uses ``axisticks.DateTicks``
    for tick placement and ``TickLabels.format`` for tick formatting.

    Works on both plain ``axis`` and ``axis-broken`` widgets because
    ``AxisBroken`` is a subclass of ``axis.Axis`` and therefore inherits
    the ``mode`` setting and the TickLabels / MajorTicks groups
    unchanged.  Compatible with both Veusz 3.4 and 4.1.

    Parameters
    ----------
    axis_widget : Veusz axis widget
        The dt_labels-page x axis (plain ``x`` or ``axis-broken``).
    fmt : str, optional
        Veusz date-format string for tick labels.
    rotate_deg : float, optional
        Tick label rotation in degrees.
    major_ticks_target : int, optional
        Hint for ``MajorTicks.number``.
    label : str, optional
        Axis label to set if non-empty.

    Returns
    -------
    The axis widget (for chaining).  All attribute writes are guarded.
    """
    if axis_widget is None:
        return axis_widget
    try:
        axis_widget.mode.val = "datetime"
    except Exception:
        # Older Veusz builds (unknown) might use a different setting name;
        # fall through and let style_datetime_x_axis still apply.
        pass
    return style_datetime_x_axis(
        axis_widget,
        rotate_deg=rotate_deg,
        fmt=fmt,
        major_ticks_target=major_ticks_target,
        label=label,
    )


def mjd_break_pairs_to_dtsec(pairs, local=True):
    """Convert MJD (start, end) gap pairs to Veusz datetime-seconds pairs.

    The output is suitable to drive an ``axis-broken`` widget on a
    dt_labels (mode='datetime') page whose xData is in Veusz datetime
    seconds.  The pair shrink-by-epsilon already applied by
    :func:`detect_time_breaks` is preserved because the conversion is
    monotonic-affine (multiply by 86400, then translate).

    Parameters
    ----------
    pairs : list of (float, float)
        MJD gap pairs as returned by :func:`detect_time_breaks`.

    Returns
    -------
    list of (float, float)
        Same pairs in Veusz datetime seconds.  Empty list if input is
        empty.
    """
    if not pairs:
        return []
    out = []
    for s, e in pairs:
        try:
            s2 = float(
                (float(s) - MJD_VEUSZ_EPOCH_MJD) * 86400.0
            )
            e2 = float(
                (float(e) - MJD_VEUSZ_EPOCH_MJD) * 86400.0
            )
            if e2 > s2:
                out.append((s2, e2))
        except Exception:
            continue
    if out and local:
        # 0.3.0: the dt_labels axes show local time; shift every gap edge
        # by its own (DST-aware) offset, as the local datasets are.
        try:
            import vszh5_writer
            flat = np.array(out, dtype=float).ravel()
            shifted = vszh5_writer.local_time_from_veusz(flat).veusz_seconds
            out = [(float(a), float(b)) for a, b in shifted.reshape(-1, 2) if b > a]
        except Exception:
            pass
    return out
