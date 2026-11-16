# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
_autoplot_native.py

Native AutoPlot project model: a small, veusz.embed-like document (datasets,
widget tree, settings) that the AutoPlot page builders fill and that is
written as a Veusz HDF5 project (.vszh5) by vszh5_writer.  No Qt or Veusz
imports.

The public SetData/Root.Add surface is deliberately small, so existing
AutoPlot page builders can target this model without a running Veusz process.
This is not a general-purpose replacement for the Veusz scripting API.

On save, finalize() names auto-named widgets after their data and lays out
every time axis: MJD/DMJD and Veusz date datasets become local ISO 8601
date-time axes, a trace's line breaks at a small gap (dt > K x median dt)
and the x axis breaks at a large gap (dt > n x the small-gap threshold) -
see TimeLayout.

Created on 2026-10-11

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.5.4: Clear gives memory back to the OS; data kept between batches.
Date: 2026-11-16
#              * NEW release_memory(log): three garbage-collection passes, the
#                GPU pools (_autoplot_gpu.release_memory, only when that
#                module is loaded), then the C heap back to the operating
#                system (_release_os_heap: glibc malloc_trim on Linux, which
#                frees whole free pages in every arena; on Windows _heapmin,
#                HeapCompact of every process heap and
#                SetProcessWorkingSetSize(-1, -1); macOS
#                malloc_zone_pressure_relief); logs the RSS before and after.
#                Measured: 400 MB of freed 80 kB arrays stayed resident
#                (1202 MiB) until malloc_trim returned them (43 MiB).
#              * NEW process_rss_mb(): psutil, /proc/self/statm or
#                GetProcessMemoryInfo.
#              * NEW KeptReads and file_stamp(): data read from the listed
#                files, kept until Clear; to_read() names only new files,
#                files changed on disk (size, mtime) or read with other
#                options; a kept file gone from disk is not forgotten.
#              * NEW NativeDocument.release(): empties the project at once
#                (the widget tree's document / parent links are cut).
#              Checked: tests/test_clear_and_add.py (all at once == first
#              half, then the second added; Clear leaves no project, plots,
#              kept data or cache; the cleared files are not in the next
#              project) on Python 3.8, 3.12 and 3.13.
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        local_veusz_seconds: Local wall-clock Veusz date seconds of the UTC Veusz date seconds ``utc_veusz``, with the per-sample offsets of the LocalTimeAxis ``axis`` (0.5.2):...
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        parse_gap_string: Parse a human-readable gap string into MJD-days.
        add_gap_arguments: Add the gap options to an argparse parser (0.5.3): --gap-k, --gap-small, --gap-n, --gap-large, --no-line-breaks, --no-axis-breaks, --no-numeric-gap...
        layout_from_args: The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3).
        TimeLayout (class): Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any sorted x, and the two consequences of a gap made optional), see vszh5_writer T...
        _number_text: A setting for the notes with every digit (repr), without a trailing ".0" (0.5.3; %g cut 12.3456789 to 12.3457).
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0).
        _set_gap_note: Put the gap-rule line ``note`` into the graph notes (0.5.3: a line from an earlier layout is replaced, so the notes state the rule in use).
        _sorted_finite: True when the finite values of ``values`` never decrease (at least 3).
        _layout_numeric_axis: Gap options on an x axis that is not time (0.5.3): frequency, angle, index ...
        _undo_numeric_layout: Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the gap options of a project turn non-time axes off between two saves, the traces d...
        _drop_unused_plot_copies: Remove line-break plot copies no widget draws any more (0.5.3): when the gap options change between two saves, copies of the old layout would other...
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        apply_settings: Set ``{setting path: value}`` on a widget (0.4.0).
        add_smith_graph: Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent`` (a page or grid widget); returns the graph widget (0.4.0).
        add_smith_page: A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).
        _data_name: The dataset a plotting widget draws (y for xy), or ''.
        _rename_from_data: Name an auto-named widget after its data (unique among siblings).
        tag_word: ``text`` as a tag word: letters, digits and '_' only (never empty).
        source_tag: Veusz tag naming where data come from: ``source_<word>``.
        type_tag: Veusz tag naming what data are: ``type_<word>``.
        tag_data: Tag the datasets ``names`` with ``source_*`` and ``type_*`` tags.
        _safe_name: ``name`` with letters, digits and '_' only (any other character becomes '_'), so a widget name reads like the dataset it draws; a leading digit sta...
        trace_widget_name: Name of the xy widget that draws ``y_dataset``: the dataset name (it holds the source file, the item and the processing type), plus ``_<variant>``...
        exact_number_text: A number as text with every digit: repr() of a float (the shortest text that gives back the same float64), str() of an integer.
        float_loses_text: True when float64 ``values`` parsed from the cell ``texts`` do not hold all of them (0.5.2): a cell that is not a number and not a missing-value wo...
        process_rss_mb: Resident set size of this process in MiB (NaN when it cannot be read).
        _release_os_heap: Hand freed heap memory back to the operating system; returns the steps that ran (names), [] when nothing applies.
        release_memory: Give the memory of dropped data back to the operating system (0.5.4).
        file_stamp: (size, mtime_ns, options) of ``path``: the data read from it are still good while this is unchanged; None when the file cannot be read.
        KeptReads (class): Data read from the listed files, kept until the user clears them (0.5.4).
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        _GAP_TOKEN_REGEX: one number and unit of a gap text (s, min, h, d, m; bare = hours)
        GAP_STRING_HELP: tool tip of the gap text boxes
        _DAYS_PER_MONTH: average Gregorian month (365.25 / 12 days) for 'm' in gap texts
        _DAYS_PER_HOUR: 1 / 24 (gap text parsing)
        _DAYS_PER_MINUTE: 1 / 1440 (gap text parsing)
        _DAYS_PER_SECOND: 1 / 86400 (gap text parsing)
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
        _TAG_WORD: characters not allowed in a tag word (replaced by '_')
        MISSING_VALUE_WORDS: cell texts that mean no value (pandas' NA strings): no exact-text copy needed
        NUMERIC_TICK_FORMAT: tick-label format of numeric axes: up to 12 significant digits
# %%%%% More Info
    Imported by every AutoPlot program and plugin; needs numpy, h5py and
    vszh5_writer only (no Qt, no Veusz).
    Tests: python -m unittest discover -s tests
    Gaps (every plot generator): a small gap (step > K x median step, or a
    fixed time) can break the line / trace; a large gap (no trace has data for
    > n x the small gap, or a fixed time) can break the x axis; each is a check
    box (GUI) or flag (--no-line-breaks, --no-axis-breaks), and "also on x axes
    that are not time" (--no-numeric-gaps to turn off) applies the rule to sorted
    frequency / angle / index axes with a log-scale test, so a log sweep is not
    cut.  Rule: vszh5_writer 0.0.30 GapRule; notes state the rule in use.
    TimeLayout holds the options; finalize() lays out time axes
    (_layout_time_axis), then other x axes (_layout_numeric_axis), then drops
    plot copies no widget draws (_drop_unused_plot_copies).
    Memory: release_memory() after
    the last reference to dropped data is gone (gc, GPU pools, malloc_trim /
    HeapCompact + working-set trim); KeptReads keeps per-file data until Clear.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.3: Gap options: line breaks and x-axis breaks each optional, on any sorted x.
Date: 2026-11-16
#              * TimeLayout gains break_lines, break_axis and numeric (all
#                True by default, so 0.5.2 projects are unchanged):
#                break_lines=False draws every line across its gaps (no NaN
#                rows, no "_gaps" copies); break_axis=False keeps one
#                continuous x axis; numeric=False limits the rule to time
#                axes.  gap_rule() gives the vszh5_writer 0.0.30 GapRule;
#                describe() states the options with every digit.
#              * NEW _layout_numeric_axis(): x axes that are not time
#                (frequency, angle, index) get the same rule on sorted 1-D x
#                data, with the geometric test of the writer (the step must
#                also be large on a log scale), so a log-spaced sweep is not
#                cut; unsorted traces (loops, polar), label and hidden axes
#                (Smith charts) are left alone.  _undo_numeric_layout()
#                restores the traces when the option is turned off.
#              * NEW _drop_unused_plot_copies(): gap / local copies of an
#                earlier save that no widget draws any more are removed with
#                their tags; _set_gap_note() replaces the "Gaps:" line of the
#                graph notes instead of adding a second one.
#              * NEW add_gap_arguments() / layout_from_args(): --gap-k,
#                --gap-small, --gap-n, --gap-large, --no-line-breaks,
#                --no-axis-breaks, --no-numeric-gaps for the batch modes.
#              * parse_gap_string, GAP_STRING_HELP and the _DAYS_PER_*
#                constants moved here from _autoplot_common (Qt-free; the
#                common module re-exports them).
#              * local_veusz_seconds() calls the writer's
#                local_veusz_exact() (same UTC + offset in one addition).
#              Checked: tests/test_gap_options.py and test_writer_0030.py
#              on Python 3.8, 3.12 and 3.13 (Veusz 4.2.1 render); with the
#              default options the datasets of every example project are
#              identical to 0.4.2.
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        local_veusz_seconds: Local wall-clock Veusz date seconds of the UTC Veusz date seconds ``utc_veusz``, with the per-sample offsets of the LocalTimeAxis ``axis`` (0.5.2):...
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        parse_gap_string: Parse a human-readable gap string into MJD-days.
        add_gap_arguments: Add the gap options to an argparse parser (0.5.3): --gap-k, --gap-small, --gap-n, --gap-large, --no-line-breaks, --no-axis-breaks, --no-numeric-gap...
        layout_from_args: The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3).
        TimeLayout (class): Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any sorted x, and the two consequences of a gap made optional), see vszh5_writer T...
        _number_text: A setting for the notes with every digit (repr), without a trailing ".0" (0.5.3; %g cut 12.3456789 to 12.3457).
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0).
        _set_gap_note: Put the gap-rule line ``note`` into the graph notes (0.5.3: a line from an earlier layout is replaced, so the notes state the rule in use).
        _sorted_finite: True when the finite values of ``values`` never decrease (at least 3).
        _layout_numeric_axis: Gap options on an x axis that is not time (0.5.3): frequency, angle, index ...
        _undo_numeric_layout: Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the gap options of a project turn non-time axes off between two saves, the traces d...
        _drop_unused_plot_copies: Remove line-break plot copies no widget draws any more (0.5.3): when the gap options change between two saves, copies of the old layout would other...
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        apply_settings: Set ``{setting path: value}`` on a widget (0.4.0).
        add_smith_graph: Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent`` (a page or grid widget); returns the graph widget (0.4.0).
        add_smith_page: A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).
        _data_name: The dataset a plotting widget draws (y for xy), or ''.
        _rename_from_data: Name an auto-named widget after its data (unique among siblings).
        tag_word: ``text`` as a tag word: letters, digits and '_' only (never empty).
        source_tag: Veusz tag naming where data come from: ``source_<word>``.
        type_tag: Veusz tag naming what data are: ``type_<word>``.
        tag_data: Tag the datasets ``names`` with ``source_*`` and ``type_*`` tags.
        _safe_name: ``name`` with letters, digits and '_' only (any other character becomes '_'), so a widget name reads like the dataset it draws; a leading digit sta...
        trace_widget_name: Name of the xy widget that draws ``y_dataset``: the dataset name (it holds the source file, the item and the processing type), plus ``_<variant>``...
        exact_number_text: A number as text with every digit: repr() of a float (the shortest text that gives back the same float64), str() of an integer.
        float_loses_text: True when float64 ``values`` parsed from the cell ``texts`` do not hold all of them (0.5.2): a cell that is not a number and not a missing-value wo...
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        _GAP_TOKEN_REGEX: one number and unit of a gap text (s, min, h, d, m; bare = hours)
        GAP_STRING_HELP: tool tip of the gap text boxes
        _DAYS_PER_MONTH: average Gregorian month (365.25 / 12 days) for 'm' in gap texts
        _DAYS_PER_HOUR: 1 / 24 (gap text parsing)
        _DAYS_PER_MINUTE: 1 / 1440 (gap text parsing)
        _DAYS_PER_SECOND: 1 / 86400 (gap text parsing)
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
        _TAG_WORD: characters not allowed in a tag word (replaced by '_')
        MISSING_VALUE_WORDS: cell texts that mean no value (pandas' NA strings): no exact-text copy needed
        NUMERIC_TICK_FORMAT: tick-label format of numeric axes: up to 12 significant digits
# %%%%% More Info
    Imported by every AutoPlot program and plugin; needs numpy, h5py and
    vszh5_writer only (no Qt, no Veusz).
    Tests: python -m unittest discover -s tests
    Gaps (every plot generator): a small gap (step > K x median step, or a
    fixed time) can break the line / trace; a large gap (no trace has data for
    > n x the small gap, or a fixed time) can break the x axis; each is a check
    box (GUI) or flag (--no-line-breaks, --no-axis-breaks), and "also on x axes
    that are not time" (--no-numeric-gaps to turn off) applies the rule to sorted
    frequency / angle / index axes with a log-scale test, so a log sweep is not
    cut.  Rule: vszh5_writer 0.0.30 GapRule; notes state the rule in use.
    TimeLayout holds the options; finalize() lays out time axes
    (_layout_time_axis), then other x axes (_layout_numeric_axis), then drops
    plot copies no widget draws (_drop_unused_plot_copies).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.2: source_/type_ tags on every dataset, data-named traces, exact numbers.
Date: 2026-11-10
#              * NEW tag_word, source_tag, type_tag, tag_data(): the Veusz
#                tags source_<origin> and type_<kind> used by every program
#                (no Qt here, so batch modes and reader processes stay
#                light); trace_widget_name(): a trace widget is named after
#                the dataset it draws; exact_number_text(): repr() of a
#                float, str() of an integer; float_loses_text() with
#                MISSING_VALUE_WORDS: can float64 hold a column's text?
#              * finalize(): local and gap copies inherit the source_ and
#                type_ tags of their data (plus type_time_local /
#                type_gap_rows); a dataset still untagged gets
#                source_<program> and a type_ of its kind; numeric axes
#                without a format get "%.12Vg" ticks (NUMERIC_TICK_FORMAT;
#                Veusz's %Vg showed MJD 61234.1 as 6.1x10^4).
#              * NEW GetDataType(name): '1d', '2d', 'text' or 'datetime'.
#              * Local-time copies are UTC + offset in one addition
#                (local_veusz_seconds); the writer's round trip through Unix
#                epoch seconds rounded fractions of a second to about
#                0.24 us (tack / Temps: at most 1.2e-7 s; whole seconds
#                unchanged).
#              * _seconds_text(): the exact seconds in brackets when the
#                short text is rounded.
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        local_veusz_seconds: Local wall-clock Veusz date seconds of the UTC Veusz date seconds ``utc_veusz``, with the per-sample offsets of the LocalTimeAxis ``axis`` (0.5.2):...
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        TimeLayout (class): Gap rule of the time axes (0.5.0), see vszh5_writer TIME GAPS.
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0).
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        apply_settings: Set ``{setting path: value}`` on a widget (0.4.0).
        add_smith_graph: Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent`` (a page or grid widget); returns the graph widget (0.4.0).
        add_smith_page: A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).
        _data_name: The dataset a plotting widget draws (y for xy), or ''.
        _rename_from_data: Name an auto-named widget after its data (unique among siblings).
        tag_word: ``text`` as a tag word: letters, digits and '_' only (never empty).
        source_tag: Veusz tag naming where data come from: ``source_<word>``.
        type_tag: Veusz tag naming what data are: ``type_<word>``.
        tag_data: Tag the datasets ``names`` with ``source_*`` and ``type_*`` tags.
        _safe_name: ``name`` with letters, digits and '_' only (any other character becomes '_'), so a widget name reads like the dataset it draws; a leading digit sta...
        trace_widget_name: Name of the xy widget that draws ``y_dataset``: the dataset name (it holds the source file, the item and the processing type), plus ``_<variant>``...
        exact_number_text: A number as text with every digit: repr() of a float (the shortest text that gives back the same float64), str() of an integer.
        float_loses_text: True when float64 ``values`` parsed from the cell ``texts`` do not hold all of them (0.5.2): a cell that is not a number and not a missing-value wo...
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
        _TAG_WORD: characters not allowed in a tag word (replaced by '_')
        MISSING_VALUE_WORDS: cell texts that mean no value (pandas' NA strings): no exact-text copy needed
        NUMERIC_TICK_FORMAT: tick-label format of numeric axes: up to 12 significant digits
# %%%%% More Info
    Imported by every AutoPlot program and plugin; needs numpy, h5py and
    vszh5_writer only (no Qt, no Veusz).
    Tests: python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.1: Faster building and saving of large projects (same output).
Date: 2026-11-09
#              * NativeDocument.utc_of: the reverse of local_of, kept up to
#                date; _layout_time_axis rebuilt it for every axis.
#              * SetDataText: NumPy U / S arrays converted in one call; the
#                NUL check searches the joined text once.
#              * TagDatasets: no copy of every dataset name per call.
#              With writer 0.0.28: 30-day OnePpsDeltas batch (319 files,
#              40 195 datasets) 17.9 s -> 8.4 s to build, 34.2 s -> 14.2 s
#              to save; projects identical.
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        TimeLayout (class): Gap rule of the time axes (0.5.0), see vszh5_writer TIME GAPS.
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0).
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        apply_settings: Set ``{setting path: value}`` on a widget (0.4.0).
        add_smith_graph: Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent`` (a page or grid widget); returns the graph widget (0.4.0).
        add_smith_page: A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).
        _data_name: The dataset a plotting widget draws (y for xy), or ''.
        _rename_from_data: Name an auto-named widget after its data (unique among siblings).
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
# %%%%% More Info
    Imported by every AutoPlot program and plugin; needs numpy, h5py and
    vszh5_writer only (no Qt, no Veusz).
    Tests: python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.0: Time axes laid out on save: local ISO 8601, small/large gaps.
Date: 2026-11-09
#              * finalize() lays out every horizontal axis whose traces are
#                against time - MJD/DMJD (1941-2132; MJD 0 padding ignored),
#                a Veusz date dataset, or a seconds column: MJD becomes a
#                Veusz date-time plot copy "<x>_local" (DST-aware local
#                clock, ISO 8601 ticks with the UTC offset, label "Local
#                time (ISO 8601, UTC-05:00)"); date datasets get the same.
#              * TimeLayout(k, n, manual_s, axis_manual_s) on the document
#                (set by the GUIs): SMALL gap - a trace's step dt >
#                k x median(dt) of that trace (or manual_s) - the line breaks
#                (NaN rows in the plot copies "<x>_local" / "<y>_gaps"; the
#                stored data are unchanged); LARGE gap - no trace on the
#                axis has data for dt > n x the largest small-gap threshold
#                (or axis_manual_s) - the x axis becomes axis-broken there.
#                Defaults k = 10, n = 100.  Re-laid out when the settings
#                change; the rule is written in each graph's notes.
#              * Pages with date-time axes and no explicit size get room for
#                the rotated ISO tick labels (13 cm per grid column, 8 cm per
#                row); date axes use 8 pt ticks and a 10 pt label; a DST
#                label names each real offset change once.
#              * Writer 0.0.27.
#              * Header in the author's template; docstrings for every
#                function, class and method.
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        TimeLayout (class): Gap rule of the time axes (0.5.0), see vszh5_writer TIME GAPS.
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0).
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        apply_settings: Set ``{setting path: value}`` on a widget (0.4.0).
        add_smith_graph: Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent`` (a page or grid widget); returns the graph widget (0.4.0).
        add_smith_page: A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).
        _data_name: The dataset a plotting widget draws (y for xy), or ''.
        _rename_from_data: Name an auto-named widget after its data (unique among siblings).
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
# %%%%% More Info
    Imported by every AutoPlot program and plugin; needs numpy, h5py and
    vszh5_writer only (no Qt, no Veusz).
    Tests: python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.4.0: Smith charts in the project, writer 0.0.26.
Date: 2026-11-09
#              * add_smith_page(root, page, grid, traces, ...) and
#                add_smith_graph(parent, grid, traces, ...): a Veusz Smith
#                chart (vszh5_writer.smith_graph_spec: aspect-locked graph,
#                hidden Re/Im Gamma axes, grid circles and arcs as xy lines,
#                grid values as point labels, traces of Im against Re Gamma,
#                legend beside the chart) on a NativeDocument - or on a
#                veusz.embed document, through apply_settings().
#              * apply_settings(widget, {path: value}): one specification for
#                both widget trees.
#              * Writer 0.0.26.
# %%%% 0.3.0: Local ISO date-time axes, names from the data, writer 0.0.25.
Date: 2026-11-09
#              * SetDataLocalTime(utc_name): from a stored UTC date dataset,
#                a second date dataset "<name>_local" on the clock of the
#                computer doing the processing (VSZH5_DISPLAY_TZ overrides),
#                DST-aware (vszh5_writer.local_time_from_veusz).  The UTC
#                values stay the stored Veusz date-time; plots bind the local
#                one.
#              * mark_local_time_axis(axis): the axis is a datetime axis whose
#                tick format and label are resolved from the local datasets
#                drawn on it when the document is finalised: ISO 8601,
#                24-hour, with the UTC offset, e.g.
#                %VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00 and the label "Local
#                time (ISO 8601, UTC-05:00)"; across a DST change the label
#                lists every offset (writer 0.0.24 rules).  Tick labels are
#                rotated 45 degrees.  Traces on a marked axis that bind a
#                UTC date dataset with a local copy draw the local copy.
#              * Widgets added without a name (Add("xy")) are named after
#                their data when the document is finalised: xy -> the y
#                dataset, nonorthpoint -> the r dataset, graph/polar -> "g_"
#                + the first trace's dataset (unique among siblings) instead
#                of "xy", "xy_1", "graph" ...  Named widgets are unchanged.
#              * finalize() runs both; save() and SaveToString() call it,
#                and the pyqtgraph Plots tab reads the finalised tree.
#              * Writer 0.0.25 (names, notes and local-time helpers).
# %%%% 0.2.0: Files written through vszh5_writer 0.0.24 (the shared master).
Date: 2026-11-05
#              The widget tree and dataset API are unchanged (SetData,
#              Root.Add, TagDatasets, save/Save ...).  Serialisation now goes
#              through the author's shared writer, vszh5_writer.py 0.0.24
#              (SHA-256 199f30a8...a901, master copy in the RF project):
#              VeuszWriter writes the datasets, tags and document atomically
#              (private ".part" file, os.replace), VszDoc builds the command
#              script and vz_text escapes captions (now also '[', ']' and a
#              backslash, Veusz's own latexEscape).  Large arrays are
#              gzip+shuffle compressed (>= 4096 elements).
#              New widget kinds for the ATR, CSV/TSV and R&S tools: polar,
#              nonorthpoint, function, bar.  New NativeWidget.typename and
#              childnames (veusz.embed names), data1/data2/lengths count as
#              dataset references (minimized saves), numpy scalars are stored
#              as plain Python values.
# %%%% 0.1.0: Direct .vszh5 serialization using h5py.
Date: 2026-10-11
=============================================================================
"""
from __future__ import annotations

import math
import os
import re
from collections import OrderedDict

import numpy as np

import vszh5_writer
from vszh5_writer import VeuszWriter, VszDoc, vz_text

VERSION = "0.5.4"
WRITER_VERSION = vszh5_writer.__version__
DATASET_SETTINGS = frozenset((
    "xData", "yData", "labels", "scalePoints", "yDataLow", "yDataHigh",
    "xDataLow", "xDataHigh", "data", "data1", "data2", "lengths",
))
WIDGET_KINDS = frozenset((
    "page", "grid", "graph", "axis", "axis-broken", "xy", "key", "image",
    "colorbar", "label", "polar", "nonorthpoint", "function", "bar",
))


# %% Safe script values
def _plain(value):
    """Plain Python literal value of a setting (numpy -> Python); validated."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite widget setting (dataset NaNs are supported)")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("Unsupported setting value: %s" % type(value).__name__)
    return value


def _literal(value):
    """Python literal text of a setting (kept for callers of 0.1.0)."""
    return repr(_plain(value))


def _escape_name(name):
    """Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name)."""
    return vszh5_writer.escape_hdf_name(name)


class _Setting:
    """Proxy for one setting path of a NativeWidget (``widget.Label.size``).

    Mimics the veusz.embed setting objects: attribute access walks deeper into
    the setting path ("Label" -> "Label/size") and ``.val`` reads or writes the
    value stored in the widget's settings dictionary.
    """
    def __init__(self, widget, path):
        """Remember the owning widget and the '/'-separated setting path."""
        self._widget = widget
        self._path = path

    def __getattr__(self, name):
        """Return the proxy for the child setting ``<path>/<name>``.

        Names starting with '_' raise AttributeError so copy/pickle and hasattr
        probes do not create bogus setting paths.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return _Setting(self._widget, self._path + "/" + name)

    @property
    def val(self):
        """Value of the setting, or None when it has never been set."""
        return self._widget.settings.get(self._path)

    @val.setter
    def val(self, value):
        """Store ``value`` (converted to plain Python/numpy types first, so a bad
        value fails here and leaves the widget unchanged).
        """
        self._widget.settings[self._path] = _plain(value)   # fail before modifying


# %% In-memory widget tree
class NativeWidget:
    """One widget of the native project tree (page, graph, axis, xy, ...).

    Mirrors the subset of the veusz.embed widget API the AutoPlot programs
    use - ``Add``, ``Remove``, ``WalkWidgets``, attribute access to children
    and settings - so the same page-building code runs against a NativeDocument
    (written with vszh5_writer, no Veusz needed) or a live Veusz document.
    """
    def __init__(self, document, kind, name, parent=None):
        """Create a widget of ``kind`` named ``name`` under ``parent``.

        ``settings`` holds the explicit setting values (path -> value) in the order
        they were set; ``children`` the child widgets in drawing order.
        """
        self.document = document
        self.kind = kind
        self.name = name
        self.parent = parent
        self.children = []
        self.settings = OrderedDict()
        self.auto_named = False
        self.time_source = None        # 0.5.0: (x, y) datasets before the time layout
        self.gap_source = None         # 0.5.3: (x, y) datasets before the numeric gap layout
        self.gap_layout_key = None     # 0.5.3: gap options applied to this numeric axis
        self.time_layout_key = None    # 0.5.0: TimeLayout applied to this axis       # 0.3.0: named from its data by finalize()
        self.local_time = None        # 0.3.0: label prefix of a local date-time axis

    @property
    def typename(self):
        """Widget type, as veusz.embed names it."""
        return self.kind

    @property
    def childnames(self):
        """Names of the child widgets, as veusz.embed names them."""
        return [c.name for c in self.children]

    @property
    def path(self):
        """Absolute widget path, e.g. ``/page_1/graph1/x``; ``/`` for the root.
        """
        if self.parent is None:
            return "/"
        return self.parent.path.rstrip("/") + "/" + self.name

    def __getattr__(self, name):
        """Child widget called ``name`` if there is one, else a setting proxy.

        As in veusz.embed, ``page.graph1`` returns the child widget and
        ``graph1.Border.hide`` the setting proxy.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        for child in self.children:
            if child.name == name:
                return child
        return _Setting(self, name)

    def Add(self, kind, name=None, autoadd=True, **settings):
        """Add a child widget of ``kind`` and return it.

        ``name`` defaults to the kind; a clash gets a numeric suffix (graph,
        graph_1, graph_2, ...).  A widget added without a name is marked
        ``auto_named`` so finalize() may rename it after its data.  Keyword
        arguments become settings.  With ``autoadd`` a new graph gets its x
        and y axes, as Veusz does; pass ``autoadd=False`` to build the axes
        yourself.  Unknown kinds and invalid names raise ValueError.
        """
        if kind not in WIDGET_KINDS:
            raise ValueError("Unsupported native widget: %s" % kind)
        existing = {c.name for c in self.children}
        stem = name or kind
        candidate = stem
        index = 1
        while candidate in existing:
            candidate = "%s_%d" % (stem, index)
            index += 1
        if not candidate or "/" in candidate or candidate in (".", ".."):
            raise ValueError("Invalid widget name: %r" % candidate)
        widget = NativeWidget(self.document, kind, candidate, self)
        widget.auto_named = name is None          # renamed from its data by finalize()
        for key, value in settings.items():
            widget.settings[key] = _plain(value)
        self.children.append(widget)
        if kind == "graph" and autoadd:
            widget.Add("axis", name="x", direction="horizontal")
            widget.Add("axis", name="y", direction="vertical")
        return widget

    def Remove(self, name=None):
        """Remove this widget (``name`` None) or its child called ``name``.

        The document root cannot be removed; an unknown child name raises
        KeyError.
        """
        if name is None:
            if self.parent is None:
                raise ValueError("Cannot remove document root")
            self.parent.children.remove(self)
            return
        for child in self.children:
            if child.name == name:
                self.children.remove(child)
                return
        raise KeyError(name)

    def WalkWidgets(self):
        """Yield this widget and every descendant, depth first, in drawing order.
        """
        yield self
        for child in self.children:
            yield from child.WalkWidgets()

    def _write(self, doc):
        """Emit this widget's settings and children into a vszh5_writer.VszDoc."""
        if self.kind == "graph" and "bottomMargin" not in self.settings:
            if any(child.kind in ("axis", "axis-broken") and
                   child.settings.get("mode") == "datetime"
                   for child in self.children):
                doc.set("bottomMargin", "3cm")
        if self.kind in ("axis", "axis-broken") and self.settings.get("mode") == "datetime":
            if "TickLabels/size" not in self.settings:
                doc.set("TickLabels/size", "8pt")
            if "Label/size" not in self.settings:          # 0.5.0: long ISO labels
                doc.set("Label/size", "10pt")
        for key, value in self.settings.items():
            # AutoPlot captions are plain text, not Veusz math/markup:
            # vz_text is Veusz's own latexEscape.  Dataset names and dataset
            # contents are never rewritten.
            if key in ("label", "key") and isinstance(value, str):
                value = vz_text(value)
            doc.set(key, value)
        for child in self.children:
            doc.add(child.kind, child.name)
            doc.to(child.name)
            child._write(doc)
            doc.up()


def local_veusz_seconds(utc_veusz, axis):
    """Local wall-clock Veusz date seconds of the UTC Veusz date seconds
    ``utc_veusz``, with the per-sample offsets of the LocalTimeAxis ``axis``
    (0.5.2): one addition, UTC + offset.

    The writer's ``axis.veusz_seconds`` goes through Unix epoch seconds
    (about 1.8e9 s, where float64 steps are 2.4e-7 s) and back, which
    rounded fractions of a second to about 0.24 microseconds; Veusz date
    values (about 5.6e8 s, steps 6e-8 s) keep them.  Whole seconds are
    identical either way; NaN stays NaN (its offset is 0).
    """
    return vszh5_writer.local_veusz_exact(utc_veusz, axis)      # 0.5.3: writer 0.0.30


def mark_local_time_axis(axis, label="Local time"):
    """
    Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis
    (0.3.0): its tick format and label are set from the local datasets drawn
    on it by NativeDocument.finalize() (ISO 8601, 24 h, with the offset).
    """
    axis.local_time = label
    axis.settings["mode"] = "datetime"
    axis.settings["autoRange"] = axis.settings.get("autoRange", "exact")
    return axis


# %% Gap texts (0.5.3; moved from _autoplot_common 0.4.2, unchanged)
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

def add_gap_arguments(parser):
    """Add the gap options to an argparse parser (0.5.3): --gap-k,
    --gap-small, --gap-n, --gap-large, --no-line-breaks, --no-axis-breaks,
    --no-numeric-gaps; layout_from_args() turns them into a TimeLayout."""
    group = parser.add_argument_group(
        "gaps", "A small gap breaks the line, a large gap breaks the x axis; either can be "
        "turned off. Time axes and (unless --no-numeric-gaps) any sorted x.")
    group.add_argument("--gap-k", type=float, default=vszh5_writer.GAP_K,
                       help="small gap: step > K x median step (default %(default)s)")
    group.add_argument("--gap-small", default="",
                       help="fixed small gap on time axes, e.g. 90s, 15min, 2h (overrides K)")
    group.add_argument("--gap-n", type=float, default=vszh5_writer.GAP_AXIS_N,
                       help="large gap: no data for > n x the small gap (0 = never; "
                            "default %(default)s)")
    group.add_argument("--gap-large", default="",
                       help="fixed large gap on time axes, e.g. 12h, 3d (overrides n)")
    group.add_argument("--no-line-breaks", action="store_true",
                       help="draw every line across its gaps")
    group.add_argument("--no-axis-breaks", action="store_true",
                       help="never break the x axis")
    group.add_argument("--no-numeric-gaps", action="store_true",
                       help="gap rule on time axes only (not frequency, angle, index ...)")
    return group


def layout_from_args(args):
    """The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3)."""
    return TimeLayout(k=args.gap_k, n=args.gap_n,
                      manual_s=parse_gap_string(args.gap_small) * 86400.0,
                      axis_manual_s=parse_gap_string(args.gap_large) * 86400.0,
                      break_lines=not args.no_line_breaks, break_axis=not args.no_axis_breaks,
                      numeric=not args.no_numeric_gaps)


# %% Time axes: local ISO 8601, line breaks and axis breaks (0.5.0)
class TimeLayout(object):
    """
    Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any
    sorted x, and the two consequences of a gap made optional), see
    vszh5_writer TIME GAPS and GAP OPTIONS (GapRule, 0.0.30).

    Small gap - the trace line breaks, the axis stays continuous:
        dt > k * median(dt) of that trace, or dt > manual_s when manual_s > 0.
    Large gap - the x axis itself breaks (Veusz axis-broken), where no trace
    on the axis has data for
        dt > n * (largest small-gap threshold of the traces on that axis),
        or dt > axis_manual_s when axis_manual_s > 0;
    n = 0 with axis_manual_s = 0: no axis breaks.

    0.5.3 options:
        break_lines - False draws every line across its gaps (no NaN rows).
        break_axis  - False keeps one continuous axis (as n = 0).
        numeric     - True applies the same rule to x axes that are not time
                      (frequency, angle, index ...) whose x data are sorted,
                      with the step also required to be large on a log
                      scale (GapRule geometric), so a log-spaced sweep is not
                      cut; the fixed gaps (seconds) apply to time axes only.
    Either break can be used alone or both together (the 0.5.2 behaviour,
    which every default keeps).
    """

    def __init__(self, k=vszh5_writer.GAP_K, n=vszh5_writer.GAP_AXIS_N, manual_s=0.0,
                 axis_manual_s=0.0, break_lines=True, break_axis=True, numeric=True):
        """Store the gap settings (negative values are clamped to 0).

        k: small-gap factor - a trace's line breaks where its step dt exceeds
        k x median(dt) of that trace.
        n: large-gap factor - the x axis breaks where no trace has data for
        dt > n x the largest small-gap threshold; 0 disables axis breaks.
        manual_s / axis_manual_s: fixed small / large gap in seconds; > 0
        overrides k / n (time axes).
        break_lines / break_axis / numeric: the 0.5.3 options (class text).
        """
        self.k = float(k)
        self.n = max(0.0, float(n))
        self.manual_s = max(0.0, float(manual_s or 0.0))
        self.axis_manual_s = max(0.0, float(axis_manual_s or 0.0))
        self.break_lines = bool(break_lines)
        self.break_axis = bool(break_axis)
        self.numeric = bool(numeric)

    def key(self):
        """Tuple of the settings; a change re-lays out the axes on save."""
        return (self.k, self.n, self.manual_s, self.axis_manual_s, self.break_lines,
                self.break_axis, self.numeric)

    def gap_rule(self, geometric=False):
        """The writer's GapRule of these settings (0.5.3): for a time axis
        (seconds, the fixed gaps apply) or, with ``geometric``, for a
        numeric axis (k and n only; the step must also be large on a log
        scale)."""
        return vszh5_writer.GapRule(
            k=self.k, n=self.n,
            manual=0.0 if geometric else self.manual_s,
            axis_manual=0.0 if geometric else self.axis_manual_s,
            break_lines=self.break_lines, break_axis=self.break_axis, geometric=bool(geometric))

    def axis_threshold(self, line_threshold):
        """Large-gap threshold for an axis whose largest small-gap threshold
        is ``line_threshold`` (seconds); 0.0 = no axis breaks."""
        if not self.break_axis:
            return 0.0
        if self.axis_manual_s > 0:
            return max(self.axis_manual_s, float(line_threshold))
        if self.n > 0 and line_threshold > 0:
            return max(self.n, 1.0) * float(line_threshold)
        return 0.0

    def describe(self, numeric=False):
        """One-line human description of the gap rule, written into graph notes
        (0.5.3: the options; every number exact; ``numeric`` describes the
        rule of a non-time axis)."""
        if numeric:
            return vszh5_writer.GapRule(
                k=self.k, n=self.n, break_lines=self.break_lines, break_axis=self.break_axis,
                geometric=True).describe()
        if not self.break_lines:
            line = "lines drawn across gaps"
        elif self.manual_s > 0:
            line = "line break (small gap) where dt > %s" % _seconds_text(self.manual_s)
        else:
            line = "line break (small gap) where dt > %s x median(dt)" % _number_text(self.k)
        if not self.break_axis:
            axis = "no x-axis breaks"
        elif self.axis_manual_s > 0:
            axis = "x-axis break (large gap) where dt > %s" % _seconds_text(self.axis_manual_s)
        elif self.n > 0:
            axis = "x-axis break (large gap) where dt > %s x the small-gap threshold" % (
                _number_text(self.n))
        else:
            axis = "no x-axis breaks"
        return "Gaps: %s; %s." % (line, axis)


#: (0.5.3) Same as TimeLayout: the gap options apply to every x axis.
GapLayout = TimeLayout


def _number_text(value):
    """A setting for the notes with every digit (repr), without a trailing
    ".0" (0.5.3; %g cut 12.3456789 to 12.3457)."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _seconds_text(seconds):
    """A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.

    0.5.2: when the short form is not exact, the exact seconds follow in
    parentheses (repr: every digit), e.g. '1.334 min (80.04000000000001 s)';
    the notes are exported with the project, so nothing is rounded away.
    """
    seconds = float(seconds)
    exact = "%s s" % repr(seconds)
    for size, unit in ((86400.0, "d"), (3600.0, "h"), (60.0, "min")):
        if seconds >= size:
            short = "%g %s" % (round(seconds / size, 3), unit)
            same = float("%g" % round(seconds / size, 3)) * size == seconds
            return short if same else "%s (%s)" % (short, exact)
    short = "%g s" % round(seconds, 3)
    return short if float("%g" % round(seconds, 3)) == seconds else "%s (%s)" % (short, exact)


#: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
MJD_VEUSZ_EPOCH = 54832.0
#: Plausible MJD range of a time column (1941 .. 2132); values outside are
#: not taken for MJD (and MJD 0 padding rows are ignored).
MJD_RANGE = (30000.0, 100000.0)
_TIME_NAME_HINT = re.compile(r"(^|[^A-Za-z])(D?MJD|JD|TIME|DATE)([^A-Za-z]|$)|__x__", re.I)
_SECONDS_NAMES = frozenset(("TIME", "TIMESTAMP", "TIMESTAMP_SEC", "SECONDS", "SEC",
                            "UT_SECONDS", "UTSECONDS", "ELAPSED", "ELAPSED_S", "T_SEC"))


def _time_kind(doc, name, axis):
    """'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time)."""
    entry = doc.datasets.get(name) if isinstance(name, str) else None
    if entry is None:
        return None
    kind, parts = entry
    if kind == "date":
        return "date"
    if kind != "1d":
        return None
    values = np.asarray(parts["data"], dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]     # MJD 0 = padding row
    label = str(axis.settings.get("label", "") or "")
    hinted = bool(_TIME_NAME_HINT.search(name) or _TIME_NAME_HINT.search(label))
    if (hinted and finite.size and np.all(finite >= MJD_RANGE[0])
            and np.all(finite <= MJD_RANGE[1])):
        return "mjd"
    if name.split("__")[-1].upper() in _SECONDS_NAMES or label.strip().upper() in _SECONDS_NAMES:
        return "seconds"
    return None


def _plot_name(doc, wanted, owner):
    """A dataset name for a plot copy: ``wanted``, or wanted_2, ... when
    another source already uses that name (owner identifies the source)."""
    used = doc._plot_copies
    name, number = wanted, 2
    while name in used and used[name] != owner or (name in doc.datasets and name not in used):
        name = "%s_%d" % (wanted, number)
        number += 1
    used[name] = owner
    return name


def _layout_time_axis(doc, graph, axis):
    """
    Lay out one x axis whose traces are against time (0.5.0).

    MJD or Veusz-date x data are stored once as a UTC date dataset
    ("<x>_utc" for MJD) and drawn from a local copy "<utc>_local" (this
    computer's clock, DST-aware, ISO 8601 ticks with the UTC offset). A
    trace whose step exceeds the line-break threshold gets NaN rows there
    (plot copies "<utc>_local" and "<y>_gaps"; the stored data stay
    unchanged). The axis becomes axis-broken only for large gaps (n > 0).
    """
    layout = getattr(doc, "time_layout", None) or TimeLayout()
    if not hasattr(doc, "_plot_copies"):
        doc._plot_copies = {}
    xys = [c for c in graph.WalkWidgets()
           if c.kind == "xy" and c.settings.get("xAxis", "x") == axis.name]
    if not xys:
        return False
    if (axis.time_layout_key == layout.key()
            and all(xy.time_source is not None for xy in xys)):
        return True
    # 0.5.1: the document keeps the reverse map (local copy -> UTC dataset);
    # rebuilding it here for every axis was O(datasets) per axis.
    reverse_local = getattr(doc, "utc_of", None)
    if reverse_local is None:                 # a document made before 0.5.1
        reverse_local = doc.utc_of = dict((v, k) for k, v in doc.local_of.items())
    plans = []
    for xy in xys:
        x_name, y_name = xy.time_source or (xy.settings.get("xData"), xy.settings.get("yData"))
        x_name = reverse_local.get(x_name, x_name)
        kind = _time_kind(doc, x_name, axis)
        if kind is None or not isinstance(y_name, str) or y_name not in doc.datasets:
            plans.append(None)
            continue
        plans.append((xy, x_name, y_name, kind))
    timed = [p for p in plans if p is not None]
    if not timed:
        return False
    kinds = set(p[3] for p in timed)
    dated = kinds <= set(("date", "mjd"))
    local_names, union, thresholds = [], [], []
    for xy, x_name, y_name, kind in timed:
        values = np.asarray(doc.datasets[x_name][1]["data"], dtype=float)
        if kind == "mjd":
            # Veusz date seconds = (MJD - 54832) * 86400 (54832 = 2009-01-01).
            # The MJD dataset stays the stored value; the plot copy below is
            # the Veusz date-time dataset (a tool stores a UTC date dataset
            # of its raw time column itself, once per file - not per trace).
            utc = (values - MJD_VEUSZ_EPOCH) * 86400.0
            utc[(values < MJD_RANGE[0]) | (values > MJD_RANGE[1])] = np.nan
            utc_name = x_name
        elif kind == "date":
            utc, utc_name = values, x_name
        else:
            utc, utc_name = values, x_name
        seconds = utc                                  # gap rule in seconds
        threshold = vszh5_writer.gap_threshold(seconds, layout.k, layout.manual_s)
        cut = vszh5_writer.gap_positions(seconds, threshold)
        if not layout.break_lines:                     # 0.5.3: lines across gaps
            cut = cut[:0]
        thresholds.append(threshold)
        if dated:
            local = vszh5_writer.local_time_from_veusz(utc)
            local_values = local_veusz_seconds(utc, local)      # 0.5.2: exact sum
            local_name = utc_name + "_local"
            doc.SetDataDateTime(local_name, vszh5_writer.insert_gaps(local_values, cut))
            doc.local_axes[local_name] = local
            doc.local_of[utc_name] = local_name
            doc.utc_of[local_name] = utc_name
            plot_x = local_name
            union.append(local_values)
            if local_name not in local_names:
                local_names.append(local_name)
        else:
            plot_x = x_name
            if cut.size:
                plot_x = _plot_name(doc, x_name + "_gaps", x_name)
                doc.SetData(plot_x, vszh5_writer.insert_gaps(values, cut))
            union.append(values)
        plot_y = y_name
        if cut.size:
            y_values = doc.datasets[y_name][1]["data"]
            if np.asarray(y_values).shape == values.shape:
                plot_y = _plot_name(doc, y_name + "_gaps", (x_name, y_name))
                doc.SetData(plot_y, vszh5_writer.insert_gaps(y_values, cut))
            else:
                # y does not line up with x: draw it unbroken against a
                # local copy without gap rows (never misalign the points).
                if dated:
                    plot_x = _plot_name(doc, utc_name + "_local_nogaps", utc_name)
                    doc.SetDataDateTime(plot_x, local_values)
                    doc.local_axes[plot_x] = local
                else:
                    plot_x = x_name
        xy.time_source = (x_name, y_name)
        xy.settings["xData"] = plot_x
        xy.settings["yData"] = plot_y
    # Large gaps break the x axis: only where no trace has data for longer
    # than the large-gap threshold (n x the largest small-gap threshold, or
    # the fixed large gap).
    pairs = []
    axis_threshold = layout.axis_threshold(max(thresholds) if thresholds else 0.0)
    if axis_threshold > 0:
        pairs = vszh5_writer.axis_break_pairs(np.concatenate(union), axis_threshold)
    if pairs:
        axis.kind = "axis-broken"
        axis.settings["breakPoints"] = [float(v) for pair in pairs for v in pair]
    else:
        axis.kind = "axis"
        axis.settings.pop("breakPoints", None)
    if dated:
        fmt, text = doc.local_axis_text(local_names, "Local time")
        axis.settings["mode"] = "datetime"
        axis.settings["TickLabels/format"] = fmt
        axis.settings.setdefault("TickLabels/rotate", str(vszh5_writer.LOCAL_DATETIME_ROTATE))
        axis.settings["label"] = text
        axis.settings.setdefault("autoRange", "exact")
    axis.time_layout_key = layout.key()
    _set_gap_note(graph, layout.describe())
    return True


def _set_gap_note(graph, note):
    """Put the gap-rule line ``note`` into the graph notes (0.5.3: a line
    from an earlier layout is replaced, so the notes state the rule in use)."""
    lines = [line for line in str(graph.settings.get("notes", "")).split("\n")
             if not line.startswith("Gaps: ")]
    lines.append(note)
    graph.settings["notes"] = "\n".join(line for line in lines if line).strip()


def _sorted_finite(values):
    """True when the finite values of ``values`` never decrease (at least 3)."""
    finite = values[np.isfinite(values)]
    return finite.size >= 3 and bool(np.all(np.diff(finite) >= 0))


def _layout_numeric_axis(doc, graph, axis):
    """
    Gap options on an x axis that is not time (0.5.3): frequency, angle,
    index ...

    For every xy trace on the axis whose x dataset is numeric, 1-D and
    sorted, the writer's geometric GapRule (TimeLayout.gap_rule) gives the
    line breaks ("<x>_gaps" / "<y>_gaps" plot copies with NaN rows; the
    stored data stay unchanged) and, when every trace on the axis is sorted,
    the axis breaks (Veusz axis-broken). A label axis, a hidden axis (Smith
    chart) or an axis with any unsorted trace (a loop, a polar trace) is
    left as it is.
    """
    layout = getattr(doc, "time_layout", None) or TimeLayout()
    xys = [c for c in graph.WalkWidgets()
           if c.kind == "xy" and c.settings.get("xAxis", "x") == axis.name]
    if not layout.numeric:
        _undo_numeric_layout(graph, axis, xys)   # an earlier save had them on
        return
    if axis.settings.get("hide", False) or axis.settings.get("mode", "numeric") != "numeric":
        return
    if not hasattr(doc, "_plot_copies"):
        doc._plot_copies = {}
    if not xys:
        return
    if axis.gap_layout_key == layout.key() and all(xy.gap_source is not None for xy in xys):
        return
    rule = layout.gap_rule(geometric=True)
    plans = []
    for xy in xys:
        x_name, y_name = xy.gap_source or (xy.settings.get("xData"), xy.settings.get("yData"))
        entry = doc.datasets.get(x_name) if isinstance(x_name, str) else None
        if entry is None or entry[0] != "1d":
            return                        # an x that is not a numeric dataset: leave the axis
        values = np.asarray(entry[1]["data"], dtype=float)
        if values.ndim != 1 or not _sorted_finite(values):
            return                        # unsorted: no gaps are defined on this axis
        plans.append((xy, x_name, y_name, values))
    series = []
    for xy, x_name, y_name, values in plans:
        series.append(values)
        cut = vszh5_writer.rule_gap_positions(values, rule)
        plot_x, plot_y = x_name, y_name
        y_entry = doc.datasets.get(y_name) if isinstance(y_name, str) else None
        if (cut.size and y_entry is not None and y_entry[0] == "1d"
                and np.asarray(y_entry[1]["data"]).shape == values.shape):
            plot_x = _plot_name(doc, x_name + "_gaps", x_name)
            doc.SetData(plot_x, vszh5_writer.insert_gaps(values, cut))
            plot_y = _plot_name(doc, y_name + "_gaps", (x_name, y_name))
            doc.SetData(plot_y, vszh5_writer.insert_gaps(y_entry[1]["data"], cut))
        xy.gap_source = (x_name, y_name)
        xy.settings["xData"] = plot_x
        xy.settings["yData"] = plot_y
    pairs = vszh5_writer.rule_axis_break_pairs(series, rule)
    if pairs:
        axis.kind = "axis-broken"
        axis.settings["breakPoints"] = vszh5_writer.break_points(pairs)
    elif axis.kind == "axis-broken" or "breakPoints" in axis.settings:
        axis.kind = "axis"
        axis.settings.pop("breakPoints", None)
    axis.gap_layout_key = layout.key()
    _set_gap_note(graph, layout.describe(numeric=True))


def _undo_numeric_layout(graph, axis, xys):
    """Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the
    gap options of a project turn non-time axes off between two saves, the
    traces draw their own data again, the axis is no longer broken and the
    "Gaps:" note goes; the unused copies are dropped by
    _drop_unused_plot_copies. An axis this module never laid out is left
    as it is."""
    if axis.gap_layout_key is None:
        return
    for xy in xys:
        if xy.gap_source is not None:
            xy.settings["xData"], xy.settings["yData"] = xy.gap_source
            xy.gap_source = None
    if axis.kind == "axis-broken" or "breakPoints" in axis.settings:
        axis.kind = "axis"
        axis.settings.pop("breakPoints", None)
    axis.gap_layout_key = None
    _set_gap_note(graph, "")


def _drop_unused_plot_copies(doc):
    """Remove line-break plot copies no widget draws any more (0.5.3): when
    the gap options change between two saves, copies of the old layout
    would otherwise stay in the file. Only names made by _plot_name."""
    copies = getattr(doc, "_plot_copies", None)
    if not copies:
        return
    used = set()
    for widget in doc.Root.WalkWidgets():
        for key in ("xData", "yData"):
            value = widget.settings.get(key)
            if isinstance(value, str):
                used.add(value)
    for name in [n for n in copies if n not in used and n in doc.datasets]:
        if name.endswith("_gaps") or "_gaps_" in name or name.endswith("_local_nogaps"):
            del doc.datasets[name]
            for members in getattr(doc, "tags", {}).values():
                if name in members:
                    members.remove(name)
            del copies[name]


def _size_dated_page(page):
    """
    Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).

    Only pages without an explicit size: Veusz's default 15 x 15 cm page
    leaves a grid of graphs no room for 25-character rotated tick labels.
    A grid gets 13 cm per column and 8 cm per row; a single graph 25 x 16 cm.
    """
    if "width" in page.settings or "height" in page.settings:
        return
    if not any(w.kind in ("axis", "axis-broken") and w.settings.get("mode") == "datetime"
               for w in page.WalkWidgets()):
        return
    graphs = [w for w in page.WalkWidgets() if w.kind == "graph"]
    grids = [w for w in page.children if w.kind == "grid"]
    if grids and graphs:
        try:
            columns = max(1, int(grids[0].settings.get("columns", 2) or 2))
        except (TypeError, ValueError):
            columns = 2
        rows = int(math.ceil(len(graphs) / float(columns)))
        page.settings["width"] = "%dcm" % (13 * columns)
        page.settings["height"] = "%dcm" % max(16, 8 * rows)
    else:
        page.settings["width"] = "25cm"
        page.settings["height"] = "16cm"


# %% Smith charts (0.4.0)
def apply_settings(widget, settings):
    """
    Set ``{setting path: value}`` on a widget (0.4.0).

    A NativeWidget stores the values directly; any other widget (a
    veusz.embed node) is reached by walking the path as attributes
    (``"PlotLine/color"`` -> ``widget.PlotLine.color.val``), so the same
    specification builds a Veusz-embedded document too.
    """
    if isinstance(widget, NativeWidget):
        for path, value in settings.items():
            widget.settings[path] = _plain(value)
        return widget
    for path, value in settings.items():
        node = widget
        for part in path.split("/"):
            node = getattr(node, part)
        node.val = value
    return widget


def add_smith_graph(parent, grid, traces, name=None, notes="", key="auto",
                    key_position=("outside", "top"), key_width="6.5cm",
                    limit=vszh5_writer.SMITH_LIMIT):
    """
    Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent``
    (a page or grid widget); returns the graph widget (0.4.0).

    ``grid`` is a vszh5_writer.SmithGrid whose datasets are written to the
    same document (``grid.write(doc)``); ``traces`` are
    vszh5_writer.smith_trace() traces.  Captions are passed unescaped:
    NativeWidget escapes "label" and "key" text when it writes the file.
    """
    graph_settings, children = vszh5_writer.smith_graph_spec(
        grid, traces, key=key, key_position=key_position, limit=limit, escape=False,
        notes=notes, key_width=key_width)
    graph = parent.Add("graph", name=name or "smith", autoadd=False)
    apply_settings(graph, graph_settings)
    for kind, child_name, settings in children:
        apply_settings(graph.Add(kind, name=child_name), settings)
    return graph


def add_smith_page(root, page_name, grid, traces, title="", notes="", graph_name="",
                   graph_notes="", size="18cm", key="auto", key_position=("outside", "top"),
                   key_width="6.5cm"):
    """
    A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a
    NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).

    ``notes`` are the page notes (source header); ``graph_notes`` are added
    below the chart type in the graph notes (e.g. "Z0 = 50 ohm").
    """
    width, height, top_margin = vszh5_writer.smith_page_layout(
        traces, title, size=size, key=key, key_position=key_position, key_width=key_width)
    page = root.Add("page", name=page_name)
    apply_settings(page, {"width": width, "height": height})
    if notes:
        apply_settings(page, {"notes": notes})
    if title:
        apply_settings(page.Add("label", name="title"),
                       vszh5_writer.title_settings(title, escape=False))
    graph = add_smith_graph(page, grid, traces,
                            name=graph_name or vszh5_writer.widget_name(page_name, prefix="g_"),
                            notes=graph_notes, key=key, key_position=key_position,
                            key_width=key_width)
    apply_settings(graph, {"topMargin": top_margin})
    return graph


_DATA_SETTING_OF = {"xy": "yData", "nonorthpoint": "data1", "image": "data",
                    "bar": "lengths", "function": "function"}


def _data_name(widget):
    """The dataset a plotting widget draws (y for xy), or ''."""
    value = widget.settings.get(_DATA_SETTING_OF.get(widget.kind, ""), "")
    if isinstance(value, (tuple, list)):
        value = value[0] if value else ""
    return value if isinstance(value, str) else ""


def _rename_from_data(widget):
    """Name an auto-named widget after its data (unique among siblings)."""
    parent = widget.parent
    if parent is None:
        return
    if widget.kind in _DATA_SETTING_OF:
        data = _data_name(widget)
        base = vszh5_writer.widget_name(data) if data else ""
    elif widget.kind in ("graph", "polar"):
        first = next((_data_name(c) for c in widget.WalkWidgets()
                      if c is not widget and _data_name(c)), "")
        base = vszh5_writer.widget_name(first, prefix="g_") if first else ""
    else:
        base = ""
    widget.auto_named = False
    if not base or base == widget.name:
        return
    used = {c.name for c in parent.children if c is not widget}
    widget.name = vszh5_writer.unique_widget_name(base, used)


# %% Dataset tags and trace names (0.5.2; shared by every AutoPlot program)
# Every dataset carries Veusz tags source_<origin> (file or program) and
# type_<kind> (what the values are); trace widgets are named after the
# data they draw.  Imported by _autoplot_common, ATR, R&S, CSV and
# Touchstone AutoPlot (no Qt here, so batch modes and workers stay light).
#: Characters allowed in a tag word; everything else becomes "_".
_TAG_WORD = re.compile(r"[^A-Za-z0-9_]+")


def tag_word(text):
    """``text`` as a tag word: letters, digits and '_' only (never empty)."""
    word = _TAG_WORD.sub("_", str(text)).strip("_")
    return word or "unnamed"


def source_tag(source):
    """Veusz tag naming where data come from: ``source_<word>``."""
    return "source_" + tag_word(source)


def type_tag(kind):
    """Veusz tag naming what data are: ``type_<word>``."""
    return "type_" + tag_word(kind)


def tag_data(doc, names, sources=(), kinds=()):
    """Tag the datasets ``names`` with ``source_*`` and ``type_*`` tags.

    ``sources`` and ``kinds`` are one string or a sequence of them.  Works
    with a NativeDocument and with Veusz's own document (plugins).
    """
    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return
    if isinstance(sources, str):
        sources = (sources,)
    if isinstance(kinds, str):
        kinds = (kinds,)
    for tag in [source_tag(s) for s in sources] + [type_tag(k) for k in kinds]:
        doc.TagDatasets(tag, names)


def _safe_name(name):
    """``name`` with letters, digits and '_' only (any other character
    becomes '_'), so a widget name reads like the dataset it draws; a
    leading digit stays (Veusz widget names allow it)."""
    text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    return text or "trace"


def trace_widget_name(y_dataset, variant=""):
    """Name of the xy widget that draws ``y_dataset``: the dataset name (it
    holds the source file, the item and the processing type), plus
    ``_<variant>`` for another drawing of the same data (e.g. "local_time",
    "date_labels").  Letters, digits and '_' only."""
    name = _safe_name(y_dataset)
    return name + ("_" + tag_word(variant) if variant else "")


def exact_number_text(value):
    """A number as text with every digit: repr() of a float (the shortest
    text that gives back the same float64), str() of an integer."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return repr(number) if math.isfinite(number) else str(number)


#: Cell texts that mean "no value" (pandas' default NA strings): a NaN read
#: from one of these loses nothing, so it needs no exact-text copy.
MISSING_VALUE_WORDS = frozenset((
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
    "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null", "+nan",
    "+NaN"))


def float_loses_text(texts, values):
    """True when float64 ``values`` parsed from the cell ``texts`` do not hold
    all of them (0.5.2): a cell that is not a number and not a missing-value
    word (MISSING_VALUE_WORDS), or a number with more significant digits
    than float64 keeps.

    A text of 15 characters or fewer has at most 15 digits, which float64
    always keeps (DBL_DIG = 15), so only longer ones are compared - exactly,
    with decimal.Decimal against repr() of the parsed value.
    """
    import decimal
    text = np.char.strip(np.asarray(texts).astype(str))
    values = np.asarray(values, dtype=float).reshape(-1)
    if text.size == 0:
        return False
    missing = np.isin(text, sorted(MISSING_VALUE_WORDS))
    if np.any(~missing & np.isnan(values)):
        return True
    long_text = np.char.str_len(text) > 15
    for token, value in zip(text[long_text].tolist(), values[long_text].tolist()):
        if not math.isfinite(value):
            continue
        try:
            if decimal.Decimal(token) != decimal.Decimal(repr(value)):
                return True
        except decimal.InvalidOperation:
            return True
    return False


#: (0.5.2) Tick-label format of numeric axes: up to 12 significant digits
#: (the MHM2010 / T4 workbench 0.2.3 uses the same); ticks are positions,
#: the data are stored with every digit.
NUMERIC_TICK_FORMAT = "%.12Vg"


# %% Memory release (0.5.4): Clear gives the memory back to the OS
def process_rss_mb():
    """Resident set size of this process in MiB (NaN when it cannot be read).

    psutil when installed; otherwise /proc/self/statm (Linux) or
    GetProcessMemoryInfo (Windows) through ctypes.
    """
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / 1048576.0
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                """PROCESS_MEMORY_COUNTERS (psapi)."""
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            # private library objects: argtypes set here do not change the
            # shared ctypes.windll function objects other code may use
            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / 1048576.0
        except Exception:
            pass
    return float("nan")


def _release_os_heap():
    """Hand freed heap memory back to the operating system; returns the steps
    that ran (names), [] when nothing applies.

    * Linux (glibc): malloc_trim(0) - free pages at the top of the heap and
      in every arena go back to the kernel (glibc keeps them otherwise once
      its mmap threshold has grown).  musl and others have no malloc_trim.
    * Windows: _heapmin() of the C runtime ("releases unused heap memory to
      the operating system"), HeapCompact() of every process heap (it
      coalesces free blocks; Windows normally does that at HeapFree
      already), then SetProcessWorkingSetSize(-1, -1), which removes as many
      pages as possible from the working set (what Task Manager shows as
      memory in use).
    * macOS: malloc_zone_pressure_relief(NULL, 0).
    """
    import ctypes
    import ctypes.util
    import sys as _sys
    steps = []
    if _sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(ctypes.c_size_t(0))
                steps.append("malloc_trim")
        except Exception:
            pass
    elif os.name == "nt":
        try:
            for crt in ("ucrtbase", "msvcrt"):
                try:
                    getattr(ctypes.cdll, crt)._heapmin()
                    steps.append("_heapmin")
                    break
                except Exception:
                    continue
            # a private kernel32 object: the argtypes set below do not change
            # the shared ctypes.windll function objects other code may use
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetProcessHeaps.restype = ctypes.c_uint32
            kernel32.GetProcessHeaps.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
            count = kernel32.GetProcessHeaps(0, None)
            if count:
                heaps = (ctypes.c_void_p * count)()
                count = kernel32.GetProcessHeaps(count, heaps)
                kernel32.HeapCompact.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                for index in range(count):
                    kernel32.HeapCompact(heaps[index], 0)
                steps.append("HeapCompact x%d" % count)
            kernel32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                                          ctypes.c_size_t]
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            minus_one = ctypes.c_size_t(-1).value
            if kernel32.SetProcessWorkingSetSize(kernel32.GetCurrentProcess(),
                                                 minus_one, minus_one):
                steps.append("working set trimmed")
        except Exception:
            pass
    elif _sys.platform == "darwin":
        try:
            libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
            libsystem.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
            steps.append("malloc_zone_pressure_relief")
        except Exception:
            pass
    return steps


def release_memory(log=None):
    """Give the memory of dropped data back to the operating system (0.5.4).

    Call it after the last reference to the data is gone (project, plots,
    caches): a full garbage collection (three passes: widget trees and
    documents form reference cycles), the GPU memory pools when the GPU
    module is loaded (_autoplot_gpu.release_memory), then the C heap
    (_release_os_heap).  numpy buffers of 128 KiB or more are mapped
    separately and go back as soon as they are freed; the heap step returns
    the smaller ones.  ``log`` (optional) gets one line with the RSS before
    and after.  Returns {"before_mb", "after_mb", "collected", "steps"}.
    """
    import gc
    import sys as _sys
    before = process_rss_mb()
    collected = 0
    for _ in range(3):
        collected += gc.collect()
    steps = []
    gpu = _sys.modules.get("_autoplot_gpu")
    if gpu is not None and hasattr(gpu, "release_memory"):
        try:
            text = gpu.release_memory()
            if text:
                steps.append(text)
        except Exception:
            pass
    steps.extend(_release_os_heap())
    after = process_rss_mb()
    if log is not None:
        try:
            log("Memory released: RSS %.1f -> %.1f MiB (%s)"
                % (before, after, ", ".join(["gc %d objects" % collected] + steps)))
        except Exception:
            pass
    return {"before_mb": before, "after_mb": after, "collected": collected, "steps": steps}


# %% Data kept between batches (0.5.4)
def file_stamp(path, options=()):
    """(size, mtime_ns, options) of ``path``: the data read from it are still
    good while this is unchanged; None when the file cannot be read."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_size, info.st_mtime_ns, tuple(options))


class KeptReads(object):
    """Data read from the listed files, kept until the user clears them (0.5.4).

    A program reads only the files it has no data for (to_read: new files,
    files changed on disk, files read with other read ``options``) and builds
    its project from every listed file (items).  A kept file that is gone
    from disk keeps its data: adding files never makes the program forget
    the ones it already has.  clear() forgets everything (the Clear button).
    """

    def __init__(self):
        """Nothing kept yet."""
        self._data = {}
        self._stamps = {}

    def __len__(self):
        """Number of files with kept data."""
        return len(self._data)

    def __contains__(self, path):
        """True when data of ``path`` are kept."""
        return path in self._data

    def to_read(self, paths, options=()):
        """The ``paths`` that must be read (in order, each once)."""
        todo = []
        for path in dict.fromkeys(paths):
            if path not in self._data:
                todo.append(path)
                continue
            stamp = file_stamp(path, options)
            if stamp is not None and stamp != self._stamps.get(path):
                todo.append(path)
        return todo

    def keep(self, path, data, options=()):
        """Keep ``data`` read from ``path`` (None forgets the file, so the next
        batch reads it again)."""
        if data is None:
            self.forget(path)
            return
        self._data[path] = data
        self._stamps[path] = file_stamp(path, options)

    def forget(self, path):
        """Drop the data of ``path``."""
        self._data.pop(path, None)
        self._stamps.pop(path, None)

    def get(self, path, default=None):
        """Kept data of ``path`` (``default`` when none)."""
        return self._data.get(path, default)

    def items(self, paths):
        """(path, data) of every listed path with kept data, in list order."""
        return [(path, self._data[path]) for path in dict.fromkeys(paths) if path in self._data]

    def clear(self):
        """Forget every file's data."""
        self._data = {}
        self._stamps = {}


# %% Dataset model and file output
class NativeDocument:
    """Build and atomically save a self-contained Veusz format-1 document."""

    def __init__(self, name="AutoPlot"):
        """Empty project: a document root, no datasets, default TimeLayout.

        ``local_axes`` maps each local date-time display dataset to its
        vszh5_writer.LocalTimeAxis (tick format and label); ``local_of`` maps a
        UTC date dataset to its local copy.
        """
        self.name = name
        self.Root = NativeWidget(self, "document", "")
        self.datasets = OrderedDict()
        self.tags = {}
        self.local_axes = {}          # local dataset name -> vszh5_writer.LocalTimeAxis
        self.local_of = {}            # UTC date dataset name -> its local copy
        self.utc_of = {}              # 0.5.1: the reverse map, local copy -> UTC name
        # 0.5.0: gap rule of the time axes resolved by finalize():
        # line break when dt > k * median(dt) (or > manual_s seconds),
        # axis break when dt > n * k * median(dt); n = 0: one axis.
        self.time_layout = TimeLayout()

    def release(self):
        """Drop every dataset, widget and tag of the project (0.5.4).

        Used when the user clears the files: the widget tree's links (each
        widget points to its document and parent) are cut so the arrays are
        freed at once, without waiting for the garbage collector, and the
        document is left empty and usable.
        """
        def cut(widget):
            """Unlink ``widget`` and everything below it."""
            for child in list(widget.children):
                cut(child)
            widget.children = []
            widget.settings = OrderedDict()
            widget.parent = None
            widget.time_source = widget.gap_source = None
        cut(self.Root)
        self.Root = NativeWidget(self, "document", "")
        self.datasets = OrderedDict()
        self.tags = {}
        self.local_axes = {}
        self.local_of = {}
        self.utc_of = {}
        if hasattr(self, "_plot_copies"):
            self._plot_copies = {}

    def _put(self, name, kind, parts):
        """Store dataset ``name`` of ``kind`` ('1d', 'date', 'text', '2d') with its
        arrays ``parts``; rejects empty names, NUL characters, '.' and '..'.
        """
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("Dataset name must be nonempty text without NUL")
        if name in (".", ".."):
            raise ValueError("Dataset name cannot be '.' or '..'")
        self.datasets[name] = (kind, parts)

    def SetData(self, name, data=None, symerr=None, negerr=None, poserr=None, val=None):
        """1-D numeric dataset (``val=`` accepted as veusz.embed spells it)."""
        if data is None:
            data = val
        parts = {}
        for key, values in (("data", data), ("serr", symerr),
                            ("nerr", negerr), ("perr", poserr)):
            if values is not None:
                array = np.array(values, dtype=np.float64, copy=True)
                if array.ndim == 0:
                    array = array.reshape(1)
                if array.ndim != 1:
                    raise ValueError("%s/%s requires a 1-D array" % (name, key))
                if key != "data" and array.shape != parts["data"].shape:
                    raise ValueError("Error array length mismatch: %s" % name)
                parts[key] = array
        self._put(name, "1d", parts)

    def SetDataText(self, name, values):
        """Store a text dataset; bytes are decoded as UTF-8 (bad bytes replaced).

        NUL characters are rejected because HDF5 strings cannot hold them.
        """
        arr = values if isinstance(values, np.ndarray) else None
        if arr is not None and arr.dtype.kind == "U":          # 0.5.1: fast paths
            data = arr.ravel().tolist()
        elif arr is not None and arr.dtype.kind == "S":
            data = np.char.decode(arr.ravel(), "utf-8", "replace").tolist()
        else:
            data = [v.decode("utf-8", "replace") if isinstance(v, bytes)
                    else str(v) for v in values]
        # 0.5.1: one C-level search of the joined text instead of a Python
        # generator over every string (2.2 M iterations for the 30d set).
        if "\x00" in "".join(data):
            raise ValueError("Text datasets cannot contain NUL")
        self._put(name, "text", {"data": data})

    def SetData2D(self, name, data, xrange=None, yrange=None,
                  xedge=None, yedge=None, xcent=None, ycent=None):
        """Store a 2-D (image) dataset with optional ranges, edges or centres.

        ``data`` is copied as float64 and must be 2-D; ``xrange``/``yrange`` are
        (min, max) pairs, ``xedge``/``yedge`` bin edges and ``xcent``/``ycent``
        bin centres, exactly as Veusz's SetData2D takes them.
        """
        array = np.array(data, dtype=np.float64, copy=True)
        if array.ndim != 2:
            raise ValueError("%s requires a 2-D array" % name)
        parts = {"data": array}
        for key, value in (("xrange", xrange), ("yrange", yrange),
                           ("xedge", xedge), ("yedge", yedge),
                           ("xcent", xcent), ("ycent", ycent)):
            if value is not None:
                parts[key] = np.array(value, dtype=np.float64, copy=True)
        self._put(name, "2d", parts)

    def SetDataDateTime(self, name, seconds):
        """Date dataset: seconds since 2009-01-01 00:00:00 UTC (Veusz date-time)."""
        self.SetData(name, seconds)
        self.datasets[name] = ("date", self.datasets[name][1])

    def SetDataLocalTime(self, utc_name, local_name=None, zone=None, label="Local time"):
        """
        Local display copy of the UTC date dataset ``utc_name`` (0.3.0).

        Writes ``local_name`` (default "<utc_name>_local") as a date dataset
        on the processing computer's clock (or ``zone``; VSZH5_DISPLAY_TZ
        overrides), DST-aware, and remembers its offsets so a marked axis
        gets an ISO-8601 tick format with the offset.  Returns the name.
        """
        kind, parts = self.datasets[utc_name]
        if kind not in ("date", "1d"):
            raise ValueError("%s is not a date dataset" % utc_name)
        axis = vszh5_writer.local_time_from_veusz(parts["data"], zone=zone, label=label)
        local_name = local_name or "%s_local" % utc_name
        # 0.5.2: UTC + offset in Veusz seconds (local_veusz_seconds), not the
        # writer's round trip through Unix epoch seconds.
        self.SetDataDateTime(local_name, local_veusz_seconds(parts["data"], axis))
        self.local_axes[local_name] = axis
        self.local_of[utc_name] = local_name
        self.utc_of[local_name] = utc_name
        return local_name

    def local_axis_text(self, dataset_names, label="Local time"):
        """(tick format, axis label) of an axis showing these local datasets."""
        axes = [self.local_axes[n] for n in dataset_names if n in self.local_axes]
        if not axes:
            return vszh5_writer.ISO_DATETIME_FORMAT, label
        offsets = []
        for axis in axes:
            # 0.5.0: only instants that exist (a NaN time has offset 0).
            finite = axis.offsets_s[np.isfinite(axis.veusz_seconds) & np.isfinite(axis.offsets_s)]
            for value in finite.tolist():
                if value not in offsets:
                    offsets.append(value)
        offsets = offsets or [0.0]
        if len(offsets) == 1:
            text = vszh5_writer.iso_offset_text(offsets[0])
            return (vszh5_writer.ISO_DATETIME_FORMAT + text,
                    "%s (ISO 8601, UTC%s)" % (label, text))
        changes = []
        for axis in axes:
            for instant, offset in axis.changes:
                if (instant, offset) not in changes:
                    changes.append((instant, offset))
        changes.sort()
        # 0.5.0: compact for grid panels - the first offset, then each real
        # change of offset (every trace starts a "change" of its own, so
        # equal neighbours are merged) with the local minute it starts.
        parts, current = [], None
        for instant, offset in changes:
            if offset == current:
                continue
            parts.append("UTC%s" % offset if current is None
                         else "UTC%s from %s" % (offset, instant[:16]))
            current = offset
        return (vszh5_writer.ISO_DATETIME_FORMAT,
                "%s (ISO 8601; %s)" % (label, ", ".join(parts)))

    def finalize(self):
        """Name unnamed widgets, then lay out every time axis: local ISO 8601
        date-times, line breaks at gaps and (optionally) axis breaks (0.5.0;
        0.3.0 resolved marked local-time axes only)."""
        for widget in list(self.Root.WalkWidgets()):
            if widget.auto_named:
                _rename_from_data(widget)
        for graph in [w for w in self.Root.WalkWidgets() if w.kind == "graph"]:
            for axis in [c for c in graph.children if c.kind in ("axis", "axis-broken")]:
                if axis.settings.get("direction", "horizontal") != "vertical":
                    # 0.5.3: an x axis that is not time gets the gap options
                    # too (TimeLayout.numeric)
                    if not _layout_time_axis(self, graph, axis):
                        _layout_numeric_axis(self, graph, axis)
        _drop_unused_plot_copies(self)
        for page in [w for w in self.Root.children if w.kind == "page"]:
            _size_dated_page(page)
        for widget in self.Root.WalkWidgets():
            label = widget.local_time
            if label is None or widget.kind not in ("axis", "axis-broken"):
                continue
            if widget.time_layout_key is not None:      # done by _layout_time_axis
                continue
            graph = widget.parent
            names = []
            for child in graph.WalkWidgets() if graph is not None else ():
                if child.kind == "xy":
                    x_name = child.settings.get("xData")
                    if x_name in self.local_of:          # UTC date -> local copy
                        x_name = self.local_of[x_name]
                        child.settings["xData"] = x_name
                    if isinstance(x_name, str) and x_name not in names:
                        names.append(x_name)
            fmt, text = self.local_axis_text(names, label)
            widget.settings["mode"] = "datetime"
            widget.settings["TickLabels/format"] = fmt
            widget.settings.setdefault("TickLabels/rotate", str(vszh5_writer.LOCAL_DATETIME_ROTATE))
            widget.settings["label"] = text
        # 0.5.2: numeric axes show up to 12 significant digits.  Veusz's
        # automatic "%Vg" cuts tick labels to about 4 digits, so MJD 61000.25
        # read "6.1x10^4" and a 100 Hz span at 10 MHz showed the same label
        # on every tick in the exported PDF/PNG.  An explicit format, a log
        # axis or a hidden axis (Smith chart) is left alone.
        for widget in self.Root.WalkWidgets():
            if widget.kind in ("axis", "axis-broken") \
                    and widget.settings.get("mode", "numeric") == "numeric" \
                    and not widget.settings.get("log", False) \
                    and not widget.settings.get("hide", False) \
                    and "TickLabels/format" not in widget.settings:
                widget.settings["TickLabels/format"] = NUMERIC_TICK_FORMAT
        self._tag_untagged()

    def GetDataType(self, name):
        """veusz.embed-compatible data type of dataset ``name`` (0.5.2):
        '1d', '2d', 'text' or 'datetime'; KeyError when it does not exist."""
        kind = self.datasets[name][0]
        return {"date": "datetime"}.get(kind, kind)

    def _tag_untagged(self):
        """Give every dataset a ``source_*`` and a ``type_*`` tag (0.5.2).

        The programs tag their datasets themselves (source file and kind,
        _autoplot_common.tag_data); this is the safety net for any dataset
        they missed: ``source_<document name>`` and ``type_numeric`` /
        ``type_time`` / ``type_text`` / ``type_image`` by data type.
        """
        self._inherit_tags()
        with_source, with_type = set(), set()
        for tag, members in self.tags.items():
            if tag.startswith("source_"):
                with_source |= members
            elif tag.startswith("type_"):
                with_type |= members
        missing_source = [n for n in self.datasets if n not in with_source]
        if missing_source:
            word = re.sub(r"[^A-Za-z0-9_]+", "_", str(self.name)).strip("_") or "AutoPlot"
            self.tags.setdefault("source_" + word, set()).update(missing_source)
        kind_tag = {"1d": "type_numeric", "date": "type_time", "text": "type_text",
                    "2d": "type_image"}
        for name in self.datasets:
            if name not in with_type:
                self.tags.setdefault(kind_tag[self.datasets[name][0]], set()).add(name)

    def _inherit_tags(self):
        """Plot copies made by finalize() take the tags of their data (0.5.2).

        A local date-time copy "<utc>_local" gets the ``source_*`` tags and
        the non-time ``type_*`` tags of its UTC dataset plus
        ``type_time_local``; a line-break copy "<y>_gaps" gets every tag of
        ``<y>`` plus ``type_gap_rows``.
        """
        tags_of = {}
        for tag, members in self.tags.items():
            for member in members:
                tags_of.setdefault(member, set()).add(tag)
        copies = {}
        for local, utc in self.utc_of.items():
            copies[local] = (utc, "type_time_local")
        for name, owner in getattr(self, "_plot_copies", {}).items():
            origin = owner if isinstance(owner, str) else owner[-1]
            extra = "type_time_local" if "_local" in name else "type_gap_rows"
            copies.setdefault(name, (origin, extra))
        for name, (origin, extra) in copies.items():
            if name not in self.datasets:
                continue
            for tag in tags_of.get(origin, ()):
                if tag.startswith("source_") or (
                        tag.startswith("type_") and not (
                            extra == "type_time_local" and tag.startswith("type_time"))):
                    self.tags[tag].add(name)
            self.tags.setdefault(extra, set()).add(name)

    def TagDatasets(self, tag, names):
        """Attach ``tag`` to the existing datasets ``names`` (Veusz dataset tags).

        The tag must be a valid HDF5 name (no '/', NUL, '.' or '..'); unknown
        dataset names raise KeyError before anything is changed.
        """
        if not tag or "/" in tag or "\x00" in tag or tag in (".", ".."):
            raise ValueError("Unsupported HDF5 tag name: %r" % tag)
        names = set(names)
        # 0.5.1: dictionary lookups; set(self.datasets) copied every dataset
        # name on every call (O(n^2) over a batch).
        missing = [n for n in names if n not in self.datasets]
        if missing:
            raise KeyError("Unknown tagged datasets: %s" % sorted(missing))
        self.tags.setdefault(tag, set()).update(names)

    def GetDatasets(self):
        """Names of all datasets, in the order they were created."""
        return list(self.datasets)

    def GetData(self, name):
        """Arrays of dataset ``name``: (data, serr, nerr, perr) for numeric and
        date datasets, (data,) for text and 2-D datasets.
        """
        kind, parts = self.datasets[name]
        if kind in ("1d", "date"):
            return tuple(parts.get(k) for k in ("data", "serr", "nerr", "perr"))
        return (parts["data"],)

    def Get(self, path):
        """Value of the setting at ``path`` (``/page/graph/x/label``), or None.

        Walks the widget tree by name; the first path part that is not a child
        widget starts the setting path.
        """
        nodes = path.strip("/").split("/")
        widget = self.Root
        while nodes:
            match = next((c for c in widget.children if c.name == nodes[0]), None)
            if match is None:
                return widget.settings.get("/".join(nodes))
            widget = match
            nodes.pop(0)
        raise KeyError(path)

    def referenced_datasets(self):
        """Exact references from the model; no RPC, regex or destructive pruning."""
        used = set()
        for widget in self.Root.WalkWidgets():
            for key, value in widget.settings.items():
                if key.split("/")[-1] not in DATASET_SETTINGS:
                    continue
                for item in value if isinstance(value, (tuple, list)) else [value]:
                    if isinstance(item, str) and item:
                        if item not in self.datasets:
                            raise ValueError("Missing dataset %r at %s/%s" %
                                             (item, widget.path, key))
                        used.add(item)
        return used

    def SaveToString(self):
        """The Veusz command script, built with vszh5_writer.VszDoc."""
        doc = VszDoc(generator="AutoPlot _autoplot_native %s, vszh5_writer %s"
                     % (VERSION, WRITER_VERSION))
        # Compatibility level of the settings baseline (Veusz 3.4 / 4.1).
        doc.lines.append("SetCompatLevel(0)")
        self.finalize()
        self.Root._write(doc)
        return doc.text()

    GetSaveString = SaveToString

    def save(self, filename, minimized=False):
        """Write the document with vszh5_writer.VeuszWriter (atomic).

        The file is written to a private same-directory ".part" file and
        moved onto ``filename`` only when everything succeeded; any error
        deletes the partial file and leaves an existing ``filename`` as it
        was.  ``minimized`` keeps only the datasets the widgets use.
        """
        filename = os.fspath(filename)
        if not filename.lower().endswith(".vszh5"):
            filename = os.path.splitext(filename)[0] + ".vszh5"
        self.finalize()
        used = self.referenced_datasets()  # validate even a full save
        names = used if minimized else set(self.datasets)
        script = self.SaveToString()
        tags_of = {}
        for tag, members in self.tags.items():
            for member in members:
                tags_of.setdefault(member, []).append(tag)
        with VeuszWriter(filename, compress=True,
                         writer_tag="AutoPlot _autoplot_native %s" % VERSION) as writer:
            for name in sorted(names):
                kind, parts = self.datasets[name]
                tags = sorted(tags_of.get(name, ()))
                if kind == "1d":
                    writer.dataset_1d(name, parts["data"], serr=parts.get("serr"),
                                      perr=parts.get("perr"), nerr=parts.get("nerr"),
                                      tags=tags)
                elif kind == "date":
                    writer.dataset_date(name, parts["data"], tags=tags)
                elif kind == "text":
                    writer.dataset_text(name, parts["data"], tags=tags)
                elif kind == "2d":
                    axes = {k: v for k, v in parts.items() if k != "data"}
                    writer.dataset_2d(name, parts["data"], tags=tags, **axes)
                else:  # pragma: no cover - _put only stores the four kinds
                    raise ValueError("Unknown dataset kind %r" % kind)
            writer.finish(script)
        return filename

    def Save(self, filename, mode="hdf5"):
        """veusz.embed-compatible alias of ``save``; only mode 'hdf5' is supported.
        """
        if mode != "hdf5":
            raise ValueError("Only native HDF5 output is supported")
        return self.save(filename)

    def Close(self):
        """No external process to close; keep data alive until object disposal
        (release() empties the project at once)."""
