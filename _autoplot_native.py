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

# %%%% 0.5.9: Page kinds; combined-axis traces and their gap layout.
Date: 2026-11-17
#              Operator request 2026-11-17: Per file, Combined axis and
#              Overlay pages in every program; a combined page looks like a
#              file's own page but holds the same quantity of every file
#              joined end to end (the full range on each axis, not overlaid),
#              listed just above the overlays; gaps and the RAM limit apply.
#              * NEW PAGE_KINDS, PAGE_KIND_TEXT, DEFAULT_PAGE_KINDS; page_kinds()
#                (None or "all": all three; "none"; names separated by commas,
#                + or spaces; an unknown word is refused), describe_page_kinds(),
#                add_page_arguments() (--pages, checked by argparse) and
#                page_kinds_from_args().
#              * NEW concatenate_on_x(): the files' pieces joined in file order,
#                then a stable sort on x (equal x keep their file order, NaN x
#                last); nothing rounded; with_sources gives the file of every
#                row (file_index_array: uint8 / uint16).  NEW
#                overlapping_ranges() and move_pages_before() (combined pages
#                just above the overlays).
#              * NEW set_combined_sources() / combined_sources() /
#                combined_gap_positions(): the gap layout of a joined trace.
#                Each file is measured by its own rule and covers the runs
#                between its own gaps; a step of the joined trace is cut only
#                when no run of any file spans it and it is longer than the
#                larger threshold of the two files beside it (and larger on a
#                log scale for non-time axes); the axis breaks see the files as
#                an overlay of them would (the largest threshold of the files).
#                One median of the joined values cut every step of a coarse
#                file joined with a fine one: R&S, a 1 MHz span joined with
#                100 Hz and 20 Hz spans gave 800 axis breaks and an 18 m wide
#                page; UWBR Vacuum gave 3804 gap rows where its files have 278.
#              * _layout_time_axis / _layout_numeric_axis use it for every x
#                dataset with file numbers; other traces are laid out as before.
#              * NativeDocument.combined_of: the file numbers (1 or 2 bytes a
#                row, not saved), emptied by release() and moved to the offload
#                file with the datasets above the RAM limit (offload()).
#              Checked (Python 3.12.3): tests/test_page_kinds.py (22 tests)
#              and the whole suite; real files of every program with every
#              page-kind set, in RAM and offloaded (the files identical):
#              each combined trace equals the files' traces joined and
#              stable-sorted, the pages are in order (combined just above the
#              overlays) and the gaps are the same in the project, the Plots
#              tab and the saved file; with Combined axis off the projects are
#              the 0.4.8 ones (every dataset and script line).
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        local_veusz_seconds: Local wall-clock Veusz date seconds of the UTC Veusz date seconds ``utc_veusz``, with the per-sample offsets of the LocalTimeAxis ``axis`` (0.5.2):...
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        _gap_fraction: The exact seconds of a gap text (0.5.8): ``(Fraction, None)`` or ``(None, problem)``; check_gap_string has the syntax.
        check_gap_seconds: Read a gap text in seconds (0.5.8): ``(seconds, problem)``, the exact value of the text rounded once to a float (check_gap_string).
        check_gap_string: Read a gap text (0.5.8): ``(days, problem)``.
        describe_gap_seconds: A read gap for people (0.5.8): ``"15 min"``, ``"1.5 h"``, ``"not set"`` at 0 (TimeLayout's _seconds_text: exact seconds follow when the short form...
        describe_gap_days: describe_gap_seconds for a gap in days (0.5.8).
        parse_gap_seconds: parse_gap_string in seconds (0.5.8; exact, for TimeLayout manual_s / axis_manual_s): 0.0 for an entry that cannot be read, logged.
        parse_gap_string: Parse a human-readable gap string into MJD-days (check_gap_string).
        gap_text: argparse type of --gap-small / --gap-large (0.5.8): the text itself when check_gap_string() reads all of it, else ArgumentTypeError with the reason...
        _gap_factor: A finite number >= ``minimum`` for argparse (0.5.8), else ArgumentTypeError.
        gap_factor_k: argparse type of --gap-k (0.5.8): a finite K >= MIN_GAP_K.
        gap_factor_n: argparse type of --gap-n (0.5.8): a finite n >= 0 (0 = no axis breaks).
        add_gap_arguments: Add the gap options to an argparse parser (0.5.3): --gap-k, --gap-small, --gap-n, --gap-large, --no-line-breaks, --no-axis-breaks, --no-numeric-gap...
        add_memory_arguments: Add --ram-limit to an argparse parser (0.5.7): offload plot data to disk when this computer's RAM in use (all programs) reaches PERCENT (0 = never;...
        ram_limit_from_args: Set the program's RAM limit from add_memory_arguments() (0.5.7); returns it.
        layout_from_args: The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3; 0.5.6: --time-axes; 0.5.8: ``log_cb`` reports a gap text that cannot be read...
        page_kinds: The page kinds asked for (0.5.9), in PAGE_KINDS order.
        describe_page_kinds: Log line of the page kinds (0.5.9): "Pages: Per file, Combined axis, Overlay".
        add_page_arguments: Add --pages to an argparse parser (0.5.9): which page kinds to draw (per_file, combined, overlay; default all); page_kinds_from_args() reads it.
        page_kinds_from_args: The page kinds of add_page_arguments() in ``args`` (0.5.9; all when absent).
        concatenate_on_x: Join the traces of several files into one (0.5.9, combined-axis pages).
        file_index_array: The file (0, 1, ...) of every row of pieces of ``rows`` rows each, joined in that order (0.5.9), in the smallest unsigned integer type (uint8 below...
        set_combined_sources: Record which file each row of the combined-axis x datasets ``names`` came from (0.5.9): ``sources`` holds one file number per row, in the rows' (so...
        combined_sources: The file number of every row of the combined x dataset ``name`` (0.5.9; set_combined_sources), or None when ``name`` is not a joined trace or its l...
        combined_gap_positions: Gap rows of a joined trace (0.5.9): ``values`` are its x values in the gap rule's units (sorted; NaN allowed), ``sources`` the file of every row.
        overlapping_ranges: How many of the files' x ranges overlap an earlier one (0.5.9): the combined trace interleaves their samples there.
        move_pages_before: Put ``pages`` (pages of ``doc``) just before the page ``anchor`` (0.5.9: combined pages just above the overlays, in the project and the Plots tab).
        time_clocks: The clocks of ``value`` as a tuple in TIME_CLOCKS order (0.5.6).
        _zone_of: The tzinfo of ``zone`` (None = VSZH5_DISPLAY_TZ, else None = this computer's zone; a text is read as by the writer) (0.5.6).
        zone_name_at: Name of the time zone at the UTC instant ``epoch_s`` (Unix seconds) (0.5.6): what the operating system calls this computer's zone then ("EST", "EDT...
        label_zone_name: ``name`` as the axis label shows it, or "" for a name that only repeats the offset or says UTC (0.5.6).
        local_zone_description: One line naming the zone local time axes use (0.5.6), for the GUI and the logs: this computer's zone (or VSZH5_DISPLAY_TZ) with its name and UTC of...
        _finite_seconds: A fixed gap in seconds (0.5.8): finite and > 0, else 0.0 (not set).
        TimeLayout (class): Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any sorted x, and the two consequences of a gap made optional), see vszh5_writer T...
        _number_text: A setting for the notes with every digit (repr), without a trailing ".0" (0.5.3; %g cut 12.3456789 to 12.3457).
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _page_of: The page widget ``widget`` is on, or None (0.5.6).
        _clock_of: The time-axis clock of the page ``widget`` is on ("local" when the page has none yet, or the widget is on no page) (0.5.6).
        _same_values: True when two arrays hold the same numbers, NaN where the other has NaN (0.5.6; numpy.array_equal(equal_nan=...) needs numpy >= 1.19).
        _mjd_name: Name of the MJD plot copy of the UTC date dataset ``utc_name`` (0.5.6): "<x>__utc" -> "<x>__mjd", "<x>_utc" -> "<x>_mjd", else "<x>_mjd".
        _exact_mjd: (MJD values, name of a stored dataset holding exactly them or None) of one time x (0.5.6).
        _clock_dataset: Plot x dataset of one trace on the UTC or MJD clock (0.5.6).
        _time_note: The "Time axis:" line of the graph notes for ``clock`` (0.5.6).
        _set_note_line: Put ``line`` into the graph notes, replacing a line that starts with ``prefix`` (0.5.6; _set_gap_note for any prefix).
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0; 0.5.6 clocks).
        _clock_break_pairs: Axis breaks found in UTC seconds, moved onto another clock's axis (0.5.8).
        _set_gap_note: Put the gap-rule line ``note`` into the graph notes (0.5.3: a line from an earlier layout is replaced, so the notes state the rule in use).
        _sorted_finite: True when the finite values of ``values`` never decrease (at least 3).
        _layout_numeric_axis: Gap options on an x axis that is not time (0.5.3): frequency, angle, index ...
        _undo_numeric_layout: Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the gap options of a project turn non-time axes off between two saves, the traces d...
        _drop_unused_plot_copies: Remove line-break plot copies no widget draws any more (0.5.3): when the gap options change between two saves, copies of the old layout would other...
        _has_clock_axis: True when ``page`` draws a date-time x axis laid out on a clock.
        _free_page_name: ``wanted``, or wanted_2, wanted_3 ...
        _copy_setting: A setting value for a page copy: lists and dicts copied, the rest (text, numbers, booleans) shared.
        _clone_widget: Copy of ``widget`` and every widget below it, under ``parent`` (0.5.6).
        _remove_clock_pages: Take the clock copies of the last finalize out of the project (0.5.6); finalize makes them again from their pages, so a copy always shows its page...
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        _fit_set: Set ``key`` on ``widget`` for the tick-label fit, remembering the value it had before (0.5.5).
        _fit_restore: Undo the previous tick-label fit on ``widget`` (0.5.5): a setting the fit wrote and nobody changed since gets its earlier value back.
        _cm: A Veusz distance setting in cm (``default`` when unset or not physical).
        _pt: A Veusz font size setting in points (``default`` when unset).
        _finite_range: (min, max) of the finite values of dataset ``name`` (1-D or date), or None; cached per finalize in ``cache``.
        _axis_data_range: (min, max) of the data a graph draws on ``axis`` (xy traces and images), or None when it draws nothing known (0.5.5).
        _axis_room: vszh5_writer.TickRoom of one axis as it is set now (0.5.5); None for a hidden axis, a label axis or an axis without known data.
        _time_label_cm: Width (cm) a date-time x axis's label needs (0.5.6), 0 for other axes.
        _dated_page_size: (width, height) cm the 0.5.0 rule gives a page with date-time axes and no size of its own: 13 cm a column and 8 cm a row (at least 16 cm) for a gri...
        _fit_page: Fit the tick labels of every graph on ``page`` (0.5.5): vertical labels where they belong, the margins every label needs and a page large enough (s...
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
        _OffloadNotes (class): Throttled offload messages (0.5.7): the first at once, then at most one every _NOTE_S with the totals so far; flush() logs the last one held back (...
        set_ram_limit: Set the RAM limit (% of this computer's RAM in use, every program) above which plot data are offloaded to disk (0.5.7); 0 or None = never.
        ram_limit: The RAM limit in force (%; 0 = never offload) (0.5.7).
        system_memory: (percent in use, total bytes, available bytes) of this computer's RAM (0.5.7), all programs together; None when it cannot be read.
        memory_text: One line for the GUI (0.5.7), e.g.
        payload_bytes: Bytes of the values in ``value`` (0.5.7): numpy arrays by nbytes, text and lists estimated, dicts / lists / tuples summed (nested to 4 levels).
        OffloadStore (class): One append-only pickle file of offloaded values (0.5.7).
        OffloadedValue (class): A value in an OffloadStore (0.5.7); load() reads it back (a new copy every time; nothing is kept in RAM).
        OffloadedParts (class): The arrays of a dataset kept in an OffloadStore (0.5.7): a read-only mapping like the dict it replaces ({"data": ..., "serr": ...}).
        loaded_parts: ``parts`` as a plain dict of arrays (0.5.7): offloaded parts are read back once; parts in RAM are returned as they are.
        _over_ram_limit: (percent in use, limit) when this computer's RAM use is at or above ``limit`` (%), else None; None too when the limit is 0 or the use is unknown (0...
        file_stamp: (size, mtime_ns, options) of ``path``: the data read from it are still good while this is unchanged; None when the file cannot be read.
        KeptReads (class): Data read from the listed files, kept until the user clears them (0.5.4).
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        _GAP_NUMBER: a number of a gap text: digits, decimals, an exponent of up to 3 digits (0.5.8)
        _GAP_TOKEN_REGEX: one number and its unit of a gap text, spaces allowed (0.5.8: matched one after another over the whole entry)
        _GAP_NUMBER_REGEX: a gap text that is a bare number (hours) (0.5.8)
        _GAP_CLOCK_REGEX: a gap text that is a clock H:MM or H:MM:SS(.s) (0.5.8)
        GAP_STRING_HELP: tool tip of the gap text boxes
        _DAYS_PER_MONTH: average Gregorian month (365.25 / 12 days) for 'm' in gap texts
        _DAYS_PER_HOUR: 1 / 24 (gap text parsing)
        _DAYS_PER_MINUTE: 1 / 1440 (gap text parsing)
        _DAYS_PER_SECOND: 1 / 86400 (gap text parsing)
        _GAP_UNIT_SECONDS: seconds per unit of a gap text, every accepted spelling; m = month = 2 629 800 s (0.5.8)
        _GAP_HINT: short unit list a log line about an unread gap text ends with (0.5.8)
        PAGE_KINDS: the page kinds a program can draw, in page order (per_file, combined, overlay)
        PAGE_KIND_TEXT: names of the page kinds for people (GUI, logs, notes)
        DEFAULT_PAGE_KINDS: default page kinds: all three
        _PAGE_KIND_WORDS: the words page_kinds() accepts for each kind
        TIME_CLOCKS: the clocks of date-time x axes, in page order (local, utc, mjd)
        DEFAULT_TIME_CLOCKS: default clocks: local time only
        CLOCK_PAGE_SUFFIX: page-name suffix of each clock ('' for local)
        CLOCK_TITLES: clock names for people (GUI, logs, notes)
        UTC_AXIS_LABEL: axis label of a UTC clock axis
        MJD_AXIS_LABEL: axis label of an MJD clock axis
        UTC_DATETIME_FORMAT: tick format of a UTC date-time axis (ISO 8601, 'Z')
        _OFFSET_ONLY_NAME: zone names that only repeat the offset or say UTC (left off labels)
        MIN_GAP_K: smallest K of the small-gap rule (0.5.8)
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _TIME_NOTE: start of the graph-notes line written by the time layout
        _MISSING: Every page is fitted on save, after the time and gap layouts: each axis's tick labels are worked out the way Veusz will draw them (vszh5_writer.axi...
        _LABEL_PAD_CM: room kept beside a time-axis label that sets the plot width (cm)
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
        _TAG_WORD: characters not allowed in a tag word (replaced by '_')
        MISSING_VALUE_WORDS: cell texts that mean no value (pandas' NA strings): no exact-text copy needed
        NUMERIC_TICK_FORMAT: tick-label format of numeric axes: up to 12 significant digits
        DEFAULT_RAM_LIMIT_PERCENT: default RAM limit: offload above this much of the computer's RAM in use (%)
        OFFLOAD_DIR_ENV: environment variable naming the folder for offload files
        OFFLOAD_MIN_BYTES: datasets smaller than this (bytes of values) stay in RAM
        _RAM_CHECK_S: RAM use is read at most this often while datasets are added (s)
        _HEAP_TRIM_S: freed heap is handed back to the system at most this often (s)
        _RAM_LIMIT: the program-wide RAM limit (set_ram_limit)
        _NOTE_S: (0.5.7) Offload messages are logged at most this often (s), with totals.
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
    Tick labels: finalize() fits every page
    (_fit_page): vertical two-line date labels and long numbers, margins and
    page sizes so Veusz draws every major tick label; undone before the next fit.
    Clocks: TimeLayout(clocks=...)
    draws date-time x axes in local time, UTC and / or MJD, one set of pages per
    clock, rebuilt on every save (see "Time-axis clocks").
    Offload: above the RAM limit
    (set_ram_limit) datasets and kept file data go to a pickle file
    (OffloadStore) and are read back where used (see "Offloading").
    Page kinds and combined-axis
    traces: see "Page kinds" (page_kinds, concatenate_on_x, combined_gap_positions).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.8: Gap options enforced: gap texts read whole and exactly, DST.
Date: 2026-11-17
#              Operator request 2026-11-17: recheck that the gap-aware
#              settings are enforced.
#              * parse_gap_string: the whole entry must be read, token after
#                token (before, tokens were found anywhere: "00:10:00" was
#                10 h, "1x2" 3 h, ".5h" 5 h); NEW clocks H:MM / H:MM:SS(.s),
#                long unit names (sec, minutes, hours, days, months) and
#                exponents; a bare number alone is still hours; an entry that
#                cannot be read gives 0 and a log line naming the entry, the
#                reason and the gap (``what``).
#              * The text is read exactly (fractions.Fraction) and rounded
#                once: "16min" is 960 s (the day-based sum gave
#                959.9999999999999 s, so a 960 s step was cut).  NEW
#                check_gap_string / check_gap_seconds / parse_gap_seconds /
#                describe_gap_seconds / describe_gap_days; TimeLayout gets
#                seconds everywhere (layout_from_args too).
#              * --gap-small / --gap-large / --gap-k / --gap-n are checked when
#                the command line is read (gap_text, gap_factor_k,
#                gap_factor_n): argparse stops with the reason instead of
#                ignoring the entry; layout_from_args(log_cb).
#              * TimeLayout: K >= 1 (MIN_GAP_K; a K below 1 cut ordinary
#                steps), K not a number = the default, n not finite = 0, fixed
#                gaps not finite = 0.
#              * Local clock: axis breaks found in UTC seconds and moved onto
#                the local axis (_clock_break_pairs): a spring-forward hour is
#                no gap any more and a hole over the fall-back is broken as on
#                the UTC and MJD pages; no sample is ever hidden.  Without a
#                daylight-saving change the breaks are the same as before.
#              Checked (Python 3.12.3): tests/test_gap_enforcement.py (20 tests)
#              and the whole suite; every option set before a run and changed
#              after it, in CSV/TSV (a file with known gaps, 64 cases at the
#              boundaries), FITS (GBT tack, UWBR Vacuum, 30-day 1PPS), ATR,
#              R&S and Touchstone: the project, the Plots tab and the saved
#              file agree; real files without a daylight-saving change give
#              the same .vszh5 as 0.4.7 (every dataset and script line).
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        local_veusz_seconds: Local wall-clock Veusz date seconds of the UTC Veusz date seconds ``utc_veusz``, with the per-sample offsets of the LocalTimeAxis ``axis`` (0.5.2):...
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        _gap_fraction: The exact seconds of a gap text (0.5.8): ``(Fraction, None)`` or ``(None, problem)``; check_gap_string has the syntax.
        check_gap_seconds: Read a gap text in seconds (0.5.8): ``(seconds, problem)``, the exact value of the text rounded once to a float (check_gap_string).
        check_gap_string: Read a gap text (0.5.8): ``(days, problem)``.
        describe_gap_seconds: A read gap for people (0.5.8): ``"15 min"``, ``"1.5 h"``, ``"not set"`` at 0 (TimeLayout's _seconds_text: exact seconds follow when the short form...
        describe_gap_days: describe_gap_seconds for a gap in days (0.5.8).
        parse_gap_seconds: parse_gap_string in seconds (0.5.8; exact, for TimeLayout manual_s / axis_manual_s): 0.0 for an entry that cannot be read, logged.
        parse_gap_string: Parse a human-readable gap string into MJD-days (check_gap_string).
        gap_text: argparse type of --gap-small / --gap-large (0.5.8): the text itself when check_gap_string() reads all of it, else ArgumentTypeError with the reason...
        _gap_factor: A finite number >= ``minimum`` for argparse (0.5.8), else ArgumentTypeError.
        gap_factor_k: argparse type of --gap-k (0.5.8): a finite K >= MIN_GAP_K.
        gap_factor_n: argparse type of --gap-n (0.5.8): a finite n >= 0 (0 = no axis breaks).
        add_gap_arguments: Add the gap options to an argparse parser (0.5.3): --gap-k, --gap-small, --gap-n, --gap-large, --no-line-breaks, --no-axis-breaks, --no-numeric-gap...
        add_memory_arguments: Add --ram-limit to an argparse parser (0.5.7): offload plot data to disk when this computer's RAM in use (all programs) reaches PERCENT (0 = never;...
        ram_limit_from_args: Set the program's RAM limit from add_memory_arguments() (0.5.7); returns it.
        layout_from_args: The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3; 0.5.6: --time-axes; 0.5.8: ``log_cb`` reports a gap text that cannot be read...
        time_clocks: The clocks of ``value`` as a tuple in TIME_CLOCKS order (0.5.6).
        _zone_of: The tzinfo of ``zone`` (None = VSZH5_DISPLAY_TZ, else None = this computer's zone; a text is read as by the writer) (0.5.6).
        zone_name_at: Name of the time zone at the UTC instant ``epoch_s`` (Unix seconds) (0.5.6): what the operating system calls this computer's zone then ("EST", "EDT...
        label_zone_name: ``name`` as the axis label shows it, or "" for a name that only repeats the offset or says UTC (0.5.6).
        local_zone_description: One line naming the zone local time axes use (0.5.6), for the GUI and the logs: this computer's zone (or VSZH5_DISPLAY_TZ) with its name and UTC of...
        _finite_seconds: A fixed gap in seconds (0.5.8): finite and > 0, else 0.0 (not set).
        TimeLayout (class): Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any sorted x, and the two consequences of a gap made optional), see vszh5_writer T...
        _number_text: A setting for the notes with every digit (repr), without a trailing ".0" (0.5.3; %g cut 12.3456789 to 12.3457).
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _page_of: The page widget ``widget`` is on, or None (0.5.6).
        _clock_of: The time-axis clock of the page ``widget`` is on ("local" when the page has none yet, or the widget is on no page) (0.5.6).
        _same_values: True when two arrays hold the same numbers, NaN where the other has NaN (0.5.6; numpy.array_equal(equal_nan=...) needs numpy >= 1.19).
        _mjd_name: Name of the MJD plot copy of the UTC date dataset ``utc_name`` (0.5.6): "<x>__utc" -> "<x>__mjd", "<x>_utc" -> "<x>_mjd", else "<x>_mjd".
        _exact_mjd: (MJD values, name of a stored dataset holding exactly them or None) of one time x (0.5.6).
        _clock_dataset: Plot x dataset of one trace on the UTC or MJD clock (0.5.6).
        _time_note: The "Time axis:" line of the graph notes for ``clock`` (0.5.6).
        _set_note_line: Put ``line`` into the graph notes, replacing a line that starts with ``prefix`` (0.5.6; _set_gap_note for any prefix).
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0; 0.5.6 clocks).
        _clock_break_pairs: Axis breaks found in UTC seconds, moved onto another clock's axis (0.5.8).
        _set_gap_note: Put the gap-rule line ``note`` into the graph notes (0.5.3: a line from an earlier layout is replaced, so the notes state the rule in use).
        _sorted_finite: True when the finite values of ``values`` never decrease (at least 3).
        _layout_numeric_axis: Gap options on an x axis that is not time (0.5.3): frequency, angle, index ...
        _undo_numeric_layout: Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the gap options of a project turn non-time axes off between two saves, the traces d...
        _drop_unused_plot_copies: Remove line-break plot copies no widget draws any more (0.5.3): when the gap options change between two saves, copies of the old layout would other...
        _has_clock_axis: True when ``page`` draws a date-time x axis laid out on a clock.
        _free_page_name: ``wanted``, or wanted_2, wanted_3 ...
        _copy_setting: A setting value for a page copy: lists and dicts copied, the rest (text, numbers, booleans) shared.
        _clone_widget: Copy of ``widget`` and every widget below it, under ``parent`` (0.5.6).
        _remove_clock_pages: Take the clock copies of the last finalize out of the project (0.5.6); finalize makes them again from their pages, so a copy always shows its page...
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        _fit_set: Set ``key`` on ``widget`` for the tick-label fit, remembering the value it had before (0.5.5).
        _fit_restore: Undo the previous tick-label fit on ``widget`` (0.5.5): a setting the fit wrote and nobody changed since gets its earlier value back.
        _cm: A Veusz distance setting in cm (``default`` when unset or not physical).
        _pt: A Veusz font size setting in points (``default`` when unset).
        _finite_range: (min, max) of the finite values of dataset ``name`` (1-D or date), or None; cached per finalize in ``cache``.
        _axis_data_range: (min, max) of the data a graph draws on ``axis`` (xy traces and images), or None when it draws nothing known (0.5.5).
        _axis_room: vszh5_writer.TickRoom of one axis as it is set now (0.5.5); None for a hidden axis, a label axis or an axis without known data.
        _time_label_cm: Width (cm) a date-time x axis's label needs (0.5.6), 0 for other axes.
        _dated_page_size: (width, height) cm the 0.5.0 rule gives a page with date-time axes and no size of its own: 13 cm a column and 8 cm a row (at least 16 cm) for a gri...
        _fit_page: Fit the tick labels of every graph on ``page`` (0.5.5): vertical labels where they belong, the margins every label needs and a page large enough (s...
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
        _OffloadNotes (class): Throttled offload messages (0.5.7): the first at once, then at most one every _NOTE_S with the totals so far; flush() logs the last one held back (...
        set_ram_limit: Set the RAM limit (% of this computer's RAM in use, every program) above which plot data are offloaded to disk (0.5.7); 0 or None = never.
        ram_limit: The RAM limit in force (%; 0 = never offload) (0.5.7).
        system_memory: (percent in use, total bytes, available bytes) of this computer's RAM (0.5.7), all programs together; None when it cannot be read.
        memory_text: One line for the GUI (0.5.7), e.g.
        payload_bytes: Bytes of the values in ``value`` (0.5.7): numpy arrays by nbytes, text and lists estimated, dicts / lists / tuples summed (nested to 4 levels).
        OffloadStore (class): One append-only pickle file of offloaded values (0.5.7).
        OffloadedValue (class): A value in an OffloadStore (0.5.7); load() reads it back (a new copy every time; nothing is kept in RAM).
        OffloadedParts (class): The arrays of a dataset kept in an OffloadStore (0.5.7): a read-only mapping like the dict it replaces ({"data": ..., "serr": ...}).
        loaded_parts: ``parts`` as a plain dict of arrays (0.5.7): offloaded parts are read back once; parts in RAM are returned as they are.
        _over_ram_limit: (percent in use, limit) when this computer's RAM use is at or above ``limit`` (%), else None; None too when the limit is 0 or the use is unknown (0...
        file_stamp: (size, mtime_ns, options) of ``path``: the data read from it are still good while this is unchanged; None when the file cannot be read.
        KeptReads (class): Data read from the listed files, kept until the user clears them (0.5.4).
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        _GAP_NUMBER: a number of a gap text: digits, decimals, an exponent of up to 3 digits (0.5.8)
        _GAP_TOKEN_REGEX: one number and its unit of a gap text, spaces allowed (0.5.8: matched one after another over the whole entry)
        _GAP_NUMBER_REGEX: a gap text that is a bare number (hours) (0.5.8)
        _GAP_CLOCK_REGEX: a gap text that is a clock H:MM or H:MM:SS(.s) (0.5.8)
        GAP_STRING_HELP: tool tip of the gap text boxes
        _DAYS_PER_MONTH: average Gregorian month (365.25 / 12 days) for 'm' in gap texts
        _DAYS_PER_HOUR: 1 / 24 (gap text parsing)
        _DAYS_PER_MINUTE: 1 / 1440 (gap text parsing)
        _DAYS_PER_SECOND: 1 / 86400 (gap text parsing)
        _GAP_UNIT_SECONDS: seconds per unit of a gap text, every accepted spelling; m = month = 2 629 800 s (0.5.8)
        _GAP_HINT: short unit list a log line about an unread gap text ends with (0.5.8)
        TIME_CLOCKS: the clocks of date-time x axes, in page order (local, utc, mjd)
        DEFAULT_TIME_CLOCKS: default clocks: local time only
        CLOCK_PAGE_SUFFIX: page-name suffix of each clock ('' for local)
        CLOCK_TITLES: clock names for people (GUI, logs, notes)
        UTC_AXIS_LABEL: axis label of a UTC clock axis
        MJD_AXIS_LABEL: axis label of an MJD clock axis
        UTC_DATETIME_FORMAT: tick format of a UTC date-time axis (ISO 8601, 'Z')
        _OFFSET_ONLY_NAME: zone names that only repeat the offset or say UTC (left off labels)
        MIN_GAP_K: smallest K of the small-gap rule (0.5.8)
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _TIME_NOTE: start of the graph-notes line written by the time layout
        _MISSING: Every page is fitted on save, after the time and gap layouts: each axis's tick labels are worked out the way Veusz will draw them (vszh5_writer.axi...
        _LABEL_PAD_CM: room kept beside a time-axis label that sets the plot width (cm)
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
        _TAG_WORD: characters not allowed in a tag word (replaced by '_')
        MISSING_VALUE_WORDS: cell texts that mean no value (pandas' NA strings): no exact-text copy needed
        NUMERIC_TICK_FORMAT: tick-label format of numeric axes: up to 12 significant digits
        DEFAULT_RAM_LIMIT_PERCENT: default RAM limit: offload above this much of the computer's RAM in use (%)
        OFFLOAD_DIR_ENV: environment variable naming the folder for offload files
        OFFLOAD_MIN_BYTES: datasets smaller than this (bytes of values) stay in RAM
        _RAM_CHECK_S: RAM use is read at most this often while datasets are added (s)
        _HEAP_TRIM_S: freed heap is handed back to the system at most this often (s)
        _RAM_LIMIT: the program-wide RAM limit (set_ram_limit)
        _NOTE_S: (0.5.7) Offload messages are logged at most this often (s), with totals.
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
    Tick labels: finalize() fits every page
    (_fit_page): vertical two-line date labels and long numbers, margins and
    page sizes so Veusz draws every major tick label; undone before the next fit.
    Clocks: TimeLayout(clocks=...)
    draws date-time x axes in local time, UTC and / or MJD, one set of pages per
    clock, rebuilt on every save (see "Time-axis clocks").
    Offload: above the RAM limit
    (set_ram_limit) datasets and kept file data go to a pickle file
    (OffloadStore) and are read back where used (see "Offloading").
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.7: Plot data offloaded to disk when this computer's RAM use passes a limit.
Date: 2026-11-17
#              Operator request 2026-11-17: offload to disk (pickle) when a
#              user value for RAM use - of the whole computer, not only this
#              program - is exceeded, and convert to the .vszh5 on save; only
#              above that value.
#              * NEW set_ram_limit() / ram_limit() (default 80 %, 0 = never),
#                system_memory() (psutil; /proc/meminfo; GlobalMemoryStatusEx),
#                memory_text(), payload_bytes(); OffloadStore (one append-only
#                pickle file, thread-safe, deleted on close or exit),
#                OffloadedValue, OffloadedParts (a read-only mapping in place
#                of a dataset's parts), loaded_parts().
#              * NativeDocument: after each dataset is stored (at most every
#                0.2 s), at the end of finalize() and on save, the RAM in use
#                on the computer is read; at or above the limit every dataset
#                of 512 bytes or more and the exact MJD arrays move to the
#                file (offload(); not when the disk lacks room), with a
#                message (offload_log, offload_messages).  Layouts, the Plots
#                tab and GetData read them back; save() writes each from the
#                file into the .vszh5; release() deletes the file.
#              * KeptReads: keep() offloads the kept file data above the limit;
#                get() / items() read them back; data that cannot be pickled
#                stay in RAM; clear() deletes the file.
#              * NEW add_memory_arguments() / ram_limit_from_args(): --ram-limit.
#              Checked (Python 3.12.3): tests/test_offload.py (16 tests) and the
#              whole suite; every program's real files (FITS GBT tack, UWBR
#              Temps / Vacuum, ATR, R&S, CSV, Touchstone) saved with every
#              dataset offloaded: identical to the in-RAM save in every
#              dataset, attribute and script line; Plots tab identical.
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
        add_memory_arguments: Add --ram-limit to an argparse parser (0.5.7): offload plot data to disk when this computer's RAM in use (all programs) reaches PERCENT (0 = never;...
        ram_limit_from_args: Set the program's RAM limit from add_memory_arguments() (0.5.7); returns it.
        layout_from_args: The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3; 0.5.6: --time-axes).
        time_clocks: The clocks of ``value`` as a tuple in TIME_CLOCKS order (0.5.6).
        _zone_of: The tzinfo of ``zone`` (None = VSZH5_DISPLAY_TZ, else None = this computer's zone; a text is read as by the writer) (0.5.6).
        zone_name_at: Name of the time zone at the UTC instant ``epoch_s`` (Unix seconds) (0.5.6): what the operating system calls this computer's zone then ("EST", "EDT...
        label_zone_name: ``name`` as the axis label shows it, or "" for a name that only repeats the offset or says UTC (0.5.6).
        local_zone_description: One line naming the zone local time axes use (0.5.6), for the GUI and the logs: this computer's zone (or VSZH5_DISPLAY_TZ) with its name and UTC of...
        TimeLayout (class): Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any sorted x, and the two consequences of a gap made optional), see vszh5_writer T...
        _number_text: A setting for the notes with every digit (repr), without a trailing ".0" (0.5.3; %g cut 12.3456789 to 12.3457).
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _page_of: The page widget ``widget`` is on, or None (0.5.6).
        _clock_of: The time-axis clock of the page ``widget`` is on ("local" when the page has none yet, or the widget is on no page) (0.5.6).
        _same_values: True when two arrays hold the same numbers, NaN where the other has NaN (0.5.6; numpy.array_equal(equal_nan=...) needs numpy >= 1.19).
        _mjd_name: Name of the MJD plot copy of the UTC date dataset ``utc_name`` (0.5.6): "<x>__utc" -> "<x>__mjd", "<x>_utc" -> "<x>_mjd", else "<x>_mjd".
        _exact_mjd: (MJD values, name of a stored dataset holding exactly them or None) of one time x (0.5.6).
        _clock_dataset: Plot x dataset of one trace on the UTC or MJD clock (0.5.6).
        _time_note: The "Time axis:" line of the graph notes for ``clock`` (0.5.6).
        _set_note_line: Put ``line`` into the graph notes, replacing a line that starts with ``prefix`` (0.5.6; _set_gap_note for any prefix).
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0; 0.5.6 clocks).
        _set_gap_note: Put the gap-rule line ``note`` into the graph notes (0.5.3: a line from an earlier layout is replaced, so the notes state the rule in use).
        _sorted_finite: True when the finite values of ``values`` never decrease (at least 3).
        _layout_numeric_axis: Gap options on an x axis that is not time (0.5.3): frequency, angle, index ...
        _undo_numeric_layout: Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the gap options of a project turn non-time axes off between two saves, the traces d...
        _drop_unused_plot_copies: Remove line-break plot copies no widget draws any more (0.5.3): when the gap options change between two saves, copies of the old layout would other...
        _has_clock_axis: True when ``page`` draws a date-time x axis laid out on a clock.
        _free_page_name: ``wanted``, or wanted_2, wanted_3 ...
        _copy_setting: A setting value for a page copy: lists and dicts copied, the rest (text, numbers, booleans) shared.
        _clone_widget: Copy of ``widget`` and every widget below it, under ``parent`` (0.5.6).
        _remove_clock_pages: Take the clock copies of the last finalize out of the project (0.5.6); finalize makes them again from their pages, so a copy always shows its page...
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        _fit_set: Set ``key`` on ``widget`` for the tick-label fit, remembering the value it had before (0.5.5).
        _fit_restore: Undo the previous tick-label fit on ``widget`` (0.5.5): a setting the fit wrote and nobody changed since gets its earlier value back.
        _cm: A Veusz distance setting in cm (``default`` when unset or not physical).
        _pt: A Veusz font size setting in points (``default`` when unset).
        _finite_range: (min, max) of the finite values of dataset ``name`` (1-D or date), or None; cached per finalize in ``cache``.
        _axis_data_range: (min, max) of the data a graph draws on ``axis`` (xy traces and images), or None when it draws nothing known (0.5.5).
        _axis_room: vszh5_writer.TickRoom of one axis as it is set now (0.5.5); None for a hidden axis, a label axis or an axis without known data.
        _time_label_cm: Width (cm) a date-time x axis's label needs (0.5.6), 0 for other axes.
        _dated_page_size: (width, height) cm the 0.5.0 rule gives a page with date-time axes and no size of its own: 13 cm a column and 8 cm a row (at least 16 cm) for a gri...
        _fit_page: Fit the tick labels of every graph on ``page`` (0.5.5): vertical labels where they belong, the margins every label needs and a page large enough (s...
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
        _OffloadNotes (class): Throttled offload messages (0.5.7): the first at once, then at most one every _NOTE_S with the totals so far; flush() logs the last one held back (...
        set_ram_limit: Set the RAM limit (% of this computer's RAM in use, every program) above which plot data are offloaded to disk (0.5.7); 0 or None = never.
        ram_limit: The RAM limit in force (%; 0 = never offload) (0.5.7).
        system_memory: (percent in use, total bytes, available bytes) of this computer's RAM (0.5.7), all programs together; None when it cannot be read.
        memory_text: One line for the GUI (0.5.7), e.g.
        payload_bytes: Bytes of the values in ``value`` (0.5.7): numpy arrays by nbytes, text and lists estimated, dicts / lists / tuples summed (nested to 4 levels).
        OffloadStore (class): One append-only pickle file of offloaded values (0.5.7).
        OffloadedValue (class): A value in an OffloadStore (0.5.7); load() reads it back (a new copy every time; nothing is kept in RAM).
        OffloadedParts (class): The arrays of a dataset kept in an OffloadStore (0.5.7): a read-only mapping like the dict it replaces ({"data": ..., "serr": ...}).
        loaded_parts: ``parts`` as a plain dict of arrays (0.5.7): offloaded parts are read back once; parts in RAM are returned as they are.
        _over_ram_limit: (percent in use, limit) when this computer's RAM use is at or above ``limit`` (%), else None; None too when the limit is 0 or the use is unknown (0...
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
        TIME_CLOCKS: the clocks of date-time x axes, in page order (local, utc, mjd)
        DEFAULT_TIME_CLOCKS: default clocks: local time only
        CLOCK_PAGE_SUFFIX: page-name suffix of each clock ('' for local)
        CLOCK_TITLES: clock names for people (GUI, logs, notes)
        UTC_AXIS_LABEL: axis label of a UTC clock axis
        MJD_AXIS_LABEL: axis label of an MJD clock axis
        UTC_DATETIME_FORMAT: tick format of a UTC date-time axis (ISO 8601, 'Z')
        _OFFSET_ONLY_NAME: zone names that only repeat the offset or say UTC (left off labels)
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _TIME_NOTE: start of the graph-notes line written by the time layout
        _MISSING: Every page is fitted on save, after the time and gap layouts: each axis's tick labels are worked out the way Veusz will draw them (vszh5_writer.axi...
        _LABEL_PAD_CM: room kept beside a time-axis label that sets the plot width (cm)
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
        _TAG_WORD: characters not allowed in a tag word (replaced by '_')
        MISSING_VALUE_WORDS: cell texts that mean no value (pandas' NA strings): no exact-text copy needed
        NUMERIC_TICK_FORMAT: tick-label format of numeric axes: up to 12 significant digits
        DEFAULT_RAM_LIMIT_PERCENT: default RAM limit: offload above this much of the computer's RAM in use (%)
        OFFLOAD_DIR_ENV: environment variable naming the folder for offload files
        OFFLOAD_MIN_BYTES: datasets smaller than this (bytes of values) stay in RAM
        _RAM_CHECK_S: RAM use is read at most this often while datasets are added (s)
        _HEAP_TRIM_S: freed heap is handed back to the system at most this often (s)
        _RAM_LIMIT: the program-wide RAM limit (set_ram_limit)
        _NOTE_S: (0.5.7) Offload messages are logged at most this often (s), with totals.
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
    Tick labels: finalize() fits every page
    (_fit_page): vertical two-line date labels and long numbers, margins and
    page sizes so Veusz draws every major tick label; undone before the next fit.
    Clocks: TimeLayout(clocks=...)
    draws date-time x axes in local time, UTC and / or MJD, one set of pages per
    clock, rebuilt on every save (see "Time-axis clocks").
    Offload: above the RAM limit
    (set_ram_limit) datasets and kept file data go to a pickle file
    (OffloadStore) and are read back where used (see "Offloading").
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.6: Time-axis clocks: Local time (this computer's zone), UTC and MJD.
Date: 2026-11-17
#              Operator request 2026-11-17: local time read from the time
#              zone of the computer processing the data (not -05:00), and
#              tick boxes for MJD, UTC and local time axes in the GUI plots
#              and the .vszh5 (default local only).
#              * The local offset was and is read from the operating system
#                for every sample (datetime.astimezone(), DST-aware); the
#                "-05:00" in a tick format is text, because a Veusz date
#                format cannot show a zone.  NOW the axis label names the
#                zone ("Local time (ISO 8601, EST, UTC-05:00)"; across a DST
#                change "UTC-04:00 EDT, UTC-05:00 EST from ..."), the graph
#                notes say where it came from, local_zone_description() /
#                zone_name_at() give the GUI its zone line.
#              * NEW TimeLayout(clocks=...) (TIME_CLOCKS local, utc, mjd;
#                time_clocks(); describe_clocks()); finalize() draws a page
#                with date-time x axes once per clock: the first keeps the
#                page, copies "<page>_utc" / "<page>_mjd" follow it
#                (_clone_widget, _remove_clock_pages, _free_page_name);
#                rebuilt on every save, so they always match their page.
#              * NEW UTC clock: the stored UTC date dataset (a "<utc>_gaps"
#                copy with gap rows), ticks ISO 8601 ending in "Z"; MJD
#                clock: the exact MJD (SetDataDateTime(mjd=...): a dataset
#                name or array; else UTC / 86400 + 54832) on a numeric axis,
#                12 significant digits.  Gaps and axis breaks are found in
#                UTC seconds for every clock (the same samples).
#              * NEW _fit_page keeps a time-axis label inside its plot
#                (_time_label_cm); tags type_time_utc / type_time_mjd;
#                add_gap_arguments(time_axes=True) adds --time-axes.
#              Checked (Python 3.12.3, Veusz 4.2.1): tests/test_time_clocks.py
#              (15 tests) and the whole suite; FITS GBT tack / UWBR Temps /
#              30-day 1PPS, synthetic FITS / Franks and CSV projects with
#              all three clocks rendered in Veusz with every tick label
#              drawn; local-only projects identical to 0.4.5 in every
#              dataset (only the axis label and graph notes changed).
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
        layout_from_args: The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3; 0.5.6: --time-axes).
        time_clocks: The clocks of ``value`` as a tuple in TIME_CLOCKS order (0.5.6).
        _zone_of: The tzinfo of ``zone`` (None = VSZH5_DISPLAY_TZ, else None = this computer's zone; a text is read as by the writer) (0.5.6).
        zone_name_at: Name of the time zone at the UTC instant ``epoch_s`` (Unix seconds) (0.5.6): what the operating system calls this computer's zone then ("EST", "EDT...
        label_zone_name: ``name`` as the axis label shows it, or "" for a name that only repeats the offset or says UTC (0.5.6).
        local_zone_description: One line naming the zone local time axes use (0.5.6), for the GUI and the logs: this computer's zone (or VSZH5_DISPLAY_TZ) with its name and UTC of...
        TimeLayout (class): Gap rule and gap options of every x axis (0.5.0 time axes; 0.5.3 any sorted x, and the two consequences of a gap made optional), see vszh5_writer T...
        _number_text: A setting for the notes with every digit (repr), without a trailing ".0" (0.5.3; %g cut 12.3456789 to 12.3457).
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _page_of: The page widget ``widget`` is on, or None (0.5.6).
        _clock_of: The time-axis clock of the page ``widget`` is on ("local" when the page has none yet, or the widget is on no page) (0.5.6).
        _same_values: True when two arrays hold the same numbers, NaN where the other has NaN (0.5.6; numpy.array_equal(equal_nan=...) needs numpy >= 1.19).
        _mjd_name: Name of the MJD plot copy of the UTC date dataset ``utc_name`` (0.5.6): "<x>__utc" -> "<x>__mjd", "<x>_utc" -> "<x>_mjd", else "<x>_mjd".
        _exact_mjd: (MJD values, name of a stored dataset holding exactly them or None) of one time x (0.5.6).
        _clock_dataset: Plot x dataset of one trace on the UTC or MJD clock (0.5.6).
        _time_note: The "Time axis:" line of the graph notes for ``clock`` (0.5.6).
        _set_note_line: Put ``line`` into the graph notes, replacing a line that starts with ``prefix`` (0.5.6; _set_gap_note for any prefix).
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0; 0.5.6 clocks).
        _set_gap_note: Put the gap-rule line ``note`` into the graph notes (0.5.3: a line from an earlier layout is replaced, so the notes state the rule in use).
        _sorted_finite: True when the finite values of ``values`` never decrease (at least 3).
        _layout_numeric_axis: Gap options on an x axis that is not time (0.5.3): frequency, angle, index ...
        _undo_numeric_layout: Undo an earlier _layout_numeric_axis of this axis (0.5.3): when the gap options of a project turn non-time axes off between two saves, the traces d...
        _drop_unused_plot_copies: Remove line-break plot copies no widget draws any more (0.5.3): when the gap options change between two saves, copies of the old layout would other...
        _has_clock_axis: True when ``page`` draws a date-time x axis laid out on a clock.
        _free_page_name: ``wanted``, or wanted_2, wanted_3 ...
        _copy_setting: A setting value for a page copy: lists and dicts copied, the rest (text, numbers, booleans) shared.
        _clone_widget: Copy of ``widget`` and every widget below it, under ``parent`` (0.5.6).
        _remove_clock_pages: Take the clock copies of the last finalize out of the project (0.5.6); finalize makes them again from their pages, so a copy always shows its page...
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        _fit_set: Set ``key`` on ``widget`` for the tick-label fit, remembering the value it had before (0.5.5).
        _fit_restore: Undo the previous tick-label fit on ``widget`` (0.5.5): a setting the fit wrote and nobody changed since gets its earlier value back.
        _cm: A Veusz distance setting in cm (``default`` when unset or not physical).
        _pt: A Veusz font size setting in points (``default`` when unset).
        _finite_range: (min, max) of the finite values of dataset ``name`` (1-D or date), or None; cached per finalize in ``cache``.
        _axis_data_range: (min, max) of the data a graph draws on ``axis`` (xy traces and images), or None when it draws nothing known (0.5.5).
        _axis_room: vszh5_writer.TickRoom of one axis as it is set now (0.5.5); None for a hidden axis, a label axis or an axis without known data.
        _time_label_cm: Width (cm) a date-time x axis's label needs (0.5.6), 0 for other axes.
        _dated_page_size: (width, height) cm the 0.5.0 rule gives a page with date-time axes and no size of its own: 13 cm a column and 8 cm a row (at least 16 cm) for a gri...
        _fit_page: Fit the tick labels of every graph on ``page`` (0.5.5): vertical labels where they belong, the margins every label needs and a page large enough (s...
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
        TIME_CLOCKS: the clocks of date-time x axes, in page order (local, utc, mjd)
        DEFAULT_TIME_CLOCKS: default clocks: local time only
        CLOCK_PAGE_SUFFIX: page-name suffix of each clock ('' for local)
        CLOCK_TITLES: clock names for people (GUI, logs, notes)
        UTC_AXIS_LABEL: axis label of a UTC clock axis
        MJD_AXIS_LABEL: axis label of an MJD clock axis
        UTC_DATETIME_FORMAT: tick format of a UTC date-time axis (ISO 8601, 'Z')
        _OFFSET_ONLY_NAME: zone names that only repeat the offset or say UTC (left off labels)
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _TIME_NOTE: start of the graph-notes line written by the time layout
        _MISSING: Every page is fitted on save, after the time and gap layouts: each axis's tick labels are worked out the way Veusz will draw them (vszh5_writer.axi...
        _LABEL_PAD_CM: room kept beside a time-axis label that sets the plot width (cm)
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
    Tick labels: finalize() fits every page
    (_fit_page): vertical two-line date labels and long numbers, margins and
    page sizes so Veusz draws every major tick label; undone before the next fit.
    Clocks: TimeLayout(clocks=...)
    draws date-time x axes in local time, UTC and / or MJD, one set of pages per
    clock, rebuilt on every save (see "Time-axis clocks").
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.5.5: Every major tick label shows: vertical date and long-number x labels.
Date: 2026-11-16
#              * NEW _fit_page (run by finalize() for every page, after the
#                time and gap layouts): each axis's labels as Veusz will draw
#                them (vszh5_writer.axis_tick_room on the data range of its
#                traces / images); x date-times and long numbers vertical
#                (-90, reading downwards; date-times on two lines, the date
#                then "T" + time); bottom and left margins every label and
#                axis label need, the largest of a grid for all its graphs;
#                pages grown (never shrunk) so each plot keeps MIN_PLOT_CM
#                and room for every label; fewer major ticks per part of a
#                broken axis whose parts are short.  What the fit writes is
#                undone before the next fit (NativeWidget.fit_base /
#                fit_written), so saving twice writes the same file.
#              * The 0.5.0 page size of date pages (13 cm a column, 8 cm a
#                row; 25 x 16 cm) is the starting size (_dated_page_size);
#                Veusz's grid uses only the columns its graphs fill.
#              * Measured (Veusz 4.2.1): x tick labels drawn, before -> now:
#                GBT tack 723 / 1925 -> 1925 / 1925; UWBR Temps 403 / 788 ->
#                788 / 788; R&S 2130 / 2563 -> 2563 / 2563; CSV 15 / 18 ->
#                18 / 18; y labels on tack 1979 / 2017 -> 2017 / 2017.
#              Checked: tests/test_writer_0031.py (16 tests: identity with
#              master 0.0.30 for pages without dates or ranges, the ports
#              against Veusz 4.2.1, the label decisions, the page helpers,
#              the native fit, a Veusz render in which every tick label is
#              drawn) on Python 3.8, 3.12 and 3.13 (the Veusz tests with
#              Veusz 4.2.1 on 3.13); every example project rendered in
#              Veusz 4.2.1 with all tick labels drawn.
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
        _fit_set: Set ``key`` on ``widget`` for the tick-label fit, remembering the value it had before (0.5.5).
        _fit_restore: Undo the previous tick-label fit on ``widget`` (0.5.5): a setting the fit wrote and nobody changed since gets its earlier value back.
        _cm: A Veusz distance setting in cm (``default`` when unset or not physical).
        _pt: A Veusz font size setting in points (``default`` when unset).
        _finite_range: (min, max) of the finite values of dataset ``name`` (1-D or date), or None; cached per finalize in ``cache``.
        _axis_data_range: (min, max) of the data a graph draws on ``axis`` (xy traces and images), or None when it draws nothing known (0.5.5).
        _axis_room: vszh5_writer.TickRoom of one axis as it is set now (0.5.5); None for a hidden axis, a label axis or an axis without known data.
        _dated_page_size: (width, height) cm the 0.5.0 rule gives a page with date-time axes and no size of its own: 13 cm a column and 8 cm a row (at least 16 cm) for a gri...
        _fit_page: Fit the tick labels of every graph on ``page`` (0.5.5): vertical labels where they belong, the margins every label needs and a page large enough (s...
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
        _MISSING: Every page is fitted on save, after the time and gap layouts: each axis's tick labels are worked out the way Veusz will draw them (vszh5_writer.axi...
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
    Tick labels: finalize() fits every page
    (_fit_page): vertical two-line date labels and long numbers, margins and
    page sizes so Veusz draws every major tick label; undone before the next fit.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
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

import copy
import datetime
import math
import os
import pickle
import re
import shutil
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from fractions import Fraction                       # 0.5.8: exact gap texts
try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 2 era
    from collections import Mapping  # type: ignore

import numpy as np

import vszh5_writer
from vszh5_writer import VeuszWriter, VszDoc, vz_text

VERSION = "0.5.9"
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
        self.fit_base = {}            # 0.5.5: settings before the tick-label fit
        self.fit_written = {}         # 0.5.5: settings the tick-label fit wrote
        # 0.5.6: time-axis clocks (TimeLayout.clocks).  A page: the clock its
        # date-time axes are drawn in; an axis: the clock it was laid out in
        # (None = not a date-time axis).
        self.time_clock = None
        self.time_base_name = None    # 0.5.6: a page's own name (no clock suffix)
        self.time_clone_of = None     # 0.5.6: the page a clock copy was made from

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
#
# 0.5.8: the whole entry must be read, token after token: a number and its
# unit, optional spaces between them and between tokens, nothing else.
# Before 0.5.8 the tokens were found anywhere in the text, so "00:10:00"
# read as 0 h + 10 h + 0 h = 10 h, "1x2" as 3 h and ".5h" as 5 h.  Also new:
# a clock "H:MM" or "H:MM:SS(.s)" (hours:minutes[:seconds]), the long unit
# names (sec, minutes, hours, days, months ...) and exponents ("1e3s").  A
# bare number alone is still hours; with several tokens each needs a unit.
_GAP_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?"
_GAP_TOKEN_REGEX = re.compile(r"\s*(" + _GAP_NUMBER + r")\s*([A-Za-z]*)\s*")
_GAP_NUMBER_REGEX = re.compile(r"\s*\+?(" + _GAP_NUMBER + r")\s*$")      # 0.5.8
_GAP_CLOCK_REGEX = re.compile(                                          # 0.5.8
    r"\s*(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?\s*$")
#: (0.4.0) Tool tip of the gap text boxes (0.5.8: clocks and unit names).
GAP_STRING_HELP = ("Units: s = seconds, min = minutes, h = hours, d = days, "
                   "m = months (30.4375 days); also sec, minutes, hours, days, months.  "
                   "Examples: '90s', '15min', '12h', '3d', '1m 3d 2h', '0:15' or "
                   "'00:15:00' (hours:minutes[:seconds]).  A bare number is hours.  "
                   "An entry that cannot be read is not used (the log says so and the "
                   "box turns red).")

# Average Gregorian month length (365.25 / 12) -- intentionally the same
# value used by other scientific tools (e.g. pandas) so 1m + 1m + ... = ~year.
_DAYS_PER_MONTH = 30.4375
_DAYS_PER_HOUR = 1.0 / 24.0
_DAYS_PER_MINUTE = _DAYS_PER_HOUR / 60.0     # 0.4.0
_DAYS_PER_SECOND = _DAYS_PER_MINUTE / 60.0   # 0.4.0
#: (0.5.8) Seconds per unit of a gap text, every accepted spelling (lower
#: case; "m" is months, as since 0.0.19 - 30.4375 d = 2 629 800 s; minutes
#: are "min").  Integers: the text is read exactly (fractions.Fraction) and
#: rounded once, so "16min" is 960 s, not 959.9999999999999 s as the
#: day-based sum of 0.4.0-0.5.7 gave.
_GAP_UNIT_SECONDS = dict(
    [(u, 1) for u in ("s", "sec", "secs", "second", "seconds")]
    + [(u, 60) for u in ("min", "mins", "minute", "minutes")]
    + [(u, 3600) for u in ("h", "hr", "hrs", "hour", "hours")]
    + [(u, 86400) for u in ("d", "day", "days")]
    + [(u, 2629800) for u in ("m", "mo", "month", "months")])


def _gap_fraction(text):
    # type: (object) -> Tuple[Optional[Fraction], Optional[str]]
    """The exact seconds of a gap text (0.5.8): ``(Fraction, None)`` or
    ``(None, problem)``; check_gap_string has the syntax."""
    if text is None:
        return Fraction(0), None
    try:
        s = str(text).strip()
    except Exception:
        return None, "not text"
    if not s:
        return Fraction(0), None
    if s.startswith("-"):
        return None, "a gap cannot be negative"
    bare = _GAP_NUMBER_REGEX.match(s)
    if bare is not None:                      # a bare number alone: hours
        return Fraction(bare.group(1)) * 3600, None
    clock = _GAP_CLOCK_REGEX.match(s)
    if clock is not None:                     # hours:minutes[:seconds]
        hours, minutes = int(clock.group(1)), int(clock.group(2))
        seconds = Fraction(clock.group(3)) if clock.group(3) else Fraction(0)
        if minutes >= 60 or seconds >= 60:
            return None, "minutes and seconds of a clock must be under 60"
        return hours * 3600 + minutes * 60 + seconds, None
    if ":" in s:
        return None, "a clock is hours:minutes or hours:minutes:seconds"
    total, position = Fraction(0), 0
    while position < len(s):
        token = _GAP_TOKEN_REGEX.match(s, position)
        if token is None or token.end() == position:
            return None, "cannot read %r" % (s[position:],)
        unit = token.group(2).lower()
        if not unit:
            return None, "give every number a unit (s, min, h, d, m) when there are several"
        if unit not in _GAP_UNIT_SECONDS:
            return None, "unknown unit %r (s, min, h, d, m)" % (token.group(2),)
        total += Fraction(token.group(1)) * _GAP_UNIT_SECONDS[unit]
        position = token.end()
    return total, None


def check_gap_seconds(text):
    # type: (object) -> Tuple[float, Optional[str]]
    """Read a gap text in seconds (0.5.8): ``(seconds, problem)``, the
    exact value of the text rounded once to a float (check_gap_string)."""
    exact, problem = _gap_fraction(text)
    if problem is not None:
        return 0.0, problem
    try:
        seconds = float(exact)
    except OverflowError:
        return 0.0, "the number is too large"
    return (seconds, None) if math.isfinite(seconds) else (0.0, "the number is too large")


def check_gap_string(text):
    # type: (object) -> Tuple[float, Optional[str]]
    """Read a gap text (0.5.8): ``(days, problem)``.

    ``problem`` is None when the whole entry was read (``days`` >= 0; an
    empty entry is 0 = not set) and otherwise says why it was not
    (``days`` is then 0.0).  The days are the exact value of the text
    rounded once (check_gap_seconds gives the seconds the same way; use
    it for a TimeLayout, whose gaps are seconds).  Accepted,
    case-insensitive, spaces allowed between a number and its unit and
    between tokens::

        "120"        -> 120 hours = 5 days  (a bare number alone is hours)
        "90s", "15min", "2h", "3d", "1m"    (m = month = 30.4375 days)
        "5m 3d 2h", "5m3d2h", "2h 30min"    (the tokens add up)
        "1.5h", ".5h", "1e3s"
        "0:15", "00:15:00", "1:30:15.5"     (hours:minutes[:seconds])
        "", None, "0"                       -> 0 (not set)

    Not accepted: anything left over ("00:10:00x", "1x2", "15 mins ago"),
    a negative gap, an unknown unit ("5ms", "2w"), a number without a unit
    next to other tokens ("1d 2"), minutes or seconds of 60 or more in a
    clock ("1:75"), inf / nan.
    """
    exact, problem = _gap_fraction(text)
    if problem is not None:
        return 0.0, problem
    try:
        days = float(exact / 86400)
    except OverflowError:
        return 0.0, "the number is too large"
    return (days, None) if math.isfinite(days) else (0.0, "the number is too large")


def describe_gap_seconds(seconds):
    # type: (float) -> str
    """A read gap for people (0.5.8): ``"15 min"``, ``"1.5 h"``, ``"not set"``
    at 0 (TimeLayout's _seconds_text: exact seconds follow when the short
    form is not exact)."""
    if not seconds or seconds <= 0:
        return "not set"
    return _seconds_text(seconds)


def describe_gap_days(days):
    # type: (float) -> str
    """describe_gap_seconds for a gap in days (0.5.8)."""
    return describe_gap_seconds((days or 0.0) * 86400.0)


#: (0.5.8) The short unit list a log line about an unread gap text ends with.
_GAP_HINT = "Units s, min, h, d, m (months) or hours:minutes[:seconds]; e.g. 90s, 15min, 2h, 0:15"


def parse_gap_seconds(text, log_cb=None, what="gap"):
    # type: (object, Optional[Callable[[str], None]], str) -> float
    """parse_gap_string in seconds (0.5.8; exact, for TimeLayout manual_s /
    axis_manual_s): 0.0 for an entry that cannot be read, logged."""
    seconds, problem = check_gap_seconds(text)
    if problem is not None:
        if log_cb:
            log_cb("  The fixed %s %r is not used: %s (%s)." % (what, str(text).strip(),
                                                              problem, _GAP_HINT))
        return 0.0
    return seconds


def parse_gap_string(text, log_cb=None, what="gap"):
    # type: (object, Optional[Callable[[str], None]], str) -> float
    """Parse a human-readable gap string into MJD-days (check_gap_string).

    ``"5m 3d 2h"`` = 5 months + 3 days + 2 hours; ``"120"`` = 120 hours;
    ``"0:15"`` = 15 minutes; empty = 0 (no fixed gap).  An entry that cannot
    be read returns 0.0 (the K / n rule applies) and, with ``log_cb``, a log
    line naming the entry, why, and ``what`` (e.g. "small gap") - 0.5.8:
    before, part of an entry could be read and the rest ignored.
    """
    days, problem = check_gap_string(text)
    if problem is not None:
        if log_cb:
            log_cb("  The fixed %s %r is not used: %s (%s)." % (what, str(text).strip(),
                                                              problem, _GAP_HINT))
        return 0.0
    return days

def gap_text(value):
    """argparse type of --gap-small / --gap-large (0.5.8): the text itself
    when check_gap_string() reads all of it, else ArgumentTypeError with
    the reason (argparse stops with it)."""
    import argparse
    _days, problem = check_gap_string(value)
    if problem is not None:
        raise argparse.ArgumentTypeError("%r: %s (%s)" % (value, problem, _GAP_HINT))
    return str(value).strip()


def _gap_factor(value, minimum, name):
    """A finite number >= ``minimum`` for argparse (0.5.8), else ArgumentTypeError."""
    import argparse
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("%s must be a number, not %r" % (name, value))
    if not math.isfinite(number) or number < minimum:
        raise argparse.ArgumentTypeError("%s must be a number >= %g, not %r"
                                         % (name, minimum, value))
    return number


def gap_factor_k(value):
    """argparse type of --gap-k (0.5.8): a finite K >= MIN_GAP_K."""
    return _gap_factor(value, MIN_GAP_K, "K")


def gap_factor_n(value):
    """argparse type of --gap-n (0.5.8): a finite n >= 0 (0 = no axis breaks)."""
    return _gap_factor(value, 0.0, "n")


def add_gap_arguments(parser, time_axes=False):
    """Add the gap options to an argparse parser (0.5.3): --gap-k,
    --gap-small, --gap-n, --gap-large, --no-line-breaks, --no-axis-breaks,
    --no-numeric-gaps; layout_from_args() turns them into a TimeLayout.
    0.5.6: ``time_axes`` also adds --time-axes (local, utc, mjd, all) for a
    tool that draws date-time x axes."""
    group = parser.add_argument_group(
        "gaps", "A small gap breaks the line, a large gap breaks the x axis; either can be "
        "turned off. Time axes and (unless --no-numeric-gaps) any sorted x.")
    # 0.5.8: every value is checked when the command line is read (argparse
    # stops with the reason); before, an unreadable gap text was ignored
    group.add_argument("--gap-k", type=gap_factor_k, default=vszh5_writer.GAP_K,
                       help="small gap: step > K x median step, K >= 1 (default %(default)s)")
    group.add_argument("--gap-small", type=gap_text, default="",
                       help="fixed small gap on time axes, e.g. 90s, 15min, 2h, 0:15 "
                            "(overrides K)")
    group.add_argument("--gap-n", type=gap_factor_n, default=vszh5_writer.GAP_AXIS_N,
                       help="large gap: no data for > n x the small gap (0 = never; "
                            "default %(default)s)")
    group.add_argument("--gap-large", type=gap_text, default="",
                       help="fixed large gap on time axes, e.g. 12h, 3d (overrides n)")
    group.add_argument("--no-line-breaks", action="store_true",
                       help="draw every line across its gaps")
    group.add_argument("--no-axis-breaks", action="store_true",
                       help="never break the x axis")
    group.add_argument("--no-numeric-gaps", action="store_true",
                       help="gap rule on time axes only (not frequency, angle, index ...)")
    if not time_axes:
        return group
    # 0.5.6: the clocks of the date-time x axes, one set of pages per clock
    group.add_argument("--time-axes", type=time_clocks, default=DEFAULT_TIME_CLOCKS,
                       help="clocks of the date-time x axes, comma separated: local (this "
                            "computer's time zone), utc, mjd, or all; one set of pages per "
                            "clock (default: local)")
    return group


def add_memory_arguments(parser):
    """Add --ram-limit to an argparse parser (0.5.7): offload plot data to
    disk when this computer's RAM in use (all programs) reaches PERCENT (0 =
    never; default DEFAULT_RAM_LIMIT_PERCENT); ram_limit_from_args() sets it."""
    parser.add_argument("--ram-limit", type=float, default=DEFAULT_RAM_LIMIT_PERCENT,
                        metavar="PERCENT",
                        help="offload plot data to a temporary pickle file when this computer's "
                             "RAM in use (all programs) reaches PERCENT; 0 = never "
                             "(default %(default)s)")
    return parser


def ram_limit_from_args(args):
    """Set the program's RAM limit from add_memory_arguments() (0.5.7); returns it."""
    return set_ram_limit(getattr(args, "ram_limit", DEFAULT_RAM_LIMIT_PERCENT))


def layout_from_args(args, log_cb=None):
    """The TimeLayout of the add_gap_arguments() options in ``args`` (0.5.3;
    0.5.6: --time-axes; 0.5.8: ``log_cb`` reports a gap text that cannot be
    read - argparse already refuses one, so only for a hand-made ``args``)."""
    return TimeLayout(k=args.gap_k, n=args.gap_n,
                      manual_s=parse_gap_seconds(args.gap_small, log_cb, "small gap"),
                      axis_manual_s=parse_gap_seconds(args.gap_large, log_cb, "large gap"),
                      break_lines=not args.no_line_breaks, break_axis=not args.no_axis_breaks,
                      numeric=not args.no_numeric_gaps,
                      clocks=getattr(args, "time_axes", None))


# %% Page kinds: per file, combined axis, overlay (0.5.9)
# Every plotter of the suite can draw three kinds of page:
#   per_file - the pages of one file (as before);
#   combined - "combined axis": pages like a file's own, but each trace holds
#              the same quantity of every file joined end to end (in file
#              order, then sorted on x, stable), so each axis spans the full
#              range of all the files; the gap rules lay the joined traces out
#              like any other (the steps between files are gaps);
#   overlay  - every file's traces stacked on one graph (as before).
# In the project and in the Plots tab the combined pages come just above the
# overlay pages (move_pages_before).
#: (0.5.9) The page kinds, in the order the GUIs list them.
PAGE_KINDS = ("per_file", "combined", "overlay")
#: (0.5.9) Their names in the GUIs and the log.
PAGE_KIND_TEXT = {"per_file": "Per file", "combined": "Combined axis", "overlay": "Overlay"}
#: (0.5.9) Default page kinds: all three.
DEFAULT_PAGE_KINDS = PAGE_KINDS
_PAGE_KIND_WORDS = {"per_file": "per_file", "perfile": "per_file", "file": "per_file",
                    "files": "per_file", "per-file": "per_file", "combined": "combined",
                    "combined_axis": "combined", "combined-axis": "combined", "combine": "combined",
                    "concatenated": "combined", "overlay": "overlay", "overlays": "overlay"}


def page_kinds(value=None):
    """
    The page kinds asked for (0.5.9), in PAGE_KINDS order.

    ``value``: None (the default: all three), an iterable of kinds, or a text
    such as "per_file,combined" / "overlay" / "all" / "none" (spaces,
    commas and "+" separate; "-" and "_" in a word are the same).  Unknown
    words raise ValueError (argparse turns it into a message).
    """
    if value is None:
        return tuple(DEFAULT_PAGE_KINDS)
    if isinstance(value, str):
        words = [w for w in re.split(r"[\s,+]+", value.strip().lower()) if w]
    else:
        words = [str(w).strip().lower() for w in value]
    if words == ["all"]:
        return tuple(PAGE_KINDS)
    if words in (["none"], []):
        return ()
    chosen = set()
    for word in words:
        kind = _PAGE_KIND_WORDS.get(word.replace("-", "_"), _PAGE_KIND_WORDS.get(word))
        if kind is None:
            raise ValueError("unknown page kind %r (per_file, combined, overlay, all, none)"
                             % word)
        chosen.add(kind)
    return tuple(k for k in PAGE_KINDS if k in chosen)


def describe_page_kinds(kinds):
    """Log line of the page kinds (0.5.9): "Pages: Per file, Combined axis, Overlay"."""
    kinds = page_kinds(kinds)
    return "Pages: %s" % (", ".join(PAGE_KIND_TEXT[k] for k in kinds) if kinds else "none")


def add_page_arguments(parser):
    """Add --pages to an argparse parser (0.5.9): which page kinds to draw
    (per_file, combined, overlay; default all); page_kinds_from_args() reads it."""
    import argparse

    def kinds(text):
        """argparse type: page_kinds() with argparse's error."""
        try:
            return page_kinds(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    parser.add_argument("--pages", type=kinds, default=tuple(DEFAULT_PAGE_KINDS),
                        metavar="KINDS",
                        help="page kinds to draw: per_file, combined (combined axis: every "
                             "file's data joined on one axis), overlay - comma separated, or "
                             "all / none (default all)")
    return parser


def page_kinds_from_args(args):
    """The page kinds of add_page_arguments() in ``args`` (0.5.9; all when absent)."""
    return page_kinds(getattr(args, "pages", None))


def concatenate_on_x(xs, ys, with_sources=False):
    """
    Join the traces of several files into one (0.5.9, combined-axis pages).

    ``xs``: the x arrays of the files, in file order; ``ys``: their y arrays
    (same lengths; or a list per file of several y arrays sharing that x).
    The pieces are joined in the order given and then sorted on x with a
    stable sort (equal x keep their file order; NaN x go last), so a gap
    rule sees one ascending axis.  Returns ``(x, y)`` - or ``(x, [y, ...])``
    for several y arrays - float64 for numbers, the values themselves
    otherwise; nothing is rounded.  ``with_sources``: also return, as a
    third item, the file (0, 1, ... in the order given) of every joined row
    (file_index_array), for set_combined_sources.
    """
    if not xs:
        empty = (np.empty(0), ([] if ys and isinstance(ys[0], (list, tuple)) else np.empty(0)))
        return empty + (file_index_array([]),) if with_sources else empty
    several = isinstance(ys[0], (list, tuple))
    x_all = np.concatenate([np.asarray(x).reshape(-1) for x in xs])
    order = np.argsort(x_all, kind="stable") if x_all.dtype.kind in "fiub" else None
    x_sorted = x_all if order is None else x_all[order]
    if with_sources:
        files = file_index_array([np.asarray(x).size for x in xs])
        files = files if order is None else files[order]

    def joined(parts):
        """One y joined like x."""
        y_all = np.concatenate([np.asarray(p).reshape(len(np.asarray(p)), -1)
                                if np.asarray(p).ndim > 1 else np.asarray(p) for p in parts])
        return y_all if order is None else y_all[order]
    if several:
        result = (x_sorted, [joined([y[i] for y in ys]) for i in range(len(ys[0]))])
    else:
        result = (x_sorted, joined(ys))
    return result + (files,) if with_sources else result


def file_index_array(rows):
    """The file (0, 1, ...) of every row of pieces of ``rows`` rows each,
    joined in that order (0.5.9), in the smallest unsigned integer type
    (uint8 below 256 files, uint16 below 65536): 1 or 2 bytes a row."""
    rows = [int(n) for n in rows]
    dtype = np.uint8 if len(rows) <= 256 else np.uint16 if len(rows) <= 65536 else np.uint32
    return np.repeat(np.arange(len(rows), dtype=dtype), rows)


def set_combined_sources(doc, names, sources):
    """
    Record which file each row of the combined-axis x datasets ``names``
    came from (0.5.9): ``sources`` holds one file number per row, in the
    rows' (sorted) order.  The gap layout of finalize() then cuts a joined
    trace only where none of its files has data (combined_gap_positions)
    and breaks the axis as an overlay of the same files would be broken.
    The array is not saved; it is kept in RAM in the smallest integer type
    and moves to the offload file with the datasets above the RAM limit.
    A name not in ``doc`` is skipped.
    """
    store = getattr(doc, "combined_of", None)
    if store is None:
        return
    array = np.asarray(sources).reshape(-1)
    if array.size and array.dtype.kind not in "ui":
        raise ValueError("file numbers must be integers")
    if array.size and int(array.min()) < 0:
        raise ValueError("file numbers must not be negative")
    if array.size:                                     # the smallest unsigned type
        top = int(array.max())
        array = array.astype(np.uint8 if top < 256 else np.uint16 if top < 65536
                             else np.uint32, copy=False)
    for name in names:
        if isinstance(name, str) and name in doc.datasets:
            store[name] = array


def combined_sources(doc, name, size):
    """The file number of every row of the combined x dataset ``name``
    (0.5.9; set_combined_sources), or None when ``name`` is not a joined
    trace or its length is not ``size``."""
    given = getattr(doc, "combined_of", {}).get(name)
    if given is None:
        return None
    if isinstance(given, OffloadedValue):              # moved to disk above the RAM limit
        given = given.load()
    given = np.asarray(given).reshape(-1)
    return given if given.size == int(size) else None


def combined_gap_positions(values, sources, line_of, log_of=None, break_lines=True):
    """
    Gap rows of a joined trace (0.5.9): ``values`` are its x values in the
    gap rule's units (sorted; NaN allowed), ``sources`` the file of every
    row.  Each file is measured on its own samples - ``line_of(piece)`` its
    small-gap threshold, ``log_of(piece)`` its log-scale threshold (0: no
    log test) - and covers the runs between its own gaps.  A step of the
    joined trace is cut only when no single run of any file spans it and it
    is longer than the larger threshold of the two files beside it (and,
    with a log test, larger on a log scale too): the trace breaks where no
    file has data, never inside a file's own sampling (a coarse sweep
    joined with a fine one, files of different cadences).

    Returns ``(positions, thresholds, pieces)``: the insertion indices as
    vszh5_writer.gap_positions gives them (empty when ``break_lines`` is
    off), each file's small-gap threshold and each file's values (for the
    axis-break rule, which then sees the files as an overlay would).
    Unsorted values give no positions.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    sources = np.asarray(sources).reshape(-1)
    empty = np.zeros(0, dtype=np.int64)
    if sources.shape != values.shape:
        return empty, [], [values]
    pieces, thresholds, log_limits, starts, ends = [], {}, {}, [], []
    for number in np.unique(sources):
        piece = values[sources == number]
        line = float(line_of(piece) or 0.0)
        log_limit = float(log_of(piece) or 0.0) if log_of is not None else 0.0
        pieces.append(piece)
        thresholds[number], log_limits[number] = line, log_limit
        real = piece[np.isfinite(piece)]
        if not real.size:
            continue
        own = vszh5_writer.gap_positions(real, line)
        if own.size and log_limit > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                own = own[(np.log(real[own]) - np.log(real[own - 1])) > log_limit]
        starts.append(real[np.concatenate(([0], own))])
        ends.append(real[np.concatenate((own - 1, [real.size - 1]))])
    lines = [thresholds[k] for k in sorted(thresholds)]
    if not break_lines or values.size < 2 or not starts:
        return empty, lines, pieces
    steps = np.diff(values)
    finite = np.isfinite(steps)
    if np.any(steps[finite] < 0):
        return empty, lines, pieces                    # not in order: no gaps defined
    # The runs of every file merged into the stretches that have data.
    run_a = np.concatenate(starts)
    run_b = np.concatenate(ends)
    by_start = np.argsort(run_a, kind="stable")
    run_a, run_b = run_a[by_start], np.maximum.accumulate(run_b[by_start])
    new = np.concatenate(([True], run_a[1:] > run_b[:-1]))
    first = np.flatnonzero(new)
    last = np.concatenate((first[1:] - 1, [run_a.size - 1]))
    have_a, have_b = run_a[first], run_b[last]
    candidates = np.flatnonzero(finite & (steps > 0))
    left, right = values[candidates], values[candidates + 1]
    where = np.searchsorted(have_a, left, side="right") - 1
    covered = (where >= 0) & (right <= have_b[np.maximum(where, 0)])
    candidates = candidates[~covered]
    if not candidates.size:
        return empty, lines, pieces
    left_file, right_file = sources[candidates], sources[candidates + 1]
    line = np.maximum([thresholds[k] for k in left_file], [thresholds[k] for k in right_file])
    keep = (line > 0.0) & (steps[candidates] > line)
    log_limit = np.maximum([log_limits[k] for k in left_file],
                           [log_limits[k] for k in right_file])
    if np.any(log_limit > 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_steps = np.log(values[candidates + 1]) - np.log(values[candidates])
        keep &= (log_limit <= 0.0) | (log_steps > log_limit)
    return (candidates[keep] + 1).astype(np.int64), lines, pieces


def overlapping_ranges(xs):
    """How many of the files' x ranges overlap an earlier one (0.5.9): the
    combined trace interleaves their samples there.  ``xs``: x arrays."""
    spans = []
    for x in xs:
        values = np.asarray(x, dtype=float).reshape(-1) if np.asarray(x).dtype.kind in "fiub" \
            else np.empty(0)
        values = values[np.isfinite(values)]
        if values.size:
            spans.append((float(values.min()), float(values.max())))
    spans.sort()
    count, highest = 0, None
    for low, high in spans:
        if highest is not None and low < highest:
            count += 1
        highest = high if highest is None else max(highest, high)
    return count


def move_pages_before(doc, pages, anchor):
    """
    Put ``pages`` (pages of ``doc``) just before the page ``anchor`` (0.5.9:
    combined pages just above the overlays, in the project and the Plots
    tab).  ``anchor`` None, or not a page of ``doc``, leaves the order as it
    is.  The pages keep their own order.
    """
    root = doc.Root
    children = root.children
    if anchor is None or anchor not in children:
        return
    moving = [p for p in pages if p in children and p is not anchor]
    if not moving:
        return
    rest = [c for c in children if c not in moving]
    at = rest.index(anchor)
    children[:] = rest[:at] + moving + rest[at:]


# %% Time-axis clocks (0.5.6)
# A date-time x axis can be drawn in three clocks; TimeLayout.clocks picks
# any of them (default: local time only, the 0.5.5 pages).  A page with
# date-time x axes is drawn once per chosen clock: the first chosen clock
# keeps the page, every other one gets a copy of the page right after it
# ("<page>_utc", "<page>_mjd").  The stored data are the same for every
# clock; each clock draws plot copies of them:
#   local - "<utc>_local": the UTC instant plus the UTC offset of the zone
#           of the computer running the program at that instant (DST-aware,
#           read from the operating system for every sample; nothing is
#           fixed), ISO 8601 ticks with the offset, the zone named on the
#           axis label.  VSZH5_DISPLAY_TZ overrides the computer's zone.
#   utc   - the UTC date dataset itself (a "<utc>_gaps" copy with gap rows),
#           ISO 8601 ticks ending in "Z".
#   mjd   - the exact MJD values (the MJD dataset or the MJD the program
#           gave SetDataDateTime(mjd=...); else UTC seconds / 86400 + 54832)
#           on a numeric axis with 12 significant digits.
#: (0.5.6) The clocks of date-time x axes, in page order.
TIME_CLOCKS = ("local", "utc", "mjd")
#: (0.5.6) Default clocks: local time only.
DEFAULT_TIME_CLOCKS = ("local",)
#: (0.5.6) Page-name suffix of each clock ("" keeps the 0.5.5 page names).
CLOCK_PAGE_SUFFIX = {"local": "", "utc": "_utc", "mjd": "_mjd"}
#: (0.5.6) Words for people (GUI, logs, notes).
CLOCK_TITLES = {"local": "Local time", "utc": "UTC", "mjd": "MJD"}
#: (0.5.6) Axis labels of the UTC and MJD clocks.
UTC_AXIS_LABEL = "UTC (ISO 8601)"
MJD_AXIS_LABEL = "MJD (UTC)"
#: (0.5.6) Tick format of a UTC date-time axis (ISO 8601, "Z" = UTC).
UTC_DATETIME_FORMAT = vszh5_writer.ISO_DATETIME_FORMAT + "Z"
#: (0.5.6) A zone name that only repeats the offset ("UTC-05:00", "+0530",
#: "GMT+1", "-03") or says "UTC" is left off the axis label.
_OFFSET_ONLY_NAME = re.compile(r"^(?:(?:UTC|GMT)?[+-]\d.*|UTC|Z)$", re.IGNORECASE)


def time_clocks(value=None):
    """
    The clocks of ``value`` as a tuple in TIME_CLOCKS order (0.5.6).

    ``value``: None (the default, local only), a text such as "local,utc"
    or "all", or an iterable of clock names (case and spaces ignored).
    Unknown names or no clock at all raise ValueError.
    """
    if value is None:
        return DEFAULT_TIME_CLOCKS
    if isinstance(value, str):
        items = [part for part in re.split(r"[\s,;+]+", value) if part]
    else:
        items = list(value)
    names = set()
    for item in items:
        word = str(item).strip().lower()
        if word == "all":
            names.update(TIME_CLOCKS)
        elif word in TIME_CLOCKS:
            names.add(word)
        elif word:
            raise ValueError("unknown time-axis clock %r (use local, utc, mjd or all)" % item)
    if not names:
        raise ValueError("at least one time-axis clock is needed (local, utc or mjd)")
    return tuple(clock for clock in TIME_CLOCKS if clock in names)


def _zone_of(zone):
    """The tzinfo of ``zone`` (None = VSZH5_DISPLAY_TZ, else None = this
    computer's zone; a text is read as by the writer) (0.5.6)."""
    if zone is None:
        zone = vszh5_writer.display_zone()
    if isinstance(zone, str):
        zone = vszh5_writer._zone_from_text(zone)
    return zone


def zone_name_at(epoch_s, zone=None):
    """
    Name of the time zone at the UTC instant ``epoch_s`` (Unix seconds)
    (0.5.6): what the operating system calls this computer's zone then
    ("EST", "EDT"; on Windows "Eastern Standard Time"), or the name of
    ``zone`` / VSZH5_DISPLAY_TZ.  "" when unknown.
    """
    try:
        when = datetime.datetime.fromtimestamp(float(epoch_s), datetime.timezone.utc)
        tz = _zone_of(zone)
        local = when.astimezone() if tz is None else when.astimezone(tz)
        return str(local.tzname() or "").strip()
    except (OverflowError, OSError, ValueError, TypeError):
        return ""


def label_zone_name(name):
    """``name`` as the axis label shows it, or "" for a name that only
    repeats the offset or says UTC (0.5.6)."""
    name = str(name or "").strip()
    return "" if not name or _OFFSET_ONLY_NAME.match(name) else name


def local_zone_description(now=None):
    """
    One line naming the zone local time axes use (0.5.6), for the GUI and
    the logs: this computer's zone (or VSZH5_DISPLAY_TZ) with its name and
    UTC offset at ``now`` (Unix seconds; default the current time), e.g.
    "this computer's time zone: EST (UTC-05:00 now; each sample gets the
    offset of its own date, daylight saving included)".
    """
    now = time.time() if now is None else float(now)
    text = os.environ.get(vszh5_writer.DISPLAY_TZ_ENV, "").strip()
    try:
        offset = float(vszh5_writer._zone_offsets(np.array([now]), None)[0])
    except Exception:                       # an unusable zone: no offset known
        offset = None
    name = zone_name_at(now)
    where = ("%s=%s" % (vszh5_writer.DISPLAY_TZ_ENV, text)
             if text and vszh5_writer.display_zone() is not None else "this computer's time zone")
    parts = [part for part in (label_zone_name(name),
                               "UTC%s now" % vszh5_writer.iso_offset_text(offset)
                               if offset is not None else "") if part]
    return "%s: %s (each sample gets the offset of its own date, daylight saving " \
           "included)" % (where, ", ".join(parts) or "unknown")


# %% Time axes: local ISO 8601, line breaks and axis breaks (0.5.0)
#: (0.5.8) Smallest K of the small-gap rule (the K boxes start there too).
MIN_GAP_K = 1.0


def _finite_seconds(value):
    """A fixed gap in seconds (0.5.8): finite and > 0, else 0.0 (not set)."""
    seconds = float(value or 0.0)
    return seconds if math.isfinite(seconds) and seconds > 0.0 else 0.0


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

    0.5.6 option:
        clocks      - the clocks the date-time x axes are drawn in, any of
                      TIME_CLOCKS ("local", "utc", "mjd"); one set of pages
                      per clock (see "Time-axis clocks"); default local only.
    """

    def __init__(self, k=vszh5_writer.GAP_K, n=vszh5_writer.GAP_AXIS_N, manual_s=0.0,
                 axis_manual_s=0.0, break_lines=True, break_axis=True, numeric=True,
                 clocks=None):
        """Store the gap settings (negative values are clamped to 0).

        k: small-gap factor - a trace's line breaks where its step dt exceeds
        k x median(dt) of that trace.  0.5.8: at least 1 (as the K boxes; a
        K below 1 cut the line at ordinary steps); not a number = the default.
        n: large-gap factor - the x axis breaks where no trace has data for
        dt > n x the largest small-gap threshold; 0 disables axis breaks.
        manual_s / axis_manual_s: fixed small / large gap in seconds; > 0
        overrides k / n (time axes).  0.5.8: not finite (inf, NaN) = 0.
        break_lines / break_axis / numeric: the 0.5.3 options (class text).
        clocks: the 0.5.6 time-axis clocks (time_clocks(); None = local only).
        """
        k = float(vszh5_writer.GAP_K if k is None else k)
        self.k = max(MIN_GAP_K, k) if math.isfinite(k) else float(vszh5_writer.GAP_K)
        n = float(n or 0.0)
        self.n = max(0.0, n) if math.isfinite(n) else 0.0
        self.manual_s = _finite_seconds(manual_s)
        self.axis_manual_s = _finite_seconds(axis_manual_s)
        self.break_lines = bool(break_lines)
        self.break_axis = bool(break_axis)
        self.numeric = bool(numeric)
        self.clocks = time_clocks(clocks)

    def key(self):
        """Tuple of the gap settings; a change re-lays out the axes on save
        (the clocks are compared per axis, 0.5.6)."""
        return (self.k, self.n, self.manual_s, self.axis_manual_s, self.break_lines,
                self.break_axis, self.numeric)

    def describe_clocks(self):
        """The chosen clocks for people (0.5.6), e.g. "Time axes: local time
        (this computer's time zone), UTC, MJD"."""
        words = [CLOCK_TITLES[c] + (" (this computer's time zone)" if c == "local" else "")
                 for c in self.clocks]
        return "Time axes: %s." % ", ".join(words)

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


def _page_of(widget):
    """The page widget ``widget`` is on, or None (0.5.6)."""
    while widget is not None and widget.kind != "page":
        widget = widget.parent
    return widget


def _clock_of(widget):
    """The time-axis clock of the page ``widget`` is on ("local" when the
    page has none yet, or the widget is on no page) (0.5.6)."""
    page = _page_of(widget)
    clock = getattr(page, "time_clock", None) if page is not None else None
    return clock if clock in TIME_CLOCKS else "local"


def _same_values(a, b):
    """True when two arrays hold the same numbers, NaN where the other has
    NaN (0.5.6; numpy.array_equal(equal_nan=...) needs numpy >= 1.19)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def _mjd_name(utc_name):
    """Name of the MJD plot copy of the UTC date dataset ``utc_name`` (0.5.6):
    "<x>__utc" -> "<x>__mjd", "<x>_utc" -> "<x>_mjd", else "<x>_mjd"."""
    for suffix in ("__utc", "_utc"):
        if utc_name.endswith(suffix):
            return utc_name[:-len(suffix)] + suffix[:-3] + "mjd"
    return utc_name + "_mjd"


def _exact_mjd(doc, x_name, kind, values, utc):
    """
    (MJD values, name of a stored dataset holding exactly them or None) of
    one time x (0.5.6).

    An MJD x is its own exact MJD.  A UTC date dataset uses the MJD its
    program gave SetDataDateTime(mjd=...) (a dataset name or an array of
    the same length); without one, UTC seconds / 86400 + 54832 (within an
    ulp of the MJD).  Samples without a UTC time (padding rows, NaN) are
    NaN, so the MJD axis draws exactly the samples the other clocks draw.
    """
    exact, source = None, None
    if kind == "mjd":
        exact, source = np.array(values, dtype=float), x_name
    else:
        given = getattr(doc, "mjd_of", {}).get(x_name)
        if isinstance(given, OffloadedValue):          # 0.5.7: read back from disk
            given = given.load()
        if isinstance(given, str) and given in doc.datasets and doc.datasets[given][0] == "1d":
            candidate, source = doc.datasets[given][1]["data"], given
        else:
            candidate = given
        if candidate is not None and np.asarray(candidate).shape == np.asarray(utc).shape:
            exact = np.array(candidate, dtype=float)
        else:
            source = None
    if exact is None:
        exact = np.asarray(utc, dtype=float) / 86400.0 + MJD_VEUSZ_EPOCH
    exact[~np.isfinite(np.asarray(utc, dtype=float))] = np.nan
    if source is not None and not _same_values(exact, doc.datasets[source][1]["data"]):
        source = None
    return exact, source


def _clock_dataset(doc, clock, x_name, utc_name, kind, values, utc, cut, gapped):
    """
    Plot x dataset of one trace on the UTC or MJD clock (0.5.6).

    ``gapped`` - the trace's y gets gap rows too, so x needs them as well
    (with no cut nothing changes).  The stored dataset itself is drawn when
    it already holds the values; otherwise a plot copy (_plot_name) is
    written: UTC "<x>_utc" (of an MJD x) / "<utc>_gaps", MJD "<utc
    stem>__mjd" / "<...>_gaps".  Copies are listed in doc._clock_copies.
    """
    if clock == "utc":
        display, is_date = np.asarray(utc, dtype=float), True
        base = utc_name if kind == "date" else None
        stem = utc_name if kind == "date" else x_name + "_utc"
    else:
        display, base = _exact_mjd(doc, x_name, kind, values, utc)
        is_date = False
        stem = base or _mjd_name(utc_name)
    with_gaps = bool(gapped and cut.size)
    if base is not None and not with_gaps:
        return base
    name = _plot_name(doc, stem + "_gaps" if with_gaps else stem, (clock, x_name))
    data = vszh5_writer.insert_gaps(display, cut) if with_gaps else display
    if is_date:
        doc.SetDataDateTime(name, data)
    else:
        doc.SetData(name, data)
    if not hasattr(doc, "_clock_copies"):
        doc._clock_copies = {}
    doc._clock_copies[name] = "type_time_" + clock
    return name


#: (0.5.6) Lines of the graph notes written by the time layout.
_TIME_NOTE = "Time axis: "


def _time_note(doc, clock, local_names):
    """The "Time axis:" line of the graph notes for ``clock`` (0.5.6)."""
    if clock == "utc":
        return _TIME_NOTE + "UTC, ISO 8601 (the 'Z' ends every tick)."
    if clock == "mjd":
        return (_TIME_NOTE + "Modified Julian Date, UTC days (MJD 54832 = "
                "2009-01-01T00:00:00Z), 12 significant digits on the ticks.")
    zones = doc.local_zone_names(local_names)
    where = ("the zone %s=%s" % (vszh5_writer.DISPLAY_TZ_ENV,
                                 os.environ.get(vszh5_writer.DISPLAY_TZ_ENV, "").strip())
             if vszh5_writer.display_zone() is not None
             else "the time zone of the computer that made this project")
    found = ", ".join("%s = UTC%s" % (name, offset) if name else "UTC%s" % offset
                      for offset, name in zones)
    return (_TIME_NOTE + "local time in %s%s; every sample has the UTC offset of its own "
            "date (daylight saving included); the stored data are UTC." % (
                where, (" (%s)" % found) if found else ""))


def _set_note_line(graph, prefix, line):
    """Put ``line`` into the graph notes, replacing a line that starts with
    ``prefix`` (0.5.6; _set_gap_note for any prefix)."""
    lines = [old for old in str(graph.settings.get("notes", "")).split("\n")
             if not old.startswith(prefix)]
    lines.append(line)
    graph.settings["notes"] = "\n".join(old for old in lines if old).strip()


def _layout_time_axis(doc, graph, axis):
    """
    Lay out one x axis whose traces are against time (0.5.0; 0.5.6 clocks).

    MJD or Veusz-date x data are stored once (MJD, or a UTC date dataset)
    and drawn on the clock of the axis's page (TimeLayout.clocks, see
    "Time-axis clocks"): local - a local copy "<utc>_local" (the computer's
    clock, DST-aware, ISO 8601 ticks with the UTC offset and the zone
    name); utc - the UTC dataset (ISO 8601 ticks ending in "Z"); mjd - the
    exact MJD on a numeric axis.  A trace whose step exceeds the line-break
    threshold gets NaN rows in its plot copies ("<utc>_local", "<x>_gaps",
    "<y>_gaps"; the stored data stay unchanged). The axis becomes
    axis-broken only for large gaps (n > 0); the gaps are found in UTC
    seconds, so every clock breaks at the same samples.
    """
    layout = getattr(doc, "time_layout", None) or TimeLayout()
    clock = _clock_of(graph)
    if not hasattr(doc, "_plot_copies"):
        doc._plot_copies = {}
    xys = [c for c in graph.WalkWidgets()
           if c.kind == "xy" and c.settings.get("xAxis", "x") == axis.name]
    if not xys:
        return False
    if (axis.time_layout_key == (layout.key(), clock)
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
    if not dated:
        clock = None                          # elapsed seconds: no clock
    local_names, union, thresholds = [], [], []
    utc_union = []                                     # 0.5.8: UTC seconds of every trace
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
        utc_union.append(utc)
        files = combined_sources(doc, x_name, values.size)
        if files is not None:
            # 0.5.9: a combined-axis trace: cut where none of its files has
            # data; the axis-break threshold is the largest of its files'
            # (as on an overlay of the same files)
            cut, file_lines, _pieces = combined_gap_positions(
                seconds, files, lambda piece: vszh5_writer.gap_threshold(
                    piece, layout.k, layout.manual_s))
            threshold = max(file_lines) if file_lines else 0.0
        else:
            threshold = vszh5_writer.gap_threshold(seconds, layout.k, layout.manual_s)
            cut = vszh5_writer.gap_positions(seconds, threshold)
        if not layout.break_lines:                     # 0.5.3: lines across gaps
            cut = cut[:0]
        thresholds.append(threshold)
        y_values = doc.datasets[y_name][1]["data"]
        lined_up = np.asarray(y_values).shape == values.shape
        if clock == "local":
            local = vszh5_writer.local_time_from_veusz(utc)
            local_values = local_veusz_seconds(utc, local)      # 0.5.2: exact sum
            local_name = utc_name + "_local"
            doc.SetDataDateTime(local_name, vszh5_writer.insert_gaps(local_values, cut))
            doc.local_axes[local_name] = local
            doc.local_zones[local_name] = None                   # 0.5.6
            doc.local_of[utc_name] = local_name
            doc.utc_of[local_name] = utc_name
            plot_x = local_name
            union.append(local_values)
            if local_name not in local_names:
                local_names.append(local_name)
            if cut.size and not lined_up:
                # y does not line up with x: draw it unbroken against a
                # local copy without gap rows (never misalign the points).
                plot_x = _plot_name(doc, utc_name + "_local_nogaps", utc_name)
                doc.SetDataDateTime(plot_x, local_values)
                doc.local_axes[plot_x] = local
                doc.local_zones[plot_x] = None
        elif clock in ("utc", "mjd"):                  # 0.5.6
            plot_x = _clock_dataset(doc, clock, x_name, utc_name, kind, values, utc, cut,
                                    gapped=lined_up)
            union.append(utc)                          # breaks found in UTC seconds
        else:
            plot_x = x_name
            if cut.size and lined_up:
                plot_x = _plot_name(doc, x_name + "_gaps", x_name)
                doc.SetData(plot_x, vszh5_writer.insert_gaps(values, cut))
            union.append(values)
        plot_y = y_name
        if cut.size and lined_up:
            plot_y = _plot_name(doc, y_name + "_gaps", (x_name, y_name))
            doc.SetData(plot_y, vszh5_writer.insert_gaps(y_values, cut))
        xy.time_source = (x_name, y_name)
        xy.settings["xData"] = plot_x
        xy.settings["yData"] = plot_y
    # Large gaps break the x axis: only where no trace has data for longer
    # than the large-gap threshold (n x the largest small-gap threshold, or
    # the fixed large gap).
    pairs = []
    axis_threshold = layout.axis_threshold(max(thresholds) if thresholds else 0.0)
    if axis_threshold > 0 and clock == "local":
        # 0.5.8: found in UTC seconds and moved onto the local axis, so a
        # daylight-saving change is neither a gap (spring forward: a 1-hour
        # jump of the local clock) nor hides one (fall back: a 3-hour hole
        # is 2 hours on the local clock); before, the local page was broken
        # by the local steps and could differ from the UTC and MJD pages.
        utc_all = np.concatenate(utc_union)
        pairs = _clock_break_pairs(utc_all, np.concatenate(union),
                                   vszh5_writer.axis_break_pairs(utc_all, axis_threshold))
    elif axis_threshold > 0:
        pairs = vszh5_writer.axis_break_pairs(np.concatenate(union), axis_threshold)
    if pairs and clock == "mjd":
        # 0.5.6: found in UTC seconds (the same samples as the other
        # clocks), drawn in MJD days; the breaks lie inside the gaps (a
        # margin of the gap is kept at each end), far from any sample.
        pairs = [(a / 86400.0 + MJD_VEUSZ_EPOCH, b / 86400.0 + MJD_VEUSZ_EPOCH)
                 for a, b in pairs]
    if pairs:
        axis.kind = "axis-broken"
        axis.settings["breakPoints"] = [float(v) for pair in pairs for v in pair]
    else:
        axis.kind = "axis"
        axis.settings.pop("breakPoints", None)
    if clock == "local":
        fmt, text = doc.local_axis_text(local_names, "Local time")
        axis.settings["mode"] = "datetime"
    elif clock == "utc":
        fmt, text = UTC_DATETIME_FORMAT, UTC_AXIS_LABEL
        axis.settings["mode"] = "datetime"
    elif clock == "mjd":
        fmt, text = NUMERIC_TICK_FORMAT, MJD_AXIS_LABEL
        axis.settings["mode"] = "numeric"
    if clock is not None:
        axis.settings["TickLabels/format"] = fmt
        axis.settings.setdefault("TickLabels/rotate", str(vszh5_writer.LOCAL_DATETIME_ROTATE))
        axis.settings["label"] = text
        axis.settings.setdefault("autoRange", "exact")
        _set_note_line(graph, _TIME_NOTE, _time_note(doc, clock, local_names))
    axis.time_clock = clock
    axis.time_layout_key = (layout.key(), _clock_of(graph))
    _set_gap_note(graph, layout.describe())
    return True


def _clock_break_pairs(utc_values, clock_values, utc_pairs,
                       margin=vszh5_writer.GAP_AXIS_MARGIN):
    """
    Axis breaks found in UTC seconds, moved onto another clock's axis (0.5.8).

    ``utc_values`` / ``clock_values``: the UTC seconds and the clock's value
    (e.g. local Veusz seconds) of the same samples; ``utc_pairs``: the
    vszh5_writer.axis_break_pairs of ``utc_values``.  Each break separates
    the samples before a UTC gap from those after it; on the clock's axis it
    runs from the latest clock value before the gap to the earliest after
    it, shrunk by ``margin`` of that length at each end (the writer's rule).
    Where the clock leaves no room (a fall-back hour folded into a short
    gap) that break is left out: no clock value may be hidden.  Without a
    daylight-saving change the result is the writer's own pairs of the
    clock values.
    """
    utc = np.asarray(utc_values, dtype=float).ravel()
    other = np.asarray(clock_values, dtype=float).ravel()
    keep = np.isfinite(utc) & np.isfinite(other)
    utc, other = utc[keep], other[keep]
    if not utc_pairs or utc.size < 2:
        return []
    order = np.argsort(utc, kind="stable")
    utc, other = utc[order], other[order]
    before = np.maximum.accumulate(other)              # latest clock value so far
    after = np.minimum.accumulate(other[::-1])[::-1]   # earliest clock value from here
    pairs = []
    for start, stop in utc_pairs:
        last = int(np.searchsorted(utc, start, side="right")) - 1
        first = int(np.searchsorted(utc, stop, side="left"))
        if last < 0 or first >= utc.size or first <= last:
            continue
        low, high = float(before[last]), float(after[first])
        if high <= low:
            continue                                   # no room on this clock
        pad = float(margin) * (high - low)
        pairs.append((low + pad, high - pad))
    return pairs


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
        files = combined_sources(doc, x_name, values.size)
        if files is not None:
            # 0.5.9: a combined-axis trace: cut where none of its files has
            # data; the axis breaks see its files as an overlay would
            cut, _lines, pieces = combined_gap_positions(
                values, files, rule.line_threshold, rule.log_threshold,
                break_lines=rule.break_lines)
            series.extend(pieces)
        else:
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
    clock_copies = getattr(doc, "_clock_copies", {})
    for name in [n for n in copies if n not in used and n in doc.datasets]:
        if (name.endswith("_gaps") or "_gaps_" in name or name.endswith("_local_nogaps")
                or name in clock_copies):              # 0.5.6: UTC / MJD copies
            del doc.datasets[name]
            for members in getattr(doc, "tags", {}).values():
                if name in members:
                    members.remove(name)
            del copies[name]
            clock_copies.pop(name, None)
            getattr(doc, "local_axes", {}).pop(name, None)
            getattr(doc, "local_zones", {}).pop(name, None)


# %% Pages per clock (0.5.6)
def _has_clock_axis(page):
    """True when ``page`` draws a date-time x axis laid out on a clock."""
    return any(w.kind in ("axis", "axis-broken") and w.time_clock in TIME_CLOCKS
               for w in page.WalkWidgets())


def _free_page_name(root, wanted, page):
    """``wanted``, or wanted_2, wanted_3 ... when another page of ``root``
    already has that name (``page`` itself does not count)."""
    taken = set(c.name for c in root.children if c is not page)
    name, number = wanted, 2
    while name in taken:
        name = "%s_%d" % (wanted, number)
        number += 1
    return name


def _copy_setting(value):
    """A setting value for a page copy: lists and dicts copied, the rest
    (text, numbers, booleans) shared."""
    return copy.deepcopy(value) if isinstance(value, (list, dict, tuple)) else value


def _clone_widget(widget, parent):
    """
    Copy of ``widget`` and every widget below it, under ``parent`` (0.5.6).

    The copy has the settings, the data sources of the layouts (time_source,
    gap_source and the numeric gap key) and the tick-label fit records of
    the original; its time axes are laid out again (time_layout_key None).
    Datasets are shared, never copied.
    """
    twin = NativeWidget(widget.document, widget.kind, widget.name, parent)
    twin.auto_named = False
    twin.settings = OrderedDict((key, _copy_setting(value))
                                for key, value in widget.settings.items())
    twin.time_source = widget.time_source
    twin.gap_source = widget.gap_source
    twin.gap_layout_key = widget.gap_layout_key
    twin.time_layout_key = None
    twin.time_clock = widget.time_clock if widget.kind != "page" else None
    twin.local_time = widget.local_time
    twin.fit_base = dict((k, _copy_setting(v)) for k, v in widget.fit_base.items())
    twin.fit_written = dict((k, _copy_setting(v)) for k, v in widget.fit_written.items())
    twin.children = [_clone_widget(child, twin) for child in widget.children]
    return twin


def _remove_clock_pages(doc):
    """Take the clock copies of the last finalize out of the project (0.5.6);
    finalize makes them again from their pages, so a copy always shows its
    page as it is now.  Their plot copies go with _drop_unused_plot_copies
    when no page draws them any more."""
    for page in [c for c in doc.Root.children if c.kind == "page" and c.time_clone_of is not None]:
        doc.Root.children.remove(page)
        for widget in page.WalkWidgets():
            widget.time_clone_of = None
        page.parent = None


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


# %% Tick labels that all show (0.5.5)
# Every page is fitted on save, after the time and gap layouts: each axis's
# tick labels are worked out the way Veusz will draw them
# (vszh5_writer.axis_tick_room, a port of Veusz's tick choice and number
# formats), an x axis with date-times or long numbers gets vertical labels
# (-90 degrees, reading downwards; a date-time on two lines, the date and
# "T" + the time), every graph gets the bottom and left margins its labels
# and axis labels need (the largest of a grid, so its plots line up), and a
# page grows - never shrinks - until each plot keeps MIN_PLOT_CM and room for
# every label; a broken axis whose parts are too short gets fewer major
# ticks per part (MajorTicks/number).  What the fit writes is undone before
# the next fit, so saving twice gives the same file.
_MISSING = object()


def _fit_set(widget, key, value):
    """Set ``key`` on ``widget`` for the tick-label fit, remembering the value
    it had before (0.5.5)."""
    if key not in widget.fit_base:
        widget.fit_base[key] = widget.settings.get(key, _MISSING)
    widget.settings[key] = value
    widget.fit_written[key] = value


def _fit_restore(widget):
    """Undo the previous tick-label fit on ``widget`` (0.5.5): a setting the
    fit wrote and nobody changed since gets its earlier value back."""
    for key, value in widget.fit_written.items():
        if widget.settings.get(key, _MISSING) == value:
            before = widget.fit_base.get(key, _MISSING)
            if before is _MISSING:
                widget.settings.pop(key, None)
            else:
                widget.settings[key] = before
    widget.fit_base = {}
    widget.fit_written = {}


def _cm(value, default):
    """A Veusz distance setting in cm (``default`` when unset or not physical)."""
    if value is None or value is _MISSING:
        return float(default)
    found = vszh5_writer._distance_cm(str(value))
    return float(default) if found is None else found


def _pt(value, default=None):
    """A Veusz font size setting in points (``default`` when unset)."""
    found = _cm(value, -1.0) if value is not None else -1.0
    return default if found < 0 else found / (2.54 / 72.0)


def _finite_range(doc, name, cache):
    """(min, max) of the finite values of dataset ``name`` (1-D or date), or
    None; cached per finalize in ``cache``."""
    if name in cache:
        return cache[name]
    found = None
    entry = doc.datasets.get(name) if isinstance(name, str) else None
    if entry is not None and entry[0] in ("1d", "date"):
        values = np.asarray(entry[1]["data"], dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            found = (float(values.min()), float(values.max()))
    elif entry is not None and entry[0] == "2d":
        found = None                              # images: their own x/y ranges below
    cache[name] = found
    return found


def _axis_data_range(doc, graph, axis, cache):
    """(min, max) of the data a graph draws on ``axis`` (xy traces and
    images), or None when it draws nothing known (0.5.5)."""
    horizontal = axis.settings.get("direction", "horizontal") != "vertical"
    axis_key, data_key = ("xAxis", "xData") if horizontal else ("yAxis", "yData")
    default_axis = "x" if horizontal else "y"
    lows, highs = [], []
    for child in graph.WalkWidgets():
        if child.kind == "xy" and child.settings.get(axis_key, default_axis) == axis.name:
            name = child.settings.get(data_key)
            if name is None and horizontal:          # no x data: Veusz uses 1 .. N
                y = _finite_range(doc, child.settings.get("yData"), cache)
                entry = doc.datasets.get(child.settings.get("yData"))
                if y is not None and entry is not None:
                    lows.append(1.0)
                    highs.append(float(np.asarray(entry[1]["data"]).size))
                continue
            found = _finite_range(doc, name, cache)
            if found is not None:
                lows.append(found[0])
                highs.append(found[1])
        elif child.kind == "image" and child.settings.get(axis_key, default_axis) == axis.name:
            entry = doc.datasets.get(child.settings.get("data"))
            if entry is not None and entry[0] == "2d":
                parts = entry[1]
                span = parts.get("xrange" if horizontal else "yrange")
                edges = parts.get("xedge" if horizontal else "yedge")
                if span is not None and len(span) == 2:
                    lows.append(float(min(span)))
                    highs.append(float(max(span)))
                elif edges is not None and len(edges):
                    lows.append(float(np.nanmin(edges)))
                    highs.append(float(np.nanmax(edges)))
    if not lows:
        return None
    return (min(lows), max(highs))


def _axis_room(doc, graph, axis, length_cm, cache):
    """vszh5_writer.TickRoom of one axis as it is set now (0.5.5); None for a
    hidden axis, a label axis or an axis without known data."""
    settings = axis.settings
    if settings.get("hide", False) or settings.get("TickLabels/hide", False):
        return None
    mode = settings.get("mode", "numeric")
    if mode not in ("numeric", "datetime"):
        return None
    data_range = _axis_data_range(doc, graph, axis, cache)
    dated = mode == "datetime"
    if data_range is None and not dated:
        return None
    horizontal = settings.get("direction", "horizontal") != "vertical"
    # NativeWidget._write gives a date axis 8 pt ticks and a 10 pt label.
    tick_pt = _pt(settings.get("TickLabels/size"), 8.0 if dated else None)
    label_pt = _pt(settings.get("Label/size"), 10.0 if dated else vszh5_writer.VEUSZ_TEXT_PT)
    rotate = settings.get("TickLabels/rotate")
    try:
        major = int(settings.get("MajorTicks/number", vszh5_writer.VEUSZ_MAJOR_TICKS))
    except (TypeError, ValueError):
        major = vszh5_writer.VEUSZ_MAJOR_TICKS
    return vszh5_writer.axis_tick_room(
        "horizontal" if horizontal else "vertical", mode, data_range=data_range,
        tick_format=str(settings.get("TickLabels/format", "") or ""),
        log=bool(settings.get("log", False)), axis_length_cm=length_cm,
        label=str(settings.get("label", "") or ""), label_size_pt=label_pt,
        tick_size_pt=tick_pt if (dated or "TickLabels/size" in settings) else None,
        autorange=str(settings.get("autoRange", "next-tick")),
        axis_min=settings.get("min", "Auto"), axis_max=settings.get("max", "Auto"),
        breaks=list(settings.get("breakPoints", ()) or ()) if axis.kind == "axis-broken" else (),
        major_ticks=major, rotate=rotate)


#: (0.5.6) Room kept beside a date-time axis label that sets the plot width.
_LABEL_PAD_CM = 0.4


def _time_label_cm(axis):
    """
    Width (cm) a date-time x axis's label needs (0.5.6), 0 for other axes.

    The label of a time axis names the clock, the offset(s) and the zone
    ("Local time (ISO 8601, EST, UTC-05:00)", "... from 2026-11-01T01:00");
    the plot is made at least that wide, so the label never runs past its
    graph into the next one.  Widths as for tick labels (tick_text_width_em).
    """
    settings = axis.settings
    if axis.time_clock not in TIME_CLOCKS and settings.get("mode") != "datetime":
        return 0.0
    label = str(settings.get("label", "") or "")
    if not label or settings.get("hide", False):
        return 0.0
    dated = settings.get("mode") == "datetime"
    size = _pt(settings.get("Label/size"), 10.0 if dated else vszh5_writer.VEUSZ_TEXT_PT)
    return vszh5_writer.tick_text_width_em(label) * size * (2.54 / 72.0) + _LABEL_PAD_CM


def _dated_page_size(page, graphs, columns):
    """(width, height) cm the 0.5.0 rule gives a page with date-time axes and
    no size of its own: 13 cm a column and 8 cm a row (at least 16 cm) for a
    grid, 25 x 16 cm for one graph; None for other pages."""
    if not any(w.kind in ("axis", "axis-broken") and w.settings.get("mode") == "datetime"
               for w in page.WalkWidgets()):
        return None
    if columns:
        rows = int(math.ceil(len(graphs) / float(columns)))
        return 13.0 * columns, max(16.0, 8.0 * rows)
    return 25.0, 16.0


def _fit_page(doc, page, cache=None):
    """
    Fit the tick labels of every graph on ``page`` (0.5.5): vertical labels
    where they belong, the margins every label needs and a page large enough
    (see the section text).  Graphs in a grid share the largest margins.
    """
    cache = {} if cache is None else cache
    graphs = [w for w in page.WalkWidgets() if w.kind == "graph"]
    for widget in [page] + [w for w in page.WalkWidgets() if w.kind in ("grid", "graph", "axis",
                                                                        "axis-broken")]:
        _fit_restore(widget)
    if not graphs:
        return
    grid = next((c for c in page.children if c.kind == "grid"), None)
    columns = 0
    if grid is not None:
        try:
            columns = max(1, int(grid.settings.get("columns", 2) or 2))
        except (TypeError, ValueError):
            columns = 2
    in_grid = [g for g in graphs if grid is not None and g.parent is grid]
    rows = int(math.ceil(len(in_grid) / float(columns))) if in_grid else 0
    defaults = vszh5_writer.VEUSZ_GRAPH_MARGINS_CM
    grid_margins = vszh5_writer.VEUSZ_GRID_MARGINS_CM
    base_size = _dated_page_size(page, in_grid if in_grid else graphs, columns if in_grid else 0)
    # Veusz uses only the columns its graphs fill (grid._gridengine): one
    # graph in a 2-column grid spans the whole grid.
    columns = min(columns, len(in_grid)) if in_grid else columns
    width = _cm(page.settings.get("width"), base_size[0] if base_size else
                vszh5_writer.VEUSZ_PAGE_CM[0])
    height = _cm(page.settings.get("height"), base_size[1] if base_size else
                 vszh5_writer.VEUSZ_PAGE_CM[1])
    page_set = {"width": "width" in page.settings, "height": "height" in page.settings}
    gm = dict((side, _cm(grid.settings.get(side), grid_margins[side]) if grid is not None else 0.0)
              for side in grid_margins)

    def margins_of(graph):
        """The graph's own margins (cm), Veusz defaults where unset."""
        return dict((side, _cm(graph.settings.get(side), defaults[side])) for side in defaults)

    def plot_lengths(graph, margin):
        """(x length, y length) of a graph's plot area on the page (cm)."""
        if graph in in_grid:
            cell_w = (width - gm["leftMargin"] - gm["rightMargin"]) / columns
            cell_h = (height - gm["topMargin"] - gm["bottomMargin"]) / max(rows, 1)
        else:
            cell_w, cell_h = width, height
        return (cell_w - margin["leftMargin"] - margin["rightMargin"],
                cell_h - margin["topMargin"] - margin["bottomMargin"])

    plans = {}
    need = {}
    for _pass in range(2):
        # 1. every axis's labels (pass 2: with the axis lengths on the page)
        for graph in graphs:
            margin = need.get(graph) or margins_of(graph)
            lengths = plot_lengths(graph, margin) if _pass else (None, None)
            rooms = []
            for axis in [c for c in graph.children if c.kind in ("axis", "axis-broken")]:
                horizontal = axis.settings.get("direction", "horizontal") != "vertical"
                room = _axis_room(doc, graph, axis, lengths[0] if horizontal else None, cache)
                if room is not None:
                    rooms.append((axis, horizontal, room))
            plans[graph] = rooms
        # 2. margins: each graph's own, or the largest of its grid
        bottoms, lefts = {}, {}
        for graph, rooms in plans.items():
            margin = margins_of(graph)
            bottoms[graph] = max([margin["bottomMargin"]] + [r.room_cm for _a, h, r in rooms if h])
            lefts[graph] = max([margin["leftMargin"]] + [r.room_cm for _a, h, r in rooms if not h])
        shared_bottom = max([bottoms[g] for g in in_grid] or [0.0])
        shared_left = max([lefts[g] for g in in_grid] or [0.0])
        for graph in graphs:
            margin = margins_of(graph)
            margin["bottomMargin"] = shared_bottom if graph in in_grid else bottoms[graph]
            margin["leftMargin"] = shared_left if graph in in_grid else lefts[graph]
            need[graph] = margin
        # 3. the page: each plot keeps MIN_PLOT_CM and room for its labels
        need_w, need_h = 0.0, 0.0
        cell_w = cell_h = 0.0
        for graph, rooms in plans.items():
            margin = need[graph]
            plot_w = max([vszh5_writer.MIN_PLOT_CM[0]] +
                         [r.min_length_cm for _a, h, r in rooms if h and r.vertical] +
                         [_time_label_cm(a) for a, h, _r in rooms if h])     # 0.5.6
            plot_h = max([vszh5_writer.MIN_PLOT_CM[1]] + [r.min_length_cm for _a, h, r in rooms if not h])
            w_cm = margin["leftMargin"] + plot_w + margin["rightMargin"]
            h_cm = margin["topMargin"] + plot_h + margin["bottomMargin"]
            if graph in in_grid:
                cell_w, cell_h = max(cell_w, w_cm), max(cell_h, h_cm)
            else:
                need_w, need_h = max(need_w, w_cm), max(need_h, h_cm)
        if in_grid:
            need_w = max(need_w, gm["leftMargin"] + gm["rightMargin"] + columns * cell_w)
            need_h = max(need_h, gm["topMargin"] + gm["bottomMargin"] + rows * cell_h)
        width, height = max(width, need_w), max(height, need_h)
    # Write what changed (undone by the next fit).
    for key, value, explicit, default in (
            ("width", width, page_set["width"], base_size[0] if base_size else None),
            ("height", height, page_set["height"], base_size[1] if base_size else None)):
        current = _cm(page.settings.get(key), default if default is not None else
                      vszh5_writer.VEUSZ_PAGE_CM[0])
        if explicit and value <= current + 1e-9:
            continue
        if not explicit and default is None and value <= vszh5_writer.VEUSZ_PAGE_CM[0] + 1e-9:
            continue                          # Veusz's own 15 cm page is big enough
        _fit_set(page, key, vszh5_writer.margin_text(value) if value > (default or 0) + 1e-9
                 else "%gcm" % value)
    for graph in graphs:
        own = margins_of(graph)
        for side in ("bottomMargin", "leftMargin"):
            if need[graph][side] > own[side] + 1e-9:
                _fit_set(graph, side, vszh5_writer.margin_text(need[graph][side]))
        for axis, _horizontal, room in plans.get(graph, ()):
            for key, value in room.settings.items():
                _fit_set(axis, key, value)


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


# %% Offloading plot data to disk above a RAM limit (0.5.7)
# When the RAM in use on this computer - every program, not only this one -
# reaches the limit the user set (ram_limit(): % of the computer's RAM;
# default 80 %; 0 = never), the project's datasets (NativeDocument) and the
# data kept from the files read (KeptReads) are written to a pickle file and
# dropped from RAM.  Below the limit nothing is written.  The data are read
# back from the file wherever they are used (time and gap layout, the Plots
# tab) and are converted into the .vszh5 when the project is saved, one
# dataset at a time.  Values come back exactly as they were (pickle stores
# float64 arrays and text bit for bit).  The file is deleted when the
# project is cleared or released, or when the program ends.
#: (0.5.7) Default RAM limit: offload above this much of the computer's RAM in use (%).
DEFAULT_RAM_LIMIT_PERCENT = 80.0
#: (0.5.7) Folder for the offload files (default: the system's temporary folder).
OFFLOAD_DIR_ENV = "AUTOPLOT_OFFLOAD_DIR"
#: (0.5.7) Datasets smaller than this (bytes of values) stay in RAM.
OFFLOAD_MIN_BYTES = 512
#: (0.5.7) RAM use is read at most this often while datasets are added (s).
_RAM_CHECK_S = 0.2
#: (0.5.7) Freed heap is handed back to the system at most this often (s).
_HEAP_TRIM_S = 2.0
#: (0.5.7) The program-wide RAM limit (set_ram_limit; one per process).
_RAM_LIMIT = {"percent": DEFAULT_RAM_LIMIT_PERCENT}
#: (0.5.7) Offload messages are logged at most this often (s), with totals.
_NOTE_S = 10.0


class _OffloadNotes(object):
    """Throttled offload messages (0.5.7): the first at once, then at most one
    every _NOTE_S with the totals so far; flush() logs the last one held back
    (the end of a batch).  A thousand small offloads make a few log lines."""

    def __init__(self):
        """Nothing logged yet."""
        self.last = None
        self.pending = None

    def add(self, text, log, messages, now=None):
        """Log ``text`` now, or hold it until _NOTE_S has passed."""
        now = time.monotonic() if now is None else now
        if self.last is None or now - self.last >= _NOTE_S:
            self._emit(text, log, messages, now)
        else:
            self.pending = text

    def flush(self, log, messages):
        """Log the message held back, if any."""
        if self.pending is not None:
            self._emit(self.pending, log, messages, time.monotonic())

    def _emit(self, text, log, messages, now):
        """Keep ``text`` in ``messages`` and give it to ``log``."""
        self.last, self.pending = now, None
        if messages is not None:
            messages.append(text)
        if log is not None:
            try:
                log(text)
            except Exception:
                pass


def set_ram_limit(percent):
    """Set the RAM limit (% of this computer's RAM in use, every program)
    above which plot data are offloaded to disk (0.5.7); 0 or None = never.
    Values are clamped to 0 .. 100.  Returns the limit now in force."""
    value = 0.0 if percent is None else float(percent)
    if not math.isfinite(value):
        value = 0.0
    _RAM_LIMIT["percent"] = min(max(value, 0.0), 100.0)
    return _RAM_LIMIT["percent"]


def ram_limit():
    """The RAM limit in force (%; 0 = never offload) (0.5.7)."""
    return _RAM_LIMIT["percent"]


def system_memory():
    """
    (percent in use, total bytes, available bytes) of this computer's RAM
    (0.5.7), all programs together; None when it cannot be read.

    psutil.memory_physical() (psutil 8; virtual_memory() before) when
    installed; otherwise /proc/meminfo (Linux:
    MemTotal and MemAvailable) or GlobalMemoryStatusEx (Windows) through
    ctypes.  "In use" = total - available, as psutil computes its percent.
    """
    try:
        import psutil  # type: ignore
        # psutil 8 names it memory_physical(); virtual_memory() stays as an alias
        reader = getattr(psutil, "memory_physical", None) or psutil.virtual_memory
        found = reader()
        return float(found.percent), float(found.total), float(found.available)
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _sep, value = line.partition(":")
                parts = value.split()
                if parts:
                    info[key.strip()] = float(parts[0]) * 1024.0
        total = info["MemTotal"]
        available = info.get("MemAvailable", info.get("MemFree", 0.0))
        return 100.0 * (total - available) / total, total, available
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _Status(ctypes.Structure):
                """MEMORYSTATUSEX (kernel32)."""
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = float(status.ullTotalPhys)
                available = float(status.ullAvailPhys)
                return 100.0 * (total - available) / total, total, available
        except Exception:
            pass
    return None


def memory_text():
    """One line for the GUI (0.5.7), e.g. "this computer: 31.7 GB of RAM,
    42 % in use now (all programs)"; "RAM use unknown" when unreadable."""
    found = system_memory()
    if found is None:
        return "RAM use of this computer unknown (psutil not installed?)"
    return ("this computer: %.1f GB of RAM, %.0f %% in use now (all programs)"
            % (found[1] / 1e9, found[0]))


def payload_bytes(value, _depth=0):
    """Bytes of the values in ``value`` (0.5.7): numpy arrays by nbytes,
    text and lists estimated, dicts / lists / tuples summed (nested to 4
    levels).  Used to choose what is worth offloading and to report it."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes) if value.dtype.kind != "O" else int(value.size) * 64
    if isinstance(value, (bytes, str)):
        return len(value)
    if _depth > 4:
        return 0
    if isinstance(value, dict):
        return sum(payload_bytes(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        if len(value) > 256:               # a text dataset: estimate, do not walk
            return 56 * len(value)
        return sum(payload_bytes(v, _depth + 1) for v in value)
    return 0


class OffloadStore(object):
    """
    One append-only pickle file of offloaded values (0.5.7).

    put() pickles a value (protocol HIGHEST_PROTOCOL) to the end of the file
    and returns (offset, length); get() reads it back.  The file is made in
    AUTOPLOT_OFFLOAD_DIR (or the system's temporary folder) on the first
    put(); close() - or the program ending - deletes it.  A value that is
    replaced stays in the file until then (the file only grows).  Safe to use
    from several threads.
    """

    def __init__(self, prefix="autoplot_offload_", folder=None):
        """No file yet; ``prefix`` starts its name, ``folder`` overrides the
        environment variable and the temporary folder."""
        self.prefix = prefix
        self.folder = folder or os.environ.get(OFFLOAD_DIR_ENV) or None
        self.path = None
        self.size = 0
        self.count = 0
        self._handle = None
        self._finalizer = None
        self._lock = threading.RLock()

    @staticmethod
    def _remove(handle, path):
        """Close and delete an offload file (finalizer: no reference to the store)."""
        try:
            handle.close()
        except Exception:
            pass
        try:
            os.remove(path)
        except OSError:
            pass

    def _open(self):
        """Make the file on first use (private: mkstemp gives the owner only access)."""
        if self._handle is None:
            descriptor, path = tempfile.mkstemp(prefix=self.prefix, suffix=".pkl", dir=self.folder)
            self._handle = os.fdopen(descriptor, "w+b")
            self.path = path
            self._finalizer = weakref.finalize(self, OffloadStore._remove, self._handle, path)

    def free_bytes(self):
        """Free space (bytes) where the file is or will be; None when unknown."""
        try:
            return float(shutil.disk_usage(self.folder or tempfile.gettempdir()).free)
        except OSError:
            return None

    def put(self, value):
        """Write ``value`` to the end of the file; (offset, length)."""
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            self._open()
            self._handle.seek(0, os.SEEK_END)
            offset = self._handle.tell()
            self._handle.write(blob)
            self._handle.flush()
            self.size = offset + len(blob)
            self.count += 1
        return offset, len(blob)

    def get(self, offset, length):
        """The value written at ``offset`` (``length`` bytes), unpickled."""
        with self._lock:
            if self._handle is None:
                raise IOError("the offload file was closed (project cleared)")
            self._handle.seek(offset)
            blob = self._handle.read(length)
        if len(blob) != length:
            raise IOError("offload file %s: %d of %d bytes at %d" % (self.path, len(blob),
                                                                     length, offset))
        return pickle.loads(blob)

    def close(self):
        """Close and delete the file; the store can be used again (new file)."""
        with self._lock:
            if self._finalizer is not None:
                self._finalizer()
            self._handle = None
            self._finalizer = None
            self.path = None
            self.size = 0
            self.count = 0


class OffloadedValue(object):
    """A value in an OffloadStore (0.5.7); load() reads it back (a new copy
    every time; nothing is kept in RAM)."""
    __slots__ = ("store", "offset", "length", "nbytes")

    def __init__(self, store, offset, length, nbytes):
        """Where the value is: ``store``, ``offset``, ``length``; ``nbytes`` of values."""
        self.store, self.offset, self.length, self.nbytes = store, offset, length, nbytes

    def load(self):
        """The value, unpickled from the file."""
        return self.store.get(self.offset, self.length)


class OffloadedParts(Mapping):
    """
    The arrays of a dataset kept in an OffloadStore (0.5.7): a read-only
    mapping like the dict it replaces ({"data": ..., "serr": ...}).  Asking
    for a part reads the dataset back from the file; load() gives every
    part at once.  The key names are kept in RAM, so ``"serr" in parts`` and
    ``parts.get("serr")`` (None when absent) need no read when absent.
    """
    __slots__ = ("_value", "_keys")

    def __init__(self, value, keys):
        """``value``: the OffloadedValue of the parts dict; ``keys``: its keys."""
        self._value = value
        self._keys = tuple(keys)

    @property
    def nbytes(self):
        """Bytes of the values (as when they were in RAM)."""
        return self._value.nbytes

    def load(self):
        """Every part (a new dict of new arrays)."""
        return self._value.load()

    def __getitem__(self, key):
        """One part, read from the file."""
        if key not in self._keys:
            raise KeyError(key)
        return self.load()[key]

    def __contains__(self, key):
        """True when the dataset has the part ``key`` (no read)."""
        return key in self._keys

    def get(self, key, default=None):
        """The part ``key`` read from the file, ``default`` when absent."""
        return self.load()[key] if key in self._keys else default

    def __iter__(self):
        """The part names."""
        return iter(self._keys)

    def __len__(self):
        """Number of parts."""
        return len(self._keys)

    def __repr__(self):
        """Short description (no read)."""
        return "OffloadedParts(%s, %d bytes in %s)" % (
            ", ".join(self._keys), self.nbytes, self._value.store.path)


def loaded_parts(parts):
    """``parts`` as a plain dict of arrays (0.5.7): offloaded parts are read
    back once; parts in RAM are returned as they are."""
    return parts.load() if isinstance(parts, OffloadedParts) else parts


def _over_ram_limit(limit):
    """(percent in use, limit) when this computer's RAM use is at or above
    ``limit`` (%), else None; None too when the limit is 0 or the use is
    unknown (0.5.7)."""
    if not limit or limit <= 0.0:
        return None
    found = system_memory()
    if found is None or found[0] < limit:
        return None
    return found[0], limit


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

    0.5.7: when this computer's RAM use reaches the limit (ram_limit(), or
    ``ram_limit_percent``), keep() moves the kept data to a pickle file
    (OffloadStore); get() and items() read them back (new copies each time).
    Data that cannot be pickled stay in RAM.
    """

    def __init__(self):
        """Nothing kept yet."""
        self._data = {}
        self._stamps = {}
        self.ram_limit_percent = None     # 0.5.7: None = the program's limit
        self.offload_log = None           # 0.5.7: callable(text)
        self._offload = None
        self._offload_totals = [0, 0]     # 0.5.7: files and bytes moved so far
        self._notes = _OffloadNotes()

    def _value(self, path, default=None):
        """Kept data of ``path``, read back from disk when offloaded."""
        value = self._data.get(path, default)
        return value.load() if isinstance(value, OffloadedValue) else value

    def offload(self, percent=None, limit=None):
        """Move every kept file's data held in RAM to the offload file (0.5.7);
        returns how many files moved."""
        if self._offload is None:
            self._offload = OffloadStore("autoplot_kept_")
        moved, total, failed = 0, 0, []
        for path, value in list(self._data.items()):
            if isinstance(value, OffloadedValue):
                continue
            try:
                offset, length = self._offload.put(value)
            except Exception as exc:          # not picklable (a file handle ...): keep it
                failed.append("%s (%s)" % (os.path.basename(str(path)), type(exc).__name__))
                continue
            size = payload_bytes(value)
            self._data[path] = OffloadedValue(self._offload, offset, length, size)
            moved += 1
            total += size
        value = None
        self._offload_totals[0] += moved
        self._offload_totals[1] += total
        why = ("RAM in use on this computer %.1f %% >= limit %g %%" % (percent, limit)
               if percent is not None else "offload asked for")
        if failed:                                # always at once
            self._notes._emit("%s: kept in RAM (cannot be written to disk): %s"
                              % (why, ", ".join(failed)), self.offload_log, None,
                              time.monotonic())
        if moved:
            self._notes.add("%s: data of %d file(s) read, %.1f MB, moved to disk so far (%s)"
                            % (why, self._offload_totals[0], self._offload_totals[1] / 1e6,
                               self._offload.path), self.offload_log, None)
        return moved

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
        limit = self.ram_limit_percent if self.ram_limit_percent is not None else ram_limit()
        found = _over_ram_limit(limit)                  # 0.5.7
        if found is not None:
            self.offload(found[0], found[1])

    def forget(self, path):
        """Drop the data of ``path``."""
        self._data.pop(path, None)
        self._stamps.pop(path, None)

    def get(self, path, default=None):
        """Kept data of ``path`` (``default`` when none; 0.5.7: read back from
        disk when offloaded)."""
        return self._value(path, default)

    def items(self, paths):
        """(path, data) of every listed path with kept data, in list order
        (0.5.7: offloaded data read back; the last offload message held back
        is logged - items() follows a batch)."""
        self._notes.flush(self.offload_log, None)
        return [(path, self._value(path)) for path in dict.fromkeys(paths) if path in self._data]

    def clear(self):
        """Forget every file's data (0.5.7: and delete the offload file)."""
        self._data = {}
        self._stamps = {}
        if self._offload is not None:
            self._offload.close()
            self._offload = None
        self._offload_totals = [0, 0]
        self._notes = _OffloadNotes()


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
        self.local_zones = {}         # 0.5.6: local dataset name -> its zone (None = computer)
        self.mjd_of = {}              # 0.5.6: UTC date dataset -> exact MJD (name or array)
        self.combined_of = {}         # 0.5.9: combined x dataset -> file of every row
        self._clock_copies = {}       # 0.5.6: UTC / MJD plot copy -> its type tag
        # 0.5.7: offloading to disk above the RAM limit (see "Offloading")
        self.ram_limit_percent = None  # None = the program's limit (ram_limit())
        self.offload_log = None        # callable(text) for the offload messages
        self.offload_messages = []     # every offload message of this project
        self._offload = None           # OffloadStore (made on the first offload)
        self._ram_checked = 0.0        # time.monotonic() of the last RAM reading
        self._heap_trimmed = 0.0       # time.monotonic() of the last heap trim
        self._offload_totals = [0, 0]  # datasets and bytes moved to disk so far
        self._notes = _OffloadNotes()  # throttled offload messages
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
            widget.time_clone_of = None            # 0.5.6
        cut(self.Root)
        self.Root = NativeWidget(self, "document", "")
        self.datasets = OrderedDict()
        self.tags = {}
        self.local_axes = {}
        self.local_of = {}
        self.utc_of = {}
        self.local_zones = {}                      # 0.5.6
        self.mjd_of = {}
        self.combined_of = {}                      # 0.5.9
        self._clock_copies = {}
        if hasattr(self, "_plot_copies"):
            self._plot_copies = {}
        if getattr(self, "_offload", None) is not None:   # 0.5.7: the file goes too
            self._offload.close()
            self._offload = None

    def _put(self, name, kind, parts):
        """Store dataset ``name`` of ``kind`` ('1d', 'date', 'text', '2d') with its
        arrays ``parts``; rejects empty names, NUL characters, '.' and '..'.
        """
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("Dataset name must be nonempty text without NUL")
        if name in (".", ".."):
            raise ValueError("Dataset name cannot be '.' or '..'")
        self.datasets[name] = (kind, parts)
        self._ram_guard()                          # 0.5.7: offload above the RAM limit

    # ----- 0.5.7: offloading to disk above the RAM limit ------------------------
    def _ram_guard(self, force=False):
        """Read this computer's RAM use (at most every _RAM_CHECK_S, every
        time with ``force``) and offload the datasets held in RAM when it is
        at or above the limit; True when datasets were offloaded."""
        limit = self.ram_limit_percent if self.ram_limit_percent is not None else ram_limit()
        if not limit or limit <= 0.0:
            return False
        now = time.monotonic()
        if not force and now - self._ram_checked < _RAM_CHECK_S:
            return False
        self._ram_checked = now
        found = _over_ram_limit(limit)
        if found is None:
            return False
        return self.offload(found[0], found[1]) > 0

    def offload(self, percent=None, limit=None):
        """
        Move every dataset held in RAM (OFFLOAD_MIN_BYTES or more), the
        exact MJD arrays and (0.5.9) the file numbers of the combined x
        datasets to this project's offload file (0.5.7); returns how many
        datasets / arrays moved.

        Called by _ram_guard when this computer's RAM use reaches the limit
        (``percent`` / ``limit`` are logged); can be called directly (tests,
        a program that wants the data on disk).  Nothing moves when the disk
        has less free space than the data plus a fifth and 64 MB.
        """
        if self._offload is None:
            self._offload = OffloadStore("autoplot_project_")
        store = self._offload
        names = []
        total = 0
        for name, (_kind, parts) in self.datasets.items():
            if isinstance(parts, OffloadedParts):
                continue
            size = payload_bytes(parts)
            if size >= OFFLOAD_MIN_BYTES:
                names.append(name)
                total += size
        mjd_keys = [k for k, v in self.mjd_of.items()
                    if isinstance(v, np.ndarray) and v.nbytes >= OFFLOAD_MIN_BYTES]
        total += sum(self.mjd_of[k].nbytes for k in mjd_keys)
        # 0.5.9: the file numbers of the combined x datasets (one array may
        # serve several names: it is written once)
        source_keys = [k for k, v in getattr(self, "combined_of", {}).items()
                       if isinstance(v, np.ndarray) and v.nbytes >= OFFLOAD_MIN_BYTES]
        total += sum(v.nbytes for v in {id(self.combined_of[k]): self.combined_of[k]
                                        for k in source_keys}.values())
        if not names and not mjd_keys and not source_keys:
            return 0
        free = store.free_bytes()
        if free is not None and free < total * 1.2 + 64e6:
            self._offload_note("RAM limit reached but only %.1f MB free on disk for %.1f MB of "
                               "plot data: kept in RAM" % (free / 1e6, total / 1e6))
            return 0
        for name in names:
            kind, parts = self.datasets[name]
            offset, length = store.put(dict(parts))
            value = OffloadedValue(store, offset, length, payload_bytes(parts))
            self.datasets[name] = (kind, OffloadedParts(value, parts.keys()))
        for key in mjd_keys:
            array = self.mjd_of[key]
            offset, length = store.put(array)
            self.mjd_of[key] = OffloadedValue(store, offset, length, int(array.nbytes))
        written = {}
        for key in source_keys:
            array = self.combined_of[key]
            if id(array) not in written:
                offset, length = store.put(array)
                written[id(array)] = (array, OffloadedValue(store, offset, length,
                                                            int(array.nbytes)))
            self.combined_of[key] = written[id(array)][1]
        written = None
        parts = array = None                       # the last references to the RAM copies
        now = time.monotonic()
        if now - self._heap_trimmed >= _HEAP_TRIM_S:     # freed numpy buffers back to the OS
            self._heap_trimmed = now
            _release_os_heap()
        if percent is not None:
            why = "RAM in use on this computer %.1f %% >= limit %g %%" % (percent, limit)
        else:
            why = "offload asked for"
        self._offload_totals[0] += len(names) + len(mjd_keys) + len(source_keys)
        self._offload_totals[1] += total
        self._notes.add("%s: %d dataset(s), %.1f MB of plot data moved to disk so far (%s, "
                        "%.1f MB)" % (why, self._offload_totals[0], self._offload_totals[1] / 1e6,
                                      store.path, store.size / 1e6),
                        self.offload_log, self.offload_messages)
        return len(names) + len(mjd_keys) + len(source_keys)

    def _offload_note(self, text):
        """Log ``text`` at once (warnings) and keep it in offload_messages."""
        self._notes._emit(text, self.offload_log, self.offload_messages, time.monotonic())

    def offloaded(self):
        """(datasets on disk, datasets in RAM, file size in bytes) (0.5.7)."""
        on_disk = sum(1 for _k, parts in self.datasets.values() if isinstance(parts, OffloadedParts))
        size = self._offload.size if self._offload is not None else 0
        return on_disk, len(self.datasets) - on_disk, size

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

    def SetDataDateTime(self, name, seconds, mjd=None):
        """Date dataset: seconds since 2009-01-01 00:00:00 UTC (Veusz date-time).

        0.5.6: ``mjd`` - the exact MJD of the same instants, as the name of
        a stored 1-D dataset or an array of the same length; the MJD clock
        draws it (else UTC seconds / 86400 + 54832).  It is not saved as a
        dataset of its own.
        """
        self.SetData(name, seconds)
        self.datasets[name] = ("date", self.datasets[name][1])
        if mjd is not None:
            self.mjd_of[name] = mjd if isinstance(mjd, str) else np.asarray(mjd, dtype=float)
        else:
            self.mjd_of.pop(name, None)

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
        self.local_zones[local_name] = zone                  # 0.5.6: names on the label
        self.local_of[utc_name] = local_name
        self.utc_of[local_name] = utc_name
        return local_name

    def local_zone_names(self, dataset_names):
        """
        [(offset "+HH:MM", zone name), ...] of the local datasets
        ``dataset_names`` (0.5.6), in the order the offsets first occur: the
        name the operating system gives the zone at the first sample with
        that offset ("EST", "EDT"; Windows "Eastern Standard Time"), "" when
        it only repeats the offset or says UTC (label_zone_name).
        """
        found = OrderedDict()
        for name in dataset_names:
            axis = self.local_axes.get(name)
            if axis is None:
                continue
            zone = self.local_zones.get(name)
            local = np.asarray(axis.veusz_seconds, dtype=float)
            offsets = np.asarray(axis.offsets_s, dtype=float)
            good = np.flatnonzero(np.isfinite(local) & np.isfinite(offsets))
            if not good.size:
                continue
            # first sample of every offset (offsets change rarely: DST)
            starts = good[np.r_[True, np.diff(offsets[good]) != 0]]
            for index in starts.tolist():
                text = vszh5_writer.iso_offset_text(offsets[index])
                if text in found:
                    continue
                epoch = float(vszh5_writer.veusz_to_epoch(local[index] - offsets[index]))
                found[text] = label_zone_name(zone_name_at(epoch, zone))
        return list(found.items())

    def local_axis_text(self, dataset_names, label="Local time"):
        """(tick format, axis label) of an axis showing these local datasets.

        0.5.6: the label names the zone as well, e.g. "Local time (ISO 8601,
        EST, UTC-05:00)"; across offsets "UTC-04:00 EDT, UTC-05:00 EST from
        ..." when every name is an abbreviation (at most 6 characters, no
        space), else the offsets alone (a long Windows name would make the
        label too wide; the graph notes name the zone).
        """
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
        names = dict(self.local_zone_names(dataset_names))
        if len(offsets) == 1:
            text = vszh5_writer.iso_offset_text(offsets[0])
            zone = names.get(text, "")
            return (vszh5_writer.ISO_DATETIME_FORMAT + text,
                    "%s (ISO 8601, %sUTC%s)" % (label, (zone + ", ") if zone else "", text))
        short = all(name and len(name) <= 6 and " " not in name for name in names.values())
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
            word = "UTC%s" % offset + ((" " + names[offset]) if short and names.get(offset)
                                       else "")
            parts.append(word if current is None else "%s from %s" % (word, instant[:16]))
            current = offset
        return (vszh5_writer.ISO_DATETIME_FORMAT,
                "%s (ISO 8601; %s)" % (label, ", ".join(parts)))

    def finalize(self):
        """Name unnamed widgets, then lay out every time axis: local ISO 8601
        date-times, line breaks at gaps and (optionally) axis breaks (0.5.0;
        0.3.0 resolved marked local-time axes only); 0.5.6: a page with
        date-time x axes once per chosen clock (TimeLayout.clocks)."""
        for widget in list(self.Root.WalkWidgets()):
            if widget.auto_named:
                _rename_from_data(widget)
        clocks = (getattr(self, "time_layout", None) or TimeLayout()).clocks
        # 0.5.6: the clock copies of the last finalize go; each page gets its
        # own name back and the first chosen clock, and is laid out in it.
        _remove_clock_pages(self)
        pages = [w for w in self.Root.children if w.kind == "page"]
        for page in pages:
            if page.time_base_name is None:
                page.time_base_name = page.name
            page.name = page.time_base_name
            page.time_clock = clocks[0]
        self._layout_axes(self.Root)
        # 0.5.6: a page drawing date-time x axes gets the suffix of its clock
        # and a copy for every other chosen clock, right after it.
        for page in pages:
            if not _has_clock_axis(page):
                continue
            page.name = _free_page_name(self.Root, page.time_base_name +
                                        CLOCK_PAGE_SUFFIX[clocks[0]], page)
            at = self.Root.children.index(page)
            for clock in clocks[1:]:
                twin = _clone_widget(page, self.Root)
                twin.time_clone_of = page
                twin.time_clock = clock
                twin.time_base_name = page.time_base_name
                at += 1
                self.Root.children.insert(at, twin)
                twin.name = _free_page_name(self.Root, page.time_base_name +
                                            CLOCK_PAGE_SUFFIX[clock], twin)
                self._layout_axes(twin)
        _drop_unused_plot_copies(self)
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
        # 0.5.5: vertical two-line date labels and long numbers, margins
        # and page sizes that show every major tick label (_fit_page).
        ranges = {}
        for page in [w for w in self.Root.children if w.kind == "page"]:
            _fit_page(self, page, ranges)
        self._tag_untagged()
        self._ram_guard(force=True)            # 0.5.7: the plot copies made above
        self._notes.flush(self.offload_log, self.offload_messages)   # the last totals

    def _layout_axes(self, top):
        """Lay out every x axis below the widget ``top`` (0.5.6: was part of
        finalize): time axes on their page's clock, other x axes with the
        numeric gap options."""
        for graph in [w for w in top.WalkWidgets() if w.kind == "graph"]:
            for axis in [c for c in graph.children if c.kind in ("axis", "axis-broken")]:
                if axis.settings.get("direction", "horizontal") != "vertical":
                    # 0.5.3: an x axis that is not time gets the gap options
                    # too (TimeLayout.numeric)
                    if not _layout_time_axis(self, graph, axis):
                        _layout_numeric_axis(self, graph, axis)

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
        clock_copies = getattr(self, "_clock_copies", {})
        for name, owner in getattr(self, "_plot_copies", {}).items():
            origin = owner if isinstance(owner, str) else owner[-1]
            if name in clock_copies:                   # 0.5.6: type_time_utc / _mjd
                extra = clock_copies[name]
            else:
                extra = "type_time_local" if "_local" in name else "type_gap_rows"
            copies.setdefault(name, (origin, extra))
        for name, (origin, extra) in copies.items():
            if name not in self.datasets:
                continue
            # A copy on another clock (local, UTC, MJD) does not keep the
            # time tags of its origin; it gets its own.
            retimed = extra.startswith("type_time")
            for tag in tags_of.get(origin, ()):
                if tag.startswith("source_") or (
                        tag.startswith("type_") and not (retimed and tag.startswith("type_time"))):
                    self.tags[tag].add(name)
            self.tags.setdefault(extra, set()).add(name)
            if retimed and name.endswith("_gaps"):
                self.tags.setdefault("type_gap_rows", set()).add(name)

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
        parts = loaded_parts(parts)                   # 0.5.7: read once when offloaded
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
        self.finalize()                        # 0.5.7: ends with a RAM check (offload)
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
                # 0.5.7: an offloaded dataset is read back from the offload
                # file (pickle) and written into the .vszh5, one at a time
                parts = loaded_parts(parts)
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
