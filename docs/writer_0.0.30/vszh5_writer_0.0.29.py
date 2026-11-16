#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 William W. Wallace <wwallace@nrao.edu>
# SPDX-FileCopyrightText: 2026 William W. Wallace, Adaptive Electrodynamics LLC dba AdaptEd
# %% ===========================================================================
#    MODULE HEADER
# =============================================================================
"""Native Veusz HDF5 document (.vszh5) writer - no Veusz needed to write.

===============================================================================
# %% Header Info
--------
 Title          : Veusz HDF5 Document Writer
 Filename       : vszh5_writer.py
 Version        : 0.0.29 (Semantic Versioning 2.0.0, MAJOR.MINOR.PATCH)
 Author         : William W. Wallace
 Company        : NRAO/GBO
 Email          : wwallace@nrao.edu
 Created        : 2026-10-11
 Modified       : 2026-11-10
 Language       : Python 3.8 through 3.12 (3.12 primary; parses at 3.7)
 Requires       : numpy, h5py   (Veusz only to open or render the result;
                  pypdf only for PDF bookmarks)
 Licence        : GPL-3.0-or-later (GNU General Public License v3.0 or later)
 Copyright      : Copyright (C) 2026 William W. Wallace
 Derived from   : isp_speed_monitor.py 1.3.0 (VeuszWriter, VszDoc, vz_text,
                  epoch_to_veusz, find_veusz_exe, _atomic_path/_commit_atomic,
                  tests/render_in_veusz.sh), Copyright (C) 2026 William W.
                  Wallace, Adaptive Electrodynamics LLC dba AdaptEd.
 Status         : Internal draft - checked against real Veusz 3.6.2 and 4.2.1
                  (2-D datasets and the page helpers of 0.0.11/0.0.12: 4.2.1;
                  the 0.0.14 fills, bar pages and step masks: 3.6.2;
                  0.0.18 render fixes: 3.6.2; 0.0.18 appender: 4.2.1;
                  0.0.20 date-axis images and reduced spectrograms: see
                  the 0.0.20 history entry; 0.0.23 shape checks and
                  failed-load renders: 3.6.2 and 4.2.1; 0.0.24 MJD and
                  local ISO-8601 time axes: 3.6.2; 0.0.25 names, notes and
                  local time axes: 4.2.1; 0.0.26 Smith charts: 4.2.1,
                  settings checked in the 3.6.2 source; 0.0.27 gap rule:
                  4.2.1 with real GBT sampler2log files; 0.0.28 faster
                  save: files identical to 0.0.27; 0.0.29 exact ISO text:
                  3.6.2, every other output identical to 0.0.28)
 Lineage        : 0.0.20 joins the two releases that were both numbered
                  0.0.18 - the R&S FSP logger 0.9.0 release (render fixes)
                  and the AB Power Meter Monitor 1.6.0 release (appender) -
                  together with 0.0.19 (AB Power Meter Monitor 1.6.2).
                  0.0.21 changes regularize_rows() only (SHR Converter
                  0.0.2); no other program calls it. 0.0.22 (MHM2010
                  Workbench review) rejects a multidimensional date array
                  in VeuszWriter.dataset_date(); 0.0.23 extends the same
                  rule to every 1-D and date entry point of VeuszAppender,
                  leaves no half-made dataset after a rejected call, and
                  makes a document whose data Veusz cannot load a failed
                  render. 0.0.24 (MHM2010 / T4 Maser Timing Workbench
                  0.2.0) adds paired MJD and local ISO-8601 date-time axes;
                  additions only. 0.0.25 (FitsAutoPlot suite, 2026-11-09)
                  names plot widgets after their data, writes source-file
                  headers into page notes and adds local date-time helpers
                  for Veusz date datasets. 0.0.26 (FitsAutoPlot suite,
                  Touchstone AutoPlot 0.7.0) adds Smith charts; 0.0.27
                  (FitsAutoPlot, FITS AutoPlot 0.4.0) the small/large gap
                  rule and exact DST change instants; 0.0.28 (FITS / Franks
                  AutoPlot 0.4.1) saves faster with identical files; 0.0.29
                  (MHM2010 / T4 Maser Timing Workbench 0.2.4) writes local
                  ISO-8601 text without dropping digits on request
                  (iso_local_strings timespec="exact"). One master copy: the
                  RF project document
                  claude/vszh5_writer.py (see claude/vszh5-writer-registry.md).
===============================================================================
# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216


LICENCE
-------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

DESCRIPTION
-----------
Writes Veusz's native HDF5 save format directly with h5py: the data as typed
Veusz datasets and the plots as the Veusz command script, in one file that
opens in Veusz with File > Open. Nothing from Veusz is imported to write it.

    from vszh5_writer import VeuszWriter, VszDoc, vz_text

    doc = VszDoc(generator="my_program 1.0")
    doc.add("page", "page1"); doc.to("page1")
    doc.add("graph", "g"); doc.to("g")
    doc.add("axis", "x"); doc.add("axis", "y")
    doc.set("y/direction", "vertical")
    doc.add("xy", "trace"); doc.set("trace/xData", "t"); doc.set("trace/yData", "v")
    doc.up(); doc.up()

    with VeuszWriter("out.vszh5", compress=True) as w:
        w.dataset_1d("t", t)
        w.dataset_1d("v", v, serr=sigma, tags=("raw",))
        w.finish(doc.text())

The plot-page helpers (0.0.11) build whole pages in one call, the same way
on every page of every program:

    from vszh5_writer import Trace, image_page, xy_page

    xy_page(doc, "spectrum", [Trace("f", "amp", "FFT"), Trace("f", "fit", "fit")],
            title="Envelope spectrum", xlabel="Frequency (Hz)", ylabel="W/m^2",
            ylog=True)
    image_page(doc, "map", "level_db", title="Spectrogram", xlabel="Time (s)",
               ylabel="Frequency (Hz)", zlabel="dB", zmin=-60, zmax=0)
    ...
    w.dataset_2d("level_db", image, xcent=time_s, ycent=freq_hz)

FILE LAYOUT  (Veusz document format 1, as written by Document.saveToHDF5File
and read by loader.loadHDF5Doc in Veusz 3.6 and 4.2 [R1])
-----------
    /Veusz                     attrs vsz_version, vsz_saved_at, vsz_format = 1
    /Veusz/Data/<name>         group, attr vsz_datatype in {'1d', '2d', 'date', 'text'}
         .../data              values; attr vsz_name = <name> (UTF-8)
         .../serr|perr|nerr    optional error arrays for '1d'
         .../xrange|xedge|xcent  '2d' only: x axis of the image (one of the
         .../yrange|yedge|ycent  three per axis); 'data' carries an HDF5
                               object reference to it in attr vsz_<key>
                               (Dataset2D.saveDataDumpToHDF5 [R1])
    /Veusz/Document/document   1-element byte-string array: the command script
    /Veusz/Document/Tags/<tag> byte-string array of the dataset names tagged
Names containing '/' or '`' are escaped as '`SL' and '`BT'. Date datasets
hold seconds since 2009-01-01 00:00:00 (naive), Veusz's date origin.

WRITING SAFELY
--------------
Every file is written to "<name>.part" and moved onto "<name>" with one
os.replace() when finish() succeeds, so Veusz or a script that opens the
file mid-update sees the old complete file or the new one, never half of
one. An exception inside the "with" block deletes the .part file.

APPENDING IN PLACE  (0.0.18, VeuszAppender)
------------------
A long-running logger cannot rewrite a growing document every poll, so
VeuszAppender grows an existing .vszh5 in place: 1-D and date datasets are
created chunked and resizable (maxshape=(None,), fill value NaN) and each
append writes only the new rows. The document script and tags can be
replaced at any time. This is NOT atomic: the file is opened 'r+' for the
few milliseconds an append takes, so

  * keep appends short and close the appender between polls;
  * an HDF5 file lock held by a reader (Veusz holds the file only while it
    loads) raises AppendLockedError - retry on the next poll;
  * the datasets are left uncompressed on purpose: a partly filled chunk is
    rewritten on every append, and a compressed chunk that grows is moved,
    leaving dead space in the file; uncompressed chunks never move;
  * unwritten rows read back as NaN (the fill value), never as zeros.
Veusz reads only the /Veusz group [R1: loader.loadHDF5Doc], so private
bookkeeping is kept beside it under /VszAppend (ignored by Veusz).

RENDERING  (needs Veusz)
---------
render_pages() renders PDF and PNG from a .vszh5. `veusz --export` cannot
open a .vszh5 (Veusz 3.6.2 and 4.2.1 read it as text, get an empty document
and fail), and `--export-dpi` and `%PAGE%` are not command-line features. So
the program is driven in `--listen --quiet` mode with a script that calls
loader.loadHDF5Doc, as tests/render_in_veusz.sh in isp_speed_monitor does.
That works with a standalone Veusz install (Windows installer) as well as
with the conda-forge package. If no Veusz program is found but the veusz
Python package imports, the same load-and-export runs in-process.

PDF BOOKMARKS  (needs pypdf)
-------------
Veusz writes a multipage PDF without an outline. render_pages() adds one
afterwards with add_pdf_bookmarks(): by default one bookmark per page, named
after the Veusz page; a caller can pass its own hierarchy (a callable that
receives the page names). The outline opens in the reader's bookmark pane.
Without pypdf the PDF is still written, only without bookmarks.

DIFFERENCES FROM isp_speed_monitor.py 1.3.0
-------------------------------------------
* vz_text() follows Veusz's own latexEscape exactly: '\\' becomes
  {\\backslash} (in Veusz text '\\\\' is a LINE BREAK, so the 1.3.0 escape
  split a label at any backslash) and '[' ']' are escaped as well.
* VeuszWriter: optional gzip+shuffle compression; perr/nerr error arrays;
  context manager; atomic .part write built in; abort().
* local_utc_offsets(): per-sample UTC offset, so a local-time axis stays
  correct across a daylight-saving change.
* render_pages(): the --listen method, every page to PNG plus one multipage
  PDF, Windows-safe paths (bound outside exec), waits for a done marker, and
  falls back to the veusz package when the program fails.
* add_pdf_bookmarks(): a bookmark hierarchy and document information in the
  rendered PDF.
* (0.0.11) dataset_2d() for images, and plot-page helpers: Trace, add_title,
  xy_page (line plots, linear/log/date axes, legend) and image_page (2-D
  image with a colour bar placed where Veusz 4.2.1 draws it).
* (0.0.12) grid_page (several graphs on one page, e.g. an overview) and
  text_page (a page of text lines, e.g. a results summary in the PDF).
* (0.0.13) dataset_names, read_text_dataset and page_names: read a written
  document back with h5py (no Veusz): datasets, text metadata, page order.
* (0.0.14) Trace.fill (shaded region below/above a trace, e.g. where a
  level exceeds a protection mask), bar_page (category bar charts with a
  labels-mode axis), step_mask (stepped mask lines with gaps) and
  reference_line (horizontal or vertical threshold/marker lines).
* (0.0.16) frequency_ghz_text and frequency_time_pages: one page per
  frequency bin against recorded datetime, plus a combined page.
* (0.0.17) linear_power_mean_dbm: chunked, max-referenced linear-power mean
  of dBm samples (never an arithmetic mean of dB values).
* (0.0.18) VeuszAppender: grow a document in place (append rows, replace
  the script and tags) for loggers that run for days; VeuszWriter
  resizable=True writes a document that can be appended to later.
* (0.0.18, FSP branch / 0.0.19 / 0.0.20) render_pages() fails loudly: a
  document Veusz could not load, or a PDF/PNG that was not written, is an
  error rather than an empty success.
* (0.0.20) image_page(xmode=, ymode=, datetime_format=): date/time image
  axes (e.g. a spectrogram with time on y); xy_page(datetime_format=);
  block_reduce(): peak-hold / minimum / mean / power-mean decimation of a
  2-D array (a waterfall too large to plot) in blocks of rows x columns,
  on NumPy or a NumPy-compatible array module (CuPy); regularize_rows():
  uneven image rows onto an even grid (fast to draw in Veusz).
* (0.0.22 / 0.0.23) Shapes are checked before anything is written: a 1-D
  or date dataset must be 1-D (a single number counts as one row) in
  VeuszWriter and VeuszAppender alike, so no input is stored in a shape
  Veusz refuses or flattened silently; a rejected call leaves no dataset
  behind. render_pages() treats a document whose data Veusz cannot load
  as a failed render and writes no output for it.
* (0.0.24) Time axes in MJD and in the processing computer's local time:
  mjd_to_epoch/mjd_to_veusz, iso_offset_text, local_time_axis (a Veusz
  date dataset shifted by the DST-aware local UTC offset, with an
  ISO-8601 tick format such as 2026-11-03T06:31:15-05:00),
  iso_local_strings, with_x and time_pair_pages (one page against MJD,
  one against local date-time); Panel.datetime_format for grid_page.
* (0.0.25) Specific widget names and page notes: graphs, traces, images
  and bars are named after their datasets ("g_<y>", "<y>", "img_<z>")
  instead of "graph"/"trace0"; widget_name/unique_widget_name make any
  text a valid Veusz widget name; header_notes builds page notes from a
  source file header (FITS cards, "Key;Value" lines, a mapping) and every
  page helper takes notes=; local_time_from_veusz turns stored UTC date
  datasets into a local display axis; VSZH5_DISPLAY_TZ overrides the
  computer's zone (e.g. a UTC build server).
* (0.0.26) Smith charts in Veusz: smith_grid (resistance / reactance or
  conductance / susceptance grid, labels, VSWR circles as datasets and xy
  widget settings), smith_trace_datasets / smith_trace (Re, Im Gamma and
  frequency), smith_page / smith_graph_spec (aspect-locked graph, hidden
  axes, legend beside the chart), gamma_from_impedance,
  impedance_from_gamma, vswr_from_gamma; title_settings.
* (0.0.27) Time gaps: median_step, gap_threshold (K x median step),
  gap_positions/insert_gaps (NaN rows: the line breaks at a small gap),
  axis_break_pairs (intervals of a broken axis at a large gap, shrunk by a
  fraction of the gap); local_time_axis names the minute a DST change
  happens.
* (0.0.28) Faster saving with identical files: a name set beside the name
  list, one create_dataset() per dataset, text in one call, UTC offsets
  cached per (zone, UTC hour).
* (0.0.29) iso_local_strings(timespec=): "seconds" (default, as before:
  fractions of a second are cut off), "milliseconds", "microseconds",
  "auto" (Python's isoformat) or "exact" - every digit of the shortest
  decimal that gives back the same float epoch (Python repr), so the text
  is never rounded or cut; ISO_TIMESPECS lists them. mjd_iso_local_strings:
  the same exact text straight from UTC MJD values, converted in decimal
  arithmetic (MJD 61234.10355 -> 22:29:06.72, not 06.7199999...).

# %%% References
[R1] Veusz Developers, Veusz 4.2.1 source: veusz/document/doc.py
     (saveToHDF5File), veusz/document/loader.py (loadHDF5Doc),
     veusz/utils/textrender.py (latexEscape, symbols), veusz/veusz_main.py.
     https://pypi.org/project/veusz/4.2.1/
[R2] h5py developers, "h5py 3.16.0." https://pypi.org/project/h5py/
[R3] W. W. Wallace, isp_speed_monitor.py 1.3.0, Adaptive Electrodynamics
     LLC dba AdaptEd, 2026 (source of the writer design).
[R4] P. H. Smith, "Transmission line calculator," Electronics, vol. 12,
     no. 1, pp. 29-31, Jan. 1939 (the Smith chart).
[R5] A. Arsenovic et al., "scikit-rf: An open source Python package for
     microwave network creation, analysis, and calibration," IEEE Microw.
     Mag., vol. 23, no. 1, pp. 98-105, Jan. 2022,
     doi: 10.1109/MMM.2021.3117139 (Smith grid values, VSWR circles,
     admittance chart convention).

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.29: MHM2010 / T4 Maser Timing Workbench 0.2.4 ("in all exports,
Date: 2026-11-10
#              all floats or numbers should never be rounded or trimmed"),
#              built on master 0.0.28 (SHA-256 13f68513...f634a64b5).
#              Additions only: every 0.0.28 call writes the same file.
#              * NEW iso_local_strings(epoch_s, zone=None,
#                timespec="seconds"): "seconds" keeps the 0.0.24-0.0.28
#                text (datetime.isoformat cuts the fraction off, so
#                12:00:00.9 reads 12:00:00); "milliseconds",
#                "microseconds" and "auto" are isoformat's; "exact" writes
#                every digit of the shortest decimal that gives back the
#                same float epoch (Python repr, e.g.
#                2026-07-12T22:29:06.1234567-04:00), so the epoch can be
#                recovered bit for bit from the text. A float64 epoch near
#                2026 resolves 0.24 us and an MJD near 61220 0.63 us, below
#                the 1 us of datetime, so "microseconds" would still round.
#                ValueError for any other timespec; ISO_TIMESPECS lists
#                the accepted values.
#              * NEW mjd_iso_local_strings(mjd, zone=None): exact local ISO
#                text of UTC MJD values. The MJD's shortest decimal (repr)
#                becomes Unix seconds in decimal arithmetic, (MJD - 40587)
#                x 86400, so the text states the time the MJD states (MJD
#                61234.10355 -> 22:29:06.72-04:00; through the float epoch
#                06.7199999...); the zone offset is taken at that instant;
#                Decimal inversion gives the identical MJD float back.
#              * Header: the "Version :" line said 0.0.27 in 0.0.28 (only
#                __version__ was raised); now 0.0.29. Lineage, Status and
#                DIFFERENCES name 0.0.28 and 0.0.29.
#              Checked: tests/test_writer_0029.py (workbench): round trip
#              of 10 000 random epochs and of 10 000 random and the real
#              Franks MJD values to the identical float (and the Franks
#              text equals the decimal MJD the file states); the default
#              text equals 0.0.28's;
#              DST zones and NaN; Python 3.8 grammar; every other function
#              unchanged (diff limited to the ISO text and the header).
# %%%%% Function Descriptions
        epoch_to_veusz: Unix epoch seconds -> Veusz date value (seconds since 2009-01-01).
        local_utc_offsets: Local UTC offset in seconds for every timestamp, DST-aware.
        _hour_offsets: UTC offset (s) at the middle of each UTC hour number in ``hours``.
        vz_text: Escape Veusz text-markup characters so a string prints literally.
        escape_hdf_name: Escape a dataset name for use as an HDF5 member name, as Veusz does.
        widget_name: Any text -> a valid Veusz widget name (0.0.25).
        unique_widget_name: ``name``, or ``name_2``, ``name_3`` ...
        header_notes: Page notes text from a source-file header (0.0.25).
        set_notes: Set the notes of the current widget (a page) when ``notes`` is not empty.
        VszDoc (class): Builder for the Veusz command script stored in the document.
        VeuszWriter (class): Write Veusz datasets and a document script to a .vszh5 file.
        _default_file_mode: Mode a new file would get from open(): 0o666 minus the process umask.
        _one_dimensional: ``values`` as a contiguous float64 1-D array, or ValueError (0.0.23).
        _create_resizable_1d: Create ``group[key]``: a chunked, resizable, uncompressed float64 array.
        _is_regular: True when the spacing of ``values`` is constant (relative 1e-6).
        _image_axis: Choose and check the stored form of one image axis (see dataset_2d).
        Trace (class): One line or marker series (Veusz 'xy' widget) for xy_page().
        add_title: Centred title label near the top of the current page.
        title_settings: Settings of add_title()'s page title label (0.0.26: shared with the widget-tree builders, same values and order as before).
        _distance_cm: A physical Veusz distance ("15cm", "6in", "20pt") in cm; None otherwise.
        _graph_name: (0.0.25) Explicit name, else "g_<y>" for one trace, else "g_<page>".
        _open_page_and_graph: Add a page (sized, with notes), its title and a graph; leave the graph current.
        _add_axes: The x and y axes of the current graph, with labels, scales and limits.
        xy_page: A page with one graph of line/marker traces; returns the graph name.
        _add_traces: Add the xy widgets of ``traces`` to the current graph (0.0.25: named after Trace.name or the y dataset, unique in the graph); returns the names.
        _trace_settings: Settings of the xy widget that draws ``trace`` (relative paths, in the order Veusz receives them); shared by _add_traces and smith_graph_spec (0.0....
        _add_key: Legend in the current graph: True, False or "auto" (more than one label).
        image_page: A page with a 2-D dataset drawn as an image, and a colour bar; returns the graph name.
        Panel (class): One graph of grid_page(): traces, axes and a small title (0.0.12).
        grid_page: A page with several graphs in a Veusz grid (rows filled left to right).
        linear_power_mean_dbm: Equal-weight power mean of finite dBm samples, returned in dBm.
        frequency_ghz_text: Scale decimal Hz to GHz without coarse rounding/name collisions.
        frequency_time_pages: Append one page per frequency and a combined page; preserve all raw data.
        mjd_to_epoch: UTC Modified Julian Date (days) -> Unix epoch seconds.
        mjd_to_veusz: UTC MJD -> Veusz date seconds; ``offset_s`` shifts to a local clock.
        iso_offset_text: UTC offset in seconds -> ISO-8601 "+HH:MM" / "-HH:MM" (e.g.
        _zone_from_text: "UTC"/"Z", "+HH:MM", "-5", "UTC-05:00" or an IANA name -> tzinfo.
        iana_zone: An IANA zone ("America/New_York") as a tzinfo on every Python (0.0.27).
        display_zone: Zone for local date-time axes (0.0.25): VSZH5_DISPLAY_TZ when set, else ``default`` (None = the time zone of this computer).
        _zone_offsets: Per-sample UTC offset (s) in ``zone`` (None = VSZH5_DISPLAY_TZ, else this computer's zone).
        LocalTimeAxis (class): A local wall-clock date axis for Veusz (0.0.24), see local_time_axis().
        _offset_change_instant: First whole UTC minute in (before, after] whose offset is ``offset_after`` (bisection; the zone changes offset once between two samples) (0.0.27).
        local_time_axis: Veusz date values and an ISO-8601 tick format on a local clock.
        veusz_to_epoch: Veusz date seconds (since 2009-01-01, UTC) -> Unix epoch seconds (0.0.25).
        local_time_from_veusz: local_time_axis() for UTC values already in Veusz date seconds (0.0.25).
        style_datetime_axis: Make ``axis`` (path from the current widget, e.g.
        iso_local_strings: ISO-8601 local date-times with offsets, e.g.
        _exact_wall_text: (0.0.29) ISO text (no offset suffix) of an exact wall-clock time in Unix-style seconds (Decimal).
        mjd_iso_local_strings: (0.0.29) ISO-8601 local date-times of UTC MJD values, every digit kept.
        with_x: The same traces drawn against another x dataset (0.0.24).
        time_pair_pages: Two xy pages from one set of traces: against MJD and against local time.
        median_step: Median of the positive, finite steps of ``t`` (0.0 when there is none).
        gap_threshold: Line-break threshold in the units of ``t``: ``manual`` when > 0, else ``k * median_step(t)`` (0.0 = no threshold, e.g.
        gap_positions: Insertion indices of the gap rows: ``i + 1`` for every step ``t[i+1] - t[i] > threshold``.
        insert_gaps: ``values`` (float) with a NaN inserted before each of ``positions`` (from gap_positions); the input is not changed.
        axis_break_pairs: (start, end) intervals to leave out of a broken axis.
        gamma_from_impedance: Reflection coefficient Gamma = (Z - Z0) / (Z + Z0) (complex array).
        impedance_from_gamma: Impedance Z = Z0 (1 + Gamma) / (1 - Gamma); Gamma = 1 gives inf.
        vswr_from_gamma: VSWR = (1 + |Gamma|) / (1 - |Gamma|); inf for |Gamma| >= 1.
        _joined: Concatenate (x, y) curves with a NaN point between them (Veusz breaks an xy line at NaN, nanHandling 'break-on', its default).
        SmithGrid (class): Datasets and widget specifications of a Smith-chart grid (0.0.26).
        _grid_line_settings: xy settings of a grid line family: no markers, no legend entry.
        _grid_label_settings: xy settings that print a text dataset at points (no line, no marker).
        smith_grid: The grid of a Smith chart as Veusz datasets and xy widget settings.
        smith_trace_datasets: Datasets of one Smith trace: "<stem>_re", "<stem>_im" and, when given, "<stem>_freq_Hz" (read by the suite's interactive Smith plot).
        smith_trace: Trace of Im(Gamma) against Re(Gamma) for the datasets of smith_trace_datasets(stem, ...); keyword arguments go to Trace.
        smith_has_key: True when a Smith graph gets a legend: key=True, or "auto" with more than one labelled trace (as xy_page).
        smith_graph_spec: Settings of a Smith graph and its children, for any Veusz builder.
        smith_page_layout: (page width, page height, graph top margin) of a Smith page: the chart about ``size`` across, 1 cm more height for the title, ``key_width`` more wi...
        smith_page: A page with one Smith chart; returns the graph name (0.0.26).
        text_page: Pages of plain text lines (escaped; monospaced by default) (0.0.12).
        step_mask: (x, y) arrays of a stepped mask: one flat segment per band.
        reference_line: (x, y) arrays of a straight threshold or marker line.
        bar_page: A page with a vertical category bar chart (0.0.14); returns the graph name.
        _reduce_axis: Reduce ``values`` along ``axis`` in groups of ``factor`` (see block_reduce).
        _reduce_groups: Collapse ``axis`` of a grouped array with ``method``.
        block_reduce: Decimate a 2-D array in blocks of ``row_factor`` x ``column_factor``.
        regularize_rows: Image rows with (possibly uneven) centres -> an even grid for drawing.
        dataset_names: Names of the datasets stored in a .vszh5 document (unescaped, sorted).
        read_text_dataset: The strings of a text dataset written by dataset_text(), or None if absent.
        page_names: Names of the pages of a written document, in page order (0.0.13).
        AppendLockedError (class): The document is locked by another process (HDF5 file locking).
        _is_lock_error: True for HDF5's "unable to lock file" open failure (any platform).
        VeuszAppender (class): Grow an existing .vszh5 document in place, a few rows at a time.
        _decode: bytes or numpy bytes -> str (UTF-8); anything else -> str().
        find_veusz_exe: Locate the Veusz program: explicit path, env var, PATH, usual folders.
        _popen_kwargs: Hide the console window of child processes on Windows.
        _render_script: The script piped into `veusz --listen`.
        _outline_parts: Split an outline entry into (title, page, children).
        add_pdf_bookmarks: Write a bookmark (outline) hierarchy into an existing PDF, in place.
        render_pages: Render a .vszh5 to one multipage PDF and per-page PNG files.
        _page_names_from_log: Veusz page names from the VSZH5_PAGES line the render script prints.
        _run_listen: Pipe a script into `veusz --listen --quiet` and wait for the marker.
        _render_in_process: Fallback: render with the veusz Python package in this process.
# %%%%% Variable Descriptions
    Define all utilized variables
        VSZH5_FORMAT: Highest document format the Veusz 3.6 and 4.2 loaders accept.
        VSZH5_LAYOUT_VERSION: Veusz release whose HDF5 layout this writer follows (written as vsz_version).
        COMPRESS_MIN_ELEMENTS: Arrays at least this long are gzip+shuffle compressed when compress=True.
        _VLEN_BYTES: text datasets
        APPEND_CHUNK_ELEMENTS: (0.0.18) Rows per chunk of a resizable (appendable) 1-D or date dataset: : 1024 float64 = 8 KiB, at the low end of the 10 KiB - 1 MiB range h5py :...
        PDF_DPI: Resolution passed to Veusz's Export() for PDF. Veusz 4.x lays a PDF page : out at the nearest resolution the Qt PDF engine supports (72) but sets t...
        VEUSZ_EPOCH: Veusz date origin, and the offset that maps Unix seconds onto it.
        UNIX_TO_VEUSZ_S: -1230768000
        _HOUR_OFFSET_CACHE: 0.0.28: UTC offset of each whole UTC hour, per zone, kept between calls. A project with thousands of time traces asked the zone database for the sa...
        _HOUR_OFFSET_CACHE_MAX: entries (about 114 years of hours)
        WIDGET_NAME_MAX: Longest widget name the helpers make (Veusz has no limit; long names make : the widget tree and PDF bookmarks hard to read).
        TRACE_COLOURS: 1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        _CM_PER_UNIT: Centimetres per unit for the physical Veusz distance units.
        MJD_UNIX_EPOCH: Modified Julian Date of the Unix epoch, 1970-01-01T00:00:00 UTC.
        ISO_DATETIME_FORMAT: Veusz tick format of an ISO-8601 date-time without offset (Veusz %VD codes).
        ISO_TIMESPECS: (0.0.29) timespec values of iso_local_strings(): seconds (default), milliseconds, microseconds, auto, exact.
        _EXACT: (0.0.29) Decimal context (80 digits) of the exact ISO text.
        DISPLAY_TZ_ENV: (0.0.25) Environment variable that overrides the computer's time zone for : local date-time axes: "UTC", a fixed offset such as "-05:00" or "+0530"...
        _FIXED_OFFSET: fixed UTC offset text (-05:00, UTC-05:00, GMT+0100) for VSZH5_DISPLAY_TZ
        LOCAL_DATETIME_ROTATE: (0.0.25) Tick-label rotation (degrees) of an ISO-8601 date-time axis; a : 25-character "2026-11-09T08:39:00-05:00" label needs it.
        GAP_K: Default line-break multiplier K (gap when dt > K * median(dt)).
        GAP_AXIS_N: Default axis-break multiple n (axis cut when dt > n * K * median(dt)); 0 = never.
        GAP_AXIS_MARGIN: Fraction of a gap kept visible at each side of an axis break.
        SMITH_RESISTANCES: Normalized resistance (conductance) circles and reactance (susceptance) : arcs drawn by default - the values of scikit-rf's and the Plots tab's chart.
        SMITH_REACTANCES: normalised reactances of the Smith grid arcs
        SMITH_VSWR: VSWR circles of scikit-rf's draw_vswr=True.
        SMITH_GRID_COLOUR: a6a6a6"
        SMITH_FRAME_COLOUR: 4d4d4d"
        SMITH_LABEL_COLOUR: 595959"
        SMITH_LIMIT: Half-width of the Re/Im Gamma axes: room outside |Gamma| = 1 for labels.
        SMITH_NOTES_PREFIX: First words of the notes of every Smith graph (how readers recognise one).
        VEUSZ_LINE_BREAK: Veusz text markup for a line break (two backslashes in the label text).
        REDUCE_METHODS: Methods of block_reduce().
        APPEND_META_GROUP: Top-level HDF5 group for the appender's private bookkeeping (e.g. a logger's : column index). Veusz reads only /Veusz [R1: loader.loadHDF5Doc], so...
# %%%%% More Info
    Master copy: RF project document claude/vszh5_writer.py (registry
    claude/vszh5-writer-registry.md).  Needs numpy and h5py; Veusz only to open
    or render; pypdf only for PDF bookmarks; on Python 3.8 backports.zoneinfo
    or python-dateutil for IANA zone names (VSZH5_DISPLAY_TZ=America/New_York).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md of FitsAutoPlot (mamba:
    environment-py38.yml, environment-py312.yml).  Release notes: RF project
    claude/vszh5-writer-0.0.29-notes.md (workbench docs/vszh5_writer_0.0.29_NOTES.md).
# %%%% 0.0.28: FitsAutoPlot suite speed round (FITS / Franks AutoPlot 0.4.1):
Date: 2026-11-09
#              faster saving, the same file contents. Built on master 0.0.27.
#              * VeuszWriter keeps a set beside the ordered name list: the
#                "written twice" check was a list search per dataset
#                (O(n^2); 9 s of a 40 000-dataset save).
#              * dataset_1d / dataset_date: create_dataset() returns the
#                dataset (one HDF5 lookup less per dataset); text datasets
#                are made in one call with the variable-length bytes type.
#              * local_utc_offsets / _zone_offsets: UTC offsets cached per
#                (zone, UTC hour) between calls (_hour_offsets); the
#                30-day OnePpsDeltas set asked the zone database 183 000
#                times.  The computer's zone is keyed with time.tzname /
#                timezone / altzone, so a TZ change is seen.
#              * block_reduce(): docstring says "mean" is for linear values
#                and "power_mean_db" for dB values (no behaviour change).
#              Checked: every dataset, attribute and the document script of
#              the tack (776 datasets), UWBR Temps (384) and 30-day 1PPS
#              (40 195) projects are identical to 0.0.27's (object-by-object
#              comparison; only the version comment of the script differs);
#              the 30-day save took 14.2 s (was 34.2 s, 2 CPUs).
# %%%%% Function Descriptions
        epoch_to_veusz: Unix epoch seconds -> Veusz date value (seconds since 2009-01-01).
        local_utc_offsets: Local UTC offset in seconds for every timestamp, DST-aware.
        _hour_offsets: UTC offset (s) at the middle of each UTC hour number in ``hours``.
        vz_text: Escape Veusz text-markup characters so a string prints literally.
        escape_hdf_name: Escape a dataset name for use as an HDF5 member name, as Veusz does.
        widget_name: Any text -> a valid Veusz widget name (0.0.25).
        unique_widget_name: ``name``, or ``name_2``, ``name_3`` ...
        header_notes: Page notes text from a source-file header (0.0.25).
        set_notes: Set the notes of the current widget (a page) when ``notes`` is not empty.
        VszDoc (class): Builder for the Veusz command script stored in the document.
        VeuszWriter (class): Write Veusz datasets and a document script to a .vszh5 file.
        _default_file_mode: Mode a new file would get from open(): 0o666 minus the process umask.
        _one_dimensional: ``values`` as a contiguous float64 1-D array, or ValueError (0.0.23).
        _create_resizable_1d: Create ``group[key]``: a chunked, resizable, uncompressed float64 array.
        _is_regular: True when the spacing of ``values`` is constant (relative 1e-6).
        _image_axis: Choose and check the stored form of one image axis (see dataset_2d).
        Trace (class): One line or marker series (Veusz 'xy' widget) for xy_page().
        add_title: Centred title label near the top of the current page.
        title_settings: Settings of add_title()'s page title label (0.0.26: shared with the widget-tree builders, same values and order as before).
        _distance_cm: A physical Veusz distance ("15cm", "6in", "20pt") in cm; None otherwise.
        _graph_name: (0.0.25) Explicit name, else "g_<y>" for one trace, else "g_<page>".
        _open_page_and_graph: Add a page (sized, with notes), its title and a graph; leave the graph current.
        _add_axes: The x and y axes of the current graph, with labels, scales and limits.
        xy_page: A page with one graph of line/marker traces; returns the graph name.
        _add_traces: Add the xy widgets of ``traces`` to the current graph (0.0.25: named after Trace.name or the y dataset, unique in the graph); returns the names.
        _trace_settings: Settings of the xy widget that draws ``trace`` (relative paths, in the order Veusz receives them); shared by _add_traces and smith_graph_spec (0.0....
        _add_key: Legend in the current graph: True, False or "auto" (more than one label).
        image_page: A page with a 2-D dataset drawn as an image, and a colour bar; returns the graph name.
        Panel (class): One graph of grid_page(): traces, axes and a small title (0.0.12).
        grid_page: A page with several graphs in a Veusz grid (rows filled left to right).
        linear_power_mean_dbm: Equal-weight power mean of finite dBm samples, returned in dBm.
        frequency_ghz_text: Scale decimal Hz to GHz without coarse rounding/name collisions.
        frequency_time_pages: Append one page per frequency and a combined page; preserve all raw data.
        mjd_to_epoch: UTC Modified Julian Date (days) -> Unix epoch seconds.
        mjd_to_veusz: UTC MJD -> Veusz date seconds; ``offset_s`` shifts to a local clock.
        iso_offset_text: UTC offset in seconds -> ISO-8601 "+HH:MM" / "-HH:MM" (e.g.
        _zone_from_text: "UTC"/"Z", "+HH:MM", "-5", "UTC-05:00" or an IANA name -> tzinfo.
        iana_zone: An IANA zone ("America/New_York") as a tzinfo on every Python (0.0.27).
        display_zone: Zone for local date-time axes (0.0.25): VSZH5_DISPLAY_TZ when set, else ``default`` (None = the time zone of this computer).
        _zone_offsets: Per-sample UTC offset (s) in ``zone`` (None = VSZH5_DISPLAY_TZ, else this computer's zone).
        LocalTimeAxis (class): A local wall-clock date axis for Veusz (0.0.24), see local_time_axis().
        _offset_change_instant: First whole UTC minute in (before, after] whose offset is ``offset_after`` (bisection; the zone changes offset once between two samples) (0.0.27).
        local_time_axis: Veusz date values and an ISO-8601 tick format on a local clock.
        veusz_to_epoch: Veusz date seconds (since 2009-01-01, UTC) -> Unix epoch seconds (0.0.25).
        local_time_from_veusz: local_time_axis() for UTC values already in Veusz date seconds (0.0.25).
        style_datetime_axis: Make ``axis`` (path from the current widget, e.g.
        iso_local_strings: ISO-8601 local date-times with offsets, e.g.
        with_x: The same traces drawn against another x dataset (0.0.24).
        time_pair_pages: Two xy pages from one set of traces: against MJD and against local time.
        median_step: Median of the positive, finite steps of ``t`` (0.0 when there is none).
        gap_threshold: Line-break threshold in the units of ``t``: ``manual`` when > 0, else ``k * median_step(t)`` (0.0 = no threshold, e.g.
        gap_positions: Insertion indices of the gap rows: ``i + 1`` for every step ``t[i+1] - t[i] > threshold``.
        insert_gaps: ``values`` (float) with a NaN inserted before each of ``positions`` (from gap_positions); the input is not changed.
        axis_break_pairs: (start, end) intervals to leave out of a broken axis.
        gamma_from_impedance: Reflection coefficient Gamma = (Z - Z0) / (Z + Z0) (complex array).
        impedance_from_gamma: Impedance Z = Z0 (1 + Gamma) / (1 - Gamma); Gamma = 1 gives inf.
        vswr_from_gamma: VSWR = (1 + |Gamma|) / (1 - |Gamma|); inf for |Gamma| >= 1.
        _joined: Concatenate (x, y) curves with a NaN point between them (Veusz breaks an xy line at NaN, nanHandling 'break-on', its default).
        SmithGrid (class): Datasets and widget specifications of a Smith-chart grid (0.0.26).
        _grid_line_settings: xy settings of a grid line family: no markers, no legend entry.
        _grid_label_settings: xy settings that print a text dataset at points (no line, no marker).
        smith_grid: The grid of a Smith chart as Veusz datasets and xy widget settings.
        smith_trace_datasets: Datasets of one Smith trace: "<stem>_re", "<stem>_im" and, when given, "<stem>_freq_Hz" (read by the suite's interactive Smith plot).
        smith_trace: Trace of Im(Gamma) against Re(Gamma) for the datasets of smith_trace_datasets(stem, ...); keyword arguments go to Trace.
        smith_has_key: True when a Smith graph gets a legend: key=True, or "auto" with more than one labelled trace (as xy_page).
        smith_graph_spec: Settings of a Smith graph and its children, for any Veusz builder.
        smith_page_layout: (page width, page height, graph top margin) of a Smith page: the chart about ``size`` across, 1 cm more height for the title, ``key_width`` more wi...
        smith_page: A page with one Smith chart; returns the graph name (0.0.26).
        text_page: Pages of plain text lines (escaped; monospaced by default) (0.0.12).
        step_mask: (x, y) arrays of a stepped mask: one flat segment per band.
        reference_line: (x, y) arrays of a straight threshold or marker line.
        bar_page: A page with a vertical category bar chart (0.0.14); returns the graph name.
        _reduce_axis: Reduce ``values`` along ``axis`` in groups of ``factor`` (see block_reduce).
        _reduce_groups: Collapse ``axis`` of a grouped array with ``method``.
        block_reduce: Decimate a 2-D array in blocks of ``row_factor`` x ``column_factor``.
        regularize_rows: Image rows with (possibly uneven) centres -> an even grid for drawing.
        dataset_names: Names of the datasets stored in a .vszh5 document (unescaped, sorted).
        read_text_dataset: The strings of a text dataset written by dataset_text(), or None if absent.
        page_names: Names of the pages of a written document, in page order (0.0.13).
        AppendLockedError (class): The document is locked by another process (HDF5 file locking).
        _is_lock_error: True for HDF5's "unable to lock file" open failure (any platform).
        VeuszAppender (class): Grow an existing .vszh5 document in place, a few rows at a time.
        _decode: bytes or numpy bytes -> str (UTF-8); anything else -> str().
        find_veusz_exe: Locate the Veusz program: explicit path, env var, PATH, usual folders.
        _popen_kwargs: Hide the console window of child processes on Windows.
        _render_script: The script piped into `veusz --listen`.
        _outline_parts: Split an outline entry into (title, page, children).
        add_pdf_bookmarks: Write a bookmark (outline) hierarchy into an existing PDF, in place.
        render_pages: Render a .vszh5 to one multipage PDF and per-page PNG files.
        _page_names_from_log: Veusz page names from the VSZH5_PAGES line the render script prints.
        _run_listen: Pipe a script into `veusz --listen --quiet` and wait for the marker.
        _render_in_process: Fallback: render with the veusz Python package in this process.
# %%%%% Variable Descriptions
    Define all utilized variables
        VSZH5_FORMAT: Highest document format the Veusz 3.6 and 4.2 loaders accept.
        VSZH5_LAYOUT_VERSION: Veusz release whose HDF5 layout this writer follows (written as vsz_version).
        COMPRESS_MIN_ELEMENTS: Arrays at least this long are gzip+shuffle compressed when compress=True.
        _VLEN_BYTES: text datasets
        APPEND_CHUNK_ELEMENTS: (0.0.18) Rows per chunk of a resizable (appendable) 1-D or date dataset: : 1024 float64 = 8 KiB, at the low end of the 10 KiB - 1 MiB range h5py :...
        PDF_DPI: Resolution passed to Veusz's Export() for PDF. Veusz 4.x lays a PDF page : out at the nearest resolution the Qt PDF engine supports (72) but sets t...
        VEUSZ_EPOCH: Veusz date origin, and the offset that maps Unix seconds onto it.
        UNIX_TO_VEUSZ_S: -1230768000
        _HOUR_OFFSET_CACHE: 0.0.28: UTC offset of each whole UTC hour, per zone, kept between calls. A project with thousands of time traces asked the zone database for the sa...
        _HOUR_OFFSET_CACHE_MAX: entries (about 114 years of hours)
        WIDGET_NAME_MAX: Longest widget name the helpers make (Veusz has no limit; long names make : the widget tree and PDF bookmarks hard to read).
        TRACE_COLOURS: 1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        _CM_PER_UNIT: Centimetres per unit for the physical Veusz distance units.
        MJD_UNIX_EPOCH: Modified Julian Date of the Unix epoch, 1970-01-01T00:00:00 UTC.
        ISO_DATETIME_FORMAT: Veusz tick format of an ISO-8601 date-time without offset (Veusz %VD codes).
        DISPLAY_TZ_ENV: (0.0.25) Environment variable that overrides the computer's time zone for : local date-time axes: "UTC", a fixed offset such as "-05:00" or "+0530"...
        _FIXED_OFFSET: fixed UTC offset text (-05:00, UTC-05:00, GMT+0100) for VSZH5_DISPLAY_TZ
        LOCAL_DATETIME_ROTATE: (0.0.25) Tick-label rotation (degrees) of an ISO-8601 date-time axis; a : 25-character "2026-11-09T08:39:00-05:00" label needs it.
        GAP_K: Default line-break multiplier K (gap when dt > K * median(dt)).
        GAP_AXIS_N: Default axis-break multiple n (axis cut when dt > n * K * median(dt)); 0 = never.
        GAP_AXIS_MARGIN: Fraction of a gap kept visible at each side of an axis break.
        SMITH_RESISTANCES: Normalized resistance (conductance) circles and reactance (susceptance) : arcs drawn by default - the values of scikit-rf's and the Plots tab's chart.
        SMITH_REACTANCES: normalised reactances of the Smith grid arcs
        SMITH_VSWR: VSWR circles of scikit-rf's draw_vswr=True.
        SMITH_GRID_COLOUR: a6a6a6"
        SMITH_FRAME_COLOUR: 4d4d4d"
        SMITH_LABEL_COLOUR: 595959"
        SMITH_LIMIT: Half-width of the Re/Im Gamma axes: room outside |Gamma| = 1 for labels.
        SMITH_NOTES_PREFIX: First words of the notes of every Smith graph (how readers recognise one).
        VEUSZ_LINE_BREAK: Veusz text markup for a line break (two backslashes in the label text).
        REDUCE_METHODS: Methods of block_reduce().
        APPEND_META_GROUP: Top-level HDF5 group for the appender's private bookkeeping (e.g. a logger's : column index). Veusz reads only /Veusz [R1: loader.loadHDF5Doc], so...
# %%%%% More Info
    Master copy: RF project document claude/vszh5_writer.py (registry
    claude/vszh5-writer-registry.md).  Needs numpy and h5py; Veusz only to open
    or render; pypdf only for PDF bookmarks; on Python 3.8 backports.zoneinfo
    or python-dateutil for IANA zone names (VSZH5_DISPLAY_TZ=America/New_York).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.27: FitsAutoPlot suite: FITS / Franks AutoPlot 0.4.0 (gap rule
Date: 2026-11-09
#              chosen by the user; real GBT sampler2log files), built on
#              master 0.0.26 (SHA-256 d826c679...a32a18). Additions plus
#              one label change:
#              * NEW TIME GAPS section: median_step(t) (median of the
#                positive finite steps), gap_threshold(t, k, manual)
#                = manual, or k * median_step(t) - a multiple of the
#                median step, never a ratio; gap_positions(t, threshold)
#                and insert_gaps(values, positions): NaN rows after
#                every step > threshold, so Veusz and pyqtgraph break
#                the line at a SMALL gap while the axis stays
#                continuous; axis_break_pairs(t, threshold, margin):
#                the (start, end) intervals of a broken axis at a LARGE
#                gap. GAP_K = 10, GAP_AXIS_N = 100 (large gap = 100 x
#                the small-gap threshold), GAP_AXIS_MARGIN = 0.02.
#              * FIX (for callers' broken axes) the interval shrink is a
#                fraction of the gap; FitsAutoPlot shrank every break by
#                1e-9 x |t|, about 5.3 s at MJD 61000, which removed
#                gaps up to 10.6 s from MJD axes.
#              * CHANGED local_time_axis(): across a DST change the
#                label gives the minute the offset changes (bisection,
#                e.g. "UTC-04:00 from 2026-03-08T03:00:00-04:00"), not
#                the first sample after it; tick formats and values
#                are unchanged.
#              Checked: tests/test_time_gaps.py 0.0.1 (FitsAutoPlot) and
#              Veusz 4.2.1 renders of the UWBR, tack and 1PPS files.
#              * FIX (Python 3.8) NEW iana_zone(name): an IANA zone name
#                (America/New_York) from zoneinfo (3.9+), else
#                backports.zoneinfo, else dateutil.tz; ValueError when
#                none knows it.  VSZH5_DISPLAY_TZ=America/New_York raised
#                ModuleNotFoundError: zoneinfo on 3.8; display_zone() now
#                warns (RuntimeWarning) and uses the computer's zone for
#                a name it cannot resolve.
#              * Header in the author's template (Header Info, Author
#                Information, References, Revisions with "# %%%%" entries,
#                Function/Variable Descriptions, More Info); docstrings for
#                every function, class and method.  No code change.
#              Checked on Python 3.8.20, 3.12.3 and 3.13.
# %%%%% Function Descriptions
        epoch_to_veusz: Unix epoch seconds -> Veusz date value (seconds since 2009-01-01).
        local_utc_offsets: Local UTC offset in seconds for every timestamp, DST-aware.
        vz_text: Escape Veusz text-markup characters so a string prints literally.
        escape_hdf_name: Escape a dataset name for use as an HDF5 member name, as Veusz does.
        widget_name: Any text -> a valid Veusz widget name (0.0.25).
        unique_widget_name: ``name``, or ``name_2``, ``name_3`` ...
        header_notes: Page notes text from a source-file header (0.0.25).
        set_notes: Set the notes of the current widget (a page) when ``notes`` is not empty.
        VszDoc (class): Builder for the Veusz command script stored in the document.
        VeuszWriter (class): Write Veusz datasets and a document script to a .vszh5 file.
        _default_file_mode: Mode a new file would get from open(): 0o666 minus the process umask.
        _one_dimensional: ``values`` as a contiguous float64 1-D array, or ValueError (0.0.23).
        _create_resizable_1d: Create ``group[key]``: a chunked, resizable, uncompressed float64 array.
        _is_regular: True when the spacing of ``values`` is constant (relative 1e-6).
        _image_axis: Choose and check the stored form of one image axis (see dataset_2d).
        Trace (class): One line or marker series (Veusz 'xy' widget) for xy_page().
        add_title: Centred title label near the top of the current page.
        title_settings: Settings of add_title()'s page title label (0.0.26: shared with the widget-tree builders, same values and order as before).
        _distance_cm: A physical Veusz distance ("15cm", "6in", "20pt") in cm; None otherwise.
        _graph_name: (0.0.25) Explicit name, else "g_<y>" for one trace, else "g_<page>".
        _open_page_and_graph: Add a page (sized, with notes), its title and a graph; leave the graph current.
        _add_axes: The x and y axes of the current graph, with labels, scales and limits.
        xy_page: A page with one graph of line/marker traces; returns the graph name.
        _add_traces: Add the xy widgets of ``traces`` to the current graph (0.0.25: named after Trace.name or the y dataset, unique in the graph); returns the names.
        _trace_settings: Settings of the xy widget that draws ``trace`` (relative paths, in the order Veusz receives them); shared by _add_traces and smith_graph_spec (0.0....
        _add_key: Legend in the current graph: True, False or "auto" (more than one label).
        image_page: A page with a 2-D dataset drawn as an image, and a colour bar; returns the graph name.
        Panel (class): One graph of grid_page(): traces, axes and a small title (0.0.12).
        grid_page: A page with several graphs in a Veusz grid (rows filled left to right).
        linear_power_mean_dbm: Equal-weight power mean of finite dBm samples, returned in dBm.
        frequency_ghz_text: Scale decimal Hz to GHz without coarse rounding/name collisions.
        frequency_time_pages: Append one page per frequency and a combined page; preserve all raw data.
        mjd_to_epoch: UTC Modified Julian Date (days) -> Unix epoch seconds.
        mjd_to_veusz: UTC MJD -> Veusz date seconds; ``offset_s`` shifts to a local clock.
        iso_offset_text: UTC offset in seconds -> ISO-8601 "+HH:MM" / "-HH:MM" (e.g.
        _zone_from_text: "UTC"/"Z", "+HH:MM", "-5", "UTC-05:00" or an IANA name -> tzinfo.
        iana_zone: An IANA zone ("America/New_York") as a tzinfo on every Python (0.0.27).
        display_zone: Zone for local date-time axes (0.0.25): VSZH5_DISPLAY_TZ when set, else ``default`` (None = the time zone of this computer).
        _zone_offsets: Per-sample UTC offset (s) in ``zone`` (None = VSZH5_DISPLAY_TZ, else this computer's zone).
        LocalTimeAxis (class): A local wall-clock date axis for Veusz (0.0.24), see local_time_axis().
        _offset_change_instant: First whole UTC minute in (before, after] whose offset is ``offset_after`` (bisection; the zone changes offset once between two samples) (0.0.27).
        local_time_axis: Veusz date values and an ISO-8601 tick format on a local clock.
        veusz_to_epoch: Veusz date seconds (since 2009-01-01, UTC) -> Unix epoch seconds (0.0.25).
        local_time_from_veusz: local_time_axis() for UTC values already in Veusz date seconds (0.0.25).
        style_datetime_axis: Make ``axis`` (path from the current widget, e.g.
        iso_local_strings: ISO-8601 local date-times with offsets, e.g.
        with_x: The same traces drawn against another x dataset (0.0.24).
        time_pair_pages: Two xy pages from one set of traces: against MJD and against local time.
        median_step: Median of the positive, finite steps of ``t`` (0.0 when there is none).
        gap_threshold: Line-break threshold in the units of ``t``: ``manual`` when > 0, else ``k * median_step(t)`` (0.0 = no threshold, e.g.
        gap_positions: Insertion indices of the gap rows: ``i + 1`` for every step ``t[i+1] - t[i] > threshold``.
        insert_gaps: ``values`` (float) with a NaN inserted before each of ``positions`` (from gap_positions); the input is not changed.
        axis_break_pairs: (start, end) intervals to leave out of a broken axis.
        gamma_from_impedance: Reflection coefficient Gamma = (Z - Z0) / (Z + Z0) (complex array).
        impedance_from_gamma: Impedance Z = Z0 (1 + Gamma) / (1 - Gamma); Gamma = 1 gives inf.
        vswr_from_gamma: VSWR = (1 + |Gamma|) / (1 - |Gamma|); inf for |Gamma| >= 1.
        _joined: Concatenate (x, y) curves with a NaN point between them (Veusz breaks an xy line at NaN, nanHandling 'break-on', its default).
        SmithGrid (class): Datasets and widget specifications of a Smith-chart grid (0.0.26).
        _grid_line_settings: xy settings of a grid line family: no markers, no legend entry.
        _grid_label_settings: xy settings that print a text dataset at points (no line, no marker).
        smith_grid: The grid of a Smith chart as Veusz datasets and xy widget settings.
        smith_trace_datasets: Datasets of one Smith trace: "<stem>_re", "<stem>_im" and, when given, "<stem>_freq_Hz" (read by the suite's interactive Smith plot).
        smith_trace: Trace of Im(Gamma) against Re(Gamma) for the datasets of smith_trace_datasets(stem, ...); keyword arguments go to Trace.
        smith_has_key: True when a Smith graph gets a legend: key=True, or "auto" with more than one labelled trace (as xy_page).
        smith_graph_spec: Settings of a Smith graph and its children, for any Veusz builder.
        smith_page_layout: (page width, page height, graph top margin) of a Smith page: the chart about ``size`` across, 1 cm more height for the title, ``key_width`` more wi...
        smith_page: A page with one Smith chart; returns the graph name (0.0.26).
        text_page: Pages of plain text lines (escaped; monospaced by default) (0.0.12).
        step_mask: (x, y) arrays of a stepped mask: one flat segment per band.
        reference_line: (x, y) arrays of a straight threshold or marker line.
        bar_page: A page with a vertical category bar chart (0.0.14); returns the graph name.
        _reduce_axis: Reduce ``values`` along ``axis`` in groups of ``factor`` (see block_reduce).
        _reduce_groups: Collapse ``axis`` of a grouped array with ``method``.
        block_reduce: Decimate a 2-D array in blocks of ``row_factor`` x ``column_factor``.
        regularize_rows: Image rows with (possibly uneven) centres -> an even grid for drawing.
        dataset_names: Names of the datasets stored in a .vszh5 document (unescaped, sorted).
        read_text_dataset: The strings of a text dataset written by dataset_text(), or None if absent.
        page_names: Names of the pages of a written document, in page order (0.0.13).
        AppendLockedError (class): The document is locked by another process (HDF5 file locking).
        _is_lock_error: True for HDF5's "unable to lock file" open failure (any platform).
        VeuszAppender (class): Grow an existing .vszh5 document in place, a few rows at a time.
        _decode: bytes or numpy bytes -> str (UTF-8); anything else -> str().
        find_veusz_exe: Locate the Veusz program: explicit path, env var, PATH, usual folders.
        _popen_kwargs: Hide the console window of child processes on Windows.
        _render_script: The script piped into `veusz --listen`.
        _outline_parts: Split an outline entry into (title, page, children).
        add_pdf_bookmarks: Write a bookmark (outline) hierarchy into an existing PDF, in place.
        render_pages: Render a .vszh5 to one multipage PDF and per-page PNG files.
        _page_names_from_log: Veusz page names from the VSZH5_PAGES line the render script prints.
        _run_listen: Pipe a script into `veusz --listen --quiet` and wait for the marker.
        _render_in_process: Fallback: render with the veusz Python package in this process.
# %%%%% Variable Descriptions
    Define all utilized variables
        VSZH5_FORMAT: Highest document format the Veusz 3.6 and 4.2 loaders accept.
        VSZH5_LAYOUT_VERSION: Veusz release whose HDF5 layout this writer follows (written as vsz_version).
        COMPRESS_MIN_ELEMENTS: Arrays at least this long are gzip+shuffle compressed when compress=True.
        APPEND_CHUNK_ELEMENTS: (0.0.18) Rows per chunk of a resizable (appendable) 1-D or date dataset: : 1024 float64 = 8 KiB, at the low end of the 10 KiB - 1 MiB range h5py :...
        PDF_DPI: Resolution passed to Veusz's Export() for PDF. Veusz 4.x lays a PDF page : out at the nearest resolution the Qt PDF engine supports (72) but sets t...
        VEUSZ_EPOCH: Veusz date origin, and the offset that maps Unix seconds onto it.
        UNIX_TO_VEUSZ_S: -1230768000
        WIDGET_NAME_MAX: Longest widget name the helpers make (Veusz has no limit; long names make : the widget tree and PDF bookmarks hard to read).
        TRACE_COLOURS: 1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        _CM_PER_UNIT: Centimetres per unit for the physical Veusz distance units.
        MJD_UNIX_EPOCH: Modified Julian Date of the Unix epoch, 1970-01-01T00:00:00 UTC.
        ISO_DATETIME_FORMAT: Veusz tick format of an ISO-8601 date-time without offset (Veusz %VD codes).
        DISPLAY_TZ_ENV: (0.0.25) Environment variable that overrides the computer's time zone for : local date-time axes: "UTC", a fixed offset such as "-05:00" or "+0530"...
        _FIXED_OFFSET: fixed UTC offset text (-05:00, UTC-05:00, GMT+0100) for VSZH5_DISPLAY_TZ
        LOCAL_DATETIME_ROTATE: (0.0.25) Tick-label rotation (degrees) of an ISO-8601 date-time axis; a : 25-character "2026-11-09T08:39:00-05:00" label needs it.
        GAP_K: Default line-break multiplier K (gap when dt > K * median(dt)).
        GAP_AXIS_N: Default axis-break multiple n (axis cut when dt > n * K * median(dt)); 0 = never.
        GAP_AXIS_MARGIN: Fraction of a gap kept visible at each side of an axis break.
        SMITH_RESISTANCES: Normalized resistance (conductance) circles and reactance (susceptance) : arcs drawn by default - the values of scikit-rf's and the Plots tab's chart.
        SMITH_REACTANCES: normalised reactances of the Smith grid arcs
        SMITH_VSWR: VSWR circles of scikit-rf's draw_vswr=True.
        SMITH_GRID_COLOUR: a6a6a6"
        SMITH_FRAME_COLOUR: 4d4d4d"
        SMITH_LABEL_COLOUR: 595959"
        SMITH_LIMIT: Half-width of the Re/Im Gamma axes: room outside |Gamma| = 1 for labels.
        SMITH_NOTES_PREFIX: First words of the notes of every Smith graph (how readers recognise one).
        VEUSZ_LINE_BREAK: Veusz text markup for a line break (two backslashes in the label text).
        REDUCE_METHODS: Methods of block_reduce().
        APPEND_META_GROUP: Top-level HDF5 group for the appender's private bookkeeping (e.g. a logger's : column index). Veusz reads only /Veusz [R1: loader.loadHDF5Doc], so...
# %%%%% More Info
    Master copy: RF project document claude/vszh5_writer.py (registry
    claude/vszh5-writer-registry.md).  Needs numpy and h5py; Veusz only to open
    or render; pypdf only for PDF bookmarks; on Python 3.8 backports.zoneinfo
    or python-dateutil for IANA zone names (VSZH5_DISPLAY_TZ=America/New_York).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.26: FitsAutoPlot suite: Touchstone AutoPlot 0.7.0 (Smith
Date: 2026-11-09
#              chart pages in the project), built on master 0.0.25
#              (SHA-256 fb481290...18b9a). Additions only: every 0.0.25
#              name, parameter, default and output is unchanged (the
#              scripts of xy/grid/image/bar pages compared byte for
#              byte; _add_traces and add_title now share their settings
#              with the new widget-tree builders).
#              * NEW smith_grid(prefix, kind="z"|"y", resistances=,
#                reactances=, vswr=, labels=, ...) -> SmithGrid: the
#                constant-resistance circles (centre r/(1+r), radius
#                1/(1+r)) and constant-reactance arcs (traced exactly as
#                Gamma(r) = (r + jx - 1)/(r + jx + 1)), the unit circle,
#                the real axis, optional VSWR circles (radius
#                (S-1)/(S+1)) and the grid values, as NaN-joined 1-D
#                datasets (one xy widget per family) and text datasets
#                printed as point labels. kind="y" rotates the grid by
#                180 degrees (admittance chart, as scikit-rf).
#                SmithGrid.write(target) writes the datasets with a
#                VeuszWriter or any object with SetData/SetDataText
#                (_autoplot_native.NativeDocument, veusz.embed).
#              * NEW smith_trace_datasets(stem, gamma, frequency_hz) ->
#                "<stem>_re", "<stem>_im", "<stem>_freq_Hz";
#                smith_trace(stem, label, ...) -> Trace(Im vs Re).
#              * NEW smith_graph_spec(grid, traces, ...) -> (graph
#                settings, [(type, name, settings), ...]): aspect ratio
#                1, hidden axes at +/-SMITH_LIMIT, no border, notes
#                "Smith chart (impedance|admittance)", the grid widgets
#                "grid_*", the traces, the legend (beside the chart by
#                default). smith_has_key(), smith_page_layout() and
#                smith_page(doc, page, traces, grid, title, ...) build a
#                whole page in a VszDoc script.
#              * NEW gamma_from_impedance, impedance_from_gamma,
#                vswr_from_gamma; SMITH_RESISTANCES, SMITH_REACTANCES,
#                SMITH_VSWR, SMITH_LIMIT, SMITH_NOTES_PREFIX and the
#                dataset suffixes; title_settings().
#              Checked: tests/test_smith_chart.py (grid geometry against
#              the circle equations, Gamma/Z/VSWR round trips, page
#              scripts) and renders of impedance and admittance charts
#              in real Veusz 4.2.1; every setting used exists in the
#              Veusz 3.6.2 source (graph aspect, axis hide, xy labels
#              and Label/posnHorz/posnVert, key horzManual).
# %%%% 0.0.25: FitsAutoPlot suite (ATR, CSV/TSV, FITS, Franks, R&S,
Date: 2026-11-09
#              Touchstone), built on master 0.0.24 (SHA-256
#              199f30a8...a901). Every 0.0.24 name and parameter is still
#              accepted; the CHANGED defaults are the widget names.
#              * CHANGED widget names of the page helpers, from the data
#                instead of generic names: xy_page graph "g_<y>" for one
#                trace, "g_<page>" for several (was "graph"); traces are
#                named after their y dataset (was "trace0", "trace1",
#                ...); grid_page graphs "g_<first y>" (was "graph0"...);
#                image_page graph "g_<dataset>", image "img_<dataset>",
#                colour bar "cbar_<dataset>" (were "graph", "image",
#                "colorbar"); bar_page graph "g_<page>" and bars
#                "bars_<first length>". New Trace.name, Panel.name and
#                graph_name= parameters choose a name explicitly
#                (graph_name="graph" gives the 0.0.24 layout). The page
#                helpers return the graph name(s); frequency_time_pages
#                no longer reaches into "graph/x" by path (it passes
#                datetime_format to the helpers), so its output is
#                unchanged apart from the names.
#              * NEW widget_name(text) (a valid Veusz widget name: no
#                '/', no surrounding blanks, not '.'/'..') and
#                unique_widget_name(name, used).
#              * NEW header_notes(header, source=, title=, extra=): page
#                notes from a source-file header - a mapping ("KEY =
#                value", like Veusz shows FITS cards), lines, or text -
#                and set_notes(doc, notes). xy_page, image_page,
#                grid_page, bar_page, text_page and time_pair_pages take
#                notes= and write it to the page's notes (Veusz:
#                Properties > notes).
#              * NEW veusz_to_epoch(), display_zone() and
#                local_time_from_veusz(veusz_utc, zone=None, label=):
#                the LocalTimeAxis of UTC date values that are already
#                in Veusz seconds (FITS MJD, CSV dates). zone=None now
#                means "VSZH5_DISPLAY_TZ if set, else this computer's
#                zone"; the variable takes "UTC", a fixed offset
#                ("-05:00") or, on Python >= 3.9, an IANA name. Fixed
#                offsets also work on Python 3.8 (no zoneinfo).
#              * NEW LOCAL_DATETIME_ROTATE (45) and
#                style_datetime_axis(doc, axis, local_axis): mode
#                datetime, ISO-8601 24-hour tick format with the offset
#                (%VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00), label naming
#                the offset, exact range, rotated tick labels.
#              Checked: tests in FitsAutoPlot (writer names, notes and
#              local axes) and renders in real Veusz 4.2.1.
# %%%% 0.0.24: Released with MHM2010 / T4 Maser Timing Workbench 0.2.0
Date: 2026-11-03
#              (Maser and timing analysis chat), built on master 0.0.23
#              (SHA-256 90833d76...eaac6). Additions only: every 0.0.23
#              name, parameter, default and output is unchanged.
#              * NEW mjd_to_epoch(mjd) and mjd_to_veusz(mjd, offset_s):
#                UTC MJD days -> Unix seconds / Veusz date seconds
#                (Unix epoch = MJD 40587).
#              * NEW iso_offset_text(offset_s): "+HH:MM" / "-HH:MM".
#              * NEW LocalTimeAxis and local_time_axis(epoch_s, zone=None):
#                Veusz date values on the wall clock of the computer that
#                processes the data (or an IANA zone, Python >= 3.9), per
#                sample and DST-aware (local_utc_offsets), plus a Veusz
#                tick format giving ISO-8601 date-times. With one offset
#                over the data the offset is part of every tick label
#                ("%VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00"); Veusz formats
#                are static, so when the data cross a DST change the
#                ticks carry no offset and the axis label lists each
#                offset and the instant it starts.
#              * NEW iso_local_strings(epoch_s, zone=None): the same
#                instants as ISO-8601 text with offsets (text datasets,
#                CSV columns).
#              * NEW with_x(traces, x): the same traces on another x
#                dataset; time_pair_pages(): one xy page against MJD and
#                one against local date-time from the same traces.
#              * NEW Panel.datetime_format (default ""): tick format of a
#                datetime x axis in grid_page().
#              Checked: tests/test_writer_time_axes.py (workbench) and
#              renders in real Veusz 3.6.2 (ISO ticks with offset).
# %%%% 0.0.23: Built on 0.0.22 (MHM2010 Workbench review), with the open
Date: 2026-11-03
#              points of the RF_Equipment_Comms and GBO Maser Control
#              reviews of 2026-11-03. Every public name, parameter and
#              default is unchanged; valid 1-D input is written exactly
#              as before.
#              * CORRECTION to the 0.0.22 note: VeuszWriter(
#                resizable=True) never flattened a 2-D date array - its
#                resizable layout is used for 1-D input only, so the
#                array was stored 2-D, as without resizable. The silent
#                flattening was in VeuszAppender (next item). Checked
#                with 0.0.21 and 0.0.22 (h5py shapes, real Veusz).
#              * FIX VeuszAppender.create_1d(), create_date() and
#                append(): input with more than one dimension is
#                rejected (ValueError) instead of being flattened with
#                ravel() into a longer column (a (2, 3) array became 6
#                rows). A single number is still one row; an empty
#                sequence still creates an empty dataset.
#              * FIX VeuszAppender.append()/truncate() on a dataset that
#                an earlier writer stored with more than one dimension
#                (0.0.21 dataset_date, 2-D input): rejected with a
#                message saying the document must be rewritten, and the
#                file is left unchanged. 0.0.22 flattened it (2 x 3 dates
#                + 1 row became 7 rows) while converting it to the
#                growable layout.
#              * FIX VeuszWriter.dataset_1d() and dataset_text(): every
#                check and conversion now runs before the dataset group
#                is created, as 0.0.22 did for dataset_date(). A wrong
#                serr/perr/nerr length, or text input that is not a
#                sequence, left an empty group behind and blocked the
#                name ("written twice") for a corrected retry.
#              * FIX render_pages() reported success for a document whose
#                pages loaded but whose data did not. Veusz's loader
#                (loader.loadHDF5Doc) builds the pages first and loads
#                the datasets afterwards, and `veusz --listen` runs each
#                script line on its own and only prints an exception, so
#                a dataset Veusz refuses ("Only 1-dimensional arrays or
#                lists allowed", e.g. a 0.0.21 2-D date) gave PDF and PNG
#                files of empty 0..1 axes and no error. The load now
#                reports its pages only when the whole load succeeded;
#                otherwise no page is exported (an earlier PDF of the
#                same name is kept) and the render fails with Veusz's own
#                message, then method "auto" tries the veusz package.
#              * FIX render_pages(method="inprocess" or the "auto"
#                fallback): a load error from the veusz package is raised
#                as RuntimeError, as documented, instead of escaping as a
#                bare ValueError (which also hid the program error in
#                "auto" mode).
#              Checked: probe_writer_issues.py 0.0.1 (16 cases, 0.0.21 /
#              0.0.22 / 0.0.23 side by side) with real Veusz 3.6.2 and
#              4.2.1; tests/test_vszh5_writer.py 0.0.3; and as a drop-in
#              in the FSP logger 0.10.0, SHR Converter 0.0.2 and AB Power
#              Meter Monitor 1.7.0 suites (see the 0.0.23 release notes).
# %%%% 0.0.22: MHM2010 Workbench review. FIX dataset_date(): reject
Date: 2026-11-03
#              multidimensional input before creating the HDF5 group.
#              Previously a 2-D date array silently produced an invalid
#              date dataset (or was flattened with resizable=True).
#              No valid 1-D caller changes; no other writer code changed.
# %%%% 0.0.21: Released with SHR Converter 0.0.2. FIX regularize_rows():
Date: 2026-10-28
#              with jittered time stamps the grid step came from the
#              median interval, whose slots then drifted from the rows
#              and left an empty (NaN) row every few dozen rows - seen
#              on a real SM200C recording (32-35 ms sweep intervals,
#              median 33 ms, mean 33.35 ms: a blank line about every
#              47 sweeps). The step is now the MEAN of the ordinary
#              intervals; an interval longer than (1 + 2 x tolerance)
#              median intervals is a gap and gets rint(gap / step)
#              slots. Rows that are not strictly increasing still
#              fall back to "even". Signature and return values are
#              unchanged; nothing else in the writer changed (drop-in
#              for every program that uses 0.0.20).
# %%%% 0.0.20: Released with SHR Converter 0.0.1. One writer again: the
Date: 2026-10-28
#              two releases numbered 0.0.18 (see that entry) had
#              diverged, and 0.0.19 had been built on the AB-monitor
#              branch only, so it lacked the FSP-branch render fixes.
#              Every public call of both branches is kept unchanged.
#              * MERGED from 0.0.18 (FSP logger 0.9.0 branch):
#                - FIX render_pages(png_pages="first"): the --listen
#                  script bounded its PNG loop with min(1, len(names));
#                  Veusz 4.2.1 evaluates listen scripts with NumPy's
#                  names imported, so min was numpy.min(1, axis=n) ->
#                  AxisError, printed and swallowed: no PNG at all. The
#                  bound is now (1 if names else 0). Present in 0.0.19.
#                - FIX render_pages(): the PDF was named with
#                  stem.with_suffix(".pdf"), so "run v0.0.2.vszh5" gave
#                  "run v0.0.pdf"; it is now "<file stem>.pdf" whatever
#                  dots the name holds (the PNGs already were). Present
#                  in 0.0.19.
#                - FIX render_pages(): an Export() error inside
#                  `veusz --listen` is printed, not raised, and the done
#                  marker still arrives; a requested PDF or PNG that was
#                  not written now raises RuntimeError instead of
#                  returning a list without it. Present in 0.0.19.
#                - Spyder cell markers of the frequency-page (0.0.16) and
#                  linear-power (0.0.17) sections put back above their
#                  own code (0.0.19 had the two markers stacked above
#                  linear_power_mean_dbm()).
#              * KEPT from 0.0.19 and widened: Veusz finishing the script
#                with no page loaded is a failed program render whether
#                or not its log holds a Traceback (0.0.19 needed one; the
#                FSP branch did not), with Veusz's own message and a hint
#                that Veusz needs h5py to open .vszh5. Method "auto"
#                then tries the veusz Python package, as before.
#              * KEPT from 0.0.18 (AB-monitor branch): VeuszAppender,
#                AppendLockedError, VeuszWriter(resizable=True),
#                APPEND_CHUNK_ELEMENTS, APPEND_META_GROUP.
#              * NEW image_page(xmode=, ymode=, datetime_format=): either
#                image axis can be a date axis ("datetime"), so a
#                spectrogram can show UTC time on y; the image range is
#                then given in Veusz date seconds (epoch_to_veusz).
#                Defaults unchanged ("numeric").
#              * NEW xy_page(datetime_format=): tick-label format of a
#                date x axis in one call (frequency_time_pages() set it
#                afterwards; it still works that way).
#              * NEW block_reduce(values, row_factor, column_factor,
#                method): decimates a 2-D array in blocks for plotting -
#                "max" (peak hold: a short or narrow signal survives),
#                "min", "mean", "power_mean_db" (dB values averaged as
#                power, 10 log10 mean 10^(x/10), max-referenced so it
#                cannot overflow) and "power_mean_linear" (amplitudes
#                averaged as power, sqrt(mean x^2), i.e. RMS). Trailing
#                partial blocks are reduced over the rows/columns they
#                have, never padded. Works on NumPy arrays or, with
#                xp=cupy, on the GPU. REDUCE_METHODS lists the names.
#              * FIX (Linux/macOS): documents published by finish() had
#                mode 0600 - the private mode of the mkstemp() temporary
#                file used since 0.0.15 - so other users (a web server,
#                Grafana, a colleague) could not read them. finish() now
#                gives the file the mode it replaces, or a new file the
#                usual 0666 minus the umask. No effect on Windows.
#              * NEW render_pages(png_background=): background colour of
#                the PNG files ("white", "#ffffffff", ...). The default
#                "" keeps Veusz's transparent background as before.
#              * NEW regularize_rows(image, centres): puts image rows
#                with uneven time stamps onto an even grid, with NaN
#                (transparent) rows where the recording paused. Veusz
#                draws an image whose x/y axis is given as uneven
#                edges/centres through its slow non-linear path:
#                measured on a 1964 x 1334 three-hour spectrogram, 30.3 s
#                and 3.2 GB to render against 2.7 s and 0.31 GB on an
#                even grid (Veusz 3.6.2). image_page() docstring notes it.
#              * DIFFERENCES list completed (0.0.16, 0.0.17) and put in
#                version order.
#              Checked: tests/test_vszh5_writer.py (stand-in Veusz render
#              protocol, block_reduce against loops, regularize_rows,
#              file modes, date-axis images rendered in real Veusz
#              3.6.2 and 4.2.1), and as a drop-in replacement in every
#              program that ships a copy (their own suites, unchanged
#              apart from version literals):
#                - R&S FSP logger 0.9.0: 44/44 with PyQt5 and PySide6,
#                  each with Veusz 3.6.2 and 4.2.1 (the About test's
#                  expected writer version updated 0.0.18 -> 0.0.20);
#                - AB Power Meter Monitor 1.7.0: 8 suites, 45/45, real
#                  renders in Veusz 4.2.1 and 3.6.2;
#                - SA Time Stability Tools 0.0.3: run_all 7/7 and
#                  validate_vszh5_in_veusz 22/22 (Veusz 4.2.1);
#                - QuickSyn package 0.0.10: 44 pass; its three checks
#                  that pin the writer's version string (0.0.10) and its
#                  pre-0.0.15 temporary name "<file>.part" fail as they
#                  do with 0.0.19 - test literals, not behaviour.
# %%%% 0.0.19: render_pages(): when Veusz runs the script but cannot load
Date: 2026-10-26
#              the document (a Traceback and no page in its log - e.g. a
#              Veusz environment without h5py: "No HDF5 support as h5py
#              module is missing"), the render now fails with Veusz's own
#              message instead of returning an empty list. Released with
#              AB Power Meter Monitor 1.6.2.
# %%%% 0.0.18: Two releases carried this number (merged in 0.0.20):
Date: 2026-10-26
#              (a) R&S FSP logger bundle 0.9.0 - fixes only; the HDF5
#              layout and every public signature unchanged:
#              * FIX render_pages(png_pages="first"): the --listen
#                script limited the PNG loop with min(1, len(names)).
#                Veusz 4.2.1 runs listen scripts with NumPy's names
#                imported, so min was numpy.min(1, axis=n) -> AxisError,
#                printed and swallowed by --listen: no PNG was written.
#                The bound is now (1 if names else 0). Reported against
#                0.0.5 by SA Time Stability Tools 0.0.3; still present
#                through 0.0.17.
#              * FIX render_pages(): the PDF path was
#                stem.with_suffix(".pdf"), so "run v0.0.2.vszh5" gave
#                "run v0.0.pdf". It is now "<file stem>.pdf" whatever
#                dots the name holds (the PNG names already were).
#              * FIX render_pages(): a load or Export() error inside
#                `veusz --listen` is printed, not raised, and the done
#                marker still arrives, so 0.0.17 returned success with
#                no files (found with a Veusz lacking h5py). Zero
#                loaded pages is now a failed program render (method
#                "auto" then tries the veusz package), and a missing
#                PDF or PNG raises RuntimeError.
#              * Revision history repaired: 0.0.17 listed two blocks
#                with the same number; the frequency-page block is
#                0.0.16. DIFFERENCES list in version order.
#              (b) AB Power Meter Monitor 1.6.0 - additions only; every
#              0.0.17 call is unchanged:
#              * VeuszAppender: open an existing .vszh5 'r+' (or create
#                an empty, valid one atomically) and append rows to
#                resizable 1-D and date datasets, create new datasets
#                back-filled with NaN, pad or truncate datasets to a
#                length, replace the document script, add tags, and keep
#                private JSON bookkeeping under /VszAppend (ignored
#                by Veusz). AppendLockedError is raised when another
#                process holds the HDF5 file lock, so a logger can
#                retry on its next poll instead of failing.
#              * VeuszWriter(resizable=True): 1-D and date datasets
#                are written chunked and resizable (fill value NaN,
#                uncompressed) so the document can be appended later.
#              * APPEND_CHUNK_ELEMENTS (1024 rows = 8 KiB per chunk).
#              * Revision history: the second "0.0.17" block (frequency
#                pages) is renumbered 0.0.16, so versions are unique
#                and in descending numeric order; the section marker
#                of frequency_time_pages() says 0.0.16 to match.
#              Checked by loading appended documents in real Veusz
#              4.2.1 (conda-forge) and exporting PNG and PDF.
# %%%% 0.0.17: Consolidate numerically stable, chunked linear-power dBm
Date: 2026-10-23
#              averaging for the merged logger/report release 0.8.3.
# %%%% 0.0.16: Add precise frequency_ghz_text() and frequency_time_pages():
Date: 2026-10-23
#              Freq_N_GHz and Combined_Freq_vs_Time with datetime axes.
#              Independent analyzer traces are never averaged together.
# %%%% 0.0.15: Python 3.8-safe typing alias; private atomic temp files;
Date: 2026-10-23
#              flushed renderer markers to prevent export-complete hangs.
# %%%% 0.0.14: Released with the GBT RFI Interference Evaluator 0.0.1.
Date: 2026-10-19
#              Additions only; every 0.0.13 call is unchanged:
#              * Trace.fill ("below" | "above" | ""), fill_colour and
#                fill_transparency: shades the region between a trace
#                and the bottom/top of the graph (Veusz xy FillBelow /
#                FillAbove), e.g. the part of a spectrum above a mask.
#              * bar_page(): category bar chart (Veusz 'bar' widget,
#                x axis in 'labels' mode from a text dataset), linear
#                or log value axis, legend for grouped bars.
#              * step_mask(): (x, y) arrays of a stepped mask from band
#                edges and levels, NaN between bands so Veusz and
#                pyqtgraph draw separate steps.
#              * reference_line(): two-point (x, y) arrays for a
#                horizontal or vertical threshold/marker line.
#              * Revision history re-ordered newest first (numeric
#                version order), the author's house style.
#              Checked by rendering in Veusz 3.6.2 (PDF and PNG).
# %%%% 0.0.13: Released with EMF Envelope FFT Analyzer 0.0.4. Additions
Date: 2026-10-15
#              only:
#              * dataset_names(), read_text_dataset() and page_names():
#                read back what a document holds without Veusz (e.g. the
#                unit sections an analyzer stored as a text dataset, to
#                rebuild the PDF bookmarks when a document is rendered
#                later, or to list its pages).
# %%%% 0.0.12: Released with EMF Envelope FFT Analyzer 0.0.3. Additions
Date: 2026-10-15
#              only; every earlier call is unchanged:
#              * Panel and grid_page(): several graphs on one page in a
#                Veusz grid widget (rows x columns), each with its own
#                traces, axes, log/date modes and a small panel title.
#              * text_page(): lines of text (monospaced by default) on a
#                page, continued on further pages when long; Veusz
#                markup is escaped and lines joined with Veusz's line
#                break, so a summary prints as written.
#              * "Veusz not found" messages name both variables that
#                find_veusz_exe() reads (VEUSZ_EXE, QUICKSYN_VEUSZ).
#              * xy_page() and grid_page() share _add_traces/_add_key.
#              Checked by rendering in Veusz 4.2.1 (PDF and PNG).
# %%%% 0.0.11: Released with EMF Envelope FFT Analyzer 0.0.2. Additions
Date: 2026-10-15
#              only; every 0.0.10 call is unchanged:
#              * VeuszWriter.dataset_2d(): 2-D (image) datasets in the
#                layout of Veusz's Dataset2D.saveDataDumpToHDF5 - 'data'
#                (ny, nx) plus xrange/xedge/xcent and yrange/yedge/ycent,
#                linked by vsz_<key> object references. Regular edges or
#                centres are stored as a range, as Veusz does on load.
#              * Plot-page helpers: TRACE_COLOURS, Trace, add_title(),
#                xy_page() and image_page(). image_page() places the
#                colour bar manually (horzPosn/vertPosn 'manual', right
#                of the graph, labels reflected): with the default
#                placement Veusz 4.2.1 does not draw it.
#              * xy_page(xmode="datetime") for date datasets on x;
#                x_autorange/y_autorange ("exact" for time axes).
#                Image axes and the colour bar use "exact" ranges, so
#                the bar is not padded with white past zmin/zmax.
#              Checked by rendering in Veusz 4.2.1 (PDF and PNG).
# %%%% 0.0.10: Released with QuickSyn package 0.0.10. The GUI now also
Date: 2026-10-14
#              uses VszDoc/VeuszWriter in-process to export a power
#              calibration table (Calibration tab). No change to this
#              module's code.
# %%%% 0.0.9: Released with QuickSyn package 0.0.9 (10 MHz output command
Date: 2026-10-12
#              path in the GUI). No change to this module's code.
# %%%% 0.0.8: Released with QuickSyn package 0.0.8 (hardware-session
Date: 2026-10-12
#              fixes in the GUI). No change to this module's code.
# %%%% 0.0.7: Released with QuickSyn package 0.0.7 (output-power fix in
Date: 2026-10-12
#              the GUI). No change to this module's code.
# %%%% 0.0.6: add_pdf_bookmarks(): writes a bookmark (outline) hierarchy
Date: 2026-10-11
#              and the document title, author and subject into a PDF with
#              pypdf, atomically, and opens the outline pane by default.
#              render_pages() calls it for the multipage PDF (new
#              parameters bookmarks= and pdf_info=; one bookmark per page
#              unless the caller supplies a hierarchy); skipped with a
#              warning, never an error, when pypdf is not installed.
#              FIX (Veusz 4.x): PDF pages were drawn at 48 % scale in the
#              top-left corner. Veusz 4.2.1 (and master) lays a PDF out
#              at 72 dpi, the only resolution Qt's PDF engine reports,
#              but sets the printer to Export()'s pdfdpi, default 150;
#              PDFs are now exported with pdfdpi=72 (PDF_DPI). Veusz
#              3.6.2 was unaffected. The PNGs were always correct, which
#              is why the 0.0.4 checks, which inspected PNGs, missed it.
#              FIX: render_pages() returns the files it wrote even when
#              they already existed. 0.0.5 compared against the folder
#              listing taken at the start, so a second render into the
#              same folder returned an empty list (the files were
#              rewritten, but the export reported no plots).
#              Released with QuickSyn package 0.0.6 (0.0.5 was never
#              packaged; its changes ship here).
# %%%% 0.0.5: render_pages(): Windows paths fixed (the PNG path sat
Date: 2026-10-11
#              inside an exec() string, so C:\\Users... was unescaped twice;
#              it is now bound on its own line); method "auto" falls back
#              to the veusz package when the Veusz program found fails to
#              render (e.g. a distribution package built for another
#              Python) and reports both errors otherwise. A program that
#              exits before reading its script (broken pipe) or cannot
#              be started is now reported as a failed render rather than
#              raising BrokenPipeError/OSError, so the fallback runs. The
#              stdout pipe is closed (ResourceWarning) and Veusz is never
#              left running after an error.
# %%%% 0.0.4: First version, released with QuickSyn package 0.0.4
Date: 2026-10-11
#              (version follows the package). Extracted from
#              isp_speed_monitor.py 1.3.0 and extended as listed above.
===============================================================================
"""

# %% ===========================================================================
#    IMPORTS
# =============================================================================
from __future__ import annotations

import datetime as _dt
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, replace as _replace
from decimal import Context, Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union, List, Mapping

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - reported when a writer is created
    h5py = None

__version__ = "0.0.29"                 # Semantic Versioning 2.0.0
__author__ = "William W. Wallace"
__email__ = "wwallace@nrao.edu"
__company__ = "NRAO/GBO"
__license__ = "GPL-3.0-or-later"      # SPDX identifier
__copyright__ = "Copyright (C) 2026 William W. Wallace"

__all__ = [
    "VSZH5_FORMAT", "VSZH5_LAYOUT_VERSION", "VEUSZ_EPOCH", "UNIX_TO_VEUSZ_S",
    "epoch_to_veusz", "local_utc_offsets", "vz_text", "escape_hdf_name",
    "VszDoc", "VeuszWriter", "find_veusz_exe", "render_pages",
    "add_pdf_bookmarks", "OutlineEntry",
    # 0.0.11: plot-page helpers
    "TRACE_COLOURS", "Trace", "add_title", "xy_page", "image_page",
    # 0.0.12: multi-panel and text pages
    "Panel", "grid_page", "text_page",
    # 0.0.16 / 0.0.17: frequency-time pages, linear-power mean
    "frequency_ghz_text", "frequency_time_pages", "linear_power_mean_dbm",
    # 0.0.13: reading a written document back
    "dataset_names", "read_text_dataset", "page_names",
    # 0.0.14: fills, bar charts, step masks, reference lines
    "bar_page", "step_mask", "reference_line",
    # 0.0.18: growing a document in place
    "APPEND_CHUNK_ELEMENTS", "APPEND_META_GROUP", "VeuszAppender", "AppendLockedError",
    # 0.0.20: decimating images for plotting
    "REDUCE_METHODS", "block_reduce", "regularize_rows",
    # 0.0.24: MJD and local ISO-8601 time axes
    "MJD_UNIX_EPOCH", "ISO_DATETIME_FORMAT", "mjd_to_epoch", "mjd_to_veusz",
    "iso_offset_text", "LocalTimeAxis", "local_time_axis", "iso_local_strings",
    "with_x", "time_pair_pages", "ISO_TIMESPECS", "mjd_iso_local_strings",
    # 0.0.25: names, page notes, local display of stored UTC dates
    "widget_name", "unique_widget_name", "header_notes", "set_notes",
    "veusz_to_epoch", "DISPLAY_TZ_ENV", "display_zone", "local_time_from_veusz",
    "LOCAL_DATETIME_ROTATE", "style_datetime_axis",
    # 0.0.26: Smith charts
    "SMITH_RESISTANCES", "SMITH_REACTANCES", "SMITH_VSWR", "SMITH_LIMIT",
    "SMITH_NOTES_PREFIX", "SMITH_RE_SUFFIX", "SMITH_IM_SUFFIX", "SMITH_FREQUENCY_SUFFIX",
    "gamma_from_impedance", "impedance_from_gamma", "vswr_from_gamma", "SmithGrid",
    "smith_grid", "smith_trace_datasets", "smith_trace", "smith_has_key",
    "smith_graph_spec", "smith_page_layout", "smith_page", "title_settings",
    # 0.0.27: time gaps
    "GAP_K", "GAP_AXIS_N", "GAP_AXIS_MARGIN", "median_step", "gap_threshold",
    "gap_positions", "insert_gaps", "axis_break_pairs",
]

# %% ===========================================================================
#    CONSTANTS AND TIME HELPERS
# =============================================================================
#: Highest document format the Veusz 3.6 and 4.2 loaders accept.
VSZH5_FORMAT = 1
#: Veusz release whose HDF5 layout this writer follows (written as vsz_version).
VSZH5_LAYOUT_VERSION = "4.2.1"
#: Arrays at least this long are gzip+shuffle compressed when compress=True.
COMPRESS_MIN_ELEMENTS = 4096
_VLEN_BYTES = h5py.special_dtype(vlen=bytes) if h5py is not None else None   # text datasets
#: (0.0.18) Rows per chunk of a resizable (appendable) 1-D or date dataset:
#: 1024 float64 = 8 KiB, at the low end of the 10 KiB - 1 MiB range h5py
#: recommends, because each append rewrites the partly filled last chunk.
APPEND_CHUNK_ELEMENTS = 1024
#: Resolution passed to Veusz's Export() for PDF. Veusz 4.x lays a PDF page
#: out at the nearest resolution the Qt PDF engine supports (72) but sets the
#: printer to the requested pdfdpi (Export() defaults to 150), so the drawing
#: lands in the top-left 48 % of each page. 72 keeps both the same; Veusz
#: 3.6 is unaffected, and PDF output is vector, so nothing is lost.
PDF_DPI = 72

#: Veusz date origin, and the offset that maps Unix seconds onto it.
VEUSZ_EPOCH = _dt.datetime(2009, 1, 1)
UNIX_TO_VEUSZ_S = (_dt.datetime(1970, 1, 1) - VEUSZ_EPOCH).total_seconds()  # -1230768000


def epoch_to_veusz(epoch_s: Any, offset_s: Any = 0.0) -> np.ndarray:
    """Unix epoch seconds -> Veusz date value (seconds since 2009-01-01).

    Pass the UTC offset in seconds (scalar or per sample) to get a local
    wall-clock axis; leave it 0 for UTC.
    """
    return (np.asarray(epoch_s, dtype=float) + np.asarray(offset_s, dtype=float)
            + UNIX_TO_VEUSZ_S)


def local_utc_offsets(epoch_s: Any) -> np.ndarray:
    """Local UTC offset in seconds for every timestamp, DST-aware.

    Offsets only change on whole hours, so the lookup is done once per
    distinct UTC hour and broadcast, which keeps it fast for long logs.
    """
    t = np.asarray(epoch_s, dtype=float)
    out = np.zeros_like(t)
    good = np.isfinite(t)
    if not good.any():
        return out
    hours = np.floor(t[good] / 3600.0)
    uniq, inverse = np.unique(hours, return_inverse=True)
    out[good] = _hour_offsets(uniq, None)[inverse]
    return out


# 0.0.28: UTC offset of each whole UTC hour, per zone, kept between calls.
# A project with thousands of time traces asked the zone database for the
# same hours again for every trace (about 183 000 astimezone() calls for the
# 30-day GBT 1PPS set).  Offsets only change on whole hours.
_HOUR_OFFSET_CACHE: dict = {}
_HOUR_OFFSET_CACHE_MAX = 1_000_000          # entries (about 114 years of hours)


def _hour_offsets(hours: np.ndarray, zone: Any) -> np.ndarray:
    """UTC offset (s) at the middle of each UTC hour number in ``hours``.

    ``zone`` None means this computer's zone.  Results are cached per
    (zone, hour); the cache is cleared when it grows past
    _HOUR_OFFSET_CACHE_MAX entries.  The computer's zone is keyed with its
    current names and offsets, so a change of TZ (time.tzset) is seen.
    """
    if zone is None:
        key_zone = ("local", tuple(time.tzname), time.timezone, time.altzone)
    else:
        key_zone = zone
    try:
        hash(key_zone)
    except TypeError:                         # an unhashable tzinfo: no cache
        key_zone = None
    out = np.empty(len(hours), dtype=float)
    if len(_HOUR_OFFSET_CACHE) > _HOUR_OFFSET_CACHE_MAX:
        _HOUR_OFFSET_CACHE.clear()
    for i, h in enumerate(hours.tolist()):
        key = (key_zone, h) if key_zone is not None else None
        value = _HOUR_OFFSET_CACHE.get(key) if key is not None else None
        if value is None:
            when = _dt.datetime.fromtimestamp(h * 3600.0 + 1800.0, _dt.timezone.utc)
            local = when.astimezone() if zone is None else when.astimezone(zone)
            value = local.utcoffset().total_seconds()
            if key is not None:
                _HOUR_OFFSET_CACHE[key] = value
        out[i] = value
    return out


# %% ===========================================================================
#    TEXT AND NAME ESCAPING
# =============================================================================
def vz_text(text: Any) -> str:
    """Escape Veusz text-markup characters so a string prints literally.

    Identical to veusz.utils.textrender.latexEscape [R1]: '_' and '^' would
    start sub/superscripts, braces and brackets group, and a backslash
    becomes {\\backslash} because '\\\\' is a line break in Veusz text.
    """
    s = str(text).replace("\\", "\ue000")
    s = re.sub(r"([_\^\[\]\{\}])", r"\\\1", s)
    return s.replace("\ue000", "{\\backslash}")


def escape_hdf_name(name: str) -> str:
    """Escape a dataset name for use as an HDF5 member name, as Veusz does."""
    return name.replace("`", "`BT").replace("/", "`SL")


# %% ===========================================================================
#    WIDGET NAMES AND PAGE NOTES (0.0.25)
# =============================================================================
#: Longest widget name the helpers make (Veusz has no limit; long names make
#: the widget tree and PDF bookmarks hard to read).
WIDGET_NAME_MAX = 96


def widget_name(text: Any, prefix: str = "", fallback: str = "widget") -> str:
    """Any text -> a valid Veusz widget name (0.0.25).

    Veusz accepts a widget name without '/' and without surrounding blanks
    that is not '.' or '..' [R1: utils.validateWidgetName]. '/' becomes '_',
    runs of blanks become one '_', and the result is cut to WIDGET_NAME_MAX.
    """
    name = re.sub(r"\s+", "_", str(text).strip()).replace("/", "_").replace("`", "_")
    name = (prefix + name)[:WIDGET_NAME_MAX].strip("_ ") if name else ""
    return name if name and name not in (".", "..") else fallback


def unique_widget_name(name: str, used: set) -> str:
    """``name``, or ``name_2``, ``name_3`` ... when already in ``used`` (added)."""
    candidate, number = name, 2
    while candidate in used:
        candidate = f"{name}_{number}"
        number += 1
    used.add(candidate)
    return candidate


def header_notes(header: Any = None, source: str = "", title: str = "",
                 extra: Sequence[str] = ()) -> str:
    """Page notes text from a source-file header (0.0.25).

    Parameters
    ----------
    header:
        A mapping (written "KEY = value", one card per line, as Veusz shows
        FITS headers), a sequence of header lines, a string, or None.
        Mapping values that are (value, comment) or (value, unit) pairs are
        written "KEY = value / comment".
    source:
        File the header came from (first line "Source: <file>").
    title:
        Optional first line (e.g. the plot title of the run).
    extra:
        Lines appended after the header (processing notes).
    """
    lines: list[str] = []
    if title:
        lines.append(str(title))
    if source:
        lines.append(f"Source: {source}")
    if lines and header:
        lines.append("")
    if isinstance(header, Mapping):
        for key, value in header.items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                main, comment = value
                lines.append(f"{key} = {main}" + (f" / {comment}" if comment else ""))
            else:
                lines.append(f"{key} = {value}")
    elif isinstance(header, str):
        lines.extend(header.splitlines())
    elif header:
        lines.extend(str(line).rstrip("\r\n") for line in header)
    if extra:
        lines.append("")
        lines.extend(str(line) for line in extra)
    return "\n".join(lines)


def set_notes(doc: "VszDoc", notes: Any, path: str = "notes") -> None:
    """Set the notes of the current widget (a page) when ``notes`` is not empty.

    ``notes`` may be text or a sequence of lines (see header_notes()).
    """
    if notes is None or notes == "" or (not isinstance(notes, str) and not list(notes)):
        return
    text = notes if isinstance(notes, str) else "\n".join(str(line) for line in notes)
    doc.set(path, text)


# %% ===========================================================================
#    DOCUMENT SCRIPT BUILDER
# =============================================================================
class VszDoc:
    """Builder for the Veusz command script stored in the document.

    Same calls as isp_speed_monitor's VszDoc: add() creates a widget under the
    current one without moving, to() moves into a widget, up() moves back,
    and set() takes a path relative to the current widget ("x/label").
    Values are written with repr(), which is valid Python and therefore what
    Veusz executes.
    """

    def __init__(self, generator: str = "", veusz_version: str = VSZH5_LAYOUT_VERSION) -> None:
        """Start a document script with the Veusz header comment, the generator
        name (if given) and the UTC save time.
        """
        stamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.lines: list[str] = [f"# Veusz saved document (version {veusz_version})"]
        if generator:
            self.lines.append(f"# Generated by {generator}")
        self.lines += [f"# Saved at {stamp}", ""]
        self.depth = 0

    def add(self, wtype: str, name: str, **kw: Any) -> None:
        """Add a widget of type ``wtype`` called ``name`` (no auto children)."""
        extra = "".join(f", {k}={v!r}" for k, v in kw.items())
        self.lines.append(f"Add({wtype!r}, name={name!r}, autoadd=False{extra})")

    def to(self, name: str) -> None:
        """Make ``name`` the current widget ('/' = root, '..' = parent)."""
        self.lines.append(f"To({name!r})")
        if name == "/":
            self.depth = 0
        elif name == "..":
            self.depth -= 1
        else:
            self.depth += 1

    def up(self) -> None:
        """Move to the parent widget."""
        self.to("..")

    def set(self, path: str, value: Any) -> None:
        """Set a setting, path relative to the current widget."""
        self.lines.append(f"Set({path!r}, {value!r})")

    def comment(self, text: str) -> None:
        """Add a comment line to the script (ignored by Veusz)."""
        for line in str(text).splitlines() or [""]:
            self.lines.append(f"# {line}")

    def text(self) -> str:
        """The complete script."""
        if self.depth < 0:
            raise ValueError("document script moves above the root widget")
        return "\n".join(self.lines) + "\n"


# %% ===========================================================================
#    HDF5 WRITER
# =============================================================================
class VeuszWriter:
    """Write Veusz datasets and a document script to a .vszh5 file.

    Parameters
    ----------
    path:
        Final file name. Data go to a private sibling .part until finish().
    compress:
        gzip (level 4) + shuffle for arrays of COMPRESS_MIN_ELEMENTS or more.
        Veusz reads compressed datasets transparently.
    atomic:
        Write via .part and os.replace (default). False writes in place.
    resizable:
        (0.0.18) Write 1-D and date arrays (data and error arrays) chunked
        and resizable, uncompressed, fill value NaN, so VeuszAppender can
        add rows later. ``compress`` is ignored for those arrays (see
        APPENDING IN PLACE in the module docstring); 2-D and text datasets
        are written as before.
    """

    def __init__(self, path: str | os.PathLike, compress: bool = False,
                 atomic: bool = True, veusz_version: str = VSZH5_LAYOUT_VERSION,
                 writer_tag: str = "", resizable: bool = False) -> None:
        """Open a new .vszh5 file for writing (h5py required).

        With ``atomic`` (default) everything goes to a private ``<name>.*.part``
        file in the same folder, renamed over ``path`` only by finish(), so a
        crash never leaves a half-written document; an existing file's
        permissions are kept.  ``compress`` gzips large arrays; ``resizable``
        uses the appendable layout of VeuszAppender for 1-D and date data;
        ``writer_tag`` is stored as an attribute Veusz ignores.
        """
        if h5py is None:
            raise RuntimeError("h5py is required to write .vszh5 files "
                               "(mamba install -c conda-forge h5py)")
        self.final_path = Path(path)
        # 0.0.20: the mode the published file gets (see finish()).
        self._final_mode = (self.final_path.stat().st_mode & 0o7777
                            if self.final_path.exists() else None)
        if atomic:
            # Private same-directory temporary file: concurrent writers must
            # not unlink one another's unfinished documents.
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=self.final_path.name + ".", suffix=".part", dir=str(self.final_path.parent))
            os.close(descriptor)
            self.path = Path(temporary_name)
        else:
            self.path = self.final_path
        self.compress = compress
        self.resizable = bool(resizable)        # 0.0.18: appendable 1-D/date arrays
        self.f = h5py.File(self.path, "w")
        self.root = self.f.create_group("Veusz")
        self.root.attrs["vsz_version"] = veusz_version
        self.root.attrs["vsz_saved_at"] = (
            _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat())
        self.root.attrs["vsz_format"] = VSZH5_FORMAT
        if writer_tag:
            self.root.attrs["writer"] = writer_tag      # ignored by Veusz
        self.data = self.root.create_group("Data")
        self.docgrp = self.root.create_group("Document")
        self.tags: dict[str, list[str]] = {}
        self.names: list[str] = []
        # 0.0.28: set beside the ordered list - "name in list" was O(n) per
        # dataset (O(n^2) per file: 9 s of a 40 000-dataset save).
        self._name_set: set = set()
        self._finished = False

    # --- context manager ------------------------------------------------------
    def __enter__(self) -> "VeuszWriter":
        """Context manager: return the writer."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Abort (delete the temporary file) on an exception or when finish()
        was never called.
        """
        if exc_type is not None or not self._finished:
            self.abort()

    # --- internals --------------------------------------------------------------
    def _group(self, name: str, datatype: str) -> Any:
        """Create the HDF5 group of dataset ``name`` (escaped) with its Veusz
        datatype; a name written twice raises ValueError.
        """
        if name in self._name_set:
            raise ValueError(f"dataset {name!r} written twice")
        self.names.append(name)
        self._name_set.add(name)
        g = self.data.create_group(escape_hdf_name(name))
        g.attrs["vsz_datatype"] = datatype
        return g

    def _array(self, group: Any, key: str, values: np.ndarray, vsz_name: str) -> Any:
        """Store ``values`` as member ``key`` of ``group`` (resizable or gzip
        compressed when configured) and record the Veusz name.
        """
        if self.resizable and values.ndim == 1 and group.attrs.get("vsz_datatype") in ("1d", "date"):
            # 0.0.18: appendable layout shared with VeuszAppender.
            ds = _create_resizable_1d(group, key, values)
            ds.attrs["vsz_name"] = vsz_name.encode("utf-8")
            return ds
        if self.compress and values.size >= COMPRESS_MIN_ELEMENTS:
            ds = group.create_dataset(key, data=values, compression="gzip",
                                      compression_opts=4, shuffle=True)
        else:
            # 0.0.28: create_dataset returns the dataset; "group[key] = values;
            # group[key]" made the same contiguous dataset with one more HDF5
            # lookup per dataset (about 25 % of the object-creation time).
            ds = group.create_dataset(key, data=values)
        ds.attrs["vsz_name"] = vsz_name.encode("utf-8")
        return ds

    def _tag(self, name: str, tags: Iterable[str]) -> None:
        """Remember that dataset ``name`` carries each of ``tags``."""
        for t in tags:
            self.tags.setdefault(t, []).append(name)

    # --- datasets -----------------------------------------------------------------
    def dataset_1d(self, name: str, data: Any, serr: Any = None, tags: Sequence[str] = (),
                   perr: Any = None, nerr: Any = None) -> None:
        """Numeric dataset (float64), with optional symmetric or +/- errors."""
        values = np.ascontiguousarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"dataset {name!r} must be 1-D, got shape {values.shape}")
        # 0.0.23: check every error array before the group is created, so a
        # wrong length leaves nothing behind and the name can be retried.
        errors = []
        for key, arr, suffix in (("serr", serr, " (+-)"), ("perr", perr, " (+)"),
                                 ("nerr", nerr, " (-)")):
            if arr is not None:
                err = np.ascontiguousarray(arr, dtype=float)
                if err.shape != values.shape:
                    raise ValueError(f"{name!r} {key} shape {err.shape} != {values.shape}")
                errors.append((key, err, suffix))
        g = self._group(name, "1d")
        self._array(g, "data", values, name)
        for key, err, suffix in errors:
            self._array(g, key, err, name + suffix)
        self._tag(name, tags)

    def dataset_date(self, name: str, veusz_seconds: Any, tags: Sequence[str] = ()) -> None:
        """Date/time dataset in Veusz date seconds (see epoch_to_veusz)."""
        values = np.ascontiguousarray(veusz_seconds, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"date dataset {name!r} must be 1-D, got shape {values.shape}")
        g = self._group(name, "date")
        ds = self._array(g, "data", values, name)
        ds.attrs["vsz_convert_datetime"] = 1
        self._tag(name, tags)

    def dataset_text(self, name: str, strings: Iterable[Any], tags: Sequence[str] = ()) -> None:
        """Text dataset, one string per element."""
        # 0.0.23: encode first, so input that is not a sequence of strings
        # raises before the group exists (no half-made dataset, name free).
        encoded = [str(s).encode("utf-8") for s in strings]
        g = self._group(name, "text")
        # 0.0.28: the same variable-length bytes dataset as "g['data'] =
        # encoded", without h5py guessing the type element by element (and
        # without the second lookup).
        if encoded:
            ds = g.create_dataset("data", data=np.array(encoded, dtype=object),
                                  dtype=_VLEN_BYTES)
        else:                                  # empty: exactly as before
            ds = g.create_dataset("data", data=encoded)
        ds.attrs["vsz_name"] = name.encode("utf-8")
        self._tag(name, tags)

    def dataset_2d(self, name: str, data: Any, xrange: Any = None, yrange: Any = None,
                   xedge: Any = None, yedge: Any = None, xcent: Any = None,
                   ycent: Any = None, tags: Sequence[str] = ()) -> None:
        """2-D numeric dataset (an image) of shape (ny, nx): rows are y, columns x.

        Each axis is given by at most one of
            ``xrange``  (start, stop) of the outer pixel edges,
            ``xedge``   nx + 1 pixel edges (irregular grids),
            ``xcent``   nx pixel centres (irregular grids);
        and likewise for y. With none given the axis is the pixel index
        0..n, Veusz's default. Regular edges or centres are stored as a
        range, which is what Veusz's own Dataset2D does when it loads them
        [R1]. Every axis array is linked from 'data' by an HDF5 object
        reference attribute vsz_<key>, as Dataset2D.saveDataDumpToHDF5 writes.
        NaN pixels are drawn transparent by Veusz.
        """
        values = np.ascontiguousarray(data, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"dataset {name!r} must be 2-D, got shape {values.shape}")
        n_rows, n_columns = values.shape
        # Check the axes before creating anything, so a bad call leaves no group.
        x_key, x_values = _image_axis(name, "x", n_columns, xrange, xedge, xcent)
        y_key, y_values = _image_axis(name, "y", n_rows, yrange, yedge, ycent)
        group = self._group(name, "2d")
        image = self._array(group, "data", values, name)
        for key, axis_values in ((x_key, x_values), (y_key, y_values)):
            group[key] = axis_values
            image.attrs["vsz_" + key] = group[key].ref
        self._tag(name, tags)

    # --- completion -------------------------------------------------------------------
    def finish(self, document_text: str) -> Path:
        """Write the tags and the script, close, and publish the file."""
        tagsgrp = self.docgrp.create_group("Tags")
        for tag, names in sorted(self.tags.items()):
            tagsgrp[tag] = [n.encode("utf-8") for n in sorted(names)]
        self.docgrp["document"] = [document_text.encode("utf-8")]
        self.f.close()
        if self.path != self.final_path:
            if os.name != "nt":
                # 0.0.20: mkstemp() made the temporary file 0600; publish it
                # with the mode of the file it replaces, or the usual mode.
                try:
                    os.chmod(self.path, self._final_mode if self._final_mode is not None
                             else _default_file_mode())
                except OSError:
                    pass
            os.replace(self.path, self.final_path)
        self._finished = True
        return self.final_path

    def abort(self) -> None:
        """Close and delete the unfinished file (the old final file is kept)."""
        try:
            self.f.close()
        except Exception:  # noqa: BLE001 - already closed or never opened
            pass
        if not self._finished and self.path != self.final_path and self.path.exists():
            self.path.unlink()


def _default_file_mode() -> int:
    """Mode a new file would get from open(): 0o666 minus the process umask.

    Read from /proc/self/status on Linux (no race); elsewhere the umask is
    read by setting and restoring it.
    """
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("Umask:"):
                    return 0o666 & ~int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    mask = os.umask(0o022)
    os.umask(mask)
    return 0o666 & ~mask


def _one_dimensional(values: Any, what: str) -> np.ndarray:
    """``values`` as a contiguous float64 1-D array, or ValueError (0.0.23).

    numpy.ascontiguousarray returns at least one dimension, so a single
    number becomes one row and an empty sequence zero rows; anything with
    two or more dimensions is refused rather than flattened, because Veusz
    reads 1-D and date datasets as one column and refuses other shapes.
    """
    array = np.ascontiguousarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{what} must be 1-D, got shape {array.shape}")
    return array


def _create_resizable_1d(group: Any, key: str, values: Any) -> Any:
    """Create ``group[key]``: a chunked, resizable, uncompressed float64 array.

    The layout every appendable 1-D/date array uses (0.0.18): unlimited
    maxshape, APPEND_CHUNK_ELEMENTS rows per chunk and a NaN fill value, so
    rows that were reserved by a resize but never written read back as NaN.
    ``values`` must already be 1-D (0.0.23: checked here too, never ravelled).
    """
    array = _one_dimensional(values, f"array {key!r}")
    dataset = group.create_dataset(key, shape=(array.size,), maxshape=(None,), dtype="f8",
                                   chunks=(APPEND_CHUNK_ELEMENTS,), fillvalue=np.nan)
    if array.size:
        dataset[:] = array
    return dataset


def _is_regular(values: np.ndarray) -> bool:
    """True when the spacing of ``values`` is constant (relative 1e-6)."""
    if values.size < 3:
        return values.size == 2
    steps = np.diff(values)
    return bool(np.allclose(steps, steps[0], rtol=1e-6, atol=abs(steps[0]) * 1e-9))


def _image_axis(name: str, axis: str, n_pixels: int, value_range: Any, edges: Any,
                centres: Any) -> tuple[str, np.ndarray]:
    """Choose and check the stored form of one image axis (see dataset_2d).

    Returns (key, values) with key '<axis>range', '<axis>edge' or '<axis>cent'.
    """
    given = [k for k, v in (("range", value_range), ("edge", edges), ("cent", centres))
             if v is not None]
    if len(given) > 1:
        raise ValueError(f"{name!r}: give only one of {axis}range/{axis}edge/{axis}cent")
    if value_range is not None:
        bounds = np.asarray(value_range, dtype=float).ravel()
        if bounds.size != 2:
            raise ValueError(f"{name!r}: {axis}range needs (start, stop)")
        return f"{axis}range", bounds
    if edges is not None:
        edge_values = np.asarray(edges, dtype=float).ravel()
        if edge_values.size != n_pixels + 1:
            raise ValueError(f"{name!r}: {axis}edge needs {n_pixels + 1} values, "
                             f"got {edge_values.size}")
        if _is_regular(edge_values):
            return f"{axis}range", edge_values[[0, -1]]
        return f"{axis}edge", edge_values
    if centres is not None:
        centre_values = np.asarray(centres, dtype=float).ravel()
        if centre_values.size != n_pixels:
            raise ValueError(f"{name!r}: {axis}cent needs {n_pixels} values, "
                             f"got {centre_values.size}")
        if n_pixels == 1:                   # one pixel: unit width around the centre
            return f"{axis}range", centre_values[0] + np.array([-0.5, 0.5])
        if _is_regular(centre_values):      # outer edges half a step outside
            half_step = 0.5 * (centre_values[1] - centre_values[0])
            return f"{axis}range", np.array([centre_values[0] - half_step,
                                             centre_values[-1] + half_step])
        return f"{axis}cent", centre_values
    return f"{axis}range", np.array([0.0, float(n_pixels)])


# %% ===========================================================================
#    PLOT-PAGE HELPERS (0.0.11)
# =============================================================================
# One call builds one complete page: a title label, a graph with two axes and
# its plot widgets. Every helper starts and ends at the widget that was
# current when it was called (normally the root "/"), so pages can be added
# one after another. Text is escaped with vz_text() unless escape=False, in
# which case Veusz markup (e.g. "W/m^{2}") is passed through.

#: Default line colours, in order (Tableau 10, as matplotlib; distinguishable
#: for the common colour-vision deficiencies when combined with line styles).
TRACE_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                 "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")


@dataclass(frozen=True)
class Trace:
    """One line or marker series (Veusz 'xy' widget) for xy_page().

    Attributes
    ----------
    x, y:
        Dataset names (a 'date' dataset on x needs xmode="datetime").
    label:
        Legend text; empty = not in the legend.
    colour:
        Line colour; empty = next entry of TRACE_COLOURS.
    width:
        Line width with unit, e.g. "1pt".
    style:
        Veusz line style: solid, dashed, dotted, dash-dot, ...
    marker:
        Veusz marker name ("none", "circle", "square", ...).
    marker_size:
        Marker size with unit.
    line:
        False draws markers only.
    fill:
        (0.0.14) "below" shades from the trace down to the bottom of the
        graph, "above" up to the top, "" (default) no shading.
    fill_colour:
        (0.0.14) Shading colour; empty = the line colour.
    fill_transparency:
        (0.0.14) Shading transparency in percent (0 opaque, 100 invisible).
    name:
        (0.0.25) Veusz widget name; empty = the y dataset name (made a
        valid, unique widget name).
    """
    x: str
    y: str
    label: str = ""
    colour: str = ""
    width: str = "1pt"
    style: str = "solid"
    marker: str = "none"
    marker_size: str = "3pt"
    line: bool = True
    fill: str = ""
    fill_colour: str = ""
    fill_transparency: int = 70
    name: str = ""


def add_title(doc: VszDoc, text: str, name: str = "title", y_position: float = 0.965,
              size: str = "14pt", escape: bool = True) -> None:
    """Centred title label near the top of the current page."""
    doc.add("label", name)
    for path, value in title_settings(text, y_position, size, escape).items():
        doc.set(f"{name}/{path}", value)


def title_settings(text: str, y_position: float = 0.965, size: str = "14pt",
                   escape: bool = True) -> dict[str, Any]:
    """Settings of add_title()'s page title label (0.0.26: shared with the
    widget-tree builders, same values and order as before)."""
    return {"label": vz_text(text) if escape else text, "xPos": [0.5], "yPos": [y_position],
            "alignHorz": "centre", "alignVert": "top", "Text/size": size}


#: Centimetres per unit for the physical Veusz distance units.
_CM_PER_UNIT = {"cm": 1.0, "mm": 0.1, "in": 2.54, "inch": 2.54, '"': 2.54, "pt": 2.54 / 72.0}


def _distance_cm(distance: str) -> float | None:
    """A physical Veusz distance ("15cm", "6in", "20pt") in cm; None otherwise."""
    match = re.fullmatch(r"\s*([0-9.]+)\s*(cm|mm|inch|in|\"|pt)\s*", str(distance))
    return float(match.group(1)) * _CM_PER_UNIT[match.group(2)] if match else None


def _graph_name(graph_name: str, page: str, traces: Sequence[Trace] = ()) -> str:
    """(0.0.25) Explicit name, else "g_<y>" for one trace, else "g_<page>"."""
    if graph_name:
        return widget_name(graph_name)
    if len(traces) == 1:
        return widget_name(traces[0].y, prefix="g_")
    return widget_name(page, prefix="g_")


def _open_page_and_graph(doc: VszDoc, page: str, title: str, width: str, height: str,
                         margins: dict[str, str], escape: bool, graph: str = "graph",
                         notes: Any = "") -> None:
    """Add a page (sized, with notes), its title and a graph; leave the graph current."""
    doc.add("page", page)
    doc.to(page)
    doc.set("width", width)
    doc.set("height", height)
    set_notes(doc, notes)
    if title:
        add_title(doc, title, escape=escape)
    doc.add("graph", graph)
    doc.to(graph)
    for side, value in margins.items():         # leftMargin, rightMargin, ...
        doc.set(side, value)


def _add_axes(doc: VszDoc, xlabel: str, ylabel: str, xlog: bool, ylog: bool,
              xmin: Any, xmax: Any, ymin: Any, ymax: Any, xmode: str, escape: bool,
              x_autorange: str = "next-tick", y_autorange: str = "next-tick") -> None:
    """The x and y axes of the current graph, with labels, scales and limits.

    ``x_autorange``/``y_autorange`` are Veusz autoRange values used when the
    limit is "Auto": "next-tick" (Veusz default), "exact", "+5%", ...
    """
    doc.add("axis", "x")
    doc.add("axis", "y")
    doc.set("y/direction", "vertical")
    doc.set("x/label", vz_text(xlabel) if escape else xlabel)
    doc.set("y/label", vz_text(ylabel) if escape else ylabel)
    doc.set("x/mode", xmode)                     # numeric | datetime | labels
    doc.set("x/log", bool(xlog))
    doc.set("y/log", bool(ylog))
    doc.set("x/autoRange", x_autorange)
    doc.set("y/autoRange", y_autorange)
    for setting_path, value in (("x/min", xmin), ("x/max", xmax),
                                ("y/min", ymin), ("y/max", ymax)):
        doc.set(setting_path, value if value == "Auto" else float(value))


def xy_page(doc: VszDoc, page: str, traces: Sequence[Trace], title: str = "",
            xlabel: str = "", ylabel: str = "", *, xlog: bool = False, ylog: bool = False,
            xmin: Any = "Auto", xmax: Any = "Auto", ymin: Any = "Auto", ymax: Any = "Auto",
            xmode: str = "numeric", x_autorange: str = "next-tick",
            y_autorange: str = "next-tick", key: bool | str = "auto",
            key_position: tuple[str, str] = ("right", "top"),
            width: str = "25cm", height: str = "15cm", left_margin: str = "2.2cm",
            top_margin: str = "1.4cm", escape: bool = True,
            datetime_format: str = "", graph_name: str = "", notes: Any = "") -> str:
    """A page with one graph of line/marker traces; returns the graph name.

    Parameters
    ----------
    doc:
        The script builder; the current widget should be the root ("/").
    page:
        Veusz page name (also the PDF bookmark name with render_pages()).
    traces:
        The series to draw, in drawing order.
    title, xlabel, ylabel:
        Page title and axis labels.
    xlog, ylog:
        Logarithmic axes. Non-positive values are not drawn on a log axis.
    xmin, xmax, ymin, ymax:
        Axis limits, or "Auto".
    xmode:
        "numeric", or "datetime" for a 'date' dataset on x.
    x_autorange, y_autorange:
        How an "Auto" limit is found: "next-tick" (Veusz default, extends
        to the next labelled tick), "exact" (data range, e.g. for time
        axes), or a margin such as "+5%".
    key:
        Legend: True, False, or "auto" (when more than one trace has a label).
    key_position:
        (horizontal, vertical) position of the legend inside the graph.
    width, height, left_margin, top_margin:
        Page size and graph margins (Veusz distances with unit).
    escape:
        Escape the texts with vz_text (False passes Veusz markup through).
    datetime_format:
        (0.0.20) Tick-label format of a date x axis, e.g.
        "%VDY-%VDm-%VDd %VDH:%VDM:%VDS"; empty = Veusz's automatic format.
        Used only with xmode="datetime".
    graph_name:
        (0.0.25) Graph widget name; empty = "g_<y>" for one trace, else
        "g_<page>" ("graph" reproduces 0.0.24).
    notes:
        (0.0.25) Page notes, e.g. header_notes(source-file header).
    """
    graph = _graph_name(graph_name, page, traces)
    _open_page_and_graph(doc, page, title, width, height,
                         {"leftMargin": left_margin, "topMargin": top_margin}, escape,
                         graph=graph, notes=notes)
    _add_axes(doc, xlabel, ylabel, xlog, ylog, xmin, xmax, ymin, ymax, xmode, escape,
              x_autorange, y_autorange)
    if xmode == "datetime" and datetime_format:
        doc.set("x/TickLabels/format", datetime_format)
    _add_traces(doc, traces, escape)
    _add_key(doc, traces, key, key_position)
    doc.up()                                     # graph -> page
    doc.up()                                     # page -> where we started
    return graph


def _add_traces(doc: VszDoc, traces: Sequence[Trace], escape: bool = True) -> list[str]:
    """Add the xy widgets of ``traces`` to the current graph (0.0.25: named
    after Trace.name or the y dataset, unique in the graph); returns the names."""
    used: set = {"x", "y", "key", "title", "panel_title"}
    names = []
    for index, trace in enumerate(traces):
        widget = unique_widget_name(widget_name(trace.name or trace.y, fallback=f"trace{index}"),
                                    used)
        names.append(widget)
        doc.add("xy", widget)
        for path, value in _trace_settings(trace, index, escape).items():
            doc.set(f"{widget}/{path}", value)
    return names


def _trace_settings(trace: Trace, index: int, escape: bool = True) -> dict[str, Any]:
    """Settings of the xy widget that draws ``trace`` (relative paths, in the
    order Veusz receives them); shared by _add_traces and smith_graph_spec
    (0.0.26, same settings as 0.0.25 wrote)."""
    colour = trace.colour or TRACE_COLOURS[index % len(TRACE_COLOURS)]
    settings: dict[str, Any] = {
        "xData": trace.x, "yData": trace.y, "marker": trace.marker,
        "markerSize": trace.marker_size, "PlotLine/color": colour,
        "PlotLine/width": trace.width, "PlotLine/style": trace.style,
        "PlotLine/hide": not trace.line, "MarkerFill/color": colour,
        "MarkerLine/color": colour,
    }
    if trace.fill in ("below", "above"):
        # Veusz xy plotters carry two edge fills, FillBelow (default edge
        # "bottom") and FillAbove (default "top"); both are hidden unless
        # switched on [R1: veusz/widgets/point.py, setting/collections.py].
        group = "FillBelow" if trace.fill == "below" else "FillAbove"
        settings[f"{group}/hide"] = False
        settings[f"{group}/fillto"] = "bottom" if trace.fill == "below" else "top"
        settings[f"{group}/color"] = trace.fill_colour or colour
        settings[f"{group}/transparency"] = int(trace.fill_transparency)
    elif trace.fill:
        raise ValueError(f"Trace.fill must be 'below', 'above' or '', not {trace.fill!r}")
    if trace.label:
        settings["key"] = vz_text(trace.label) if escape else trace.label
    return settings


def _add_key(doc: VszDoc, traces: Sequence[Trace], key: bool | str = "auto",
             key_position: tuple[str, str] = ("right", "top")) -> None:
    """Legend in the current graph: True, False or "auto" (more than one label)."""
    n_labelled = sum(bool(trace.label) for trace in traces)
    if key is True or (key == "auto" and n_labelled > 1):
        doc.add("key", "key")
        doc.set("key/horzPosn", key_position[0])
        doc.set("key/vertPosn", key_position[1])


def image_page(doc: VszDoc, page: str, dataset: str, title: str = "", xlabel: str = "",
               ylabel: str = "", zlabel: str = "", *, colormap: str = "viridis",
               zmin: Any = "Auto", zmax: Any = "Auto", zscaling: str = "linear",
               xlog: bool = False, ylog: bool = False, colorbar: bool = True,
               width: str = "25cm", height: str = "15cm", right_margin: str = "3.2cm",
               top_margin: str = "1.4cm", bottom_margin: str = "1.7cm",
               escape: bool = True, xmode: str = "numeric", ymode: str = "numeric",
               datetime_format: str = "", left_margin: str = "", graph_name: str = "",
               notes: Any = "") -> str:
    """A page with a 2-D dataset drawn as an image, and a colour bar;
    returns the graph name.

    Parameters
    ----------
    dataset:
        Name of a dataset written with VeuszWriter.dataset_2d().
    colormap:
        Veusz colour map name (viridis, magma, heat, grey, ...).
    zmin, zmax:
        Colour-scale limits, or "Auto".
    zscaling:
        "linear", "sqrt", "log" or "squared" (Veusz colorScaling).
    colorbar:
        Draw the colour bar right of the graph. It is placed manually
        (left edge at 1.03 of the graph width, top at the graph top, labels
        on the outside): with the default automatic placement Veusz 4.2.1
        does not draw a colour bar that belongs to a graph. Its height is the
        graph height, height - top_margin - bottom_margin, when those are
        physical distances, else Veusz's automatic height.
    bottom_margin:
        Graph bottom margin (Veusz default 1.7cm), used for the bar height.
    Performance (0.0.20): give the image even axes (xrange/yrange, or even
    edges/centres, which dataset_2d stores as a range). With uneven
    edges/centres Veusz resamples through its non-linear path - ten times
    slower and ten times the memory on a 2000 x 1300 image; use
    regularize_rows() for time axes with jitter or pauses.
    xmode, ymode:
        (0.0.20) "numeric" (default) or "datetime": a date axis, for an
        image whose x or y range was given in Veusz date seconds
        (epoch_to_veusz), e.g. a spectrogram with UTC time on y.
    datetime_format:
        (0.0.20) Tick-label format of the date axis/axes; empty = automatic.
    left_margin:
        (0.0.20) Graph left margin; empty = Veusz default. A date y axis
        needs about 3.6cm for "%VDY-%VDm-%VDd %VDH:%VDM:%VDS" labels.
    graph_name, notes:
        (0.0.25) As xy_page(); the graph defaults to "g_<dataset>", the image
        widget is "img_<dataset>" and the colour bar "cbar_<dataset>".
    Other parameters as xy_page().
    """
    for mode in (xmode, ymode):
        if mode not in ("numeric", "datetime"):
            raise ValueError(f"image axis mode must be numeric or datetime, not {mode!r}")
    margins = {"rightMargin": right_margin, "topMargin": top_margin,
               "bottomMargin": bottom_margin}
    if left_margin:
        margins["leftMargin"] = left_margin
    graph = widget_name(graph_name) if graph_name else widget_name(dataset, prefix="g_")
    image = widget_name(dataset, prefix="img_")
    bar = widget_name(dataset, prefix="cbar_")
    _open_page_and_graph(doc, page, title, width, height, margins, escape, graph=graph,
                         notes=notes)
    # Image axes span the pixel edges exactly (no padding to the next tick).
    _add_axes(doc, xlabel, ylabel, xlog, ylog, "Auto", "Auto", "Auto", "Auto", xmode, escape,
              "exact", "exact")
    if ymode != "numeric":
        doc.set("y/mode", ymode)
    for axis_name, mode in (("x", xmode), ("y", ymode)):
        if mode == "datetime" and datetime_format:
            doc.set(f"{axis_name}/TickLabels/format", datetime_format)
    doc.add("image", image)
    doc.set(f"{image}/data", dataset)
    doc.set(f"{image}/colorMap", colormap)
    doc.set(f"{image}/colorScaling", zscaling)
    doc.set(f"{image}/min", zmin if zmin == "Auto" else float(zmin))
    doc.set(f"{image}/max", zmax if zmax == "Auto" else float(zmax))
    if colorbar:
        doc.add("colorbar", bar)
        doc.set(f"{bar}/widgetName", image)
        doc.set(f"{bar}/label", vz_text(zlabel) if escape else zlabel)
        doc.set(f"{bar}/direction", "vertical")
        # The bar is an axis: "exact" keeps it on the image's colour range
        # (the default "next-tick" pads it with white beyond zmin/zmax).
        doc.set(f"{bar}/autoRange", "exact")
        doc.set(f"{bar}/horzPosn", "manual")
        doc.set(f"{bar}/vertPosn", "manual")
        doc.set(f"{bar}/horzManual", 1.03)
        doc.set(f"{bar}/vertManual", 0.0)
        doc.set(f"{bar}/width", "0.45cm")
        sizes_cm = [_distance_cm(d) for d in (height, top_margin, bottom_margin)]
        if None not in sizes_cm and sizes_cm[0] - sizes_cm[1] - sizes_cm[2] > 1.0:
            graph_height_cm = sizes_cm[0] - sizes_cm[1] - sizes_cm[2]
            doc.set(f"{bar}/height", f"{graph_height_cm:.3f}cm")
        doc.set(f"{bar}/reflect", True)
    doc.up()                                     # graph -> page
    doc.up()                                     # page -> where we started
    return graph


@dataclass(frozen=True)
class Panel:
    """One graph of grid_page(): traces, axes and a small title (0.0.12).

    The fields mean the same as the parameters of xy_page().
    """
    traces: Sequence[Trace]
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    xlog: bool = False
    ylog: bool = False
    xmin: Any = "Auto"
    xmax: Any = "Auto"
    ymin: Any = "Auto"
    ymax: Any = "Auto"
    xmode: str = "numeric"
    x_autorange: str = "next-tick"
    key: Any = "auto"
    datetime_format: str = ""                    # 0.0.24: datetime x tick format
    name: str = ""                               # 0.0.25: graph name ("g_<first y>")


def grid_page(doc: VszDoc, page: str, panels: Sequence[Panel], title: str = "", columns: int = 2,
              width: str = "25cm", height: str = "15cm", escape: bool = True,
              notes: Any = "") -> list[str]:
    """A page with several graphs in a Veusz grid (rows filled left to right).

    Each panel gets its own axes, traces, legend and a title above the plot
    area. Use it for overview pages (e.g. time series, spectrum, PSD and zoom
    of one measurement together). (0.0.25) Graphs are named Panel.name or
    "g_<first y dataset>" (unique in the grid); ``notes`` go to the page
    notes. Returns the graph names.
    """
    rows = max(1, math.ceil(len(panels) / max(columns, 1)))
    doc.add("page", page)
    doc.to(page)
    doc.set("width", width)
    doc.set("height", height)
    set_notes(doc, notes)
    if title:
        add_title(doc, title, escape=escape)
    doc.add("grid", "grid")
    doc.to("grid")
    doc.set("rows", rows)
    doc.set("columns", columns)
    for side, value in (("topMargin", "1.2cm" if title else "0.3cm"), ("bottomMargin", "0.2cm"),
                        ("leftMargin", "0.2cm"), ("rightMargin", "0.3cm"), ("internalMargin", "0cm")):
        doc.set(side, value)
    used: set = set()
    names = []
    for index, panel in enumerate(panels):
        default = (widget_name(panel.traces[0].y, prefix="g_") if panel.traces
                   else f"graph{index}")
        graph = unique_widget_name(widget_name(panel.name) if panel.name else default, used)
        names.append(graph)
        doc.add("graph", graph)
        doc.to(graph)
        for side, value in (("leftMargin", "1.8cm"), ("rightMargin", "0.3cm"),
                            ("topMargin", "0.7cm"), ("bottomMargin", "1.3cm")):
            doc.set(side, value)
        _add_axes(doc, panel.xlabel, panel.ylabel, panel.xlog, panel.ylog, panel.xmin, panel.xmax,
                  panel.ymin, panel.ymax, panel.xmode, escape, panel.x_autorange)
        if panel.xmode == "datetime" and panel.datetime_format:     # 0.0.24
            doc.set("x/TickLabels/format", panel.datetime_format)
        _add_traces(doc, panel.traces, escape)
        _add_key(doc, panel.traces, panel.key)
        if panel.title:                          # small title just above the plot area
            doc.add("label", "panel_title")
            doc.set("panel_title/label", vz_text(panel.title) if escape else panel.title)
            doc.set("panel_title/xPos", [0.5])
            doc.set("panel_title/yPos", [1.02])
            doc.set("panel_title/alignHorz", "centre")
            doc.set("panel_title/alignVert", "bottom")
            doc.set("panel_title/Text/size", "10pt")
        doc.up()                                 # graph -> grid
    doc.up()                                     # grid -> page
    doc.up()                                     # page -> where we started
    return names


# %% LINEAR-POWER AVERAGING (0.0.17) — SHARED BY REPORTS AND FUTURE CALLERS
def linear_power_mean_dbm(levels_dbm: Any, axis: int = 0,
                          chunk_size: int = 128) -> np.ndarray:
    """Equal-weight power mean of finite dBm samples, returned in dBm.

    Averages relative milliwatts, not logarithms. Maximum-referenced scaling
    avoids overflow even outside ordinary RF ranges. Only one chunk of
    relative linear powers is allocated at a time. This supports disk-backed
    input without loading a second complete matrix into memory. Independent
    analyzer channels must be passed separately, never stacked as samples.

    No NaN skipping, weighting, smoothing, interpolation or decimation is
    implicit. Empty/nonfinite inputs fail rather than produce biased reports.
    The reduction axis is removed, matching NumPy reduction conventions.
    """
    input_levels = np.asarray(levels_dbm, dtype=float)
    if input_levels.ndim == 0:
        raise ValueError("Supply at least one sample dimension.")
    if not isinstance(axis, (int, np.integer)) or not -input_levels.ndim <= axis < input_levels.ndim:
        raise ValueError("Invalid reduction axis.")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    sample_levels = np.moveaxis(input_levels, axis, 0)
    if not sample_levels.size:
        raise ValueError("Cannot average empty power samples.")
    sample_count = sample_levels.shape[0]
    maximum_levels = np.full(sample_levels.shape[1:], -np.inf)
    for first_sample in range(0, sample_count, chunk_size):
        sample_block = sample_levels[first_sample:first_sample + chunk_size]
        if not np.all(np.isfinite(sample_block)):
            raise ValueError("Power samples must all be finite dBm values.")
        maximum_levels = np.maximum(maximum_levels, np.max(sample_block, axis=0))
    relative_power_sum = np.zeros_like(maximum_levels)
    for first_sample in range(0, sample_count, chunk_size):
        sample_block = sample_levels[first_sample:first_sample + chunk_size]
        with np.errstate(under="ignore", over="ignore"):
            relative_power_sum += np.sum(
                np.power(10.0, (sample_block - maximum_levels) / 10.0), axis=0)
    return maximum_levels + 10.0 * np.log10(relative_power_sum / sample_count)


# %% FREQUENCY-VERSUS-TIME PAGES (0.0.16)
def frequency_ghz_text(frequency_hz: Any) -> str:
    """Scale decimal Hz to GHz without coarse rounding/name collisions.

    Decimal(str(value)) respects the supplied float's decimal representation;
    fixed-point GHz avoids scientific notation in page names and trace keys.
    """
    value = Decimal(str(frequency_hz))
    if not value.is_finite() or value < 0:
        raise ValueError("Frequency must be finite and nonnegative.")
    result = format(value / Decimal("1000000000"), "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return "0" if result in ("", "-0") else result


def frequency_time_pages(doc: VszDoc, frequency_hz: Sequence[float], time_dataset: str,
                         channel_frequency_datasets: Mapping[str, Sequence[str]], *,
                         xlabel: str = "Time", xmode: str = "datetime",
                         datetime_format: str = "%VDY-%VDm-%VDd %VDH:%VDM:%VDS") -> list[str]:
    """Append one page per frequency and a combined page; preserve all raw data.

    Mapping keys identify independent analyzer channels. Mapping values list
    one raw 1-D dataset per frequency, in input order. Each Freq_N_GHz page
    overlays the analyzer channels at that frequency. Combined_Freq_vs_Time
    uses one panel per analyzer channel, so every panel's legend is exactly
    'N GHz', without merging incompatible hold/detector channels or creating
    ambiguous duplicate labels. One analyzer channel produces one graph.

    No averaging/interpolation/decimation occurs here. All bins are included;
    large sweeps necessarily create many pages and crowded combined legends.
    The caller must avoid naming conflicts with existing document pages.
    """
    frequency_names = [frequency_ghz_text(value) for value in frequency_hz]
    channels = list(channel_frequency_datasets.items())
    if not frequency_names or len(frequency_names) != len(set(frequency_names)):
        raise ValueError("At least one frequency with unique GHz text is required.")
    if not channels or any(len(names) != len(frequency_names) for _, names in channels):
        raise ValueError("Each analyzer channel requires one dataset per frequency.")
    if xmode not in ("numeric", "datetime"):
        raise ValueError("xmode must be numeric or datetime.")
    pages = []
    for frequency_index, ghz_text in enumerate(frequency_names):
        page_name = "Freq_" + ghz_text + "_GHz"
        traces = [Trace(time_dataset, names[frequency_index], channel_name,
                        colour=TRACE_COLOURS[channel_index % len(TRACE_COLOURS)],
                        marker="circle", marker_size="2pt")
                  for channel_index, (channel_name, names) in enumerate(channels)]
        xy_page(doc, page_name, traces, title=ghz_text + " GHz versus time",
                xlabel=xlabel, ylabel="Level (dBm)", xmode=xmode, key=True,
                datetime_format=datetime_format if xmode == "datetime" else "")
        pages.append(page_name)
    panels = []
    styles = ("solid", "dashed", "dotted", "dash-dot")
    for channel_name, datasets in channels:
        traces = [Trace(time_dataset, dataset_name, frequency_names[index] + " GHz",
                        colour=TRACE_COLOURS[index % len(TRACE_COLOURS)],
                        style=styles[(index // len(TRACE_COLOURS)) % len(styles)],
                        marker="circle", marker_size="2pt")
                  for index, dataset_name in enumerate(datasets)]
        panels.append(Panel(traces, title=channel_name, xlabel=xlabel,
                            ylabel="Level (dBm)", xmode=xmode, key=True,
                            datetime_format=datetime_format if xmode == "datetime" else "",
                            name="g_" + str(channel_name)))
    page_name = "Combined_Freq_vs_Time"
    if len(panels) == 1:
        xy_page(doc, page_name, panels[0].traces, title="All frequencies versus time — " + channels[0][0],
                xlabel=xlabel, ylabel="Level (dBm)", xmode=xmode, key=True, width="30cm", height="18cm",
                datetime_format=datetime_format if xmode == "datetime" else "",
                graph_name="g_" + str(channels[0][0]))
    else:
        grid_page(doc, page_name, panels, columns=1, title="All frequencies versus time",
                  width="30cm", height=str(max(18, 10 * len(panels))) + "cm")
    pages.append(page_name)
    return pages


# %% ===========================================================================
#    MJD AND LOCAL ISO-8601 TIME AXES (0.0.24)
# =============================================================================
#: Modified Julian Date of the Unix epoch, 1970-01-01T00:00:00 UTC.
MJD_UNIX_EPOCH = 40587.0
#: Veusz tick format of an ISO-8601 date-time without offset (Veusz %VD codes).
ISO_DATETIME_FORMAT = "%VDY-%VDm-%VDdT%VDH:%VDM:%VDS"
#: (0.0.29) timespec values of iso_local_strings(): seconds (default),
#: milliseconds, microseconds, auto, exact.
ISO_TIMESPECS = ("seconds", "milliseconds", "microseconds", "auto", "exact")
# (0.0.29) Decimal arithmetic of the exact ISO text: 80 digits, far above the
# 17 + 5 digits an MJD or epoch times 86400 can need, so nothing is rounded.
_EXACT = Context(prec=80)
_MJD_UNIX_EPOCH_DECIMAL = Decimal("40587")
_SECONDS_PER_DAY_DECIMAL = Decimal("86400")


def mjd_to_epoch(mjd: Any) -> np.ndarray:
    """UTC Modified Julian Date (days) -> Unix epoch seconds.

    A UTC MJD counts 86400 s per day, as Unix time does, so the conversion is
    exact except inside a leap second (which neither scale can label).
    """
    return (np.asarray(mjd, dtype=float) - MJD_UNIX_EPOCH) * 86400.0


def mjd_to_veusz(mjd: Any, offset_s: Any = 0.0) -> np.ndarray:
    """UTC MJD -> Veusz date seconds; ``offset_s`` shifts to a local clock."""
    return epoch_to_veusz(mjd_to_epoch(mjd), offset_s)


def iso_offset_text(offset_s: float) -> str:
    """UTC offset in seconds -> ISO-8601 "+HH:MM" / "-HH:MM" (e.g. -18000 -> "-05:00")."""
    seconds = int(round(float(offset_s)))
    sign = "-" if seconds < 0 else "+"
    hours, remainder = divmod(abs(seconds), 3600)
    return f"{sign}{hours:02d}:{remainder // 60:02d}"


#: (0.0.25) Environment variable that overrides the computer's time zone for
#: local date-time axes: "UTC", a fixed offset such as "-05:00" or "+0530",
#: or (Python >= 3.9) an IANA name such as "America/New_York".
DISPLAY_TZ_ENV = "VSZH5_DISPLAY_TZ"
_FIXED_OFFSET = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2}):?(\d{2})?$", re.IGNORECASE)


def _zone_from_text(text: str) -> Any:
    """"UTC"/"Z", "+HH:MM", "-5", "UTC-05:00" or an IANA name -> tzinfo."""
    text = text.strip()
    if text.upper() in ("UTC", "Z", "GMT"):
        return _dt.timezone.utc
    match = _FIXED_OFFSET.match(text)
    if match:
        sign = -1 if match.group(1) == "-" else 1
        minutes = int(match.group(2)) * 60 + int(match.group(3) or 0)
        return _dt.timezone(_dt.timedelta(minutes=sign * minutes))
    return iana_zone(text)


def iana_zone(name: str) -> Any:
    """An IANA zone ("America/New_York") as a tzinfo on every Python (0.0.27).

    zoneinfo (Python >= 3.9), else backports.zoneinfo, else dateutil.tz
    (installed with pandas and matplotlib) - so Python 3.8 accepts IANA
    names too. ValueError when none of them knows the name.
    """
    try:
        from zoneinfo import ZoneInfo           # Python >= 3.9
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo     # pip install backports.zoneinfo
        except ImportError:
            ZoneInfo = None
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception as exc:                # unknown name, missing tzdata
            raise ValueError(f"unknown time zone {name!r}: {exc}") from exc
    try:
        from dateutil import tz as _dateutil_tz
    except ImportError as exc:
        raise ValueError(f"time zone {name!r} needs Python >= 3.9, backports.zoneinfo "
                         "or python-dateutil") from exc
    zone = _dateutil_tz.gettz(name)
    if zone is None:
        raise ValueError(f"unknown time zone {name!r}")
    return zone


def display_zone(default: Any = None) -> Any:
    """Zone for local date-time axes (0.0.25): VSZH5_DISPLAY_TZ when set,
    else ``default`` (None = the time zone of this computer)."""
    text = os.environ.get(DISPLAY_TZ_ENV, "").strip()
    if not text:
        return default
    try:
        return _zone_from_text(text)
    except ValueError as exc:                   # 0.0.27: warn, use this computer's zone
        warnings.warn(f"{DISPLAY_TZ_ENV}={text!r} ignored: {exc}", RuntimeWarning, stacklevel=2)
        return default


def _zone_offsets(epoch_s: np.ndarray, zone: Any) -> np.ndarray:
    """Per-sample UTC offset (s) in ``zone`` (None = VSZH5_DISPLAY_TZ, else
    this computer's zone)."""
    if zone is None:
        zone = display_zone()
    if zone is None:
        return local_utc_offsets(epoch_s)
    if isinstance(zone, str):
        zone = _zone_from_text(zone)
    out = np.zeros_like(epoch_s)
    good = np.isfinite(epoch_s)
    hours = np.floor(epoch_s[good] / 3600.0)
    unique, inverse = np.unique(hours, return_inverse=True)
    if unique.size:
        out[good] = _hour_offsets(unique, zone)[inverse]     # 0.0.28: cached
    return out


@dataclass(frozen=True)
class LocalTimeAxis:
    """A local wall-clock date axis for Veusz (0.0.24), see local_time_axis()."""
    veusz_seconds: np.ndarray        # write with VeuszWriter.dataset_date()
    offsets_s: np.ndarray            # UTC offset of every sample, seconds
    datetime_format: str             # x/TickLabels/format for the date axis
    label: str                       # axis label naming the offset(s)
    changes: tuple                   # ((ISO instant, "+HH:MM"), ...) offset starts


def _offset_change_instant(before: float, after: float, offset_after: float, zone: Any) -> float:
    """First whole UTC minute in (before, after] whose offset is ``offset_after``
    (bisection; the zone changes offset once between two samples) (0.0.27)."""
    low = math.floor(before / 60.0) + 1                 # minutes since the Unix epoch
    high = math.floor(after / 60.0)
    if high < low:
        return after
    while low < high:
        middle = (low + high) // 2
        if _zone_offsets(np.array([middle * 60.0]), zone)[0] == offset_after:
            high = middle
        else:
            low = middle + 1
    return low * 60.0


def local_time_axis(epoch_s: Any, zone: Any = None, label: str = "Local time") -> LocalTimeAxis:
    """Veusz date values and an ISO-8601 tick format on a local clock.

    Parameters
    ----------
    epoch_s:
        Unix epoch seconds (UTC), 1-D (see mjd_to_epoch for MJD input).
    zone:
        None = the time zone of the computer running this code (DST-aware,
        local_utc_offsets); or an IANA name such as "America/New_York"
        (zoneinfo, Python >= 3.9) or a tzinfo.
    label:
        Start of the axis label.

    Veusz stores naive date seconds, so the local wall clock is the UTC time
    plus each sample's offset. Veusz tick formats are static: with a single
    offset over the data every tick reads like 2026-11-03T06:31:15-05:00;
    across a DST change the ticks omit the offset and the label lists every
    offset with the instant it begins, so no tick can be misread.
    """
    epoch = _one_dimensional(epoch_s, "epoch_s")
    offsets = _zone_offsets(epoch, zone)
    finite = np.isfinite(epoch)
    distinct = list(dict.fromkeys(offsets[finite].tolist())) or [0.0]
    changes = []
    if finite.any():
        index = np.flatnonzero(finite)
        changed = np.r_[True, np.diff(offsets[index]) != 0]
        for position in np.flatnonzero(changed):
            start = index[position]
            when = float(epoch[start])
            if position > 0:
                # 0.0.27: the instant the offset really changes (a DST
                # transition), to the minute, not the first sample after it.
                when = _offset_change_instant(float(epoch[index[position - 1]]), when,
                                              float(offsets[start]), zone)
            instant = _dt.datetime.fromtimestamp(when + offsets[start], _dt.timezone.utc
                                                 ).replace(tzinfo=None).isoformat(timespec="seconds")
            changes.append((instant + iso_offset_text(offsets[start]), iso_offset_text(offsets[start])))
    if len(distinct) == 1:
        text = iso_offset_text(distinct[0])
        fmt = ISO_DATETIME_FORMAT + text
        axis_label = f"{label} (ISO 8601, UTC{text})"
    else:
        fmt = ISO_DATETIME_FORMAT
        axis_label = f"{label} (ISO 8601; " + "; ".join(
            f"UTC{offset} from {instant}" for instant, offset in changes) + ")"
    return LocalTimeAxis(epoch_to_veusz(epoch, offsets), offsets, fmt, axis_label, tuple(changes))


def veusz_to_epoch(veusz_s: Any) -> np.ndarray:
    """Veusz date seconds (since 2009-01-01, UTC) -> Unix epoch seconds (0.0.25)."""
    return np.asarray(veusz_s, dtype=float) - UNIX_TO_VEUSZ_S


def local_time_from_veusz(veusz_utc: Any, zone: Any = None,
                          label: str = "Local time") -> LocalTimeAxis:
    """local_time_axis() for UTC values already in Veusz date seconds (0.0.25).

    Store the UTC values as the date dataset (VeuszWriter.dataset_date) and
    plot ``.veusz_seconds`` as a second date dataset on a datetime axis
    styled with style_datetime_axis(); NaN stays NaN.
    """
    return local_time_axis(veusz_to_epoch(_one_dimensional(veusz_utc, "veusz_utc")),
                           zone=zone, label=label)


#: (0.0.25) Tick-label rotation (degrees) of an ISO-8601 date-time axis; a
#: 25-character "2026-11-09T08:39:00-05:00" label needs it.
LOCAL_DATETIME_ROTATE = 45


def style_datetime_axis(doc: "VszDoc", axis: str, local_axis: Any = None,
                        datetime_format: str = "", label: str = "",
                        rotate: int = LOCAL_DATETIME_ROTATE,
                        autorange: str = "exact") -> None:
    """Make ``axis`` (path from the current widget, e.g. "x" or
    "g_level/x") a datetime axis (0.0.25).

    With a LocalTimeAxis the tick format is its ISO-8601 24-hour format with
    the UTC offset and the axis label names the offset; ``datetime_format``
    and ``label`` override them.
    """
    fmt = datetime_format or (local_axis.datetime_format if local_axis is not None
                              else ISO_DATETIME_FORMAT)
    doc.set(f"{axis}/mode", "datetime")
    doc.set(f"{axis}/TickLabels/format", fmt)
    doc.set(f"{axis}/TickLabels/rotate", str(int(rotate)))
    doc.set(f"{axis}/autoRange", autorange)
    text = label or (local_axis.label if local_axis is not None else "")
    if text:
        doc.set(f"{axis}/label", vz_text(text))


def iso_local_strings(epoch_s: Any, zone: Any = None, timespec: str = "seconds") -> list[str]:
    """ISO-8601 local date-times with offsets, e.g. "2026-11-03T06:31:15-05:00".

    Parameters
    ----------
    epoch_s:
        Unix epoch seconds (UTC), 1-D; NaN gives "".
    zone:
        As local_time_axis() (None = VSZH5_DISPLAY_TZ, else this computer).
    timespec:
        (0.0.29) "seconds" (default, the text of 0.0.24-0.0.28: the fraction
        of a second is cut off, not rounded), "milliseconds",
        "microseconds" or "auto" (datetime.isoformat; the epoch is first
        rounded to whole microseconds), or "exact": every digit of the
        shortest decimal that gives back the same float epoch (Python
        repr), e.g. "2026-07-12T22:29:06.1234567-04:00" - nothing rounded
        or cut, float(text seconds) recovers the value bit for bit. A whole
        second has no fraction in "exact" text.
    """
    if timespec not in ISO_TIMESPECS:
        raise ValueError(f"timespec must be one of {ISO_TIMESPECS}, not {timespec!r}")
    epoch = _one_dimensional(epoch_s, "epoch_s")
    offsets = _zone_offsets(epoch, zone)
    out = []
    for value, offset in zip(epoch, offsets):
        if not np.isfinite(value):
            out.append("")
            continue
        if timespec == "exact":
            wall = _EXACT.add(Decimal(repr(float(value))), Decimal(repr(float(offset))))
            out.append(_exact_wall_text(wall) + iso_offset_text(offset))
            continue
        stamp = _dt.datetime.fromtimestamp(value + offset, _dt.timezone.utc).replace(tzinfo=None)
        out.append(stamp.isoformat(timespec=timespec) + iso_offset_text(offset))
    return out


def _exact_wall_text(wall: Decimal) -> str:
    """(0.0.29) ISO text (no offset suffix) of an exact wall-clock time in
    Unix-style seconds (Decimal): whole seconds through datetime, then every
    digit of the fraction - nothing rounded, trailing zeros dropped.
    """
    whole = wall.to_integral_value(rounding=ROUND_FLOOR)
    fraction = _EXACT.subtract(wall, whole)         # 0 <= fraction < 1, exact
    stamp = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(whole))
    text = stamp.isoformat(timespec="seconds")
    if fraction:
        digits = format(fraction, "f").partition(".")[2].rstrip("0")
        if digits:
            text += "." + digits
    return text


def mjd_iso_local_strings(mjd: Any, zone: Any = None) -> list[str]:
    """(0.0.29) ISO-8601 local date-times of UTC MJD values, every digit kept.

    The shortest round-trip decimal of each MJD (repr) is turned into Unix
    seconds in decimal arithmetic, (MJD - 40587) x 86400, with no binary
    rounding, so the text states the time the MJD states: MJD 61234.10355
    gives "2026-07-12T22:29:06.72-04:00" (America/New_York), where
    iso_local_strings(mjd_to_epoch(mjd), timespec="exact") gives the digits
    of the float epoch, "...06.7199999...", and the default text cuts it to
    "...06-04:00". The offset is the zone's offset at that instant (zone as
    local_time_axis()). Inverting the formula in Decimal arithmetic gives the
    identical MJD float back. NaN -> "".
    """
    values = _one_dimensional(mjd, "mjd")
    if zone is None:
        zone = display_zone()
    if isinstance(zone, str):
        zone = _zone_from_text(zone)
    seconds, hours = [], []
    for value in values.tolist():
        if not math.isfinite(value):
            seconds.append(None)
            continue
        epoch = _EXACT.multiply(_EXACT.subtract(Decimal(repr(value)), _MJD_UNIX_EPOCH_DECIMAL),
                                _SECONDS_PER_DAY_DECIMAL)
        seconds.append(epoch)
        hours.append(float(_EXACT.divide(epoch, 3600).to_integral_value(rounding=ROUND_FLOOR)))
    offsets = iter(_hour_offsets(np.array(hours, dtype=float), zone).tolist()) if hours else iter(())
    out = []
    for epoch in seconds:
        if epoch is None:
            out.append("")
            continue
        offset = next(offsets)
        out.append(_exact_wall_text(_EXACT.add(epoch, Decimal(repr(float(offset)))))
                   + iso_offset_text(offset))
    return out


def with_x(traces: Sequence[Trace], x: str) -> list[Trace]:
    """The same traces drawn against another x dataset (0.0.24)."""
    return [_replace(trace, x=x) for trace in traces]


def time_pair_pages(doc: VszDoc, page: str, traces: Sequence[Trace], mjd_x: str, local_x: str,
                    local_axis: LocalTimeAxis, title: str = "", ylabel: str = "",
                    mjd_label: str = "MJD (UTC)", **xy_kwargs: Any) -> list[str]:
    """Two xy pages from one set of traces: against MJD and against local time.

    ``mjd_x`` is a numeric dataset of MJD values; ``local_x`` a date dataset
    written from ``local_axis.veusz_seconds``. The trace x names are replaced
    per page. Returns ["<page>_mjd", "<page>_local"]. Extra keyword arguments
    go to xy_page() (xmode/datetime_format are set here).
    """
    for key in ("xmode", "datetime_format", "xlabel"):
        xy_kwargs.pop(key, None)
    xy_page(doc, page + "_mjd", with_x(traces, mjd_x), title=title, xlabel=mjd_label,
            ylabel=ylabel, **xy_kwargs)
    xy_kwargs.setdefault("x_autorange", "exact")
    xy_page(doc, page + "_local", with_x(traces, local_x), title=title, xlabel=local_axis.label,
            ylabel=ylabel, xmode="datetime", datetime_format=local_axis.datetime_format, **xy_kwargs)
    return [page + "_mjd", page + "_local"]


# %% ===========================================================================
#    TIME GAPS: LINE BREAKS AND AXIS BREAKS (0.0.27)
# =============================================================================
# A time series has a GAP between samples i and i+1 when
#
#     dt_i = t[i+1] - t[i]  >  K * median(dt)                      (line break)
#
# where median(dt) is the median of the positive sample steps of THAT trace
# and K a multiplier (default GAP_K = 10). The threshold is a multiple of the
# median step: nothing is divided. The trace is broken there (a NaN row is
# inserted after sample i, so Veusz and pyqtgraph stop the line and restart
# it after the gap), while the time axis stays one continuous axis.
#
# A LARGE gap may also break the axis itself (an axis-broken widget that
# leaves the empty interval out):
#
#     dt  >  n * K * median(dt)                                   (axis break)
#
# with n = 0 meaning "never" (one continuous axis); the default n = 100 cuts
# the axis only at gaps 100 times longer than a line break. On a graph
# with several traces the axis is cut only where NO trace has data for
# longer than n times the largest line-break threshold of its traces, so an
# axis break always falls where every line is already broken.
# A manual threshold in seconds (manual_s > 0) replaces K * median(dt).

#: Default line-break multiplier K (gap when dt > K * median(dt)).
GAP_K = 10.0
#: Default axis-break multiple n (axis cut when dt > n * K * median(dt)); 0 = never.
GAP_AXIS_N = 100.0
#: Fraction of a gap kept visible at each side of an axis break.
GAP_AXIS_MARGIN = 0.02


def median_step(t: Any) -> float:
    """Median of the positive, finite steps of ``t`` (0.0 when there is none).

    ``t`` should be in time order; NaN samples (and the steps next to them)
    are ignored, so a series that already holds NaN gap rows is measured on
    its real samples.
    """
    values = np.asarray(t, dtype=float).ravel()
    steps = np.diff(values)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(np.median(steps)) if steps.size else 0.0


def gap_threshold(t: Any, k: float = GAP_K, manual: float = 0.0) -> float:
    """Line-break threshold in the units of ``t``: ``manual`` when > 0, else
    ``k * median_step(t)`` (0.0 = no threshold, e.g. fewer than 2 samples)."""
    if manual and float(manual) > 0.0:
        return float(manual)
    return float(k) * median_step(t)


def gap_positions(t: Any, threshold: float) -> np.ndarray:
    """Insertion indices of the gap rows: ``i + 1`` for every step
    ``t[i+1] - t[i] > threshold``. Empty when ``threshold`` <= 0 or ``t``
    is not in time order (unsorted data cannot be cut at gaps)."""
    values = np.asarray(t, dtype=float).ravel()
    if values.size < 2 or not threshold or threshold <= 0.0:
        return np.zeros(0, dtype=np.int64)
    steps = np.diff(values)
    finite = np.isfinite(steps)
    if np.any(steps[finite] < 0):
        return np.zeros(0, dtype=np.int64)
    return (np.flatnonzero(finite & (steps > threshold)) + 1).astype(np.int64)


def insert_gaps(values: Any, positions: Any) -> np.ndarray:
    """``values`` (float) with a NaN inserted before each of ``positions``
    (from gap_positions); the input is not changed."""
    array = np.asarray(values, dtype=float).ravel()
    where = np.asarray(positions, dtype=np.int64).ravel()
    return array.copy() if where.size == 0 else np.insert(array, where, np.nan)


def axis_break_pairs(t: Any, threshold: float,
                     margin: float = GAP_AXIS_MARGIN) -> list[tuple[float, float]]:
    """(start, end) intervals to leave out of a broken axis.

    ``t`` holds the x values of every trace on the axis (any order, NaN
    allowed); an interval is every step of the sorted values longer than
    ``threshold`` (> 0), shrunk by ``margin`` of its own length at each end
    so the samples beside the gap stay visible. Units are those of ``t``.
    0.0.27: the shrink is relative to the gap; an absolute 1e-9 x |t| (the
    FitsAutoPlot rule before) is about 5 s at MJD 61000, which swallowed
    short gaps on MJD axes.
    """
    if not threshold or threshold <= 0.0:
        return []
    values = np.unique(np.asarray(t, dtype=float).ravel())
    values = values[np.isfinite(values)]
    if values.size < 2:
        return []
    steps = np.diff(values)
    pairs = []
    for index in np.flatnonzero(steps > threshold):
        start, stop = float(values[index]), float(values[index + 1])
        pad = float(margin) * (stop - start)
        pairs.append((start + pad, stop - pad))
    return pairs


# %% ===========================================================================
#    SMITH CHARTS (0.0.26)
# =============================================================================
# Veusz has no Smith-chart widget, so the chart is built from what it does
# have: a graph whose aspect ratio is fixed at 1 and whose (hidden) axes are
# Re(Gamma) and Im(Gamma), the grid as xy lines read from datasets (circles
# and arcs joined with NaN breaks, so one xy widget draws a whole family),
# the grid values as point labels of marker-less xy widgets, and the data as
# ordinary xy traces of Re(Gamma) against Im(Gamma). Everything is a normal
# Veusz widget, so the chart can be edited, restyled and exported in Veusz.
#
# Grid geometry. With the normalized impedance z = Z / Z0 = r + jx,
#     Gamma = (z - 1) / (z + 1).
# Holding r constant and eliminating x gives the circle
#     (Re Gamma - r / (1 + r))^2 + (Im Gamma)^2 = (1 / (1 + r))^2,
# and holding x constant gives the circle
#     (Re Gamma - 1)^2 + (Im Gamma - 1 / x)^2 = (1 / x)^2,
# of which only the part inside |Gamma| <= 1 (r >= 0) is drawn; it is traced
# exactly as Gamma(r) = (r + jx - 1) / (r + jx + 1) for r = tan(theta),
# 0 <= theta < pi/2, so every arc starts on the unit circle (r = 0) and ends
# at Gamma = 1 (r -> infinity). An arc meets the unit circle at angle
# 2 atan(1 / x). For the admittance chart, y = Y / Y0 = g + jb and
# Gamma = (1 - y) / (1 + y) = -(y - 1) / (y + 1): the same grid rotated by
# 180 degrees, as scikit-rf draws it. A VSWR circle S has radius
# |Gamma| = (S - 1) / (S + 1).

#: Normalized resistance (conductance) circles and reactance (susceptance)
#: arcs drawn by default - the values of scikit-rf's and the Plots tab's chart.
SMITH_RESISTANCES = (0.2, 0.5, 1.0, 2.0, 5.0)
SMITH_REACTANCES = (0.2, 0.5, 1.0, 2.0, 5.0)
#: VSWR circles of scikit-rf's draw_vswr=True.
SMITH_VSWR = (1.5, 2.0, 3.0, 5.0)
#: Grid colours: minor lines, the unit circle / real axis, and grid labels.
SMITH_GRID_COLOUR = "#a6a6a6"
SMITH_FRAME_COLOUR = "#4d4d4d"
SMITH_LABEL_COLOUR = "#595959"
#: Half-width of the Re/Im Gamma axes: room outside |Gamma| = 1 for labels.
SMITH_LIMIT = 1.14
#: First words of the notes of every Smith graph (how readers recognise one).
SMITH_NOTES_PREFIX = "Smith chart"
#: Suffixes of the datasets of a Smith trace: Re, Im and (optional) frequency.
SMITH_RE_SUFFIX, SMITH_IM_SUFFIX, SMITH_FREQUENCY_SUFFIX = "_re", "_im", "_freq_Hz"


def gamma_from_impedance(impedance: Any, z0: Any = 50.0) -> np.ndarray:
    """Reflection coefficient Gamma = (Z - Z0) / (Z + Z0) (complex array)."""
    z = np.asarray(impedance, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (z - z0) / (z + z0)


def impedance_from_gamma(gamma: Any, z0: Any = 50.0) -> np.ndarray:
    """Impedance Z = Z0 (1 + Gamma) / (1 - Gamma); Gamma = 1 gives inf."""
    g = np.asarray(gamma, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return z0 * (1.0 + g) / (1.0 - g)


def vswr_from_gamma(gamma: Any) -> np.ndarray:
    """VSWR = (1 + |Gamma|) / (1 - |Gamma|); inf for |Gamma| >= 1."""
    rho = np.abs(np.asarray(gamma, dtype=complex))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rho < 1.0, (1.0 + rho) / (1.0 - rho), np.inf)


def _joined(parts: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (x, y) curves with a NaN point between them (Veusz breaks
    an xy line at NaN, nanHandling 'break-on', its default)."""
    xs: list = []
    ys: list = []
    for x_values, y_values in parts:
        if xs:
            xs.append(np.array([np.nan]))
            ys.append(np.array([np.nan]))
        xs.append(np.asarray(x_values, dtype=float))
        ys.append(np.asarray(y_values, dtype=float))
    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(xs), np.concatenate(ys)


@dataclass(frozen=True)
class SmithGrid:
    """Datasets and widget specifications of a Smith-chart grid (0.0.26).

    Build with smith_grid(); write the datasets with write() (VeuszWriter or
    an object with SetData/SetDataText, e.g. _autoplot_native.NativeDocument
    or veusz.embed), then draw the chart with smith_page() or
    smith_graph_spec().
    """
    prefix: str                       # dataset name prefix, e.g. "smith_z"
    kind: str                         # "z" (impedance) or "y" (admittance)
    numeric: Mapping[str, np.ndarray]  # 1-D datasets (grid lines, label positions)
    text: Mapping[str, list]          # text datasets (grid labels)
    widgets: tuple                    # ((xy widget name, {setting path: value}), ...)

    @property
    def notes(self) -> str:
        """Notes text that marks a graph as this Smith chart."""
        return "%s (%s)" % (SMITH_NOTES_PREFIX, "admittance" if self.kind == "y" else "impedance")

    def dataset_names(self) -> list[str]:
        """Every dataset the grid needs."""
        return list(self.numeric) + list(self.text)

    def write(self, target: Any) -> None:
        """Write the grid datasets with a VeuszWriter (dataset_1d / dataset_text)
        or a document with SetData / SetDataText (NativeDocument, veusz.embed)."""
        for name, values in self.numeric.items():
            if hasattr(target, "dataset_1d"):
                target.dataset_1d(name, values)
            else:
                target.SetData(name, values)
        for name, values in self.text.items():
            if hasattr(target, "dataset_text"):
                target.dataset_text(name, values)
            else:
                target.SetDataText(name, values)


def _grid_line_settings(x_name: str, y_name: str, colour: str, width: str,
                        style: str = "solid") -> dict[str, Any]:
    """xy settings of a grid line family: no markers, no legend entry."""
    return {"xData": x_name, "yData": y_name, "marker": "none", "PlotLine/color": colour,
            "PlotLine/width": width, "PlotLine/style": style}


def _grid_label_settings(x_name: str, y_name: str, text_name: str, colour: str, size: str,
                         horizontal: str, vertical: str) -> dict[str, Any]:
    """xy settings that print a text dataset at points (no line, no marker)."""
    return {"xData": x_name, "yData": y_name, "labels": text_name, "marker": "none",
            "PlotLine/hide": True, "Label/size": size, "Label/color": colour,
            "Label/posnHorz": horizontal, "Label/posnVert": vertical}


def smith_grid(prefix: str = "", kind: str = "z", *,
               resistances: Sequence[float] = SMITH_RESISTANCES,
               reactances: Sequence[float] = SMITH_REACTANCES,
               vswr: Sequence[float] = (), labels: bool = True, points: int = 721,
               colour: str = SMITH_GRID_COLOUR, frame_colour: str = SMITH_FRAME_COLOUR,
               label_colour: str = SMITH_LABEL_COLOUR, label_size: str = "7pt",
               escape: bool = True) -> SmithGrid:
    """The grid of a Smith chart as Veusz datasets and xy widget settings.

    Parameters
    ----------
    prefix:
        Dataset name prefix; empty = "smith_<kind>". One grid can serve every
        Smith page of a document (write it once).
    kind:
        "z" impedance chart (resistance circles, reactance arcs) or "y"
        admittance chart (the same grid rotated 180 degrees).
    resistances, reactances:
        Normalized values of the circles and of the arcs (each arc drawn for
        +x and -x). All must be positive.
    vswr:
        VSWR circles to draw, e.g. SMITH_VSWR; empty = none.
    labels:
        Print the circle and arc values (and the VSWR values).
    points:
        Points per full circle (arcs get half as many).
    colour, frame_colour, label_colour, label_size:
        Grid lines, unit circle and real axis, label text.
    escape:
        Escape the label text with vz_text.

    Returns
    -------
    SmithGrid
    """
    if kind not in ("z", "y"):
        raise ValueError(f"kind must be 'z' or 'y', not {kind!r}")
    values = [float(v) for v in tuple(resistances) + tuple(reactances)]
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError("Smith grid resistances and reactances must be finite and > 0")
    if any(not math.isfinite(float(s)) or float(s) <= 1.0 for s in vswr):
        raise ValueError("VSWR circles must be finite and > 1")
    prefix = prefix or f"smith_{kind}"
    turn = -1.0 if kind == "y" else 1.0                 # admittance: rotate 180 degrees
    n_points = max(int(points), 16)
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    numeric: dict[str, np.ndarray] = {}
    text: dict[str, list] = {}
    widgets: list = []

    def add_lines(part: str, curves: Sequence[tuple[np.ndarray, np.ndarray]], line_colour: str,
                  width: str, style: str = "solid") -> None:
        """Add one family of grid curves as joined x/y datasets (NaN between
        curves, rotated for an admittance chart) and its xy line widget.
        """
        x_values, y_values = _joined(curves)
        x_name, y_name = f"{prefix}_{part}_x", f"{prefix}_{part}_y"
        numeric[x_name], numeric[y_name] = turn * x_values, turn * y_values
        widgets.append((f"grid_{part}", _grid_line_settings(x_name, y_name, line_colour,
                                                            width, style)))

    def add_labels(part: str, positions: Sequence[tuple[float, float]], strings: Sequence[str],
                   horizontal: str, vertical: str) -> None:
        """Add grid value labels for one family (nothing when labels are off or
        there are no strings).
        """
        if not labels or not strings:
            return
        x_name, y_name, t_name = (f"{prefix}_{part}_lx", f"{prefix}_{part}_ly",
                                  f"{prefix}_{part}_text")
        numeric[x_name] = turn * np.array([p[0] for p in positions], dtype=float)
        numeric[y_name] = turn * np.array([p[1] for p in positions], dtype=float)
        text[t_name] = [vz_text(s) if escape else str(s) for s in strings]
        widgets.append((f"grid_{part}_labels", _grid_label_settings(
            x_name, y_name, t_name, label_colour, label_size, horizontal, vertical)))

    # Resistance circles: centre r / (1 + r), radius 1 / (1 + r).
    circles = []
    for r in resistances:
        centre, radius = r / (1.0 + r), 1.0 / (1.0 + r)
        circles.append((centre + radius * np.cos(angle), radius * np.sin(angle)))
    add_lines("r", circles, colour, "0.5pt")
    # Reactance arcs, traced exactly: Gamma(r) for r = tan(theta), then Gamma = 1.
    theta = np.linspace(0.0, 0.5 * np.pi, max(n_points // 2, 8), endpoint=False)
    resistance = np.tan(theta)
    arcs = []
    for x in reactances:
        for signed in (x, -x):
            z = resistance + 1j * signed
            gamma = np.append((z - 1.0) / (z + 1.0), 1.0 + 0.0j)
            arcs.append((gamma.real, gamma.imag))
    add_lines("x", arcs, colour, "0.5pt")
    if vswr:
        add_lines("vswr", [(rho * np.cos(angle), rho * np.sin(angle))
                           for rho in ((float(s) - 1.0) / (float(s) + 1.0) for s in vswr)],
                  colour, "0.5pt", "dashed")
    # Frame: the unit circle (r = 0, |Gamma| = 1) and the real axis (x = 0).
    add_lines("unit", [(np.cos(angle), np.sin(angle))], frame_colour, "1pt")
    add_lines("real", [(np.array([-1.0, 1.0]), np.array([0.0, 0.0]))], frame_colour, "0.75pt")
    # Labels: r on the real axis at the circle's left edge (r - 1) / (r + 1);
    # x just outside the unit circle where the arc meets it, 2 atan(1 / x).
    add_labels("r", [((r - 1.0) / (r + 1.0), 0.0) for r in resistances],
               ["%g" % r for r in resistances], "right" if kind == "z" else "left", "top")
    x_positions, x_strings = [], []
    for x in reactances:
        for signed in (x, -x):
            phi = 2.0 * math.atan(1.0 / signed)
            x_positions.append((1.07 * math.cos(phi), 1.07 * math.sin(phi)))
            x_strings.append("%+gj" % signed)
    add_labels("x", x_positions, x_strings, "centre", "centre")
    if vswr:
        add_labels("vswr", [(0.0, -(float(s) - 1.0) / (float(s) + 1.0)) for s in vswr],
                   ["VSWR %g" % float(s) for s in vswr], "centre", "bottom")
    return SmithGrid(prefix, kind, numeric, text, tuple(widgets))


def smith_trace_datasets(stem: str, gamma: Any, frequency_hz: Any = None) -> dict[str, np.ndarray]:
    """Datasets of one Smith trace: "<stem>_re", "<stem>_im" and, when given,
    "<stem>_freq_Hz" (read by the suite's interactive Smith plot)."""
    g = np.asarray(gamma, dtype=complex).ravel()
    out = {stem + SMITH_RE_SUFFIX: g.real.copy(), stem + SMITH_IM_SUFFIX: g.imag.copy()}
    if frequency_hz is not None:
        f = np.asarray(frequency_hz, dtype=float).ravel()
        if f.shape != g.shape:
            raise ValueError(f"{stem!r}: {f.size} frequencies for {g.size} Gamma values")
        out[stem + SMITH_FREQUENCY_SUFFIX] = f
    return out


def smith_trace(stem: str, label: str = "", **trace_fields: Any) -> Trace:
    """Trace of Im(Gamma) against Re(Gamma) for the datasets of
    smith_trace_datasets(stem, ...); keyword arguments go to Trace."""
    trace_fields.setdefault("width", "1.2pt")
    return Trace(stem + SMITH_RE_SUFFIX, stem + SMITH_IM_SUFFIX, label, **trace_fields)


def smith_has_key(traces: Sequence[Trace], key: Any = "auto") -> bool:
    """True when a Smith graph gets a legend: key=True, or "auto" with more
    than one labelled trace (as xy_page)."""
    return key is True or (key == "auto" and sum(bool(t.label) for t in traces) > 1)


def smith_graph_spec(grid: SmithGrid, traces: Sequence[Trace], *, key: Any = "auto",
                     key_position: tuple[str, str] = ("outside", "top"),
                     limit: float = SMITH_LIMIT, escape: bool = True, notes: str = "",
                     margins: str = "0.6cm", key_width: str = "6.5cm"
                     ) -> tuple[dict[str, Any], list[tuple[str, str, dict]]]:
    """Settings of a Smith graph and its children, for any Veusz builder.

    Returns (graph_settings, children): graph_settings maps setting paths of
    the graph (aspect ratio 1, margins, notes, no border) to values;
    children is a list of (widget type, name, {setting path: value}) in
    drawing order - the two hidden axes, the grid, the traces, the legend.
    smith_page() applies them to a VszDoc script; _autoplot_native applies
    them to a NativeDocument widget tree.

    The legend sits right of the chart by default (key_position[0] =
    "outside": Veusz manual placement at the right edge of the chart area), in a
    right margin of ``key_width``; ("right", "top") etc. put it inside the
    graph (it then covers part of the grid).
    """
    has_key = smith_has_key(traces, key)
    outside = has_key and key_position[0] == "outside"
    graph_settings: dict[str, Any] = {
        "aspect": 1.0, "leftMargin": margins,
        "rightMargin": key_width if outside else margins, "topMargin": margins,
        "bottomMargin": margins, "Border/hide": True,
        "notes": grid.notes + (("\n" + notes) if notes else ""),
    }
    children: list[tuple[str, str, dict]] = []
    for axis, direction in (("x", "horizontal"), ("y", "vertical")):
        children.append(("axis", axis, {
            "direction": direction, "min": -float(limit), "max": float(limit), "hide": True,
            "label": "Re(Gamma)" if axis == "x" else "Im(Gamma)"}))
    used: set = {"x", "y", "key"}
    for name, settings in grid.widgets:
        used.add(name)
        children.append(("xy", name, dict(settings)))
    for index, trace in enumerate(traces):
        name = unique_widget_name(widget_name(trace.name or trace.y, fallback=f"trace{index}"),
                                  used)
        children.append(("xy", name, _trace_settings(trace, index, escape)))
    if outside:
        children.append(("key", "key", {"horzPosn": "manual", "horzManual": 1.0,
                                         "vertPosn": key_position[1], "Text/size": "10pt"}))
    elif has_key:
        children.append(("key", "key", {"horzPosn": key_position[0],
                                         "vertPosn": key_position[1]}))
    return graph_settings, children


def smith_page_layout(traces: Sequence[Trace], title: str = "", *, size: str = "18cm",
                      key: Any = "auto", key_position: tuple[str, str] = ("outside", "top"),
                      key_width: str = "6.5cm") -> tuple[str, str, str]:
    """(page width, page height, graph top margin) of a Smith page: the chart
    about ``size`` across, 1 cm more height for the title, ``key_width`` more
    width when the legend is outside the chart."""
    size_cm = _distance_cm(size)
    key_cm = _distance_cm(key_width) or 0.0
    outside = smith_has_key(traces, key) and key_position[0] == "outside"
    height = f"{size_cm + 1.0:.3f}cm" if size_cm is not None else size
    width = f"{size_cm + key_cm:.3f}cm" if (size_cm is not None and outside) else size
    return width, height, ("1.2cm" if title else "0.6cm")


def smith_page(doc: VszDoc, page: str, traces: Sequence[Trace], grid: SmithGrid,
               title: str = "", *, graph_name: str = "", notes: Any = "",
               size: str = "18cm", key: Any = "auto",
               key_position: tuple[str, str] = ("outside", "top"), key_width: str = "6.5cm",
               limit: float = SMITH_LIMIT, escape: bool = True) -> str:
    """A page with one Smith chart; returns the graph name (0.0.26).

    ``grid`` comes from smith_grid() and its datasets must be written to the
    same document (grid.write(writer)); ``traces`` from smith_trace().
    The chart is about ``size`` across; the page is ``size`` + 1 cm high
    (title) and ``size`` wide, plus ``key_width`` when the legend is outside
    the chart. The graph keeps an aspect ratio of 1, so circles stay circles
    at any page size. ``notes`` are the page notes (e.g. header_notes(...));
    the graph notes say which chart it is (SmithGrid.notes).
    """
    graph = widget_name(graph_name) if graph_name else widget_name(page, prefix="g_")
    width, height, top_margin = smith_page_layout(traces, title, size=size, key=key,
                                                  key_position=key_position, key_width=key_width)
    _open_page_and_graph(doc, page, title, width, height, {"topMargin": top_margin}, escape,
                         graph=graph, notes=notes)
    graph_settings, children = smith_graph_spec(grid, traces, key=key, key_position=key_position,
                                                limit=limit, escape=escape, key_width=key_width)
    for path, value in graph_settings.items():
        if path != "topMargin":                  # keep room for the page title
            doc.set(path, value)
    for kind, name, settings in children:
        doc.add(kind, name)
        for path, value in settings.items():
            doc.set(f"{name}/{path}", value)
    doc.up()                                     # graph -> page
    doc.up()                                     # page -> where we started
    return graph


#: Veusz text markup for a line break (two backslashes in the label text).
VEUSZ_LINE_BREAK = "\\\\"


def text_page(doc: VszDoc, page: str, title: str, lines: Sequence[str], font: str = "Courier New",
              size: str = "9pt", lines_per_page: int = 40, width: str = "25cm",
              height: str = "15cm", notes: Any = "") -> list[str]:
    """Pages of plain text lines (escaped; monospaced by default) (0.0.12).

    Long texts continue on pages "<page>_2", "<page>_3", ...; returns the
    page names. Lines are escaped with vz_text and joined with Veusz's line
    break, so they print exactly as given. "Courier New" maps to a
    monospaced font on every platform (fontconfig substitutes on Linux).
    """
    pages: list[str] = []
    chunks = [list(lines[i:i + lines_per_page]) for i in range(0, max(len(lines), 1), lines_per_page)] or [[]]
    for number, chunk in enumerate(chunks, start=1):
        name = page if number == 1 else f"{page}_{number}"
        doc.add("page", name)
        doc.to(name)
        doc.set("width", width)
        doc.set("height", height)
        set_notes(doc, notes)                    # 0.0.25
        add_title(doc, title if number == 1 else f"{title} (continued, {number})")
        doc.add("label", "body")
        doc.set("body/label", VEUSZ_LINE_BREAK.join(vz_text(line) if line else " " for line in chunk))
        doc.set("body/xPos", [0.04])
        doc.set("body/yPos", [0.90])
        doc.set("body/alignHorz", "left")
        doc.set("body/alignVert", "top")
        doc.set("body/Text/font", font)
        doc.set("body/Text/size", size)
        doc.up()
        pages.append(name)
    return pages


# %% ===========================================================================
#    BAR PAGES, STEP MASKS AND REFERENCE LINES (0.0.14)
# =============================================================================
def step_mask(lower_edges: Any, upper_edges: Any, levels: Any) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) arrays of a stepped mask: one flat segment per band.

    Each band i is drawn from ``lower_edges[i]`` to ``upper_edges[i]`` at
    ``levels[i]``; a NaN point follows every band so plotting programs
    (Veusz, pyqtgraph with connect="finite") leave a gap between bands
    instead of joining them with a sloping line. Overlapping bands are
    drawn as given (the caller decides which level applies).

    Returns
    -------
    (x, y): float arrays of length 3 * number of bands.
    """
    lower = np.asarray(lower_edges, dtype=float).ravel()
    upper = np.asarray(upper_edges, dtype=float).ravel()
    level = np.asarray(levels, dtype=float).ravel()
    if not lower.size == upper.size == level.size:
        raise ValueError("step_mask: lower_edges, upper_edges and levels differ in length")
    x_values = np.column_stack([lower, upper, np.full(lower.size, np.nan)]).ravel()
    y_values = np.column_stack([level, level, np.full(lower.size, np.nan)]).ravel()
    return x_values, y_values


def reference_line(value: float, span: Sequence[float], orientation: str = "horizontal"
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) arrays of a straight threshold or marker line.

    ``orientation="horizontal"``: y = value for x from span[0] to span[1]
    (e.g. a 0 dB margin line). ``"vertical"``: x = value for y over span
    (e.g. the 1 GHz upper limit of a standard).
    """
    start, stop = float(span[0]), float(span[1])
    if orientation == "horizontal":
        return np.array([start, stop]), np.array([float(value)] * 2)
    if orientation == "vertical":
        return np.array([float(value)] * 2), np.array([start, stop])
    raise ValueError(f"orientation must be 'horizontal' or 'vertical', not {orientation!r}")


def bar_page(doc: VszDoc, page: str, labels: str, lengths: Sequence[str], title: str = "",
             xlabel: str = "", ylabel: str = "", *, keys: Sequence[str] = (),
             colours: Sequence[str] = (), ylog: bool = False, ymin: Any = "Auto",
             ymax: Any = "Auto", width: str = "25cm", height: str = "15cm",
             left_margin: str = "2.4cm", bottom_margin: str = "2.2cm",
             top_margin: str = "1.4cm", escape: bool = True, graph_name: str = "",
             notes: Any = "") -> str:
    """A page with a vertical category bar chart (0.0.14); returns the graph name.

    Parameters
    ----------
    labels:
        Name of a *text* dataset with one category label per bar position;
        the x axis is put in Veusz 'labels' mode to print them.
    lengths:
        Names of numeric datasets, one per bar group member (grouped mode).
    keys:
        Legend text per dataset in ``lengths`` (a legend is added when more
        than one is given).
    colours:
        Fill colour per dataset; default TRACE_COLOURS in order.
    ylog, ymin, ymax:
        Value axis scale and limits ("Auto" or a number). Bars on a log axis
        start at the axis minimum.
    graph_name, notes:
        (0.0.25) Graph name (default "g_<page>"; the bar widget is
        "bars_<first length dataset>") and page notes.
    Other parameters as xy_page().
    """
    graph = widget_name(graph_name) if graph_name else widget_name(page, prefix="g_")
    bars = widget_name(lengths[0] if lengths else page, prefix="bars_")
    _open_page_and_graph(doc, page, title, width, height,
                         {"leftMargin": left_margin, "bottomMargin": bottom_margin,
                          "topMargin": top_margin}, escape, graph=graph, notes=notes)
    _add_axes(doc, xlabel, ylabel, False, ylog, "Auto", "Auto", ymin, ymax, "labels", escape)
    doc.add("bar", bars)
    doc.set(f"{bars}/lengths", tuple(lengths))
    doc.set(f"{bars}/labels", labels)
    doc.set(f"{bars}/direction", "vertical")
    doc.set(f"{bars}/mode", "grouped")
    fills = []
    for index, _name in enumerate(lengths):
        colour = colours[index] if index < len(colours) else TRACE_COLOURS[index % len(TRACE_COLOURS)]
        fills.append(("solid", colour, False))
    doc.set(f"{bars}/BarFill/fills", fills)
    if keys:
        doc.set(f"{bars}/keys", tuple(vz_text(k) if escape else k for k in keys))
        if len(keys) > 1:
            doc.add("key", "key")
            doc.set("key/horzPosn", "right")
            doc.set("key/vertPosn", "top")
    doc.up()                                     # graph -> page
    doc.up()                                     # page -> where we started
    return graph


# %% ===========================================================================
#    DECIMATING IMAGES FOR PLOTTING (0.0.20)
# =============================================================================
#: Methods of block_reduce().
REDUCE_METHODS = ("max", "min", "mean", "power_mean_db", "power_mean_linear")


def _reduce_axis(values: Any, factor: int, axis: int, method: str, xp: Any) -> Any:
    """Reduce ``values`` along ``axis`` in groups of ``factor`` (see block_reduce)."""
    length = values.shape[axis]
    if factor <= 1 or length == 0:
        return values
    full = (length // factor) * factor
    parts = []
    if full:
        head = values[:full] if axis == 0 else values[:, :full]
        shape = ((full // factor, factor, values.shape[1]) if axis == 0
                 else (values.shape[0], full // factor, factor))
        parts.append(_reduce_groups(head.reshape(shape), axis + 1, method, xp))
    if full < length:                       # trailing partial block: what it has
        tail = values[full:] if axis == 0 else values[:, full:]
        # One group holding all the remaining rows (or columns).
        grouped_tail = (tail.reshape((1,) + tail.shape) if axis == 0
                        else tail.reshape((tail.shape[0], 1, tail.shape[1])))
        parts.append(_reduce_groups(grouped_tail, axis + 1, method, xp))
    return parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=axis)


def _reduce_groups(grouped: Any, axis: int, method: str, xp: Any) -> Any:
    """Collapse ``axis`` of a grouped array with ``method``."""
    if method == "max":
        return grouped.max(axis=axis)
    if method == "min":
        return grouped.min(axis=axis)
    if method == "mean":
        return grouped.mean(axis=axis)
    if method == "power_mean_db":
        # 10 log10(mean 10^(x/10)), referenced to the group maximum so that
        # 10^(x/10) never overflows or underflows (as linear_power_mean_dbm).
        peak = grouped.max(axis=axis, keepdims=True)
        relative = xp.power(10.0, (grouped - peak) / 10.0)
        return (xp.squeeze(peak, axis=axis)
                + 10.0 * xp.log10(relative.mean(axis=axis)))
    if method == "power_mean_linear":
        return xp.sqrt((grouped * grouped).mean(axis=axis))
    raise ValueError(f"method must be one of {REDUCE_METHODS}, not {method!r}")


def block_reduce(values: Any, row_factor: int = 1, column_factor: int = 1,
                 method: str = "max", xp: Any = None) -> Any:
    """Decimate a 2-D array in blocks of ``row_factor`` x ``column_factor``.

    For plotting arrays too large for Veusz or the screen, e.g. a waterfall
    of 20 000 sweeps x 100 000 bins. Every output pixel summarises one block
    of input pixels:

        "max"                peak hold - a short burst or a narrow carrier
                             survives the reduction (the usual choice for
                             RFI spectrograms);
        "min"                minimum hold (noise floor);
        "mean"               arithmetic mean of the values - for linear
                             quantities only: averaging dB numbers directly
                             understates the power (0 and -10 dB give -5 dB,
                             the power mean is -2.6 dB); use "power_mean_db"
                             for dB, dBm, dBuV ... values;
        "power_mean_db"      values in dB averaged as power:
                             10 log10( (1/N) sum 10^(x_i/10) );
        "power_mean_linear"  linear amplitudes averaged as power (RMS):
                             sqrt( (1/N) sum x_i^2 ).

    Rows are reduced first, then columns. Every pixel of a block has the
    same weight, so the result equals the method applied to the whole block
    at once (checked against a brute-force loop for every method). A
    trailing partial block is reduced over the rows/columns it has - never
    padded, so no fill value can leak into a pixel. A factor of 1 leaves
    that axis untouched (no abs() for "power_mean_linear"). Non-finite
    inputs propagate (NaN in, NaN out).

    Parameters
    ----------
    values:
        2-D array (rows x columns), NumPy or another array module's.
    row_factor, column_factor:
        Block size; 1 leaves that axis alone.
    method:
        One of REDUCE_METHODS.
    xp:
        Array module of ``values`` (numpy by default; pass cupy for a CuPy
        array so the work stays on the GPU).

    Returns
    -------
    array of shape (ceil(rows / row_factor), ceil(columns / column_factor)),
    float64 for NumPy input.
    """
    xp = np if xp is None else xp
    if method not in REDUCE_METHODS:
        raise ValueError(f"method must be one of {REDUCE_METHODS}, not {method!r}")
    row_factor, column_factor = int(row_factor), int(column_factor)
    if row_factor < 1 or column_factor < 1:
        raise ValueError("row_factor and column_factor must be >= 1")
    array = xp.asarray(values, dtype=xp.float64)
    if array.ndim != 2:
        raise ValueError(f"block_reduce needs a 2-D array, got shape {array.shape}")
    array = _reduce_axis(array, row_factor, 0, method, xp)
    return _reduce_axis(array, column_factor, 1, method, xp)


def regularize_rows(image: Any, centres: Any, max_growth: float = 4.0,
                    tolerance: float = 0.25) -> tuple[np.ndarray, tuple[float, float], str]:
    """Image rows with (possibly uneven) centres -> an even grid for drawing.

    Parameters
    ----------
    image:
        2-D array (rows x columns); row k is centred at ``centres[k]``.
    centres:
        Increasing row centres, e.g. the mean time of each spectrogram row
        in Veusz date seconds.
    max_growth:
        The grid may have at most this many times the rows of ``image``
        (more empty slots than that means the pauses dominate; the rows
        are then spread evenly instead).
    tolerance:
        Gap threshold: an interval longer than (1 + 2 x tolerance) median
        intervals is a gap (default 0.25: 1.5 intervals). Shorter
        intervals are ordinary rows, however much they jitter.

    Returns
    -------
    (image_on_grid, (start, stop), how)
        ``(start, stop)`` are the outer edges for
        VeuszWriter.dataset_2d(yrange=(start, stop)). ``how`` is
        "regular" - the rows already lie on an even grid (image unchanged);
        "gaps"    - rows placed in slots of the mean ordinary step, NaN
                    rows where nothing was recorded (pauses show as blank
                    bands);
        "even"    - the centres cannot be slotted (repeated or crowded time
                    stamps): rows spread evenly between the first and last
                    centre (image unchanged), times are then approximate.
    """
    values = np.asarray(image, dtype=float)
    c = np.asarray(centres, dtype=float).ravel()
    n = c.size
    if n != values.shape[0]:
        raise ValueError(f"{n} centres for {values.shape[0]} rows")
    if n == 0:
        return values, (0.0, 1.0), "regular"
    if n == 1:
        return values, (c[0] - 0.5, c[0] + 0.5), "regular"
    steps = np.diff(c)
    if np.all(np.isfinite(c)) and np.all(steps > 0):
        # Ordinary intervals vs gaps, judged against the median interval.
        # The grid step is the MEAN of the ordinary intervals: for jittered
        # time stamps (e.g. 32..35 ms around a 33.35 ms sweep period) the
        # median is biased, and slots taken from it drift from the rows
        # until a slot is skipped (a false blank row). Each ordinary
        # interval is one slot; a gap gets rint(gap / step) slots.
        median = float(np.median(steps))
        ordinary = steps <= (1.0 + 2.0 * float(tolerance)) * median
        step = float(steps[ordinary].mean())
        increments = np.where(ordinary, 1, np.maximum(1, np.rint(steps / step))).astype(np.int64)
        slots = np.r_[0, np.cumsum(increments)]
        if slots[-1] == n - 1:
            return values, (c[0] - step / 2.0, c[0] + (n - 0.5) * step), "regular"
        if slots[-1] + 1 <= max_growth * n:
            grid = np.full((int(slots[-1]) + 1, values.shape[1]), np.nan)
            grid[slots] = values
            return grid, (c[0] - step / 2.0, c[0] + (slots[-1] + 0.5) * step), "gaps"
    span = c[-1] - c[0]
    pitch = span / (n - 1) if span > 0 else 1.0
    return values, (c[0] - pitch / 2.0, c[0] + (n - 0.5) * pitch), "even"


# %% ===========================================================================
#    READING A WRITTEN DOCUMENT BACK (0.0.13)
# =============================================================================
def dataset_names(path: str | os.PathLike) -> list[str]:
    """Names of the datasets stored in a .vszh5 document (unescaped, sorted).

    Reads /Veusz/Data with h5py only; Veusz is not needed.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to read .vszh5 documents")
    names = []
    with h5py.File(str(path), "r") as hdf:
        data_group = hdf.get("Veusz/Data")
        for member in (data_group or {}):
            stored = data_group[member].get("data")
            raw = stored.attrs.get("vsz_name") if stored is not None else None
            names.append(raw.decode("utf-8") if isinstance(raw, bytes) else
                         member.replace("`SL", "/").replace("`BT", "`"))
    return sorted(names)


def read_text_dataset(path: str | os.PathLike, name: str) -> list[str] | None:
    """The strings of a text dataset written by dataset_text(), or None if absent."""
    if h5py is None:
        raise RuntimeError("h5py is required to read .vszh5 documents")
    with h5py.File(str(path), "r") as hdf:
        group = hdf.get(f"Veusz/Data/{escape_hdf_name(name)}")
        if group is None or "data" not in group:
            return None
        return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in group["data"][()]]


def page_names(path: str | os.PathLike) -> list[str]:
    """Names of the pages of a written document, in page order (0.0.13).

    Read from the Veusz command script stored in /Veusz/Document/document
    (the top-level Add('page', name=...) lines written by VszDoc), so the
    page list is known without Veusz, e.g. to build PDF bookmarks.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to read .vszh5 documents")
    with h5py.File(str(path), "r") as hdf:
        stored = hdf["Veusz/Document/document"][()]
    script = b"".join(stored).decode("utf-8") if not isinstance(stored, (bytes, str)) else (
        stored.decode("utf-8") if isinstance(stored, bytes) else stored)
    return [match.group(2) for match in
            re.finditer(r"^Add\('page', name=(['\"])(.*?)\1", script, flags=re.MULTILINE)]


# %% ===========================================================================
#    GROWING A DOCUMENT IN PLACE (0.0.18)
# =============================================================================
#: Top-level HDF5 group for the appender's private bookkeeping (e.g. a logger's
#: column index). Veusz reads only /Veusz [R1: loader.loadHDF5Doc], so nothing
#: stored here appears in, or disturbs, the Veusz document.
APPEND_META_GROUP = "VszAppend"


class AppendLockedError(OSError):
    """The document is locked by another process (HDF5 file locking).

    Raised by VeuszAppender when the file cannot be opened for writing because
    a reader (e.g. Veusz loading it) holds the HDF5 lock. Not fatal for a
    logger: keep the rows in memory and try again on the next poll.
    """


def _is_lock_error(exc: BaseException) -> bool:
    """True for HDF5's "unable to lock file" open failure (any platform)."""
    text = str(exc).lower()
    return "lock" in text and "unable" in text


class VeuszAppender:
    """Grow an existing .vszh5 document in place, a few rows at a time.

    Typical use, once per poll of a logger::

        with VeuszAppender("log.vszh5") as app:      # created if missing
            if not app.has("t"):
                app.create_date("t")
                app.create_1d("v")
            app.append("t", epoch_to_veusz(new_epochs))
            app.append("v", new_values)
            app.set_document(doc.text())              # only when pages change

    Parameters
    ----------
    path:
        The document. When it does not exist and ``create`` is True an empty
        but valid document (Data, Document, Tags) is written atomically first.
    create:
        Create a missing document (default) or raise FileNotFoundError.
    locking:
        Passed to h5py.File(locking=...) when not None (h5py >= 3.5): True,
        False or "best-effort". None keeps the HDF5 library default.
    veusz_version, writer_tag:
        Written into a newly created document (see VeuszWriter).

    Raises
    ------
    AppendLockedError
        Another process holds the HDF5 file lock.
    OSError
        The file exists but is not a readable HDF5 file (e.g. truncated by a
        power loss); the caller decides whether to move it aside.
    ValueError
        The file is HDF5 but not a Veusz document of a supported format.

    Notes
    -----
    Not atomic: see APPENDING IN PLACE in the module docstring. Only 1-D
    datasets without error arrays and date datasets can be appended to.
    """

    def __init__(self, path: str | os.PathLike, create: bool = True, locking: Any = None,
                 veusz_version: str = VSZH5_LAYOUT_VERSION, writer_tag: str = "") -> None:
        """Open ``path`` for appending (created when missing and ``create``).

        ``locking`` is passed to h5py (None: h5py's default); a file locked by
        another process raises AppendLockedError.
        """
        if h5py is None:
            raise RuntimeError("h5py is required to append to .vszh5 files "
                               "(mamba install -c conda-forge h5py)")
        self.path = Path(path)
        self._closed = True
        if not self.path.exists():
            if not create:
                raise FileNotFoundError(str(self.path))
            # A complete, empty document written atomically: Veusz can open it
            # at any moment from now on, even before the first append.
            with VeuszWriter(self.path, veusz_version=veusz_version, writer_tag=writer_tag) as writer:
                writer.finish("")
        open_options = {} if locking is None else {"locking": locking}
        try:
            self.f = h5py.File(self.path, "r+", **open_options)
        except OSError as exc:
            if _is_lock_error(exc):
                raise AppendLockedError(f"{self.path} is locked by another process: {exc}") from exc
            raise
        self._closed = False
        try:
            if "Veusz" not in self.f:
                raise ValueError(f"{self.path} is HDF5 but not a Veusz document")
            root = self.f["Veusz"]
            document_format = int(root.attrs.get("vsz_format", 0))
            if not 1 <= document_format <= VSZH5_FORMAT:
                raise ValueError(f"{self.path}: unsupported Veusz document format {document_format}")
            self.root = root
            self.data = root.require_group("Data")
            self.docgrp = root.require_group("Document")
            # Veusz 4.2.1 tagHDF5Datasets() reads Document/Tags unconditionally.
            self.tags_group = self.docgrp.require_group("Tags")
            self._tags: dict[str, list[str]] = {
                tag: [_decode(value) for value in self.tags_group[tag][()]]
                for tag in self.tags_group
            }
        except Exception:
            self.close(flush=False)
            raise
        self._dirty_tags: set[str] = set()

    # --- context manager ------------------------------------------------------
    def __enter__(self) -> "VeuszAppender":
        """Context manager: return the appender."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the file, flushing unless an exception occurred."""
        # Flush tags/timestamps only when the block succeeded; always close.
        self.close(flush=exc_type is None)

    # --- inspection -----------------------------------------------------------
    def _group(self, name: str) -> Any:
        """HDF5 group of dataset ``name``, or None when it does not exist."""
        return self.data.get(escape_hdf_name(name))

    def has(self, name: str) -> bool:
        """True when a dataset called ``name`` exists."""
        return self._group(name) is not None

    def names(self) -> list[str]:
        """All dataset names in the document (unescaped, unsorted)."""
        out = []
        for member in self.data:
            stored = self.data[member].get("data")
            raw = stored.attrs.get("vsz_name") if stored is not None else None
            out.append(_decode(raw) if raw is not None else
                       member.replace("`SL", "/").replace("`BT", "`"))
        return out

    def length(self, name: str) -> int:
        """Number of rows of dataset ``name`` (KeyError when missing)."""
        group = self._group(name)
        if group is None:
            raise KeyError(name)
        return int(group["data"].shape[0])

    # --- creating datasets ------------------------------------------------------
    def _create(self, name: str, datatype: str, values: Any, prefix_nan: int,
                tags: Sequence[str]) -> None:
        """Create appendable dataset ``name`` of ``datatype`` ('1d' or 'date').

        Refuses an existing name and multidimensional ``values`` before anything
        is written (0.0.23); ``prefix_nan`` NaN rows come first so a new column
        lines up with older ones; ``tags`` are attached.
        """
        if self.has(name):
            raise ValueError(f"dataset {name!r} already exists")
        # 0.0.23: refuse 2-D (or higher) input before the group exists; it
        # was ravelled into a longer column before.
        array = _one_dimensional(values, f"dataset {name!r}")
        if prefix_nan > 0:                       # back-fill rows before this column existed
            array = np.concatenate([np.full(int(prefix_nan), np.nan), array])
        group = self.data.create_group(escape_hdf_name(name))
        group.attrs["vsz_datatype"] = datatype
        dataset = _create_resizable_1d(group, "data", array)
        dataset.attrs["vsz_name"] = name.encode("utf-8")
        if datatype == "date":
            dataset.attrs["vsz_convert_datetime"] = 1
        self.add_tags(name, tags)

    def create_1d(self, name: str, values: Any = (), prefix_nan: int = 0,
                  tags: Sequence[str] = ()) -> None:
        """New appendable numeric dataset; ``prefix_nan`` NaN rows come first."""
        self._create(name, "1d", values, prefix_nan, tags)

    def create_date(self, name: str, veusz_seconds: Any = (), prefix_nan: int = 0,
                    tags: Sequence[str] = ()) -> None:
        """New appendable date dataset in Veusz date seconds (see epoch_to_veusz)."""
        self._create(name, "date", veusz_seconds, prefix_nan, tags)

    # --- growing datasets ------------------------------------------------------
    def _appendable(self, name: str) -> Any:
        """The 'data' array of ``name``, converted to the resizable layout if needed."""
        group = self._group(name)
        if group is None:
            raise KeyError(name)
        datatype = _decode(group.attrs.get("vsz_datatype", b""))
        if datatype not in ("1d", "date"):
            raise ValueError(f"dataset {name!r} is {datatype!r}; only 1d and date can grow")
        if any(key in group for key in ("serr", "perr", "nerr")):
            raise ValueError(f"dataset {name!r} has error arrays; appending would misalign them")
        dataset = group["data"]
        if dataset.ndim != 1:
            # 0.0.23: e.g. a 2-D date array stored by writer 0.0.21. Veusz
            # cannot load such a file at all, and flattening it here (0.0.22
            # did) would silently turn it into a different column. Leave the
            # file as it is and say so.
            raise ValueError(f"dataset {name!r} is stored with shape {dataset.shape}; only "
                             f"1-D datasets can grow (Veusz cannot load this document - "
                             f"rewrite it with 1-D data)")
        if dataset.maxshape[0] is None:
            return dataset
        # Written by a non-resizable writer: rewrite it once in the growable
        # layout, keeping its attributes (vsz_name, vsz_convert_datetime).
        values = dataset[()]
        attributes = dict(dataset.attrs)
        del group["data"]
        dataset = _create_resizable_1d(group, "data", values)
        for key, value in attributes.items():
            dataset.attrs[key] = value
        return dataset

    def append(self, name: str, values: Any) -> int:
        """Append rows to a 1-D or date dataset; returns its new length.

        ``values`` is a 1-D sequence or a single number (one row); input
        with two or more dimensions raises ValueError (0.0.23; it was
        flattened before).
        """
        array = _one_dimensional(values, f"rows for {name!r}")
        dataset = self._appendable(name)
        start = int(dataset.shape[0])
        if array.size:
            dataset.resize((start + array.size,))
            dataset[start:] = array
        return start + int(array.size)

    def pad_to(self, name: str, length: int) -> int:
        """Append NaN rows until ``name`` has ``length`` rows; returns the length."""
        current = self.length(name)
        if current < length:
            return self.append(name, np.full(int(length) - current, np.nan))
        return current

    def truncate(self, name: str, length: int) -> int:
        """Shrink ``name`` to ``length`` rows (no-op when already shorter).

        Used by a logger to realign a dataset that an interrupted append left
        longer than its time axis. Returns the resulting length.
        """
        dataset = self._appendable(name)
        if int(dataset.shape[0]) > int(length):
            dataset.resize((max(0, int(length)),))
        return int(dataset.shape[0])

    # --- document script and tags --------------------------------------------------
    def set_document(self, document_text: str) -> None:
        """Replace the Veusz command script (e.g. after new datasets appeared)."""
        if "document" in self.docgrp:
            del self.docgrp["document"]
        self.docgrp["document"] = [document_text.encode("utf-8")]

    def add_tags(self, name: str, tags: Iterable[str]) -> None:
        """Tag dataset ``name``; tag lists are rewritten on flush()/close()."""
        for tag in tags:
            members = self._tags.setdefault(str(tag), [])
            if name not in members:
                members.append(name)
                self._dirty_tags.add(str(tag))

    # --- private bookkeeping (ignored by Veusz) ------------------------------------
    def get_meta(self, key: str) -> str | None:
        """Text stored with set_meta(), or None."""
        group = self.f.get(APPEND_META_GROUP)
        if group is None or key not in group:
            return None
        stored = group[key][()]
        return _decode(stored[0] if getattr(stored, "ndim", 0) else stored)

    def set_meta(self, key: str, text: str) -> None:
        """Store text (e.g. JSON) under /VszAppend/<key>, outside the document."""
        group = self.f.require_group(APPEND_META_GROUP)
        if key in group:
            del group[key]
        group[key] = [str(text).encode("utf-8")]

    # --- completion ------------------------------------------------------------------
    def flush(self) -> None:
        """Write changed tag lists and the save time, then flush HDF5 buffers."""
        for tag in sorted(self._dirty_tags):
            if tag in self.tags_group:
                del self.tags_group[tag]
            self.tags_group[tag] = [n.encode("utf-8") for n in sorted(self._tags[tag])]
        self._dirty_tags.clear()
        self.root.attrs["vsz_saved_at"] = (
            _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat())
        self.f.flush()

    def close(self, flush: bool = True) -> None:
        """Flush (optional) and close; safe to call more than once."""
        if self._closed:
            return
        try:
            if flush:
                self.flush()
        finally:
            self._closed = True
            self.f.close()


def _decode(value: Any) -> str:
    """bytes or numpy bytes -> str (UTF-8); anything else -> str()."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


# %% ===========================================================================
#    FINDING AND DRIVING VEUSZ (only to open or render)
# =============================================================================
def find_veusz_exe(explicit: str = "", env_vars: Sequence[str] = ("QUICKSYN_VEUSZ", "VEUSZ_EXE")) -> str:
    """Locate the Veusz program: explicit path, env var, PATH, usual folders."""
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    for var in env_vars:
        p = os.environ.get(var, "").strip().strip('"')
        if p and os.path.isfile(p):
            return p
    names = ["veusz.exe", "veusz"] if os.name == "nt" else ["veusz"]
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    extra: list[str] = []
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", ""), os.environ.get("LOCALAPPDATA", "")):
            if base:
                extra += [os.path.join(base, "Veusz", "veusz.exe"),
                          os.path.join(base, "Programs", "Veusz", "veusz.exe")]
    elif sys.platform == "darwin":
        extra.append("/Applications/Veusz.app/Contents/MacOS/veusz")
    else:
        extra += ["/opt/veusz/veusz", os.path.expanduser("~/.local/opt/veusz/veusz")]
    return next((e for e in extra if os.path.isfile(e)), "")


def _popen_kwargs() -> dict[str, Any]:
    """Hide the console window of child processes on Windows."""
    return {"creationflags": 0x08000000} if os.name == "nt" else {}


def _render_script(src: Path, pdf: Path | None, png_stem: Path | None,
                   png_pages: str, dpi: int, png_background: str = "") -> str:
    """The script piped into `veusz --listen`. Paths go through repr() so
    Windows back-slashes are safe."""
    # 0.0.23: `veusz --listen` runs every line on its own and only prints an
    # exception, and loadHDF5Doc() builds the pages BEFORE it loads the
    # datasets. A dataset Veusz refuses therefore left the pages in place and
    # the old script exported empty plots as a success. The flag is set on
    # the same line as the load, after it, so it is True only when the whole
    # load (pages, data, tags) succeeded; otherwise no page is reported (the
    # caller's "no pages loaded" rule fails the render) and nothing is
    # exported, so an earlier PDF of the same name survives.
    lines = [
        "import veusz.document.loader as L",
        "doc = Root._ci.document",
        "vszh5_loaded = False",
        f"L.loadHDF5Doc(doc, {str(src)!r}); vszh5_loaded = True",
        "names = [p.name for p in doc.basewidget.children] if vszh5_loaded else []",
        "print('VSZH5_PAGES', len(names), '|'.join(names), flush=True)",
    ]
    if pdf is not None:
        lines.append(f"Export({str(pdf)!r}, page=list(range(len(names))), "
                     f"pdfdpi={PDF_DPI}) if names else None")
    if png_stem is not None:
        # 0.0.18 (FSP branch): not min(1, ...) - Veusz 4.2.1 evaluates listen
        # scripts with NumPy's names imported, where min is numpy.min (axis).
        upto = "len(names)" if png_pages == "all" else "(1 if names else 0)"
        # The path is bound on its own line, where repr() gives a valid
        # literal; the exec() string carries no path, so a Windows back-slash
        # is never unescaped a second time (C:\\Users would become \\U...).
        lines.append(f"vszh5_png_stem = {str(png_stem)!r}")
        # 0.0.20: optional PNG background (Export's backcolor, Veusz 3.6/4.2).
        background = f", backcolor={png_background!r}" if png_background else ""
        lines.append(
            f"exec(\"for i in range({upto}): Export(vszh5_png_stem + '_p%02d_%s.png' "
            f"% (i, names[i]), page=[i], dpi={int(dpi)}{background})\")")
    lines += ["print('VSZH5_DONE', flush=True)", "Quit()", ""]
    return "\n".join(lines)


#: One bookmark: (title, page index) or (title, page index, children). A page
#: index of None points at the first page of the entry's first child.
OutlineEntry = tuple

#: What render_pages() accepts as bookmarks: True (one per page), False (none),
#: an outline list, or a callable mapping the Veusz page names to an outline.
Bookmarks = Union[bool, Sequence[OutlineEntry], Callable[[List[str]], Sequence[OutlineEntry]]]


def _outline_parts(entry: OutlineEntry) -> tuple[str, int | None, Sequence[OutlineEntry]]:
    """Split an outline entry into (title, page, children)."""
    if len(entry) == 2:
        return str(entry[0]), entry[1], ()
    title, page, children = entry
    return str(title), page, children or ()


def add_pdf_bookmarks(pdf_path: str | os.PathLike, outline: Sequence[OutlineEntry],
                      title: str = "", author: str = "", subject: str = "",
                      keywords: str = "", open_outline: bool = True) -> int:
    """Write a bookmark (outline) hierarchy into an existing PDF, in place.

    Parameters
    ----------
    pdf_path:
        The PDF to annotate. It is rewritten through "<name>.part" and
        os.replace(), so a reader never sees a half-written file.
    outline:
        Nested entries, see OutlineEntry. Page indices count from 0.
    title, author, subject, keywords:
        Document information; empty strings leave the existing value.
    open_outline:
        Expand every level and open the reader's bookmark pane on load.

    Returns
    -------
    int
        Number of bookmarks written.

    Raises
    ------
    ImportError
        pypdf is not installed.
    ValueError
        An entry points outside the document.
    """
    try:
        from pypdf import PdfWriter
    except ImportError as exc:
        raise ImportError("pypdf is needed for PDF bookmarks "
                          "(mamba install -c conda-forge pypdf)") from exc
    path = Path(pdf_path)
    writer = PdfWriter(clone_from=str(path))
    n_pages = len(writer.pages)

    def first_page(entry: OutlineEntry) -> int | None:
        """Page of an outline entry: its own page, else the first page of its
        children (None when neither has one).
        """
        _, page, children = _outline_parts(entry)
        if page is not None:
            return int(page)
        return next((p for p in map(first_page, children) if p is not None), None)

    count = 0

    def add(entries: Sequence[OutlineEntry], parent: Any) -> None:
        """Add ``entries`` (and their children, recursively) under ``parent``;
        an entry pointing outside the PDF raises ValueError.
        """
        nonlocal count
        for entry in entries:
            name, _, children = _outline_parts(entry)
            target = first_page(entry)
            if target is None or not 0 <= target < n_pages:
                raise ValueError(f"bookmark {name!r} points at page {target}; "
                                 f"the PDF has {n_pages} pages")
            item = writer.add_outline_item(name, target, parent=parent, is_open=open_outline)
            count += 1
            add(children, item)

    add(outline, None)
    info = {key: value for key, value in (("/Title", title), ("/Author", author),
                                          ("/Subject", subject), ("/Keywords", keywords))
            if value}
    if info:
        writer.add_metadata(info)
    if open_outline:
        writer.page_mode = "/UseOutlines"
    part = path.with_name(path.name + ".part")
    try:
        with open(part, "wb") as handle:
            writer.write(handle)
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()
    return count


def render_pages(vszh5: str | os.PathLike, outdir: str | os.PathLike | None = None,
                 dpi: int = 150, pdf: bool = True, png_pages: str = "all",
                 veusz_exe: str = "", timeout: float = 600.0,
                 method: str = "auto", bookmarks: Bookmarks = True,
                 pdf_info: dict[str, str] | None = None,
                 png_background: str = "") -> list[Path]:
    """Render a .vszh5 to one multipage PDF and per-page PNG files.

    Parameters
    ----------
    vszh5:
        The document.
    outdir:
        Output folder; default is the document's folder.
    dpi:
        PNG resolution.
    pdf:
        Write "<stem>.pdf" holding every page.
    png_pages:
        "all" (every page), "first", or "none".
    veusz_exe:
        Veusz program to drive; found with find_veusz_exe() when empty.
    method:
        "auto" (the program if found, falling back to the Python package if
        the program fails or is absent), "program" (the Veusz executable
        only) or "inprocess" (the Python package only).
    bookmarks:
        Outline for the PDF: True (default) = one bookmark per page named
        after the Veusz page; a list of OutlineEntry; a callable that gets
        the page names and returns such a list; False = none. Needs pypdf;
        without it a RuntimeWarning is issued and the PDF has no outline.
    pdf_info:
        Optional document information for the PDF: title, author, subject,
        keywords (see add_pdf_bookmarks).
    png_background:
        (0.0.20) Background of the PNG files: a colour name or #RRGGBBAA
        ("white", "#ffffffff"). "" (default) keeps Veusz's transparent
        background.

    Returns
    -------
    list of Path
        Files written by this call (re-rendered files included).

    Raises
    ------
    RuntimeError
        Veusz is neither installed as a program nor importable, or the
        render did not complete - including (0.0.23) a document whose
        pages load but whose data Veusz refuses; nothing is exported then.
    """
    src = Path(vszh5).resolve()
    out = Path(outdir).resolve() if outdir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / src.stem
    # 0.0.18 (FSP branch): append, not with_suffix() - "run v0.0.2" must
    # not become "run v0.0".
    pdf_path = stem.parent / (stem.name + ".pdf") if pdf else None
    png_stem = stem if png_pages in ("all", "first") else None
    started = time.time()

    if method not in ("auto", "program", "inprocess"):
        raise ValueError(f"method must be auto, program or inprocess, not {method!r}")
    exe = "" if method == "inprocess" else (veusz_exe or find_veusz_exe())
    if method == "program" and not exe:
        raise RuntimeError("Veusz program not found (set VEUSZ_EXE or QUICKSYN_VEUSZ to veusz.exe)")
    if exe:
        env = dict(os.environ)
        if sys.platform.startswith("linux") and not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
        script = _render_script(src, pdf_path, png_stem, png_pages, dpi, png_background)
        ok, log = _run_listen(exe, script, env, timeout)
        names = _page_names_from_log(log)
        if ok and not names:
            # 0.0.18 (FSP branch) / 0.0.19, merged in 0.0.20: a load error
            # inside --listen (e.g. "No HDF5 support as h5py module is
            # missing", or an unreadable file) is printed, not raised, and
            # the done marker still arrives. No page loaded is a failed
            # render, with or without a Traceback in the log. 0.0.23: this
            # includes a document whose pages loaded but whose data did not
            # (the render script then reports no pages).
            ok = False
            log += ("\n[vszh5_writer] Veusz finished the script but loaded no pages: the "
                    "document or its data could not be loaded (see the load error above; "
                    "Veusz needs h5py to open .vszh5, and 1-D and date datasets must be 1-D).")
        if not ok:
            program_error = (f"Veusz program {exe} did not finish the render "
                             f"(check that `{os.path.basename(exe)} --version` runs; set "
                             f"VEUSZ_EXE or QUICKSYN_VEUSZ to a working Veusz): {log.strip()[-600:]}")
            if method == "program":
                raise RuntimeError(program_error)
            # auto: a Veusz found on the PATH can be broken (e.g. a Linux
            # distribution package built for another Python), so try the
            # veusz Python package before giving up.
            try:
                names = _render_in_process(src, pdf_path, png_stem, png_pages, dpi, png_background)
            except RuntimeError as exc:
                raise RuntimeError(f"{program_error}\nIn-process fallback: {exc}") from exc
    else:
        names = _render_in_process(src, pdf_path, png_stem, png_pages, dpi, png_background)

    if pdf_path is not None and pdf_path.is_file() and bookmarks is not False:
        if bookmarks is True:
            outline = [(name, i) for i, name in enumerate(names)]
        elif callable(bookmarks):
            outline = list(bookmarks(list(names)))
        else:
            outline = list(bookmarks)
        try:
            if outline:
                add_pdf_bookmarks(pdf_path, outline, **(pdf_info or {}))
        except ImportError as exc:
            warnings.warn(f"{pdf_path.name} written without bookmarks: {exc}",
                          RuntimeWarning, stacklevel=2)

    # Report what this call wrote, including files that existed before and
    # were overwritten (0.0.5 compared against the folder listing taken at
    # the start, so a re-render returned nothing).
    written = [pdf_path] if pdf_path is not None else []
    if png_stem is not None:
        # Frequency-per-page exports can exceed 99 pages; :02d is a
        # minimum width, not a two-digit limit. Include p100, p1000, ... .
        written += sorted(path for path in out.glob(f"{glob.escape(png_stem.name)}_p*_*.png")
                          if re.match(re.escape(png_stem.name) + r"_p[0-9]+_", path.name))
    result = sorted(p for p in written
                    if p.is_file() and p.stat().st_mtime >= started - 2.0)
    # 0.0.18 (FSP branch): an Export() error inside --listen is also printed,
    # not raised, so check that every requested file was written.
    missing = []
    if pdf_path is not None and pdf_path not in result:
        missing.append(pdf_path.name)
    if png_stem is not None and names and not any(p.suffix.lower() == ".png" for p in result):
        missing.append(png_stem.name + "_p*.png")
    if missing:
        raise RuntimeError(f"Veusz reported a finished render of {src.name} but did not write "
                           f"{', '.join(missing)}")
    return result


def _page_names_from_log(log: str) -> list[str]:
    """Veusz page names from the VSZH5_PAGES line the render script prints."""
    for line in log.splitlines():
        if line.startswith("VSZH5_PAGES"):
            parts = line.split(maxsplit=2)
            return parts[2].split("|") if len(parts) == 3 and parts[2] else []
    return []


def _run_listen(exe: str, script: str, env: dict[str, str], timeout: float) -> tuple[bool, str]:
    """Pipe a script into `veusz --listen --quiet` and wait for the marker.

    Veusz 3.6.2 exits on Quit(); Veusz 4.2.1 only closes its hidden window
    and keeps running. Every Export() has completed by the time the script
    prints VSZH5_DONE, so after the marker the program gets two seconds to
    exit on its own and is then terminated.
    """
    import threading

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        try:
            proc = subprocess.Popen([exe, "--listen", "--quiet"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=err, text=True,
                                    encoding="utf-8", errors="replace", env=env,
                                    **_popen_kwargs())
        except OSError as exc:      # missing, not executable, wrong architecture
            return False, f"cannot start {exe}: {type(exc).__name__}: {exc}"
        killer = threading.Timer(timeout, proc.kill)
        killer.start()
        lines: list[str] = []
        done = False
        try:
            try:
                proc.stdin.write(script)
                proc.stdin.close()
            except OSError:
                # BrokenPipeError: the program exited before reading its
                # script (typically it cannot start at all). Not fatal here:
                # its stderr says why, and the missing VSZH5_DONE marker
                # reports the failure to the caller, which may fall back.
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            for line in proc.stdout:
                lines.append(line)
                if line.startswith("VSZH5_DONE"):
                    done = True
                    break
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            killer.cancel()
            if proc.poll() is None:     # an exception above: never leave Veusz behind
                proc.kill()
                proc.wait()
            proc.stdout.close()         # the pipe is ours to close (ResourceWarning)
        err.seek(0)
        return done, "".join(lines) + err.read()


def _render_in_process(src: Path, pdf_path: Path | None, png_stem: Path | None,
                       png_pages: str, dpi: int, png_background: str = "") -> list[str]:
    """Fallback: render with the veusz Python package in this process.

    Returns the page names, for the PDF bookmarks.
    """
    try:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import veusz.qtall as vqt
        _app = vqt.QApplication.instance() or vqt.QApplication([])
        import veusz.document as vdoc
        import veusz.widgets  # noqa: F401  registers widget types
        import veusz.dataimport  # noqa: F401  registers Import* commands used by the loader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Veusz not found: install it (conda-forge 'veusz' or the "
                           "Veusz installer) or set VEUSZ_EXE (or QUICKSYN_VEUSZ) to veusz.exe "
                           f"[{type(exc).__name__}: {exc}]") from exc
    doc = vdoc.Document()
    try:
        doc.load(str(src), mode="hdf5")
    except Exception as exc:  # noqa: BLE001 - 0.0.23: report as documented
        # e.g. ValueError "Only 1-dimensional arrays or lists allowed" for a
        # date dataset of the wrong shape, or LoadError without h5py.
        raise RuntimeError(f"the veusz package could not load {src.name}: "
                           f"{type(exc).__name__}: {exc}") from exc
    pages = doc.basewidget.children
    if not pages:
        raise RuntimeError(f"{src.name} loaded as an empty Veusz document")
    ci = vdoc.CommandInterface(doc)
    if pdf_path is not None:
        ci.Export(str(pdf_path), page=list(range(len(pages))), pdfdpi=PDF_DPI)
    if png_stem is not None:
        upto = len(pages) if png_pages == "all" else 1
        extra = {"backcolor": png_background} if png_background else {}
        for i in range(upto):
            ci.Export(f"{png_stem}_p{i:02d}_{pages[i].name}.png", page=[i], dpi=dpi, **extra)
    return [page.name for page in pages]


# %% ===========================================================================
#    INTERACTIVE SCRATCH CELL (Spyder)
# =============================================================================
# import numpy as np
# from vszh5_writer import VeuszWriter, VszDoc, render_pages
# t = np.linspace(0, 10, 501); v = np.sin(t)
# doc = VszDoc("scratch"); doc.add("page", "p"); doc.to("p")
# doc.add("graph", "g"); doc.to("g"); doc.add("axis", "x"); doc.add("axis", "y")
# doc.set("y/direction", "vertical"); doc.add("xy", "c")
# doc.set("c/xData", "t"); doc.set("c/yData", "v"); doc.set("c/marker", "none")
# doc.up(); doc.up()
# with VeuszWriter("scratch.vszh5") as w:
#     w.dataset_1d("t", t); w.dataset_1d("v", v); w.finish(doc.text())
# print(render_pages("scratch.vszh5", png_pages="first"))
