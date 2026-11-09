#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
ATR_AutoPlot.py

GBO Outdoor Antenna Range (.atr) pattern files -> native Veusz project
(.vszh5) and interactive pyqtgraph plots:

    * every file: magnitude and phase versus azimuth (Cartesian) and as
      polar patterns; multi-frequency files give one trace per frequency;
    * overlay pages of all imported magnitudes and phases;
    * the .vszh5 is written by the author's shared writer (vszh5_writer.py
      0.0.24 through _autoplot_native.NativeDocument) - Veusz is NOT needed
      to create the file, only (optionally) to view it;
    * the Plots tab draws the same pages with pyqtgraph (polar plots with r
      at the cursor angle), in the suite's light or dark theme.

Batch use without the GUI:
    python ATR_AutoPlot.py --output pattern.vszh5 file1.ATR file2.ATR ...

Created on 2025-06-28

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later (GNU General Public License v3.0 or later)
Compatible with: Python 3.8+ (qtpy with PySide6 or PyQt5)

# %%% ATR file layout (from the example files in Example_Data_Files/atr)
Lines 1-12 are "Key: Value" header lines (the first may be quoted and lines
may end in tabs when the file went through a spreadsheet):
    Time and Date, Receiver Name, Telescope, Focus, "Sweep by Position CW|CCW"
    (no colon), Plane, Left Azimuth Limit, Right Azimuth Limit,
    Upper Frequency, Lower Frequency (MHz), Frequency Increment (MHz),
    Power level (dBm)
then one data line per frequency: "phase(deg) magnitude(dB)" pairs, one pair
per azimuth from the left to the right limit (optionally ending in '#'),
then a line "*".  The pair order (phase first) and the left-to-right order
for CW and CCW sweeps are confirmed by the annotated example
peaking_8_ccw_0p1azaccuracy.ATR ("Deg dBm ... Angle of measurement" table,
first pair at -135 deg).  Not documented anywhere available here: whether
the data lines of a multi-frequency file run from the lower to the upper
frequency; this program assumes lower -> upper (see DATA_GAPS).

# %%% References
[1] W. W. Wallace, "vszh5_writer.py 0.0.24," shared Veusz HDF5 writer, 2026.
[2] J. Sanders, "Veusz - a scientific plotting package," Veusz 4.2.
    https://veusz.github.io/
[3] pyqtgraph developers, "pyqtgraph documentation."
    https://pyqtgraph.readthedocs.io/en/latest/
[4] IEEE Standard Test Procedures for Antennas, IEEE Std 149-2021
    (Revision of IEEE Std 149-1979), 2022. doi: 10.1109/IEEESTD.2022.9714428

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.2.2: Header template, docstrings and tooltips.
Date: 2026-11-09
#              * Header in the author's template: author contact lines and,
#                under this entry, Function Descriptions, Variable
#                Descriptions and More Info.
#              * Docstrings for every function, class and method.  No change
#                to what the program does.
#              * Tooltips (pop-up help) on every button, box and field that
#                had none (file buttons, options, log, Process, Save, Open in
#                Veusz, Close).
# %%%%% Function Descriptions
        ATRFormatError (class): The file is not an ATR pattern file this program can read.
        ATRPattern (class): One parsed .atr file (n_freq traces of n_az points).
        parse_header_time: "Tue, Jul 02, 2024 12:19 PM" (and close variants) -> naive datetime.
        _header_number: First number in header field ``key`` of an ATR file.
        parse_atr_text: Parse the text of one ATR file.
        _is_float: True when ``token`` parses as a float.
        read_atr_file: Read and parse one file (ASCII; undecodable bytes are replaced).
        _read_worker: Process-pool worker: (path, pattern or None, error text).
        read_files: Read files in input order, in a 'spawn' process pool when it pays off.
        PlotOptions (class): Plot settings chosen in the GUI (or on the command line).
        ATRPlotter (class): Builds the Veusz project of a batch of patterns.
        build_project: ATRPlotter with every pattern added (input order).
        _build_gui_classes: Import Qt and define the GUI classes on first use.
        batch: Command-line conversion; returns the process exit code.
        main: Command line entry point; returns the exit code.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP_NAME = "ATR AutoPlot"
        APP_VERSION = "0.2.2"
        APP_DESCRIPTION: one-line description (command line help, About box)
        DATA_GAPS: Known gaps in the format description (shown in the log once per run).
        DEFAULT_MAG_RANGE: dB, as in the 2025 version
        PHASE_RANGE: deg
        AZIMUTH_RANGE: deg, full-circle axis (default)
        PARALLEL_MIN_FILES: fewer files: a process pool costs more than it saves
        _NUMBER_LINE: data line: starts with a number (inf and nan allowed)
# %%%%% More Info
    Run:  python ATR_AutoPlot.py [files]   (GUI; View menu: light/dark)
          python ATR_AutoPlot.py *.atr --output atr.vszh5   (batch, no GUI)
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.2.1: Measurement time as a Veusz date-time; headers on every page.
Date: 2026-11-09
#              * The header "Time and Date" (e.g. "Tue, Jul 02, 2024 12:19
#                PM") is stored as a Veusz DATE dataset "<file>_datetime"
#                (the range computer's wall clock as written; the file does
#                not say which zone) and as ISO 8601 in the page notes.
#              * Page notes: "Source: <file>", the header lines and the
#                frequencies of the file (unchanged layout, source added);
#                the overlay pages list every file with its header.
#              * Multi-frequency files: the traces are named after their y
#                dataset ("<file>_1200MHz_mag", ...) instead of "1200MHz".
#                Page and graph names are those of the 2025 script.
#              * Writer 0.0.25 / _autoplot_native 0.3.0.
# %%%% 0.2.0: Native writer, pyqtgraph Plots tab, suite theme, parser fixes.
Date: 2026-11-05
#              * .vszh5 written by vszh5_writer 0.0.24 through
#                _autoplot_native.NativeDocument (no veusz.embed, no Veusz
#                window, no extra legacy .vsz copy in "Beware_oldVersion").
#              * Plots tab (pyqtgraph): Cartesian and polar pages, r at the
#                cursor angle, light / dark with the rest of the suite.
#              * View > Light / Dark Mode (shared suite theme), Help > About
#                (NRAO / GBO), File menu with shortcuts.
#              * Parser: header read by key, not by line number; quoted and
#                tab-padded header lines; all data lines of a file (one per
#                frequency) instead of the first only; azimuth from the
#                header limits and the number of points (was a fixed 1 deg
#                step); the frequency of a single-frequency file is no longer
#                the "Upper" value by accident; odd value counts, missing
#                lines and non-numeric values are reported, not crashes;
#                files are read as ASCII with replacement characters.
#              * File dialog accepts *.ATR as well as *.atr.
#              * Check boxes use toggled(bool): the old stateChanged ==
#                Qt.Checked test was always False under PySide6, so the
#                multiprocessing switch could not be turned on again.
#              * Process pool uses the 'spawn' start method and receives only
#                file paths (the old worker argument carried a GPU context
#                that cannot be pickled).  GPU code removed: the files hold
#                a few hundred numbers and no array operation was applied to
#                them, so the GPU path was never used.
#              * "Plot Title" now titles the overlay pages; the unused
#                "Dataset Name" field was removed.  Magnitude range and
#                "fit azimuth axis to the data" are options.
#              * Dataset names stay {file}_mag, {file}_phase, {file}_Az,
#                {file}_freq for single-frequency files (as before);
#                multi-frequency files add the frequency to the name.
# %%%% 0.1.0: Un-numbered 2025-06-28 version (veusz.embed, Qt GUI).
Date: 2025-06-28
=============================================================================
"""
# %% Import Modules
from __future__ import annotations

import argparse
import datetime
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("QT_API", "pyside6")   # before qtpy is imported

import numpy as np  # noqa: E402
from vszh5_writer import VEUSZ_EPOCH, header_notes  # noqa: E402

APP_NAME = "ATR AutoPlot"
APP_VERSION = "0.2.2"
__version__ = APP_VERSION
APP_DESCRIPTION = ("GBO Outdoor Antenna Range (.atr) pattern files to native Veusz "
                   ".vszh5 projects, with interactive pyqtgraph plots.")

#: Known gaps in the format description (shown in the log once per run).
DATA_GAPS = ("Multi-frequency files: the data lines are assumed to run from the "
             "lower to the upper frequency in steps of the frequency increment.",)

DEFAULT_MAG_RANGE = (-60.0, 20.0)     # dB, as in the 2025 version
PHASE_RANGE = (-180.0, 180.0)         # deg
AZIMUTH_RANGE = (-180.0, 180.0)       # deg, full-circle axis (default)
PARALLEL_MIN_FILES = 8                # fewer files: a process pool costs more than it saves


# %% ATR parser (no Qt; runs in worker processes)
class ATRFormatError(ValueError):
    """The file is not an ATR pattern file this program can read."""


@dataclass
class ATRPattern:
    """One parsed .atr file (n_freq traces of n_az points)."""
    path: str
    header: Dict[str, str]
    header_lines: List[str]
    sweep: str                       # "CW", "CCW" or ""
    frequencies_mhz: np.ndarray      # (n_freq,)
    azimuth_deg: np.ndarray          # (n_az,)
    magnitude_db: np.ndarray         # (n_freq, n_az)
    phase_deg: np.ndarray            # (n_freq, n_az)
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """File name of the pattern (no folder)."""
        return os.path.basename(self.path)

    @property
    def measured(self) -> Optional[datetime.datetime]:
        """Header "Time and Date" (wall clock of the range computer), or None."""
        return parse_header_time(self.header.get("Time and Date", ""))


def parse_header_time(text: str) -> Optional[datetime.datetime]:
    """"Tue, Jul 02, 2024   12:19 PM" (and close variants) -> naive datetime."""
    text = " ".join(str(text).replace(",", " ").split())
    for fmt in ("%a %b %d %Y %I:%M %p", "%a %b %d %Y %I:%M:%S %p", "%b %d %Y %I:%M %p",
                "%a %b %d %Y %H:%M", "%a %b %d %Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p",
                "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


_NUMBER_LINE = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+|inf|nan)([eE][-+]?\d+)?[\s,;]",
                          re.IGNORECASE)


def _header_number(header: Dict[str, str], key: str, path: str) -> float:
    """First number in header field ``key`` of an ATR file.

    Raises ATRFormatError (naming ``path``) when the field is missing or holds
    no number.
    """
    try:
        return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", header[key])[0])
    except (KeyError, IndexError, ValueError):
        raise ATRFormatError("%s: header value %r missing or not a number"
                             % (os.path.basename(path), key))


def parse_atr_text(text: str, path: str = "<text>") -> ATRPattern:
    """
    Parse the text of one ATR file.

    Raises
    ------
    ATRFormatError
        When the header limits or the data lines are missing.
    """
    lines = text.splitlines()
    header: Dict[str, str] = {}
    header_lines: List[str] = []
    sweep = ""
    index = 0
    for index, raw in enumerate(lines):           # header until the first numeric line
        line = raw.strip().strip('"').strip()
        if _NUMBER_LINE.match(line + " "):
            break
        if not line:
            continue
        header_lines.append(line)
        match = re.match(r"sweep by position\s+(\w+)", line, re.IGNORECASE)
        if match:
            sweep = match.group(1).upper()
        elif ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    else:
        raise ATRFormatError("%s: no data line found" % os.path.basename(path))
    warnings: List[str] = []
    rows: List[np.ndarray] = []
    for number, raw in enumerate(lines[index:], start=index + 1):
        line = raw.strip().rstrip("#").strip()
        if not line:
            continue
        if line.startswith("*") or not _NUMBER_LINE.match(line + " "):
            break                                   # end marker or a trailing table
        tokens = line.replace(",", " ").split()
        values = np.array([float(t) if _is_float(t) else np.nan for t in tokens])
        bad = int(np.isnan(values).sum())
        if bad:
            warnings.append("line %d: %d value(s) are not numbers (kept as gaps)" % (number, bad))
        if values.size % 2:
            warnings.append("line %d: odd number of values (%d); the last one is "
                            "ignored" % (number, values.size))
            values = values[:-1]
        rows.append(values)
    if not rows:
        raise ATRFormatError("%s: no data line found" % os.path.basename(path))
    n_points = {row.size // 2 for row in rows}
    if len(n_points) != 1:
        raise ATRFormatError("%s: data lines have different lengths %s"
                             % (os.path.basename(path), sorted(n_points)))
    n_az = n_points.pop()
    if n_az < 1:
        raise ATRFormatError("%s: empty data line" % os.path.basename(path))
    data = np.vstack(rows)
    phase, magnitude = data[:, 0::2], data[:, 1::2]   # pairs: phase (deg), magnitude (dB)
    for values, what in ((magnitude, "magnitude"), (phase, "phase")):
        infinite = np.isinf(values)     # "-Inf" dB: the receiver saw no signal
        if infinite.any():
            values[infinite] = np.nan
            warnings.append("%d %s value(s) were +/-Inf (no signal); stored as gaps (NaN)%s"
                            % (int(infinite.sum()), what,
                               "; no finite values left" if np.isnan(values).all() else ""))

    left = _header_number(header, "Left Azimuth Limit", path)
    right = _header_number(header, "Right Azimuth Limit", path)
    azimuth = np.linspace(left, right, n_az) if n_az > 1 else np.array([left])
    if n_az > 1:
        step = (right - left) / (n_az - 1)
        if abs(step - round(step)) > 1e-9 or abs(step) != 1.0:
            warnings.append("azimuth step %.6g deg (%d points from %g to %g deg)"
                            % (step, n_az, left, right))

    upper = _header_number(header, "Upper Frequency", path)
    lower = _header_number(header, "Lower Frequency", path)
    try:
        increment = _header_number(header, "Frequency Increment", path)
    except ATRFormatError:
        increment = 0.0
    n_freq = len(rows)
    if n_freq == 1:
        frequencies = np.array([lower])
        if upper != lower:
            warnings.append("one data line but the header spans %g-%g MHz; labelled %g MHz"
                            % (lower, upper, lower))
    else:
        expected = int(round((upper - lower) / increment)) + 1 if increment > 0 else 0
        if expected == n_freq:
            frequencies = lower + increment * np.arange(n_freq)
        elif upper == lower:
            frequencies = np.full(n_freq, lower)
            warnings.append("%d data lines at one frequency: treated as repeated sweeps"
                            % n_freq)
        else:
            frequencies = np.linspace(lower, upper, n_freq)
            warnings.append("%d data lines do not match %g-%g MHz in %g MHz steps; "
                            "frequencies spread evenly" % (n_freq, lower, upper, increment))
    return ATRPattern(path=path, header=header, header_lines=header_lines, sweep=sweep,
                      frequencies_mhz=frequencies, azimuth_deg=azimuth,
                      magnitude_db=magnitude, phase_deg=phase, warnings=warnings)


def _is_float(token: str) -> bool:
    """True when ``token`` parses as a float."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_atr_file(path: str) -> ATRPattern:
    """Read and parse one file (ASCII; undecodable bytes are replaced)."""
    with open(path, "rb") as handle:
        text = handle.read().decode("ascii", errors="replace")
    return parse_atr_text(text, path)


def _read_worker(path: str) -> Tuple[str, Optional[ATRPattern], str]:
    """Process-pool worker: (path, pattern or None, error text)."""
    try:
        return path, read_atr_file(path), ""
    except Exception as exc:  # reported per file, the batch goes on
        return path, None, "%s: %s" % (type(exc).__name__, exc)


def read_files(paths: Sequence[str], workers: int = 1,
               progress: Optional[Any] = None) -> Tuple[List[ATRPattern], List[str]]:
    """
    Read files in input order, in a 'spawn' process pool when it pays off.

    Returns (patterns, errors); ``progress(done, total)`` after each file.
    """
    paths = list(dict.fromkeys(paths))                 # drop duplicates, keep order
    results: Dict[str, Tuple[Optional[ATRPattern], str]] = {}
    if workers > 1 and len(paths) >= PARALLEL_MIN_FILES:
        context = multiprocessing.get_context("spawn")  # fork can deadlock with Qt threads
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_read_worker, p) for p in paths]
            for done, future in enumerate(as_completed(futures), start=1):
                path, pattern, error = future.result()
                results[path] = (pattern, error)
                if progress:
                    progress(done, len(paths))
    else:
        for done, path in enumerate(paths, start=1):
            _p, pattern, error = _read_worker(path)
            results[path] = (pattern, error)
            if progress:
                progress(done, len(paths))
    patterns, errors = [], []
    for path in paths:
        pattern, error = results[path]
        if pattern is None:
            errors.append("%s: %s" % (os.path.basename(path), error))
        else:
            patterns.append(pattern)
    return patterns, errors


# %% Veusz project (NativeDocument; written by vszh5_writer)
@dataclass
class PlotOptions:
    """Plot settings chosen in the GUI (or on the command line)."""
    title: str = "GBO Outdoor Antenna Range Pattern"
    mag_range: Tuple[float, float] = DEFAULT_MAG_RANGE
    fit_azimuth: bool = False


class ATRPlotter:
    """Builds the Veusz project of a batch of patterns."""

    AZ_LABEL = "Azimuth (degrees)"
    MAG_LABEL = "Magnitude (dB)"
    PHASE_LABEL = "Phase (degrees)"

    def __init__(self, options: Optional[PlotOptions] = None) -> None:
        """Empty ATR project with ``options`` (title, magnitude range, azimuth fit).

        Pages are added by add_pattern() / the overlay builders; the document is a
        NativeDocument, written by vszh5_writer.
        """
        from _autoplot_native import NativeDocument
        self.options = options or PlotOptions()
        self.doc = NativeDocument("ATR AutoPlot")
        self.doc.Root.colorTheme.val = "max128"
        self._stems: Dict[str, int] = {}
        self._overlays: Optional[Tuple[Any, Any]] = None
        self._az_span: List[float] = []
        self._overlay_notes: List[str] = []

    # ----- names ----------------------------------------------------------------
    def _unique_stem(self, pattern: ATRPattern) -> str:
        """Page-name stem for ``pattern``: its file name without extension, with
        ``_2``, ``_3``, ... for repeated names.
        """
        stem = os.path.splitext(pattern.name)[0]
        count = self._stems.get(stem, 0) + 1
        self._stems[stem] = count
        return stem if count == 1 else "%s_%d" % (stem, count)

    @staticmethod
    def _freq_tag(mhz: float) -> str:
        """Dataset-name tag for a frequency in MHz, e.g. 1420.4 -> '1420p4MHz'.
        """
        return ("%g" % mhz).replace(".", "p") + "MHz"

    # ----- public ---------------------------------------------------------------
    def add_pattern(self, pattern: ATRPattern) -> str:
        """Datasets and pages of one file; returns its dataset stem."""
        if self._overlays is None:
            self._overlays = self._create_overlay_pages()
        stem = self._unique_stem(pattern)
        doc = self.doc
        az_name, freq_name = "%s_Az" % stem, "%s_freq" % stem
        doc.SetData(az_name, pattern.azimuth_deg)
        doc.SetData(freq_name, pattern.frequencies_mhz)
        single = pattern.frequencies_mhz.size == 1
        traces = []
        for k, mhz in enumerate(pattern.frequencies_mhz):
            suffix = "" if single else "_" + self._freq_tag(mhz) + ("" if np.sum(
                pattern.frequencies_mhz == mhz) == 1 else "_%d" % (k + 1))
            mag_name, phase_name = "%s%s_mag" % (stem, suffix), "%s%s_phase" % (stem, suffix)
            doc.SetData(mag_name, pattern.magnitude_db[k])
            doc.SetData(phase_name, pattern.phase_deg[k])
            key = "%g MHz" % mhz
            traces.append((mag_name, phase_name, key))
        names = [az_name, freq_name] + [n for t in traces for n in t[:2]]
        measured = pattern.measured
        if measured is not None:            # range computer's clock (zone not recorded)
            time_name = "%s_datetime" % stem
            doc.SetDataDateTime(time_name, [(measured - VEUSZ_EPOCH).total_seconds()])
            names.append(time_name)
        doc.TagDatasets(stem, names)
        self._az_span.extend([float(np.nanmin(pattern.azimuth_deg)),
                              float(np.nanmax(pattern.azimuth_deg))])
        notes = header_notes(pattern.header_lines, source=pattern.path, extra=[
            "Frequencies (MHz): %s" % ", ".join("%g" % f for f in pattern.frequencies_mhz),
            "Measured (range computer clock, zone not recorded): %s" % (
                measured.isoformat(timespec="minutes") if measured else "not readable")])
        self._overlay_notes.append(notes)
        title = stem.replace("_", " ")
        colour = "red" if single else "auto"
        x_range = self._x_range([pattern])
        for kind, label, y_range in (("mag", self.MAG_LABEL, self.options.mag_range),
                                     ("phase", self.PHASE_LABEL, PHASE_RANGE)):
            page = doc.Root.Add("page", name="%s_%s" % (stem, kind))
            page.notes.val = notes
            graph = page.Add("grid", columns=2).Add(
                "graph", name="%s_%s" % (stem, "Mag" if kind == "mag" else "Phase"))
            self._configure_graph(graph, title, label, y_range, x_range)
            for mag_name, phase_name, key in traces:
                y_name = mag_name if kind == "mag" else phase_name
                xy = graph.Add("xy", name=stem if single else y_name)
                self._configure_xy(xy, az_name, mag_name if kind == "mag" else phase_name,
                                   colour, key, fill=single)
        for kind, r_range in (("mag", self.options.mag_range), ("phase", PHASE_RANGE)):
            page = doc.Root.Add("page", name="%s_polar_%s" % (stem, kind))
            self._add_page_title(page, title)
            page.notes.val = notes
            polar = page.Add("grid", columns=2).Add(
                "polar", name="%s_Polar_%s" % (stem, "mag" if kind == "mag" else "Phase"))
            self._configure_polar_graph(polar, *r_range)
            for mag_name, phase_name, key in traces:
                r_name = mag_name if kind == "mag" else phase_name
                point = polar.Add("nonorthpoint", name=stem if single else r_name)
                self._configure_polar_plot(point, mag_name if kind == "mag" else phase_name,
                                           az_name, colour, fill=single)
        mag_graph, phase_graph = self._overlays
        for mag_name, phase_name, key in traces:
            label = stem if single else "%s %s" % (stem, key)
            self._configure_overlay_xy(mag_graph.Add("xy", name=mag_name), az_name, mag_name, label)
            self._configure_overlay_xy(phase_graph.Add("xy", name=phase_name), az_name,
                                       phase_name, label)
        return stem

    def finish(self) -> None:
        """Overlay x ranges once all files are in (fit-to-data option)."""
        if self._overlays is not None:
            text = "\n\n".join(["Overlay of %d file(s)" % len(self._overlay_notes)]
                                + self._overlay_notes)
            for graph in self._overlays:
                graph.parent.parent.notes.val = text          # page notes
        if self._overlays is not None and self.options.fit_azimuth and self._az_span:
            for graph in self._overlays:
                graph.x.min.val = float(min(self._az_span))
                graph.x.max.val = float(max(self._az_span))

    def save(self, filename: str) -> str:
        """Write the .vszh5 (atomic, vszh5_writer); returns the path written."""
        return self.doc.save(filename)

    # ----- pages ----------------------------------------------------------------
    def _x_range(self, patterns: Sequence[ATRPattern]) -> Tuple[float, float]:
        """Azimuth axis range: the fixed AZIMUTH_RANGE, or the data's min/max over
        ``patterns`` when "fit azimuth" is set.
        """
        if not self.options.fit_azimuth:
            return AZIMUTH_RANGE
        lows = [float(np.nanmin(p.azimuth_deg)) for p in patterns]
        highs = [float(np.nanmax(p.azimuth_deg)) for p in patterns]
        low, high = min(lows), max(highs)
        return (low, high) if high > low else (low - 1.0, high + 1.0)

    def _create_overlay_pages(self) -> Tuple[Any, Any]:
        """Create the two overlay pages (all files, magnitude and phase) and
        return their graphs.
        """
        graphs = []
        for kind, label, y_range in (("mag", self.MAG_LABEL, self.options.mag_range),
                                     ("phase", self.PHASE_LABEL, PHASE_RANGE)):
            page = self.doc.Root.Add("page", name="Overlay_All_%s" % kind)
            graph = page.Add("grid", columns=2).Add("graph", name="Overlay_All_%s" % kind)
            title = "%s: overlay of imported %s" % (
                self.options.title or "ATR", "magnitude" if kind == "mag" else "phase")
            self._configure_graph(graph, title, label, y_range, AZIMUTH_RANGE)
            graphs.append(graph)
        return graphs[0], graphs[1]

    def _configure_graph(self, graph: Any, title: str, y_label: str,
                         y_range: Tuple[float, float], x_range: Tuple[float, float]) -> None:
        """Title, margins, axis labels and fixed ranges of one x-y graph."""
        graph.Add("label", name="plotTitle")
        graph.topMargin.val = "1cm"
        graph.plotTitle.Text.size.val = "10pt"
        graph.plotTitle.label.val = title
        graph.plotTitle.alignHorz.val = "centre"
        graph.plotTitle.yPos.val = 1.05
        graph.plotTitle.xPos.val = 0.5
        graph.x.label.val = self.AZ_LABEL
        graph.y.label.val = y_label
        for axis in (graph.x, graph.y):
            axis.GridLines.hide.val = False
            axis.MinorGridLines.hide.val = False
        graph.y.min.val, graph.y.max.val = float(y_range[0]), float(y_range[1])
        graph.x.min.val, graph.x.max.val = float(x_range[0]), float(x_range[1])

    @staticmethod
    def _configure_xy(xy: Any, x_data: str, y_data: str, colour: str, key: str,
                      fill: bool) -> None:
        """Style one trace: small translucent circle markers, line ``colour``,
        legend ``key`` and, with ``fill``, a translucent fill down to the axis.
        """
        xy.xData.val = x_data
        xy.yData.val = y_data
        xy.key.val = key
        xy.marker.val = "circle"
        xy.markerSize.val = "2pt"
        xy.MarkerLine.color.val = "transparent"
        xy.MarkerFill.color.val = "auto"
        xy.MarkerFill.transparency.val = 80
        xy.MarkerFill.style.val = "solid"
        xy.FillBelow.transparency.val = 90
        xy.FillBelow.style.val = "solid"
        xy.FillBelow.fillto.val = "bottom"
        xy.FillBelow.color.val = "darkgreen"
        xy.FillBelow.hide.val = not fill
        xy.PlotLine.color.val = colour

    @staticmethod
    def _configure_overlay_xy(xy: Any, x_data: str, y_data: str, key: str) -> None:
        """Overlay trace style: automatic colour, no fill."""
        ATRPlotter._configure_xy(xy, x_data, y_data, "auto", key, fill=False)

    @staticmethod
    def _configure_polar_graph(graph: Any, min_radius: float, max_radius: float) -> None:
        """Polar graph as an antenna pattern: degrees, clockwise, 0 at the top,
        radius from ``min_radius`` to ``max_radius`` dB.
        """
        graph.topMargin.val = "1cm"
        graph.units.val = "degrees"
        graph.direction.val = "clockwise"
        graph.position0.val = "top"
        graph.minradius.val = float(min_radius)
        graph.maxradius.val = float(max_radius)

    @staticmethod
    def _configure_polar_plot(point: Any, data1: str, data2: str, colour: str,
                              fill: bool) -> None:
        """Style one polar trace (``data1`` radius, ``data2`` angle) in ``colour``,
        optionally filled; the widget name carries the trace name (nonorthpoint
        has no key setting).
        """
        point.data1.val = data1          # nonorthpoint has no 'key' setting in Veusz;
        point.data2.val = data2          # the widget name carries the trace name
        point.PlotLine.color.val = colour
        point.PlotLine.width.val = "2pt"
        point.MarkerLine.transparency.val = 75
        point.MarkerFill.transparency.val = 75
        point.Fill1.transparency.val = 65
        point.Fill1.color.val = "green"
        point.Fill1.filltype.val = "center"
        point.Fill1.hide.val = not fill

    @staticmethod
    def _add_page_title(page: Any, title: str) -> None:
        """Centred 10 pt title label at the top of ``page``."""
        page.Add("label", name="plotTitle")
        page.plotTitle.Text.size.val = "10pt"
        page.plotTitle.label.val = title
        page.plotTitle.alignHorz.val = "centre"
        page.plotTitle.yPos.val = 0.95
        page.plotTitle.xPos.val = 0.5


def build_project(patterns: Sequence[ATRPattern],
                  options: Optional[PlotOptions] = None) -> ATRPlotter:
    """ATRPlotter with every pattern added (input order)."""
    plotter = ATRPlotter(options)
    for pattern in patterns:
        plotter.add_pattern(pattern)
    plotter.finish()
    return plotter


# %% GUI (imported lazily so the parser and --output work without Qt)
def _build_gui_classes():
    """Import Qt and define the GUI classes on first use.

    Keeps the batch mode (``--output``) free of Qt; returns (QApplication,
    ATRMainWindow).
    """
    from qtpy.QtCore import QThread, Signal
    from qtpy.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFileDialog,
                                QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
                                QListWidget, QMainWindow, QMessageBox, QProgressBar,
                                QPushButton, QSpinBox, QTabWidget, QTextEdit,
                                QVBoxLayout, QWidget)
    import _suite_gui as suite

    class ATRProcessingThread(QThread):
        """Reads the files off the GUI thread (pool inside when it pays off)."""
        progress_updated = Signal(int)
        processing_finished = Signal(object, object)

        def __init__(self, paths: Sequence[str], workers: int) -> None:
            """Thread that reads ``paths`` with ``workers`` processes."""
            super().__init__()
            self.paths, self.workers = list(paths), int(workers)

        def run(self) -> None:
            """Read the files (progress in percent), then emit
            ``processing_finished(patterns, errors)``; an unexpected exception is
            reported as an error, never raised in the thread.
            """
            try:
                patterns, errors = read_files(
                    self.paths, self.workers,
                    lambda done, total: self.progress_updated.emit(int(100 * done / total)))
            except Exception as exc:  # pool failure: report, never hang the GUI
                patterns, errors = [], ["batch failed: %s: %s" % (type(exc).__name__, exc)]
            self.processing_finished.emit(patterns, errors)

    class ATRMainWindow(QMainWindow):
        """Processing and Plots tabs; File / View / Help menus."""

        def __init__(self) -> None:
            """Main window: Processing and Plots tabs, menus with the light/dark
            theme, and the suite theme applied.
            """
            super().__init__()
            self.setWindowTitle("%s %s" % (APP_NAME, APP_VERSION))
            self.resize(1150, 820)
            self.selected_files: List[str] = []
            self.plotter: Optional[ATRPlotter] = None
            self.last_saved: Optional[str] = None
            self.thread: Optional[ATRProcessingThread] = None
            self.theme = suite.apply_theme(suite.saved_theme(), persist=False)
            suite.add_file_menu(self, [
                ("&Add ATR Files...", self._browse_files, "Ctrl+O"),
                ("&Clear File List", self._clear_files, ""),
                ("&Process Files", self._process, "F5"),
                ("&Save Veusz Project (.vszh5)...", self._save_project, "Ctrl+S"),
                ("Open Last Project in &Veusz", self._open_in_veusz, ""),
            ])
            self.act_light, self.act_dark = suite.add_view_menu(self, self._set_theme, self.theme)
            suite.add_help_menu(self, APP_NAME, APP_VERSION, APP_DESCRIPTION)
            self._build_ui()
            self._log("%s %s ready; theme %s." % (APP_NAME, APP_VERSION, self.theme))
            for gap in DATA_GAPS:
                self._log("Note: " + gap)

        # ----- layout -------------------------------------------------------------
        def _build_ui(self) -> None:
            """Lay out the Processing tab (file list, plot options, workers, progress,
            log, buttons) and the Plots tab (pyqtgraph panel).
            """
            central = QWidget()
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)
            self.tabs = QTabWidget()
            outer.addWidget(self.tabs, 1)
            processing = QWidget()
            self.tabs.addTab(processing, "Processing")
            layout = QVBoxLayout(processing)

            files = QGroupBox("ATR File Selection")
            file_layout = QVBoxLayout(files)
            self.file_list = QListWidget()
            self.file_list.setToolTip("Files to process; add them with Browse, remove all with Clear.")
            self.file_list.setMinimumHeight(150)
            file_layout.addWidget(self.file_list)
            row = QHBoxLayout()
            for text, slot, tip in (
                    ("Browse ATR Files...", self._browse_files,
                     "Add GBO outdoor range pattern files (.atr) to the list."),
                    ("Clear Files", self._clear_files,
                     "Remove every file from the list (the files are not touched).")):
                button = QPushButton(text)
                button.setToolTip(tip)
                button.clicked.connect(slot)
                row.addWidget(button)
            file_layout.addLayout(row)
            layout.addWidget(files)

            options = QGroupBox("Processing and Plot Options")
            form = QFormLayout(options)
            self.mp_box = QCheckBox("Read files in parallel processes (%d or more files)"
                                    % PARALLEL_MIN_FILES)
            self.mp_box.setToolTip(
                "Read the files in separate processes when there are enough of "
                "them (faster for big batches).")
            self.mp_box.setChecked(True)
            self.mp_box.toggled.connect(lambda on: self._log(
                "Parallel reading %s" % ("on" if on else "off")))
            form.addRow(self.mp_box)
            self.cpu_spin = QSpinBox()
            self.cpu_spin.setToolTip("Number of processes used for parallel reading.")
            self.cpu_spin.setRange(1, max(1, multiprocessing.cpu_count()))
            self.cpu_spin.setValue(max(1, multiprocessing.cpu_count()))
            form.addRow("CPU processes:", self.cpu_spin)
            self.title_edit = QLineEdit(PlotOptions.title)
            self.title_edit.setToolTip("Title written on every page.")
            form.addRow("Plot title:", self.title_edit)
            mag_row = QHBoxLayout()
            self.mag_min = QDoubleSpinBox()
            self.mag_min.setToolTip("Lower end of the magnitude axis (dB).")
            self.mag_max = QDoubleSpinBox()
            self.mag_max.setToolTip(
                "Upper end of the magnitude axis (dB); must be above the lower "
                "end.")
            for spin, value in ((self.mag_min, DEFAULT_MAG_RANGE[0]),
                                (self.mag_max, DEFAULT_MAG_RANGE[1])):
                spin.setRange(-300.0, 300.0)
                spin.setDecimals(1)
                spin.setSuffix(" dB")
                spin.setValue(value)
                mag_row.addWidget(spin)
            mag_row.addStretch(1)
            form.addRow("Magnitude axis / polar radius:", mag_row)
            self.fit_box = QCheckBox("Fit the azimuth axis to the data (otherwise -180 to 180 deg)")
            self.fit_box.setToolTip(
                "Fit the azimuth axis to the measured range instead of -180 to "
                "180 degrees.")
            form.addRow(self.fit_box)
            layout.addWidget(options)

            self.status_text = QTextEdit()
            self.status_text.setToolTip("Progress and messages of this session, with time stamps.")
            self.status_text.setReadOnly(True)
            self.status_text.setMinimumHeight(140)
            layout.addWidget(self.status_text, 1)

            widget, self.plot_panel = suite.make_plot_panel(self.theme == "dark")
            self.tabs.addTab(widget, "Plots")

            self.progress = QProgressBar()
            self.progress.setVisible(False)
            outer.addWidget(self.progress)
            buttons = QHBoxLayout()
            self.process_button = QPushButton("Process and Plot ATR Files")
            self.process_button.setToolTip(
                "Read the listed .atr files and build the pages (shown on the "
                "Plots tab).")
            self.process_button.clicked.connect(self._process)
            self.save_button = QPushButton("Save Veusz Project (.vszh5)")
            self.save_button.setToolTip(
                "Write the project as a native Veusz HDF5 file (.vszh5); Veusz "
                "is not needed.")
            self.save_button.clicked.connect(self._save_project)
            self.save_button.setEnabled(False)
            self.veusz_button = QPushButton("Open in Veusz...")
            self.veusz_button.setToolTip("Open the last saved project in Veusz (needs Veusz installed).")
            self.veusz_button.clicked.connect(self._open_in_veusz)
            self.veusz_button.setEnabled(False)
            close_button = QPushButton("Close")
            close_button.setToolTip("Close this window (unsaved projects are lost).")
            close_button.clicked.connect(self.close)
            for button in (self.process_button, self.save_button, self.veusz_button,
                           close_button):
                buttons.addWidget(button)
            outer.addLayout(buttons)

        # ----- helpers ------------------------------------------------------------
        def _log(self, message: str) -> None:
            """Append ``message`` with an HH:MM:SS time stamp to the log."""
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.append("[%s] %s" % (stamp, message))

        def _set_theme(self, mode: str) -> None:
            """Apply and save the suite theme and recolour the Plots tab."""
            self.theme = suite.apply_theme(mode)
            if self.plot_panel is not None:
                self.plot_panel.apply_theme(self.theme == "dark")
            self._log("Theme set to %s mode" % self.theme)

        def _browse_files(self) -> None:
            """Ask for .atr files and add them to the list."""
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select ATR Files", "", "GBO ATR Files (*.atr *.ATR);;All Files (*)")
            self.add_files(files)

        def add_files(self, files: Sequence[str]) -> None:
            """Add the new ``files`` (duplicates ignored) and refresh the list.
            """
            new = [f for f in files if f not in self.selected_files]
            self.selected_files.extend(new)
            self.file_list.clear()
            self.file_list.addItems([os.path.basename(f) for f in self.selected_files])
            if new:
                self._log("Added %d file(s); %d in the list" % (len(new), len(self.selected_files)))

        def _clear_files(self) -> None:
            """Empty the file list."""
            self.selected_files.clear()
            self.file_list.clear()
            self._log("File list cleared")

        def options(self) -> PlotOptions:
            """PlotOptions from the GUI (an inverted magnitude range is widened to
            1 dB above the minimum).
            """
            low, high = self.mag_min.value(), self.mag_max.value()
            if high <= low:
                high = low + 1.0
            return PlotOptions(title=self.title_edit.text().strip(),
                               mag_range=(low, high), fit_azimuth=self.fit_box.isChecked())

        # ----- processing ---------------------------------------------------------
        def _process(self) -> None:
            """Start reading the selected files in an ATRProcessingThread."""
            if not self.selected_files:
                QMessageBox.warning(self, APP_NAME, "Please select ATR files first.")
                return
            if self.thread is not None and self.thread.isRunning():
                return
            self.process_button.setEnabled(False)
            self.progress.setValue(0)
            self.progress.setVisible(True)
            workers = self.cpu_spin.value() if self.mp_box.isChecked() else 1
            self.thread = ATRProcessingThread(self.selected_files, workers)
            self.thread.progress_updated.connect(self.progress.setValue)
            self.thread.processing_finished.connect(self._finished)
            self.thread.start()
            self._log("Reading %d file(s)..." % len(self.selected_files))

        def _finished(self, patterns: List[ATRPattern], errors: List[str]) -> None:
            """Log errors and warnings, build the project from the patterns read and
            show it on the Plots tab.
            """
            self.progress.setVisible(False)
            self.process_button.setEnabled(True)
            for error in errors:
                self._log("FAILED " + error)
            for pattern in patterns:
                for warning in pattern.warnings:
                    self._log("%s: %s" % (pattern.name, warning))
            if not patterns:
                QMessageBox.warning(self, APP_NAME, "No file could be read. See the log.")
                return
            try:
                self.plotter = build_project(patterns, self.options())
            except Exception as exc:
                self._log("Project build failed: %s" % exc)
                QMessageBox.critical(self, APP_NAME, "Project build failed:\n%s" % exc)
                return
            n_traces = sum(p.frequencies_mhz.size for p in patterns)
            self._log("Project ready: %d file(s), %d trace(s), %d page(s)." % (
                len(patterns), n_traces, len(self.plotter.doc.Root.children)))
            self.save_button.setEnabled(True)
            if self.plot_panel is not None:
                try:
                    pages = self.plot_panel.set_document(self.plotter.doc)
                    self._log("Plots tab: %d page(s)." % pages)
                    self.tabs.setCurrentIndex(1)
                except Exception as exc:
                    self._log("Plots tab could not draw the project: %s" % exc)
            if errors:
                QMessageBox.warning(self, APP_NAME, "%d file(s) failed. See the log."
                                    % len(errors))

        def _save_project(self) -> None:
            """Ask for a file name, save the .vszh5 project and offer to open it in
            Veusz.
            """
            if self.plotter is None:
                QMessageBox.information(self, APP_NAME, "Process files first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save ATR Veusz Project", "",
                                                  "Veusz HDF5 Project (*.vszh5)")
            if not path:
                return
            try:
                self.last_saved = self.plotter.save(path)
            except Exception as exc:
                QMessageBox.critical(self, APP_NAME, "Failed to save the project:\n%s" % exc)
                return
            self._log("Project saved: %s" % self.last_saved)
            self.veusz_button.setEnabled(True)
            answer = QMessageBox.question(self, "Open in Veusz",
                                          "Open the saved project in Veusz?")
            if answer == QMessageBox.StandardButton.Yes:
                self._open_in_veusz()

        def _open_in_veusz(self) -> None:
            """Open the last saved project in Veusz (suite.open_in_veusz)."""
            if not self.last_saved:
                QMessageBox.information(self, APP_NAME, "Save a project first.")
                return
            if suite.open_in_veusz(self, self.last_saved):
                self._log("Opening in Veusz: %s" % self.last_saved)

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
            """Wait up to 5 s for a running read, clear the plots, then close.
            """
            if self.thread is not None and self.thread.isRunning():
                self.thread.wait(5000)
            if self.plot_panel is not None:
                self.plot_panel.clear()
            super().closeEvent(event)

    return QApplication, ATRMainWindow


# %% Main Execution
def batch(paths: Sequence[str], output: str, options: PlotOptions, workers: int) -> int:
    """Command-line conversion; returns the process exit code."""
    patterns, errors = read_files(paths, workers)
    for error in errors:
        print("FAILED", error, file=sys.stderr)
    for pattern in patterns:
        for warning in pattern.warnings:
            print("%s: %s" % (pattern.name, warning))
    if not patterns:
        return 1
    written = build_project(patterns, options).save(output)
    print("Wrote %s (%d file(s))" % (written, len(patterns)))
    return 0 if not errors else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point; returns the exit code.

    With ``--output`` the files are processed in batch (no GUI) into that
    project; otherwise the GUI opens with the files already in its list.
    """
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)
    parser.add_argument("files", nargs="*", help=".atr files (GUI: added to the list)")
    parser.add_argument("--output", help="write this .vszh5 without opening the GUI")
    parser.add_argument("--title", default=PlotOptions.title)
    parser.add_argument("--mag-range", nargs=2, type=float, default=DEFAULT_MAG_RANGE)
    parser.add_argument("--fit-azimuth", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--version", action="version", version="%s %s" % (APP_NAME, APP_VERSION))
    args = parser.parse_args(argv)
    if args.output:
        return batch(args.files, args.output,
                     PlotOptions(args.title, tuple(args.mag_range), args.fit_azimuth),
                     args.workers)
    QApplication, ATRMainWindow = _build_gui_classes()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = ATRMainWindow()
    window.add_files(args.files)
    window.show()
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()           # frozen Windows builds
    sys.exit(main())
