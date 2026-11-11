#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
CSV_TSV_AutoPlot.py

Comma / tab / custom-delimited text files -> native Veusz project (.vszh5)
and interactive pyqtgraph plots:

    * one overlay page (every selected Y column of every file against that
      file's own X column) and one page per file;
    * X column offered only when it exists in ALL files (button turns red
      otherwise), Y columns from any file (missing ones skipped and logged);
    * numeric columns, and date/time X columns (Veusz datetime axis);
    * the .vszh5 is written by the author's shared writer (vszh5_writer.py
      0.0.24 through _autoplot_native.NativeDocument) - Veusz is NOT needed
      to create the file, only (optionally) to view or export it;
    * Plots tab (pyqtgraph), light / dark theme shared with the suite.

Batch use without the GUI:
    python CSV_TSV_AutoPlot.py --x "Frequency [Hz]" --y "Magnitude [dBm]" \\
        --output out.vszh5 file1.csv file2.csv

Created on 2025 (V1-V6 by William W. Wallace's framework)

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later (GNU General Public License v3.0 or later)
Compatible with: Python 3.8+ (pandas >= 1.3, qtpy with PySide6 or PyQt5)

# %%% References
[1] W. W. Wallace, "vszh5_writer.py 0.0.24," shared Veusz HDF5 writer, 2026.
[2] pandas developers, "pandas.read_csv," pandas documentation.
    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
[3] Y. Shafranovich, "Common Format and MIME Type for Comma-Separated Values
    (CSV) Files," IETF RFC 4180, Oct. 2005.
    https://www.rfc-editor.org/rfc/rfc4180
[4] J. Sanders, "Veusz - a scientific plotting package," Veusz 4.2
    (datetime axes count seconds from 2009-01-01T00:00:00 UTC).
    https://veusz.github.io/

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 6.1.3: Exact numbers and time stamps, kept text, source_/type_ tags.
Date: 2026-11-10
#              * pandas' default C parser is not exact everywhere: pandas
#                2.0.3 (Python 3.8) and 3.0.5 read 212 971 of 400 000
#                17-digit values wrongly (up to 7 375 ulp, relative 1e-12),
#                also 15-digit numbers with a power of ten beyond +/-22.
#                A file with such a number (scan_number_forms: 16+ digits /
#                points in a row, or an exponent) is read with
#                float_precision="round_trip" (exact, slower) when the
#                installed pandas needs it (_default_parser_is_exact;
#                pandas 3.1.1 does not).  Mixed numeric columns use float()
#                there too.
#              * Cells float64 cannot hold exactly - integers beyond
#                +/-2**53, words among numbers, more digits than float64
#                keeps - keep their text in "<file>_<column>_exact_text".
#              * Date-time X: seconds from integer nanoseconds (pandas 2.0
#                total_seconds() was 1 ulp off); the stamps as written are
#                kept in "<file>_<column>_text"; ISO 8601 stamps of mixed
#                form (fraction digits, Z, +02:00) are read with
#                format="ISO8601" instead of the file being skipped.
#              * Tags source_<file> and type_<column> plus x_data / y_data
#                / time_utc / time_local / time_text / exact_text (was one
#                tag "<stem>"); traces named after their data.
#              Checked: temps.tsv identical to 6.1.2 plus the time text;
#              60 files of 50 000 x 5 load in 2.1 s on Python 3.8 (1.1 s in
#              6.1.2; the scan), 0.6 s on 3.12.
# %%%%% Function Descriptions
        _default_parser_is_exact: True when pandas' default C float parser reads _PARSER_PROBE exactly.
        scan_number_forms: (long digit run, exponent notation) found in the file (6.1.3); read in 4 MiB blocks, a run across two blocks included; (True, True) when the file c...
        CSVProcessingConfig (class): How the files are read.
        PlotConfig (class): Plot formatting.
        LoadedFile (class): One delimited file in memory.
        exact_cell_texts: {column: every cell as text} for the numeric ``columns`` whose float64 ``values`` cannot hold the cells exactly (6.1.3): integers beyond +/-2**53,...
        delimiter_from_text: GUI text -> delimiter ('\\t' typed as two characters means a tab).
        detect_delimiter: Most frequent of , TAB ; | in the first non-comment lines.
        read_file_metadata: The lines before the header line.
        load_csv_file: (LoadedFile or None, status message) of one file.
        load_files: Load files in parallel threads; (files in input order, messages).
        get_common_columns: Columns in ALL files (intersection).
        get_all_columns: Columns of ANY file (union).
        _naive_zone: Zone of time stamps written without one, when they are local time.
        column_values: ('numeric' | 'date' | 'text', values) of one column, rows kept.
        VeuszPlotter (class): Builds the project: one overlay page and one page per file.
        _build_gui_classes: Import Qt and define the GUI classes on first use.
        main: Command line entry point; returns the exit code.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP_NAME = "CSV/TSV AutoPlot"
        APP_VERSION = "6.1.3"
        APP_DESCRIPTION: one-line description (command line help, About box)
        VEUSZ_EPOCH: Veusz datetime zero (UTC)
        MIN_PARSED_FRACTION: a column is numeric / date when >= 80 % of its cells parse
        _PARSER_PROBE: numbers the inexact pandas parsers read wrongly (checked once at import)
        PANDAS_NEEDS_ROUND_TRIP: True when pandas' default C parser is not exact (pandas 2.0.3 / 3.0.5 measured; 3.1.1 is exact)
        _FORM_TABLE: byte map d = digit or point, e = e/E, s = sign, x = other (scan_number_forms)
        COLOURS: trace colours (Tableau 10, hex, the same in Veusz and the Plots tab)
# %%%%% More Info
    Run:  python CSV_TSV_AutoPlot.py [files]   (GUI; View menu: light/dark)
          python CSV_TSV_AutoPlot.py a.csv --x Time --y V --output out.vszh5
    Date-time columns are stored as Veusz date-times (UTC) and drawn in local
    ISO 8601 (VSZH5_DISPLAY_TZ overrides the computer's zone); the stamps as
    written are kept as text.  Numbers are read exactly (PANDAS_NEEDS_ROUND_TRIP).
    Every dataset carries the Veusz tags source_<file> and type_<kind>, traces
    are named after their data, and values float64 cannot hold keep their text
    (*_exact_text).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 6.1.2: Header template, docstrings and tooltips.
Date: 2026-11-09
#              * Header in the author's template: author contact lines and,
#                under this entry, Function Descriptions, Variable
#                Descriptions and More Info.
#              * Docstrings for every function, class and method.  No change
#                to what the program does.
#              * Tooltips (pop-up help) on every button, box and field that
#                had none (20 widgets, including the delimiter presets).
# %%%%% Function Descriptions
        CSVProcessingConfig (class): How the files are read.
        PlotConfig (class): Plot formatting.
        LoadedFile (class): One delimited file in memory.
        delimiter_from_text: GUI text -> delimiter ('\\t' typed as two characters means a tab).
        detect_delimiter: Most frequent of , TAB ; | in the first non-comment lines.
        read_file_metadata: The lines before the header line.
        load_csv_file: (LoadedFile or None, status message) of one file.
        load_files: Load files in parallel threads; (files in input order, messages).
        get_common_columns: Columns in ALL files (intersection).
        get_all_columns: Columns of ANY file (union).
        _naive_zone: Zone of time stamps written without one, when they are local time.
        column_values: ('numeric' | 'date' | 'text', values) of one column, rows kept.
        VeuszPlotter (class): Builds the project: one overlay page and one page per file.
        _build_gui_classes: Import Qt and define the GUI classes on first use.
        main: Command line entry point; returns the exit code.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP_NAME = "CSV/TSV AutoPlot"
        APP_VERSION = "6.1.2"
        APP_DESCRIPTION: one-line description (command line help, About box)
        VEUSZ_EPOCH: Veusz datetime zero (UTC)
        MIN_PARSED_FRACTION: a column is numeric / date when >= 80 % of its cells parse
        COLOURS: 4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
# %%%%% More Info
    Run:  python CSV_TSV_AutoPlot.py [files]   (GUI; View menu: light/dark)
          python CSV_TSV_AutoPlot.py a.csv --x Time --y V --output out.vszh5
    Date-time columns are stored as Veusz date-times (UTC) and drawn in local
    ISO 8601 (VSZH5_DISPLAY_TZ overrides the computer's zone).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 6.1.1: Date-times stored as Veusz date-times, shown in local ISO 8601.
Date: 2026-11-09
#              * A date/time X column is stored as a Veusz DATE dataset in
#                UTC ("<file>_<column>") plus "<file>_<column>_local" on the
#                clock of the computer doing the processing (DST-aware;
#                VSZH5_DISPLAY_TZ overrides).  The pages draw the local one;
#                the ticks read 2026-11-09T08:39:00-05:00 (ISO 8601, 24 h,
#                with the UTC offset) and the axis label names the offset.
#              * Time stamps with an offset or "Z" are exact; time stamps
#                without one are read as UTC (default) or as local time of
#                this computer ("Times without a zone" option, --naive-time).
#              * The source-file lines before the header stay in the notes
#                of every page; the overlay page lists every file.
#              * Writer 0.0.25 / _autoplot_native 0.3.0.
# %%%% 6.1.0: Native writer, pyqtgraph Plots tab, suite theme, data fixes.
Date: 2026-11-05
#              * .vszh5 written by vszh5_writer 0.0.24 through
#                _autoplot_native.NativeDocument (no veusz.embed window per
#                "Generate"; Veusz is optional, for viewing / PDF export).
#              * Plots tab (pyqtgraph) with the same pages, legend values at
#                the cursor; View > Light / Dark (suite theme); Help > About.
#              * X and Y were each passed through dropna() separately, so a
#                blank cell in one column shifted every later point of that
#                trace against X.  Now both keep their rows; non-numeric
#                cells become gaps (NaN) and rows stay aligned.
#              * Text columns were sent to Veusz as strings (save failure or
#                garbage); columns are now classified numeric / date-time /
#                text.  A date-time X column gives a Veusz datetime axis;
#                text columns are skipped with a log line.
#              * Multi-character delimiters were read as regular expressions
#                by pandas; they are now literal ("\\s+" and "\\t" keep their
#                usual meaning).  Undecodable bytes are replaced instead of
#                stopping the load.  Files are loaded in parallel threads.
#              * Files with the same name in different folders no longer
#                overwrite each other (keys are full paths, dataset names
#                are made unique).
#              * Log X / Log Y options; trace colours are hex, so Veusz and
#                the Plots tab show the same colours.
#              * The optional log file is written next to the saved project
#                (it used to land in the current working directory).
# %%%% 6.0.0: V6 (2026-02-03) - separate pages per file, X-column
Date: 2026-02-03
#              intersection / Y-column union, red/green Generate button,
#              logging, Save to Veusz.
=============================================================================
"""
# %% Import Modules
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("QT_API", "pyside6")   # before qtpy is imported

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from vszh5_writer import header_notes  # noqa: E402
# 6.1.3: tags source_<file> / type_<kind>, trace names from their data and the
# exact-text check (Qt-free helpers shared by the AutoPlot programs)
from _autoplot_native import (float_loses_text, tag_data,  # noqa: E402
                              trace_widget_name)

APP_NAME = "CSV/TSV AutoPlot"
APP_VERSION = "6.1.3"
__version__ = APP_VERSION
APP_DESCRIPTION = ("Delimited text files (CSV, TSV, custom) to native Veusz .vszh5 "
                   "projects, with interactive pyqtgraph plots.")

VEUSZ_EPOCH = pd.Timestamp("2009-01-01 00:00:00")   # Veusz datetime zero (UTC)
MIN_PARSED_FRACTION = 0.8      # a column is numeric / date when >= 80 % of its cells parse
#: 6.1.3: the default C parser of pandas 2.x and 3.0 is not correctly rounded
#: for every number: it gathers at most 17 digits into an integer and then
#: multiplies or divides by a power of ten, so a number with more than 15
#: significant digits, or whose power of ten (the written exponent minus the
#: digits after the point) is beyond +/-22, can come out wrong - measured
#: with pandas 2.0.3 and 3.0.5: 212 971 of 400 000 17-digit values, up to
#: 7 375 ulp (relative 1e-12).  float_precision="round_trip" is exact but
#: about 7 times slower (60 files of 50 000 x 5: 7.6 s instead of 1.1 s), so
#: it is used only for a file that has such a number (scan_number_forms).
#: pandas 3.1 is exact by default and deprecates the option; the check below
#: runs the default parser on numbers it got wrong and decides from that.
_PARSER_PROBE = ("0.12345678901234567", "1.2345678901234567e-17", "-0.0005227484414807474",
                 "1.234567e-30", "9.87654321098765e-10")


def _default_parser_is_exact() -> bool:
    """True when pandas' default C float parser reads _PARSER_PROBE exactly."""
    import io
    try:
        frame = pd.read_csv(io.StringIO("x\n" + "\n".join(_PARSER_PROBE)), engine="c")
        return [float(v) for v in frame["x"].tolist()] == [float(t) for t in _PARSER_PROBE]
    except Exception:
        return False


PANDAS_NEEDS_ROUND_TRIP = not _default_parser_is_exact()

#: 6.1.3: the file bytes are mapped (bytes.translate, about 0.4 GB/s) to
#: d = digit or point, e = e/E, s = sign, x = anything else; then a run of 16
#: d (a number with 15 or more digits, which float64 may not hold and the
#: fast parser may round wrongly) and an exponent ("e" + digit, the form
#: whose power of ten can pass +/-22) are plain substring searches.  A
#: number of 15 characters or fewer without an exponent has at most 15
#: digits and a power of ten within +/-15: the fast parser is exact for it.
_FORM_TABLE = bytes((ord("d") if (48 <= i <= 57 or i == 46) else
                     ord("e") if i in (69, 101) else
                     ord("s") if i in (43, 45) else ord("x")) for i in range(256))


def scan_number_forms(file_path: str) -> Tuple[bool, bool]:
    """(long digit run, exponent notation) found in the file (6.1.3); read in
    4 MiB blocks, a run across two blocks included; (True, True) when the
    file cannot be read (check everything)."""
    long_run = exponent = False
    tail = b""
    try:
        with open(file_path, "rb") as handle:
            while not (long_run and exponent):
                block = handle.read(1 << 22)
                if not block:
                    break
                forms = (tail + block).translate(_FORM_TABLE)
                long_run = long_run or b"dddddddddddddddd" in forms
                exponent = exponent or b"ed" in forms or b"esd" in forms
                tail = block[-15:]
    except OSError:
        return True, True
    return long_run, exponent
#: Trace colours (Tableau 10, as the Plots tab) - hex so Veusz shows the same colours.
COLOURS = ("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac")


# %% Configuration
@dataclass
class CSVProcessingConfig:
    """How the files are read."""
    delimiter: str = ","
    header_line: int = 0          # 0-indexed line holding the column names
    encoding: str = "utf-8"
    skip_empty_lines: bool = True
    workers: int = max(1, min(8, os.cpu_count() or 1))


@dataclass
class PlotConfig:
    """Plot formatting."""
    title: str = "Data Plot"
    x_label: str = "X Axis"
    y_label: str = "Y Axis"
    x_scale: str = "linear"       # 'linear' or 'log'
    y_scale: str = "linear"
    show_grid: bool = True
    show_legend: bool = True
    line_style: str = "-"         # matplotlib-style notation
    line_width: float = 2.0
    marker_style: str = "o"
    marker_size: float = 6.0
    naive_time: str = "utc"       # 6.1.1: date-times without a zone: "utc" or "local"


@dataclass
class LoadedFile:
    """One delimited file in memory."""
    path: str
    frame: pd.DataFrame
    metadata: List[str] = field(default_factory=list)
    #: 6.1.3: pandas.read_csv() keywords the frame was read with (to read
    #: columns again as text for the exact-text check)
    read_options: Dict[str, Any] = field(default_factory=dict)
    #: 6.1.3: scan_number_forms() of the file, once known
    number_forms: Optional[Tuple[bool, bool]] = None

    @property
    def name(self) -> str:
        """File name (no folder)."""
        return os.path.basename(self.path)

    def may_have_long_numbers(self) -> bool:
        """True when the file has a run of 16+ digits / points (6.1.3), so a
        number in it may have more digits than float64 keeps."""
        if self.number_forms is None:
            self.number_forms = scan_number_forms(self.path)
        return self.number_forms[0]

    def column_texts(self, columns: Sequence[str]) -> Dict[str, np.ndarray]:
        """Cells of ``columns`` as written (6.1.3): the file read again with
        every cell as text, only those columns; "" for an empty cell."""
        names = list(self.frame.columns)
        index = [names.index(c) for c in columns]
        options = dict(self.read_options)
        options.pop("float_precision", None)
        frame = pd.read_csv(self.path, usecols=index, dtype=str, keep_default_na=False,
                            **options)
        return {names[i]: frame.iloc[:, k].to_numpy(dtype=str)
                for k, i in enumerate(sorted(index))}


def exact_cell_texts(loaded: LoadedFile, columns: Sequence[str],
                     values: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    """{column: every cell as text} for the numeric ``columns`` whose float64
    ``values`` cannot hold the cells exactly (6.1.3): integers beyond
    +/-2**53, cells that are not numbers (kept as gaps in the plots) and
    numbers with more digits than float64 keeps.  Other columns are left out.
    """
    out: Dict[str, List[str]] = {}
    long_check: List[str] = []
    for column in columns:
        series = loaded.frame[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_integer_dtype(series):
            present = series.dropna().to_numpy()
            if present.size and (np.any(present > 2 ** 53) or
                                 (present.dtype.kind == "i" and np.any(present < -2 ** 53))):
                out[column] = ["" if pd.isna(v) else str(int(v)) for v in series.tolist()]
            continue
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            # text cells (pandas 3: "str" dtype): the frame already has them
            texts = ["" if (not isinstance(v, str) and pd.isna(v)) else str(v)
                     for v in series.tolist()]
            if float_loses_text(texts, values[column]):
                out[column] = texts
            continue
        long_check.append(column)
    if long_check and loaded.may_have_long_numbers():
        for column, texts in loaded.column_texts(long_check).items():
            if float_loses_text(texts, values[column]):
                out[column] = texts.tolist()
    return out


# %% Reading
def delimiter_from_text(text: str) -> str:
    """GUI text -> delimiter ('\\t' typed as two characters means a tab)."""
    if text in ("\\t", "tab", "TAB"):
        return "\t"
    return text or ","


def detect_delimiter(file_path: str, sample_size: int = 20) -> str:
    """Most frequent of , TAB ; | in the first non-comment lines."""
    counts = {d: 0 for d in (",", "\t", ";", "|")}
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            for _ in range(sample_size):
                line = handle.readline()
                if not line:
                    break
                if not line.strip() or line.startswith("#"):
                    continue
                for delimiter in counts:
                    counts[delimiter] += line.count(delimiter)
    except OSError:
        return ","
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_file_metadata(file_path: str, header_line: int) -> List[str]:
    """The lines before the header line."""
    lines: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            for _ in range(header_line):
                line = handle.readline()
                if not line:
                    break
                lines.append(line.rstrip("\r\n"))
    except OSError:
        pass
    return lines


def load_csv_file(file_path: str, config: Optional[CSVProcessingConfig] = None
                  ) -> Tuple[Optional[LoadedFile], str]:
    """(LoadedFile or None, status message) of one file."""
    config = config or CSVProcessingConfig()
    delimiter = config.delimiter
    kwargs: Dict[str, Any] = {}
    if len(delimiter) == 1:
        kwargs["engine"] = "c"                 # fast; releases the GIL (thread pool)
    elif delimiter == r"\s+":
        kwargs["engine"] = "python"
    else:                                      # pandas treats longer separators as regex
        delimiter = re.escape(delimiter)
        kwargs["engine"] = "python"
    options = dict(sep=delimiter, encoding=config.encoding, encoding_errors="replace",
                   skiprows=range(config.header_line) if config.header_line else None,
                   skip_blank_lines=config.skip_empty_lines, **kwargs)
    forms = None
    try:
        if kwargs["engine"] == "c" and PANDAS_NEEDS_ROUND_TRIP:
            # 6.1.3: the exact (slower) parser only for a file that has a
            # number the fast one may read wrongly (scan_number_forms)
            forms = scan_number_forms(file_path)
            if any(forms):
                options["float_precision"] = "round_trip"
        frame = pd.read_csv(file_path, **options)
    except Exception as exc:
        return None, "Error loading %s: %s" % (file_path, exc)
    frame.columns = [str(c).strip() for c in frame.columns]
    loaded = LoadedFile(file_path, frame, read_file_metadata(file_path, config.header_line),
                        read_options=options, number_forms=forms)
    return loaded, ("Loaded %s: %d rows, %d columns (header at line %d)"
                    % (loaded.name, frame.shape[0], frame.shape[1], config.header_line + 1))


def load_files(paths: Sequence[str], config: CSVProcessingConfig
               ) -> Tuple[List[LoadedFile], List[str]]:
    """Load files in parallel threads; (files in input order, messages)."""
    paths = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        results = list(pool.map(lambda p: load_csv_file(p, config), paths))
    files = [loaded for loaded, _msg in results if loaded is not None]
    return files, [msg for _loaded, msg in results]


def get_common_columns(files: Sequence[LoadedFile]) -> Set[str]:
    """Columns in ALL files (intersection)."""
    if not files:
        return set()
    common = set(files[0].frame.columns)
    for loaded in files[1:]:
        common &= set(loaded.frame.columns)
    return common


def get_all_columns(files: Sequence[LoadedFile]) -> Set[str]:
    """Columns of ANY file (union)."""
    columns: Set[str] = set()
    for loaded in files:
        columns |= set(loaded.frame.columns)
    return columns


def _naive_zone():
    """Zone of time stamps written without one, when they are local time."""
    import vszh5_writer
    zone = vszh5_writer.display_zone()
    if zone is None:
        from dateutil import tz           # a pandas dependency
        zone = tz.tzlocal()
    return zone


def column_values(series: pd.Series, naive_time: str = "utc") -> Tuple[str, np.ndarray]:
    """
    ('numeric' | 'date' | 'text', values) of one column, rows kept.

    numeric : float64, cells that do not parse -> NaN
    date    : seconds since 2009-01-01 UTC (Veusz datetime), NaN where missing.
              Stamps with an offset or "Z" are exact; stamps without one are
              UTC (``naive_time="utc"``) or this computer's local time
              ("local"; VSZH5_DISPLAY_TZ overrides the zone).
    text    : the original values (not plotted)
    """
    non_empty = series.dropna()
    if series.dtype == object:
        non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    n_cells = len(non_empty)
    if pd.api.types.is_bool_dtype(series):
        return "numeric", series.astype(float).to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", series.to_numpy(dtype=float, na_value=np.nan)
    numbers = pd.to_numeric(series, errors="coerce")
    if n_cells and numbers.notna().sum() >= MIN_PARSED_FRACTION * n_cells:
        values = numbers.to_numpy(dtype=float, na_value=np.nan)
        if PANDAS_NEEDS_ROUND_TRIP:
            # 6.1.3: where pandas' own parser is not exact (pandas before
            # 3.1, PANDAS_NEEDS_ROUND_TRIP), to_numeric is not either;
            # Python's float() is.  Cells float() rejects keep to_numeric.
            ok = np.flatnonzero(numbers.notna().to_numpy())
            cells = series.to_numpy()
            for i in ok.tolist():
                try:
                    values[i] = float(str(cells[i]).strip())
                except ValueError:
                    pass
        return "numeric", values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # "could not infer format" etc.
        stamps = pd.to_datetime(series, errors="coerce", utc=True)
        if naive_time == "local":
            plain = pd.to_datetime(series, errors="coerce")
            if pd.api.types.is_datetime64_dtype(plain) and \
                    not isinstance(plain.dtype, pd.DatetimeTZDtype):
                stamps = plain.dt.tz_localize(_naive_zone(), ambiguous="NaT",
                                              nonexistent="NaT").dt.tz_convert("UTC")
        if n_cells and stamps.notna().sum() < MIN_PARSED_FRACTION * n_cells \
                and naive_time != "local":
            # 6.1.3: ISO 8601 stamps that differ in form from row to row
            # (fraction digits, "Z" or "+02:00") failed the format inferred
            # from the first row and the whole file was skipped; pandas 2+
            # reads every ISO 8601 form with format="ISO8601".
            try:
                iso = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
                if iso.notna().sum() > stamps.notna().sum():
                    stamps = iso
            except (TypeError, ValueError):
                pass
    if n_cells and stamps.notna().sum() >= MIN_PARSED_FRACTION * n_cells:
        stamps = stamps.dt.tz_convert(None)    # UTC, naive
        # 6.1.3: seconds from whole integer nanoseconds, split into whole
        # seconds and the rest so each part is exact before the one final
        # float64 rounding; pandas 2.0 total_seconds() was 1 ulp off for
        # some nanosecond stamps (the text is also kept, see the project).
        ns = (stamps - VEUSZ_EPOCH).to_numpy(dtype="timedelta64[ns]")
        missing = np.isnat(ns)
        whole, rest = np.divmod(ns.astype(np.int64), 10 ** 9)
        seconds = whole.astype(float) + rest.astype(float) / 1e9
        seconds[missing] = np.nan
        return "date", seconds
    return "text", series.to_numpy()


# %% Veusz project (NativeDocument; written by vszh5_writer)
class VeuszPlotter:
    """Builds the project: one overlay page and one page per file."""

    LINE_STYLE_MAP = {"-": "solid", "--": "dashed", "-.": "dash-dot", ":": "dotted",
                      "solid": "solid", "dashed": "dashed", "dashdot": "dash-dot",
                      "dotted": "dotted"}
    MARKER_STYLE_MAP = {"o": "circle", "s": "square", "^": "triangle",
                        "v": "triangledown", "d": "diamond", "+": "plus", "x": "cross",
                        "*": "star", "circle": "circle", "square": "square",
                        "triangle": "triangle", "diamond": "diamond", "none": "none"}

    def __init__(self, plot_title: str = "CSV Data Plot") -> None:
        """Empty CSV/TSV project titled ``plot_title``; ``messages`` collects the
        time-stamped log lines of the build.
        """
        from _autoplot_native import NativeDocument
        self.plot_title = plot_title
        self.doc = NativeDocument("CSV/TSV AutoPlot")
        self.messages: List[str] = []

    def log_message(self, message: str) -> None:
        """Add ``message`` with an HH:MM:SS time stamp to ``messages``."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append("[%s] %s" % (stamp, message))

    @staticmethod
    def _unique(stem: str, used: Set[str]) -> str:
        """``stem``, or ``stem_2``, ``stem_3``, ... - the first name not in ``used``.
        """
        candidate, index = stem, 2
        while candidate in used:
            candidate = "%s_%d" % (stem, index)
            index += 1
        used.add(candidate)
        return candidate

    def create_overlay_and_individual_plots(self, files: Sequence[LoadedFile], x_column: str,
                                            y_columns: Sequence[str],
                                            plot_config: Optional[PlotConfig] = None) -> int:
        """Build every page; returns the number of traces."""
        config = plot_config or PlotConfig()
        doc = self.doc
        notes = ["File Metadata:", "=" * 60, ""]
        for loaded in files:
            notes += ["File: %s" % loaded.path, "-" * 60]
            notes += loaded.metadata or ["(No metadata lines before header)"]
            notes.append("")
        notes_text = "\n".join(notes)
        line_style = self.LINE_STYLE_MAP.get(config.line_style, "solid")
        marker = self.MARKER_STYLE_MAP.get(config.marker_style, "circle")

        overlay_page = doc.Root.Add("page", name="OverlayPlot")
        overlay_page.notes.val = notes_text
        overlay = overlay_page.Add("graph", name="OverlayGraph")
        overlay_kind: Optional[str] = None
        used_stems: Set[str] = set()
        per_file: List[Tuple[LoadedFile, str, str, str, List[Tuple[str, str]]]] = []
        n_traces = 0
        for loaded in files:
            if x_column not in loaded.frame.columns:
                self.log_message("WARNING: X column '%s' not found in %s" % (x_column, loaded.name))
                continue
            stem = self._unique(os.path.splitext(loaded.name)[0], used_stems)
            x_kind, x_values = column_values(loaded.frame[x_column], config.naive_time)
            if x_kind == "text":
                self.log_message("WARNING: X column '%s' of %s is text, not numbers or "
                                 "dates - file skipped" % (x_column, loaded.name))
                continue
            x_name = "%s_%s" % (stem, x_column)
            source = os.path.splitext(loaded.name)[0]          # 6.1.3: source_<file>
            if x_kind == "date":
                doc.SetDataDateTime(x_name, x_values)          # UTC
                tagged_local = doc.SetDataLocalTime(x_name)    # local display copy
                tag_data(doc, [x_name], source, (x_column, "x_data", "time_utc"))
                tag_data(doc, [tagged_local], source, (x_column, "x_data", "time_local"))
                # 6.1.3: the time stamps as written (offsets, every digit of
                # the seconds) - Veusz date-times are float64 seconds.
                texts = ["" if (not isinstance(v, str) and pd.isna(v)) else str(v)
                         for v in loaded.frame[x_column].tolist()]
                doc.SetDataText(x_name + "_text", texts)
                tag_data(doc, [x_name + "_text"], source, (x_column, "time_text", "text"))
            else:
                doc.SetData(x_name, x_values)
                tag_data(doc, [x_name], source, (x_column, "x_data"))
                exact = exact_cell_texts(loaded, [x_column], {x_column: x_values})
                if x_column in exact:
                    doc.SetDataText(x_name + "_exact_text", exact[x_column])
                    tag_data(doc, [x_name + "_exact_text"], source,
                             (x_column, "exact_text", "text"))
            traces: List[Tuple[str, str]] = []
            y_values_of: Dict[str, np.ndarray] = {}
            for y_col in y_columns:
                if y_col not in loaded.frame.columns:
                    self.log_message("WARNING: Y column '%s' not found in %s - skipping"
                                     % (y_col, loaded.name))
                    continue
                y_kind, y_values = column_values(loaded.frame[y_col])
                if y_kind != "numeric":
                    self.log_message("WARNING: Y column '%s' of %s is %s, not numbers - "
                                     "skipping" % (y_col, loaded.name, y_kind))
                    continue
                gaps = int(np.isnan(y_values).sum())
                if gaps:
                    self.log_message("%s: '%s' has %d empty or non-numeric cell(s), drawn "
                                     "as gaps" % (loaded.name, y_col, gaps))
                y_name = "%s_%s" % (stem, y_col)
                doc.SetData(y_name, y_values)
                tag_data(doc, [y_name], source, (y_col, "y_data"))
                traces.append((y_col, y_name))
                y_values_of[y_col] = y_values
            # 6.1.3: cells float64 cannot hold exactly keep their text
            exact = exact_cell_texts(loaded, list(y_values_of), y_values_of)
            for y_col, y_name in traces:
                if y_col in exact:
                    doc.SetDataText(y_name + "_exact_text", exact[y_col])
                    tag_data(doc, [y_name + "_exact_text"], source,
                             (y_col, "exact_text", "text"))
                    self.log_message("%s: '%s' has cells float64 cannot hold exactly; "
                                     "their text is kept in %s_exact_text"
                                     % (loaded.name, y_col, y_name))
            per_file.append((loaded, stem, x_kind, x_name, traces))
            if overlay_kind is None:
                overlay_kind = x_kind
            for y_col, y_name in traces:
                if x_kind != overlay_kind:
                    self.log_message("WARNING: %s has a %s X column; the overlay uses %s X "
                                     "values - not overlaid" % (loaded.name, x_kind, overlay_kind))
                    break
                self._style(overlay.Add("xy", name=trace_widget_name(y_name), xData=x_name,
                                        yData=y_name), COLOURS[n_traces % len(COLOURS)],
                            line_style, marker, config, "%s_%s" % (stem, y_col))
                n_traces += 1
        self._configure_graph(overlay, "Overlay: %s" % config.title, config,
                              overlay_kind == "date")
        for loaded, stem, x_kind, x_name, traces in per_file:
            page = doc.Root.Add("page", name="Page_%s" % stem)
            page.notes.val = header_notes(
                loaded.metadata or ["(No metadata lines before header)"], source=loaded.path,
                extra=["X: %s; Y: %s" % (x_column, ", ".join(t[0] for t in traces))])
            graph = page.Add("graph", name="Graph_%s" % stem)
            for index, (y_col, y_name) in enumerate(traces):
                self._style(graph.Add("xy", name=trace_widget_name(y_name), xData=x_name,
                                      yData=y_name),
                            COLOURS[index % len(COLOURS)], line_style, marker, config, y_col)
            self._configure_graph(graph, stem, config, x_kind == "date")
        self.log_message("Created %d pages: 1 overlay + %d file pages, %d overlay trace(s)"
                         % (1 + len(per_file), len(per_file), n_traces))
        return n_traces

    @staticmethod
    def _style(xy: Any, colour: str, line_style: str, marker: str, config: PlotConfig,
               key: str) -> None:
        """Style one trace: line colour/style/width, marker and size from
        ``config`` and the legend ``key``.
        """
        xy.PlotLine.color.val = colour
        xy.PlotLine.style.val = line_style
        xy.PlotLine.width.val = "%gpt" % config.line_width
        xy.marker.val = marker
        xy.markerSize.val = "%gpt" % config.marker_size
        xy.MarkerFill.color.val = colour
        xy.key.val = key

    @staticmethod
    def _configure_graph(graph: Any, title: str, config: PlotConfig, date_x: bool) -> None:
        """Title, axis labels, log axes and, for ``date_x``, a local ISO 8601
        date-time x axis.
        """
        graph.Add("label", name="plotTitle")
        graph.plotTitle.Text.size.val = "12pt"
        graph.plotTitle.label.val = title
        graph.plotTitle.alignHorz.val = "centre"
        graph.plotTitle.yPos.val = 1.05
        graph.plotTitle.xPos.val = 0.5
        graph.topMargin.val = "1cm"
        graph.x.label.val = config.x_label
        graph.y.label.val = config.y_label
        if date_x:
            from _autoplot_native import mark_local_time_axis
            default_label = config.x_label in ("", PlotConfig.x_label)
            mark_local_time_axis(graph.x, "Local time" if default_label else config.x_label)
            graph.x.TickLabels.rotate.val = "45"
        else:
            graph.x.log.val = config.x_scale == "log"
        graph.y.log.val = config.y_scale == "log"
        graph.x.GridLines.hide.val = not config.show_grid
        graph.y.GridLines.hide.val = not config.show_grid
        if config.show_legend:
            key = graph.Add("key")
            key.horzPosn.val = "right"
            key.vertPosn.val = "top"

    def save_project(self, file_path: str, log_file: bool = False) -> str:
        """Write the .vszh5 (atomic); optionally the log next to it. Returns the path."""
        written = self.doc.save(file_path)
        self.log_message("Saved Veusz project: %s" % written)
        if log_file:
            log_path = os.path.splitext(written)[0] + "_log.txt"
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("%s %s log - %s\n%s\n\n" % (
                    APP_NAME, APP_VERSION,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "=" * 60))
                handle.write("\n".join(self.messages) + "\n")
        return written


# %% GUI (imported lazily so --output works without Qt)
def _build_gui_classes():
    """Import Qt and define the GUI classes on first use.

    Keeps the batch mode (``--output``) free of Qt; returns (QApplication,
    CSVAutoPlotMainWindow).
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                                QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                                QLineEdit, QListWidget, QMainWindow, QMessageBox,
                                QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                                QTabWidget, QTextEdit, QVBoxLayout, QWidget)
    import _suite_gui as suite

    class CSVAutoPlotMainWindow(QMainWindow):
        """Processing and Plots tabs; File / View / Help menus."""

        def __init__(self) -> None:
            """Main window: Processing and Plots tabs, menus with the light/dark
            theme, and the suite theme applied.
            """
            super().__init__()
            self.setWindowTitle("%s %s" % (APP_NAME, APP_VERSION))
            self.resize(1400, 950)
            self.selected_files: List[str] = []
            self.loaded: Dict[str, LoadedFile] = {}       # full path -> file
            self.csv_config = CSVProcessingConfig()
            self.plot_config = PlotConfig()
            self.plotter: Optional[VeuszPlotter] = None
            self.last_saved: Optional[str] = None
            self.theme = suite.apply_theme(suite.saved_theme(), persist=False)
            suite.add_file_menu(self, [
                ("&Add Files...", self.browse_files, "Ctrl+O"),
                ("&Clear Files", self.clear_files, ""),
                ("&Generate Plots", self.generate_plots, "F5"),
                ("&Save Veusz Project (.vszh5)...", self.save_to_veusz, "Ctrl+S"),
                ("Open Last Project in &Veusz", self.open_in_veusz, ""),
                ("&Export Last Project to PDF / PNG (Veusz)", self.export_pages, ""),
            ])
            self.act_light, self.act_dark = suite.add_view_menu(self, self._set_theme, self.theme)
            suite.add_help_menu(self, APP_NAME, APP_VERSION, APP_DESCRIPTION, ("pandas",))
            self.setup_ui()
            self.log_message("%s %s ready; theme %s." % (APP_NAME, APP_VERSION, self.theme))

        # ----- layout -------------------------------------------------------------
        def setup_ui(self) -> None:
            """Lay out the Processing tab (files, delimiter, header line, columns,
            preview, plot options, log, buttons) and the Plots tab.
            """
            central = QWidget()
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)
            self.tabs = QTabWidget()
            outer.addWidget(self.tabs, 1)
            processing = QWidget()
            self.tabs.addTab(processing, "Processing")
            main_layout = QVBoxLayout(processing)

            file_group = QGroupBox("File Selection and Delimiter Configuration")
            file_layout = QHBoxLayout(file_group)
            left = QVBoxLayout()
            row = QHBoxLayout()
            for text, slot, tip in (
                    ("Browse Files", self.browse_files,
                     "Add delimited text files (.csv, .tsv, .txt, .dat) and load them."),
                    ("Clear Files", self.clear_files,
                     "Remove every file from the list (the files are not touched).")):
                button = QPushButton(text)
                button.setToolTip(tip)
                button.clicked.connect(slot)
                row.addWidget(button)
            row.addStretch()
            left.addLayout(row)
            self.file_list_widget = QListWidget()
            self.file_list_widget.setToolTip("Loaded files; select one to preview its first rows.")
            self.file_list_widget.itemSelectionChanged.connect(self.on_file_selection_changed)
            left.addWidget(QLabel("Loaded Files:"))
            left.addWidget(self.file_list_widget)
            right = QVBoxLayout()
            delimiter_group = QGroupBox("Delimiter Configuration")
            form = QFormLayout(delimiter_group)
            detect = QPushButton("Auto-Detect Delimiter")
            detect.setToolTip(
                "Guess the delimiter from the selected (or first) file and "
                "reload.")
            detect.clicked.connect(self.auto_detect_delimiter)
            form.addRow("Auto-Detection:", detect)
            self.delimiter_input = QLineEdit(",")
            self.delimiter_input.setMaximumWidth(100)
            self.delimiter_input.setToolTip("Literal text; \\t = tab, \\s+ = any whitespace")
            self.delimiter_input.editingFinished.connect(self.reload_files)
            form.addRow("Delimiter:", self.delimiter_input)
            self.header_spin = QSpinBox()
            self.header_spin.setRange(1, 100000)
            self.header_spin.setMaximumWidth(100)
            self.header_spin.setToolTip("Line number holding the column names (1 = first line)")
            self.header_spin.valueChanged.connect(lambda _v: self.reload_files())
            form.addRow("Header Line Number:", self.header_spin)
            self.logging_box = QCheckBox("Write a log file next to the saved project")
            self.logging_box.setToolTip(("Also write this session's log as a text file next to the "
                'saved project.'))
            self.logging_box.setChecked(True)
            form.addRow("", self.logging_box)
            right.addWidget(delimiter_group)
            presets = QGroupBox("Delimiter Presets")
            preset_row = QHBoxLayout(presets)
            for name, delimiter in (("Comma", ","), ("Tab", "\\t"), ("Semicolon", ";"),
                                    ("Pipe", "|"), ("Spaces", "\\s+")):
                button = QPushButton(name)
                button.setToolTip("Use %s as the delimiter (%r) and reload the files."
                                  % (name.lower(), delimiter))
                button.clicked.connect(lambda _c=False, d=delimiter: self.set_delimiter(d))
                preset_row.addWidget(button)
            right.addWidget(presets)
            right.addStretch()
            file_layout.addLayout(left, 1)
            file_layout.addLayout(right, 0)
            main_layout.addWidget(file_group)

            preview_group = QGroupBox("Data Preview (first 10 rows)")
            preview_layout = QVBoxLayout(preview_group)
            self.data_preview_table = QTableWidget()
            self.data_preview_table.setToolTip(
                "First 10 rows of the selected file, as read with the current "
                "settings.")
            self.data_preview_table.setMaximumHeight(200)
            preview_layout.addWidget(self.data_preview_table)
            main_layout.addWidget(preview_group)

            column_group = QGroupBox("Column Selection for Plotting")
            column_layout = QVBoxLayout(column_group)
            selection = QFormLayout()
            self.x_column_combo = QComboBox()
            self.x_column_combo.setToolTip(
                "Column used as x for every file; it must exist in all loaded "
                "files.")
            self.x_column_combo.currentTextChanged.connect(lambda _t: self.validate_x_axis_selection())
            selection.addRow("X-Axis Column (common to all files):", self.x_column_combo)
            self.x_axis_warning_label = QLabel("")
            self.x_axis_warning_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.x_axis_warning_label.setWordWrap(True)
            selection.addRow("", self.x_axis_warning_label)
            column_layout.addLayout(selection)
            column_layout.addWidget(QLabel("Y-Axis Columns (select one or more, from any file):"))
            self.y_columns_list = QListWidget()
            self.y_columns_list.setToolTip("Columns to plot against x (select one or more).")
            self.y_columns_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            self.y_columns_list.setMaximumHeight(120)
            column_layout.addWidget(self.y_columns_list)
            main_layout.addWidget(column_group)

            plot_group = QGroupBox("Plot Configuration")
            plot_form = QFormLayout(plot_group)
            self.title_input = QLineEdit(self.plot_config.title)
            self.title_input.setToolTip("Title written on every page.")
            plot_form.addRow("Plot Title:", self.title_input)
            self.x_label_input = QLineEdit(self.plot_config.x_label)
            self.x_label_input.setToolTip(("Label of the x axis (a date-time x axis gets 'Local time' "
                'when left at the default).'))
            plot_form.addRow("X-Axis Label:", self.x_label_input)
            self.y_label_input = QLineEdit(self.plot_config.y_label)
            self.y_label_input.setToolTip("Label of the y axis.")
            plot_form.addRow("Y-Axis Label:", self.y_label_input)
            toggles = QHBoxLayout()
            self.show_legend_checkbox = QCheckBox("Show Legend")
            self.show_legend_checkbox.setToolTip("Show a legend (key) on each graph.")
            self.show_legend_checkbox.setChecked(True)
            self.log_x_box = QCheckBox("Log X")
            self.log_x_box.setToolTip("Logarithmic x axis (not used for date-time x).")
            self.log_y_box = QCheckBox("Log Y")
            self.log_y_box.setToolTip("Logarithmic y axis.")
            self.grid_box = QCheckBox("Grid lines")
            self.grid_box.setToolTip("Draw grid lines on both axes.")
            self.grid_box.setChecked(True)
            self.naive_combo = QComboBox()
            self.naive_combo.addItems(["Times without a zone are UTC",
                                       "Times without a zone are local time"])
            self.naive_combo.setToolTip("Date/time X columns are stored as Veusz date-times "
                                        "(UTC) and shown in this computer's local time "
                                        "(ISO 8601 with the UTC offset).")
            for box in (self.show_legend_checkbox, self.log_x_box, self.log_y_box, self.grid_box,
                        self.naive_combo):
                toggles.addWidget(box)
            toggles.addStretch()
            plot_form.addRow(toggles)
            main_layout.addWidget(plot_group)

            self.status_text = QTextEdit()
            self.status_text.setToolTip("Progress and messages of this session, with time stamps.")
            self.status_text.setReadOnly(True)
            self.status_text.setMaximumHeight(130)
            main_layout.addWidget(self.status_text)

            widget, self.plot_panel = suite.make_plot_panel(self.theme == "dark")
            self.tabs.addTab(widget, "Plots")

            buttons = QHBoxLayout()
            self.generate_btn = QPushButton("Generate Plots")
            self.generate_btn.setToolTip(
                "Build the pages from the loaded files and chosen columns "
                "(shown on the Plots tab).")
            self.generate_btn.clicked.connect(self.generate_plots)
            self.save_btn = QPushButton("Save Veusz Project (.vszh5)")
            self.save_btn.setToolTip(
                "Write the project as a native Veusz HDF5 file (.vszh5); Veusz "
                "is not needed.")
            self.save_btn.clicked.connect(self.save_to_veusz)
            self.save_btn.setEnabled(False)
            self.veusz_btn = QPushButton("Open in Veusz...")
            self.veusz_btn.setToolTip("Open the last saved project in Veusz (needs Veusz installed).")
            self.veusz_btn.clicked.connect(self.open_in_veusz)
            self.veusz_btn.setEnabled(False)
            close = QPushButton("Close")
            close.setToolTip("Close this window (unsaved projects are lost).")
            close.clicked.connect(self.close)
            for button in (self.generate_btn, self.save_btn, self.veusz_btn, close):
                buttons.addWidget(button)
            outer.addLayout(buttons)
            self.validate_x_axis_selection()

        # ----- helpers ------------------------------------------------------------
        def log_message(self, message: str) -> None:
            """Append ``message`` with an HH:MM:SS time stamp to the log."""
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.append("[%s] %s" % (stamp, message))

        def _set_theme(self, mode: str) -> None:
            """Apply and save the suite theme, recolour the Plots tab and re-apply
            the Generate button colour (the theme resets style sheets).
            """
            self.theme = suite.apply_theme(mode)
            if self.plot_panel is not None:
                self.plot_panel.apply_theme(self.theme == "dark")
            self.validate_x_axis_selection()
            self.log_message("Theme set to %s mode" % self.theme)

        def _config(self) -> CSVProcessingConfig:
            """Current CSV reading settings (delimiter, 0-based header line).
            """
            self.csv_config.delimiter = delimiter_from_text(self.delimiter_input.text())
            self.csv_config.header_line = self.header_spin.value() - 1
            return self.csv_config

        # ----- files --------------------------------------------------------------
        def browse_files(self) -> None:
            """Ask for delimited files and load them."""
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select CSV/TSV Files", "",
                "Delimited Files (*.csv *.tsv *.txt *.dat);;All Files (*)")
            self.add_files(files)

        def add_files(self, files: Sequence[str]) -> None:
            """Load the new ``files`` (duplicates ignored)."""
            new = [f for f in files if f not in self.selected_files]
            if not new:
                return
            self.selected_files.extend(new)
            self._load(new)

        def _load(self, paths: Sequence[str]) -> None:
            """Read ``paths`` with the current settings, log the messages and refresh
            the file list and column selectors.
            """
            files, messages = load_files(paths, self._config())
            for message in messages:
                self.log_message(message)
            for loaded in files:
                self.loaded[loaded.path] = loaded
            self.update_file_list()
            self.update_column_selectors()

        def reload_files(self) -> None:
            """Read every selected file again (after a delimiter/header change).
            """
            self.loaded.clear()
            if self.selected_files:
                self._load(self.selected_files)

        def update_file_list(self) -> None:
            """Show the files that loaded; each item keeps its full path."""
            self.file_list_widget.clear()
            for path in self.selected_files:
                if path in self.loaded:
                    self.file_list_widget.addItem(os.path.basename(path))
                    self.file_list_widget.item(self.file_list_widget.count() - 1).setToolTip(path)
                    self.file_list_widget.item(self.file_list_widget.count() - 1).setData(
                        Qt.ItemDataRole.UserRole, path)

        def clear_files(self) -> None:
            """Forget all files and clear the selectors and preview."""
            self.selected_files.clear()
            self.loaded.clear()
            self.update_file_list()
            self.update_column_selectors()
            self.log_message("File list cleared")

        def on_file_selection_changed(self) -> None:
            """Preview the file selected in the list."""
            item = self.file_list_widget.currentItem()
            if item is not None:
                self.display_data_preview(item.data(Qt.ItemDataRole.UserRole))

        def _files(self) -> List[LoadedFile]:
            """Loaded files in selection order."""
            return [self.loaded[p] for p in self.selected_files if p in self.loaded]

        def update_column_selectors(self) -> None:
            """Refill the X combo box and the Y list with the columns of the loaded
            files, keeping the current choices where they still exist.
            """
            current_x = self.x_column_combo.currentText()
            selected_y = {i.text() for i in self.y_columns_list.selectedItems()}
            self.x_column_combo.blockSignals(True)
            self.x_column_combo.clear()
            self.y_columns_list.clear()
            files = self._files()
            common = sorted(get_common_columns(files))
            self.x_column_combo.addItems(common)
            if current_x in common:
                self.x_column_combo.setCurrentText(current_x)
            self.x_column_combo.blockSignals(False)
            self.y_columns_list.addItems(sorted(get_all_columns(files)))
            for index in range(self.y_columns_list.count()):
                item = self.y_columns_list.item(index)
                item.setSelected(item.text() in selected_y)
            if files:
                self.log_message("X-axis options: %d common column(s); Y-axis options: %d"
                                 % (len(common), self.y_columns_list.count()))
                if not common:
                    self.log_message("WARNING: No common columns found across all files!")
            self.validate_x_axis_selection()

        def validate_x_axis_selection(self) -> bool:
            """True when the chosen X column exists in every loaded file.

            Enables the Generate button (green) only then; when files lack the
            column the button turns red and a warning names the column.
            """
            x_column = self.x_column_combo.currentText()
            files = self._files()
            missing = [f.name for f in files if x_column not in f.frame.columns]
            ok = bool(x_column and files and not missing)
            self.generate_btn.setEnabled(ok)
            colour = "#4CAF50" if ok else ("#DC143C" if missing else "#7f8c8d")
            self.generate_btn.setStyleSheet("background-color: %s; color: white; "
                                            "font-weight: bold; padding: 6px;" % colour)
            self.x_axis_warning_label.setText(
                "WARNING: NOT ALL LOADED FILES CONTAIN THE SELECTED X AXIS COLUMN '%s'"
                % x_column if missing else "")
            return ok

        def display_data_preview(self, path: str) -> None:
            """Show the first 10 rows of the file at ``path`` in the preview table.
            """
            loaded = self.loaded.get(path)
            if loaded is None:
                return
            preview = loaded.frame.head(10)
            self.data_preview_table.setRowCount(preview.shape[0])
            self.data_preview_table.setColumnCount(preview.shape[1])
            self.data_preview_table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
            for i, row in enumerate(preview.itertuples(index=False)):
                for j, value in enumerate(row):
                    self.data_preview_table.setItem(i, j, QTableWidgetItem(str(value)))
            self.data_preview_table.resizeColumnsToContents()

        def auto_detect_delimiter(self) -> None:
            """Guess the delimiter of the selected (or first) file and reload.
            """
            item = self.file_list_widget.currentItem()
            path = item.data(Qt.ItemDataRole.UserRole) if item else (
                self.selected_files[0] if self.selected_files else "")
            if not path:
                QMessageBox.warning(self, APP_NAME, "Add a file first.")
                return
            detected = detect_delimiter(path)
            self.set_delimiter("\\t" if detected == "\t" else detected)
            self.log_message("Auto-detected delimiter: %r" % detected)

        def set_delimiter(self, text: str) -> None:
            """Use delimiter ``text`` (',' ';' '|', '\\t' or 'tab' for a tab, or
            '\\s+' for runs of spaces) and reload the files.
            """
            self.delimiter_input.setText(text)
            self.reload_files()
            self.log_message("Delimiter set to: %r" % delimiter_from_text(text))

        # ----- plots and project -------------------------------------------------
        def _plot_config(self) -> PlotConfig:
            """PlotConfig from the plot option widgets."""
            config = self.plot_config
            config.title = self.title_input.text()
            config.x_label = self.x_label_input.text()
            config.y_label = self.y_label_input.text()
            config.show_legend = self.show_legend_checkbox.isChecked()
            config.show_grid = self.grid_box.isChecked()
            config.x_scale = "log" if self.log_x_box.isChecked() else "linear"
            config.y_scale = "log" if self.log_y_box.isChecked() else "linear"
            config.naive_time = "local" if self.naive_combo.currentIndex() == 1 else "utc"
            return config

        def generate_plots(self) -> None:
            """Build the project from the loaded files and chosen columns and show it
            on the Plots tab.
            """
            if not self.validate_x_axis_selection():
                QMessageBox.warning(self, APP_NAME, "Load files and choose an X column that "
                                    "exists in every file.")
                return
            y_columns = [i.text() for i in self.y_columns_list.selectedItems()]
            if not y_columns:
                QMessageBox.warning(self, APP_NAME, "Select one or more Y columns.")
                return
            config = self._plot_config()
            plotter = VeuszPlotter(config.title)
            try:
                n_traces = plotter.create_overlay_and_individual_plots(
                    self._files(), self.x_column_combo.currentText(), y_columns, config)
            except Exception as exc:
                QMessageBox.critical(self, APP_NAME, "Plot generation failed:\n%s" % exc)
                self.log_message("Plot generation error: %s" % exc)
                return
            for message in plotter.messages:
                self.status_text.append(message)
            self.plotter = plotter
            self.save_btn.setEnabled(True)
            if self.plot_panel is not None:
                try:
                    pages = self.plot_panel.set_document(plotter.doc)
                    self.log_message("Plots tab: %d page(s), %d overlay trace(s)."
                                     % (pages, n_traces))
                    self.tabs.setCurrentIndex(1)
                except Exception as exc:
                    self.log_message("Plots tab could not draw the project: %s" % exc)

        def save_to_veusz(self) -> None:
            """Ask for a file name and save the .vszh5 project."""
            if self.plotter is None:
                QMessageBox.warning(self, APP_NAME, "Generate plots first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Veusz Project", "",
                                                  "Veusz HDF5 Project (*.vszh5)")
            if not path:
                return
            try:
                self.last_saved = self.plotter.save_project(path, self.logging_box.isChecked())
            except Exception as exc:
                QMessageBox.critical(self, APP_NAME, "Error saving:\n%s" % exc)
                self.log_message("Save error: %s" % exc)
                return
            self.log_message("Saved Veusz project: %s" % self.last_saved)
            self.veusz_btn.setEnabled(True)
            if QMessageBox.question(self, "Open in Veusz", "Open the saved project in Veusz?") \
                    == QMessageBox.StandardButton.Yes:
                self.open_in_veusz()

        def open_in_veusz(self) -> None:
            """Open the last saved project in Veusz."""
            if not self.last_saved:
                QMessageBox.information(self, APP_NAME, "Save a project first.")
                return
            suite.open_in_veusz(self, self.last_saved)

        def export_pages(self) -> None:
            """Export the pages of the last saved project with Veusz (PDF/PNG).
            """
            if not self.last_saved:
                QMessageBox.information(self, APP_NAME, "Save a project first.")
                return
            suite.export_with_veusz(self, self.last_saved, self.log_message)

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
            """Clear the plots, then close."""
            if self.plot_panel is not None:
                self.plot_panel.clear()
            super().closeEvent(event)

    return QApplication, CSVAutoPlotMainWindow


# %% Main Execution
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point; returns the exit code.

    With ``--output`` the files are plotted in batch (no GUI; ``--x`` and
    ``--y`` choose the columns) into that project; otherwise the GUI opens.
    """
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--output", help="write this .vszh5 without opening the GUI")
    parser.add_argument("--x", help="X column (batch)")
    parser.add_argument("--y", action="append", default=[], help="Y column (repeatable)")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--header-line", type=int, default=1, help="1 = first line")
    parser.add_argument("--title", default=PlotConfig.title)
    parser.add_argument("--naive-time", choices=("utc", "local"), default="utc",
                        help="date-times written without a zone are UTC or local time")
    parser.add_argument("--version", action="version", version="%s %s" % (APP_NAME, APP_VERSION))
    args = parser.parse_args(argv)
    if args.output:
        if not args.x or not args.y:
            parser.error("--output needs --x and at least one --y")
        config = CSVProcessingConfig(delimiter=delimiter_from_text(args.delimiter),
                                     header_line=max(0, args.header_line - 1))
        files, messages = load_files(args.files, config)
        print("\n".join(messages))
        plotter = VeuszPlotter(args.title)
        plotter.create_overlay_and_individual_plots(
            files, args.x, args.y, PlotConfig(title=args.title, x_label=args.x,
                                              y_label=", ".join(args.y),
                                              naive_time=args.naive_time))
        print("\n".join(plotter.messages))
        print("Wrote", plotter.save_project(args.output))
        return 0
    QApplication, Window = _build_gui_classes()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = Window()
    window.add_files(args.files)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
