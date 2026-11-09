#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
RAndS_FSW_ASCII_Plotter.py

Rohde & Schwarz spectrum-analyzer trace files (.sft: FSP/FSW "Section"
files in ASCII or binary, and FSW ASCII x;y exports) -> native Veusz
project (.vszh5) and interactive pyqtgraph plots:

    * x axis from the header: frequency (x-Unit Hz; LIN -> linspace,
      LOG -> geomspace from Start to Stop) or time (x-Unit s, zero span:
      0 .. SWT);
    * every section (sweep) of every file, per-file power averages and an
      average of all files that share one x axis;
    * plot modes: all plots / average + overlay / average only;
    * optional split into several project files of at most N section pages;
    * the .vszh5 is written by the author's shared writer (vszh5_writer.py
      0.0.24 through _autoplot_native.NativeDocument) - Veusz is NOT needed
      to create the file, only (optionally) to view or export it.

Batch use without the GUI:
    python RAndS_FSW_ASCII_Plotter.py --mode 0 --output out.vszh5 a.sft b.sft

Created on 2025-06-28

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later (GNU General Public License v3.0 or later)
Compatible with: Python 3.8+ (qtpy with PySide6 or PyQt5; CuPy optional)

# %%% File layout (from the examples in Example_Data_Files/sft)
Header lines "Key;Value;Unit;" (Type, Version, Mode, Center Freq, Span,
x-Axis LIN|LOG, Start, Stop, Ref Level, RBW, VBW, SWT, Trace Mode, Detector,
x-Unit, y-Unit, Values N, ...).  Then one block per sweep:
    "Section N;YYYY.M.D_h.m.s:ms" newline, then N values, either one
    comma-separated ASCII line (FSP firmware 4.50 examples) or N little-
    endian float32 values followed by one byte (binary example
    ngRADAR_003.sft: 2004 + 1 bytes for N = 501).
FSW ASCII exports without sections ("x;y;" rows after "Values") are read
as one trace per y column.

# %%% References
[1] Rohde & Schwarz, "R&S FSW Signal and Spectrum Analyzer User Manual,"
    ch. "Trace export / ASCII file export format," Munich, Germany.
    https://www.rohde-schwarz.com/manual/fsw/
[2] Rohde & Schwarz, "R&S FSP Spectrum Analyzer Operating Manual," ch.
    "ASCII file export format (TRACE ... ASCII FILE EXPORT)," Munich, Germany.
[3] W. W. Wallace, "vszh5_writer.py 0.0.24," shared Veusz HDF5 writer,
    2026 (linear_power_mean_dbm: average of mW, not of dB).
[4] IEEE Standard for Spectrum Analyzers, IEEE Std 748-1979 (R2007) -
    power (not log) averaging of traces.

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 1.1.2: Header template, docstrings and tooltips.
Date: 2026-11-09
#              * Header in the author's template: author contact lines and,
#                under this entry, Function Descriptions, Variable
#                Descriptions and More Info.
#              * Docstrings for every function, class and method.  No change
#                to what the program does.
#              * Tooltips (pop-up help) on every button, box and field that
#                had none (14 widgets, including the three plot modes).
# %%%%% Function Descriptions
        _cupy: CuPy module when a CUDA device works, else None (checked on demand).
        SFTFormatError (class): The file holds no trace this program can read.
        SFTTrace (class): One sweep (section) of an R&S SFT file: its number, the instrument time stamp text and the trace values.
        SFTFile (class): A parsed R&S FSP/FSW ASCII trace file (.sft).
        _header: Parse the 'key;value;unit' header lines of an SFT file.
        parse_sweep_time: FSP/FSW section stamp "2025.6.17_9.26.44:783" -> naive datetime.
        _float: float(text), or ``default`` (NaN) when text is not a number.
        _ascii_values: Values of an ASCII section, or None when the block is not ASCII numbers.
        parse_sft_bytes: Parse one .sft file (see the file layout in the module header).
        read_sft: Read and parse one SFT file (bytes, any encoding the parser accepts).
        _read_worker: Process-pool worker: (path, SFTFile or None, error text).
        read_files: Read in input order ('spawn' process pool for 4+ files); (files, errors).
        power_mean: Average of traces (rows) in their unit.
        PlotOptions (class): Plot choices: ``mode`` 0 = every sweep plus the average, 1 = average and overlay, 2 = average only; ``use_gpu`` averages with CuPy when available.
        RnSPlotter (class): Builds the project of a batch of .sft files.
        build_project: RnSPlotter with every file of ``files`` added and finished.
        batches: Split whole files into groups of at most ``plots_per_file`` sections (a file with more sections than that is a group of its own).
        save_split: Write {base}_001.vszh5, {base}_002.vszh5 ...
        _build_gui_classes: Import Qt and define the GUI classes on first use.
        main: Command line entry point; returns the exit code.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP_NAME = "R&S FSW / FSP Trace Plotter"
        APP_VERSION = "1.1.2"
        APP_DESCRIPTION: one-line description (command line help, About box)
        PARALLEL_MIN_FILES: fewest files read with a process pool
        MODE_NAMES: plot mode names (0 all, 1 average + overlay, 2 average only)
        _SECTION: the 'Section<n>;<time stamp>' line that starts each sweep (bytes)
        _NUMBER: one numeric token
# %%%%% More Info
    Run:  python RAndS_FSW_ASCII_Plotter.py [files]   (GUI; View menu)
          python RAndS_FSW_ASCII_Plotter.py *.sft --output rns.vszh5 [--split N]
    Sweep averages can use CuPy (--gpu) when a CUDA device is present.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 1.1.1: Original page names, sweep times as Veusz date-times, notes.
Date: 2026-11-09
#              * Page and graph names follow the 2025 script: a section page
#                is "<file>::Section N;<time stamp>" (underscores shown as
#                spaces, as before), an average page "<file> average"; the
#                graphs are named after their y dataset ("<file>_Section1",
#                "<file>_avg_dB") instead of "graph".
#              * The sweep time stamps ("2025.6.17_9.26.44:783") are stored
#                as a Veusz DATE dataset "<file>_sweep_time" (instrument
#                clock; the file does not say which zone) and listed in the
#                page notes.
#              * Page notes: "Source: <file>", the full header and the sweep
#                count; a section page adds its own sweep time, the file's
#                average page lists every sweep time; the overlay and
#                Averages pages list every file with its header.
#              * Writer 0.0.25 / _autoplot_native 0.3.0.
# %%%% 1.1.0: Native writer, Plots tab, time-domain axis, binary sections.
Date: 2026-11-05
#              * .vszh5 written by vszh5_writer 0.0.24 through
#                _autoplot_native.NativeDocument (no veusz.embed window, no
#                extra legacy .vsz in "Beware_oldVersion").
#              * Plots tab (pyqtgraph), View > Light / Dark (suite theme),
#                Help > About, File menu, Export PDF / PNG (Veusz).
#              * Time domain: x-Unit "s" (zero span) gives a time axis
#                0 .. SWT labelled "Time (s)"; it was drawn against
#                Start .. Stop as if it were a frequency sweep.  x-Axis LOG
#                gives logarithmically spaced points (was linear) and a log
#                x axis.
#              * Header read by key (the old pattern search matched
#                "ENDSTOP" for "Stop" and "Trace Mode" for "Trace"), binary
#                float32 sections read (ngRADAR_003.sft was unreadable),
#                empty sections and short traces reported, not fatal.
#              * Averages: power mean in mW (vszh5_writer.
#                linear_power_mean_dbm; NaN-aware fallback), optional CuPy.
#                "Average only" produced no plot at all and "Average +
#                Overlay" only added the average to the overlay: every mode
#                now has a page per file average and an "Averages" page
#                (each file, plus all files when they share one x axis).
#              * Dataset names: {file}_freq / {file}_time and
#                {file}_Section{N} (the bare "Section N;time" names of
#                different files could collide); the time stamp is in the
#                legend text.
#              * Check boxes use toggled(bool) (stateChanged == Qt.Checked
#                was always False under PySide6).  The no-op GPU "process"
#                step (multiply by 1.0) and the per-worker GPU context were
#                removed; GPU is used for the averages when CuPy works.
#              * Split saving writes {base}_001.vszh5 ... next to the chosen
#                file when saving (it used to create a folder in the current
#                working directory during processing).
#              * The hang monitor (Hanging_Threads) only runs with
#                AUTOPLOT_HANG_MONITOR=1: an idle Qt event loop looked like
#                a hang every 20 s.
# %%%% 1.0.3: Corrected Average+Overlay only mode with AutoSave (2026-01-26).
Date: 2026-01-26
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

APP_NAME = "R&S FSW / FSP Trace Plotter"
APP_VERSION = "1.1.2"
__version__ = APP_VERSION
APP_DESCRIPTION = ("Rohde & Schwarz .sft trace files to native Veusz .vszh5 projects, "
                   "with averages and interactive pyqtgraph plots.")
PARALLEL_MIN_FILES = 4
MODE_NAMES = ("All plots", "Average + Overlay", "Average only")


def _cupy():
    """CuPy module when a CUDA device works, else None (checked on demand)."""
    try:
        import cupy
        cupy.cuda.Device(0).use()
        return cupy
    except Exception:
        return None


# %% SFT reader (no Qt; runs in worker processes)
class SFTFormatError(ValueError):
    """The file holds no trace this program can read."""


@dataclass
class SFTTrace:
    """One sweep (section) of an R&S SFT file: its number, the instrument time
    stamp text and the trace values.
    """
    number: int
    stamp: str
    values: np.ndarray


@dataclass
class SFTFile:
    """A parsed R&S FSP/FSW ASCII trace file (.sft).

    ``header`` maps lower-case keys to (value, unit); ``header_lines`` keeps
    the header as written (for page notes); ``x`` is frequency, time or index
    (``x_kind``) with its label and log flag; ``traces`` are the sweeps and
    ``warnings`` notes made while parsing.
    """
    path: str
    header: Dict[str, Tuple[str, str]]      # lower-case key -> (value, unit)
    header_lines: List[str]
    x: np.ndarray
    x_kind: str                             # "frequency", "time" or "index"
    x_label: str
    x_log: bool
    y_unit: str
    traces: List[SFTTrace]
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """File name (no folder)."""
        return os.path.basename(self.path)

    def value(self, key: str, default: str = "") -> str:
        """Header value of ``key`` (any case), or ``default``."""
        return self.header.get(key.lower(), (default, ""))[0]


_SECTION = re.compile(rb"Section\s*(\d+);([^\r\n]*)\r?\n")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _header(text: str) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """Parse the 'key;value;unit' header lines of an SFT file.

    Returns ({lower-case key: (value, unit)}, non-empty header lines); the
    first occurrence of a key wins.
    """
    header: Dict[str, Tuple[str, str]] = {}
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lines.append(line)
        parts = [p.strip() for p in line.split(";")]
        key = parts[0].rstrip(":").lower()
        if key and key not in header:
            header[key] = (parts[1] if len(parts) > 1 else "",
                           parts[2] if len(parts) > 2 else "")
    return header, lines


def parse_sweep_time(stamp: str) -> Optional[datetime.datetime]:
    """FSP/FSW section stamp "2025.6.17_9.26.44:783" -> naive datetime."""
    match = re.match(r"\s*(\d{4})\.(\d{1,2})\.(\d{1,2})[_ T](\d{1,2})\.(\d{1,2})\.(\d{1,2})"
                     r"(?::(\d{1,3}))?", stamp or "")
    if not match:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    millis = int((match.group(7) or "0").ljust(3, "0"))
    try:
        return datetime.datetime(year, month, day, hour, minute, second, millis * 1000)
    except ValueError:
        return None


def _float(text: str, default: float = float("nan")) -> float:
    """float(text), or ``default`` (NaN) when text is not a number."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _ascii_values(block: bytes) -> Optional[np.ndarray]:
    """Values of an ASCII section, or None when the block is not ASCII numbers."""
    try:
        text = block.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    tokens = [t for t in re.split(r"[,;\s]+", text) if t]
    if not tokens or not all(_NUMBER.match(t) for t in tokens):
        return None
    return np.array(tokens, dtype=float)


def parse_sft_bytes(raw: bytes, path: str = "<bytes>") -> SFTFile:
    """Parse one .sft file (see the file layout in the module header)."""
    name = os.path.basename(path)
    warnings: List[str] = []
    matches = list(_SECTION.finditer(raw))
    head_end = matches[0].start() if matches else len(raw)
    header, header_lines = _header(raw[:head_end].decode("ascii", errors="replace"))
    declared = _float(header.get("values", ("", ""))[0])
    n_values = int(declared) if np.isfinite(declared) and declared > 0 else None
    traces: List[SFTTrace] = []
    rows_x: Optional[np.ndarray] = None
    if matches:
        for index, match in enumerate(matches):
            start = match.end()
            stop = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            block = raw[start:stop]
            number = int(match.group(1))
            stamp = match.group(2).decode("ascii", errors="replace").strip()
            values = _ascii_values(block)
            if values is None and n_values:
                if len(block) >= 4 * n_values:           # binary float32 trace
                    values = np.frombuffer(block[:4 * n_values], dtype="<f4").astype(float)
            if values is None or values.size == 0:
                warnings.append("section %d (%s) holds no readable trace; skipped" % (number, stamp))
                continue
            if n_values and values.size != n_values:
                warnings.append("section %d has %d values, the header says %d; skipped"
                                % (number, values.size, n_values))
                continue
            traces.append(SFTTrace(number, stamp, values))
    else:                                               # FSW ASCII export: x;y;... rows
        text = raw.decode("ascii", errors="replace").splitlines()
        rows = []
        after_values = False
        for line in text:
            if line.strip().lower().startswith("values"):
                after_values = True
                continue
            if not after_values:
                continue
            tokens = [t for t in re.split(r"[;,\s]+", line.strip()) if t]
            if len(tokens) >= 2 and all(_NUMBER.match(t) for t in tokens):
                rows.append([float(t) for t in tokens])
        if rows and len({len(r) for r in rows}) == 1:
            table = np.array(rows)
            rows_x = table[:, 0]
            for column in range(1, table.shape[1]):
                traces.append(SFTTrace(column, "column %d" % column, table[:, column]))
            n_values = table.shape[0]
    if not traces:
        raise SFTFormatError("%s: no trace data (header only or unknown layout)" % name)
    n = traces[0].values.size
    x_unit = header.get("x-unit", ("", ""))[0]
    x_log = header.get("x-axis", ("LIN", ""))[0].upper().startswith("LOG")
    start = _float(header.get("start", ("nan", ""))[0])
    stop = _float(header.get("stop", ("nan", ""))[0])
    sweep = _float(header.get("swt", ("nan", ""))[0])
    if rows_x is not None:
        x, x_kind = rows_x, ("time" if x_unit.lower() == "s" else "frequency")
        x_label = "Time (s)" if x_kind == "time" else "Frequency (%s)" % (x_unit or "Hz")
    elif x_unit.lower() == "s":                         # zero span: time domain
        if not np.isfinite(sweep) or sweep <= 0:
            raise SFTFormatError("%s: x-Unit s but no sweep time (SWT)" % name)
        x = np.linspace(0.0, sweep, n)
        x_kind, x_label = "time", "Time (s)"
        warnings.append("x-Unit s: time axis 0 to %g s (sweep time); Start/Stop are not "
                        "used" % sweep)
        x_log = False
    elif np.isfinite(start) and np.isfinite(stop):
        if x_log and start > 0 and stop > 0:
            x = np.geomspace(start, stop, n)
        else:
            if x_log:
                warnings.append("x-Axis LOG with Start <= 0: linear points used")
            x_log = False
            x = np.linspace(start, stop, n)
        x_kind, x_label = "frequency", "Frequency (%s)" % (x_unit or "Hz")
    else:
        x = np.arange(n, dtype=float)
        x_kind, x_label = "index", "Point"
        warnings.append("no Start/Stop in the header: x is the point number")
    y_unit = header.get("y-unit", ("dBm", ""))[0] or "dBm"
    return SFTFile(path=path, header=header, header_lines=header_lines, x=x, x_kind=x_kind,
                   x_label=x_label, x_log=x_log, y_unit=y_unit, traces=traces,
                   warnings=warnings)


def read_sft(path: str) -> SFTFile:
    """Read and parse one SFT file (bytes, any encoding the parser accepts)."""
    with open(path, "rb") as handle:
        return parse_sft_bytes(handle.read(), path)


def _read_worker(path: str) -> Tuple[str, Optional[SFTFile], str]:
    """Process-pool worker: (path, SFTFile or None, error text)."""
    try:
        return path, read_sft(path), ""
    except Exception as exc:
        return path, None, "%s: %s" % (type(exc).__name__, exc)


def read_files(paths: Sequence[str], workers: int = 1,
               progress: Optional[Any] = None) -> Tuple[List[SFTFile], List[str]]:
    """Read in input order ('spawn' process pool for 4+ files); (files, errors)."""
    paths = list(dict.fromkeys(paths))
    results: Dict[str, Tuple[Optional[SFTFile], str]] = {}
    if workers > 1 and len(paths) >= PARALLEL_MIN_FILES:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_read_worker, p) for p in paths]
            for done, future in enumerate(as_completed(futures), start=1):
                path, parsed, error = future.result()
                results[path] = (parsed, error)
                if progress:
                    progress(done, len(paths))
    else:
        for done, path in enumerate(paths, start=1):
            _p, parsed, error = _read_worker(path)
            results[path] = (parsed, error)
            if progress:
                progress(done, len(paths))
    files, errors = [], []
    for path in paths:
        parsed, error = results[path]
        if parsed is None:
            errors.append("%s: %s" % (os.path.basename(path), error))
        else:
            files.append(parsed)
    return files, errors


# %% Averages
def power_mean(stack: np.ndarray, unit: str, use_gpu: bool = False) -> np.ndarray:
    """
    Average of traces (rows) in their unit.

    Log units (dBm, dB, dBuV ...): mean of the linear powers, back in dB:
        P_avg = 10 log10( (1/K) sum_k 10^(P_k / 10) )
    Linear units: arithmetic mean.  NaN samples are left out.
    """
    stack = np.asarray(stack, dtype=float)
    if not unit.lower().startswith("db"):
        return np.nanmean(stack, axis=0)
    if np.all(np.isfinite(stack)) and not use_gpu:
        from vszh5_writer import linear_power_mean_dbm
        return linear_power_mean_dbm(stack, axis=0)
    xp = _cupy() if use_gpu else None
    xp = xp or np
    data = xp.asarray(stack)
    peak = xp.nanmax(data, axis=0)                      # overflow-safe reference
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = xp.nanmean(xp.power(10.0, (data - peak) / 10.0), axis=0)
        result = peak + 10.0 * xp.log10(mean)
    return xp.asnumpy(result) if xp is not np else result


# %% Veusz project (NativeDocument; written by vszh5_writer)
@dataclass
class PlotOptions:
    """Plot choices: ``mode`` 0 = every sweep plus the average, 1 = average and
    overlay, 2 = average only; ``use_gpu`` averages with CuPy when available.
    """
    mode: int = 0                    # 0 all plots, 1 average + overlay, 2 average only
    use_gpu: bool = False


class RnSPlotter:
    """Builds the project of a batch of .sft files."""

    def __init__(self, options: Optional[PlotOptions] = None) -> None:
        """Empty R&S project with ``options``; pages are added by add_file() and
        completed by finish().
        """
        from _autoplot_native import NativeDocument
        self.options = options or PlotOptions()
        self.doc = NativeDocument("R&S trace plotter")
        self.doc.Root.colorTheme.val = "max128"
        self.messages: List[str] = []
        self._stems: Dict[str, int] = {}
        self._overlays: Dict[str, Any] = {}
        self._averages: Dict[str, List[Tuple[str, str, SFTFile]]] = {}
        self._file_notes: List[str] = []
        self._overlay_pages: List[Any] = []

    def _stem(self, sft: SFTFile) -> str:
        """Name stem for ``sft``: its file name without extension, with ``_2``,
        ``_3``, ... for repeated names.
        """
        stem = os.path.splitext(sft.name)[0]
        count = self._stems.get(stem, 0) + 1
        self._stems[stem] = count
        return stem if count == 1 else "%s_%d" % (stem, count)

    def add_file(self, sft: SFTFile) -> str:
        """Add one file's datasets and pages; return its name stem.

        Stores the x axis, the sweep times (Veusz date-times on the instrument
        clock, zone not recorded) and each sweep; draws the section pages, the
        average page and the overlay trace as ``options.mode`` asks.  A section
        page's notes name only its own sweep (1.1.1 fix: listing every sweep on
        every page made the notes grow as sweeps squared).
        """
        doc, mode = self.doc, self.options.mode
        stem = self._stem(sft)
        x_name = "%s_%s" % (stem, "time" if sft.x_kind == "time" else "freq")
        doc.SetData(x_name, sft.x)
        names = [x_name]
        stamps = [parse_sweep_time(t.stamp) for t in sft.traces]
        if any(s is not None for s in stamps):       # instrument clock, zone not recorded
            time_name = "%s_sweep_time" % stem
            doc.SetDataDateTime(time_name, [(s - VEUSZ_EPOCH).total_seconds() if s else np.nan
                                            for s in stamps])
            names.append(time_name)
        summary = "%d sweep(s); x: %s; y: %s" % (len(sft.traces), sft.x_label, sft.y_unit)
        # The average page lists every sweep time; a section page names only
        # its own sweep (listing all of them on every section page made the
        # notes grow as sweeps^2: an 11.5 MB script for 588 pages).
        notes = header_notes(sft.header_lines, source=sft.path, extra=[summary]
            + ["Section %d: %s" % (t.number, t.stamp) for t in sft.traces[:2000]]
            + (["... %d more" % (len(sft.traces) - 2000)] if len(sft.traces) > 2000 else []))
        self._file_notes.append(header_notes(sft.header_lines, source=sft.path))
        y_label = "Uncalibrated (%s)" % sft.y_unit
        for trace in sft.traces:
            y_name = "%s_Section%d" % (stem, trace.number)
            doc.SetData(y_name, trace.values)
            names.append(y_name)
            key = "%s S%d %s" % (stem, trace.number, trace.stamp)
            if mode in (0, 1):
                overlay = self._overlay(sft)
                xy = overlay.Add("xy", name=y_name)
                self._style_overlay(xy, x_name, y_name, key)
            if mode == 0:
                title = ("%s::Section %d;%s" % (stem, trace.number, trace.stamp)).replace("_", " ")
                section_notes = header_notes(sft.header_lines, source=sft.path, extra=[
                    summary, "Section %d of %d: %s" % (trace.number, len(sft.traces), trace.stamp)])
                graph = self._page_graph(title, title, section_notes, sft, y_label,
                                         graph_name=y_name)
                self._style_single(graph.Add("xy", name=y_name), x_name, y_name, key)
        stack = np.vstack([t.values for t in sft.traces])
        avg_db = power_mean(stack, sft.y_unit, self.options.use_gpu)
        db_name = "%s_avg_dB" % stem
        doc.SetData(db_name, avg_db)
        names.append(db_name)
        if sft.y_unit.lower().startswith("db"):
            lin_name = "%s_avg_lin" % stem
            doc.SetData(lin_name, np.power(10.0, avg_db / 10.0))
            names.append(lin_name)
            doc.TagDatasets("Avg_Linear", [lin_name])
        doc.TagDatasets("Avg_dB", [db_name])
        doc.TagDatasets(stem, names)
        title = ("%s average of %d sweep(s)" % (stem, len(sft.traces))).replace("_", " ")
        graph = self._page_graph("%s average" % stem, title, notes, sft, y_label,
                                 graph_name=db_name)
        self._style_single(graph.Add("xy", name=db_name), x_name, db_name,
                           "%s average (%d)" % (stem, len(sft.traces)))
        self._averages.setdefault(sft.x_kind, []).append((x_name, db_name, sft))
        for warning in sft.warnings:
            self.messages.append("%s: %s" % (sft.name, warning))
        return stem

    def finish(self) -> None:
        """Averages page(s): every file average, and all files on one x axis."""
        for page in self._overlay_pages:
            page.notes.val = "\n\n".join(["All Imported and Plottable Data Overlay"]
                                          + self._file_notes)
        for kind, entries in self._averages.items():
            suffix = "" if kind == "frequency" else "_" + kind
            sft0 = entries[0][2]
            graph = self._page_graph("Averages%s" % suffix, "Average of each file",
                                     "\n\n".join(["Power averages of every file"]
                                                  + self._file_notes), sft0,
                                     "Uncalibrated (%s)" % sft0.y_unit,
                                     graph_name="Averages%s_avg_dB" % suffix)
            for index, (x_name, db_name, sft) in enumerate(entries):
                xy = graph.Add("xy", name=db_name)
                self._style_overlay(xy, x_name, db_name, db_name.replace("_avg_dB", ""))
            same = all(np.array_equal(e[2].x, sft0.x) and e[2].y_unit == sft0.y_unit
                       for e in entries)
            if len(entries) > 1 and same:
                stack = np.vstack([t.values for e in entries for t in e[2].traces])
                name = "AllFiles%s_avg_dB" % suffix
                self.doc.SetData(name, power_mean(stack, sft0.y_unit, self.options.use_gpu))
                self.doc.TagDatasets("Avg_dB", [name])
                xy = graph.Add("xy", name=name)
                self._style_single(xy, entries[0][0], name,
                                   "All files (%d sweeps)" % stack.shape[0])
                xy.PlotLine.color.val = "black"
                xy.PlotLine.width.val = "1.5pt"
                xy.FillBelow.hide.val = True
            elif len(entries) > 1:
                self.messages.append("Averages%s: the files have different x axes or units; "
                                     "no all-files average" % suffix)

    # ----- pages ---------------------------------------------------------------
    def _overlay(self, sft: SFTFile) -> Any:
        """Overlay graph for the x kind of ``sft`` (frequency, time, index),
        created on first use on page AllImported / AllImported_<kind>.
        """
        kind = sft.x_kind
        if kind not in self._overlays:
            name = "AllImported" if kind == "frequency" else "AllImported_%s" % kind
            page = self.doc.Root.Add("page", name=name)
            page.notes.val = "All Imported and Plottable Data Overlay"
            self._overlay_pages.append(page)
            graph = page.Add("grid", columns=2).Add("graph", name="Imported_Overlay")
            self._decorate(graph, "Overlay of All Imported", sft, "Uncalibrated (%s)" % sft.y_unit,
                           "centre", 0.5)
            graph.notes.val = "All imported overlay, see individual plots for specifics."
            self._overlays[kind] = graph
        return self._overlays[kind]

    def _page_graph(self, page_name: str, title: str, notes: str, sft: SFTFile,
                    y_label: str, graph_name: str = "graph") -> Any:
        """New page ``page_name`` with ``notes`` holding one titled graph."""
        page = self.doc.Root.Add("page", name=page_name)
        page.notes.val = notes
        graph = page.Add("grid", columns=2).Add("graph", name=graph_name)
        self._decorate(graph, title, sft, y_label, "left", -0.3)
        graph.notes.val = notes
        return graph

    @staticmethod
    def _decorate(graph: Any, title: str, sft: SFTFile, y_label: str, align: str,
                  x_pos: float) -> None:
        """Title, axis labels (x from the file, y ``y_label``), log x when the
        file says so, and the legend position (``align``, ``x_pos``).
        """
        graph.Add("label", name="plotTitle")
        graph.topMargin.val = "1cm"
        graph.plotTitle.Text.size.val = "10pt"
        graph.plotTitle.label.val = title
        graph.plotTitle.alignHorz.val = align
        graph.plotTitle.yPos.val = 1.05
        graph.plotTitle.xPos.val = x_pos
        graph.x.label.val = sft.x_label
        graph.y.label.val = y_label
        graph.x.log.val = bool(sft.x_log)

    @staticmethod
    def _style_overlay(xy: Any, x_name: str, y_name: str, key: str) -> None:
        """Overlay trace style: small translucent markers, automatic colour."""
        xy.xData.val, xy.yData.val, xy.key.val = x_name, y_name, key
        xy.marker.val = "circle"
        xy.markerSize.val = "2pt"
        xy.MarkerLine.color.val = "transparent"
        xy.MarkerFill.color.val = "auto"
        xy.MarkerFill.transparency.val = 80
        xy.MarkerFill.style.val = "solid"
        xy.FillBelow.hide.val = True
        xy.PlotLine.color.val = "auto"

    @staticmethod
    def _style_single(xy: Any, x_name: str, y_name: str, key: str) -> None:
        """Single-page trace style: small markers and a red line."""
        xy.xData.val, xy.yData.val, xy.key.val = x_name, y_name, key
        xy.marker.val = "circle"
        xy.markerSize.val = "2pt"
        xy.MarkerLine.color.val = "transparent"
        xy.MarkerFill.color.val = "foreground"
        xy.MarkerFill.transparency.val = 80
        xy.MarkerFill.style.val = "solid"
        xy.FillBelow.transparency.val = 90
        xy.FillBelow.style.val = "solid"
        xy.FillBelow.fillto.val = "bottom"
        xy.FillBelow.color.val = "darkgreen"
        xy.FillBelow.hide.val = False
        xy.PlotLine.color.val = "red"


def build_project(files: Sequence[SFTFile], options: Optional[PlotOptions] = None) -> RnSPlotter:
    """RnSPlotter with every file of ``files`` added and finished."""
    plotter = RnSPlotter(options)
    for sft in files:
        plotter.add_file(sft)
    plotter.finish()
    return plotter


def batches(files: Sequence[SFTFile], plots_per_file: int) -> List[List[SFTFile]]:
    """Split whole files into groups of at most ``plots_per_file`` sections
    (a file with more sections than that is a group of its own)."""
    groups: List[List[SFTFile]] = []
    current: List[SFTFile] = []
    count = 0
    for sft in files:
        n = len(sft.traces)
        if current and count + n > plots_per_file:
            groups.append(current)
            current, count = [], 0
        current.append(sft)
        count += n
    if current:
        groups.append(current)
    return groups


def save_split(files: Sequence[SFTFile], base_path: str, plots_per_file: int,
               options: PlotOptions) -> List[str]:
    """Write {base}_001.vszh5, {base}_002.vszh5 ... (one project per group)."""
    root = os.path.splitext(base_path)[0]
    written = []
    for index, group in enumerate(batches(files, plots_per_file), start=1):
        written.append(build_project(group, options).doc.save("%s_%03d.vszh5" % (root, index)))
    return written


# %% GUI (imported lazily so --output works without Qt)
def _build_gui_classes():
    """Import Qt and define the GUI classes on first use.

    Keeps the batch mode (``--output``) free of Qt; returns (QApplication,
    EnhancedMainWindow).
    """
    from qtpy.QtCore import QThread, Signal
    from qtpy.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFileDialog,
                                QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
                                QListWidget, QMainWindow, QMessageBox, QProgressBar,
                                QPushButton, QRadioButton, QSpinBox, QTabWidget,
                                QTextEdit, QVBoxLayout, QWidget)
    import _suite_gui as suite

    class FileProcessingThread(QThread):
        """Qt thread that reads SFT files with a process pool and reports progress.
        """
        progress_updated = Signal(int)
        processing_finished = Signal(object, object)

        def __init__(self, paths: Sequence[str], workers: int) -> None:
            """Thread that reads ``paths`` with ``workers`` processes."""
            super().__init__()
            self.paths, self.workers = list(paths), workers

        def run(self) -> None:
            """Read the files (progress in percent), then emit
            ``processing_finished(files, errors)``; an unexpected exception is
            reported as an error, never raised in the thread.
            """
            try:
                files, errors = read_files(
                    self.paths, self.workers,
                    lambda done, total: self.progress_updated.emit(int(100 * done / total)))
            except Exception as exc:
                files, errors = [], ["batch failed: %s: %s" % (type(exc).__name__, exc)]
            self.processing_finished.emit(files, errors)

    class EnhancedMainWindow(QMainWindow):
        """Processing and Plots tabs; File / View / Help menus."""

        def __init__(self) -> None:
            """Main window: Processing and Plots tabs, menus with the light/dark
            theme, and the suite theme applied.
            """
            super().__init__()
            self.setWindowTitle("%s %s" % (APP_NAME, APP_VERSION))
            self.resize(1150, 900)
            self.selected_files: List[str] = []
            self.files: List[SFTFile] = []
            self.plotter: Optional[RnSPlotter] = None
            self.last_saved: Optional[str] = None
            self.thread: Optional[FileProcessingThread] = None
            self.theme = suite.apply_theme(suite.saved_theme(), persist=False)
            suite.add_file_menu(self, [
                ("&Add SFT Files...", self._browse_files, "Ctrl+O"),
                ("&Clear File List", self._clear_files, ""),
                ("&Process and Plot", self._process_and_plot, "F5"),
                ("&Save Veusz Project (.vszh5)...", self._save_project, "Ctrl+S"),
                ("Open Last Project in &Veusz", self._open_in_veusz, ""),
                ("&Export Last Project to PDF / PNG (Veusz)", self._export, ""),
            ])
            self.act_light, self.act_dark = suite.add_view_menu(self, self._set_theme, self.theme)
            suite.add_help_menu(self, APP_NAME, APP_VERSION, APP_DESCRIPTION, ("cupy",))
            self._setup_ui()
            self._log_message("%s %s ready; theme %s." % (APP_NAME, APP_VERSION, self.theme))

        def _setup_ui(self) -> None:
            """Lay out the Processing tab (files, plot mode, GPU, split projects,
            workers, progress, log, buttons) and the Plots tab.
            """
            central = QWidget()
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)
            self.tabs = QTabWidget()
            outer.addWidget(self.tabs, 1)
            processing = QWidget()
            self.tabs.addTab(processing, "Processing")
            layout = QVBoxLayout(processing)
            group = QGroupBox("File Selection")
            box = QVBoxLayout(group)
            self.file_list_widget = QListWidget()
            self.file_list_widget.setToolTip("Files to process; add them with Browse, remove all with Clear.")
            box.addWidget(self.file_list_widget)
            row = QHBoxLayout()
            for text, slot, tip in (
                    ("Browse Files", self._browse_files,
                     "Add R&S FSP/FSW ASCII trace files (.sft) to the list."),
                    ("Clear Files", self._clear_files,
                     "Remove every file from the list (the files are not touched).")):
                button = QPushButton(text)
                button.setToolTip(tip)
                button.clicked.connect(slot)
                row.addWidget(button)
            box.addLayout(row)
            layout.addWidget(group)

            options = QGroupBox("Processing Options")
            form = QFormLayout(options)
            self.enable_mp_checkbox = QCheckBox("Read files in parallel processes (%d or more files)"
                                                % PARALLEL_MIN_FILES)
            self.enable_mp_checkbox.setToolTip(
                "Read the files in separate processes when there are enough of "
                "them (faster for big batches).")
            self.enable_mp_checkbox.setChecked(True)
            self.enable_mp_checkbox.toggled.connect(
                lambda on: self._log_message("Multiprocessing %s" % ("on" if on else "off")))
            form.addRow(self.enable_mp_checkbox)
            self.cpu_spinbox = QSpinBox()
            self.cpu_spinbox.setToolTip("Number of processes used for parallel reading.")
            self.cpu_spinbox.setRange(1, max(1, multiprocessing.cpu_count()))
            self.cpu_spinbox.setValue(max(1, multiprocessing.cpu_count()))
            form.addRow("CPU processes:", self.cpu_spinbox)
            self.enable_gpu_checkbox = QCheckBox("Average on the GPU (CuPy / CUDA)")
            self.enable_gpu_checkbox.setToolTip(
                "Compute the power averages with CuPy on an NVIDIA GPU; unticks"
                " itself when no GPU is found.")
            self.enable_gpu_checkbox.toggled.connect(self._gpu_toggled)
            form.addRow(self.enable_gpu_checkbox)
            layout.addWidget(options)

            plot_group = QGroupBox("Plot Options")
            plot_layout = QVBoxLayout(plot_group)
            self.plot_mode_group = QButtonGroup(self)
            mode_tips = (
                "One page per sweep (section), the power average of each file and an "
                "overlay of all files - the most pages.",
                "Only the power average of each file and the overlay of all files.",
                "Only the power average of each file (fewest pages).")
            for index, text in enumerate(("All plots (overlay, every section, averages)",
                                          "Average + Overlay", "Average only")):
                radio = QRadioButton(text)
                radio.setToolTip(mode_tips[index])
                radio.setChecked(index == 0)
                self.plot_mode_group.addButton(radio, index)
                plot_layout.addWidget(radio)
            layout.addWidget(plot_group)

            split = QGroupBox("Split Saving")
            split_form = QFormLayout(split)
            self.split_box = QCheckBox("Save several projects of at most N section plots each")
            self.split_box.setToolTip(
                "Save the pages as several smaller projects (each opens faster "
                "in Veusz).")
            split_form.addRow(self.split_box)
            self.plots_spin = QSpinBox()
            self.plots_spin.setToolTip("Largest number of section plots in one project when splitting.")
            self.plots_spin.setRange(1, 1000000)
            self.plots_spin.setValue(1000)
            split_form.addRow("Plots per project:", self.plots_spin)
            self.base_edit = QLineEdit("RnS_Plots_Batch")
            self.base_edit.setToolTip("File name stem of the split projects (_1, _2, ... are added).")
            split_form.addRow("Base file name:", self.base_edit)
            for widget in (self.plots_spin, self.base_edit):
                widget.setEnabled(False)
                self.split_box.toggled.connect(widget.setEnabled)
            layout.addWidget(split)

            self.status_text = QTextEdit()
            self.status_text.setToolTip("Progress and messages of this session, with time stamps.")
            self.status_text.setReadOnly(True)
            self.status_text.setMinimumHeight(110)
            layout.addWidget(self.status_text, 1)

            widget, self.plot_panel = suite.make_plot_panel(self.theme == "dark")
            self.tabs.addTab(widget, "Plots")
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            outer.addWidget(self.progress_bar)
            buttons = QHBoxLayout()
            self.plot_button = QPushButton("Process and Plot")
            self.plot_button.setToolTip(
                "Read the listed .sft files and build the pages (shown on the "
                "Plots tab).")
            self.plot_button.clicked.connect(self._process_and_plot)
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
            close = QPushButton("Close")
            close.setToolTip("Close this window (unsaved projects are lost).")
            close.clicked.connect(self.close)
            for button in (self.plot_button, self.save_button, self.veusz_button, close):
                buttons.addWidget(button)
            outer.addLayout(buttons)

        # ----- helpers ----------------------------------------------------------
        def _log_message(self, message: str) -> None:
            """Append ``message`` with an HH:MM:SS time stamp to the log."""
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.append("[%s] %s" % (stamp, message))

        def _set_theme(self, mode: str) -> None:
            """Apply and save the suite theme and recolour the Plots tab."""
            self.theme = suite.apply_theme(mode)
            if self.plot_panel is not None:
                self.plot_panel.apply_theme(self.theme == "dark")
            self._log_message("Theme set to %s mode" % self.theme)

        def _gpu_toggled(self, on: bool) -> None:
            """Untick GPU averaging (with a log line) when CuPy or a CUDA device is
            missing.
            """
            if on and _cupy() is None:
                self._log_message("GPU averaging: CuPy with a CUDA device is not available; "
                                  "the CPU is used")
                self.enable_gpu_checkbox.setChecked(False)
            else:
                self._log_message("GPU averaging %s" % ("on" if on else "off"))

        def options(self) -> PlotOptions:
            """PlotOptions from the GUI (plot mode, GPU)."""
            return PlotOptions(mode=max(0, self.plot_mode_group.checkedId()),
                               use_gpu=self.enable_gpu_checkbox.isChecked())

        def _browse_files(self) -> None:
            """Ask for .sft files and add them to the list."""
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select SFT Files", "", "R&S SFT Files (*.sft *.SFT);;All Files (*)")
            self.add_files(files)

        def add_files(self, files: Sequence[str]) -> None:
            """Add the new ``files`` (duplicates ignored) and refresh the list.
            """
            new = [f for f in files if f not in self.selected_files]
            self.selected_files.extend(new)
            self.file_list_widget.clear()
            self.file_list_widget.addItems([os.path.basename(f) for f in self.selected_files])
            if new:
                self._log_message("Added %d file(s)" % len(new))

        def _clear_files(self) -> None:
            """Empty the file list."""
            self.selected_files.clear()
            self.file_list_widget.clear()
            self._log_message("File list cleared")

        # ----- processing -------------------------------------------------------
        def _process_and_plot(self) -> None:
            """Start reading the selected files in a FileProcessingThread."""
            if not self.selected_files:
                QMessageBox.warning(self, APP_NAME, "Please select SFT files first.")
                return
            if self.thread is not None and self.thread.isRunning():
                return
            self.plot_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            workers = self.cpu_spinbox.value() if self.enable_mp_checkbox.isChecked() else 1
            self.thread = FileProcessingThread(self.selected_files, workers)
            self.thread.progress_updated.connect(self.progress_bar.setValue)
            self.thread.processing_finished.connect(self._on_processing_finished)
            self.thread.start()
            self._log_message("Processing %d file(s), mode: %s"
                              % (len(self.selected_files), MODE_NAMES[self.options().mode]))

        def _on_processing_finished(self, files: List[SFTFile], errors: List[str]) -> None:
            """Log errors and warnings, build the project and show it on the Plots
            tab.
            """
            self.progress_bar.setVisible(False)
            self.plot_button.setEnabled(True)
            for error in errors:
                self._log_message("FAILED " + error)
            if not files:
                QMessageBox.warning(self, APP_NAME, "No file could be read. See the log.")
                return
            try:
                self.plotter = build_project(files, self.options())
            except Exception as exc:
                QMessageBox.critical(self, APP_NAME, "Plot creation failed:\n%s" % exc)
                return
            self.files = files
            for message in self.plotter.messages:
                self._log_message(message)
            self._log_message("Project ready: %d file(s), %d sweep(s), %d page(s)." % (
                len(files), sum(len(f.traces) for f in files), len(self.plotter.doc.Root.children)))
            self.save_button.setEnabled(True)
            if self.plot_panel is not None:
                try:
                    pages = self.plot_panel.set_document(self.plotter.doc)
                    self._log_message("Plots tab: %d page(s)." % pages)
                    self.tabs.setCurrentIndex(1)
                except Exception as exc:
                    self._log_message("Plots tab could not draw the project: %s" % exc)
            if errors:
                QMessageBox.warning(self, APP_NAME, "%d file(s) failed. See the log." % len(errors))

        def _save_project(self) -> None:
            """Save the project: one file, or - with "split" - several projects of at
            most N sections each in a chosen folder; then offer to open the first in
            Veusz.
            """
            if self.plotter is None:
                QMessageBox.information(self, APP_NAME, "Process files first.")
                return
            if self.split_box.isChecked():
                folder = QFileDialog.getExistingDirectory(self, "Folder for the split projects")
                if not folder:
                    return
                base = os.path.join(folder, self.base_edit.text().strip() or "RnS_Plots_Batch")
                try:
                    written = save_split(self.files, base, self.plots_spin.value(), self.options())
                except Exception as exc:
                    QMessageBox.critical(self, APP_NAME, "Save failed:\n%s" % exc)
                    return
                for path in written:
                    self._log_message("Saved %s" % path)
                self.last_saved = written[0] if written else None
            else:
                path, _ = QFileDialog.getSaveFileName(self, "Save Veusz Project", "",
                                                      "Veusz HDF5 Project (*.vszh5)")
                if not path:
                    return
                try:
                    self.last_saved = self.plotter.doc.save(path)
                except Exception as exc:
                    QMessageBox.critical(self, APP_NAME, "Save failed:\n%s" % exc)
                    return
                self._log_message("Project saved: %s" % self.last_saved)
            self.veusz_button.setEnabled(bool(self.last_saved))
            if self.last_saved and QMessageBox.question(
                    self, "Open in Veusz", "Open the saved project in Veusz?") \
                    == QMessageBox.StandardButton.Yes:
                self._open_in_veusz()

        def _open_in_veusz(self) -> None:
            """Open the last saved project in Veusz."""
            if not self.last_saved:
                QMessageBox.information(self, APP_NAME, "Save a project first.")
                return
            suite.open_in_veusz(self, self.last_saved)

        def _export(self) -> None:
            """Export the pages of the last saved project with Veusz (PDF/PNG).
            """
            if not self.last_saved:
                QMessageBox.information(self, APP_NAME, "Save a project first.")
                return
            suite.export_with_veusz(self, self.last_saved, self._log_message)

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
            """Wait up to 5 s for a running read, clear the plots, then close.
            """
            if self.thread is not None and self.thread.isRunning():
                self.thread.wait(5000)
            if self.plot_panel is not None:
                self.plot_panel.clear()
            super().closeEvent(event)

    return QApplication, EnhancedMainWindow


# %% Main Application
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point; returns the exit code.

    With ``--output`` the files are plotted in batch (no GUI; ``--mode``,
    ``--split``, ``--workers``, ``--gpu``) into that project; otherwise the
    GUI opens with the files in its list.
    """
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--output", help="write this .vszh5 without opening the GUI")
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), default=0,
                        help="0 all plots, 1 average + overlay, 2 average only")
    parser.add_argument("--split", type=int, default=0,
                        help="with --output: projects of at most N section plots")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu", action="store_true", help="average with CuPy")
    parser.add_argument("--version", action="version", version="%s %s" % (APP_NAME, APP_VERSION))
    args = parser.parse_args(argv)
    monitor = None
    if os.environ.get("AUTOPLOT_HANG_MONITOR") == "1":
        from Hanging_Threads import start_monitoring
        monitor = start_monitoring()
    try:
        if args.output:
            files, errors = read_files(args.files, args.workers)
            for error in errors:
                print("FAILED", error, file=sys.stderr)
            if not files:
                return 1
            options = PlotOptions(args.mode, args.gpu)
            if args.split > 0:
                for path in save_split(files, args.output, args.split, options):
                    print("Wrote", path)
            else:
                plotter = build_project(files, options)
                print("\n".join(plotter.messages))
                print("Wrote", plotter.doc.save(args.output))
            return 0 if not errors else 2
        QApplication, Window = _build_gui_classes()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = Window()
        window.add_files(args.files)
        window.show()
        return app.exec()
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
