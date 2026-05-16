# -*- coding: utf-8 -*-
"""
=============================================================================
Franks_AutoPlot.py
-----------------------------------------------------------------------------
# %% Header Info

GUI-driven batch processor for the "FranksProcessed" ASCII time-series data
files distributed alongside the NRAO 1PPS sync FITS bundle.  Two file
flavours are recognised automatically:

    * ``YYYY_MM.t00new`` - GB Timing Data (medians over 60 min), columns:
          MJD  date  time  Site1Hz  TAC  MicroSem  CNS_GPS1  MSM-GPS
          CNS_GPS  CNS2018  GBT_VLBA  RA  GBTRtn

    * ``time_gbt.dat``   - long-term GBT timing log, columns:
          MJD  EECO-REF  NIST-REF  NS  DATE  [COMMENTS]

Both flavours are read as numpy column arrays, pushed into a Veusz embedded
document as one dataset per column tagged with the file's base name (plus
``raw`` and ``sorted`` tags), and plotted column-vs-MJD on dedicated pages.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Initial implementation, FranksProcessed file support
Date: 2026-05-16
# %%%% 0.0.2: Added "Generate MJD->date strings" checkbox.  Whenever the
#             parsed file contains an MJD column, ticking the option emits
#             two additional Veusz text datasets (raw + sorted) of
#             YYYY-MM-DD_HH:MM:SS strings tagged with the file's base name
#             plus ``datestr``.
Date: 2026-05-16
# %%%% 0.0.3: NaN-preserving emission policy.  When a Franks token is
#             missing or non-numeric the parser now records ``NaN`` in the
#             numeric column (length-preserving) AND keeps the text token
#             ("" for fully missing) in the text companion -- rows are
#             never dropped.  Veusz numeric datasets carry the NaN through;
#             non-finite MJDs in the optional date-string text datasets
#             become the sentinel string ``"NaN"`` so all per-row arrays
#             retain the same length and stay index-aligned.
Date: 2026-05-16
# %%%%% Function Descriptions
        main: build QApplication and open the AutoPlot main window.
        FranksAutoPlotWindow: qtpy main window with the Touchstone-style
            file list / options / log / progress layout and the View menu
            dark/light theme toggle.
        FranksBatchWorker: QThread that parses selected files in parallel.
        parse_franks_file: format-autodetecting reader.  Returns a dict of
            column-shaped numpy arrays plus the sort key.
        push_franks_to_veusz: push column arrays into the embedded Veusz
            document, tagged with filename + raw/sorted, and create one
            scatter plot per numeric column versus MJD.
# %%%%% Variable Descriptions
        MAX_THREADS: top-of-file thread-pool size knob.
        DEFAULT_RSS_HIGH_WATER_MB: RSS threshold for memmap spill.
        T00_COLUMNS / TIMEDAT_COLUMNS: hardcoded column lists for the two
            file flavours (used when the on-file header is malformed).
# %%%%% More Info
        Because Veusz cannot natively import these custom ASCII formats
        (mixed date + numeric columns), we always import the raw text via
        Python (numpy.loadtxt) and produce derived "sorted" datasets in
        Veusz; both the raw and sorted variants are kept so the user can
        compare them, and both are tagged accordingly.
=============================================================================
"""
from __future__ import annotations

# ============================================================================
# IMPORTS - Standard library
# ============================================================================
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# IMPORTS - Scientific
# ============================================================================
import numpy as np

# ============================================================================
# IMPORTS - Veusz embedded
# ============================================================================
import veusz.embed as vz_embed

# ============================================================================
# IMPORTS - shared GUI helpers
# ============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _autoplot_common import (   # noqa: E402
    Qt, QThread, Signal,
    QApplication, QFileDialog, QFormLayout, QLabel, QMessageBox,
    QSpinBox, QComboBox, QCheckBox, QLineEdit,
    AutoPlotMainWindow,
    MemoryAwareCache, MemoryMonitor, MemoryMonitorConfig,
    open_embedded, save_vszh5, run_in_threadpool, safe_dsname,
    mjd_to_datestr,
)

# ============================================================================
# SCRIPT-LEVEL KNOBS
# ============================================================================
MAX_THREADS = max(1, (os.cpu_count() or 4))
DEFAULT_RSS_HIGH_WATER_MB = 1024

T00_COLUMNS = ["MJD", "date", "time", "Site1Hz", "TAC", "MicroSem",
               "CNS_GPS1", "MSM_GPS", "CNS_GPS", "CNS2018",
               "GBT_VLBA", "RA", "GBTRtn"]

TIMEDAT_COLUMNS = ["MJD", "EECO_REF", "NIST_REF", "NS", "DATE"]


# ============================================================================
# FILE PARSER
# ============================================================================
def _detect_flavour(path: str) -> str:
    name = os.path.basename(path).lower()
    if name.endswith(".t00new") or ".t00new." in name:
        return "t00new"
    if "time_gbt" in name:
        return "time_gbt"
    # try sniff
    with open(path, "rt", errors="replace") as fh:
        head = fh.readline()
    if "GB Timing Data" in head:
        return "t00new"
    return "time_gbt"


def _safe_columns(raw_names: List[str]) -> List[str]:
    out = []
    for n in raw_names:
        n2 = re.sub(r"[^A-Za-z0-9_]", "_", n.strip())
        if not n2:
            n2 = "col%d" % len(out)
        if n2[0].isdigit():
            n2 = "c_" + n2
        out.append(n2)
    return out


def parse_franks_file(path: str, cache: MemoryAwareCache) -> Dict[str, Any]:
    """
    Read one Franks-style ASCII file into a dict of column arrays.

    Returns
    -------
    {
      'flavour'      : 't00new' or 'time_gbt'
      'columns'      : OrderedDict-like {colname: np.ndarray}
      'sort_key'     : 'MJD'
      'header_lines' : list[str]
      'base_name'    : str
    }
    """
    flavour = _detect_flavour(path)
    base = os.path.basename(path)
    base = re.sub(r"\.(t00new|dat)(\..*)?$", "", base)
    base = safe_dsname(base)

    header_lines: List[str] = []
    columns: Dict[str, np.ndarray] = {}

    with open(path, "rt", errors="replace") as fh:
        lines = fh.readlines()

    # find data start
    if flavour == "t00new":
        # line[0]=TITLE, line[1]=column names, line[2+]=data
        header_lines = [lines[0].rstrip()] if lines else []
        colnames_line = lines[1] if len(lines) > 1 else ""
        names = _safe_columns(colnames_line.split())
        if len(names) < 13:
            names = T00_COLUMNS[:]
        data_lines = lines[2:]
    else:  # time_gbt
        # line[0]=column names, line[1]=separator '====', line[2+]=data
        header_lines = [lines[0].rstrip()] if lines else []
        colnames_line = lines[0] if lines else ""
        names = _safe_columns(colnames_line.split())
        # Drop the COMMENTS column from the parsing target since most rows
        # have no comment text and the header has 6 names with often only
        # 5 numeric tokens per row.
        names = names[:5] if len(names) >= 5 else names
        data_lines = lines[2:]

    # Parse rows: keep MJD as float, parse numeric columns where possible,
    # keep text columns (dates / times) as strings.
    numeric_columns: Dict[str, List[float]] = {}
    text_columns: Dict[str, List[str]] = {}
    for n in names:
        text_columns[n] = []
        numeric_columns[n] = []

    for ln in data_lines:
        s = ln.rstrip()
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        # Handle the embedded "==" separator (time_gbt) that contains '='
        if "====" in s:
            continue
        toks = s.split()
        # Don't run past the names list (extra COMMENTS tokens in time_gbt).
        for i, n in enumerate(names):
            if i >= len(toks):
                text_columns[n].append("")
                numeric_columns[n].append(np.nan)
                continue
            tok = toks[i]
            text_columns[n].append(tok)
            try:
                numeric_columns[n].append(float(tok))
            except ValueError:
                numeric_columns[n].append(np.nan)

    # Decide which columns are usefully numeric (mostly not-nan) vs text.
    # NaN preservation: numeric columns keep their NaN floats verbatim --
    # we never strip rows.  Veusz numeric datasets natively support NaN
    # (samples are skipped at plot time only).  Text columns also keep
    # every row (missing tokens recorded as "" earlier in this loop).
    for n in names:
        arr_num = np.asarray(numeric_columns[n], dtype=float)
        finite_frac = np.isfinite(arr_num).mean() if arr_num.size else 0.0
        if finite_frac > 0.5:
            columns[n] = cache.store("%s.%s" % (base, n), arr_num)
        else:
            arr_txt = np.asarray(text_columns[n])
            columns[n] = cache.store("%s.%s.txt" % (base, n), arr_txt)

    return {
        "flavour": flavour,
        "columns": columns,
        "sort_key": "MJD" if "MJD" in columns else None,
        "header_lines": header_lines,
        "base_name": base,
    }


# ============================================================================
# VEUSZ INTEGRATION
# ============================================================================
def push_franks_to_veusz(doc, data: Dict[str, Any], log_cb=None,
                         emit_datestr: bool = False) -> None:
    """Push parsed Franks columns into the running embedded Veusz document."""
    base = data["base_name"]
    cols = data["columns"]
    sort_key = data["sort_key"]

    raw_names: List[str] = []
    sorted_names: List[str] = []

    # Compute sort permutation once
    if sort_key is not None and sort_key in cols:
        sort_idx = np.argsort(np.asarray(cols[sort_key]))
    else:
        sort_idx = None

    for cname, arr in cols.items():
        arr = np.asarray(arr)
        raw_name = "%s__%s__raw" % (base, safe_dsname(cname))
        sorted_name = "%s__%s__sorted" % (base, safe_dsname(cname))
        try:
            if arr.dtype.kind in ("U", "S", "O"):
                values = [v.decode("ascii", "replace") if isinstance(v, bytes) else str(v)
                          for v in arr]
                doc.SetDataText(raw_name, values)
                if sort_idx is not None:
                    doc.SetDataText(sorted_name, [values[i] for i in sort_idx])
                else:
                    doc.SetDataText(sorted_name, values)
            else:
                # NaN-preserving: Veusz handles NaN natively in numeric
                # datasets, so we push the float array through unchanged.
                fa = np.ascontiguousarray(arr, dtype=float)
                doc.SetData(raw_name, fa)
                if sort_idx is not None:
                    doc.SetData(sorted_name, np.ascontiguousarray(fa[sort_idx]))
                else:
                    doc.SetData(sorted_name, fa)
        except Exception as exc:
            if log_cb:
                log_cb("  SetData failed for %s: %s" % (cname, exc))
            continue
        raw_names.append(raw_name)
        sorted_names.append(sorted_name)

    # Tag everything
    try:
        doc.TagDatasets(base, raw_names + sorted_names)
        doc.TagDatasets("raw", raw_names)
        doc.TagDatasets("sorted", sorted_names)
    except Exception:
        pass

    # ---- Optional MJD -> date-string text datasets -----------------------
    if emit_datestr and sort_key is not None and sort_key in cols:
        arr_mjd = np.asarray(cols[sort_key])
        if arr_mjd.dtype.kind in ("f", "i", "u"):
            try:
                date_strings = mjd_to_datestr(arr_mjd.astype(float))
                raw_ds = "%s__%s__datestr" % (base, safe_dsname(sort_key))
                sorted_ds = "%s__%s__datestr_sorted" % (base, safe_dsname(sort_key))
                doc.SetDataText(raw_ds, list(date_strings))
                if sort_idx is not None:
                    doc.SetDataText(sorted_ds,
                                    list(np.asarray(date_strings)[sort_idx]))
                else:
                    doc.SetDataText(sorted_ds, list(date_strings))
                try:
                    doc.TagDatasets(base, [raw_ds, sorted_ds])
                    doc.TagDatasets("datestr", [raw_ds, sorted_ds])
                except Exception:
                    pass
                if log_cb:
                    log_cb("  Date-string datasets created (2)")
            except Exception as exc:
                if log_cb:
                    log_cb("  MJD->date conversion failed: %s" % exc)

    # Build pages: one page named after the file, one graph per numeric col vs MJD
    page = doc.Root.Add("page", name=safe_dsname(base))
    try:
        page.notes.val = "\n".join(data.get("header_lines", []))
    except Exception:
        pass

    grid = page.Add("grid", columns=2)
    x_ds = "%s__%s__sorted" % (base, safe_dsname(sort_key)) if sort_key else None

    for cname in cols.keys():
        if cname == sort_key:
            continue
        arr = np.asarray(cols[cname])
        if arr.dtype.kind not in ("f", "i", "u"):
            # skip text columns in plotting
            continue
        ds = "%s__%s__sorted" % (base, safe_dsname(cname))
        graph = grid.Add("graph", name=safe_dsname("g_%s" % cname))
        try:
            graph.x.label.val = sort_key or "index"
            graph.y.label.val = cname
            graph.x.GridLines.hide.val = False
            graph.y.GridLines.hide.val = False
        except Exception:
            pass
        xy = graph.Add("xy", name=safe_dsname("xy_%s" % cname))
        if x_ds is not None:
            xy.xData.val = x_ds
        xy.yData.val = ds
        try:
            xy.marker.val = "circle"
            xy.markerSize.val = "2pt"
            xy.PlotLine.hide.val = True
        except Exception:
            pass

    if log_cb:
        log_cb("  Page built with %d graphs" %
               sum(1 for c in cols if c != sort_key
                   and np.asarray(cols[c]).dtype.kind in ("f", "i", "u")))


# ============================================================================
# WORKER THREAD
# ============================================================================
class FranksBatchWorker(QThread):
    """Background QThread that parses Franks files in parallel."""

    progress = Signal(int, int, str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, files: List[str], max_threads: int,
                 cache: MemoryAwareCache, parent=None) -> None:
        super().__init__(parent)
        self.files = files
        self.max_threads = max_threads
        self.cache = cache

    def run(self) -> None:
        try:
            work = [(p, parse_franks_file, (p, self.cache)) for p in self.files]
            results = run_in_threadpool(
                work, max_workers=self.max_threads,
                progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit("%s\n%s" % (exc, traceback.format_exc()))


# ============================================================================
# MAIN WINDOW
# ============================================================================
class FranksAutoPlotWindow(AutoPlotMainWindow):
    """Top-level window for the Franks AutoPlot script."""

    def __init__(self) -> None:
        super().__init__("FranksProcessed AutoPlot - Veusz Embedded GUI",
                         default_mode="dark")
        self._file_filter = (
            "Franks files (*.t00new *.dat *.bak* *.OrigBack* *.back);;All files (*)"
        )
        self.cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=DEFAULT_RSS_HIGH_WATER_MB
        ))
        self.monitor = MemoryMonitor(
            self.cache,
            callback=lambda rss: self.log(
                "RSS crossed high-water mark (%.1f MiB) -- spilling to disk." % rss
            )
        )
        self.monitor.start()
        self.veusz_doc = None
        self.worker = None
        self.log("MAX_THREADS = %d" % MAX_THREADS)

    # ----- options form ----------------------------------------------------
    def _populate_options(self, form: QFormLayout) -> None:
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

    # ----- run -------------------------------------------------------------
    def _process_files(self) -> None:
        if not self.selected_files:
            QMessageBox.information(self, "No files",
                                    "Select one or more Franks data files first.")
            return
        self.cache.cfg.rss_high_water_mb = int(self.rss_spin.value())

        if self.veusz_doc is None and self.show_embed_cb.isChecked():
            self.veusz_doc = open_embedded("Franks_AutoPlot")
        elif self.veusz_doc is None:
            self.veusz_doc = vz_embed.Embedded("Franks_AutoPlot", hidden=True)

        self.process_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.selected_files))
        self.progress_bar.setValue(0)

        self.worker = FranksBatchWorker(
            self.selected_files, int(self.thread_spin.value()),
            self.cache, parent=self
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.log("Starting batch: %d file(s), threads=%d" %
                 (len(self.selected_files), int(self.thread_spin.value())))
        self.worker.start()

    def _on_progress(self, done: int, total: int, key: str) -> None:
        self.progress_bar.setValue(done)
        self.log("  parsed [%d/%d] %s" % (done, total, os.path.basename(key)))
        self.update_mem_label(self.cache.rss_mb())

    def _on_failed(self, msg: str) -> None:
        self.log("Batch failed: %s" % msg)
        QMessageBox.critical(self, "Batch failed", msg)
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_done(self, results: Dict[str, Any]) -> None:
        emit_datestr = bool(self.datestr_cb.isChecked())
        for path, data in results.items():
            if isinstance(data, Exception):
                self.log("  ERROR processing %s: %s" % (path, data))
                continue
            self.log("Inserting datasets for %s" % os.path.basename(path))
            try:
                push_franks_to_veusz(self.veusz_doc, data, log_cb=self.log,
                                     emit_datestr=emit_datestr)
            except Exception as exc:
                self.log("  push failed: %s" % exc)
        self.log("Batch complete.")
        self.progress_bar.setVisible(False)
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
        try:
            written = save_vszh5(self.veusz_doc, fn)
            self.log("Saved %s" % written)
            QMessageBox.information(self, "Saved", "Wrote %s" % written)
        except Exception as exc:
            self.log("Save failed: %s" % exc)
            QMessageBox.critical(self, "Save failed", str(exc))

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
# ENTRY POINT
# ============================================================================
def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = FranksAutoPlotWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
