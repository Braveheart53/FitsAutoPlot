# -*- coding: utf-8 -*-
"""
=============================================================================
FITS_AutoPlot.py
-----------------------------------------------------------------------------
# %% Header Info

Replacement for the older FITS_AutoPlot.py in the FitsAutoPlot repo.
GUI-driven batch processor for NRAO Green Bank FITS files (OnePpsDeltas
samplers and any other FITS BinTable / image HDUs) that produces a Veusz
3.4 / 4.1 compatible ``.vszh5`` project file with:

    * column-oriented sorted datasets (one Veusz dataset per FITS column),
    * tagged datasets (filename tag + raw/sorted tag),
    * appropriate plots per HDU (scatter for tables, image for image HDUs),
    * a live embedded Veusz preview while the file is built.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: Initial port of FITS_AutoPlot to the Touchstone-style GUI
Date: 2026-05-16
# %%%%% Function Descriptions
        main: build QApplication, open AutoPlot main window, run event loop.
        FITSAutoPlotWindow: qtpy main window subclassing AutoPlotMainWindow;
            adds the FITS-specific processing-options form (import backend
            choice, thread count, RSS high-water mark) and orchestrates the
            batch run through a worker QThread.
        FITSBatchWorker: QThread that runs the parallel batch read using
            ``run_in_threadpool``; emits per-file progress signals.
        FITSProcessor: per-file reader.  Supports both Veusz native FITS
            import and astropy fallback; transparently handles ``.fits`` and
            ``.fits.gz`` files; produces a normalised dict of column-shaped
            arrays keyed by ``(hdu_name, column_name)`` or ``(hdu_name,
            "IMAGE")`` for image HDUs.
        push_to_veusz: install one file's worth of column datasets into the
            running embedded Veusz document, with raw/sorted tagging and
            per-HDU plot creation.
# %%%%% Variable Descriptions
        MAX_THREADS: top-of-file knob for the worker thread pool size.
        DEFAULT_RSS_HIGH_WATER_MB: RSS threshold for memmap spill.
        SORTED_KEY_HINT: ordered list of preferred sort keys (DMJD first).
# %%%%% More Info
        FITS files from NRAO Green Bank carry a single BinTableHDU named
        ``OnePpsDeltas`` with columns DMJD (1D), CHANNELA (8A), CHANNELB
        (8A), DELTAT (J).  We expose every column as its own Veusz dataset
        plus an additional "sorted" pass (sorted by DMJD) so derived plots
        match the convention used in the NRAO 1PPS documentation.  The
        sorted datasets are produced from the raw arrays inside Python
        because Veusz's internal expressions cannot reorder a structured
        BinTable on import; the raw datasets are still imported so the
        user can pivot back to the unsorted representation if desired.
=============================================================================
"""
from __future__ import annotations

# ============================================================================
# IMPORTS - Standard library
# ============================================================================
import gzip
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# IMPORTS - Scientific
# ============================================================================
import numpy as np
from astropy.io import fits
from astropy.table import QTable

# ============================================================================
# IMPORTS - Veusz embedded (deferred to runtime in workers, OK at module
#           level here because the standalone GUI is also a Veusz client)
# ============================================================================
import veusz.embed as vz_embed

# ============================================================================
# IMPORTS - shared GUI / threading / cache helpers
# ============================================================================
# Allow this file to be launched directly even if _autoplot_common.py lives
# in the same folder but the folder is not on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _autoplot_common import (   # noqa: E402
    Qt, QThread, Signal,
    QApplication, QFileDialog, QFormLayout, QLabel, QMessageBox,
    QSpinBox, QComboBox, QCheckBox, QLineEdit,
    AutoPlotMainWindow,
    MemoryAwareCache, MemoryMonitor, MemoryMonitorConfig,
    apply_theme, open_embedded, save_vszh5,
    run_in_threadpool, open_maybe_gzipped, safe_dsname,
)

# ============================================================================
# SCRIPT-LEVEL KNOBS  (kept at the top per spec)
# ============================================================================
MAX_THREADS = max(1, (os.cpu_count() or 4))   # used by ThreadPoolExecutor
DEFAULT_RSS_HIGH_WATER_MB = 1024              # spill threshold per process
DEFAULT_BACKEND = "both"                      # 'veusz' | 'astropy' | 'both'
SORTED_KEY_HINT = ["DMJD", "MJD", "TIME", "TIMESTAMP", "JD"]
ALLOWED_EXT = (".fits", ".fit", ".fits.gz", ".fit.gz")


# ============================================================================
# PER-FILE PROCESSOR
# ============================================================================
class FITSProcessor:
    """
    Read a single FITS file (compressed or not) and return a normalised dict
    of column-shaped numpy arrays plus minimal metadata.

    Two backends are supported:

        * ``astropy``  -- uses ``astropy.io.fits``; works for any HDU.
        * ``veusz``    -- uses ``veusz.embed.Embedded.ImportFileFITS`` which
                          imports each column as its own dataset.  We mirror
                          the resulting column data back into the same dict
                          structure so the caller can attach raw/sorted
                          datasets uniformly.

    The two backends are not redundant: ``veusz`` keeps the link to the
    original FITS file (if requested) so the project file follows updates
    to the source; ``astropy`` is the fallback path for image HDUs and for
    NRAO files where Veusz's importer skips header-only HDUs.

    Per the spec, when Veusz's internal import does not yield the desired
    column representation we additionally produce the column arrays
    ourselves via astropy and tag them as 'sorted'.
    """

    def __init__(self, backend: str, cache: MemoryAwareCache,
                 linked: bool = False) -> None:
        if backend not in ("veusz", "astropy", "both"):
            raise ValueError("backend must be 'veusz', 'astropy' or 'both'")
        self.backend = backend
        self.cache = cache
        self.linked = linked

    # ------------------------------------------------------------------
    def _open(self, path: str):
        # astropy.io.fits transparently opens .gz, but we still need a
        # local copy on disk for Veusz's importer if the user picked the
        # veusz backend on a gzipped file (Veusz cannot open .fits.gz).
        if path.lower().endswith(".gz"):
            tmp = tempfile.NamedTemporaryFile(prefix="autoplot_fits_",
                                              suffix=".fits", delete=False)
            tmp.close()
            with gzip.open(path, "rb") as src, open(tmp.name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return tmp.name, True
        return path, False

    # ------------------------------------------------------------------
    def read(self, path: str) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:

            'columns'     : {(hdu_name, col_name) : np.ndarray}  -- 1D table cols
            'images'      : {hdu_name             : np.ndarray}  -- nD images
            'sort_key'    : (hdu_name, col_name) or None
            'header'      : list[str]  -- pretty-printed PRIMARY header lines
            'fits_for_vz' : path to a non-gzipped FITS file usable by Veusz
            'tmp_uncompressed' : bool
        """
        local_path, did_decompress = self._open(path)
        base = os.path.splitext(os.path.basename(path))[0]
        if base.endswith(".fits"):
            base = base[:-5]
        out = {
            "columns": {},
            "images": {},
            "sort_key": None,
            "header": [],
            "fits_for_vz": local_path,
            "tmp_uncompressed": did_decompress,
            "base_name": base,
        }
        with fits.open(local_path, memmap=True) as hdul:
            # Collect a compact header dump from PRIMARY
            try:
                out["header"] = [
                    "%s = %r" % (k, hdul[0].header[k])
                    for k in list(hdul[0].header.keys())[:20]
                ]
            except Exception:
                pass

            sort_key: Optional[Tuple[str, str]] = None
            for i, hdu in enumerate(hdul):
                hname = hdu.name or "HDU%d" % i
                if hdu.data is None:
                    continue
                if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
                    qt = QTable.read(hdu)
                    for col in qt.colnames:
                        try:
                            data = np.asarray(qt[col])
                        except Exception:
                            data = np.asarray(hdu.data[col])
                        # store via cache (will memmap-spill if needed)
                        key = (hname, col)
                        out["columns"][key] = self.cache.store(
                            "%s.%s.%s" % (base, hname, col), data
                        )
                        if sort_key is None:
                            for hint in SORTED_KEY_HINT:
                                if col.upper() == hint:
                                    sort_key = (hname, col)
                                    break
                else:
                    # Image / primary array HDU
                    data = np.asarray(hdu.data)
                    out["images"][hname] = self.cache.store(
                        "%s.%s.IMAGE" % (base, hname), data
                    )
            out["sort_key"] = sort_key
        return out


# ============================================================================
# VEUSZ INTEGRATION  (runs on the GUI thread)
# ============================================================================
def push_to_veusz(doc, file_path: str, data: Dict[str, Any],
                  backend: str, log_cb=None) -> None:
    """
    Push a single processed FITS file into the running embedded Veusz
    document.

    Strategy
    --------
    1.  If backend in ('veusz', 'both'): call ``ImportFileFITS`` so the
        raw FITS columns/images appear as Veusz datasets, tagged with the
        base filename + 'raw'.  This keeps the Veusz-internal link to the
        FITS file when ``linked=True``.
    2.  If backend in ('astropy', 'both') or the FITS file was gzipped:
        push python-level numpy column arrays as their own Veusz datasets,
        tagged with the base filename + 'sorted' (sorted by DMJD when the
        column is present, otherwise plain copies).
    3.  Build one Veusz page per HDU containing scatter plots for table
        HDUs (DELTAT vs DMJD, etc.) and image widgets for image HDUs.
    """
    base = safe_dsname(data["base_name"])
    raw_names: List[str] = []
    sorted_names: List[str] = []

    # ---- 1) Veusz native import (raw datasets) ---------------------------
    if backend in ("veusz", "both") and not data["tmp_uncompressed"]:
        try:
            existing = set(doc.GetDatasets())
            doc.ImportFileFITS(
                filename=data["fits_for_vz"],
                items=["/"],          # import every HDU/column
                namemap={},
                linked=False,         # we already point at a real file
            )
            added = [n for n in doc.GetDatasets() if n not in existing]
            for n in added:
                # rename with base prefix so multi-file runs don't collide
                new_name = "%s__%s__raw" % (base, safe_dsname(n))
                try:
                    doc.RenameDataset(n, new_name)
                except Exception:
                    new_name = n
                raw_names.append(new_name)
            doc.TagDatasets(base, raw_names)
            doc.TagDatasets("raw", raw_names)
            if log_cb:
                log_cb("  Veusz native import: %d raw datasets" % len(raw_names))
        except Exception as exc:
            if log_cb:
                log_cb("  Veusz native import failed: %s" % exc)

    # ---- 2) Astropy/column-oriented sorted datasets ----------------------
    if backend in ("astropy", "both") or backend == "veusz" and data["tmp_uncompressed"]:
        sort_key = data["sort_key"]
        sort_idx_by_hdu: Dict[str, np.ndarray] = {}
        if sort_key is not None:
            hkey, ckey = sort_key
            sort_idx_by_hdu[hkey] = np.argsort(np.asarray(data["columns"][sort_key]))

        for (hname, col), arr in data["columns"].items():
            arr = np.asarray(arr)
            if hname in sort_idx_by_hdu:
                arr_s = arr[sort_idx_by_hdu[hname]]
            else:
                arr_s = arr
            ds_name = "%s__%s__%s__sorted" % (base, safe_dsname(hname), safe_dsname(col))
            # SetData refuses to accept bytes columns; convert to str list
            if arr_s.dtype.kind in ("S", "U", "O"):
                arr_s = np.asarray([
                    v.decode("ascii", "replace") if isinstance(v, bytes) else str(v)
                    for v in arr_s
                ])
                try:
                    doc.SetDataText(ds_name, list(arr_s))
                except Exception:
                    # Older Veusz: drop text columns silently
                    continue
            else:
                doc.SetData(ds_name, np.ascontiguousarray(arr_s, dtype=float))
            sorted_names.append(ds_name)

        for hname, img in data["images"].items():
            img = np.asarray(img)
            ds_name = "%s__%s__IMAGE" % (base, safe_dsname(hname))
            try:
                doc.SetData2D(ds_name, np.ascontiguousarray(img, dtype=float))
            except Exception:
                doc.SetData(ds_name, np.ascontiguousarray(img.ravel(), dtype=float))
            sorted_names.append(ds_name)

        if sorted_names:
            doc.TagDatasets(base, sorted_names)
            doc.TagDatasets("sorted", sorted_names)
        if log_cb:
            log_cb("  Column datasets created: %d" % len(sorted_names))

    # ---- 3) Build the plots ---------------------------------------------
    _build_pages(doc, base, data, sorted_names)

    # cleanup any temporary uncompressed copy
    if data["tmp_uncompressed"]:
        try:
            os.remove(data["fits_for_vz"])
        except OSError:
            pass


def _build_pages(doc, base: str, data: Dict[str, Any],
                 sorted_names: List[str]) -> None:
    """Create one Veusz page per HDU with the appropriate plot widgets."""
    # Group sorted datasets by HDU
    by_hdu: Dict[str, List[Tuple[str, str]]] = {}
    for (hname, col) in data["columns"].keys():
        ds = "%s__%s__%s__sorted" % (base, safe_dsname(hname), safe_dsname(col))
        if ds in sorted_names:
            by_hdu.setdefault(hname, []).append((col, ds))

    for hname, cols in by_hdu.items():
        page = doc.Root.Add("page", name=safe_dsname("%s_%s" % (base, hname)))
        try:
            page.notes.val = "\n".join(data.get("header", []))
        except Exception:
            pass
        grid = page.Add("grid", columns=2)
        # x-axis defaults to the sort key column (DMJD/MJD/etc.) when present
        x_col = None
        for c, ds in cols:
            if c.upper() in [k.upper() for k in SORTED_KEY_HINT]:
                x_col = (c, ds)
                break
        if x_col is None:
            # fall back to the first numeric column
            x_col = cols[0]
        x_name = x_col[1]

        for c, ds in cols:
            if ds == x_name:
                continue
            graph = grid.Add("graph", name=safe_dsname("g_%s" % c))
            try:
                graph.x.label.val = x_col[0]
                graph.y.label.val = c
                graph.x.GridLines.hide.val = False
                graph.y.GridLines.hide.val = False
            except Exception:
                pass
            xy = graph.Add("xy", name=safe_dsname("xy_%s" % c))
            xy.xData.val = x_name
            xy.yData.val = ds
            xy.marker.val = "circle"
            try:
                xy.markerSize.val = "2pt"
                xy.PlotLine.hide.val = True
                xy.MarkerFill.color.val = "blue"
                xy.MarkerLine.color.val = "blue"
            except Exception:
                pass

    # Image pages
    for hname, img in data["images"].items():
        ds_name = "%s__%s__IMAGE" % (base, safe_dsname(hname))
        page = doc.Root.Add("page", name=safe_dsname("%s_%s_img" % (base, hname)))
        graph = page.Add("graph", name="g_img")
        try:
            image_w = graph.Add("image", name="imageIn",
                                data=ds_name, colorMap="plasma")
            graph.Add("colorbar", direction="vertical",
                      name="colorbar1", widgetName="imageIn")
        except Exception:
            # Older Veusz API: omit colorbar
            graph.Add("image", name="imageIn", data=ds_name)


# ============================================================================
# WORKER THREAD (parallel read)
# ============================================================================
class FITSBatchWorker(QThread):
    """Background QThread that reads a batch of FITS files concurrently."""

    progress = Signal(int, int, str)   # done, total, last_key
    log = Signal(str)
    finished_ok = Signal(dict)         # {path: read_dict}
    failed = Signal(str)

    def __init__(self, files: List[str], backend: str, max_threads: int,
                 cache: MemoryAwareCache, parent=None) -> None:
        super().__init__(parent)
        self.files = files
        self.backend = backend
        self.max_threads = max_threads
        self.cache = cache

    def run(self) -> None:
        try:
            proc = FITSProcessor(self.backend, self.cache)
            work = [(p, proc.read, (p,)) for p in self.files]
            results = run_in_threadpool(
                work,
                max_workers=self.max_threads,
                progress_cb=lambda d, t, k: self.progress.emit(d, t, k),
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit("%s\n%s" % (exc, traceback.format_exc()))


# ============================================================================
# MAIN WINDOW
# ============================================================================
class FITSAutoPlotWindow(AutoPlotMainWindow):
    """Top-level window for the FITS AutoPlot script."""

    def __init__(self) -> None:
        super().__init__("FITS AutoPlot - Veusz Embedded GUI", default_mode="dark")
        self._file_filter = (
            "FITS files (*.fits *.fit *.fits.gz *.fit.gz);;All files (*)"
        )
        self.cache = MemoryAwareCache(MemoryMonitorConfig(
            rss_high_water_mb=DEFAULT_RSS_HIGH_WATER_MB,
        ))
        self.monitor = MemoryMonitor(
            self.cache,
            callback=lambda rss: self.log("RSS crossed high-water mark "
                                          "(%.1f MiB) -- spilling to disk." % rss),
        )
        self.monitor.start()
        self.veusz_doc = None
        self.worker = None
        self.log("MAX_THREADS = %d" % MAX_THREADS)

    # ----- options form ----------------------------------------------------
    def _populate_options(self, form: QFormLayout) -> None:
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["both (Veusz + astropy)",
                                     "veusz (native FITS import)",
                                     "astropy"])
        form.addRow(QLabel("Import backend:"), self.backend_combo)

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

    # ----- run -------------------------------------------------------------
    def _process_files(self) -> None:
        if not self.selected_files:
            QMessageBox.information(self, "No files", "Select one or more FITS files first.")
            return
        backend_idx = self.backend_combo.currentIndex()
        backend = ["both", "veusz", "astropy"][backend_idx]
        self.cache.cfg.rss_high_water_mb = int(self.rss_spin.value())

        if self.veusz_doc is None and self.show_embed_cb.isChecked():
            self.veusz_doc = open_embedded("FITS_AutoPlot")
        elif self.veusz_doc is None:
            # invisible doc: still required to build the project
            self.veusz_doc = vz_embed.Embedded("FITS_AutoPlot", hidden=True)

        self.process_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.selected_files))
        self.progress_bar.setValue(0)

        self.worker = FITSBatchWorker(
            self.selected_files, backend, int(self.thread_spin.value()),
            self.cache, parent=self
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_failed)
        self.log("Starting batch: %d file(s), backend=%s, threads=%d" %
                 (len(self.selected_files), backend, int(self.thread_spin.value())))
        self.worker.start()

    def _on_worker_progress(self, done: int, total: int, key: str) -> None:
        self.progress_bar.setValue(done)
        self.log("  read [%d/%d] %s" % (done, total, os.path.basename(key)))
        self.update_mem_label(self.cache.rss_mb())

    def _on_worker_failed(self, msg: str) -> None:
        self.log("Batch failed: %s" % msg)
        QMessageBox.critical(self, "Batch failed", msg)
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_worker_done(self, results: Dict[str, Any]) -> None:
        backend_idx = self.backend_combo.currentIndex()
        backend = ["both", "veusz", "astropy"][backend_idx]
        for path, data in results.items():
            if isinstance(data, Exception):
                self.log("  ERROR processing %s: %s" % (path, data))
                continue
            self.log("Inserting datasets for %s" % os.path.basename(path))
            try:
                push_to_veusz(self.veusz_doc, path, data, backend, log_cb=self.log)
            except Exception as exc:
                self.log("  push_to_veusz failed for %s: %s" % (path, exc))
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

    # ----- shutdown --------------------------------------------------------
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
    win = FITSAutoPlotWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
