# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_clear_and_add.py

Clear and add in every plot generator (2026-11-16).  Clear forgets the files,
the data read from them, the project (nothing cleared is saved later), the
Plots tab, the caches and spill files, and gives the memory back to the
operating system.  Adding files without Clear keeps the data already read
(only new or changed files are read) and the next Process builds the project
from every listed file: the same project as all files processed at once.

Run:  python -m unittest discover -s tests

Created on 2026-11-16

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version.
Date: 2026-11-16
#              KeptReads, release_memory, NativeDocument.release,
#              MemoryAwareCache.clear; the clear / add steps in FITS and
#              Franks AutoPlot (synthetic files) and in ATR, R&S, Touchstone
#              and CSV/TSV AutoPlot (example files).
# %%%%% Function Descriptions
        dispose: Close ``widget`` and delete its C++ widget tree now.
        wait_for: Pump Qt events until ``condition()`` is true (or ``limit`` seconds).
        project_objects: Every HDF5 dataset of a saved project (bytes) and its script lines (save stamp left out).
        KeptReadsTests (class): The store of data read from listed files.
        MemoryTests (class): release_memory, NativeDocument.release and the cache.
        WindowTests (class): Clear and add in every program's window.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        EXAMPLES: example data folder
        APP: the QApplication of the tests (None without Qt)
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import glob
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "Example_Data_Files"
sys.path.insert(0, str(ROOT))

from _autoplot_native import (KeptReads, NativeDocument, process_rss_mb,  # noqa: E402
                              release_memory)

try:
    from qtpy.QtWidgets import QApplication, QMessageBox
    APP = QApplication.instance() or QApplication([])
except Exception:                           # pragma: no cover - no Qt
    APP = None


# %% Helpers
def dispose(widget):
    """Close ``widget`` and delete its C++ widget tree now."""
    from qtpy.QtCore import QCoreApplication, QEvent
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    APP.processEvents()


def wait_for(condition, limit=300.0):
    """Pump Qt events until ``condition()`` is true (or ``limit`` seconds)."""
    start = time.time()
    while not condition() and time.time() - start < limit:
        APP.processEvents()
        time.sleep(0.01)
    APP.processEvents()


def project_objects(path):
    """Every HDF5 dataset of a saved project (bytes) and its script lines
    (save stamp left out)."""
    import h5py
    out = {}
    with h5py.File(str(path), "r") as handle:
        def visit(name, obj):
            """Record one HDF5 dataset."""
            if isinstance(obj, h5py.Dataset) and name != "Veusz/Document/document":
                value = obj[()]
                out[name] = (value.tobytes() if hasattr(value, "tobytes") and obj.dtype.kind != "O"
                             else repr(value.tolist() if hasattr(value, "tolist") else value))
        handle.visititems(visit)
        script = handle["Veusz/Document/document"][0].decode().splitlines()
    return out, [line for line in script if not line.startswith("# Saved")]


# %% Kept data
class KeptReadsTests(unittest.TestCase):
    """The store of data read from listed files."""

    def test_read_keep_change_gone(self):
        """New files are read once; a changed file or other options are read
        again; a file gone from disk keeps its data; clear forgets."""
        with tempfile.TemporaryDirectory() as folder:
            a, b = os.path.join(folder, "a.txt"), os.path.join(folder, "b.txt")
            for path in (a, b):
                with open(path, "w") as handle:
                    handle.write("1\n")
            kept = KeptReads()
            self.assertEqual(kept.to_read([a, b, a]), [a, b])
            kept.keep(a, {"v": 1})
            self.assertEqual(kept.to_read([a, b]), [b])
            self.assertEqual(kept.to_read([a], options=("other",)), [a])
            stamp = os.stat(a).st_mtime_ns
            with open(a, "w") as handle:
                handle.write("1\n2\n")
            os.utime(a, ns=(stamp + 10**9, stamp + 10**9))
            self.assertEqual(kept.to_read([a]), [a])
            kept.keep(a, {"v": 2})
            os.remove(a)
            self.assertEqual(kept.to_read([a]), [])          # gone: not forgotten
            self.assertEqual(kept.items([b, a]), [(a, {"v": 2})])
            kept.keep(a, None)                                # a failed read forgets
            self.assertNotIn(a, kept)
            kept.keep(b, {"v": 3})
            kept.clear()
            self.assertEqual(len(kept), 0)


# %% Memory
class MemoryTests(unittest.TestCase):
    """release_memory, NativeDocument.release and the cache."""

    def test_release_memory(self):
        """The RSS falls back after freed heap blocks are handed back (glibc
        keeps them otherwise); the result names the steps."""
        before_alloc = process_rss_mb()
        blocks = [np.ones(10000) for _ in range(5000)]       # 80 kB each, ~400 MB
        del blocks
        result = release_memory()
        self.assertEqual(sorted(result), ["after_mb", "before_mb", "collected", "steps"])
        if sys.platform.startswith("linux") and "malloc_trim" in result["steps"]:
            self.assertLess(result["after_mb"], before_alloc + 100.0)
        logs = []
        release_memory(logs.append)
        self.assertTrue(logs and logs[0].startswith("Memory released: RSS"))

    def test_document_release(self):
        """A released project is empty and still usable."""
        doc = NativeDocument("t")
        doc.SetData("x", np.arange(5.0))
        page = doc.Root.Add("page", name="p")
        page.Add("graph", name="g")
        doc.release()
        self.assertEqual((len(doc.datasets), len(doc.Root.children), doc.tags), (0, 0, {}))
        doc.SetData("y", [1.0])
        self.assertIn("y", doc.datasets)

    def test_cache_clear_deletes_spill_files(self):
        """Spilled arrays: memmaps closed, files deleted on clear()."""
        from _autoplot_common import MemoryAwareCache, MemoryMonitorConfig
        cache = MemoryAwareCache(MemoryMonitorConfig(rss_high_water_mb=0, array_min_bytes=1))
        try:
            value = cache.store("k", np.arange(1000.0))
            self.assertIsInstance(value, np.memmap)
            files = os.listdir(cache.cache_dir)
            self.assertEqual(len(files), 1)
            del value
            cache.clear()
            self.assertEqual(os.listdir(cache.cache_dir), [])
            self.assertEqual(cache.keys(), [])
        finally:
            cache.cleanup()


# %% Windows
@unittest.skipIf(APP is None, "Qt not available")
class WindowTests(unittest.TestCase):
    """Clear and add in every program's window."""

    def setUp(self):
        """Temporary folder; message boxes do not block."""
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.boxes = []
        self._patches = [patch.object(QMessageBox, name, side_effect=lambda *a, **k: self.boxes.append(a))
                         for name in ("information", "warning", "critical")]
        for item in self._patches:
            item.start()

    def tearDown(self):
        """Undo the patches, delete the folder."""
        for item in self._patches:
            item.stop()
        self.tmp.cleanup()

    # ----- synthetic inputs ----------------------------------------------------
    def fits_files(self):
        """Four small OnePpsDeltas-style FITS files, one day apart."""
        from astropy.io import fits
        paths = []
        for day in range(4):
            x = 60805.0 + day + np.arange(50) / 1440.0
            table = fits.BinTableHDU.from_columns(
                [fits.Column(name="DMJD", format="D", array=x),
                 fits.Column(name="DELTAT", format="D", unit="ns", array=np.sin(x * 50.0))],
                name="OnePpsDeltas")
            path = self.folder / ("2026_01_%02d_00_00_00.fits" % (day + 1))
            fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
            paths.append(str(path))
        return paths

    def franks_files(self):
        """Four small .t00new-style Franks files, one month each."""
        paths = []
        for month in range(4):
            path = self.folder / ("2026_%02d.t00new" % (month + 1))
            rows = ["%.5f %d %.3f %.3f" % (60805.0 + 30 * month + i / 24.0, i, np.sin(i), np.cos(i))
                    for i in range(48)]
            path.write_text("MJD SEQ CNS2018 MicroSem\n" + "\n".join(rows) + "\n", encoding="utf-8")
            paths.append(str(path))
        return paths

    # ----- FITS / Franks (shared window) ---------------------------------------------
    def check_shared_window(self, window, files, save):
        """All at once == A then add B; Clear empties everything; B only."""
        def process():
            """Click Process and wait for the batch."""
            window.save_button.setEnabled(False)
            window._process_files()
            wait_for(lambda: window.save_button.isEnabled())

        def selected(paths):
            """Put ``paths`` into the list."""
            window.selected_files = list(paths)
            window._refresh_file_list()

        half = len(files) // 2
        selected(files)
        process()
        save(window.veusz_doc, self.folder / "all.vszh5")
        window._clear_files()
        selected(files[:half])
        process()
        selected(files)                       # add B without Clear
        reads = []
        original = window.files_to_read
        with patch.object(window, "files_to_read",
                          side_effect=lambda options: reads.append(original(options)) or reads[-1]):
            process()
        self.assertEqual(reads, [files[half:]])          # only B was read
        save(window.veusz_doc, self.folder / "added.vszh5")
        one, script_one = project_objects(self.folder / "all.vszh5")
        two, script_two = project_objects(self.folder / "added.vszh5")
        self.assertGreater(len(one), 4 * len(files))     # every file made datasets
        self.assertEqual(one, two)
        self.assertEqual(script_one, script_two)
        process()                             # nothing new: rebuilt from kept data
        save(window.veusz_doc, self.folder / "again.vszh5")
        self.assertEqual(project_objects(self.folder / "again.vszh5"), (one, script_one))
        window._clear_files()
        self.assertIsNone(window.veusz_doc)
        self.assertEqual(len(window.kept_reads), 0)
        self.assertEqual(window.selected_files, [])
        self.assertFalse(window.save_button.isEnabled())
        if window.plot_panel is not None:
            self.assertEqual(window.plot_panel.page_box.count(), 0)
        self.assertEqual(window.cache.keys(), [])
        selected(files[half:])
        process()
        save(window.veusz_doc, self.folder / "b_only.vszh5")
        _data, script = project_objects(self.folder / "b_only.vszh5")
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0].split(".")[0]
            found = [line for line in script if stem in line]
            if path in files[:half]:
                self.assertFalse(found, stem)               # cleared: gone
            else:
                self.assertTrue(found, stem)                # listed: there

    def test_fits(self):
        """FITS AutoPlot: clear and add."""
        import FITS_AutoPlot as fits_tool
        window = fits_tool.FITSAutoPlotWindow()
        window.process_reader_cb.setChecked(False)
        try:
            self.check_shared_window(window, self.fits_files(),
                                     lambda doc, path: fits_tool.save_vszh5(doc, str(path)))
        finally:
            dispose(window)

    def test_franks(self):
        """Franks AutoPlot: clear and add."""
        import Franks_AutoPlot as franks_tool
        window = franks_tool.FranksAutoPlotWindow()
        window.process_reader_cb.setChecked(False)
        try:
            self.check_shared_window(window, self.franks_files(),
                                     lambda doc, path: franks_tool.save_vszh5(doc, str(path)))
        finally:
            dispose(window)

    # ----- ATR, R&S, Touchstone, CSV/TSV -------------------------------------------
    def check_window(self, window, files, add, process, save, project, kept, save_button, clear):
        """The same steps for the programs with their own window."""
        half = len(files) // 2
        add(window, files)
        process(window)
        save(window, self.folder / "all.vszh5")
        clear(window)
        add(window, files[:half])
        process(window)
        add(window, files[half:])
        process(window)
        save(window, self.folder / "added.vszh5")
        self.assertEqual(project_objects(self.folder / "all.vszh5"),
                         project_objects(self.folder / "added.vszh5"))
        clear(window)
        self.assertIsNone(project(window))
        self.assertEqual(kept(window), 0)
        self.assertFalse(save_button(window).isEnabled())
        if window.plot_panel is not None:
            self.assertEqual(window.plot_panel.page_box.count(), 0)
        add(window, files[half:])
        process(window)
        save(window, self.folder / "b_only.vszh5")
        _data, script = project_objects(self.folder / "b_only.vszh5")
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0].replace("-", "_")
            found = [line for line in script if stem in line.replace("-", "_")]
            if path in files[:half]:
                self.assertFalse(found, stem)               # cleared: gone
            else:
                self.assertTrue(found, stem)                # listed: there

    def test_atr(self):
        """ATR AutoPlot: clear and add."""
        import ATR_AutoPlot as atr
        files = sorted(glob.glob(str(EXAMPLES / "atr" / "Band_1_XPol*.ATR")))
        if len(files) < 2:
            self.skipTest("example ATR files missing")
        _app, window_class = atr._build_gui_classes()
        window = window_class()

        def process(w):
            """Process and wait."""
            w._process()
            wait_for(lambda: not (w.thread is not None and w.thread.isRunning()))

        def save(w, path):
            """Save the project."""
            w.plotter.save(str(path))
        try:
            self.check_window(window, files, lambda w, f: w.add_files(f), process, save,
                              lambda w: w.plotter, lambda w: len(w.kept),
                              lambda w: w.save_button, lambda w: w._clear_files())
        finally:
            dispose(window)

    def test_rns(self):
        """R&S plotter: clear and add."""
        import RAndS_FSW_ASCII_Plotter as rns
        files = sorted(glob.glob(str(EXAMPLES / "sft" / "*.sft")))
        if len(files) < 2:
            self.skipTest("example .sft files missing")
        _app, window_class = rns._build_gui_classes()
        window = window_class()

        def process(w):
            """Process and wait."""
            w._process_and_plot()
            wait_for(lambda: not (w.thread is not None and w.thread.isRunning()))

        def save(w, path):
            """Save the project."""
            w.plotter.doc.save(str(path))
        try:
            self.check_window(window, files, lambda w, f: w.add_files(f), process, save,
                              lambda w: w.plotter, lambda w: len(w.kept),
                              lambda w: w.save_button, lambda w: w._clear_files())
        finally:
            dispose(window)

    def test_touchstone(self):
        """Touchstone AutoPlot: clear and add."""
        import Touchstone_AutoPlot as ts
        files = sorted(glob.glob(str(EXAMPLES / "Touchstone" / "s2p" / "*.s2p")))[:4]
        if len(files) < 2:
            self.skipTest("example .s2p files missing")
        window = ts.TouchstoneMainWindow()

        def add(w, paths):
            """Add files as Browse does."""
            for path in paths:
                if path not in w.selected_files:
                    w.selected_files.append(path)
            w._update_file_list()

        def process(w):
            """Process and wait."""
            w._process_files()
            wait_for(lambda: w._worker is None)

        def save(w, path):
            """Save the project."""
            w.touchstone_plotter.save(str(path))
        try:
            self.check_window(window, files, add, process, save,
                              lambda w: w.touchstone_plotter,
                              lambda w: len(w.kept) + len(w.processed_data) + len(w.td_results),
                              lambda w: w.save_button, lambda w: w._clear_files())
        finally:
            dispose(window)

    def test_csv(self):
        """CSV/TSV AutoPlot: clear and add."""
        import CSV_TSV_AutoPlot as csvp
        files = []
        for index in range(4):
            path = self.folder / ("run%d.csv" % index)
            rows = ["%d,%.3f" % (i, np.sin(i + index)) for i in range(20)]
            path.write_text("Freq,Level\n" + "\n".join(rows) + "\n", encoding="utf-8")
            files.append(str(path))
        _app, window_class = csvp._build_gui_classes()
        window = window_class()

        def process(w):
            """Choose the columns and generate."""
            w.x_column_combo.setCurrentText("Freq")
            for i in range(w.y_columns_list.count()):
                item = w.y_columns_list.item(i)
                item.setSelected(item.text() == "Level")
            w.generate_plots()
            APP.processEvents()

        def save(w, path):
            """Save the project."""
            w.plotter.save_project(str(path), False)
        try:
            self.check_window(window, files, lambda w, f: w.add_files(f), process, save,
                              lambda w: w.plotter, lambda w: len(w.loaded),
                              lambda w: w.save_btn, lambda w: w.clear_files())
        finally:
            dispose(window)


if __name__ == "__main__":
    unittest.main()
