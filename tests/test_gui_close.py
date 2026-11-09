# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_gui_close.py

Closing and deleting the windows must never crash (regression test for the
Python 3.12 / PySide6 6.12 segmentation fault of 2026-11-09).

The cause was the memory-monitor thread: it held lambdas that referred to the
window or plugin dialog, so that thread could release the last reference and
destroy Qt widgets outside the GUI thread; it also wrote to the log widget
from its own thread.  These tests make the monitor fire from its thread (RSS
threshold 0 MiB), fill and page through the Plots tab, run the garbage
collector in a second thread, then close and delete every window - several
times in a row.

Run:  QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests

Created on 2026-11-09

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First tests: windows, Plots tab, plugin dialogs, monitor thread.
Date: 2026-11-09
# %%%%% Function Descriptions
        wait_for: Process Qt events until ``condition()`` is true or ``timeout`` s pass.
        delete_now: Close ``widget`` and delete its C++ object in the GUI thread, now.
        _NoVeusz (class): Import hook that keeps Veusz out of these tests.
        GcHammer (class): Background thread that runs gc.collect() in a loop until stopped.
        write_fits: Small OnePpsDeltas-like FITS file with a 7-day hole; returns its path.
        WindowCloseTests (class): Open, use and close every window type several times without a crash.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP = QApplication.instance() or QApplication([])
        CYCLES = 2
# %%%%% More Info
    See README.md and REQUIREMENTS.md.
=============================================================================
"""
# %% Imports
import gc
import importlib.abc
import os
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _NoVeusz(importlib.abc.MetaPathFinder):
    """Make ``import veusz`` fail: the programs must close cleanly without Veusz,
    and test discovery imports this module before test_native_export (whose
    "no Veusz imported" check would otherwise see the plugins' Veusz import)."""

    def find_spec(self, fullname, path=None, target=None):
        """Raise ImportError for veusz and its submodules; defer the rest."""
        if fullname == "veusz" or fullname.startswith("veusz."):
            raise ImportError("Veusz deliberately unavailable during GUI close tests")
        return None


if not any(isinstance(f, _NoVeusz) or type(f).__name__ == "NoVeusz" for f in sys.meta_path):
    sys.meta_path.insert(0, _NoVeusz())

import numpy as np                                       # noqa: E402
from astropy.io import fits                              # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent         # noqa: E402

from _autoplot_common import (                           # noqa: E402
    QApplication, MemoryAwareCache, MemoryMonitor, MemoryMonitorConfig,
)
from _autoplot_native import NativeDocument              # noqa: E402
import FITS_AutoPlot as fitstool                         # noqa: E402
import Franks_AutoPlot as frankstool                     # noqa: E402
import FITS_AutoPlot_plugin as fitsplugin                # noqa: E402
import Franks_AutoPlot_plugin as franksplugin            # noqa: E402

APP = QApplication.instance() or QApplication([])
CYCLES = 2


# %% Helpers
def wait_for(condition, timeout=5.0):
    """Process Qt events until ``condition()`` is true or ``timeout`` s pass."""
    end = time.time() + timeout
    while time.time() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def delete_now(widget):
    """Close ``widget`` and delete its C++ object in the GUI thread, now."""
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    APP.processEvents()


class GcHammer(threading.Thread):
    """Background thread that runs gc.collect() in a loop until stopped.

    The cyclic garbage collector runs in whichever thread allocates; this
    makes "Qt objects freed outside the GUI thread" likely if the program
    still left Qt objects for the collector.
    """

    def __init__(self):
        """Daemon thread; call stop() to end it."""
        super().__init__(daemon=True)
        self.halt = threading.Event()
        self.runs = 0

    def run(self):
        """Collect garbage every 2 ms until stop()."""
        while not self.halt.is_set():
            gc.collect()
            self.runs += 1
            self.halt.wait(0.002)

    def stop(self):
        """End the loop and wait for the thread."""
        self.halt.set()
        self.join(5.0)


def write_fits(folder):
    """Small OnePpsDeltas-like FITS file with a 7-day hole; returns its path."""
    x = np.array([60805, 60805.1, 60805.2, 60812, 60812.1, 60812.2])
    cols = [fits.Column(name="DMJD", format="D", array=x),
            fits.Column(name="DELTAT", format="D", unit="ns", array=[1, 2, np.nan, 3, 4, 5]),
            fits.Column(name="CHANNELA", format="8A", array=["A"] * 6),
            fits.Column(name="CHANNELB", format="8A", array=["B"] * 6)]
    path = os.path.join(folder, "close_test.fits")
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(cols, name="OnePpsDeltas")]).writeto(path)
    return path


# %% Tests
class WindowCloseTests(unittest.TestCase):
    """Open, use and close every window type several times without a crash."""

    @classmethod
    def setUpClass(cls):
        """One FITS file and its project, shared by the tests."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_fits(cls.tmp.name)
        cls.cache = MemoryAwareCache(MemoryMonitorConfig())
        cls.data = fitstool.FITSProcessor("astropy", cls.cache).read(cls.path)

    @classmethod
    def tearDownClass(cls):
        """Remove the cache files and the temporary folder."""
        cls.cache.cleanup()
        cls.tmp.cleanup()

    def setUp(self):
        """Start the garbage-collector thread."""
        self.hammer = GcHammer()
        self.hammer.start()

    def tearDown(self):
        """Stop the garbage-collector thread."""
        self.hammer.stop()

    def project(self):
        """A fresh NativeDocument with the per-file pages of the test file."""
        doc = NativeDocument("close test")
        fitstool.push_to_veusz(doc, self.path, self.data, "astropy", gap_absolute=1.0)
        return doc

    def test_main_windows_with_monitor_and_plots(self):
        """FITS and Franks windows: the monitor fires from its thread, the Plots tab
        pages through a project, the window closes; the monitor thread ends and
        does not keep the window alive."""
        for cls in (fitstool.FITSAutoPlotWindow, frankstool.FranksAutoPlotWindow):
            for cycle in range(CYCLES):
                with self.subTest(window=cls.__name__, cycle=cycle):
                    window = cls()
                    window.show()
                    window.cache.cfg.rss_high_water_mb = 0          # fire at once
                    self.assertTrue(wait_for(lambda: "RSS crossed" in window.status_text.toPlainText()),
                                    "monitor message not logged through the GUI thread")
                    pages = window.plot_panel.set_document(self.project())
                    for index in range(pages):
                        window.plot_panel.page_box.setCurrentIndex(index)
                        APP.processEvents()
                    monitor = window.monitor
                    delete_now(window)
                    self.assertFalse(monitor.is_alive(), "monitor thread still running after close")
                    # The monitor thread holds the window only weakly, so it can
                    # never be the one to release it (the 3.12 crash).
                    self.assertIsInstance(monitor._callback_ref, weakref.WeakMethod)
                    del window
                    gc.collect()

    def test_plugin_dialogs_with_monitor(self):
        """Plugin dialogs: a monitor bound to the dialog logs through the GUI
        thread and lets the dialog go once it is deleted."""
        for module in (fitsplugin, franksplugin):
            for cycle in range(CYCLES):
                with self.subTest(plugin=module.__name__, cycle=cycle):
                    dialog = module._PluginBatchDialog()
                    cache = MemoryAwareCache(MemoryMonitorConfig(rss_high_water_mb=0))
                    monitor = MemoryMonitor(cache, callback=dialog.rss_crossed)
                    monitor.start()
                    self.assertTrue(wait_for(lambda: "RSS over" in dialog.log.toPlainText()))
                    monitor.stop()
                    self.assertFalse(monitor.is_alive())
                    self.assertIsInstance(monitor._callback_ref, weakref.WeakMethod)
                    delete_now(dialog)
                    del dialog
                    gc.collect()
                    cache.cleanup()

    def test_other_windows_open_close(self):
        """ATR, CSV/TSV, R&S and Touchstone windows open and close repeatedly."""
        import ATR_AutoPlot
        import CSV_TSV_AutoPlot
        import RAndS_FSW_ASCII_Plotter
        import Touchstone_AutoPlot
        makers = [ATR_AutoPlot._build_gui_classes()[1],
                  CSV_TSV_AutoPlot._build_gui_classes()[1],
                  RAndS_FSW_ASCII_Plotter._build_gui_classes()[1],
                  Touchstone_AutoPlot.TouchstoneMainWindow]
        for make in makers:
            for cycle in range(CYCLES):
                with self.subTest(window=make.__name__, cycle=cycle):
                    window = make()
                    window.show()
                    APP.processEvents()
                    delete_now(window)
                    del window

    def test_monitor_stop_is_prompt_and_join_works(self):
        """stop() wakes the monitor at once and join() works (the old '_stop'
        attribute shadowed threading.Thread._stop and broke join())."""
        cache = MemoryAwareCache(MemoryMonitorConfig())
        cache.cfg.poll_interval_sec = 30.0
        monitor = MemoryMonitor(cache)
        monitor.start()
        start = time.time()
        monitor.stop()
        self.assertFalse(monitor.is_alive())
        self.assertLess(time.time() - start, 2.0)
        monitor.join(1.0)
        monitor.stop()                                          # twice is fine
        cache.cleanup()


if __name__ == "__main__":
    unittest.main()
