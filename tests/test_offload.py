# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_offload.py

_autoplot_native 0.5.7: plot data offloaded to a pickle file on disk when
this computer's RAM in use (all programs) reaches the user's limit, read
back for the plots and converted into the .vszh5 on save.  Nothing is
written below the limit; every value comes back bit for bit; the saved
project and the Plots tab are the same as without offloading.

Run:  python -m unittest discover -s tests

Created on 2026-11-17

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version.
Date: 2026-11-17
#              The RAM reading and the limit; no file below the limit; every
#              dataset kind bit for bit; saved projects identical with and
#              without offloading (clock pages, synthetic FITS and Franks,
#              random datasets); the Plots tab identical; overwrite, drop,
#              release; kept file data; threads; the offload folder; too
#              little disk; the GUI control and --ram-limit.
# %%%%% Function Descriptions
        offload_always: Make ``doc`` offload at every check: its RAM limit far below any real use (the check reads the computer's real RAM use).
        EveryDataset (class): Context: OFFLOAD_MIN_BYTES 0, so even the smallest dataset goes to disk (the strongest test of the save and the plots).
        hdf_objects: {name: (attributes, bytes)} of every HDF5 object of a .vszh5 (the save stamps and the script left out) and the script lines without comments.
        random_project: A NativeDocument with every dataset kind and awkward values (NaN, inf, -0.0, subnormal, 2**53 + 1, unicode), one page drawing some of them.
        same_save: Save the project ``make()`` builds once in RAM and once offloaded; assert both files are identical; return the offloaded document.
        LimitTests (class): system_memory(), set_ram_limit(), the guard below and above the limit.
        RoundTripTests (class): Every dataset kind comes back bit for bit.
        SavedProjectTests (class): The saved .vszh5 is identical with and without offloading.
        PlotsTabTests (class): document_pages() and trace_arrays() see the same data.
        KeptReadsTests (class): Kept file data offloaded and read back.
        StoreTests (class): OffloadStore: threads, folder, disk space, release.
        OptionTests (class): RamLimitBox (isolated settings) and --ram-limit.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        TINY: a RAM limit (%) below any real use, so every check offloads
        APP: the QApplication of the Qt tests (kept alive)
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import argparse
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py                                                # noqa: E402
import vszh5_writer as w                                   # noqa: E402
import _autoplot_native as native                          # noqa: E402
from _autoplot_native import (KeptReads, NativeDocument, OffloadedParts,  # noqa: E402
                              OffloadStore, TimeLayout)
from test_time_clocks import clock_project                 # noqa: E402

TINY = 1e-9
APP = None


# %% Helpers
def offload_always(doc):
    """Make ``doc`` offload at every check: its RAM limit far below any real
    use (the check reads the computer's real RAM use)."""
    doc.ram_limit_percent = TINY
    doc._ram_checked = 0.0
    return doc


class EveryDataset(object):
    """Context: OFFLOAD_MIN_BYTES 0, so even the smallest dataset goes to
    disk (the strongest test of the save and the plots)."""

    def __enter__(self):
        """Lower the minimum to 0 bytes."""
        self.before = native.OFFLOAD_MIN_BYTES
        native.OFFLOAD_MIN_BYTES = 0

    def __exit__(self, *exc):
        """Restore the minimum."""
        native.OFFLOAD_MIN_BYTES = self.before


def hdf_objects(path):
    """{name: (attributes, bytes)} of every HDF5 object of a .vszh5 (the save
    stamps and the script left out) and the script lines without comments."""
    out = {}
    with h5py.File(str(path), "r") as handle:
        def visit(name, obj):
            """One object: attributes and data bytes."""
            attrs = {k: (v.tobytes() if hasattr(v, "tobytes") else repr(v))
                     for k, v in obj.attrs.items() if k not in ("vsz_saved_at", "vsz_writer")}
            data = b""
            if isinstance(obj, h5py.Dataset) and name != "Veusz/Document/document":
                data = obj[()].tobytes() if obj.dtype.kind != "O" else repr(obj[()].tolist())
            out[name] = (attrs, data)
        handle.visititems(visit)
        script = [line for line in handle["Veusz/Document/document"][0].decode().splitlines()
                  if not line.startswith("#")]
    return out, script


def random_project(seed=7):
    """A NativeDocument with every dataset kind and awkward values (NaN,
    inf, -0.0, subnormal, 2**53 + 1, unicode), one page drawing some of them."""
    rng = np.random.default_rng(seed)
    doc = NativeDocument("offload")
    n = 2000
    x = np.cumsum(rng.uniform(0.5, 1.5, n))
    x[700:] += 500.0                                       # a gap
    y = rng.normal(size=n)
    y[[3, 9]] = [np.nan, np.inf]
    y[10] = -0.0
    y[11] = 5e-324                                         # subnormal
    y[12] = float(2 ** 53 + 1)
    doc.SetData("x", x)
    doc.SetData("y", y, symerr=np.abs(y) * 0.1, negerr=-np.abs(y) * 0.2,
                poserr=np.abs(y) * 0.3)
    doc.SetDataDateTime("t", w.epoch_to_veusz(1762678740.0 + np.arange(n) * 6.0),
                        mjd=61349.5 + np.arange(n) * 6.0 / 86400.0)
    doc.SetDataText("label", ["µs", "", "naïve", "日本"] * 500)
    doc.SetData2D("img", rng.normal(size=(64, 48)), xrange=(0.0, 4.8), yrange=(-1.0, 1.0))
    doc.SetData2D("img_edges", rng.normal(size=(8, 5)), xedge=np.linspace(0, 1, 6),
                  yedge=np.geomspace(1, 100, 9))
    doc.SetData("small", [1.0, 2.0])                       # below OFFLOAD_MIN_BYTES
    doc.TagDatasets("source_test", ["x", "y", "t", "label"])
    page = doc.Root.Add("page", name="p")
    graph = page.Add("graph", name="g")
    graph.Add("xy", name="a", xData="x", yData="y")
    graph2 = page.Add("graph", name="g2")
    graph2.Add("xy", name="b", xData="t", yData="y")
    page.Add("graph", name="g3").Add("image", name="im", data="img")
    return doc


def same_save(test, make, layout=None):
    """Save the project ``make()`` builds once in RAM and once offloaded;
    assert both files are identical; return the offloaded document."""
    with tempfile.TemporaryDirectory() as folder:
        ram = make()
        ram.ram_limit_percent = 0.0
        if layout is not None:
            ram.time_layout = layout
        first = ram.save(os.path.join(folder, "ram.vszh5"))
        with EveryDataset():
            disk = offload_always(make())
            if layout is not None:
                disk.time_layout = layout
            second = disk.save(os.path.join(folder, "disk.vszh5"))
        test.assertEqual(disk.offloaded()[1], 0)          # every dataset is on disk
        a, b = hdf_objects(first), hdf_objects(second)
        test.assertEqual(sorted(a[0]), sorted(b[0]))
        for name in a[0]:
            test.assertEqual(a[0][name], b[0][name], name)
        test.assertEqual(a[1], b[1])
        on_disk, _in_ram, size = disk.offloaded()
        test.assertGreater(on_disk, 0)
        test.assertGreater(size, 0)
        ram.release()
    return disk


# %% Tests
class LimitTests(unittest.TestCase):
    """system_memory(), set_ram_limit(), the guard below and above the limit."""

    def tearDown(self):
        """Back to the default limit."""
        native.set_ram_limit(native.DEFAULT_RAM_LIMIT_PERCENT)

    def test_reading(self):
        """The computer's RAM: total > 0, 0 <= use <= 100, available <= total."""
        found = native.system_memory()
        self.assertIsNotNone(found)
        percent, total, available = found
        self.assertGreater(total, 0)
        self.assertTrue(0.0 <= percent <= 100.0)
        self.assertLessEqual(available, total)
        self.assertIn("GB of RAM", native.memory_text())
        if os.path.exists("/proc/meminfo"):
            # without psutil (Linux): /proc/meminfo gives the same reading
            import builtins
            real = builtins.__import__

            def no_psutil(name, *args, **kwargs):
                """Import as usual, but psutil is missing."""
                if name == "psutil":
                    raise ImportError(name)
                return real(name, *args, **kwargs)
            builtins.__import__ = no_psutil
            try:
                fallback = native.system_memory()
            finally:
                builtins.__import__ = real
            self.assertEqual(fallback[1], total)
            self.assertLess(abs(fallback[0] - percent), 5.0)

    def test_limit_values(self):
        """Clamped to 0 .. 100; None / NaN = never; default 80 %."""
        self.assertEqual(native.DEFAULT_RAM_LIMIT_PERCENT, 80.0)
        self.assertEqual(native.set_ram_limit(150), 100.0)
        self.assertEqual(native.set_ram_limit(-5), 0.0)
        self.assertEqual(native.set_ram_limit(None), 0.0)
        self.assertEqual(native.set_ram_limit(float("nan")), 0.0)
        self.assertEqual(native.set_ram_limit(55.5), 55.5)
        self.assertEqual(native.ram_limit(), 55.5)

    def test_nothing_below_the_limit(self):
        """Below the limit (or limit 0) no dataset moves and no file is made."""
        use = native.system_memory()[0]
        for limit in (0.0, min(99.99, use + 5.0)):
            if limit and limit <= use:
                continue                          # this computer is fuller than that
            doc = random_project()
            doc.ram_limit_percent = limit
            doc._ram_checked = 0.0
            doc.SetData("more", np.arange(1000.0))
            doc.finalize()
            self.assertIsNone(doc._offload)
            self.assertEqual(doc.offloaded(), (0, len(doc.datasets), 0))
            self.assertEqual(doc.offload_messages, [])

    def test_above_the_limit(self):
        """Above the limit every dataset of 512 bytes or more moves at the next
        check, with one message; the program-wide limit is used by default."""
        native.set_ram_limit(TINY)
        doc = NativeDocument("t")
        seen = []
        doc.offload_log = seen.append
        doc.SetData("a", np.arange(1000.0))
        self.assertIsInstance(doc.datasets["a"][1], OffloadedParts)
        doc.SetData("tiny", [1.0])
        self.assertIsInstance(doc.datasets["tiny"][1], dict)
        self.assertEqual(len(seen), 1)
        self.assertIn("RAM in use on this computer", seen[0])
        self.assertTrue(os.path.exists(doc._offload.path))
        # 200 more offloads within 10 s: held back, then one line with the
        # totals when finalize() ends (not 200 lines)
        for i in range(200):
            doc._ram_checked = 0.0
            doc.SetData("more%d" % i, np.arange(100.0))
        self.assertEqual(len(seen), 1)
        doc.finalize()
        self.assertEqual(len(seen), 2)
        self.assertIn("201 dataset(s)", seen[1])
        self.assertEqual(doc.offload_messages, seen)


class RoundTripTests(unittest.TestCase):
    """Every dataset kind comes back bit for bit."""

    def test_every_kind(self):
        """1-D with errors, date (and its exact MJD), text, 2-D with ranges
        and edges: identical bytes after offload; GetData / GetDataType work."""
        ram = random_project()
        disk = random_project()
        big = [n for n, (_k, parts) in disk.datasets.items()
               if native.payload_bytes(parts) >= native.OFFLOAD_MIN_BYTES]
        self.assertEqual(sorted(big), ["img", "label", "t", "x", "y"])   # img_edges: 440 B
        moved = disk.offload()
        self.assertEqual(moved, len(big) + 1)         # and the exact MJD array of "t"
        self.assertIsInstance(disk.mjd_of["t"], native.OffloadedValue)
        for name, (kind, parts) in ram.datasets.items():
            kind2, parts2 = disk.datasets[name]
            self.assertEqual(kind, kind2)
            self.assertEqual(disk.GetDataType(name), ram.GetDataType(name))
            loaded = native.loaded_parts(parts2)
            self.assertEqual(sorted(parts), sorted(loaded))
            for key, value in parts.items():
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.dtype, loaded[key].dtype)
                    self.assertEqual(value.tobytes(), loaded[key].tobytes(), (name, key))
                else:
                    self.assertEqual(value, loaded[key])
            for a, b in zip(ram.GetData(name), disk.GetData(name)):
                if a is None:
                    self.assertIsNone(b)
                elif isinstance(a, np.ndarray):
                    self.assertEqual(a.tobytes(), b.tobytes())
                else:
                    self.assertEqual(a, b)
        self.assertEqual(disk.mjd_of["t"].load().tobytes(), ram.mjd_of["t"].tobytes())
        self.assertIn("serr", disk.datasets["y"][1])
        self.assertIsNone(disk.datasets["x"][1].get("serr"))
        self.assertEqual(disk.offload(), 0)            # nothing left to move

    def test_overwrite_and_drop(self):
        """SetData over an offloaded dataset replaces it in RAM; dropping a
        plot copy deletes the entry; tags stay."""
        doc = random_project()
        doc.offload()
        doc.SetData("x", np.arange(5.0))
        self.assertIsInstance(doc.datasets["x"][1], dict)
        self.assertEqual(doc.GetData("x")[0].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertIn("y", doc.tags["source_test"])
        del doc.datasets["label"]
        self.assertNotIn("label", doc.datasets)


class SavedProjectTests(unittest.TestCase):
    """The saved .vszh5 is identical with and without offloading."""

    def setUp(self):
        """A fixed zone for the local time labels."""
        self.saved = os.environ.get(w.DISPLAY_TZ_ENV)
        os.environ[w.DISPLAY_TZ_ENV] = "America/New_York"

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self.saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self.saved

    def test_random_project(self):
        """Every kind of dataset, gaps and a local time axis."""
        disk = same_save(self, random_project)
        self.assertTrue(any("moved to" in m for m in disk.offload_messages))
        disk.release()

    def test_clock_pages(self):
        """Local, UTC and MJD pages (plot copies made while offloaded)."""
        same_save(self, lambda: clock_project()[0], TimeLayout(n=10.0, clocks="all")).release()

    def test_synthetic_fits_and_franks(self):
        """The synthetic FITS (tagged, untagged) and Franks projects of
        tests/make_demo_projects.py, built while offloading."""
        from test_native_export import (MemoryAwareCache, MemoryMonitorConfig, NativeTests,
                                        fitstool, frankstool)
        fixtures = NativeTests()
        fixtures.setUp()
        try:
            def fits(tagged):
                """Build the synthetic FITS project into a new document."""
                path = fixtures.write_fits(tagged)

                def make():
                    """A new document with the synthetic FITS pages."""
                    cache = MemoryAwareCache(MemoryMonitorConfig())
                    data = fitstool.FITSProcessor("astropy", cache).read(str(path))
                    doc = NativeDocument()
                    if make.offload:
                        offload_always(doc)
                    fitstool.push_to_veusz(doc, str(path), data, "astropy",
                                           datetime_duplicate=True, hide_keys=True,
                                           gap_absolute=1.0)
                    fitstool.build_unit_overlay_pages(
                        doc, [dict(data, base=data["base_name"])], datetime_duplicate=True,
                        hide_keys=True, gap_absolute=1.0)
                    cache.cleanup()
                    return doc
                return make

            def franks():
                """Build the synthetic Franks project into a new document."""
                path = fixtures.write_franks()

                def make():
                    """A new document with the synthetic Franks pages."""
                    cache = MemoryAwareCache(MemoryMonitorConfig())
                    data = frankstool.parse_franks_file(str(path), cache)
                    doc = NativeDocument()
                    if make.offload:
                        offload_always(doc)
                    frankstool.push_franks_to_veusz(doc, data, datetime_duplicate=True,
                                                    hide_keys=True, gap_absolute=1.0)
                    frankstool.build_unit_overlay_pages_franks(
                        doc, [dict(data, base=data["base_name"])], datetime_duplicate=True,
                        hide_keys=True, gap_absolute=1.0)
                    cache.cleanup()
                    return doc
                return make
            for builder in (fits(True), fits(False), franks()):
                with tempfile.TemporaryDirectory() as folder:
                    builder.offload = False
                    ram = builder()
                    ram.ram_limit_percent = 0.0
                    first = ram.save(os.path.join(folder, "ram.vszh5"))
                    builder.offload = True
                    with EveryDataset():
                        disk = builder()
                        second = disk.save(os.path.join(folder, "disk.vszh5"))
                    self.assertGreater(disk.offloaded()[0], 0)
                    self.assertEqual(disk.offloaded()[1], 0)
                    self.assertEqual(hdf_objects(first), hdf_objects(second))
                    ram.release()
                    disk.release()
        finally:
            fixtures.tearDown()


class PlotsTabTests(unittest.TestCase):
    """document_pages() and trace_arrays() see the same data."""

    def test_pages_and_arrays(self):
        """Every page, graph and trace array is the same offloaded or not."""
        from _autoplot_pyqtgraph import document_pages, trace_arrays
        os.environ[w.DISPLAY_TZ_ENV] = "America/New_York"
        try:
            ram = clock_project()[0]
            ram.time_layout = TimeLayout(n=10.0, clocks="all")
            with EveryDataset():
                disk = offload_always(clock_project()[0])
                disk.time_layout = TimeLayout(n=10.0, clocks="all")
                pages_ram, pages_disk = document_pages(ram), document_pages(disk)
            self.assertEqual(disk.offloaded()[1], 0)
        finally:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        self.assertGreater(disk.offloaded()[0], 0)
        self.assertEqual(pages_ram, pages_disk)
        for page in pages_ram:
            for graph in page["graphs"]:
                for trace in graph["traces"]:
                    for mjd_as_date in (True, False):
                        days = graph.get("x_clock") == "mjd"
                        a = trace_arrays(ram, trace, graph["x_kind"], mjd_as_date, days)
                        b = trace_arrays(disk, trace, graph["x_kind"], mjd_as_date, days)
                        self.assertEqual(a[0].tobytes(), b[0].tobytes())
                        self.assertEqual(a[1].tobytes(), b[1].tobytes())
                        self.assertEqual(a[2], b[2])
        disk.release()


class KeptReadsTests(unittest.TestCase):
    """Kept file data offloaded and read back."""

    def test_kept(self):
        """Above the limit keep() moves the data; get() / items() read them
        back equal; data that cannot be pickled stay in RAM; clear() deletes
        the file."""
        kept = KeptReads()
        kept.ram_limit_percent = TINY
        notes = []
        kept.offload_log = notes.append
        data = {"x": np.arange(10000.0), "name": "f1", "cols": {"a": np.ones(3, dtype=np.int64)}}
        kept.keep("f1.fits", data)
        self.assertIsInstance(kept._data["f1.fits"], native.OffloadedValue)
        back = kept.get("f1.fits")
        self.assertIsNot(back, data)
        self.assertEqual(back["x"].tobytes(), data["x"].tobytes())
        self.assertEqual(back["cols"]["a"].dtype, np.int64)
        self.assertEqual(back["name"], "f1")
        self.assertEqual([p for p, _d in kept.items(["f1.fits", "none"])], ["f1.fits"])
        kept.keep("f2.fits", {"handle": threading.Lock()})   # cannot be pickled
        self.assertIsInstance(kept._data["f2.fits"], dict)
        self.assertTrue(any("kept in RAM" in n for n in notes))
        path = kept._offload.path
        kept.clear()
        self.assertFalse(os.path.exists(path))
        self.assertEqual(len(kept), 0)


class StoreTests(unittest.TestCase):
    """OffloadStore: threads, folder, disk space, release."""

    def test_threads(self):
        """Eight threads writing and reading at once get their own values."""
        store = OffloadStore("autoplot_test_")
        errors = []

        def work(seed):
            """Write and read back 40 arrays."""
            rng = np.random.default_rng(seed)
            for _ in range(40):
                value = rng.normal(size=rng.integers(1, 3000))
                where = store.put(value)
                if store.get(*where).tobytes() != value.tobytes():
                    errors.append(seed)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(store.count, 320)
        path = store.path
        store.close()
        self.assertFalse(os.path.exists(path))
        with self.assertRaises(IOError):
            store.get(0, 10)

    def test_folder_and_disk(self):
        """AUTOPLOT_OFFLOAD_DIR is used; too little free space: nothing
        moves and the message says so."""
        with tempfile.TemporaryDirectory() as folder:
            os.environ[native.OFFLOAD_DIR_ENV] = folder
            try:
                doc = random_project()
                doc.offload()
                self.assertEqual(os.path.dirname(doc._offload.path), folder)
                doc.release()
                self.assertEqual(os.listdir(folder), [])
            finally:
                os.environ.pop(native.OFFLOAD_DIR_ENV, None)
        doc = random_project()
        doc._offload = OffloadStore("autoplot_test_")
        doc._offload.free_bytes = lambda: 1000.0
        self.assertEqual(doc.offload(), 0)
        self.assertIn("kept in RAM", doc.offload_messages[-1])
        self.assertEqual(doc.offloaded()[0], 0)

    def test_finalizer(self):
        """A store that is dropped without close() deletes its file."""
        store = OffloadStore("autoplot_test_")
        store.put(np.arange(10.0))
        path = store.path
        del store
        import gc
        gc.collect()
        self.assertFalse(os.path.exists(path))


class OptionTests(unittest.TestCase):
    """RamLimitBox (isolated settings) and --ram-limit."""

    def tearDown(self):
        """Back to the default limit."""
        native.set_ram_limit(native.DEFAULT_RAM_LIMIT_PERCENT)

    def test_command_line(self):
        """--ram-limit sets the program's limit; default 80."""
        parser = argparse.ArgumentParser()
        native.add_memory_arguments(parser)
        self.assertEqual(native.ram_limit_from_args(parser.parse_args([])), 80.0)
        self.assertEqual(native.ram_limit_from_args(parser.parse_args(["--ram-limit", "0"])), 0.0)
        self.assertEqual(native.ram_limit(), 0.0)

    def test_box(self):
        """Default 80 %, 0 shows "never", a change sets the limit and is saved
        (in a test settings folder, not the user's)."""
        global APP
        try:
            from qtpy.QtCore import QSettings
            from qtpy.QtWidgets import QApplication
            import _suite_gui
        except Exception as exc:                      # pragma: no cover - no Qt
            self.skipTest("Qt not available: %s" % exc)
        APP = QApplication.instance() or QApplication([])
        original = _suite_gui.suite_settings
        with tempfile.TemporaryDirectory() as folder:
            # an ini file of its own: never the user's settings (the registry on Windows)
            ini = os.path.join(folder, "suite.ini")
            _suite_gui.suite_settings = lambda: QSettings(ini, QSettings.Format.IniFormat)
            try:
                self._check_box(_suite_gui)
            finally:
                _suite_gui.suite_settings = original

    def _check_box(self, _suite_gui):
        """The RamLimitBox checks, with the settings in a test ini file."""
        if True:
            box = _suite_gui.RamLimitBox()
            self.assertEqual(box.limit_percent(), 80.0)
            self.assertEqual(native.ram_limit(), 80.0)
            heard = []
            box.limit_changed.connect(heard.append)
            box.spin.setValue(0)
            self.assertEqual(box.spin.text(), "never")
            self.assertEqual(native.ram_limit(), 0.0)
            box.spin.setValue(65)
            self.assertEqual(heard, [0.0, 65.0])
            self.assertEqual(_suite_gui.saved_ram_limit(), 65.0)
            self.assertIn("GB of RAM", box.ram_label.text())
            for widget in (box.spin, box.ram_label):
                self.assertTrue(widget.toolTip())
            again = _suite_gui.RamLimitBox()
            self.assertEqual(again.limit_percent(), 65.0)


if __name__ == "__main__":
    unittest.main()
