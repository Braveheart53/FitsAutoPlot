# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_native_export.py

AutoPlot native writer regression tests.

Run: QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v

Created on 2026-10-11

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.3.1: Widgets deleted in the GUI thread; 3.12 crash found and fixed.
Date: 2026-11-09
#             * dispose(): the main windows and the Plots panel are closed
#               and deleted in the GUI thread (deleteLater + DeferredDelete).
#             * The random segmentation faults / hangs under Python 3.12
#               (PySide6 6.12; also seen once as a hang on 3.8) came from the
#               program, not the tests: the memory monitor thread could drop
#               the last reference to a plugin dialog and destroy it off the
#               GUI thread (fixed in _autoplot_common 0.4.0, plugins 0.3.0).
#               The committed 2026-11-09 suite crashed 3 runs in 4 on 3.12.
#             * Docstrings for every test; header template; Spyder cells.
# %%%%% Function Descriptions
        NoVeusz (class): Import hook that makes ``import veusz`` fail, proving the native path needs no Veusz installation.
        dispose: Close ``widget`` and delete its C++ widget tree now (0.3.1).
        NativeTests (class): Native project model, writer, readers, GUIs and plugins without Veusz.
# %%%%% Variable Descriptions
    Define all utilized variables
        APP: the QApplication of the tests
# %%%%% More Info
    Run:  QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.3.0: small gaps break the line, large gaps (dt > n x small gap,
Date: 2026-11-09
#             default n = 100) break the x axis; MJD drawn as local date-time.
# %%%% 0.2.0: pyqtgraph Plots tab (pages, traces, MJD axes, images, paging);
Date: 2026-11-05
#             files written through vszh5_writer 0.0.24 (no *.part left).
# %%%% 0.1.0: Native file, engine, GUI and plugin workflow coverage.
Date: 2026-10-11
=============================================================================
"""
# %% Imports
import ast
import importlib.abc
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# %% Tests
class NoVeusz(importlib.abc.MetaPathFinder):
    """Import hook that makes ``import veusz`` fail, proving the native path
    needs no Veusz installation.
    """
    def find_spec(self, fullname, path=None, target=None):
        """Raise ImportError for veusz and its submodules; defer everything else.
        """
        if fullname == "veusz" or fullname.startswith("veusz."):
            raise ImportError("Veusz deliberately unavailable during native tests")
        return None


sys.meta_path.insert(0, NoVeusz())
import h5py
import numpy as np
from astropy.io import fits
from _autoplot_native import NativeDocument, TimeLayout
from _autoplot_common import (
    QApplication, MemoryAwareCache, MemoryMonitorConfig, parse_gap_string,
    save_vszh5_minimized,
)
import FITS_AutoPlot as fitstool
import Franks_AutoPlot as frankstool
import FITS_AutoPlot_plugin as fitsplugin
import Franks_AutoPlot_plugin as franksplugin

APP = QApplication.instance() or QApplication([])


def dispose(widget):
    """Close ``widget`` and delete its C++ widget tree now (0.3.1).

    deleteLater() plus an explicit DeferredDelete pass deletes the tree in
    Qt's own order, in the GUI thread.  (The random segmentation faults
    under Python 3.12 / PySide6 6.12 came from the memory monitor thread
    releasing the last reference to a plugin dialog, which then died outside
    the GUI thread; fixed in _autoplot_common 0.4.0 and the plugins 0.3.0.)
    """
    from qtpy.QtCore import QCoreApplication, QEvent
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    APP.processEvents()


class NativeTests(unittest.TestCase):
    """Native project model, writer, readers, GUIs and plugins without Veusz.
    """
    def setUp(self):
        """Fresh temporary folder for each test."""
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        """Delete the temporary folder."""
        self.tmp.cleanup()

    def basic_document(self):
        """Small project: numeric x, y with symmetric errors and a NaN, a text
        dataset, one tag and one page with a graph and an xy trace.
        """
        doc = NativeDocument()
        doc.SetData("x", [1, 2, 3])
        doc.SetData("y", [4, np.nan, 6], symerr=[.1, .2, .3])
        doc.SetDataText("label", ["alpha", "μs", ""])
        doc.SetData("unused", np.arange(5000.0))
        doc.TagDatasets("test", doc.GetDatasets())
        graph = doc.Root.Add("page", name="page").Add("graph", name="graph")
        xy = graph.Add("xy", name="trace")
        xy.xData.val = "x"
        xy.yData.val = "y"
        xy.labels.val = "label"
        return doc

    def test_no_veusz_imported(self):
        """No veusz module was imported by the native code."""
        self.assertFalse(any(k == "veusz" or k.startswith("veusz.") for k in sys.modules))

    def test_roundtrip_types_nan_errors_tags_and_minimization(self):
        """Saved file keeps NaN, errors, UTF-8 text and tags; the minimized file
        holds only the used datasets and is smaller.
        """
        doc = self.basic_document()
        full = doc.save(self.folder / "full.vszh5")
        small = save_vszh5_minimized(doc, str(self.folder / "small.vszh5"))
        self.assertEqual(len(doc.datasets), 4)
        with h5py.File(full) as h:
            self.assertEqual(h["Veusz"].attrs["vsz_format"], 1)
            self.assertTrue(np.isnan(h["Veusz/Data/y/data"][1]))
            np.testing.assert_array_equal(h["Veusz/Data/y/serr"], [.1, .2, .3])
            self.assertEqual(h["Veusz/Data/label/data"][1].decode("utf-8"), "μs")
            self.assertEqual(h["Veusz/Document/document"].shape, (1,))
        with h5py.File(small) as h:
            self.assertEqual(set(h["Veusz/Data"]), {"x", "y", "label"})
            self.assertEqual(set(h["Veusz/Document/Tags/test"].asstr()[:]),
                             {"x", "y", "label"})
        self.assertLess(os.path.getsize(small), os.path.getsize(full))

    def test_two_dimensional_and_date(self):
        """2-D datasets (with NaN) and date datasets get the right Veusz types and
        the date conversion flag.
        """
        doc = NativeDocument()
        doc.SetData2D("img", [[1, 2], [3, np.nan]], xrange=[0, 2], yrange=[5, 7])
        doc.SetDataDateTime("date", [0, 86400, np.nan])
        path = doc.save(self.folder / "types.vszh5")
        with h5py.File(path) as h:
            self.assertEqual(h["Veusz/Data/img"].attrs["vsz_datatype"], "2d")
            self.assertTrue(np.isnan(h["Veusz/Data/img/data"][1, 1]))
            self.assertEqual(h["Veusz/Data/date"].attrs["vsz_datatype"], "date")
            self.assertEqual(h["Veusz/Data/date/data"].attrs["vsz_convert_datetime"], 1)

    def test_empty_and_escaped_unicode_names(self):
        """Empty datasets save; '/' and '`' in names are escaped as Veusz does.
        """
        doc = NativeDocument()
        doc.SetDataText("µ/a`b", [])
        doc.SetData("empty", [])
        path = doc.save(self.folder / "empty.vszh5")
        with h5py.File(path) as h:
            self.assertIn("µ`SLa`BTb", h["Veusz/Data"])
            self.assertEqual(h["Veusz/Data/µ`SLa`BTb/data"].shape, (0,))

    def test_missing_reference_does_not_overwrite(self):
        """A trace naming a missing dataset raises ValueError and leaves the
        existing file untouched.
        """
        doc = self.basic_document()
        path = self.folder / "keep.vszh5"
        path.write_bytes(b"prior file")
        doc.Root.children[0].children[0].children[-1].xData.val = "missing"
        with self.assertRaises(ValueError):
            doc.save(path)
        self.assertEqual(path.read_bytes(), b"prior file")

    def test_atomic_cleanup_on_replace_failure(self):
        """A failed final rename keeps the old file and leaves no temporary
        (.autoplot-* or *.part) files behind.
        """
        doc = self.basic_document()
        path = self.folder / "keep.vszh5"
        path.write_bytes(b"prior file")
        with patch("_autoplot_native.os.replace", side_effect=OSError("test")):
            with self.assertRaises(OSError):
                doc.save(path)
        self.assertEqual(path.read_bytes(), b"prior file")
        self.assertEqual(list(self.folder.glob(".autoplot-*")), [])
        self.assertEqual(list(self.folder.glob("*.part")), [])   # vszh5_writer temp files

    def test_script_literals_cannot_inject_commands(self):
        """Quotes and new lines in a label cannot add commands to the saved
        document script.
        """
        doc = self.basic_document()
        graph = doc.Root.children[0].children[0]
        graph.x.label.val = "');\nImportFileCSV('not-real'); #"
        tree = ast.parse(doc.SaveToString())
        commands = {n.func.id for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertEqual(commands, {"SetCompatLevel", "Add", "To", "Set"})

    def test_plain_text_caption_escaping_not_dataset_names(self):
        """Labels are escaped for Veusz's text renderer; dataset references are
        not.
        """
        doc = self.basic_document()
        graph = doc.Root.children[0].children[0]
        graph.y.label.val = "A_B^{C}"
        statements = ast.parse(doc.SaveToString()).body
        values = [(ast.literal_eval(n.value.args[0]),
                   ast.literal_eval(n.value.args[1])) for n in statements
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and getattr(n.value.func, "id", None) == "Set"]
        self.assertIn(("label", r"A\_B\^\{C\}"), values)
        self.assertIn(("yData", "y"), values)

    def test_axis_replacement_and_collision_handling(self):
        """Duplicate names get a suffix; an axis can be replaced by a broken axis
        in date-time mode.
        """
        doc = NativeDocument()
        page = doc.Root.Add("page", name="page")
        self.assertEqual(doc.Root.Add("page", name="page").name, "page_1")
        graph = page.Add("graph")
        graph.Remove("x")
        graph.Add("axis-broken", name="x", breakPoints=[10.0, 20.0])
        graph.x.mode.val = "datetime"
        self.assertEqual([w.kind for w in graph.children], ["axis", "axis-broken"])
        self.assertIn("Set('mode', 'datetime')", doc.SaveToString())

    def test_gap_parser(self):
        """Gap text: months/days/hours units add up; a bare number is hours."""
        self.assertAlmostEqual(parse_gap_string("5M 3d 2H"), 5 * 30.4375 + 3 + 2 / 24)
        self.assertEqual(parse_gap_string("120"), 5.0)

    def write_fits(self, tagged=True):
        """Write a small OnePpsDeltas-style FITS file (a 7-day hole, a NaN, an
        image HDU), with CHANNELA/B tag columns when ``tagged``.
        """
        x = np.array([60805, 60805.1, 60805.2, 60812, 60812.1, 60812.2])
        cols = [fits.Column(name="DMJD", format="D", array=x),
                fits.Column(name="DELTAT", format="D", unit="ns",
                            array=[1, 2, np.nan, 3, 4, 5])]
        if tagged:
            cols += [fits.Column(name="CHANNELA", format="8A",
                                 array=["A"] * 6),
                     fits.Column(name="CHANNELB", format="8A",
                                 array=["B"] * 6)]
        table = fits.BinTableHDU.from_columns(cols, name="OnePpsDeltas")
        path = self.folder / ("tagged.fits" if tagged else "untagged.fits")
        fits.HDUList([fits.PrimaryHDU(), table,
                      fits.ImageHDU(np.arange(12.0).reshape(3, 4), name="IMAGE")]).writeto(path)
        return path

    def write_franks(self):
        """Write a small time_gbt.dat Franks file with a 7-day hole and a NaN.
        """
        path = self.folder / "time_gbt.dat"
        path.write_text("MJD A B C D\n================\n"
                        "60805 1 2 3 4\n60805.1 2 nan 4 5\n"
                        "60812 3 4 5 6\n60812.1 4 5 6 7\n", encoding="utf-8")
        return path

    def test_fits_tagged_untagged_images_and_axes(self):
        """FITS per-file and overlay pages: local date-time axes; the 7-day hole
        breaks the line (n = 0) and also the x axis (n = 2); keys hidden; legacy
        _dt pages stay numeric.
        """
        for tagged in (True, False):
            with self.subTest(tagged=tagged):
                cache = MemoryAwareCache(MemoryMonitorConfig())
                path = self.write_fits(tagged)
                data = fitstool.FITSProcessor("astropy", cache).read(str(path))
                doc = NativeDocument()
                logs = []
                fitstool.push_to_veusz(doc, str(path), data, "both", log_cb=logs.append,
                                      datetime_duplicate=True, hide_keys=True,
                                      gap_absolute=1.0)
                fitstool.build_unit_overlay_pages(
                    doc, [dict(data, base=data["base_name"])],
                    datetime_duplicate=True, hide_keys=True,
                    gap_absolute=1.0, log_cb=logs.append)
                # 0.3.0: the 7-day hole is a small gap (line break) with
                # n = 0 and also a large gap (x-axis break) with n = 2
                doc.time_layout = TimeLayout(k=10.0, n=0.0)
                script = doc.SaveToString()
                self.assertNotIn("axis-broken", script)
                self.assertIn("Set('mode', 'datetime')", script)
                self.assertIn("_gaps", script)
                doc.time_layout = TimeLayout(k=10.0, n=2.0)
                self.assertIn("axis-broken", doc.SaveToString())
                self.assertFalse([line for line in logs if "failed" in line.lower()], logs)
                for widget in doc.Root.WalkWidgets():
                    if widget.kind == "key":
                        self.assertIs(widget.hide.val, True)
                doc.save(self.folder / ("fits_%s.vszh5" % tagged), minimized=True)
                for page in doc.Root.children:
                    if page.name.endswith("_dt"):
                        for w in page.WalkWidgets():
                            if w.kind.startswith("axis"):
                                self.assertNotEqual(w.mode.val, "datetime")
                cache.cleanup()

    def test_franks_per_file_and_overlay(self):
        """Franks per-file and overlay pages build without errors, with date-time
        axes.
        """
        cache = MemoryAwareCache(MemoryMonitorConfig())
        data = frankstool.parse_franks_file(str(self.write_franks()), cache)
        doc = NativeDocument()
        logs = []
        frankstool.push_franks_to_veusz(doc, data, log_cb=logs.append,
                                      datetime_duplicate=True, hide_keys=True)
        frankstool.build_unit_overlay_pages_franks(
            doc, [dict(data, base=data["base_name"])],
            datetime_duplicate=True, hide_keys=True, log_cb=logs.append)
        self.assertFalse([line for line in logs if "failed" in line.lower()], logs)
        self.assertIn("Set('mode', 'datetime')", doc.SaveToString())
        doc.save(self.folder / "franks.vszh5", minimized=True)
        cache.cleanup()

    def test_gui_construction_without_veusz(self):
        """Both main windows open without Veusz (Open-in-Veusz disabled)."""
        for cls in (fitstool.FITSAutoPlotWindow, frankstool.FranksAutoPlotWindow):
            window = cls()
            self.assertFalse(window.show_embed_cb.isEnabled())
            dispose(window)

    def test_both_plugin_export_workflows_leave_host_untouched(self):
        """Both plugins write minimized and full projects without touching the
        Veusz host document.
        """
        class HostMustNotBeUsed:
            """Stand-in host document that fails the test on any use."""
            def __getattr__(self, name):
                """Any attribute access means the plugin used the host: fail.
                """
                raise AssertionError("Host document accessed: " + name)
        cases = [(fitsplugin, fitsplugin.FITSAutoPlotPlugin, self.write_fits()),
                 (franksplugin, franksplugin.FranksAutoPlotPlugin, self.write_franks())]
        for module, cls, input_path in cases:
            with self.subTest(plugin=cls.__name__):
                output = self.folder / (cls.__name__ + ".vszh5")
                fields = dict(preselected_files=str(input_path), output_file=str(output),
                              minimized=True, also_full=True, datetime_duplicate=True,
                              hide_keys=True, gap_string="5d")
                plugin = object.__new__(cls)
                with patch.object(module._PluginBatchDialog, "exec_", return_value=1), \
                     patch.object(module.QMessageBox, "information"):
                    plugin._apply(HostMustNotBeUsed(), fields)
                self.assertTrue(output.exists())
                self.assertTrue(output.with_name(output.stem + "_full.vszh5").exists())
                with h5py.File(output) as h:
                    self.assertGreater(len(h["Veusz/Data"]), 0)
                    self.assertIn("datetime", h["Veusz/Document/document"][0].decode())

    def test_pyqtgraph_plots_tab(self):
        """The Plots tab draws a FITS project: date-time axes, images, line breaks
        at gaps and page stepping.
        """
        import _autoplot_pyqtgraph as pgview
        self.assertIsNone(pgview.self_test())
        cache = MemoryAwareCache(MemoryMonitorConfig())
        path = self.write_fits(True)
        data = fitstool.FITSProcessor("astropy", cache).read(str(path))
        doc = NativeDocument()
        fitstool.push_to_veusz(doc, str(path), data, "astropy",
                               datetime_duplicate=True, gap_absolute=1.0)
        pages = pgview.document_pages(doc)
        self.assertTrue(pages)
        kinds = {g["x_kind"] for p in pages for g in p["graphs"] if g["traces"]}
        # 0.3.0: MJD axes are laid out as local date-time axes before drawing
        self.assertEqual(kinds, {"veusz_date"})
        self.assertTrue(any(g["images"] for p in pages for g in p["graphs"]))
        # gaps: the 7-day jump in the fixture breaks the line
        x, y = pgview.break_gaps(np.array([0.0, 1, 2, 30, 31]), np.ones(5))
        self.assertEqual(int(np.isnan(x).sum()), 1)
        panel = pgview.ProjectPlotPanel()
        self.assertEqual(panel.set_document(doc), len(pages))
        first = len(panel.plots)
        self.assertGreater(first, 0)
        for index in range(len(pages)):
            panel.page_box.setCurrentIndex(index)
            APP.processEvents()
        plot = next(p for p in panel.plots if isinstance(p, pgview.TracePlot)) \
            if any(isinstance(p, pgview.TracePlot) for p in panel.plots) else None
        if plot is not None:
            label = next(iter(plot.data))
            xs = plot.data[label][0]
            values = plot.set_cursor(float(np.nanmin(xs)))
            self.assertIn(label, values)
        panel.apply_theme(True)
        panel.clear()
        self.assertEqual(panel.plots, [])
        dispose(panel)
        for cls in (fitstool.FITSAutoPlotWindow, frankstool.FranksAutoPlotWindow):
            window = cls()
            self.assertIsNotNone(window.plot_panel)
            self.assertEqual(window.main_tabs.count(), 2)
            dispose(window)
        cache.cleanup()

    def test_python38_grammar_and_history_order(self):
        """Every program parses as Python 3.8 and its revision history is
        newest-first.
        """
        for path in Path(__file__).resolve().parents[1].glob("*.py"):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=path.name, feature_version=(3, 8))
            import re
            versions = [tuple(map(int, v.split("."))) for v in
                        re.findall(r"^# %%%% (\d+\.\d+\.\d+):", text, re.M)]
            self.assertEqual(versions, sorted(versions, reverse=True), path.name)


# %% Run
if __name__ == "__main__":
    unittest.main()
