# -*- coding: utf-8 -*-
"""AutoPlot native writer regression tests.
# %%%% 0.2.0: pyqtgraph Plots tab (pages, traces, MJD axes, images, paging);
#             files written through vszh5_writer 0.0.24 (no *.part left).
# Date: 2026-11-05
# %%%% 0.1.0: Native file, engine, GUI and plugin workflow coverage.
# Date: 2026-10-11
Run: QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
"""
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


class NoVeusz(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "veusz" or fullname.startswith("veusz."):
            raise ImportError("Veusz deliberately unavailable during native tests")
        return None


sys.meta_path.insert(0, NoVeusz())
import h5py
import numpy as np
from astropy.io import fits
from _autoplot_native import NativeDocument
from _autoplot_common import (
    QApplication, MemoryAwareCache, MemoryMonitorConfig, parse_gap_string,
    save_vszh5_minimized,
)
import FITS_AutoPlot as fitstool
import Franks_AutoPlot as frankstool
import FITS_AutoPlot_plugin as fitsplugin
import Franks_AutoPlot_plugin as franksplugin

APP = QApplication.instance() or QApplication([])


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def basic_document(self):
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
        self.assertFalse(any(k == "veusz" or k.startswith("veusz.") for k in sys.modules))

    def test_roundtrip_types_nan_errors_tags_and_minimization(self):
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
        doc = NativeDocument()
        doc.SetDataText("µ/a`b", [])
        doc.SetData("empty", [])
        path = doc.save(self.folder / "empty.vszh5")
        with h5py.File(path) as h:
            self.assertIn("µ`SLa`BTb", h["Veusz/Data"])
            self.assertEqual(h["Veusz/Data/µ`SLa`BTb/data"].shape, (0,))

    def test_missing_reference_does_not_overwrite(self):
        doc = self.basic_document()
        path = self.folder / "keep.vszh5"
        path.write_bytes(b"prior file")
        doc.Root.children[0].children[0].children[-1].xData.val = "missing"
        with self.assertRaises(ValueError):
            doc.save(path)
        self.assertEqual(path.read_bytes(), b"prior file")

    def test_atomic_cleanup_on_replace_failure(self):
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
        doc = self.basic_document()
        graph = doc.Root.children[0].children[0]
        graph.x.label.val = "');\nImportFileCSV('not-real'); #"
        tree = ast.parse(doc.SaveToString())
        commands = {n.func.id for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertEqual(commands, {"SetCompatLevel", "Add", "To", "Set"})

    def test_plain_text_caption_escaping_not_dataset_names(self):
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
        self.assertAlmostEqual(parse_gap_string("5M 3d 2H"), 5 * 30.4375 + 3 + 2 / 24)
        self.assertEqual(parse_gap_string("120"), 5.0)

    def write_fits(self, tagged=True):
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
        path = self.folder / "time_gbt.dat"
        path.write_text("MJD A B C D\n================\n"
                        "60805 1 2 3 4\n60805.1 2 nan 4 5\n"
                        "60812 3 4 5 6\n60812.1 4 5 6 7\n", encoding="utf-8")
        return path

    def test_fits_tagged_untagged_images_and_axes(self):
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
                script = doc.SaveToString()
                self.assertIn("axis-broken", script)
                self.assertIn("Set('mode', 'datetime')", script)
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
        for cls in (fitstool.FITSAutoPlotWindow, frankstool.FranksAutoPlotWindow):
            window = cls()
            self.assertFalse(window.show_embed_cb.isEnabled())
            window.close()

    def test_both_plugin_export_workflows_leave_host_untouched(self):
        class HostMustNotBeUsed:
            def __getattr__(self, name):
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
        self.assertIn("mjd", kinds)
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
        for cls in (fitstool.FITSAutoPlotWindow, frankstool.FranksAutoPlotWindow):
            window = cls()
            self.assertIsNotNone(window.plot_panel)
            self.assertEqual(window.main_tabs.count(), 2)
            window.close()
        cache.cleanup()

    def test_python38_grammar_and_history_order(self):
        for path in Path(__file__).resolve().parents[1].glob("*.py"):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=path.name, feature_version=(3, 8))
            import re
            versions = [tuple(map(int, v.split("."))) for v in
                        re.findall(r"^# %%%% (\d+\.\d+\.\d+):", text, re.M)]
            self.assertEqual(versions, sorted(versions, reverse=True), path.name)


if __name__ == "__main__":
    unittest.main()
