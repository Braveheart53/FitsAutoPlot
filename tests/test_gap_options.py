# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_gap_options.py

Gap options of every plot generator (2026-11-16): a small gap breaks the
line / trace, a large gap breaks the x axis, each can be turned off, and the
rule also works on x axes that are not time (_autoplot_native 0.5.3,
_autoplot_pyqtgraph 0.3.3, _suite_gui 0.1.2, _autoplot_common 0.4.3, ATR
0.2.5, CSV/TSV 6.1.4, R&S 1.1.5, Touchstone 0.7.4, FITS / Franks 0.4.3).

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
#              Time and numeric axes with each option on and off, stale
#              copies removed, notes, the Plots tab, the GUI boxes and the
#              command-line flags of ATR, CSV/TSV and R&S.
# %%%%% Function Descriptions
        graph_with: Add a page with one graph whose x axis is labelled ``label`` and one xy trace per (x, y) pair; return (graph, x axis).
        TimeAxisOptions (class): Line and axis breaks on a time axis, each optional.
        NumericAxisOptions (class): The same options on axes that are not time.
        PlotsTabTests (class): The Plots tab follows the options; log sweeps are not cut.
        GuiTests (class): GapOptionsBox and the FITS / Franks gap controls give the TimeLayout.
        CommandLineTests (class): --gap-* flags of ATR, CSV/TSV and R&S.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        EXAMPLES: example data folder
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
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "Example_Data_Files"
sys.path.insert(0, str(ROOT))

import vszh5_writer as w                                         # noqa: E402
from _autoplot_native import NativeDocument, TimeLayout          # noqa: E402


# %% Helpers
def graph_with(doc, pairs, label="time"):
    """Add a page with one graph whose x axis is labelled ``label`` and one
    xy trace per (x, y) pair; return (graph, x axis)."""
    page = doc.Root.Add("page", name="p%d" % len(doc.Root.children))
    graph = page.Add("graph", name="g", autoadd=False)
    axis = graph.Add("axis", name="x")
    axis.settings["label"] = label
    graph.Add("axis", name="y").settings["direction"] = "vertical"
    for index, (x_name, y_name) in enumerate(pairs):
        graph.Add("xy", name="t%d" % index, xData=x_name, yData=y_name)
    return graph, axis


# %% Time axes
class TimeAxisOptions(unittest.TestCase):
    """Line and axis breaks on a time axis, each optional."""

    def setUp(self):
        """UWBR-like MJD data (1 s steps, then 60 s steps, a 2.3 h hole); zone -05:00."""
        self._saved = os.environ.get(w.DISPLAY_TZ_ENV)
        os.environ[w.DISPLAY_TZ_ENV] = "-05:00"
        mjd = np.r_[61349.5 + np.arange(300) / 86400.0,
                    61349.6 + np.arange(50) * 60.0 / 86400.0]
        self.doc = NativeDocument("t")
        self.doc.SetData("ds__DMJD__sorted", mjd)
        self.doc.SetData("ds__T1__sorted", np.arange(mjd.size, dtype=float))
        self.graph, self.axis = graph_with(self.doc, [("ds__DMJD__sorted", "ds__T1__sorted")])

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self._saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self._saved

    def xy(self):
        """The trace widget of the test graph."""
        return [c for c in self.graph.children if c.kind == "xy"][0]

    def test_defaults_unchanged(self):
        """Defaults: line breaks at the 60 s steps and an axis break at the hole."""
        self.doc.finalize()
        self.assertEqual(self.xy().settings["yData"], "ds__T1__sorted_gaps")
        self.assertEqual(self.axis.kind, "axis-broken")
        self.assertIn("Gaps: line break (small gap) where dt > 10 x median(dt)",
                      self.graph.settings["notes"])

    def test_lines_off_axis_on(self):
        """break_lines=False: no gap rows; the axis still breaks."""
        self.doc.time_layout = TimeLayout(break_lines=False)
        self.doc.finalize()
        self.assertEqual(self.xy().settings["yData"], "ds__T1__sorted")
        local = self.doc.datasets[self.xy().settings["xData"]][1]["data"]
        self.assertFalse(np.isnan(local).any())
        self.assertEqual(self.axis.kind, "axis-broken")
        self.assertIn("lines drawn across gaps", self.graph.settings["notes"])

    def test_lines_on_axis_off(self):
        """break_axis=False: one continuous axis; the lines still break."""
        self.doc.time_layout = TimeLayout(break_axis=False)
        self.doc.finalize()
        self.assertEqual(self.axis.kind, "axis")
        self.assertNotIn("breakPoints", self.axis.settings)
        self.assertEqual(self.xy().settings["yData"], "ds__T1__sorted_gaps")

    def test_switching_options_removes_stale_copies(self):
        """A second save with lines off drops the gap copies of the first."""
        self.doc.finalize()
        self.assertIn("ds__T1__sorted_gaps", self.doc.datasets)
        self.doc.time_layout = TimeLayout(break_lines=False, break_axis=False)
        self.doc.finalize()
        self.assertNotIn("ds__T1__sorted_gaps", self.doc.datasets)
        self.assertEqual(self.axis.kind, "axis")
        notes = self.graph.settings["notes"]
        self.assertEqual(notes.count("Gaps: "), 1)          # the old rule line is replaced
        self.assertIn("lines drawn across gaps; no x-axis breaks", notes)


# %% Numeric axes
class NumericAxisOptions(unittest.TestCase):
    """The same options on axes that are not time."""

    def doc_with(self, x, label="Frequency (Hz)", layout=None):
        """A document with one trace of ``x`` (and y = x) on an axis labelled ``label``."""
        doc = NativeDocument("t")
        doc.SetData("f", np.asarray(x, dtype=float))
        doc.SetData("v", np.arange(len(x), dtype=float))
        graph, axis = graph_with(doc, [("f", "v")], label=label)
        if layout is not None:
            doc.time_layout = layout
        doc.finalize()
        xy = [c for c in graph.children if c.kind == "xy"][0]
        return doc, graph, axis, xy

    def test_segmented_sweep(self):
        """A hole in a frequency sweep breaks the line; n = 10 also the axis."""
        x = np.r_[np.linspace(1e9, 2e9, 101), np.linspace(1e10, 1.1e10, 101)]
        doc, graph, axis, xy = self.doc_with(x)
        self.assertEqual(xy.settings["xData"], "f_gaps")
        self.assertEqual(int(np.isnan(doc.datasets["f_gaps"][1]["data"]).sum()), 1)
        self.assertEqual(axis.kind, "axis")                 # 8 GHz < 100 x 100 MHz
        doc, graph, axis, xy = self.doc_with(x, layout=TimeLayout(n=10.0))
        self.assertEqual(axis.kind, "axis-broken")
        self.assertIn("(and on a log scale)", graph.settings["notes"])

    def test_log_sweep_not_cut(self):
        """A log-spaced sweep has no gaps (geometric rule)."""
        doc, graph, axis, xy = self.doc_with(np.geomspace(1e7, 1e10, 201))
        self.assertEqual(xy.settings["xData"], "f")
        self.assertNotIn("f_gaps", doc.datasets)

    def test_numeric_off_and_unsorted(self):
        """numeric=False leaves the axis alone; an unsorted x is never cut."""
        x = np.r_[np.arange(10.0), 100.0 + np.arange(10.0)]
        doc, graph, axis, xy = self.doc_with(x, layout=TimeLayout(numeric=False))
        self.assertEqual(xy.settings["xData"], "f")
        self.assertNotIn("Gaps:", str(graph.settings.get("notes", "")))
        doc, graph, axis, xy = self.doc_with(x[::-1].copy())
        self.assertEqual(xy.settings["xData"], "f")

    def test_numeric_turned_off_on_second_save(self):
        """Numeric gaps on, then off: the traces draw their data again."""
        x = np.r_[np.linspace(1e9, 2e9, 101), np.linspace(1e10, 1.1e10, 101)]
        doc = NativeDocument("t")
        doc.SetData("f", x)
        doc.SetData("v", np.arange(x.size, dtype=float))
        graph, axis = graph_with(doc, [("f", "v")], label="Frequency (Hz)")
        doc.time_layout = TimeLayout(n=10.0)
        doc.finalize()
        xy = [c for c in graph.children if c.kind == "xy"][0]
        self.assertEqual((xy.settings["xData"], axis.kind), ("f_gaps", "axis-broken"))
        doc.time_layout = TimeLayout(n=10.0, numeric=False)
        doc.finalize()
        self.assertEqual((xy.settings["xData"], xy.settings["yData"]), ("f", "v"))
        self.assertEqual(axis.kind, "axis")
        self.assertNotIn("f_gaps", doc.datasets)
        self.assertNotIn("Gaps:", str(graph.settings.get("notes", "")))

    def test_disjoint_traces_break_axis_only(self):
        """Two traces far apart: no line breaks, the axis is cut between them."""
        doc = NativeDocument("t")
        doc.SetData("a", np.linspace(1e7, 1.01e7, 101))
        doc.SetData("b", np.linspace(1.4e10, 1.4001e10, 101))
        doc.SetData("ya", np.zeros(101))
        doc.SetData("yb", np.zeros(101))
        graph, axis = graph_with(doc, [("a", "ya"), ("b", "yb")], label="Frequency (Hz)")
        doc.finalize()
        self.assertEqual(axis.kind, "axis-broken")
        self.assertEqual(len(axis.settings["breakPoints"]), 2)
        self.assertNotIn("a_gaps", doc.datasets)


# %% Plots tab
class PlotsTabTests(unittest.TestCase):
    """The Plots tab follows the options; log sweeps are not cut."""

    @classmethod
    def setUpClass(cls):
        """Import the Plots tab module (skip without pyqtgraph)."""
        try:
            import _autoplot_pyqtgraph as plots
        except Exception as exc:                  # optional stack
            raise unittest.SkipTest("pyqtgraph not importable: %s" % exc)
        cls.plots = plots

    def test_trace_arrays_and_regions(self):
        """Line breaks follow break_lines; axis regions follow break_axis."""
        doc = NativeDocument("t")
        x = np.r_[np.arange(100.0), 1e5 + np.arange(100.0)]
        doc.SetData("x", x)
        doc.SetData("y", np.ones(x.size))
        trace = {"x_name": "x", "y_name": "y"}
        bx, _by, _t = self.plots.trace_arrays(doc, trace, "numeric", False)
        self.assertEqual(int(np.isnan(bx).sum()), 1)
        self.assertEqual(len(self.plots.axis_break_regions(doc, [x], False)), 1)
        doc.time_layout = TimeLayout(break_lines=False, break_axis=False)
        bx, _by, _t = self.plots.trace_arrays(doc, trace, "numeric", False)
        self.assertFalse(np.isnan(bx).any())
        self.assertEqual(self.plots.axis_break_regions(doc, [x], False), [])

    def test_log_sweep(self):
        """A log-spaced sweep is drawn in one piece (0.3.2 cut it 33 times)."""
        doc = NativeDocument("t")
        f = np.geomspace(1e7, 1e10, 201)
        doc.SetData("f", f)
        doc.SetData("s", np.zeros(f.size))
        bx, _by, _t = self.plots.trace_arrays(doc, {"x_name": "f", "y_name": "s"},
                                              "numeric", False)
        self.assertFalse(np.isnan(bx).any())


# %% GUI
class GuiTests(unittest.TestCase):
    """GapOptionsBox and the FITS / Franks gap controls give the TimeLayout."""

    @classmethod
    def setUpClass(cls):
        """One offscreen QApplication."""
        try:
            from qtpy.QtWidgets import QApplication
        except Exception as exc:
            raise unittest.SkipTest("Qt not importable: %s" % exc)
        cls.app = QApplication.instance() or QApplication([])

    def test_gap_box(self):
        """Check boxes and values reach the TimeLayout; spin boxes follow the boxes."""
        import _suite_gui
        box = _suite_gui.GapOptionsBox()
        layout = box.time_layout()
        self.assertEqual((layout.break_lines, layout.break_axis, layout.numeric), (True, True, True))
        box.lines_cb.setChecked(False)
        # K still scales the large gap (n x small gap) while the axis breaks are on
        self.assertTrue(box.k_spin.isEnabled())
        self.assertTrue(box.n_spin.isEnabled())
        box.axis_cb.setChecked(False)
        box.numeric_cb.setChecked(False)
        box.k_spin.setValue(25.0)
        box.small_text.setText("90s")
        layout = box.time_layout()
        self.assertEqual((layout.break_lines, layout.break_axis, layout.numeric),
                         (False, False, False))
        self.assertEqual((layout.k, layout.manual_s), (25.0, 90.0))
        self.assertFalse(box.k_spin.isEnabled())
        self.assertFalse(box.n_spin.isEnabled())
        for child in (box.lines_cb, box.axis_cb, box.numeric_cb, box.k_spin, box.n_spin,
                      box.small_text, box.large_text):
            self.assertTrue(child.toolTip())

    def test_fits_controls(self):
        """FITS AutoPlot's gap controls carry the three new check boxes."""
        import FITS_AutoPlot as fits
        window = fits.FITSAutoPlotWindow()
        try:
            window.gap_lines_cb.setChecked(False)
            self.assertTrue(window.gap_k_spin.isEnabled())
            window.gap_numeric_cb.setChecked(False)
            layout = window.time_layout()
            self.assertEqual((layout.break_lines, layout.break_axis, layout.numeric),
                             (False, True, False))
        finally:
            window.close()


# %% Command line
class CommandLineTests(unittest.TestCase):
    """--gap-* flags of ATR, CSV/TSV and R&S."""

    def test_layout_from_args(self):
        """The flags become the TimeLayout."""
        import argparse
        from _autoplot_native import add_gap_arguments, layout_from_args
        parser = argparse.ArgumentParser()
        add_gap_arguments(parser)
        layout = layout_from_args(parser.parse_args(
            ["--gap-k", "5", "--gap-n", "0", "--gap-small", "90s", "--gap-large", "2h",
             "--no-line-breaks", "--no-numeric-gaps"]))
        self.assertEqual((layout.k, layout.n, layout.manual_s, layout.axis_manual_s),
                         (5.0, 0.0, 90.0, 7200.0))
        self.assertEqual((layout.break_lines, layout.break_axis, layout.numeric),
                         (False, True, False))

    def test_csv_batch(self):
        """CSV/TSV batch: a 2 h hole in 1 s data, axis break on, lines off."""
        import h5py
        import CSV_TSV_AutoPlot as csvplot
        lines = ["Time,V"]
        stamps = list(range(0, 300)) + list(range(7500, 7800))
        for second in stamps:
            lines.append("2026-11-16T%02d:%02d:%02dZ,%d" % (second // 3600, second // 60 % 60,
                                                          second % 60, second))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hole.csv")
            with open(path, "w") as handle:
                handle.write("\n".join(lines) + "\n")
            out = os.path.join(folder, "hole.vszh5")
            code = csvplot.main([path, "--x", "Time", "--y", "V", "--output", out,
                                 "--no-line-breaks"])
            self.assertEqual(code, 0)
            with h5py.File(out, "r") as handle:
                script = handle["Veusz/Document/document"][0].decode()
                names = list(handle["Veusz/Data"])
        self.assertIn("axis-broken", script)
        self.assertFalse(any(name.endswith("_gaps") for name in names))
        self.assertIn("lines drawn across gaps", script)

    def test_atr_and_rns_batch(self):
        """ATR and R&S accept the flags (no axis breaks requested -> none)."""
        import h5py
        import ATR_AutoPlot as atr
        import RAndS_FSW_ASCII_Plotter as rns
        atr_files = sorted(glob.glob(str(EXAMPLES / "atr" / "*.ATR")))[:2]
        rns_files = sorted(glob.glob(str(EXAMPLES / "sft" / "R&S_DataFiles" / "All" / "*.sft")))
        if not atr_files or not rns_files:
            self.skipTest("example files missing")
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "atr.vszh5")
            self.assertEqual(atr.main(atr_files + ["--output", out, "--no-axis-breaks"]), 0)
            out2 = os.path.join(folder, "rns.vszh5")
            self.assertIn(rns.main(rns_files + ["--output", out2, "--no-axis-breaks"]), (0, 2))
            for name in (out, out2):
                with h5py.File(name, "r") as handle:
                    script = handle["Veusz/Document/document"][0].decode()
                self.assertNotIn("axis-broken", script)
                self.assertIn("no x-axis breaks", script)


if __name__ == "__main__":
    unittest.main()
