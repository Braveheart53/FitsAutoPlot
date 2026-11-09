# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_time_gaps.py

Gap rule and time axes: writer 0.0.27, _autoplot_native 0.5.0, FITS 0.4.0.

Line break where a trace's step dt > K x median(dt); axis break only where no
trace has data for dt > n x K x median(dt) (n = 0: one axis); MJD time axes
drawn as local ISO 8601 date-times; padding rows (DMJD <= 0) left out.

Run:  python -m unittest discover -s tests

Created on 2026-11-09

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First tests (from the UWBR, tack and 1PPS files of 2026-11-09).
Date: 2026-11-09
# %%%%% Function Descriptions
        GapHelpers (class): vszh5_writer 0.0.27: thresholds are multiples of the median step.
        _graph: Add a page with a graph, an x axis labelled ``label`` and one xy trace per name in ``y_names``; return the graph.
        TimeLayoutPass (class): _autoplot_native 0.5.0: every MJD axis becomes a local date-time axis.
        FitsReader (class): FITS AutoPlot 0.4.0: padding rows, full headers, page order.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT = Path(__file__).resolve().parents[1]
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vszh5_writer as w                                  # noqa: E402
from _autoplot_native import NativeDocument, TimeLayout   # noqa: E402


# %% Tests
class GapHelpers(unittest.TestCase):
    """vszh5_writer 0.0.27: thresholds are multiples of the median step."""

    def test_threshold_is_k_times_median(self):
        """Small-gap threshold is K x median(dt) (or the fixed value); the cut is
        after each larger step; unsorted data is not cut; NaN steps are ignored.
        """
        t = np.array([0.0, 1, 2, 3, 63, 64, 65])
        self.assertEqual(w.median_step(t), 1.0)
        self.assertEqual(w.gap_threshold(t, k=10.0), 10.0)
        self.assertEqual(w.gap_threshold(t, k=10.0, manual=100.0), 100.0)
        self.assertEqual(w.gap_positions(t, 10.0).tolist(), [4])
        self.assertEqual(w.gap_positions(t, 100.0).tolist(), [])
        self.assertEqual(w.gap_positions(t[::-1], 10.0).tolist(), [])   # unsorted: no cut
        y = w.insert_gaps(np.arange(7.0), [4])
        self.assertEqual(int(np.isnan(y).sum()), 1)
        self.assertTrue(np.isnan(y[4]))
        self.assertEqual(w.median_step([0.0, np.nan, 2.0, 3.0]), 1.0)  # NaN rows ignored

    def test_axis_break_margin_is_relative(self):
        """Axis break pairs are shrunk by 2 % of the gap, not a fixed time."""
        # A 10.6 s gap at MJD 61349 in Veusz seconds (~5.6e8): the old
        # absolute shrink (1e-9 x |t| ~ 5.6 s per side) removed it.
        start = (61349.5 - 54832.0) * 86400.0
        t = np.r_[start + np.arange(10.0), start + 9.0 + 10.6 + np.arange(5.0)]
        pairs = w.axis_break_pairs(t, 5.0)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0][1] - pairs[0][0], 10.6 * (1 - 2 * w.GAP_AXIS_MARGIN), 5)
        self.assertEqual(w.axis_break_pairs(t, 0.0), [])


def _graph(doc, x_name, y_names, label="DMJD"):
    """Add a page with a graph, an x axis labelled ``label`` and one xy trace
    per name in ``y_names``; return the graph.
    """
    graph = doc.Root.Add("page", name="p%d" % len(doc.Root.children)).Add("graph", name="g", autoadd=False)
    axis = graph.Add("axis", name="x")
    axis.settings["label"] = label
    graph.Add("axis", name="y").settings["direction"] = "vertical"
    for y_name in y_names:
        graph.Add("xy", name=y_name, xData=x_name, yData=y_name)
    return graph, axis


class TimeLayoutPass(unittest.TestCase):
    """_autoplot_native 0.5.0: every MJD axis becomes a local date-time axis."""

    def setUp(self):
        """Fix the display zone at -05:00 for reproducible labels."""
        self._saved = os.environ.get(w.DISPLAY_TZ_ENV)
        os.environ[w.DISPLAY_TZ_ENV] = "-05:00"

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self._saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self._saved

    def _multi_rate(self):
        """UWBR-like data: 300 s at 1 s steps then 60 s steps; returns (doc,
        graph, x axis).
        """
        # 300 samples at 1 s, then 50 at 60 s (as the UWBR temperature logs)
        mjd = np.r_[61349.5 + np.arange(300) / 86400.0,
                    61349.6 + np.arange(50) * 60.0 / 86400.0]
        doc = NativeDocument("t")
        doc.SetData("ds__DMJD__sorted", mjd)
        doc.SetData("ds__T1__sorted", np.arange(mjd.size, dtype=float))
        graph, axis = _graph(doc, "ds__DMJD__sorted", ["ds__T1__sorted"])
        return doc, graph, axis

    def test_line_breaks_one_axis(self):
        """n = 0: lines break at the 60 s steps, no axis break, local date-time
        axis.
        """
        doc, graph, axis = self._multi_rate()
        doc.time_layout = TimeLayout(k=10.0, n=0.0)            # no axis breaks
        doc.finalize()
        xy = graph.children[-1]
        self.assertEqual(xy.settings["xData"], "ds__DMJD__sorted_local")
        self.assertEqual(xy.settings["yData"], "ds__T1__sorted_gaps")
        self.assertEqual(doc.datasets["ds__DMJD__sorted_local"][0], "date")
        x = doc.datasets["ds__DMJD__sorted_local"][1]["data"]
        self.assertEqual(int(np.isnan(x).sum()), 50)       # 49 slow steps + the 1 s -> 60 s jump
        self.assertEqual(axis.kind, "axis")
        self.assertEqual(axis.settings["mode"], "datetime")
        self.assertEqual(axis.settings["TickLabels/format"], "%VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00")
        self.assertEqual(axis.settings["label"], "Local time (ISO 8601, UTC-05:00)")
        # stored data unchanged
        self.assertEqual(doc.datasets["ds__T1__sorted"][1]["data"].size, 350)
        self.assertIn("no x-axis breaks", graph.settings["notes"])

    def test_default_small_and_large_gaps(self):
        """Defaults (K = 10, n = 100): lines break at the 60 s steps; the x axis
        stays whole.
        """
        # default K = 10, n = 100: the 60 s steps (> 10 s) break the line;
        # the 2.3 h hole (> 1000 s) also breaks the x axis
        doc, graph, axis = self._multi_rate()
        doc.finalize()
        x = doc.datasets[graph.children[-1].settings["xData"]][1]["data"]
        self.assertEqual(int(np.isnan(x).sum()), 50)
        self.assertEqual(axis.kind, "axis-broken")
        self.assertEqual(len(axis.settings["breakPoints"]), 2)
        # a fixed large gap overrides n; never below the small gap
        doc.time_layout = TimeLayout(k=10.0, n=100.0, axis_manual_s=3.0 * 3600.0)
        doc.finalize()
        self.assertEqual(axis.kind, "axis")
        doc.time_layout = TimeLayout(k=10.0, axis_manual_s=1.0)   # raised to the 10 s small gap
        doc.finalize()
        self.assertEqual(len(axis.settings["breakPoints"]), 2 * 50)
        self.assertIn("x-axis break (large gap) where dt > 1 s", doc.time_layout.describe())

    def test_gap_text_units(self):
        """Gap text units s, min, h, d and m (months); a bare number is hours.
        """
        from _autoplot_common import parse_gap_string
        for text, seconds in (("90s", 90.0), ("15min", 900.0), ("2h", 7200.0),
                              ("3d", 259200.0), ("1m 3d 2h", 2896200.0), ("2", 7200.0)):
            self.assertAlmostEqual(parse_gap_string(text) * 86400.0, seconds, 3)

    def test_axis_break_multiple_n(self):
        """n = 2 (20 s): the x axis breaks at the 60 s steps."""
        doc, graph, axis = self._multi_rate()
        doc.time_layout = TimeLayout(k=10.0, n=2.0)          # 20 s: every 60 s step
        doc.finalize()
        self.assertEqual(axis.kind, "axis-broken")
        self.assertEqual(len(axis.settings["breakPoints"]), 2 * 50)
        doc.time_layout = TimeLayout(k=10.0, n=100.0)        # 1000 s: the 2.3 h hole only
        doc.finalize()
        self.assertEqual(axis.kind, "axis-broken")
        self.assertEqual(len(axis.settings["breakPoints"]), 2)
        doc.time_layout = TimeLayout(k=10.0, n=1000.0)       # 10000 s: none
        doc.finalize()
        self.assertEqual(axis.kind, "axis")
        self.assertNotIn("breakPoints", axis.settings)
        doc.time_layout = TimeLayout(k=100.0)                # redo with K = 100 (100 s)
        doc.finalize()
        x = doc.datasets[graph.children[-1].settings["xData"]][1]["data"]
        self.assertEqual(int(np.isnan(x).sum()), 1)           # the 2.3 h hole only
        self.assertEqual(axis.kind, "axis")

    def test_axis_break_only_where_no_trace_has_data(self):
        """The axis breaks only where no trace on it has data."""
        doc = NativeDocument("t")
        a = 61349.5 + np.arange(100) * 10.0 / 86400.0          # 10 s, then a 1 h hole
        a = np.r_[a, a[-1] + 3600.0 / 86400.0 + np.arange(100) * 10.0 / 86400.0]
        b = 61349.5 + np.arange(200) * 270.0 / 86400.0         # 270 s cadence, no hole
        doc.SetData("a__x__sorted", a)
        doc.SetData("a", np.ones(a.size))
        doc.SetData("b__x__sorted", b)
        doc.SetData("b", np.ones(b.size))
        graph = doc.Root.Add("page", name="p").Add("graph", name="g", autoadd=False)
        axis = graph.Add("axis", name="x")
        axis.settings["label"] = "time"
        graph.Add("xy", name="a", xData="a__x__sorted", yData="a")
        graph.Add("xy", name="b", xData="b__x__sorted", yData="b")
        doc.time_layout = TimeLayout(k=10.0, n=1.0)
        doc.finalize()
        # trace a breaks (3600 s > 100 s); b covers the hole -> no axis break
        self.assertEqual(axis.kind, "axis")
        self.assertTrue(np.isnan(doc.datasets[graph.children[1].settings["xData"]][1]["data"]).any())

    def test_padding_rows_and_non_time_axes(self):
        """MJD 0 padding rows are ignored; axes that are not time are left
        alone.
        """
        doc = NativeDocument("t")
        doc.SetData("f__freq__sorted", np.linspace(1e9, 2e9, 5))     # not time
        doc.SetData("level", np.arange(5.0))
        graph, axis = _graph(doc, "f__freq__sorted", ["level"], label="Frequency")
        doc.finalize()
        self.assertEqual(graph.children[-1].settings["xData"], "f__freq__sorted")
        self.assertNotIn("mode", axis.settings)


class FitsReader(unittest.TestCase):
    """FITS AutoPlot 0.4.0: padding rows, full headers, page order."""

    def test_padding_rows_headers_and_order(self):
        """Padding rows (DMJD 0, blank tags) are dropped and counted; page notes
        hold every header card; results sort by first time, not file name.
        """
        from astropy.io import fits
        import FITS_AutoPlot as F
        from _autoplot_common import MemoryAwareCache, MemoryMonitorConfig
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index, start in enumerate((61044.0, 61043.0)):
                t = np.r_[start + np.arange(6) / 86400.0, 0.0, 0.0]
                cols = [fits.Column(name="DMJD", format="D", unit="d", array=t),
                        fits.Column(name="CHANNELA", format="8A",
                                    array=["A", "A", "C", "C", "A", "C", "", ""]),
                        fits.Column(name="DELTAT", format="J", unit="NanoSeconds",
                                    array=np.arange(8))]
                hdu = fits.BinTableHDU.from_columns(cols, name="OnePpsDeltas")
                hdu.header["COMMENT"] = "DMJD: time sample taken"
                path = os.path.join(folder, "f%d.fits" % index)
                fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
                paths.append(path)
            cache = MemoryAwareCache(MemoryMonitorConfig())
            reader = F.FITSProcessor("astropy", cache)
            results = {p: reader.read(p) for p in paths}
            data = results[paths[0]]
            self.assertEqual(data["dropped_rows"], {"ONEPPSDELTAS": 2})
            self.assertEqual(set(data["tag_groups"]["ONEPPSDELTAS"]), {("A",), ("C",)})
            self.assertTrue(np.all(np.asarray(data["columns"][("ONEPPSDELTAS", "DMJD")]) > 0))
            self.assertIn("--- HDU 1: ONEPPSDELTAS ---", data["header"])
            self.assertTrue(any(line.startswith("EXTNAME") for line in data["header"]))
            self.assertTrue(any("time sample taken" in line for line in data["header"]))
            order = [p for p, _d in sorted(results.items(), key=F._result_order)]
            self.assertEqual(order, [paths[1], paths[0]])      # earlier data first
            doc = NativeDocument()
            F.push_to_veusz(doc, paths[0], data, "astropy")
            self.assertEqual(doc.datasets["f0__ONEPPSDELTAS__DMJD__utc"][0], "date")
            script = doc.SaveToString()
            self.assertNotIn("axis-broken", script)
            self.assertIn("Local time (ISO 8601", script)


# %% Run
if __name__ == "__main__":
    unittest.main()
