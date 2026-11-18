# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_gap_enforcement.py

Are the gap options enforced everywhere (2026-11-17, task 3 of the
2026-11-17 requests)?  The fixed-gap texts are read whole and exactly
(_autoplot_native 0.5.8: "00:10:00" was 10 h, "1x2" 3 h, ".5h" 5 h and
"16min" 959.9999999999999 s), an entry that cannot be read is reported (log,
red frame, argparse error) instead of ignored, K / n / the fixed gaps are
kept in range, the local clock is broken where UTC is (a daylight-saving
change is no gap), and the Plots tab follows every option changed after a
run and draws the project's own breaks (_suite_gui 0.1.6, _autoplot_common
0.4.8, _autoplot_pyqtgraph 0.3.7).

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
#              Gap texts (accepted, exact, rejected), TimeLayout ranges,
#              the --gap-* flags, UTC-found breaks on the local clock across
#              both US daylight-saving changes, the Plots tab (no second cut
#              of a clock trace, the project's breaks shaded), the GUI boxes
#              (red frame, log lines, layout_changed, follow()) and the FITS
#              hours-box fallback.
# %%%%% Function Descriptions
        veusz_seconds: Veusz date seconds (UTC, since 2009-01-01) of the ISO 8601 UTC time ``iso`` plus ``minutes`` (an array).
        dst_project: A NativeDocument: page "p" with one trace of 1-minute UTC samples from ``start`` at ``minutes``, laid out by ``layout`` (finalized).
        axis_of: The x axis of the first graph of page ``name``.
        GapTextTests (class): check_gap_string / check_gap_seconds / parse_gap_string.
        LayoutRangeTests (class): TimeLayout keeps K, n and the fixed gaps in range.
        CommandLineTests (class): --gap-* values are checked when the command line is read.
        DaylightSavingTests (class): Axis breaks of the local clock are found in UTC (0.5.8).
        PlotsTabTests (class): The Plots tab draws the project's cuts and breaks of a clock axis.
        GuiTests (class): GapOptionsBox and the FITS AutoPlot gap controls.
        PluginTests (class): Both plugins: an unreadable small-gap text falls back to the hours box, an unreadable large-gap text is not used, both are logged.
        _suite_gui_help: The tool tip the small-gap box of GapOptionsBox starts with.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        NEW_YORK: the zone of the daylight-saving tests (US Eastern)
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Release notes: RELEASE_suite-2026-11-09.md section 18.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import argparse
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vszh5_writer as w                                         # noqa: E402
import _autoplot_native as native                                # noqa: E402
from _autoplot_native import (NativeDocument, TimeLayout, check_gap_string,   # noqa: E402
                              check_gap_seconds, parse_gap_string, parse_gap_seconds)

NEW_YORK = "America/New_York"


# %% Helpers
def veusz_seconds(iso, minutes):
    """Veusz date seconds (UTC, since 2009-01-01) of the ISO 8601 UTC time
    ``iso`` plus ``minutes`` (an array)."""
    start = (np.datetime64(iso, "s") - np.datetime64("2009-01-01T00:00:00", "s")).astype(float)
    return start + 60.0 * np.asarray(minutes, dtype=float)


def dst_project(start, minutes, layout):
    """A NativeDocument: page "p" with one trace of 1-minute UTC samples from
    ``start`` at ``minutes``, laid out by ``layout`` (finalized)."""
    doc = NativeDocument("dst")
    doc.SetDataDateTime("t", veusz_seconds(start, minutes))
    doc.SetData("v", np.sin(np.asarray(minutes, dtype=float) / 7.0))
    doc.Root.Add("page", name="p").Add("graph", name="g").Add("xy", name="s", xData="t",
                                                              yData="v")
    doc.time_layout = layout
    doc.finalize()
    return doc


def axis_of(doc, name):
    """The x axis of the first graph of page ``name``."""
    page = next(c for c in doc.Root.children if c.name == name)
    graph = next(g for g in page.WalkWidgets() if g.kind == "graph")
    return next(c for c in graph.children if c.kind in ("axis", "axis-broken") and c.name == "x")


# %% Gap texts
class GapTextTests(unittest.TestCase):
    """check_gap_string / check_gap_seconds / parse_gap_string."""

    def test_accepted_exact(self):
        """Every documented form, read whole and exactly (seconds)."""
        cases = {"": 0.0, None: 0.0, "0": 0.0, "120": 432000.0, "2": 7200.0, "90s": 90.0,
                 "15min": 900.0, "16min": 960.0, "2h": 7200.0, "3d": 259200.0,
                 "1m": 2629800.0, "1m 3d 2h": 2896200.0, "5M 3d 2H": 13415400.0,
                 "5m3d2h": 13415400.0, "2h 30min": 9000.0, "1.5h": 5400.0, ".5h": 1800.0,
                 "1e3s": 1000.0, "0.1h": 360.0, "0:15": 900.0, "00:15:00": 900.0,
                 "00:10:00": 600.0, "1:30:15.5": 5415.5, "10 minutes": 600.0,
                 "1 hour": 3600.0, "2 days 1 hour": 176400.0, "47h 59min": 172740.0,
                 "+3": 10800.0, "3.": 10800.0, " 15min ": 900.0}
        for text, seconds in cases.items():
            got, problem = check_gap_seconds(text)
            self.assertIsNone(problem, text)
            self.assertEqual(got, seconds, text)              # exact, not almost
            days, problem = check_gap_string(text)
            self.assertIsNone(problem, text)
            self.assertEqual(days, seconds / 86400.0 if seconds else 0.0, text)

    def test_rejected(self):
        """Anything not read whole is refused with a reason (0 = not used)."""
        for text in ("1x2", "00:10:00x", "1:75", "1:00:60", "1:2:3:4", "-2h", "5ms", "2w",
                     "1d 2", "12 30", "inf", "nan", "abc", "15 mins ago", "1d,2h", "1e400",
                     "h", ":", "1:"):
            seconds, problem = check_gap_seconds(text)
            self.assertEqual(seconds, 0.0, text)
            self.assertTrue(problem, text)
            self.assertEqual(check_gap_string(text), (0.0, problem), text)

    def test_logged_not_ignored(self):
        """parse_gap_string / parse_gap_seconds log the entry, why and which gap."""
        heard = []
        self.assertEqual(parse_gap_string("00:10:00x", heard.append, "small gap"), 0.0)
        self.assertEqual(parse_gap_seconds("5ms", heard.append, "large gap"), 0.0)
        self.assertEqual(len(heard), 2)
        self.assertIn("small gap '00:10:00x' is not used", heard[0])
        self.assertIn("large gap '5ms' is not used: unknown unit 'ms'", heard[1])
        self.assertEqual(parse_gap_seconds("16min", heard.append), 960.0)
        self.assertEqual(len(heard), 2)                       # a readable entry: silent

    def test_describe(self):
        """The read-back of a gap for the tool tip."""
        self.assertEqual(native.describe_gap_seconds(900.0), "15 min")
        self.assertEqual(native.describe_gap_seconds(0.0), "not set")
        self.assertEqual(native.describe_gap_days(1.5), "1.5 d")


# %% TimeLayout ranges
class LayoutRangeTests(unittest.TestCase):
    """TimeLayout keeps K, n and the fixed gaps in range."""

    def test_ranges(self):
        """K >= 1 (not a number = default), n >= 0, fixed gaps finite or 0."""
        self.assertEqual(TimeLayout(k=0.5).k, 1.0)
        self.assertEqual(TimeLayout(k=-3).k, 1.0)
        self.assertEqual(TimeLayout(k=float("nan")).k, w.GAP_K)
        self.assertEqual(TimeLayout(k=float("inf")).k, w.GAP_K)
        self.assertEqual(TimeLayout(k=25.0).k, 25.0)
        self.assertEqual(TimeLayout(n=float("nan")).n, 0.0)
        self.assertEqual(TimeLayout(n=-1).n, 0.0)
        for bad in (float("inf"), float("nan"), -5.0, None):
            layout = TimeLayout(manual_s=bad, axis_manual_s=bad)
            self.assertEqual((layout.manual_s, layout.axis_manual_s), (0.0, 0.0))
        self.assertNotIn("inf", TimeLayout(manual_s=float("inf")).describe())

    def test_below_one_cut_ordinary_steps(self):
        """Why K >= 1: a K below 1 cut the line at the median step."""
        doc = dst_project("2026-06-01T00:00:00", np.arange(100), TimeLayout(k=0.5))
        page = doc.Root.children[0]
        xy = next(c for c in page.WalkWidgets() if c.kind == "xy")
        y = np.asarray(doc.GetData(xy.settings["yData"])[0], float)
        self.assertFalse(np.isnan(y).any())


# %% Command line
class CommandLineTests(unittest.TestCase):
    """--gap-* values are checked when the command line is read."""

    def parser(self):
        """An argparse parser with the gap options (errors raise, not exit)."""
        parser = argparse.ArgumentParser()
        native.add_gap_arguments(parser, time_axes=True)

        def error(message):
            """Raise instead of printing usage and exiting."""
            raise ValueError(message)
        parser.error = error
        return parser

    def test_good_values(self):
        """Clocks and units become the exact TimeLayout seconds."""
        args = self.parser().parse_args(["--gap-small", "00:10:00", "--gap-large", "47h 59min",
                                         "--gap-k", "1", "--gap-n", "0"])
        layout = native.layout_from_args(args)
        self.assertEqual((layout.manual_s, layout.axis_manual_s, layout.k, layout.n),
                         (600.0, 172740.0, 1.0, 0.0))

    def test_bad_values(self):
        """An unreadable text, K < 1, n < 0 or not a number stop the batch."""
        for argv, word in ((["--gap-small", "1x2"], "1x2"), (["--gap-large", "5ms"], "ms"),
                           (["--gap-k", "0.5"], "K must"), (["--gap-k", "nan"], "K must"),
                           (["--gap-n", "-1"], "n must"), (["--gap-n", "x"], "n must")):
            with self.assertRaises(ValueError) as caught:
                self.parser().parse_args(argv)
            self.assertIn(word, str(caught.exception))

    def test_hand_made_args_logged(self):
        """layout_from_args logs an unreadable text of a hand-made Namespace."""
        heard = []
        args = argparse.Namespace(gap_k=10.0, gap_n=100.0, gap_small="1x2", gap_large="",
                                  no_line_breaks=False, no_axis_breaks=False,
                                  no_numeric_gaps=False)
        layout = native.layout_from_args(args, heard.append)
        self.assertEqual(layout.manual_s, 0.0)
        self.assertEqual(len(heard), 1)


# %% Daylight saving
class DaylightSavingTests(unittest.TestCase):
    """Axis breaks of the local clock are found in UTC (0.5.8)."""

    def setUp(self):
        """The US Eastern zone (changes 2026-03-08 07:00 UTC and 2026-11-01 06:00 UTC)."""
        self.saved = os.environ.get(w.DISPLAY_TZ_ENV)
        os.environ[w.DISPLAY_TZ_ENV] = NEW_YORK

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self.saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self.saved

    def breaks(self, start, minutes, **options):
        """{clock page: number of axis breaks} of one trace on all clocks."""
        doc = dst_project(start, minutes, TimeLayout(clocks="all", **options))
        return dict((name, len(axis_of(doc, name).settings.get("breakPoints", [])) // 2)
                    for name in ("p", "p_utc", "p_mjd")), doc

    def test_spring_forward_is_no_gap(self):
        """No UTC gap: the 1-hour jump of the local clock breaks nothing (n = 2 was
        enough to break the local page before)."""
        counts, doc = self.breaks("2026-03-08T05:00:00", np.arange(240), n=2.0)
        self.assertEqual(counts, {"p": 0, "p_utc": 0, "p_mjd": 0})
        self.assertEqual(axis_of(doc, "p").kind, "axis")

    def test_fall_back_keeps_the_hole(self):
        """A 3-hour UTC hole is 2 hours on the local clock: a 2.5 h large gap breaks
        every clock (the local page was not broken before), no sample hidden."""
        minutes = np.r_[np.arange(100), 280 + np.arange(100)]
        counts, doc = self.breaks("2026-11-01T04:00:00", minutes, axis_manual_s=9000.0)
        self.assertEqual(counts, {"p": 1, "p_utc": 1, "p_mjd": 1})
        for name in ("p", "p_utc", "p_mjd"):
            start, stop = axis_of(doc, name).settings["breakPoints"]
            xy = next(c for c in doc.Root.children if c.name == name).WalkWidgets()
            x_name = next(c for c in xy if c.kind == "xy").settings["xData"]
            x = np.asarray(doc.datasets[x_name][1]["data"], float)
            x = x[np.isfinite(x)]
            self.assertEqual(int(np.sum((x > start) & (x < stop))), 0, name)
            self.assertEqual(int(np.sum(x <= start)), 100, name)

    def test_folded_gap_left_out(self):
        """A 41-minute UTC hole over the fall-back goes 19 minutes back on the
        local clock: no room for a break there (none is made, no sample is
        hidden); a 70-minute hole leaves 10 minutes, broken inside them."""
        utc = np.array([0.0, 60.0, 2520.0, 2580.0])          # a 2460 s hole
        local = utc - np.array([4, 4, 5, 5]) * 3600.0         # EDT, then EST
        pairs = w.axis_break_pairs(utc, 1800.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(native._clock_break_pairs(utc, local, pairs), [])
        utc = np.array([0.0, 60.0, 4260.0, 4320.0])          # a 4200 s hole
        local = utc - np.array([4, 4, 5, 5]) * 3600.0
        (start, stop), = native._clock_break_pairs(utc, local, w.axis_break_pairs(utc, 1800.0))
        self.assertTrue(local[1] < start < stop < local[2])
        self.assertAlmostEqual(stop - start, (1 - 2 * w.GAP_AXIS_MARGIN) * 600.0, 6)

    def test_same_as_writer_without_dst(self):
        """Without a daylight-saving change: exactly the writer's pairs."""
        utc = veusz_seconds("2026-06-01T00:00:00", np.r_[np.arange(50), 5000 + np.arange(50)])
        local = utc - 4 * 3600.0
        pairs = w.axis_break_pairs(utc, 1000.0)
        self.assertEqual(native._clock_break_pairs(utc, local, pairs),
                         w.axis_break_pairs(local, 1000.0))


# %% Plots tab
class PlotsTabTests(unittest.TestCase):
    """The Plots tab draws the project's cuts and breaks of a clock axis."""

    @classmethod
    def setUpClass(cls):
        """Import the Plots tab module (skip without pyqtgraph)."""
        try:
            import _autoplot_pyqtgraph as plots
        except Exception as exc:                  # optional stack
            raise unittest.SkipTest("pyqtgraph not importable: %s" % exc)
        cls.plots = plots

    def setUp(self):
        """The US Eastern zone."""
        self.saved = os.environ.get(w.DISPLAY_TZ_ENV)
        os.environ[w.DISPLAY_TZ_ENV] = NEW_YORK

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self.saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self.saved

    def test_spring_forward_not_cut_again(self):
        """The local copy is drawn as laid out (cut in UTC): no cut at the
        spring-forward jump; recut=True would add one."""
        doc = dst_project("2026-03-08T05:00:00", np.arange(240), TimeLayout())
        graph = self.plots.document_pages(doc)[0]["graphs"][0]
        self.assertEqual(graph["x_clock"], "local")
        trace = graph["traces"][0]
        _x, y, _t = self.plots.trace_arrays(doc, trace, graph["x_kind"], False, recut=False)
        self.assertFalse(np.isnan(y).any())
        _x, y, _t = self.plots.trace_arrays(doc, trace, graph["x_kind"], False, recut=True)
        self.assertEqual(int(np.isnan(y).sum()), 1)

    def test_project_breaks_listed(self):
        """document_pages gives the axis' breakPoints as x_breaks."""
        minutes = np.r_[np.arange(100), 3000 + np.arange(100)]
        doc = dst_project("2026-06-01T00:00:00", minutes, TimeLayout(clocks="all"))
        for page in self.plots.document_pages(doc):
            graph = page["graphs"][0]
            axis = axis_of(doc, page["name"])
            points = axis.settings["breakPoints"]
            self.assertEqual(graph["x_breaks"], [tuple(points)])


# %% GUI
class GuiTests(unittest.TestCase):
    """GapOptionsBox and the FITS AutoPlot gap controls."""

    @classmethod
    def setUpClass(cls):
        """One offscreen QApplication."""
        try:
            from qtpy.QtWidgets import QApplication
        except Exception as exc:
            raise unittest.SkipTest("Qt not importable: %s" % exc)
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, seconds=0.8):
        """Let timers fire."""
        start = time.time()
        while time.time() - start < seconds:
            self.app.processEvents()
            time.sleep(0.01)

    def test_text_checked_while_typed(self):
        """Red frame and the reason for an unreadable entry; the read value otherwise."""
        import _suite_gui
        box = _suite_gui.GapOptionsBox()
        box.small_text.setText("00:10:00x")
        self.assertTrue(box.small_text.property("gap_problem"))
        self.assertEqual(box.small_text.styleSheet(), _suite_gui.BAD_GAP_TEXT_STYLE)
        self.assertIn("Not used:", box.small_text.toolTip())
        box.small_text.setText("00:10:00")
        self.assertFalse(box.small_text.property("gap_problem"))
        self.assertEqual(box.small_text.styleSheet(), "")
        self.assertIn("Now: 10 min.", box.small_text.toolTip())
        self.assertTrue(box.small_text.toolTip().startswith(_suite_gui_help()))

    def test_time_layout_logs(self):
        """time_layout(log): the unreadable entry, then the rule in force."""
        import _suite_gui
        box = _suite_gui.GapOptionsBox(time_axes=True)
        box.small_text.setText("1x2")
        box.large_text.setText("2 days 1 hour")
        heard = []
        layout = box.time_layout(heard.append)
        self.assertEqual((layout.manual_s, layout.axis_manual_s), (0.0, 176400.0))
        self.assertIn("small gap '1x2' is not used", heard[0])
        self.assertTrue(heard[1].startswith("Gaps: "))
        self.assertTrue(heard[2].startswith("Time axes: "))
        self.assertEqual(box.time_layout().manual_s, 0.0)    # no log: quiet

    def test_layout_changed_and_follow(self):
        """layout_changed after a change settles; follow() redraws only a
        finished project whose options differ."""
        import _suite_gui
        box = _suite_gui.GapOptionsBox()
        heard = []
        box.layout_changed.connect(lambda: heard.append(1))

        class Panel(object):
            """A Plots tab stand-in that counts its redraws."""
            def __init__(self):
                self.drawn = 0

            def set_document(self, doc):
                self.drawn += 1
                return 1
        panel, doc, ready = Panel(), NativeDocument("f"), [True]
        doc.time_layout = box.time_layout()
        log = []
        box.follow(lambda: doc, lambda: panel, log.append, ready=lambda: ready[0])
        box.k_spin.setValue(12.0)
        box.k_spin.setValue(20.0)                 # two changes, one signal
        self.wait()
        self.assertEqual(heard, [1])
        self.assertEqual(panel.drawn, 1)
        self.assertEqual(doc.time_layout.k, 20.0)
        self.assertTrue(any(line.startswith("Plots tab redrawn") for line in log))
        box.small_text.setText("20min")
        box.small_text.editingFinished.emit()
        self.wait()
        self.assertEqual((panel.drawn, doc.time_layout.manual_s), (2, 1200.0))
        box.small_text.editingFinished.emit()     # same options: no redraw
        self.wait()
        self.assertEqual(panel.drawn, 2)
        ready[0] = False                          # during a run: nothing
        box.axis_cb.setChecked(False)
        self.wait()
        self.assertEqual(panel.drawn, 2)
        self.assertTrue(doc.time_layout.break_axis)

    def test_fits_hours_fallback(self):
        """FITS AutoPlot: an unreadable small-gap text is logged and the hours box
        is used; a readable one overrides it, exactly; the Plots tab follows."""
        import FITS_AutoPlot as fits
        window = fits.FITSAutoPlotWindow()
        heard = []
        window.log = heard.append
        try:
            window.gap_abs_spin.setValue(0.5)
            window.gap_abs_str.setText("1x2")
            self.assertEqual(window.time_layout().manual_s, 1800.0)
            self.assertTrue(any("is not used" in line and "hours box" in line for line in heard))
            heard.clear()
            self.assertEqual(window.time_layout(quiet=True).manual_s, 1800.0)
            self.assertEqual(heard, [])
            window.gap_abs_str.setText("16min")
            self.assertEqual(window.time_layout().manual_s, 960.0)
            window.gap_axis_str.setText("00:10:00x")
            self.assertEqual(window.time_layout().axis_manual_s, 0.0)
            self.assertTrue(window.gap_axis_str.property("gap_problem"))
        finally:
            window.close()


# %% Veusz plugins
class PluginTests(unittest.TestCase):
    """Both plugins: an unreadable small-gap text falls back to the hours box,
    an unreadable large-gap text is not used, both are logged."""

    def setUp(self):
        """A temporary folder with a FITS and a Franks file (7-day hole)."""
        import tempfile
        try:
            from astropy.io import fits
        except Exception as exc:                  # optional stack
            self.skipTest("astropy not importable: %s" % exc)
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        x = np.array([60805, 60805.1, 60805.2, 60812, 60812.1, 60812.2])
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name="DMJD", format="D", array=x),
             fits.Column(name="DELTAT", format="D", unit="ns", array=[1, 2, 3, 4, 5, 6])],
            name="OnePpsDeltas")
        self.fits_path = self.folder / "plugin.fits"
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(self.fits_path)
        self.franks_path = self.folder / "time_gbt.dat"
        self.franks_path.write_text("MJD A B\n================\n60805 1 2\n60805.1 2 3\n"
                                    "60812 3 4\n60812.1 4 5\n", encoding="utf-8")

    def tearDown(self):
        """Remove the temporary folder."""
        self.tmp.cleanup()

    def test_fallback_and_log(self):
        """The saved notes state the hours box (2 h) and no fixed large gap."""
        from unittest.mock import patch
        import h5py
        try:
            import FITS_AutoPlot_plugin as fitsplugin
            import Franks_AutoPlot_plugin as franksplugin
        except Exception as exc:                  # optional stack
            self.skipTest("plugins not importable: %s" % exc)
        from qtpy.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        for module, cls, path in ((fitsplugin, fitsplugin.FITSAutoPlotPlugin, self.fits_path),
                                  (franksplugin, franksplugin.FranksAutoPlotPlugin,
                                   self.franks_path)):
            with self.subTest(plugin=cls.__name__):
                heard = []
                output = self.folder / (cls.__name__ + ".vszh5")
                fields = dict(preselected_files=str(path), output_file=str(output),
                              minimized=False, also_full=False, gap_absolute=2.0,
                              gap_string="1x2", gap_axis_string="00:10:00x")
                plugin = object.__new__(cls)
                with patch.object(module._PluginBatchDialog, "exec_", return_value=1), \
                     patch.object(module._PluginBatchDialog, "append_log",
                                  lambda _self, msg: heard.append(str(msg))), \
                     patch.object(module.QMessageBox, "information"):
                    plugin._apply(None, fields)
                self.assertTrue(any("small gap '1x2' is not used" in line
                                    and "hours box" in line for line in heard), heard)
                self.assertTrue(any("large gap '00:10:00x' is not used" in line
                                    for line in heard), heard)
                with h5py.File(output) as handle:
                    script = handle["Veusz/Document/document"][0].decode()
                self.assertIn("line break (small gap) where dt > 2 h", script)
                self.assertNotIn("x-axis break (large gap) where dt >", script.replace(
                    "x-axis break (large gap) where dt > 100 x", ""))


def _suite_gui_help():
    """The tool tip the small-gap box of GapOptionsBox starts with."""
    from _autoplot_native import GAP_STRING_HELP
    return GAP_STRING_HELP


if __name__ == "__main__":
    unittest.main()
