# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_time_clocks.py

_autoplot_native 0.5.6 time-axis clocks: local time read from the time zone
of the computer that runs the program (every sample its own offset, the
zone named on the axis label), and the Local time / UTC / MJD options that
draw every page with date-time x axes once per ticked clock, in the Plots
tab and in the saved .vszh5 (default: local time only).

Run:  python -m unittest discover -s tests
      (the Veusz render test runs when the module is run alone:
      python -m unittest tests.test_time_clocks)

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
#              Clock names; pages per clock (names, order, the non-time page
#              left alone); exact UTC and MJD values and the same gaps and
#              axis breaks on every clock; switching clocks between saves;
#              the computer's zone in five TZ settings and across DST; the
#              Plots tab descriptions; the GUI boxes and --time-axes; a
#              saved .vszh5 read back with h5py and rendered in Veusz.
# %%%%% Function Descriptions
        clock_project: A NativeDocument: one time page "p1" (an MJD column with a padding row, 1-minute samples and a 23 h hole, drawn from its UTC date dataset with the...
        page_names: Names of the pages of ``doc`` in order.
        x_axis: The x axis widget of the first graph of page ``name``.
        plotted: (x dataset name, x values) the first trace of page ``name`` draws.
        zone_label_in: (label, tick format) of a local axis made by a child Python whose ``TZ`` is ``tz`` and VSZH5_DISPLAY_TZ unset - the computer's zone.
        ClockNameTests (class): time_clocks() and TimeLayout.clocks.
        ClockPageTests (class): Pages per clock, their data, gaps and breaks.
        ComputerZoneTests (class): Local time follows the computer's time zone (nothing is fixed).
        PlotsTabTests (class): document_pages() of the clock pages.
        OptionTests (class): The GUI boxes and the --time-axes option.
        SavedFileTests (class): The saved .vszh5 and a Veusz render.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        MJD0: first MJD of the test series (2026-10-27, daylight saving time in New York)
        APP: the QApplication of the Qt tests (kept alive)
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    (the Veusz render test runs when the module is run alone:
    python -m unittest tests.test_time_clocks)
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import argparse
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import vszh5_writer as w                                  # noqa: E402
import _autoplot_native as native                          # noqa: E402
from _autoplot_native import (MJD_VEUSZ_EPOCH, NativeDocument, TimeLayout,  # noqa: E402
                              time_clocks)

MJD0 = 61340.0
APP = None


# %% Helpers
def clock_project():
    """A NativeDocument: one time page "p1" (an MJD column with a padding
    row, 1-minute samples and a 23 h hole, drawn from its UTC date dataset
    with the exact MJD given) and one frequency page "spec"."""
    doc = NativeDocument("clocks")
    mjd = np.r_[MJD0 + np.arange(60) / 1440.0, MJD0 + 1.0 + np.arange(60) / 1440.0]
    mjd[3] = 0.0                                          # padding row
    utc = (mjd - MJD_VEUSZ_EPOCH) * 86400.0
    utc[mjd <= 0.0] = np.nan
    doc.SetData("f__DMJD__raw", mjd)
    doc.SetDataDateTime("f__DMJD__utc", utc, mjd="f__DMJD__raw")
    doc.SetData("f__temp", np.cos(np.arange(120.0) / 7.0))
    graph = doc.Root.Add("page", name="p1").Add("graph", name="g")
    graph.Add("xy", name="t", xData="f__DMJD__utc", yData="f__temp")
    doc.SetData("freq", np.linspace(1e9, 2e9, 11))
    doc.SetData("amp", np.arange(11.0))
    doc.Root.Add("page", name="spec").Add("graph", name="g").Add(
        "xy", name="s", xData="freq", yData="amp")
    return doc, mjd, utc


def page_names(doc):
    """Names of the pages of ``doc`` in order."""
    return [c.name for c in doc.Root.children if c.kind == "page"]


def x_axis(doc, name):
    """The x axis widget of the first graph of page ``name``."""
    page = next(c for c in doc.Root.children if c.name == name)
    graph = next(w_ for w_ in page.WalkWidgets() if w_.kind == "graph")
    return next(c for c in graph.children if c.kind in ("axis", "axis-broken") and c.name == "x")


def plotted(doc, name):
    """(x dataset name, x values) the first trace of page ``name`` draws."""
    page = next(c for c in doc.Root.children if c.name == name)
    xy = next(w_ for w_ in page.WalkWidgets() if w_.kind == "xy")
    x_name = xy.settings["xData"]
    return x_name, np.asarray(doc.datasets[x_name][1]["data"], dtype=float)


def zone_label_in(tz, mjd):
    """(label, tick format) of a local axis made by a child Python whose
    ``TZ`` is ``tz`` and VSZH5_DISPLAY_TZ unset - the computer's zone."""
    code = (
        "import sys, numpy as np; sys.path.insert(0, %r)\n"
        "from _autoplot_native import NativeDocument, MJD_VEUSZ_EPOCH\n"
        "doc = NativeDocument('z')\n"
        "mjd = np.array(%r)\n"
        "doc.SetDataDateTime('u', (mjd - MJD_VEUSZ_EPOCH) * 86400.0)\n"
        "doc.SetData('v', np.arange(mjd.size, dtype=float))\n"
        "g = doc.Root.Add('page', name='p').Add('graph', name='g')\n"
        "g.Add('xy', name='t', xData='u', yData='v')\n"
        "doc.finalize()\n"
        "x = [c for c in g.children if c.name == 'x'][0]\n"
        "print(x.settings['label']); print(x.settings['TickLabels/format'])\n"
    ) % (str(ROOT), [float(v) for v in mjd])
    env = dict(os.environ, TZ=tz)
    env.pop(w.DISPLAY_TZ_ENV, None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=120, check=True).stdout.strip().split("\n")
    return out[-2], out[-1]


# %% Tests
class ClockNameTests(unittest.TestCase):
    """time_clocks() and TimeLayout.clocks."""

    def test_names(self):
        """Default local; text, lists and "all" in page order; errors."""
        self.assertEqual(time_clocks(None), ("local",))
        self.assertEqual(TimeLayout().clocks, ("local",))
        self.assertEqual(time_clocks("mjd, UTC"), ("utc", "mjd"))
        self.assertEqual(time_clocks(["mjd", "local"]), ("local", "mjd"))
        self.assertEqual(time_clocks("all"), ("local", "utc", "mjd"))
        for bad in ("", [], "gps", ["local", "tai"]):
            with self.assertRaises(ValueError):
                time_clocks(bad)
        # the gap key is unchanged by the clocks (clocks are compared per axis)
        self.assertEqual(TimeLayout(clocks="all").key(), TimeLayout().key())
        self.assertIn("UTC, MJD", TimeLayout(clocks="all").describe_clocks())


class ClockPageTests(unittest.TestCase):
    """Pages per clock, their data, gaps and breaks."""

    def setUp(self):
        """A fixed zone (-05:00) so labels do not depend on this computer."""
        self.saved = os.environ.get(w.DISPLAY_TZ_ENV)
        os.environ[w.DISPLAY_TZ_ENV] = "America/New_York"

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self.saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self.saved

    def test_default_is_local_only(self):
        """Default: the 0.5.5 pages - same names, local axis, no copies."""
        doc, _mjd, _utc = clock_project()
        doc.finalize()
        self.assertEqual(page_names(doc), ["p1", "spec"])
        axis = x_axis(doc, "p1")
        self.assertEqual(axis.time_clock, "local")
        self.assertEqual(axis.settings["label"], "Local time (ISO 8601, EDT, UTC-04:00)")
        self.assertTrue(axis.settings["TickLabels/format"].endswith("-04:00"))
        self.assertFalse([n for n in doc.datasets if n.endswith("__mjd") or "_mjd_" in n
                          or n.endswith("__utc_gaps")])

    def test_all_clocks(self):
        """Local, UTC and MJD pages side by side; the frequency page once."""
        doc, mjd, utc = clock_project()
        doc.time_layout = TimeLayout(n=10.0, clocks="all")
        doc.finalize()
        self.assertEqual(page_names(doc), ["p1", "p1_utc", "p1_mjd", "spec"])
        local, utc_axis, mjd_axis = (x_axis(doc, n) for n in ("p1", "p1_utc", "p1_mjd"))
        self.assertEqual((local.time_clock, utc_axis.time_clock, mjd_axis.time_clock),
                         ("local", "utc", "mjd"))
        self.assertEqual(utc_axis.settings["label"], "UTC (ISO 8601)")
        self.assertEqual(utc_axis.settings["mode"], "datetime")
        self.assertTrue(utc_axis.settings["TickLabels/format"].endswith("%VDSZ"))
        self.assertEqual(mjd_axis.settings["label"], "MJD (UTC)")
        self.assertEqual(mjd_axis.settings["mode"], "numeric")
        self.assertEqual(mjd_axis.settings["TickLabels/format"], "%.12Vg")
        self.assertIsNone(x_axis(doc, "spec").time_clock)
        # the data: UTC exactly the stored UTC, MJD exactly the stored MJD
        _n, u = plotted(doc, "p1_utc")
        _n, m = plotted(doc, "p1_mjd")
        good = np.isfinite(utc)
        self.assertTrue(np.array_equal(u[np.isfinite(u)], utc[good]))
        self.assertTrue(np.array_equal(m[np.isfinite(m)], mjd[good]))
        # the same gap rows (padding row and the hole) on every clock
        _n, lo = plotted(doc, "p1")
        for values in (lo, u, m):
            self.assertEqual(values.size, 121)
            self.assertEqual(list(np.flatnonzero(np.isnan(values))),
                             list(np.flatnonzero(np.isnan(lo))))
        # the same axis break, between the same two samples
        for axis, values in ((local, lo), (utc_axis, u), (mjd_axis, m)):
            self.assertEqual(axis.kind, "axis-broken")
            start, stop = axis.settings["breakPoints"]
            finite = values[np.isfinite(values)]
            self.assertEqual(int(np.sum(finite <= start)), 59)
            self.assertEqual(int(np.sum(finite >= stop)), 60)
        # notes name the clock
        self.assertIn("Time axis: UTC", doc.Root.children[1].children[0].settings["notes"])
        self.assertIn("Modified Julian Date", doc.Root.children[2].children[0].settings["notes"])
        # tags: the copies are UTC / MJD time data with gap rows
        self.assertIn(plotted(doc, "p1_mjd")[0], doc.tags["type_time_mjd"])
        self.assertIn(plotted(doc, "p1_utc")[0], doc.tags["type_time_utc"])

    def test_switching_clocks(self):
        """Saving again with other clocks adds and removes pages and their
        copies; the same clocks twice give the same script."""
        doc, _mjd, _utc = clock_project()
        doc.finalize()
        local_only = set(doc.datasets)
        doc.time_layout = TimeLayout(clocks="all")

        def script():
            """The saved script without its stamp lines (# ...)."""
            return [line for line in doc.SaveToString().split("\n") if not line.startswith("#")]
        first = script()
        self.assertEqual(first, script())
        doc.time_layout = TimeLayout(clocks="utc")
        doc.finalize()
        self.assertEqual(page_names(doc), ["p1_utc", "spec"])
        self.assertEqual(x_axis(doc, "p1_utc").settings["label"], "UTC (ISO 8601)")
        doc.time_layout = TimeLayout(clocks="mjd,utc")
        doc.finalize()
        self.assertEqual(page_names(doc), ["p1_utc", "p1_mjd", "spec"])
        doc.time_layout = TimeLayout()
        doc.finalize()
        self.assertEqual(page_names(doc), ["p1", "spec"])
        self.assertEqual(set(doc.datasets), local_only)
        doc.time_layout = TimeLayout(clocks="all")
        self.assertEqual(script(), first)

    def test_clock_without_exact_mjd(self):
        """A date dataset without SetDataDateTime(mjd=...) draws UTC / 86400
        + 54832, within one float64 step of the MJD."""
        doc = NativeDocument("t")
        mjd = MJD0 + np.arange(100) / 86400.0
        doc.SetDataDateTime("u", (mjd - MJD_VEUSZ_EPOCH) * 86400.0)
        doc.SetData("v", np.arange(100.0))
        doc.Root.Add("page", name="p").Add("graph", name="g").Add(
            "xy", name="t", xData="u", yData="v")
        doc.time_layout = TimeLayout(clocks="mjd")
        doc.finalize()
        name, values = plotted(doc, "p_mjd")
        self.assertEqual(name, "u_mjd")
        self.assertLessEqual(float(np.max(np.abs(values - mjd))), 2.0 * np.spacing(MJD0))

    def test_long_label_fits(self):
        """The page is made wide enough for a long time-axis label."""
        doc, _mjd, _utc = clock_project()
        doc.finalize()
        axis = x_axis(doc, "p1")
        axis.settings["label"] = "Local time " + "x" * 120
        page = doc.Root.children[0]
        native._fit_page(doc, page, {})
        graph = page.children[0]
        width = native._cm(page.settings.get("width"), 25.0)
        margins = sum(native._cm(graph.settings.get(side), d) for side, d in
                      (("leftMargin", w.VEUSZ_GRAPH_MARGINS_CM["leftMargin"]),
                       ("rightMargin", w.VEUSZ_GRAPH_MARGINS_CM["rightMargin"])))
        self.assertGreaterEqual(width - margins + 1e-9, native._time_label_cm(axis))


@unittest.skipUnless(hasattr(time, "tzset"), "TZ is read this way on Linux / macOS only")
class ComputerZoneTests(unittest.TestCase):
    """Local time follows the computer's time zone (nothing is fixed)."""

    def test_five_zones(self):
        """The same instants in five TZ settings of the computer."""
        winter = [61360.5, 61360.5 + 1.0 / 24.0]           # 2026-11-16 12:00 UTC
        for tz, offset, name in (("America/New_York", "-05:00", "EST"),
                                 ("America/Los_Angeles", "-08:00", "PST"),
                                 ("Europe/Berlin", "+01:00", "CET"),
                                 ("Asia/Kolkata", "+05:30", "IST"),
                                 ("UTC", "+00:00", "")):
            label, fmt = zone_label_in(tz, winter)
            self.assertTrue(fmt.endswith(offset), (tz, fmt))
            self.assertEqual(label, "Local time (ISO 8601, %sUTC%s)" % (
                name + ", " if name else "", offset), tz)

    def test_daylight_saving(self):
        """Summer data in New York read UTC-04:00 EDT; data across the
        change list both offsets with the minute the clocks changed."""
        label, fmt = zone_label_in("America/New_York", [61240.5, 61240.6])   # 2026-07
        self.assertEqual(label, "Local time (ISO 8601, EDT, UTC-04:00)")
        across = [61344.0 + h / 24.0 for h in range(0, 48, 2)]              # 2026-10-31 .. 11-02
        label, fmt = zone_label_in("America/New_York", across)
        self.assertEqual(label, "Local time (ISO 8601; UTC-04:00 EDT, UTC-05:00 EST from "
                                "2026-11-01T01:00)")
        self.assertTrue(fmt.endswith("%VDS"))


class PlotsTabTests(unittest.TestCase):
    """document_pages() of the clock pages."""

    def test_descriptions(self):
        """Local offset, UTC "Z", MJD drawn as numbers (not as dates)."""
        from _autoplot_pyqtgraph import document_pages
        os.environ[w.DISPLAY_TZ_ENV] = "America/New_York"
        try:
            doc, _mjd, _utc = clock_project()
            doc.time_layout = TimeLayout(clocks="all")
            pages = {p["name"]: p["graphs"][0] for p in document_pages(doc)}
        finally:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        self.assertEqual(list(pages), ["p1", "p1_utc", "p1_mjd", "spec"])
        self.assertEqual((pages["p1"]["x_kind"], pages["p1"]["x_tz"]), ("veusz_date", "-04:00"))
        self.assertEqual((pages["p1_utc"]["x_kind"], pages["p1_utc"]["x_tz"]), ("veusz_date", "Z"))
        self.assertEqual(pages["p1_mjd"]["x_kind"], "numeric")
        self.assertEqual(pages["p1_mjd"]["x_clock"], "mjd")

    def test_mjd_page_gaps_and_ticks(self):
        """An MJD page uses the time gap rule (on seconds) and its ticks keep
        the digits their spacing needs (pyqtgraph alone: "61150.1" on all)."""
        from _autoplot_pyqtgraph import axis_break_regions, tick_texts, trace_arrays
        doc, mjd, _utc = clock_project()
        doc.time_layout = TimeLayout(n=10.0, clocks="mjd")
        doc.finalize()
        x_name = plotted(doc, "p1_mjd")[0]
        trace = {"x_name": x_name, "y_name": "f__temp_gaps"}
        x, _y, is_time = trace_arrays(doc, trace, "numeric", True, days=True)
        self.assertFalse(is_time)
        self.assertEqual(int(np.isnan(x).sum()), 2)            # padding row + hole, no more
        regions = axis_break_regions(doc, [x[np.isfinite(x)]], False, days=True)
        self.assertEqual(len(regions), 1)
        self.assertTrue(mjd[59] < regions[0][0] < regions[0][1] < mjd[60])
        # where the saved project breaks the MJD axis (found in UTC seconds)
        np.testing.assert_allclose(regions[0], x_axis(doc, "p1_mjd").settings["breakPoints"],
                                   rtol=0.0, atol=1e-9)
        self.assertEqual(tick_texts([61150.07, 61150.08, 61150.09], 0.01),
                         ["61150.07", "61150.08", "61150.09"])
        self.assertEqual(tick_texts([10000000.0, 10000020.0], 20.0), ["10000000", "10000020"])
        self.assertIsNone(tick_texts([1.0, 2.0], 1.0))


class OptionTests(unittest.TestCase):
    """The GUI boxes and the --time-axes option."""

    def test_command_line(self):
        """--time-axes only where asked for; parsed into TimeLayout.clocks."""
        parser = argparse.ArgumentParser()
        native.add_gap_arguments(parser, time_axes=True)
        self.assertEqual(native.layout_from_args(parser.parse_args([])).clocks, ("local",))
        args = parser.parse_args(["--time-axes", "utc,mjd"])
        self.assertEqual(native.layout_from_args(args).clocks, ("utc", "mjd"))
        plain = argparse.ArgumentParser()
        native.add_gap_arguments(plain)
        self.assertEqual(native.layout_from_args(plain.parse_args([])).clocks, ("local",))
        with self.assertRaises(SystemExit):
            plain.parse_args(["--time-axes", "utc"])

    def test_boxes(self):
        """Local only by default; the last ticked box stays ticked; the
        Gaps box hands the clocks to its TimeLayout; the zone is named."""
        global APP
        try:
            from qtpy.QtWidgets import QApplication
            import _suite_gui
        except Exception as exc:                      # pragma: no cover - no Qt
            self.skipTest("Qt not available: %s" % exc)
        APP = QApplication.instance() or QApplication([])
        box = _suite_gui.GapOptionsBox(time_axes=True)
        times = box.time_axes_box
        self.assertEqual(box.time_layout().clocks, ("local",))
        times.local_cb.setChecked(False)
        self.assertTrue(times.local_cb.isChecked())
        times.utc_cb.setChecked(True)
        times.mjd_cb.setChecked(True)
        times.local_cb.setChecked(False)
        self.assertEqual(box.time_layout().clocks, ("utc", "mjd"))
        times.set_clocks(["local", "mjd"])
        self.assertEqual(times.clocks(), ("local", "mjd"))
        self.assertIn("UTC", times.zone_label.text())
        for check in (times.local_cb, times.utc_cb, times.mjd_cb):
            self.assertTrue(check.toolTip())
        self.assertIsNone(_suite_gui.GapOptionsBox().time_axes_box)
        # every change is announced (the tools redraw their Plots tab)
        heard = []
        times.clocks_changed.connect(heard.append)
        times.utc_cb.setChecked(True)
        times.set_clocks(["utc"])
        times.utc_cb.setChecked(False)                   # the last box: ticked again, no change
        self.assertTrue(times.utc_cb.isChecked())
        self.assertEqual(heard, [("local", "utc", "mjd"), ("utc",)])

    def test_zone_description(self):
        """The GUI line names the zone and today's offset."""
        os.environ[w.DISPLAY_TZ_ENV] = "Asia/Kolkata"
        try:
            text = native.local_zone_description(1763294400.0)      # 2025-11-16
        finally:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        self.assertIn("IST", text)
        self.assertIn("UTC+05:30 now", text)


class SavedFileTests(unittest.TestCase):
    """The saved .vszh5 and a Veusz render."""

    def test_saved_project(self):
        """Every clock page and its data are in the file, values exact."""
        import h5py
        os.environ[w.DISPLAY_TZ_ENV] = "America/New_York"
        try:
            doc, mjd, utc = clock_project()
            doc.time_layout = TimeLayout(clocks="all")
            with tempfile.TemporaryDirectory() as folder:
                path = doc.save(os.path.join(folder, "clocks.vszh5"))
                mjd_name = plotted(doc, "p1_mjd")[0]
                with h5py.File(path, "r") as handle:
                    script = handle["Veusz"]["Document"]["document"][()][0]
                    script = script.decode() if isinstance(script, bytes) else str(script)
                    stored = np.asarray(handle["Veusz"]["Data"][mjd_name]["data"])
        finally:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        for page in ("p1", "p1_utc", "p1_mjd", "spec"):
            self.assertIn("Add('page', name=%r" % page, script)
        self.assertIn("UTC (ISO 8601)", script)
        self.assertIn("MJD (UTC)", script)
        self.assertTrue(np.array_equal(stored[np.isfinite(stored)], mjd[np.isfinite(utc)]))

    def test_veusz_render(self):
        """Veusz opens the project and draws every page, in a child Python
        (skipped without Veusz, and in a full test run that blocks it)."""
        try:
            import veusz.document                     # noqa: F401
        except Exception as exc:
            self.skipTest("Veusz not installed: %s" % exc)
        os.environ[w.DISPLAY_TZ_ENV] = "America/New_York"
        try:
            doc, _mjd, _utc = clock_project()
            doc.time_layout = TimeLayout(n=10.0, clocks="all")
            with tempfile.TemporaryDirectory() as folder:
                path = doc.save(os.path.join(folder, "clocks.vszh5"))
                code = (
                    "import os\n"
                    "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
                    "from veusz import document, qtall, widgets, dataimport\n"
                    "app = qtall.QApplication.instance() or qtall.QApplication([])\n"
                    "doc = document.Document()\n"
                    "doc.load(%r, mode='hdf5')\n"
                    "cmd = document.CommandInterface(doc)\n"
                    "for i, page in enumerate(doc.basewidget.children):\n"
                    "    cmd.Export(%r %% page.name, page=[i])\n") % (
                        path, os.path.join(folder, "%s.png"))
                out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                     text=True, timeout=300)
                pngs = sorted(p.name for p in Path(folder).glob("*.png") if p.stat().st_size > 1000)
        finally:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(pngs, ["p1.png", "p1_mjd.png", "p1_utc.png", "spec.png"])


if __name__ == "__main__":
    unittest.main()
