# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_writer_0031.py

vszh5_writer 0.0.31 and _autoplot_native 0.5.5: x tick labels with
date-times or long numbers turned vertical (-90 degrees, reading downwards;
a date-time on two lines, the date then "T" + the time), margins and page
sizes that let Veusz draw every major tick label, and pages without a date
axis or data ranges written exactly as by master 0.0.30.

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
#              Identity with master 0.0.30 (docs/writer_0.0.31) for pages
#              without dates or ranges; the ports against Veusz; the label
#              decisions and margins; the page helpers; the native fit
#              (save twice = same file); a Veusz render in which every tick
#              label is drawn and stays inside its cell and the page.
# %%%%% Function Descriptions
        veusz_modules: (document, axis, axisticks, formatting, utils, qtall) of the veusz package, or None when Veusz is not installed.
        date_project: A NativeDocument: a grid of four date-time graphs (one hour at 6 s), one graph of Hz values near 10 MHz, one of short numbers, one of long y values...
        LabelRecorder (class): Records, while active, every tick label Veusz makes and whether it draws it (Axis._drawTextWithoutOverlap replayed), with the page size.
        render_and_record: Render every page of ``path``; [(axis path, tick?, drawn?, box, page px)].
        IdentityWith0030 (class): Pages without a date axis or data ranges are written as by 0.0.30.
        PortTests (class): veusz_ticks, veusz_date_ticks and veusz_number_text equal Veusz's own.
        LabelTests (class): datetime_two_line, widths, margin text and axis_tick_room decisions.
        PageHelperTests (class): xy_page, grid_page, style_datetime_axis and time_pair_pages.
        NativeFitTests (class): _autoplot_native._fit_page through NativeDocument.save().
        VeuszRenderTests (class): Veusz draws every tick label, inside the page (skipped without Veusz).
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        MASTER_0030: the registry master 0.0.30 kept in docs/writer_0.0.31
        SHA_0030: SHA-256 of the registry master 0.0.30
        TWO_LINE: the local ISO format on two lines with a -05:00 offset
        APP: the QApplication of the Veusz tests (kept so fonts stay usable)
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    (the Veusz tests run when the module is run alone: python -m unittest
    tests.test_writer_0031; test_native_export blocks Veusz in a full run)
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import hashlib
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import vszh5_writer as w                                  # noqa: E402
from _autoplot_native import NativeDocument, mark_local_time_axis      # noqa: E402
from test_writer_0030 import build_document, document_objects, load_writer   # noqa: E402

MASTER_0030 = ROOT / "docs" / "writer_0.0.31" / "vszh5_writer_0.0.30.py"
SHA_0030 = "26194c8e96572fce2106bb5588d33d7c3f024fbd5169ff626cab2e40aa01f604"
TWO_LINE = "%VDY-%VDm-%VDd\\\\T%VDH:%VDM:%VDS-05:00"
APP = None                    # the QApplication of the Veusz tests (veusz_modules)


# %% Helpers
def veusz_modules():
    """(document, axis, axisticks, formatting, utils, qtall) of the veusz
    package, or None when Veusz is not installed."""
    try:
        from veusz import qtall
        global APP                       # keep it: a collected QApplication aborts on fonts
        APP = qtall.QApplication.instance() or qtall.QApplication([])
        from veusz import document, widgets, dataimport  # noqa: F401  (registers)
        from veusz.widgets import axis, axisticks
        from veusz.utils import formatting
        import veusz.utils as utils
    except Exception:                                      # Veusz is optional
        return None
    return document, axis, axisticks, formatting, utils, qtall


def date_project():
    """A NativeDocument: a grid of four date-time graphs (one hour at 6 s),
    one graph of Hz values near 10 MHz, one of short numbers, one of long
    y values; the UTC dates stored as Veusz date seconds."""
    doc = NativeDocument()
    t = w.epoch_to_veusz(1762678740.0 + 6.0 * np.arange(600))
    doc.SetDataDateTime("t_utc", t)
    doc.SetDataLocalTime("t_utc")
    doc.SetData("level", np.sin(np.arange(600) / 30.0))
    doc.SetData("hz", 9999940.0 + 0.2 * np.arange(600))
    doc.SetData("short", np.arange(600.0) / 10.0)
    doc.SetData("long_y", -0.25 + np.arange(600.0) / 6000.0)
    page = doc.Root.Add("page", name="p")
    grid = page.Add("grid", columns=2)
    for index in range(4):
        graph = grid.Add("graph", name="g%d" % index)
        graph.Add("xy", xData="t_utc", yData="level")
        mark_local_time_axis(graph.x)
    hz = doc.Root.Add("page", name="hz").Add("graph", name="g_hz")
    hz.Add("xy", xData="hz", yData="level")
    hz.x.label.val = "Frequency (Hz)"
    short = doc.Root.Add("page", name="short").Add("graph", name="g_short")
    short.Add("xy", xData="short", yData="long_y")
    short.y.label.val = "Torque"
    return doc


class LabelRecorder(object):
    """Records, while active, every tick label Veusz makes and whether it
    draws it (Axis._drawTextWithoutOverlap replayed), with the page size."""

    def __init__(self, veusz):
        """Patch the veusz axis module (undone by close())."""
        self.document, self.axis, _ticks, _fmt, self.utils, _qt = veusz
        self.rows = []
        self._ticks = self.axis.Axis._drawTickLabels
        self._text = self.axis.Axis._drawTextWithoutOverlap
        recorder = self

        def ticks(this, phelper, painter, coordticks, sign, outerbounds, tickvals, texttorender):
            """Remember which renderers are tick labels."""
            before = len(texttorender)
            recorder._ticks(this, phelper, painter, coordticks, sign, outerbounds, tickvals,
                            texttorender)
            this._chk_ids = getattr(this, "_chk_ids", set()) | set(
                id(r) for r, _p in texttorender[before:])

        def text(this, painter, texttorender):
            """Replay the overlap test, then let Veusz draw."""
            tester = recorder.utils.RectangleOverlapTester()
            ids = getattr(this, "_chk_ids", set())
            for r, _pen in texttorender:
                rect = r.getTightBounds()
                drawn = not tester.willOverlap(rect)
                if drawn:
                    tester.addRect(rect)
                box = rect.makePolygon().boundingRect()
                recorder.rows.append((this.path, id(r) in ids, drawn,
                                      (box.left(), box.top(), box.right(), box.bottom())))
            this._chk_ids = set()
            recorder._text(this, painter, texttorender)
        self.axis.Axis._drawTickLabels = ticks
        self.axis.Axis._drawTextWithoutOverlap = text

    def close(self):
        """Undo the patches."""
        self.axis.Axis._drawTickLabels = self._ticks
        self.axis.Axis._drawTextWithoutOverlap = self._text


def render_and_record(veusz, path, dpi=72):
    """Render every page of ``path``; [(axis path, tick?, drawn?, box, page px)]."""
    recorder = LabelRecorder(veusz)
    out = []
    try:
        doc = veusz[0].Document()
        doc.load(path, mode="hdf5")
        cmd = veusz[0].CommandInterface(doc)
        with tempfile.TemporaryDirectory() as folder:
            for index, page in enumerate(doc.basewidget.children):
                size = [w._distance_cm(page.settings.get(k).val) / 2.54 * dpi
                        for k in ("width", "height")]
                start = len(recorder.rows)
                cmd.Export(os.path.join(folder, "p%d.png" % index), page=[index], dpi=dpi)
                out += [row + (size,) for row in recorder.rows[start:]]
    finally:
        recorder.close()
    return out


# %% Tests
class IdentityWith0030(unittest.TestCase):
    """Pages without a date axis or data ranges are written as by 0.0.30."""

    @classmethod
    def setUpClass(cls):
        """Load master 0.0.30 from docs/writer_0.0.31 (skip when it is missing)."""
        if not MASTER_0030.is_file():
            raise unittest.SkipTest("docs/writer_0.0.31/vszh5_writer_0.0.30.py missing")
        if hashlib.sha256(MASTER_0030.read_bytes()).hexdigest() != SHA_0030:
            raise AssertionError("the stored 0.0.30 is not the registry master")
        cls.old = load_writer(MASTER_0030, "vszh5_writer_0030")

    def test_versions(self):
        """The writer is 0.0.31 and its header says so."""
        self.assertEqual(w.__version__, "0.0.31")
        self.assertIn("Version        : 0.0.31", (ROOT / "vszh5_writer.py").read_text(encoding="utf-8"))
        self.assertEqual(self.old.__version__, "0.0.30")

    def test_documents_identical_without_dates(self):
        """xy, grid, image, bar, text and Smith pages: every dataset,
        attribute and script line as in 0.0.30."""
        with tempfile.TemporaryDirectory() as folder:
            build_document(self.old, Path(folder) / "old.vszh5", pair=False)
            build_document(w, Path(folder) / "new.vszh5", pair=False)
            old_objects, old_script = document_objects(Path(folder) / "old.vszh5")
            new_objects, new_script = document_objects(Path(folder) / "new.vszh5")
        self.assertEqual(sorted(old_objects), sorted(new_objects))
        for name in old_objects:
            self.assertEqual(old_objects[name][1], new_objects[name][1], name)
        self.assertEqual(old_script, new_script)

    def test_pair_pages_turned(self):
        """The time-pair pages are the ones 0.0.31 changes: vertical labels,
        the MJD axis at 12 significant digits, two-line ISO labels."""
        with tempfile.TemporaryDirectory() as folder:
            build_document(w, Path(folder) / "new.vszh5")
            _objects, script = document_objects(Path(folder) / "new.vszh5")
        text = "\n".join(script)
        self.assertEqual(text.count("Set('x/TickLabels/rotate', '-90')"), 2)
        self.assertIn("Set('x/TickLabels/format', '%.12Vg')", text)
        self.assertIn("Set('x/TickLabels/format', '%s')" % TWO_LINE.replace("\\", "\\\\"), text)


class PortTests(unittest.TestCase):
    """veusz_ticks, veusz_date_ticks and veusz_number_text equal Veusz's own."""

    @classmethod
    def setUpClass(cls):
        """Import Veusz (skip without it)."""
        cls.veusz = veusz_modules()
        if cls.veusz is None:
            raise unittest.SkipTest("Veusz not installed")

    def test_numeric_ticks_and_texts(self):
        """300 random ranges, three tick numbers, both extensions, six formats."""
        _doc, _axis, axisticks, formatting, _u, _q = self.veusz
        rng = random.Random(31)
        for _ in range(300):
            centre = 10 ** rng.uniform(-8, 10) * rng.choice((1, -1))
            low, high = centre, centre + abs(centre) * 10 ** rng.uniform(-9, 1) + 1e-12
            for numticks in (3, 6, 8):
                for extend in (False, True):
                    ref = axisticks.AxisTicks(low, high, numticks, 20, extendmin=extend,
                                              extendmax=extend)
                    ref.getTicks()
                    mine, mn, mx = w.veusz_ticks(low, high, numticks, extend, extend)
                    self.assertEqual(len(mine), len(ref.tickvals))
                    self.assertTrue(np.allclose(mine, ref.tickvals, rtol=0,
                                                atol=abs(high - low) * 1e-9 + 1e-300))
                    for value in ref.tickvals[:2]:
                        for fmt in ("%.12Vg", "%Vg", "%Ve", "%VE", "%g", "%.2f"):
                            self.assertEqual(w.veusz_number_text(float(value), fmt),
                                             formatting.formatNumber(value, fmt))

    def test_date_ticks(self):
        """300 random date ranges from 0.1 s to 30 years, extended or not."""
        _doc, _axis, axisticks, _f, _u, _q = self.veusz
        rng = random.Random(32)
        for _ in range(300):
            low = rng.uniform(5e8, 6e8)
            span = 10 ** rng.uniform(-1, 9)
            for extend in (False, True):
                ref = axisticks.DateTicks(low, low + span, 6, 20, extendmin=extend, extendmax=extend)
                ref.getTicks()
                ticks, fmt, mn, mx = w.veusz_date_ticks(low, low + span, 6, extend, extend)
                self.assertEqual(fmt, ref.autoformat)
                self.assertEqual(len(ticks), len(ref.tickvals))
                self.assertTrue(np.allclose(ticks, ref.tickvals, atol=1e-6))
                self.assertAlmostEqual(mn, ref.minval, places=5)
                self.assertAlmostEqual(mx, ref.maxval, places=5)


class LabelTests(unittest.TestCase):
    """datetime_two_line, widths, margin text and axis_tick_room decisions."""

    def test_two_line(self):
        """The line ends between the date and the time; others unchanged."""
        self.assertEqual(w.datetime_two_line(w.ISO_DATETIME_FORMAT + "-05:00"), TWO_LINE)
        self.assertEqual(w.datetime_two_line(w.ISO_DATETIME_FORMAT), w.ISO_DATETIME_FORMAT_2LINE)
        self.assertEqual(w.datetime_two_line("%VDY-%VDm-%VDd %VDH:%VDM:%VDS"),
                         "%VDY-%VDm-%VDd\\\\%VDH:%VDM:%VDS")
        for fmt in ("%VDH:%VDM", "%VDY-%VDm-%VDd", "", TWO_LINE):
            self.assertEqual(w.datetime_two_line(fmt), fmt)

    def test_widths_and_margin_text(self):
        """Character widths add up; the widest line counts; cm round up."""
        self.assertAlmostEqual(w.tick_text_width_em("2026"), 4 * 0.556)
        # "2026" (4 x 0.556 em) is wider than "T00" (0.611 + 2 x 0.556 em)
        self.assertAlmostEqual(w.tick_text_width_em("2026\\\\T00"), 4 * 0.556)
        self.assertEqual(w.tick_text_lines("a\\\\b"), 2)
        self.assertEqual(w.margin_text(2.6213), "2.65cm")
        self.assertEqual(w.margin_text(2.6), "2.6cm")

    def test_axis_tick_room(self):
        """Dates and long numbers go vertical, short numbers stay level,
        unknown numeric axes are left alone, y axes get their room."""
        t0 = float(w.mjd_to_veusz(np.array([61349.0]))[0])
        dates = w.axis_tick_room("horizontal", "datetime", data_range=(t0, t0 + 3600),
                                 tick_format=w.ISO_DATETIME_FORMAT + "-05:00",
                                 label="Local time", autorange="exact")
        self.assertTrue(dates.vertical)
        self.assertEqual(dates.settings, {"TickLabels/rotate": "-90", "TickLabels/size": "8pt",
                                          "TickLabels/format": TWO_LINE})
        self.assertGreater(dates.room_cm, 2.0)
        hz = w.axis_tick_room("horizontal", data_range=(9999940, 10000060), tick_format="%.12Vg")
        self.assertTrue(hz.vertical)
        level = w.axis_tick_room("horizontal", data_range=(-150, 180), tick_format="%.12Vg",
                                 axis_length_cm=22.0)
        self.assertFalse(level.vertical)
        self.assertEqual(level.settings, {})
        unknown = w.axis_tick_room("horizontal", tick_format="%.12Vg")
        self.assertEqual((unknown.room_cm, unknown.settings), (0.0, {}))
        never = w.axis_tick_room("horizontal", "datetime", tick_format=w.ISO_DATETIME_FORMAT,
                                 vertical=False)
        self.assertFalse(never.vertical)
        y = w.axis_tick_room("vertical", data_range=(-0.25, -0.15), tick_format="%.12Vg",
                             label="Torque")
        self.assertGreater(y.room_cm, 2.0)                 # Veusz's 1.7 cm dropped labels
        self.assertGreater(y.min_length_cm, 3.0)

    def test_broken_axis_fewer_ticks(self):
        """A broken axis with short parts gets fewer major ticks per part,
        and its labels then fit apart."""
        breaks = []
        for start in np.arange(61349.1, 61349.9, 0.1):
            breaks += [float(start), float(start) + 0.05]
        room = w.axis_tick_room("horizontal", data_range=(61349.0, 61350.0), tick_format="%.12Vg",
                                breaks=breaks, axis_length_cm=10.0)
        self.assertTrue(room.vertical)
        self.assertLess(room.settings.get("MajorTicks/number", 6), 6)
        parts = len(breaks) // 2 + 1
        part = 10.0 * (1 - 0.05 * (parts - 1)) / parts
        self.assertGreaterEqual(part / max(room.ticks - 1, 1), room.along_cm * 1.2 - 1e-9)


class PageHelperTests(unittest.TestCase):
    """xy_page, grid_page, style_datetime_axis and time_pair_pages."""

    def test_xy_page_dates(self):
        """A date page: vertical two-line labels, 8 pt, the room below."""
        doc = w.VszDoc()
        w.xy_page(doc, "p", [w.Trace("t", "v")], xlabel="Local time", xmode="datetime",
                  datetime_format=w.ISO_DATETIME_FORMAT + "-05:00", x_autorange="exact")
        text = doc.text()
        self.assertIn("Set('x/TickLabels/rotate', '-90')", text)
        self.assertIn("Set('x/TickLabels/size', '8pt')", text)
        self.assertEqual(text.count("x/TickLabels/format"), 1)
        self.assertIn("Set('bottomMargin', ", text)

    def test_xy_page_numeric_unchanged(self):
        """Without a range a numeric page has no new settings; x_vertical
        False keeps a date axis level."""
        doc = w.VszDoc()
        w.xy_page(doc, "p", [w.Trace("t", "v")])
        self.assertNotIn("rotate", doc.text())
        self.assertNotIn("bottomMargin", doc.text())
        doc = w.VszDoc()
        w.xy_page(doc, "p", [w.Trace("t", "v")], xmode="datetime",
                  datetime_format=w.ISO_DATETIME_FORMAT, x_vertical=False)
        self.assertNotIn("rotate", doc.text())

    def test_grid_page_grows(self):
        """Six date panels on a 15 cm page: the page grows, every panel has
        the same fitted bottom margin."""
        doc = w.VszDoc()
        panels = [w.Panel([w.Trace("t", "v")], "Local time", "v", xmode="datetime",
                          datetime_format=w.ISO_DATETIME_FORMAT, x_autorange="exact")
                  for _ in range(6)]
        w.grid_page(doc, "g", panels, columns=2, height="15cm")
        text = doc.text()
        height = [line for line in text.splitlines() if line.startswith("Set('height'")][0]
        self.assertGreater(w._distance_cm(height.split("'")[3]), 15.0)
        margins = set(line for line in text.splitlines() if line.startswith("Set('bottomMargin'"))
        margins.discard("Set('bottomMargin', '0.2cm')")           # the grid's own margin
        self.assertEqual(len(margins), 1)
        self.assertGreater(w._distance_cm(sorted(margins)[0].split("'")[3]), 2.0)

    def test_style_datetime_axis(self):
        """-90, two lines, 8 pt and the graph's bottom margin."""
        doc = w.VszDoc()
        doc.add("page", "p")
        doc.to("p")
        doc.add("graph", "g")
        doc.to("g")
        doc.add("axis", "x")
        axis = w.local_time_axis(np.array([1.7e9, 1.7e9 + 3600.0]), zone="-05:00")
        w.style_datetime_axis(doc, "x", axis)
        text = doc.text()
        self.assertIn("Set('x/TickLabels/rotate', '-90')", text)
        self.assertIn("Set('x/TickLabels/format', '%s')" % TWO_LINE.replace("\\", "\\\\"), text)
        self.assertIn("Set('bottomMargin', ", text)
        self.assertEqual(axis.datetime_format_2line, TWO_LINE)


class NativeFitTests(unittest.TestCase):
    """_autoplot_native._fit_page through NativeDocument.save()."""

    def test_fit_and_save_twice(self):
        """Dates and Hz vertical, short numbers level, the y margin for long
        y labels, pages grown; saving twice writes the same script."""
        doc = date_project()
        with tempfile.TemporaryDirectory() as folder:
            first = doc.save(os.path.join(folder, "a.vszh5"))
            _o1, script1 = document_objects(Path(first))
            second = doc.save(os.path.join(folder, "b.vszh5"))
            _o2, script2 = document_objects(Path(second))
        self.assertEqual(script1, script2)
        grid = doc.Root.p.grid
        for graph in [c for c in grid.children if c.kind == "graph"]:
            self.assertEqual(graph.x.settings["TickLabels/rotate"], "-90")
            self.assertEqual(graph.x.settings["TickLabels/format"], TWO_LINE)
            self.assertGreater(w._distance_cm(graph.settings["bottomMargin"]), 2.0)
        self.assertEqual(doc.Root.hz.g_hz.x.settings["TickLabels/rotate"], "-90")
        self.assertNotIn("TickLabels/rotate", doc.Root.short.g_short.x.settings)
        self.assertGreater(w._distance_cm(doc.Root.short.g_short.settings["leftMargin"]), 1.7)

    def test_restore(self):
        """finalize() twice leaves the same settings (the fit undoes itself first)."""
        doc = date_project()
        doc.finalize()
        before = [(wd.path, dict(wd.settings)) for wd in doc.Root.WalkWidgets()]
        doc.finalize()
        after = [(wd.path, dict(wd.settings)) for wd in doc.Root.WalkWidgets()]
        self.assertEqual(before, after)


class VeuszRenderTests(unittest.TestCase):
    """Veusz draws every tick label, inside the page (skipped without Veusz)."""

    def test_every_label_drawn(self):
        """Writer pages and the native project: no tick label dropped,
        none outside the page."""
        veusz = veusz_modules()
        if veusz is None:
            self.skipTest("Veusz not installed")
        t = np.arange(600.0)
        mjd = 61349.5 + t * 6.0 / 86400.0
        axis = w.local_time_axis(w.mjd_to_epoch(mjd), zone="-05:00")
        doc = w.VszDoc()
        w.xy_page(doc, "xy", [w.Trace("local", "v")], xlabel=axis.label, ylabel="v",
                  xmode="datetime", datetime_format=axis.datetime_format, x_autorange="exact")
        w.grid_page(doc, "grid", [w.Panel([w.Trace("local", "v")], axis.label, "v", xmode="datetime",
                                          datetime_format=axis.datetime_format,
                                          x_autorange="exact", y_range=(-1, 1))
                                  for _ in range(5)], columns=2)
        w.time_pair_pages(doc, "pair", [w.Trace("mjd", "v")], "mjd", "local", axis, ylabel="v")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pages.vszh5")
            with w.VeuszWriter(path) as out:
                out.dataset_1d("v", np.sin(t / 30.0))
                out.dataset_1d("mjd", mjd)
                out.dataset_date("local", axis.veusz_seconds)
                out.finish(doc.text())
            native = date_project().save(os.path.join(folder, "native.vszh5"))
            rows = render_and_record(veusz, path) + render_and_record(veusz, native)
        ticks = [row for row in rows if row[1]]
        self.assertGreater(len(ticks), 50)
        dropped = [row[0] for row in ticks if not row[2]]
        self.assertEqual(dropped, [])
        for _path, _tick, drawn, box, size in rows:
            if drawn:
                self.assertGreaterEqual(box[0], -0.5)
                self.assertGreaterEqual(box[1], -0.5)
                self.assertLessEqual(box[2], size[0] + 0.5)
                self.assertLessEqual(box[3], size[1] + 0.5)


# %% Main
if __name__ == "__main__":
    unittest.main()
