# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_page_kinds.py

Per file / Combined axis / Overlay pages in every plotter (2026-11-17
request): the combined-axis pages hold the same quantity of every file
joined end to end and sorted on x (stable), look like a file's own pages,
come just above the overlays in the project and the Plots tab, are laid out
by the gap rules and follow the RAM limit (offload).

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
#              page_kinds / --pages, concatenate_on_x, overlapping_ranges,
#              move_pages_before, the combined pages of FITS (tagged and
#              untagged), Franks, CSV/TSV, ATR, R&S and Touchstone (values
#              exactly, page order, gap layout), offloaded = in RAM, the
#              PageTypesBox and the FITS window's rebuild on a changed box;
#              the gap layout of joined traces (cut only where no file has
#              data, hand-worked: a hole between files, files back to back, a
#              coarse sweep with a fine one inside it, a hole another file
#              fills, NaN x, 1-s and 60-s cadences on an MJD axis), the file
#              numbers offloaded; both plugins' "pages" / "combined_only".
# %%%%% Function Descriptions
        write_fits: A small OnePpsDeltas-style FITS file: ``rows`` DMJD samples ``step_min`` minutes apart from ``start`` (MJD).
        joined: The per-file pieces joined in file order, stable-sorted on x (the reference).
        page_names: Page names of a NativeDocument, in order.
        data: A dataset's values as float.
        gap_graph: A page with one graph (x axis labelled ``label``) drawing one trace.
        PageKindTests (class): page_kinds, describe_page_kinds and --pages.
        JoinTests (class): concatenate_on_x, overlapping_ranges and move_pages_before.
        CombinedGapTests (class): combined_gap_positions and the project layout of joined traces, hand-worked.
        FitsFranksTests (class): FITS and Franks combined-axis pages.
        CsvTests (class): CSV/TSV CombinedPlot.
        InstrumentTests (class): ATR, R&S and Touchstone combined pages.
        GuiTests (class): PageTypesBox and the FITS window.
        PluginTests (class): Both plugins: the "pages" field and the old "combined_only" field.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        EXAMPLES: the repository's example data folder
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Release notes: RELEASE_suite-2026-11-09.md section 19.
=============================================================================
"""
# %% Imports
import argparse
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
EXAMPLES = ROOT / "Example_Data_Files"

import _autoplot_native as native                                # noqa: E402
from _autoplot_native import (NativeDocument, TimeLayout, page_kinds,   # noqa: E402
                              concatenate_on_x, overlapping_ranges, move_pages_before)


# %% Helpers
def write_fits(path, start, rows=60, tagged=True, step_min=1.0):
    """A small OnePpsDeltas-style FITS file: ``rows`` DMJD samples ``step_min``
    minutes apart from ``start`` (MJD), DELTAT = start + row, CHANNELA/B tags
    alternating A/B (when ``tagged``)."""
    from astropy.io import fits
    x = start + np.arange(rows) * step_min / 1440.0
    cols = [fits.Column(name="DMJD", format="D", array=x),
            fits.Column(name="DELTAT", format="D", unit="ns", array=start + np.arange(rows))]
    if tagged:
        cols += [fits.Column(name="CHANNELA", format="8A",
                             array=["A" if i % 2 else "C" for i in range(rows)]),
                 fits.Column(name="CHANNELB", format="8A", array=["B"] * rows)]
    table = fits.BinTableHDU.from_columns(cols, name="OnePpsDeltas")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(str(path))
    return path


def joined(xs, ys):
    """The per-file pieces joined in file order, stable-sorted on x (the
    reference the combined traces must equal exactly)."""
    x = np.concatenate([np.asarray(a, float) for a in xs])
    y = np.concatenate([np.asarray(b, float) for b in ys])
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def page_names(doc):
    """Page names of a NativeDocument, in order."""
    return [p.name for p in doc.Root.children if p.kind == "page"]


def data(doc, name):
    """A dataset's values as float."""
    return np.asarray(doc.GetData(name)[0], dtype=float)


# %% Page kinds
class PageKindTests(unittest.TestCase):
    """page_kinds, describe_page_kinds and --pages."""

    def test_words(self):
        """Every spelling; PAGE_KINDS order; unknown words refused."""
        self.assertEqual(page_kinds(None), ("per_file", "combined", "overlay"))
        self.assertEqual(page_kinds("all"), ("per_file", "combined", "overlay"))
        self.assertEqual(page_kinds("none"), ())
        self.assertEqual(page_kinds("overlay, per-file"), ("per_file", "overlay"))
        self.assertEqual(page_kinds("combined+overlay"), ("combined", "overlay"))
        self.assertEqual(page_kinds(["Combined_Axis"]), ("combined",))
        self.assertEqual(page_kinds("files"), ("per_file",))
        with self.assertRaises(ValueError):
            page_kinds("per_file,stacked")
        self.assertEqual(native.describe_page_kinds(("combined",)), "Pages: Combined axis")
        self.assertEqual(native.describe_page_kinds(()), "Pages: none")

    def test_flag(self):
        """--pages: default all; a wrong word stops argparse."""
        parser = argparse.ArgumentParser()
        native.add_page_arguments(parser)
        self.assertEqual(native.page_kinds_from_args(parser.parse_args([])),
                         ("per_file", "combined", "overlay"))
        self.assertEqual(native.page_kinds_from_args(parser.parse_args(["--pages", "combined"])),
                         ("combined",))
        parser.error = lambda message: (_ for _ in ()).throw(ValueError(message))
        with self.assertRaises(ValueError):
            parser.parse_args(["--pages", "stacked"])


# %% Joining
class JoinTests(unittest.TestCase):
    """concatenate_on_x, overlapping_ranges and move_pages_before."""

    def test_stable_exact(self):
        """Equal x keep their file order, NaN x go last, values untouched."""
        x, (a, b) = concatenate_on_x(
            [np.array([3.0, 1.0, np.nan]), np.array([1.0, 2.0])],
            [[np.array([30.0, 10.0, 99.0]), np.array([0.1, 0.2, 0.3])],
             [np.array([11.0, 20.0]), np.array([2.0 ** 53 + 2, 5e-324])]])
        self.assertTrue(np.array_equal(x, [1.0, 1.0, 2.0, 3.0, np.nan], equal_nan=True))
        self.assertEqual(a.tolist(), [10.0, 11.0, 20.0, 30.0, 99.0])
        self.assertEqual(b.tolist(), [0.2, 2.0 ** 53 + 2, 5e-324, 0.1, 0.3])
        x, y = concatenate_on_x([np.array([2.0]), np.array([1.0])],
                                [np.array([[1, 2]]), np.array([[3, 4]])])
        self.assertEqual(y.tolist(), [[3, 4], [1, 2]])                  # 2-D cells
        x, y = concatenate_on_x([np.array(["b", "a"])], [np.array([1.0, 2.0])])
        self.assertEqual(list(x), ["b", "a"])                            # text x: file order

    def test_overlap(self):
        """Overlapping x ranges are counted (the combined trace interleaves)."""
        self.assertEqual(overlapping_ranges([np.arange(5.0), np.arange(5.0, 9.0)]), 0)
        self.assertEqual(overlapping_ranges([np.arange(5.0), np.arange(3.0, 9.0),
                                             np.array([np.nan, 1.0])]), 2)

    def test_move_pages(self):
        """Pages moved as a block just before the anchor; None leaves them."""
        doc = NativeDocument("order")
        pages = [doc.Root.Add("page", name=n) for n in ("O1", "O2", "F1", "C1", "C2")]
        move_pages_before(doc, pages[3:], pages[0])
        self.assertEqual(page_names(doc), ["C1", "C2", "O1", "O2", "F1"])
        move_pages_before(doc, pages[3:], None)
        self.assertEqual(page_names(doc), ["C1", "C2", "O1", "O2", "F1"])


# %% Gaps of a joined trace
def gap_graph(doc, x_name, y_name, label="Frequency (GHz)"):
    """A page with one graph (x axis labelled ``label``) drawing one trace;
    returns (graph, x axis, xy)."""
    page = doc.Root.Add("page", name="p%d" % len(doc.Root.children))
    graph = page.Add("graph", name="g", autoadd=False)
    axis = graph.Add("axis", name="x")
    axis.settings["label"] = label
    graph.Add("axis", name="y").settings["direction"] = "vertical"
    xy = graph.Add("xy", name="t", xData=x_name, yData=y_name)
    return graph, axis, xy


class CombinedGapTests(unittest.TestCase):
    """combined_gap_positions: a joined trace is cut only where none of its
    files has data (each file judged by its own gap rule), hand-worked."""

    @staticmethod
    def cuts(pieces, k=10.0):
        """Gap rows of the pieces joined (K x median step of each file)."""
        import vszh5_writer
        x, _y, files = concatenate_on_x(pieces, [np.zeros(np.asarray(p).size) for p in pieces],
                                        with_sources=True)
        positions, lines, parts = native.combined_gap_positions(
            x, files, lambda piece: vszh5_writer.gap_threshold(piece, k, 0.0))
        return x, positions.tolist(), lines, parts

    def test_hand_worked(self):
        """Hole between files: cut; files back to back: no cut; a coarse sweep
        joined with a fine one inside it: no cut (the old single-median rule
        cut every coarse step); a hole of one file that another file fills:
        no cut; the same hole not filled: cut; NaN x: ignored."""
        _x, cut, lines, _p = self.cuts([np.arange(60.0), 10000 + np.arange(60.0)])
        self.assertEqual(cut, [60])                       # 59 -> 10000 > 10 x 1
        self.assertEqual(lines, [10.0, 10.0])
        self.assertEqual(self.cuts([np.arange(60.0), 60 + np.arange(60.0)])[1], [])
        coarse, fine = np.arange(0.0, 1001.0, 100.0), 400 + 0.1 * np.arange(101)
        x, cut, lines, parts = self.cuts([coarse, fine])
        self.assertEqual(cut, [])
        self.assertEqual(lines, [1000.0, 10.0 * float(np.median(np.diff(fine)))])
        import vszh5_writer
        self.assertTrue(vszh5_writer.gap_positions(x, vszh5_writer.gap_threshold(x, 10.0, 0.0)).size)
        holed = np.concatenate((np.arange(40.0), 60 + np.arange(40.0)))      # 40..59 missing
        self.assertEqual(self.cuts([holed, 30 + np.arange(41.0)])[1], [])   # 30..70 fills it
        x, cut, _l, _p = self.cuts([holed, np.arange(21.0)])               # 0..20 does not
        self.assertEqual(cut, [61])
        self.assertEqual((x[60], x[61]), (39.0, 60.0))
        x, cut, _l, _p = self.cuts([np.array([0.0, 1.0, 2.0, np.nan]),
                                    np.array([100.0, 101.0, 102.0])])
        self.assertEqual(cut, [3])
        self.assertTrue(np.isnan(x[-1]))

    def test_one_file_is_the_plain_rule(self):
        """With one file the result is gap_positions with its own threshold."""
        import vszh5_writer
        x = np.array([0.0, 1, 2, 3, 50, 51, 52, 200, 201])
        expect = vszh5_writer.gap_positions(x, vszh5_writer.gap_threshold(x, 10.0, 0.0))
        self.assertEqual(expect.tolist(), [4, 7])
        self.assertEqual(self.cuts([x])[1], [4, 7])

    def test_options(self):
        """Line breaks off: no rows (the thresholds still come back); unsorted
        x: none; sources of another length: the plain values."""
        positions, lines, _p = native.combined_gap_positions(
            np.array([0.0, 1, 2, 100, 101]), np.array([0, 0, 0, 1, 1]), lambda p: 10.0,
            break_lines=False)
        self.assertEqual((positions.tolist(), lines), ([], [10.0, 10.0]))
        positions, _l, _p = native.combined_gap_positions(
            np.array([5.0, 1, 100]), np.array([0, 0, 1]), lambda p: 1.0)
        self.assertEqual(positions.tolist(), [])
        positions, lines, parts = native.combined_gap_positions(
            np.array([0.0, 100]), np.array([0]), lambda p: 1.0)
        self.assertEqual((positions.tolist(), lines, len(parts)), ([], [], 1))

    def test_project_layout(self):
        """finalize: a coarse sweep with a fine one inside it gives no line or
        axis breaks; two bands far apart are cut and broken (as an overlay of
        the same files would be); the file numbers move to the offload file
        with the datasets and the layout is the same."""
        import vszh5_writer
        coarse, fine = np.arange(0.0, 1001.0, 100.0), 400 + 0.1 * np.arange(101)
        doc = NativeDocument("join")
        x, y, files = concatenate_on_x([coarse, fine], [coarse * 0, fine * 0 + 1],
                                       with_sources=True)
        self.assertEqual(files.dtype, np.uint8)
        doc.SetData("cx", x)
        doc.SetData("cy", y)
        native.set_combined_sources(doc, ["cx", "not_there"], files)
        self.assertEqual(sorted(doc.combined_of), ["cx"])
        _g, axis, xy = gap_graph(doc, "cx", "cy")
        doc.finalize()
        self.assertEqual((xy.settings["xData"], xy.settings["yData"]), ("cx", "cy"))
        self.assertEqual(axis.kind, "axis")
        far = [np.arange(1.0, 2.001, 0.01), np.arange(5000.0, 5001.001, 0.01)]
        x2, y2, files2 = concatenate_on_x(far, [p * 0 for p in far], with_sources=True)
        doc.SetData("fx", x2)
        doc.SetData("fy", y2)
        native.set_combined_sources(doc, ["fx"], files2)
        _g, axis2, xy2 = gap_graph(doc, "fx", "fy")
        doc.finalize()
        self.assertEqual(int(np.isnan(data(doc, xy2.settings["yData"])).sum()), 1)
        self.assertEqual(axis2.kind, "axis-broken")
        rule = TimeLayout().gap_rule(geometric=True)
        self.assertEqual(axis2.settings["breakPoints"],
                         vszh5_writer.break_points(vszh5_writer.rule_axis_break_pairs(far, rule)))
        first = (list(axis2.settings["breakPoints"]), data(doc, xy2.settings["yData"]).tolist())
        # the Plots tab (0.3.8): draws and shades what the project laid out
        try:
            import _autoplot_pyqtgraph as pgt
        except Exception:                          # pyqtgraph not installed
            pgt = None
        if pgt is not None:
            graphs = [g for page in pgt.document_pages(doc) for g in page["graphs"]]
            self.assertEqual(len(graphs), 2)
            for graph, cuts in zip(graphs, (0, 1)):
                self.assertTrue(graph["x_laid_out"])
                self.assertFalse(pgt.graph_recut(graph))
                x, y, is_time = pgt.trace_arrays(doc, graph["traces"][0], graph["x_kind"], False,
                                                 recut=pgt.graph_recut(graph))
                self.assertEqual(int(np.isnan(y).sum()), cuts)
                regions = pgt.graph_break_regions(doc, graph, [x[np.isfinite(x)]], is_time, False)
                self.assertEqual(regions, graph["x_breaks"])        # numeric x: same units
                self.assertEqual(len(regions), cuts)
            # cut again from the joined values (before 0.3.8) the coarse sweep
            # broke at its own steps
            _x, y_old, _t = pgt.trace_arrays(doc, graphs[0]["traces"][0], graphs[0]["x_kind"],
                                             False, recut=True)
            self.assertGreater(int(np.isnan(y_old).sum()), 0)
            self.assertEqual(pgt.graph_break_regions(doc, graphs[0], [], False, False), [])
        native.OFFLOAD_MIN_BYTES, keep = 0, native.OFFLOAD_MIN_BYTES
        try:
            moved = doc.offload()
        finally:
            native.OFFLOAD_MIN_BYTES = keep
        self.assertGreater(moved, 0)
        self.assertIsInstance(doc.combined_of["fx"], native.OffloadedValue)
        self.assertTrue(np.array_equal(native.combined_sources(doc, "fx", x2.size), files2))
        self.assertIsNone(native.combined_sources(doc, "fx", x2.size + 1))
        doc.time_layout = TimeLayout(k=10.0)          # a new key: laid out again from disk
        doc.time_layout = TimeLayout()
        for widget in doc.Root.WalkWidgets():
            widget.gap_layout_key = None
        doc.finalize()
        self.assertEqual(list(axis2.settings["breakPoints"]), first[0])
        self.assertTrue(np.array_equal(data(doc, xy2.settings["yData"]), first[1],
                                       equal_nan=True))
        doc.release()
        self.assertEqual(doc.combined_of, {})

    def test_time_axis_cadences(self):
        """MJD time axis: a 1-s file inside a 60-s file's span with a 5-minute
        hole the 60-s file fills: no cut; a day after both: cut and broken."""
        one_s = 61000.0 + np.concatenate((np.arange(600.0), 900 + np.arange(600.0))) / 86400.0
        sixty = 61000.0 + np.arange(0.0, 1800.0, 60.0) / 86400.0
        later = 61002.0 + np.arange(0.0, 600.0, 60.0) / 86400.0
        doc = NativeDocument("cadence")
        doc.time_layout = TimeLayout(clocks="mjd")
        x, y, files = concatenate_on_x([one_s, sixty, later],
                                       [one_s * 0, sixty * 0, later * 0], with_sources=True)
        doc.SetData("mx", x)
        doc.SetData("my", y)
        native.set_combined_sources(doc, ["mx"], files)
        _g, axis, xy = gap_graph(doc, "mx", "my", label="MJD")
        doc.finalize()
        plotted = data(doc, xy.settings["yData"])
        self.assertEqual(int(np.isnan(plotted).sum()), 1)        # only before the later file
        self.assertEqual(axis.kind, "axis-broken")
        seconds = (data(doc, xy.settings["xData"]) - 61000.0) * 86400.0
        at = int(np.flatnonzero(np.isnan(plotted))[0])
        self.assertLess(seconds[at - 1], 1800.0)
        self.assertGreater(seconds[at + 1], 86400.0)


# %% FITS and Franks
class FitsFranksTests(unittest.TestCase):
    """FITS and Franks combined-axis pages."""

    def setUp(self):
        """Three FITS files: two 60-minute runs a week apart (tagged) and
        one untagged; Franks files with a 2-day step."""
        try:
            import FITS_AutoPlot  # noqa: F401
        except Exception as exc:
            self.skipTest("FITS AutoPlot not importable: %s" % exc)
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        """Remove the folder."""
        self.tmp.cleanup()

    def fits_project(self, layout=None, kinds=("per_file", "combined", "overlay"), offload=False):
        """Read the files and build the FITS project as the window does."""
        import FITS_AutoPlot as fitstool
        from _autoplot_common import MemoryAwareCache, MemoryMonitorConfig
        paths = [write_fits(self.folder / "b.fits", 61007.0),          # later file first
                 write_fits(self.folder / "a.fits", 61000.0)]
        cache = MemoryAwareCache(MemoryMonitorConfig())
        results = {str(p): fitstool.FITSProcessor("astropy", cache).read(str(p)) for p in paths}
        ordered = sorted(results.items(), key=fitstool._result_order)
        doc = NativeDocument("fits")
        if offload:
            doc.ram_limit_percent = 1e-9
            doc._ram_checked = 0.0
        doc.time_layout = layout or TimeLayout()
        records = []
        for path, item in ordered:
            fitstool.push_to_veusz(doc, path, item, "astropy", plot_individual="per_file" in kinds)
            records.append(dict(item, base=fitstool.safe_dsname(item["base_name"])))
        if "combined" in kinds:
            fitstool.push_combined_pages(doc, ordered)
        if "overlay" in kinds:
            fitstool.build_unit_overlay_pages(doc, records)
        cache.cleanup()
        return doc, ordered

    def test_fits_values_order_gaps(self):
        """Combined_ONEPPSDELTAS: every tag's DELTAT of both files joined and
        sorted on DMJD, exactly; after the per-file pages, before the overlay;
        the week between the files cuts the line and breaks the axis."""
        doc, ordered = self.fits_project(TimeLayout(clocks="all"))
        doc.finalize()
        names = page_names(doc)
        combined = [i for i, n in enumerate(names) if n.startswith("Combined")]
        overlay = [i for i, n in enumerate(names) if n.startswith("Overlay")]
        self.assertTrue(combined and overlay)
        self.assertEqual(combined[-1] + 1, overlay[0])
        self.assertIn("Combined_ONEPPSDELTAS", names)
        self.assertIn("Combined_ONEPPSDELTAS_utc", names)          # clock copies follow
        self.assertFalse([n for n in doc.datasets if n.startswith("Combined__") and
                          n.endswith("__raw")])                     # stored once, sorted
        page = next(p for p in doc.Root.children if p.name == "Combined_ONEPPSDELTAS")
        traces = [c for c in page.WalkWidgets() if c.kind == "xy"]
        self.assertTrue(traces)
        for xy in traces:
            x_src, y_src = xy.time_source
            rest_x, rest_y = x_src.split("__", 1)[1], y_src.split("__", 1)[1]
            bases = [d["base_name"] for _p, d in ordered]
            ex, ey = joined([data(doc, "%s__%s" % (b, rest_x)) for b in bases],
                            [data(doc, "%s__%s" % (b, rest_y)) for b in bases])
            self.assertTrue(np.array_equal(data(doc, x_src), ex))
            self.assertTrue(np.array_equal(data(doc, y_src), ey))
            pieces = [data(doc, "%s__%s" % (b, rest_x)) for b in bases]
            _x, files = joined(pieces, [np.full(p.size, i) for i, p in enumerate(pieces)])
            self.assertEqual(native.combined_sources(doc, x_src, ex.size).tolist(),
                             files.astype(int).tolist())                # file of every row
            y_plot = data(doc, xy.settings["yData"])
            self.assertEqual(int(np.isnan(y_plot).sum()), 1)          # the week: one cut
            axis = next(c for c in xy.parent.children if c.name == "x")
            self.assertEqual(axis.kind, "axis-broken")
            self.assertEqual(len(axis.settings["breakPoints"]), 2)

    def test_fits_kinds_and_groups(self):
        """Only the kinds asked for; an HDU of another layout is its own group."""
        import FITS_AutoPlot as fitstool
        doc, _ordered = self.fits_project(kinds=("combined",))
        self.assertEqual(page_names(doc), ["Combined_ONEPPSDELTAS"])
        a = {"columns": {("H", "DMJD"): np.array([1.0, 2.0]), ("H", "V"): np.array([1, 2])},
             "units": {}, "sort_key": ("H", "DMJD"), "base_name": "a", "header": []}
        b = {"columns": {("H", "DMJD"): np.array([1.5]), ("H", "V"): np.array([7.5])},
             "units": {}, "sort_key": ("H", "DMJD"), "base_name": "b", "header": []}
        c = {"columns": {("H", "DMJD"): np.array([9.0]), ("H", "W"): np.array([3.0])},
             "units": {}, "sort_key": ("H", "DMJD"), "base_name": "c", "header": []}
        groups = fitstool.combined_fits_data([("a", a), ("b", b), ("c", c)])
        self.assertEqual([g["base_name"] for g in groups], ["Combined", "Combined_2"])
        self.assertEqual(groups[0]["columns"][("H", "V")].tolist(), [1.0, 2.0, 7.5])
        self.assertEqual(groups[0]["columns"][("H", "DMJD")].tolist(), [1.0, 2.0, 1.5])
        self.assertEqual(groups[0]["combined_files"], ["a", "b"])
        self.assertEqual(groups[1]["combined_files"], ["c"])
        self.assertIn("Note: the time ranges of 1 file(s) overlap", "\n".join(groups[0]["header"]))

    def test_fits_offloaded_same(self):
        """Every dataset offloaded: the saved project equals the in-RAM one."""
        from test_offload import EveryDataset, hdf_objects
        ram, _o = self.fits_project(TimeLayout(clocks="all"))
        ram.ram_limit_percent = 0.0
        first = ram.save(str(self.folder / "ram.vszh5"))
        for name in ("a.fits", "b.fits"):
            os.remove(str(self.folder / name))
        with EveryDataset():
            disk, _o = self.fits_project(TimeLayout(clocks="all"), offload=True)
            second = disk.save(str(self.folder / "disk.vszh5"))
        self.assertEqual(disk.offloaded()[1], 0)
        self.assertGreater(disk.offloaded()[0], 0)
        self.assertEqual(hdf_objects(first), hdf_objects(second))
        ram.release()
        disk.release()

    def test_franks(self):
        """Franks 'Combined': both files joined on MJD, exactly; before the overlays."""
        import Franks_AutoPlot as frankstool
        from _autoplot_common import MemoryAwareCache, MemoryMonitorConfig
        paths = []
        for n, start in ((2, 61002.0), (1, 61000.0)):
            path = self.folder / ("time_gbt_%d.dat" % n)
            rows = "".join("%.10f %d %d\n" % (start + k / 1440.0, k + 10 * n, -k) for k in range(30))
            path.write_text("MJD A B\n================\n" + rows, encoding="utf-8")
            paths.append(path)
        cache = MemoryAwareCache(MemoryMonitorConfig())
        ordered = [(str(p), frankstool.parse_franks_file(str(p), cache)) for p in sorted(paths)]
        doc = NativeDocument("franks")
        records = []
        for _p, item in ordered:
            frankstool.push_franks_to_veusz(doc, item)
            records.append(dict(item, base=item["base_name"]))
        frankstool.push_combined_franks_pages(doc, ordered)
        frankstool.build_unit_overlay_pages_franks(doc, records)
        doc.finalize()
        names = page_names(doc)
        self.assertEqual(names.index("Combined") + 1, names.index("Overlay_A"))
        bases = [d["base_name"] for _p, d in ordered]
        ex, ey = joined([data(doc, "%s__MJD__sorted" % b) for b in bases],
                        [data(doc, "%s__A__sorted" % b) for b in bases])
        self.assertTrue(np.array_equal(data(doc, "Combined__A__sorted"), ey))
        self.assertTrue(np.array_equal(data(doc, "Combined__MJD__sorted"), ex))
        pieces = [data(doc, "%s__MJD__sorted" % b) for b in bases]
        _x, files = joined(pieces, [np.full(p.size, i) for i, p in enumerate(pieces)])
        self.assertEqual(native.combined_sources(doc, "Combined__MJD__sorted", ex.size).tolist(),
                         files.astype(int).tolist())
        cache.cleanup()


# %% CSV/TSV
class CsvTests(unittest.TestCase):
    """CSV/TSV CombinedPlot."""

    def setUp(self):
        """Two CSV files (dates, out of order) with a 2-day step between them."""
        try:
            import CSV_TSV_AutoPlot  # noqa: F401
        except Exception as exc:
            self.skipTest("CSV/TSV AutoPlot not importable: %s" % exc)
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        start = np.datetime64("2026-11-10T12:00:00")
        self.paths = []
        for name, first in (("late.csv", 3000), ("early.csv", 0)):
            path = self.folder / name
            lines = ["Time,Index,A"] + ["%sZ,%d,%r" % (start + np.timedelta64(first + k, "m"),
                                                      first + k, 0.1 * (first + k))
                                        for k in range(40)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.paths.append(str(path))

    def tearDown(self):
        """Remove the folder."""
        self.tmp.cleanup()

    def build(self, kinds, x="Time"):
        """The CSV project of both files with ``kinds``."""
        import CSV_TSV_AutoPlot as csvtool
        files, _messages = csvtool.load_files(self.paths, csvtool.CSVProcessingConfig())
        plotter = csvtool.VeuszPlotter("t")
        plotter.create_overlay_and_individual_plots(files, x, ["A", "Index"], kinds=kinds)
        return plotter.doc

    def test_order_values_gaps(self):
        """CombinedPlot first (just above OverlayPlot); values exact; the 2-day
        step: one cut and one axis break; the kinds asked for only."""
        doc = self.build(None)
        doc.finalize()
        names = page_names(doc)
        self.assertEqual(names, ["CombinedPlot", "OverlayPlot", "Page_late", "Page_early"])
        ex, ey = joined([data(doc, "late_Time"), data(doc, "early_Time")],
                        [data(doc, "late_A"), data(doc, "early_A")])
        self.assertTrue(np.array_equal(data(doc, "Combined_Time"), ex))
        self.assertTrue(np.array_equal(data(doc, "Combined_A"), ey))
        page = doc.Root.children[0]
        xy = next(c for c in page.WalkWidgets() if c.kind == "xy" and "Combined_A" in
                  c.settings["yData"])
        self.assertEqual(int(np.isnan(data(doc, xy.settings["yData"])).sum()), 1)
        axis = next(c for c in xy.parent.children if c.name == "x")
        self.assertEqual(axis.kind, "axis-broken")
        self.assertEqual(page_names(self.build(("combined",))), ["CombinedPlot"])
        self.assertEqual(page_names(self.build(("per_file", "overlay"))),
                         ["OverlayPlot", "Page_late", "Page_early"])
        self.assertEqual(page_names(self.build(("per_file", "combined"))),
                         ["CombinedPlot", "Page_late", "Page_early"])

    def test_numeric_x(self):
        """A numeric X column: joined and sorted on it."""
        doc = self.build(("combined",), x="Index")
        self.assertEqual(data(doc, "Combined_Index").tolist(),
                         sorted(data(doc, "late_Index").tolist() + data(doc, "early_Index").tolist()))


# %% ATR, R&S, Touchstone
class InstrumentTests(unittest.TestCase):
    """ATR, R&S and Touchstone combined pages."""

    def test_atr(self):
        """Combined_mag / _phase / _polar_*: before the overlays; each frequency's
        pattern joined and sorted on azimuth, exactly."""
        try:
            import ATR_AutoPlot as atr
        except Exception as exc:
            self.skipTest("ATR AutoPlot not importable: %s" % exc)
        paths = sorted((EXAMPLES / "atr").glob("Band_1_CoPol_P1_El45_2p7GHz_ccw*.ATR"))
        if len(paths) < 2:
            self.skipTest("no ATR example files")
        patterns = [atr.read_atr_file(str(p)) for p in paths]
        plotter = atr.build_project(patterns)
        names = page_names(plotter.doc)
        self.assertEqual(names[:4], ["Combined_mag", "Combined_phase", "Combined_polar_mag",
                                     "Combined_polar_phase"])
        self.assertTrue(names[4].startswith("Overlay"))
        by_key = {}
        for pattern in patterns:                          # (MHz, occurrence in the file)
            seen = {}
            for k, mhz in enumerate(pattern.frequencies_mhz):
                seen[float(mhz)] = seen.get(float(mhz), 0) + 1
                by_key.setdefault((float(mhz), seen[float(mhz)]), []).append(
                    (pattern.azimuth_deg, pattern.magnitude_db[k], pattern.phase_deg[k]))
        for page_index, column in ((0, 1), (1, 2)):
            page = plotter.doc.Root.children[page_index]
            traces = [c for c in page.WalkWidgets() if c.kind == "xy"]
            self.assertEqual(len(traces), len(by_key))
            for xy, key in zip(traces, by_key):
                parts = by_key[key]
                ex, ey = joined([p[0] for p in parts], [p[column] for p in parts])
                self.assertEqual(len(parts), len(paths))
                self.assertTrue(np.array_equal(data(plotter.doc, xy.settings["xData"]), ex))
                self.assertTrue(np.array_equal(data(plotter.doc, xy.settings["yData"]), ey,
                                               equal_nan=True))
        none = atr.build_project(patterns, atr.PlotOptions(pages=("per_file",)))
        self.assertFalse([n for n in page_names(none.doc) if n.startswith(("Combined", "Overlay"))])

    def test_rs(self):
        """Combined average: every file's average joined and sorted on x, before
        AllImported; sections only in the All plots mode."""
        try:
            import RAndS_FSW_ASCII_Plotter as rs
        except Exception as exc:
            self.skipTest("R&S plotter not importable: %s" % exc)
        paths = sorted((EXAMPLES / "sft" / "R&S_DataFiles" / "Processed").glob("*.sft"))[:4]
        files, _errors = rs.read_files([str(p) for p in paths], 1)
        if len(files) < 2:
            self.skipTest("no R&S example files")
        for mode in (0, 1):
            plotter = rs.RnSPlotter(rs.PlotOptions(mode=mode))
            stems = [plotter.add_file(sft) for sft in files]
            plotter.finish()
            doc = plotter.doc
            names = page_names(doc)
            combined = [i for i, n in enumerate(names) if n.startswith("Combined")]
            self.assertEqual(combined[-1] + 1, names.index("AllImported"))
            self.assertEqual(any("Section" in names[i] for i in combined), mode == 0)
            first = files[0]
            group = [(s, f) for s, f in zip(stems, files)
                     if f.x_kind == first.x_kind and f.y_unit == first.y_unit]
            axis = "time" if first.x_kind == "time" else "freq"
            suffix = "" if first.x_kind == "frequency" else "_" + first.x_kind
            ex, ey = joined([data(doc, "%s_%s" % (s, axis)) for s, _f in group],
                            [data(doc, "%s_avg_dB" % s) for s, _f in group])
            self.assertTrue(np.array_equal(data(doc, "Combined%s_x" % suffix), ex))
            self.assertTrue(np.array_equal(data(doc, "Combined%s_avg_dB" % suffix), ey,
                                           equal_nan=True))
            if mode == 0:                                     # section 1 of every file
                sx, sy = joined([np.asarray(f.x, float) for _s, f in group for t in f.traces
                                 if t.number == 1],
                                [t.values for _s, f in group for t in f.traces if t.number == 1])
                self.assertTrue(np.array_equal(data(doc, "Combined%s_Section1_x" % suffix), sx))
                self.assertTrue(np.array_equal(data(doc, "Combined%s_Section1" % suffix), sy,
                                               equal_nan=True))

    def test_touchstone(self):
        """Combined_SParam just above Overlay_SParam; two bands joined on
        frequency exactly; the step between the bands is laid out by the
        writer's numeric rule: 1-2 GHz and 5-6 GHz cut the lines but do not
        break the axis (3 GHz < n K median step = 10 GHz); 1-2 MHz and
        50-51 GHz also break it."""
        try:
            import skrf
            import vszh5_writer
            import Touchstone_AutoPlot as ts
        except Exception as exc:
            self.skipTest("Touchstone AutoPlot not importable: %s" % exc)
        for bands, broken in ((((5.0, 6.0), (1.0, 2.0)), False),
                              (((50.0, 51.0), (0.001, 0.002)), True)):
            plotter = ts.TouchstonePlotter()
            files = []
            for key, (f0, f1) in zip(("high", "low"), bands):
                frequency = skrf.Frequency(f0, f1, 101, unit="GHz")
                s = np.zeros((101, 2, 2), dtype=complex)
                s[:, 1, 0] = np.exp(-1j * np.linspace(0, 3, 101)) * (0.5 + f0 / 100)
                s[:, 0, 0] = 0.1
                network = skrf.Network(frequency=frequency, s=s, name=key)
                item = {"network": network, "file_path": key + ".s2p"}
                files.append((key, item, plotter.create_plots_from_data(key, item)))
            plotter.create_multi_file_pages(files, ("per_file", "combined", "overlay"))
            doc = plotter.doc
            self.assertEqual(page_names(doc), ["SParam_high", "SParam_low", "Combined_SParam",
                                               "Overlay_SParam"])
            ex, ey = joined([data(doc, "high_freq_GHz"), data(doc, "low_freq_GHz")],
                            [data(doc, "high_S21_mag_dB"), data(doc, "low_S21_mag_dB")])
            self.assertTrue(np.array_equal(data(doc, "Combined_S21_freq_GHz"), ex))
            self.assertTrue(np.array_equal(data(doc, "Combined_S21_mag_dB"), ey))
            doc.finalize()
            rule = TimeLayout().gap_rule(geometric=True)
            self.assertEqual(bool(vszh5_writer.rule_axis_break_pairs([ex], rule)), broken)
            page = next(p for p in doc.Root.children if p.name == "Combined_SParam")
            for graph in [g for g in page.WalkWidgets() if g.kind == "graph"]:
                axis = next(c for c in graph.children if c.name == "x")
                self.assertEqual(axis.kind, "axis-broken" if broken else "axis")
                for xy in [c for c in graph.children if c.kind == "xy"]:
                    self.assertEqual(int(np.isnan(data(doc, xy.settings["yData"])).sum()), 1)
        plotter.create_multi_file_pages(files, ("combined",))
        self.assertEqual(page_names(doc), ["SParam_high", "SParam_low", "Combined_SParam"])


# %% GUI
class GuiTests(unittest.TestCase):
    """PageTypesBox and the FITS window."""

    @classmethod
    def setUpClass(cls):
        """One offscreen QApplication."""
        try:
            from qtpy.QtWidgets import QApplication
        except Exception as exc:
            raise unittest.SkipTest("Qt not importable: %s" % exc)
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, test, seconds=60.0):
        """Run the event loop until ``test()`` or ``seconds``."""
        start = time.time()
        while not test() and time.time() - start < seconds:
            self.app.processEvents()
            time.sleep(0.01)

    def test_box(self):
        """All ticked; set_kinds; kinds_changed sends the tuple."""
        import _suite_gui
        box = _suite_gui.PageTypesBox(notes={"combined": "note"})
        self.assertEqual(box.kinds(), ("per_file", "combined", "overlay"))
        heard = []
        box.kinds_changed.connect(heard.append)
        box.set_kinds(("combined",))
        self.assertEqual(box.kinds(), ("combined",))
        self.assertEqual(heard[-1], ("combined",))
        self.assertTrue(box.combined_cb.toolTip().endswith("note"))

    def test_fits_window_rebuild(self):
        """FITS AutoPlot: the pages follow the boxes after a batch (rebuilt from
        the files already read), in the project and the Plots tab."""
        import FITS_AutoPlot as fitstool
        with tempfile.TemporaryDirectory() as folder:
            paths = [str(write_fits(Path(folder) / "a.fits", 61000.0)),
                     str(write_fits(Path(folder) / "b.fits", 61007.0))]
            window = fitstool.FITSAutoPlotWindow()
            try:
                window.selected_files = paths
                window._refresh_file_list()
                window._process_files()
                self.wait(window.save_button.isEnabled)
                names = page_names(window.veusz_doc)
                self.assertIn("Combined_ONEPPSDELTAS", names)
                self.assertLess(names.index("Combined_ONEPPSDELTAS"),
                                min(i for i, n in enumerate(names) if n.startswith("Overlay")))
                tab = [p["name"] for p in window.plot_panel.pages]
                self.assertIn("Combined_ONEPPSDELTAS", tab)
                window.page_types_box.per_file_cb.setChecked(False)
                window.page_types_box.overlay_cb.setChecked(False)
                self.wait(lambda: window.save_button.isEnabled() and
                          page_names(window.veusz_doc) == ["Combined_ONEPPSDELTAS"], 30.0)
                self.assertEqual(page_names(window.veusz_doc), ["Combined_ONEPPSDELTAS"])
                self.assertEqual([p["name"] for p in window.plot_panel.pages],
                                 ["Combined_ONEPPSDELTAS"])
            finally:
                window.close()


# %% Veusz plugins
class PluginTests(unittest.TestCase):
    """Both plugins: the "pages" field picks the page kinds; the old
    "combined_only" field still drops the per-file pages."""

    def setUp(self):
        """Two FITS files and two Franks files a week apart."""
        try:
            import FITS_AutoPlot_plugin  # noqa: F401
            import Franks_AutoPlot_plugin  # noqa: F401
        except Exception as exc:                  # optional stack
            self.skipTest("plugins not importable: %s" % exc)
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.fits = [str(write_fits(self.folder / "p1.fits", 61000.0)),
                     str(write_fits(self.folder / "p2.fits", 61007.0))]
        self.franks = []
        for n, start in ((1, 61000.0), (2, 61007.0)):
            path = self.folder / ("time_gbt_%d.dat" % n)
            rows = "".join("%.10f %d %d\n" % (start + k / 1440.0, k, -k) for k in range(20))
            path.write_text("MJD A B\n================\n" + rows, encoding="utf-8")
            self.franks.append(str(path))

    def tearDown(self):
        """Remove the folder."""
        self.tmp.cleanup()

    def pages_of(self, module, cls, paths, **extra):
        """Run the plugin's batch on ``paths``; the page names of the file."""
        from unittest.mock import patch
        import re
        import h5py
        from qtpy.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        output = self.folder / ("%s_%d.vszh5" % (cls.__name__, len(os.listdir(self.folder))))
        fields = dict(preselected_files=";".join(paths), output_file=str(output),
                      minimized=False, also_full=False, **extra)
        plugin = object.__new__(cls)
        with patch.object(module._PluginBatchDialog, "exec_", return_value=1), \
             patch.object(module._PluginBatchDialog, "append_log", lambda _self, msg: None), \
             patch.object(module.QMessageBox, "information"):
            plugin._apply(None, fields)
        with h5py.File(output) as handle:
            script = handle["Veusz/Document/document"][0].decode()
        return re.findall(r"Add\('page', name='([^']+)'", script)

    def test_fields(self):
        """Default: per file, combined (just above the overlays), overlay;
        pages=combined: the combined pages only; combined_only: no per-file."""
        import FITS_AutoPlot_plugin as fitsplugin
        import Franks_AutoPlot_plugin as franksplugin
        for module, cls, paths in ((fitsplugin, fitsplugin.FITSAutoPlotPlugin, self.fits),
                                   (franksplugin, franksplugin.FranksAutoPlotPlugin,
                                    self.franks)):
            with self.subTest(plugin=cls.__name__):
                names = self.pages_of(module, cls, paths)
                combined = [i for i, n in enumerate(names) if n.startswith("Combined")]
                overlay = [i for i, n in enumerate(names) if n.startswith("Overlay")]
                self.assertTrue(combined and overlay, names)
                self.assertEqual(combined[-1] + 1, overlay[0], names)
                self.assertLess(0, combined[0], names)                 # per-file pages first
                only = self.pages_of(module, cls, paths, pages="combined")
                self.assertTrue(only and all(n.startswith("Combined") for n in only), only)
                old = self.pages_of(module, cls, paths, combined_only=True)
                self.assertTrue(all(n.startswith(("Combined", "Overlay")) for n in old), old)
                self.assertEqual(sorted(old), sorted(n for n in names
                                                     if n.startswith(("Combined", "Overlay"))))


if __name__ == "__main__":
    unittest.main()
