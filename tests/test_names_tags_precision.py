# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_names_tags_precision.py

Tests of the 2026-11-10 suite update (FITS / Franks 0.4.2, ATR 0.2.4,
CSV/TSV 6.1.3, R&S 1.1.4, Touchstone 0.7.3, _autoplot_native 0.5.2,
_autoplot_pyqtgraph 0.3.2):

* nothing is cut: numbers keep every digit in notes, titles and exports;
  values float64 cannot hold (long numbers, words among numbers, integers
  beyond 2**53, nanosecond time stamps) also keep their text;
* every dataset is tagged source_<origin> and type_<kind>;
* trace widgets are named after the data they draw;
* results of a forced process pool stay in input order (the Touchstone
  0.7.1 time-gated page-order bug, fixed in 0.7.2).

Run:  python -m unittest discover -s tests

Created on 2026-11-10

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version.
Date: 2026-11-10
#              Shared helpers, Franks / ATR / CSV / R&S / Touchstone parsers
#              and projects, pyqtgraph CSV export, forced-pool order.
# %%%%% Function Descriptions
        write_text: Write ``text`` to ``folder/name`` and return the path.
        sleep_then_return: Process-pool job: sleep ``item[1]`` seconds, return ``item[0]``.
        tags_of: Sorted tags of dataset ``name`` in a NativeDocument.
        NativeHelperTests (class): _autoplot_native 0.5.2: tags, names, exact numbers, local time.
        MjdTextTests (class): _autoplot_common.mjd_to_datestr: every digit of the time.
        FranksTests (class): Franks AutoPlot 0.4.2 parser and tags.
        ATRTests (class): ATR AutoPlot 0.2.4: nothing dropped, exact frequencies, tags.
        CSVTests (class): CSV/TSV AutoPlot 6.1.3: exact numbers, kept text, tags.
        RnSTests (class): R&S plotter 1.1.4: inf / nan words, kept sections, tags.
        TouchstoneTests (class): Touchstone AutoPlot 0.7.3: exact notes, tags, pool order.
        PyqtgraphCsvTests (class): _autoplot_pyqtgraph 0.3.2: the CSV export keeps every digit.
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

import _autoplot_native as native                         # noqa: E402


# %% Helpers
def write_text(folder: str, name: str, text: str) -> str:
    """Write ``text`` to ``folder/name`` and return the path."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def sleep_then_return(item):
    """Process-pool job: sleep ``item[1]`` seconds, return ``item[0]``."""
    import time
    time.sleep(item[1])
    return item[0]


def tags_of(doc, name):
    """Sorted tags of dataset ``name`` in a NativeDocument."""
    return sorted(tag for tag, members in doc.tags.items() if name in members)


# %% Shared helpers
class NativeHelperTests(unittest.TestCase):
    """_autoplot_native 0.5.2: tags, names, exact numbers, local time."""

    def test_tag_words(self):
        """source_/type_ tags use letters, digits and '_' only."""
        self.assertEqual(native.source_tag("2026_04.t00new"), "source_2026_04_t00new")
        self.assertEqual(native.type_tag("Temp [C]"), "type_Temp_C")
        self.assertEqual(native.tag_word("///"), "unnamed")

    def test_trace_widget_name(self):
        """A trace is named after its dataset; a leading digit stays."""
        self.assertEqual(native.trace_widget_name("10MHz clk_Section1"), "10MHz_clk_Section1")
        self.assertEqual(native.trace_widget_name("a__b__sorted", "local time"),
                         "a__b__sorted_local_time")

    def test_exact_number_text(self):
        """repr() for floats, str() for integers: float(text) is the same value."""
        for value in (0.1, 1420.40575, 1e-300, 61234.10355, 2.0 ** 60 + 0.5):
            self.assertEqual(float(native.exact_number_text(value)), value)
        self.assertEqual(native.exact_number_text(np.int64(9007199254740993)),
                         "9007199254740993")

    def test_float_loses_text(self):
        """Words, extra digits and missing-value words."""
        texts = np.array(["1.5", "2.000", "", "NaN", "N/A"])
        values = np.array([1.5, 2.0, np.nan, np.nan, np.nan])
        self.assertFalse(native.float_loses_text(texts, values))
        self.assertTrue(native.float_loses_text(np.array(["1.5", "oops"]),
                                                np.array([1.5, np.nan])))
        long_text = "0.12345678901234567891"
        self.assertTrue(native.float_loses_text(np.array([long_text]),
                                                np.array([float(long_text)])))
        exact_long = repr(0.12345678901234568)
        self.assertFalse(native.float_loses_text(np.array([exact_long]),
                                                 np.array([float(exact_long)])))

    def test_tag_data_and_local_copy(self):
        """tag_data() tags; a local copy is UTC + offset with every digit."""
        doc = native.NativeDocument("test")
        utc = np.array([562716000.1234568, np.nan, 562716002.0])
        doc.SetDataDateTime("t", utc)
        local = doc.SetDataLocalTime("t", zone="Europe/Berlin")
        native.tag_data(doc, ["t"], "file 1", ("time_utc", "x"))
        self.assertEqual(tags_of(doc, "t"), ["source_file_1", "type_time_utc", "type_x"])
        axis = doc.local_axes[local]
        stored = doc.datasets[local][1]["data"]
        np.testing.assert_array_equal(stored, utc + axis.offsets_s)
        self.assertTrue(np.isnan(stored[1]))

    def test_finalize_tags_everything(self):
        """finalize() gives every dataset a source_ and a type_ tag."""
        doc = native.NativeDocument("My Program")
        doc.SetData("x", np.arange(3.0))
        doc.SetDataText("words", ["a", "b"])
        doc.finalize()
        for name in ("x", "words"):
            tags = tags_of(doc, name)
            self.assertTrue(any(t.startswith("source_") for t in tags), tags)
            self.assertTrue(any(t.startswith("type_") for t in tags), tags)


class MjdTextTests(unittest.TestCase):
    """_autoplot_common.mjd_to_datestr: every digit of the time."""

    def test_exact_fraction(self):
        """MJD 61234.10355 -> 02:29:06.72 (0.10355 d = 8946.72 s), not rounded."""
        import _autoplot_common as common
        out = common.mjd_to_datestr(np.array([61234.10355, 61220.0, np.nan]))
        self.assertEqual(list(out), ["2026-07-13_02:29:06.72", "2026-06-29_00:00:00", "NaN"])


# %% Programs
class FranksTests(unittest.TestCase):
    """Franks AutoPlot 0.4.2 parser and tags."""

    TEXT = ("MJD      EECO-REF  NIST-REF   NS    DATE        COMMENTS\n"
            "==========================================\n"
            "60700.00000  4.978  2.000  1.658  2025-01-23\n"
            "60700.04167  4.984  2.000  0.308  2025-01-23  maser  swap,  see log\n"
            "# operator note\n"
            "60700.08333  4.989  bad  0.12345678901234567891  2025-01-23\n")

    def test_comments_and_exact_text_kept(self):
        """COMMENTS text, '#' lines and text float64 cannot hold are kept."""
        import Franks_AutoPlot as franks
        from _autoplot_common import MemoryAwareCache
        with tempfile.TemporaryDirectory() as folder:
            path = write_text(folder, "time_gbt.dat", self.TEXT)
            data = franks.parse_franks_file(path, MemoryAwareCache())
            columns = {k: np.asarray(v) for k, v in data["columns"].items()}
        self.assertEqual(list(columns["COMMENTS"]), ["", "maser  swap,  see log", ""])
        self.assertEqual(list(columns["NIST_REF_exact_text"]), ["2.000", "2.000", "bad"])
        self.assertEqual(columns["NS_exact_text"][2], "0.12345678901234567891")
        self.assertIn("# operator note", data["header_lines"])
        self.assertEqual(data["source_name"], "time_gbt")

    def test_push_tags_every_dataset(self):
        """push_franks_to_veusz(): source_<file> and type_ tags on all datasets."""
        import Franks_AutoPlot as franks
        from _autoplot_common import MemoryAwareCache
        with tempfile.TemporaryDirectory() as folder:
            path = write_text(folder, "time_gbt.dat", self.TEXT)
            data = franks.parse_franks_file(path, MemoryAwareCache())
            doc = native.NativeDocument("Franks AutoPlot")
            franks.push_franks_to_veusz(doc, data, emit_datestr=True)
        for name in doc.datasets:
            tags = tags_of(doc, name)
            self.assertIn("source_time_gbt", tags, name)
            self.assertTrue(any(t.startswith("type_") for t in tags), (name, tags))
        self.assertIn("type_text", tags_of(doc, "time_gbt__COMMENTS__raw"))
        names = [w.name for w in doc.Root.WalkWidgets() if w.kind == "xy"]
        self.assertTrue(names and all(not n.startswith("xy_") for n in names), names)


class ATRTests(unittest.TestCase):
    """ATR AutoPlot 0.2.4: nothing dropped, exact frequencies, tags."""

    TEXT = ("Time and Date: Thu, Jul 11, 2024   10:40 AM\n"
            "Left Azimuth Limit: -1\nRight Azimuth Limit: 1\n"
            "Upper Frequency: 1420.4052\nLower Frequency: 1420.4051\n"
            "Frequency Increment: 0.0001\n\n"
            "10.0 -1.0 11.0 -Inf 12.0 -3.0\n"
            "20.0 -4.0 junk -5.0 22.0 -6.0\n"
            "*\ntrailing table line 1\n")

    def test_parse_keeps_everything(self):
        """Text tokens, +/-Inf and the lines after the data are kept."""
        import ATR_AutoPlot as atr
        pattern = atr.parse_atr_text(self.TEXT, "test.atr")
        self.assertEqual(sorted(pattern.text_rows), [0, 1])
        self.assertEqual(pattern.text_rows[0][3], "-Inf")
        self.assertEqual(pattern.text_rows[1][2], "junk")
        self.assertEqual(pattern.trailer_lines, ["*", "trailing table line 1"])
        self.assertEqual(atr._mhz_text(2700.0), "2700")
        self.assertEqual(atr._mhz_text(1420.4051), "1420.4051")

    def test_close_frequencies_do_not_collide(self):
        """1420.4051 and 1420.4052 MHz get two datasets (6-digit %g merged them)."""
        import ATR_AutoPlot as atr
        plotter = atr.build_project([atr.parse_atr_text(self.TEXT, "test.atr")])
        doc = plotter.doc
        self.assertIn("test_1420p4051MHz_mag", doc.datasets)
        self.assertIn("test_1420p4052MHz_mag", doc.datasets)
        self.assertIn("test_1420p4051MHz_mag_exact_text", doc.datasets)
        self.assertEqual(tags_of(doc, "test_1420p4051MHz_mag"),
                         ["source_test", "type_magnitude_dB"])
        notes = "".join(str(p.settings.get("notes", "")) for p in doc.Root.children)
        self.assertIn("Lines after the data, as written:", notes)
        self.assertIn("trailing table line 1", notes)


class CSVTests(unittest.TestCase):
    """CSV/TSV AutoPlot 6.1.3: exact numbers, kept text, tags."""

    TEXT = ("Time,Counter,Value,Mixed\n"
            "2026-11-01T00:00:00.123456789+02:00,9007199254740993,0.12345678901234567891,1.5\n"
            "2026-11-01T00:00:01.5Z,12,2.000,oops\n"
            "2026-11-01T00:00:02+02:00,13,3.25,2.5\n"
            "2026-11-01T00:00:03+02:00,14,4,3.5\n"
            "2026-11-01T00:00:04+02:00,15,5,4.5\n")

    def test_project(self):
        """Time text, exact-text copies, correctly rounded seconds, tags."""
        import CSV_TSV_AutoPlot as csvplot
        with tempfile.TemporaryDirectory() as folder:
            path = write_text(folder, "exact.csv", self.TEXT)
            loaded, _msg = csvplot.load_csv_file(path)
            plotter = csvplot.VeuszPlotter()
            plotter.create_overlay_and_individual_plots(
                [loaded], "Time", ["Counter", "Value", "Mixed"])
            doc = plotter.doc
            # ISO 8601 stamps of mixed form are dates (6.1.2 skipped the file)
            self.assertEqual(doc.datasets["exact_Time"][0], "date")
            # 00:00:00.123456789+02:00 = 562716000.123456789 s: the nearest float64
            self.assertEqual(float(doc.datasets["exact_Time"][1]["data"][0]),
                             562716000.123456789)
            self.assertEqual(list(doc.datasets["exact_Time_text"][1]["data"])[0],
                             "2026-11-01T00:00:00.123456789+02:00")
            self.assertEqual(list(doc.datasets["exact_Counter_exact_text"][1]["data"])[0],
                             "9007199254740993")
            self.assertEqual(list(doc.datasets["exact_Value_exact_text"][1]["data"])[0],
                             "0.12345678901234567891")
            self.assertEqual(list(doc.datasets["exact_Mixed_exact_text"][1]["data"])[1], "oops")
            self.assertEqual(tags_of(doc, "exact_Value"),
                             ["source_exact", "type_Value", "type_y_data"])

    def test_scan_number_forms(self):
        """Long digit runs and exponent notation are found (the forms the fast
        pandas parser may read wrongly); short positional numbers are not."""
        import CSV_TSV_AutoPlot as csvplot
        with tempfile.TemporaryDirectory() as folder:
            short = write_text(folder, "a.csv", "a,b\n1.5,-2.25\n123456789012.5,7\n")
            long_run = write_text(folder, "b.csv", "a\n0.12345678901234567\n")
            exponent = write_text(folder, "c.csv", "a\n1.5e-30\n")
            self.assertEqual(csvplot.scan_number_forms(short), (False, False))
            self.assertEqual(csvplot.scan_number_forms(long_run), (True, False))
            self.assertEqual(csvplot.scan_number_forms(exponent), (False, True))

    def test_exact_parser_when_needed(self):
        """17-digit values are read exactly (round_trip where pandas' default
        parser is not exact: pandas 2.0.3 and 3.0.5 were measured)."""
        import CSV_TSV_AutoPlot as csvplot
        values = np.random.default_rng(2).normal(size=2000) * 1e-3
        with tempfile.TemporaryDirectory() as folder:
            path = write_text(folder, "d.csv", "x\n" + "\n".join(repr(float(v)) for v in values))
            loaded, _msg = csvplot.load_csv_file(path)
        np.testing.assert_array_equal(loaded.frame["x"].to_numpy(), values)


class RnSTests(unittest.TestCase):
    """R&S plotter 1.1.4: inf / nan words, kept sections, tags."""

    HEADER = (b"Type;FSP;\r\nStart;100;Hz\r\nStop;300;Hz\r\nx-Unit;Hz;\r\n"
              b"y-Unit;dBm;\r\nValues;3;\r\n")

    def test_sections_kept(self):
        """A -inf word is a number; a short section and a text section are kept."""
        import RAndS_FSW_ASCII_Plotter as rns
        raw = (self.HEADER + b"Section 1;2025.6.17_9.26.44:783\r\n-10.5;-inf;-12.25\r\n"
               b"Section 2;2025.6.17_9.26.45:000\r\n-1;-2\r\n"
               b"Section 3;2025.6.17_9.26.46:000\r\nOVLD;UNCAL;x\r\n")
        sft = rns.parse_sft_bytes(raw, "probe.sft")
        self.assertEqual(len(sft.traces), 1)
        self.assertTrue(np.isneginf(sft.traces[0].values[1]))
        self.assertEqual([t.number for t in sft.unaligned], [2])
        self.assertEqual(sft.text_sections[0][2], ["OVLD", "UNCAL", "x"])
        plotter = rns.RnSPlotter()
        stem = plotter.add_file(sft)
        doc = plotter.doc
        self.assertIn("%s_Section2_unaligned" % stem, doc.datasets)
        self.assertIn("%s_Section3_text" % stem, doc.datasets)
        self.assertEqual(tags_of(doc, "%s_avg_dB" % stem),
                         ["source_probe", "type_average_dB", "type_dBm"])

    def test_notes_list_every_sweep(self):
        """The average page lists all sweeps (1.1.3 stopped at 2000)."""
        import RAndS_FSW_ASCII_Plotter as rns
        sections = b"".join(b"Section %d;2025.6.17_9.26.44:783\r\n-1;-2;-3\r\n" % (n + 1)
                            for n in range(2105))
        sft = rns.parse_sft_bytes(self.HEADER + sections, "many.sft")
        plotter = rns.RnSPlotter(rns.PlotOptions(mode=2))
        plotter.add_file(sft)
        notes = "".join(str(p.settings.get("notes", "")) for p in plotter.doc.Root.children)
        self.assertIn("Section 2105:", notes)
        self.assertNotIn("more", notes)


class TouchstoneTests(unittest.TestCase):
    """Touchstone AutoPlot 0.7.3: exact notes, tags, pool order."""

    @classmethod
    def setUpClass(cls):
        """Import the program; skip when scikit-rf is missing."""
        try:
            import Touchstone_AutoPlot
        except Exception as exc:  # pragma: no cover - optional stack
            raise unittest.SkipTest("Touchstone AutoPlot not importable: %s" % exc)
        cls.T = Touchstone_AutoPlot
        cls.files = sorted(glob.glob(str(EXAMPLES / "Touchstone" / "s2p" / "*.s2p")))[:6]
        if len(cls.files) < 6:
            raise unittest.SkipTest("example .s2p files missing")

    def test_number_text(self):
        """_num(): every digit, no trailing .0."""
        self.assertEqual(self.T._num(1.23456789), "1.23456789")
        self.assertEqual(self.T._num(50.0), "50")

    def test_tags_and_names(self):
        """Frequency pages: source_<file>, source_<run>, typed tags; data-named traces."""
        import skrf
        plotter = self.T.TouchstonePlotter(dataset_name="Run A")
        path = self.files[0]
        key = os.path.basename(path)
        plotter.create_plots_from_data(key, {"network": skrf.Network(path), "file_path": path,
                                             "header_info": []})
        doc = plotter.doc
        base = self.T.sanitize_name(key)
        stem = os.path.splitext(key)[0]
        self.assertEqual(tags_of(doc, "%s_S21_mag_dB" % base),
                         sorted(["source_" + native.tag_word(stem), "source_Run_A",
                                 "type_S21", "type_magnitude_dB"]))
        names = [w.name for w in doc.Root.WalkWidgets() if w.kind == "xy"]
        self.assertIn("%s_S21_mag_dB" % base, names)

    def test_forced_pool_keeps_file_order(self):
        """Time domain in a forced process pool returns the files in input
        order even when a later file finishes first (0.7.1 built the
        time-gated pages in completion order; fixed in 0.7.2)."""
        import skrf
        from dataclasses import asdict
        import _autoplot_parallel as par
        items = [(os.path.basename(p), skrf.Network(p)) for p in self.files]
        # the first file after the 2-file probe is made much larger, so the
        # pool finishes the ones after it first
        big = items[2][1]
        f = big.frequency
        dense = skrf.Frequency(f.start, f.stop, 200001, unit="hz")
        items[2] = (items[2][0], big.interpolate(dense))
        finished = []
        original = self.T.adaptive_map

        def forced(func, jobs, workers, **kwargs):
            """adaptive_map with the pool forced on and completion order recorded."""
            kwargs["force"] = True
            kwargs["on_result"] = lambda index, _result: finished.append(index)
            return par.adaptive_map(func, jobs, workers, **kwargs)

        self.T.adaptive_map = forced
        try:
            out = self.T.run_time_domain_batch(items, asdict(self.T.TimeDomainConfig()), False,
                                               2, True, lambda _p: None)
        finally:
            self.T.adaptive_map = original
        self.assertEqual(list(out), [k for k, _n in items])
        self.assertEqual(sorted(finished), list(range(len(items))))

    def test_pool_order_when_completion_order_differs(self):
        """adaptive_map() with a forced pool: item 2 sleeps 2 s, items 3-5 finish
        first, and the results still come back in input order (the property
        the Touchstone 0.7.2 fix relies on)."""
        import _autoplot_parallel as par
        if not par.main_module_is_spawn_safe():
            self.skipTest("main script cannot start spawn workers")
        jobs = [("a", 0.0), ("b", 0.0), ("c", 2.0), ("d", 0.05), ("e", 0.05), ("f", 0.05)]
        finished = []
        out = par.adaptive_map(sleep_then_return, jobs, 2, force=True,
                               on_result=lambda index, _r: finished.append(index))
        self.assertEqual(out, ["a", "b", "c", "d", "e", "f"])
        self.assertNotEqual(finished, sorted(finished), finished)
        self.assertEqual(finished[-1], 2)


class PyqtgraphCsvTests(unittest.TestCase):
    """_autoplot_pyqtgraph 0.3.2: the CSV export keeps every digit."""

    def test_csv_export_round_trip(self):
        """Values written by the Export dialog's CSV exporter read back exactly."""
        try:
            import _autoplot_pyqtgraph as plots
            import pyqtgraph as pg
            from pyqtgraph.exporters import listExporters
        except Exception as exc:  # pragma: no cover - optional stack
            self.skipTest("pyqtgraph not importable: %s" % exc)
        app = pg.mkQApp()
        self.assertIsNotNone(app)
        exporters = [e for e in listExporters() if "CSV" in e.Name]
        self.assertEqual([e.Name for e in exporters], ["CSV of original plot data (every digit)"])
        widget = pg.PlotWidget()
        x = np.array([562716000.1234568, 1.0])
        y = np.array([1.234e-12, 0.12345678901234568])
        widget.plot(x, y, name="trace")
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "plot.csv")
            exporters[0](widget.getPlotItem()).export(out)
            with open(out, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines[0], "trace_x,trace_y")
        values = [float(v) for line in lines[1:] for v in line.split(",")]
        self.assertEqual(values, [x[0], y[0], x[1], y[1]])
        self.assertEqual(plots.exact_cell_text(np.int64(7)), "7")


if __name__ == "__main__":
    unittest.main()
