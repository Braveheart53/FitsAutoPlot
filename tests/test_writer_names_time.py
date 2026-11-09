# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_writer_names_time.py

Writer 0.0.25 / _autoplot_native 0.3.0: widget names from the data, page
notes from source headers, Veusz date-times shown in local ISO 8601.

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

# %%%% 0.0.2: Header template, Spyder cells; docstrings for every test.
Date: 2026-11-09
# %%%%% Function Descriptions
        WriterNamesAndNotes (class): Names and notes made by the writer's page helpers.
        LocalDateTimes (class): UTC Veusz date datasets shown on the processing computer's clock.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.1: First tests (names, notes, local date-time axes).
Date: 2026-11-09
=============================================================================
"""
# %% Imports
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vszh5_writer as w                           # noqa: E402
from _autoplot_native import NativeDocument, mark_local_time_axis  # noqa: E402


# %% Tests
class WriterNamesAndNotes(unittest.TestCase):
    """Names and notes made by the writer's page helpers."""

    def test_widget_name_rules(self):
        """widget_name() makes safe, non-empty Veusz widget names."""
        self.assertEqual(w.widget_name(" a/b  c "), "a_b_c")
        self.assertEqual(w.widget_name(".."), "widget")
        self.assertEqual(w.widget_name("S11 mag", prefix="g_"), "g_S11_mag")
        used = set()
        self.assertEqual([w.unique_widget_name("xy", used) for _ in range(3)],
                         ["xy", "xy_2", "xy_3"])

    def test_header_notes(self):
        """header_notes() writes title, source, a blank line, the cards and extra
        lines.
        """
        text = w.header_notes({"ORIGIN": ("NRAO Green Bank", "site"), "NAXIS": 0},
                              source="a.fits", title="Run 1", extra=["sorted by DMJD"])
        self.assertEqual(text.splitlines(), ["Run 1", "Source: a.fits", "",
                                             "ORIGIN = NRAO Green Bank / site", "NAXIS = 0",
                                             "", "sorted by DMJD"])
        self.assertEqual(w.header_notes(["Type;FSP;", "Values;801;"]),
                         "Type;FSP;\nValues;801;")

    def test_page_helpers_name_widgets_after_data(self):
        """xy_page() names its graph and trace after the data and stores the
        notes.
        """
        doc = w.VszDoc()
        graph = w.xy_page(doc, "spectrum", [w.Trace("f", "level_dBm", "level")],
                          notes=w.header_notes(["Ref Level;0;dBm;"], source="x.sft"))
        self.assertEqual(graph, "g_level_dBm")
        script = doc.text()
        self.assertIn("Add('graph', name='g_level_dBm'", script)
        self.assertIn("Add('xy', name='level_dBm'", script)
        self.assertIn("Set('notes', 'Source: x.sft", script)
        names = w.grid_page(doc, "overview", [w.Panel([w.Trace("t", "a")]),
                                              w.Panel([w.Trace("t", "a")])])
        self.assertEqual(names, ["g_a", "g_a_2"])
        self.assertEqual(w.image_page(doc, "map", "level_db"), "g_level_db")
        self.assertIn("Set('cbar_level_db/widgetName', 'img_level_db')", doc.text())
        self.assertEqual(w.xy_page(doc, "old", [w.Trace("x", "y")], graph_name="graph"), "graph")


class LocalDateTimes(unittest.TestCase):
    """UTC Veusz date datasets shown on the processing computer's clock."""

    def setUp(self):
        """Remember VSZH5_DISPLAY_TZ so each test can set its own zone."""
        self._saved = os.environ.get(w.DISPLAY_TZ_ENV)

    def tearDown(self):
        """Restore VSZH5_DISPLAY_TZ."""
        if self._saved is None:
            os.environ.pop(w.DISPLAY_TZ_ENV, None)
        else:
            os.environ[w.DISPLAY_TZ_ENV] = self._saved

    def test_fixed_offset_zone(self):
        """A fixed offset (-05:00) shifts the display copy; NaN stays NaN; the
        label names the offset.
        """
        os.environ[w.DISPLAY_TZ_ENV] = "-05:00"
        utc = w.epoch_to_veusz(np.array([1762678740.0, np.nan]))   # 2025-11-09T08:59:00Z
        axis = w.local_time_from_veusz(utc)
        self.assertEqual(axis.datetime_format, "%VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00")
        self.assertEqual(axis.label, "Local time (ISO 8601, UTC-05:00)")
        self.assertAlmostEqual(axis.veusz_seconds[0] - utc[0], -18000.0)
        self.assertTrue(np.isnan(axis.veusz_seconds[1]))

    def test_native_axis_resolved_on_save(self):
        """A NativeDocument date axis is laid out on the local clock when saved.
        """
        os.environ[w.DISPLAY_TZ_ENV] = "UTC-05:00"
        doc = NativeDocument("t")
        doc.SetDataDateTime("t_utc", w.epoch_to_veusz(1762678740.0 + 60.0 * np.arange(5)))
        doc.SetData("level", np.arange(5.0))
        self.assertEqual(doc.SetDataLocalTime("t_utc"), "t_utc_local")
        graph = doc.Root.Add("page", name="p").Add("graph", name="g")
        graph.Add("xy", xData="t_utc", yData="level")          # no name: from its data
        mark_local_time_axis(graph.x)
        with tempfile.TemporaryDirectory() as folder:
            path = doc.save(os.path.join(folder, "t.vszh5"))
            self.assertTrue(os.path.exists(path))
        xy = graph.children[-1]
        self.assertEqual(xy.name, "level")
        self.assertEqual(xy.settings["xData"], "t_utc_local")
        self.assertEqual(graph.x.settings["TickLabels/format"],
                         "%VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00")
        self.assertEqual(graph.x.settings["label"], "Local time (ISO 8601, UTC-05:00)")
        self.assertEqual(doc.datasets["t_utc"][0], "date")
        self.assertEqual(doc.datasets["t_utc_local"][0], "date")


# %% Run
if __name__ == "__main__":
    unittest.main()
