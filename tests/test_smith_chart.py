# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_smith_chart.py

Writer 0.0.26 / _autoplot_native 0.4.0 / _autoplot_pyqtgraph 0.3.0: Smith
charts built from ordinary Veusz widgets.

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
        _curves: Split NaN-joined grid arrays into their curves.
        SmithGeometry (class): The grid satisfies the circle equations of the Smith chart.
        SmithPages (class): Page scripts, widget trees and the Plots-tab description.
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.1: First tests (grid geometry, conversions, page scripts, widget
Date: 2026-11-09
#              tree, Plots-tab recognition).
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

import vszh5_writer as w                                    # noqa: E402
from _autoplot_native import NativeDocument, add_smith_page  # noqa: E402


# %% Tests
def _curves(x, y):
    """Split NaN-joined grid arrays into their curves."""
    breaks = np.flatnonzero(np.isnan(x))
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, x.size]
    return [(x[a:b], y[a:b]) for a, b in zip(starts, stops)]


class SmithGeometry(unittest.TestCase):
    """The grid satisfies the circle equations of the Smith chart."""

    def test_resistance_circles(self):
        """One circle per SMITH_RESISTANCES value, with centre r/(1+r), radius
        1/(1+r), mapping back to resistance r.
        """
        grid = w.smith_grid("t", "z")
        curves = _curves(grid.numeric["t_r_x"], grid.numeric["t_r_y"])
        self.assertEqual(len(curves), len(w.SMITH_RESISTANCES))
        for r, (x, y) in zip(w.SMITH_RESISTANCES, curves):
            # (Re G - r/(1+r))^2 + (Im G)^2 = (1/(1+r))^2
            np.testing.assert_allclose((x - r / (1 + r)) ** 2 + y ** 2, (1 / (1 + r)) ** 2,
                                       atol=1e-12)
            gamma = x + 1j * y
            away = np.abs(1 - gamma) > 1e-6        # z is infinite at Gamma = 1 (both ends)
            z = (1 + gamma[away]) / (1 - gamma[away])
            np.testing.assert_allclose(z.real, r, rtol=1e-9)

    def test_reactance_arcs_inside_unit_circle(self):
        """Two arcs (+x and -x) per reactance value, all inside the unit circle.
        """
        grid = w.smith_grid("t", "z")
        curves = _curves(grid.numeric["t_x_x"], grid.numeric["t_x_y"])
        self.assertEqual(len(curves), 2 * len(w.SMITH_REACTANCES))
        signed = [s for x in w.SMITH_REACTANCES for s in (x, -x)]
        for x_value, (x, y) in zip(signed, curves):
            gamma = x[:-1] + 1j * y[:-1]                       # last point is Gamma = 1
            self.assertLessEqual(np.max(np.abs(gamma)), 1.0 + 1e-12)
            self.assertAlmostEqual(abs(gamma[0]), 1.0, places=12)   # starts on |G| = 1
            self.assertEqual((x[-1], y[-1]), (1.0, 0.0))
            z = (1 + gamma) / (1 - gamma)
            np.testing.assert_allclose(z.imag, x_value, rtol=1e-9)
            self.assertGreaterEqual(np.min(z.real), -1e-12)

    def test_admittance_is_rotated_impedance(self):
        """The admittance grid is the impedance grid rotated by 180 degrees."""
        gz, gy = w.smith_grid("a", "z"), w.smith_grid("a", "y")
        for part in ("r_x", "r_y", "x_x", "x_y", "unit_x", "unit_y"):
            np.testing.assert_allclose(gy.numeric["a_" + part], -gz.numeric["a_" + part])
        self.assertEqual(gy.notes, "Smith chart (admittance)")

    def test_vswr_and_labels(self):
        """VSWR 2 circle has |Gamma| = 1/3; grid value labels (optional) and
        invalid arguments.
        """
        grid = w.smith_grid("v", vswr=(2.0,))
        x, y = grid.numeric["v_vswr_x"], grid.numeric["v_vswr_y"]
        np.testing.assert_allclose(np.hypot(x, y), 1.0 / 3.0)        # (2 - 1) / (2 + 1)
        self.assertEqual(grid.text["v_r_text"], ["0.2", "0.5", "1", "2", "5"])
        self.assertIn("+1j", grid.text["v_x_text"])
        self.assertNotIn("v_r_text", w.smith_grid("v", labels=False).text)
        with self.assertRaises(ValueError):
            w.smith_grid(kind="q")
        with self.assertRaises(ValueError):
            w.smith_grid(vswr=(1.0,))

    def test_conversions(self):
        """Impedance <-> Gamma round trip and VSWR (1 for a match, 2 for
        |Gamma| = 1/3, infinite for |Gamma| = 1).
        """
        z = np.array([50.0, 25.0 + 30j, 0.0, 1e9])
        gamma = w.gamma_from_impedance(z, 50.0)
        np.testing.assert_allclose(w.impedance_from_gamma(gamma, 50.0)[:3], z[:3])
        np.testing.assert_allclose(w.vswr_from_gamma([0.0, 1 / 3]), [1.0, 2.0])
        self.assertTrue(np.isinf(w.vswr_from_gamma([1.0])[0]))


class SmithPages(unittest.TestCase):
    """Page scripts, widget trees and the Plots-tab description."""

    def _traces_and_data(self):
        """One synthetic S11 trace (|Gamma| = 0.5, 1-2 GHz) as Smith datasets.
        """
        f = np.linspace(1e9, 2e9, 11)
        gamma = 0.5 * np.exp(-2j * np.pi * f * 1e-9)
        data = w.smith_trace_datasets("dut_S11_gamma", gamma, f)
        return [w.smith_trace("dut_S11_gamma", "S11"),
                w.smith_trace("dut_S11_gamma", "S11 again")], data

    def test_page_script(self):
        """The Smith page script names the trace datasets, locks the aspect and
        hides the Re/Im axes.
        """
        traces, data = self._traces_and_data()
        self.assertEqual(sorted(data), ["dut_S11_gamma_freq_Hz", "dut_S11_gamma_im",
                                        "dut_S11_gamma_re"])
        doc = w.VszDoc()
        grid = w.smith_grid("smith_z", vswr=w.SMITH_VSWR)
        name = w.smith_page(doc, "Smith_dut", traces, grid, title="dut", notes="Source: dut.s2p")
        script = doc.text()
        self.assertEqual(name, "g_Smith_dut")
        for text in ("Set('aspect', 1.0)", "Set('notes', 'Smith chart (impedance)')",
                     "Add('xy', name='grid_x'", "Set('grid_x_labels/labels', 'smith_z_x_text')",
                     "Add('xy', name='dut_S11_gamma_im'", "Add('xy', name='dut_S11_gamma_im_2'",
                     "Set('key/horzPosn', 'manual')", "Set('x/hide', True)",
                     "Set('width', '24.500cm')", "Set('height', '19.000cm')"):
            self.assertIn(text, script)
        with tempfile.TemporaryDirectory() as folder:          # the whole file writes
            path = os.path.join(folder, "s.vszh5")
            with w.VeuszWriter(path) as writer:
                grid.write(writer)
                for key, values in data.items():
                    writer.dataset_1d(key, values)
                writer.finish(script)
            self.assertIn("smith_z_x_text", w.dataset_names(path))
            self.assertEqual(w.page_names(path), ["Smith_dut"])

    def test_native_page_and_plots_tab(self):
        """A Smith page on a NativeDocument saves and is drawn by the Plots tab.
        """
        traces, data = self._traces_and_data()
        doc = NativeDocument("t")
        grid = w.smith_grid("smith_y", "y")
        grid.write(doc)
        for key, values in data.items():
            doc.SetData(key, values)
        graph = add_smith_page(doc.Root, "Smith_dut", grid, traces[:1], title="dut",
                               notes="Source: dut.s2p", graph_name="dut_Smith_Graph",
                               graph_notes="Z0 = 75 ohm")
        self.assertEqual(graph.settings["aspect"], 1.0)
        self.assertEqual(graph.settings["notes"], "Smith chart (admittance)\nZ0 = 75 ohm")
        self.assertEqual([c.name for c in graph.children][-1], "dut_S11_gamma_im")  # no key
        self.assertEqual(doc.datasets["smith_y_x_text"][0], "text")
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(os.path.exists(doc.save(os.path.join(folder, "n.vszh5"))))
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            import _autoplot_pyqtgraph as pgmod
        except Exception as exc:                               # pragma: no cover
            self.skipTest("pyqtgraph not importable: %s" % exc)
        pages = pgmod.document_pages(doc)
        smith = pages[0]["graphs"][0]
        self.assertEqual(smith["kind"], "smith")
        self.assertEqual((smith["chart_type"], smith["z0"], smith["labels"], smith["vswr"]),
                         ("y", 75.0, True, False))
        self.assertEqual(smith["traces"][0]["freq_name"], "dut_S11_gamma_freq_Hz")
        self.assertEqual(smith["title"], "dut")


# %% Run
if __name__ == "__main__":
    unittest.main()
