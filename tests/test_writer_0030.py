# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_writer_0030.py

vszh5_writer 0.0.30: gap options for any plotted series (GapRule, gap_plan,
x_breaks= on the page helpers, local_veusz_exact) - and every 0.0.29 call
still writes the same file.

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

# %%%% 0.0.2: Writer 0.0.31 and later; time-pair pages left out of the identity.
Date: 2026-11-16
#              * test_versions accepts 0.0.30 or later.
#              * build_document(pair=False) and test_documents_identical
#                without the time-pair pages, which writer 0.0.31 draws with
#                vertical labels on purpose (tests/test_writer_0031.py).
# %%%%% Function Descriptions
        load_writer: Import the writer file ``path`` as module ``name``.
        build_document: Write a document with every 0.0.29 page helper through writer module ``w`` (xy, grid, time pair, image, bar, text and Smith pages); ``pair`` False...
        document_objects: Every dataset, attribute and the script of a .vszh5, save stamps left out ({name: (attrs, bytes)} plus the script lines).
        IdentityWith0029 (class): Documents of every 0.0.29 helper are identical in 0.0.29 and 0.0.30.
        GapRuleTests (class): GapRule, rule_gap_positions, rule_axis_break_pairs and gap_plan.
        PageHelperTests (class): x_breaks on xy_page, Panel / grid_page and time_pair_pages.
        VeuszRenderTests (class): Veusz loads and renders the broken axes (skipped without Veusz).
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        MASTER_0029: the registry master 0.0.29 kept in docs/writer_0.0.30
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Needs docs/writer_0.0.30/vszh5_writer_0.0.29.py (the registry master
    0.0.29, SHA-256 checked); the render test needs a Python with Veusz.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.1: First version.
Date: 2026-11-16
#              Identity with master 0.0.29 (docs/writer_0.0.30), the gap
#              options, the geometric rule, the page helpers and a Veusz
#              load / render of broken axes when Veusz is installed.
# %%%%% Function Descriptions
        load_writer: Import the writer file ``path`` as module ``name``.
        build_document: Write a document with every 0.0.29 page helper through writer module ``w`` (xy, grid, time pair, image, bar, text and Smith pages).
        document_objects: Every dataset, attribute and the script of a .vszh5, save stamps left out ({name: (attrs, bytes)} plus the script lines).
        IdentityWith0029 (class): Documents of every 0.0.29 helper are identical in 0.0.29 and 0.0.30.
        GapRuleTests (class): GapRule, rule_gap_positions, rule_axis_break_pairs and gap_plan.
        PageHelperTests (class): x_breaks on xy_page, Panel / grid_page and time_pair_pages.
        VeuszRenderTests (class): Veusz loads and renders the broken axes (skipped without Veusz).
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
        MASTER_0029: the registry master 0.0.29 kept in docs/writer_0.0.30
# %%%%% More Info
    Run:  python -m unittest discover -s tests
    Needs docs/writer_0.0.30/vszh5_writer_0.0.29.py (the registry master
    0.0.29, SHA-256 checked); the render test needs a Python with Veusz.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
MASTER_0029 = ROOT / "docs" / "writer_0.0.30" / "vszh5_writer_0.0.29.py"
sys.path.insert(0, str(ROOT))

import vszh5_writer as w                                  # noqa: E402


# %% Helpers
def load_writer(path, name):
    """Import the writer file ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                # dataclasses look the module up
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # no __pycache__ in docs/
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


def build_document(writer_module, path, pair=True):
    """Write a document with every 0.0.29 page helper through writer module
    ``w`` (xy, grid, time pair, image, bar, text and Smith pages); ``pair``
    False leaves out the time-pair pages (0.0.2)."""
    m = writer_module
    t = np.arange(50.0)
    mjd = 61349.5 + t / 86400.0
    doc = m.VszDoc(generator="test")
    m.xy_page(doc, "xy", [m.Trace("t", "v", "volts"), m.Trace("t", "v2", "fit", style="dashed")],
              title="Volts", xlabel="Time (s)", ylabel="V", notes="header line")
    m.grid_page(doc, "grid", [m.Panel([m.Trace("t", "v")], "t", "v", "panel A"),
                              m.Panel([m.Trace("t", "v2")], "t", "v2", "panel B", ylog=True)])
    axis = m.local_time_axis(m.mjd_to_epoch(mjd), zone="-05:00")
    if pair:
        m.time_pair_pages(doc, "pair", [m.Trace("mjd", "v", "v")], "mjd", "local", axis,
                          title="Pair", ylabel="V")
    m.image_page(doc, "img", "z", title="Image")
    m.bar_page(doc, "bars", "a", ["c1"], title="Bars")
    m.text_page(doc, "text", "Text", ["line 1", "line 2"])
    grid = m.smith_grid("smith_z")
    m.smith_page(doc, "smith", [m.Trace("s_re", "s_im", "S11")], grid, title="Smith")
    with m.VeuszWriter(str(path), compress=True) as out:
        grid.write(out)
        out.dataset_1d("s_re", np.linspace(-0.5, 0.5, 11))
        out.dataset_1d("s_im", np.linspace(0.2, -0.2, 11))
        out.dataset_1d("t", t)
        out.dataset_1d("v", np.sin(t))
        out.dataset_1d("v2", np.cos(t) + 2.0)
        out.dataset_1d("mjd", mjd)
        out.dataset_date("local", axis.veusz_seconds)
        out.dataset_2d("z", np.outer(t[:5], t[:4]))
        out.dataset_text("a", ["x", "y"])
        out.dataset_1d("c1", np.array([1.0, 2.0]))
        out.finish(doc.text())


def document_objects(path):
    """Every dataset, attribute and the script of a .vszh5, save stamps left
    out ({name: (attrs, bytes)} plus the script lines)."""
    out = {}
    with h5py.File(str(path), "r") as handle:
        def visit(name, obj):
            """Record one HDF5 object (attributes, data bytes)."""
            attrs = {k: (v.tobytes() if hasattr(v, "tobytes") else repr(v))
                     for k, v in obj.attrs.items() if k not in ("vsz_saved_at",)}
            data = obj[()].tobytes() if isinstance(obj, h5py.Dataset) and obj.dtype.kind != "O" \
                else (repr(obj[()].tolist()) if isinstance(obj, h5py.Dataset) else b"")
            if name == "Veusz/Document/document":
                data = b""          # the script is compared line by line below
            out[name] = (attrs, data)
        handle.visititems(visit)
        script = handle["Veusz/Document/document"][0].decode().splitlines()
    # Only the save stamp and the writer's own version comment may differ
    return out, [line for line in script if not line.startswith("# Saved")
                 and not line.startswith("# vszh5_writer")]


# %% Tests
class IdentityWith0029(unittest.TestCase):
    """Documents of every 0.0.29 helper are identical in 0.0.29 and 0.0.30."""

    @classmethod
    def setUpClass(cls):
        """Load master 0.0.29 from docs/writer_0.0.30 (skip when it is missing)."""
        if not MASTER_0029.is_file():
            raise unittest.SkipTest("docs/writer_0.0.30/vszh5_writer_0.0.29.py missing")
        digest = hashlib.sha256(MASTER_0029.read_bytes()).hexdigest()
        if digest != "50631331470ebd50162030f971529044076a6c813a264393067522a504f9d701":
            raise AssertionError("the stored 0.0.29 is not the registry master")
        cls.old = load_writer(MASTER_0029, "vszh5_writer_0029")

    def test_versions(self):
        """The writer is 0.0.30 or later and its header says so (0.0.2)."""
        self.assertGreaterEqual(tuple(int(p) for p in w.__version__.split(".")), (0, 0, 30))
        self.assertIn("Version        : " + w.__version__,
                      (ROOT / "vszh5_writer.py").read_text(encoding="utf-8"))
        self.assertEqual(self.old.__version__, "0.0.29")

    def test_documents_identical(self):
        """Datasets, attributes and script lines are the same (save stamps and
        the version comment aside).  0.0.2: without the time-pair pages,
        which writer 0.0.31 draws with vertical labels on purpose
        (tests/test_writer_0031.py)."""
        with tempfile.TemporaryDirectory() as folder:
            build_document(self.old, Path(folder) / "old.vszh5", pair=False)
            build_document(w, Path(folder) / "new.vszh5", pair=False)
            old_objects, old_script = document_objects(Path(folder) / "old.vszh5")
            new_objects, new_script = document_objects(Path(folder) / "new.vszh5")
        self.assertEqual(sorted(old_objects), sorted(new_objects))
        for name in old_objects:
            self.assertEqual(old_objects[name][1], new_objects[name][1], name)
        self.assertEqual(old_script, new_script)
        self.assertNotIn("axis-broken", "\n".join(new_script))

    def test_gap_functions_unchanged(self):
        """0.0.27 gap helpers give the same results (axis_break_pairs default)."""
        t = np.r_[np.arange(100.0), 500.0 + np.arange(100.0)]
        for name in ("median_step",):
            self.assertEqual(getattr(w, name)(t), getattr(self.old, name)(t))
        self.assertEqual(w.gap_threshold(t), self.old.gap_threshold(t))
        np.testing.assert_array_equal(w.gap_positions(t, 10.0), self.old.gap_positions(t, 10.0))
        self.assertEqual(w.axis_break_pairs(t, 100.0), self.old.axis_break_pairs(t, 100.0))


class GapRuleTests(unittest.TestCase):
    """GapRule, rule_gap_positions, rule_axis_break_pairs and gap_plan."""

    T = np.r_[np.arange(100.0), 200.0 + np.arange(100.0), 100000.0 + np.arange(10.0)]

    def test_both_breaks_default(self):
        """Defaults: line breaks at both holes, the axis at the large one."""
        positions, breaks = w.gap_plan([self.T])
        np.testing.assert_array_equal(positions[0], [100, 200])
        self.assertEqual(len(breaks), 2)
        self.assertAlmostEqual(breaks[0], 299.0 + 0.02 * (100000.0 - 299.0))

    def test_either_break_alone(self):
        """break_lines / break_axis each turn one consequence off."""
        positions, breaks = w.gap_plan([self.T], w.GapRule(break_lines=False))
        self.assertEqual(positions[0].size, 0)
        self.assertEqual(len(breaks), 2)                 # axis still breaks
        positions, breaks = w.gap_plan([self.T], w.GapRule(break_axis=False))
        self.assertEqual(positions[0].size, 2)
        self.assertEqual(breaks, [])
        positions, breaks = w.gap_plan([self.T], w.GapRule(break_lines=False, break_axis=False))
        self.assertEqual((positions[0].size, breaks), (0, []))
        self.assertEqual(w.gap_plan([self.T], w.GapRule(n=0))[1], [])

    def test_geometric_log_sweep(self):
        """A log-spaced sweep is not gapped with geometric=True; a hole is."""
        sweep = np.geomspace(1e7, 1e10, 201)
        self.assertEqual(w.rule_gap_positions(sweep, w.GapRule()).size, 33)   # linear rule
        self.assertEqual(w.rule_gap_positions(sweep, w.GapRule(geometric=True)).size, 0)
        holed = np.r_[sweep[:100], sweep[150:]]
        np.testing.assert_array_equal(w.rule_gap_positions(holed, w.GapRule(geometric=True)),
                                      [100])
        segmented = np.r_[np.linspace(1e9, 2e9, 101), np.linspace(1e10, 1.1e10, 101)]
        np.testing.assert_array_equal(
            w.rule_gap_positions(segmented, w.GapRule(geometric=True)), [101])
        # axis break of the segmented sweep needs n x the small gap on both scales
        self.assertEqual(len(w.gap_plan([segmented], w.GapRule(geometric=True, n=10))[1]), 2)
        self.assertEqual(w.gap_plan([segmented], w.GapRule(geometric=True))[1], [])

    def test_fixed_gaps_and_several_traces(self):
        """A fixed small / large gap; the axis breaks only where no trace has data."""
        a = np.r_[np.arange(10.0), 50.0 + np.arange(10.0)]
        b = np.arange(0.0, 60.0, 5.0)
        positions, breaks = w.gap_plan([a, b], w.GapRule(manual=20.0, axis_manual=30.0))
        np.testing.assert_array_equal(positions[0], [10])      # 41 > 20
        self.assertEqual(positions[1].size, 0)
        self.assertEqual(breaks, [])                            # b covers the hole

    def test_describe_exact(self):
        """describe() writes every digit and names the options."""
        self.assertEqual(w.GapRule(k=12.3456789, n=0).describe(),
                         "Gaps: line break where dx > 12.3456789 x median(dx); no x-axis breaks.")
        self.assertIn("lines drawn across gaps", w.GapRule(break_lines=False).describe())
        self.assertIn("(and on a log scale)", w.GapRule(geometric=True).describe())

    def test_local_veusz_exact(self):
        """UTC + offset in one addition keeps the fraction of a second."""
        utc = np.array([562716000.1234568, np.nan])
        axis = w.local_time_from_veusz(utc, zone="-05:00")
        local = w.local_veusz_exact(utc, axis)
        self.assertEqual(local[0], 562716000.1234568 - 18000.0)
        self.assertTrue(np.isnan(local[1]))


class PageHelperTests(unittest.TestCase):
    """x_breaks on xy_page, Panel / grid_page and time_pair_pages."""

    def test_xy_page_breaks(self):
        """A non-empty x_breaks writes an axis-broken x axis with breakPoints."""
        doc = w.VszDoc()
        w.xy_page(doc, "p", [w.Trace("t", "v")], x_breaks=[1.5, 2.5])
        text = doc.text()
        self.assertIn("Add('axis-broken', name='x', autoadd=False)", text)
        self.assertIn("Set('x/breakPoints', [1.5, 2.5])", text)
        with self.assertRaises(ValueError):
            w.xy_page(w.VszDoc(), "p", [w.Trace("t", "v")], x_breaks=[1.0])

    def test_grid_and_pair_pages(self):
        """Panel.x_breaks and the two pages of time_pair_pages get their breaks."""
        doc = w.VszDoc()
        w.grid_page(doc, "g", [w.Panel([w.Trace("t", "v")], x_breaks=(1.0, 2.0)),
                               w.Panel([w.Trace("t", "v2")])])
        self.assertEqual(doc.text().count("axis-broken"), 1)
        mjd = 61349.5 + np.arange(10) / 86400.0
        axis = w.local_time_axis(w.mjd_to_epoch(mjd), zone="UTC")
        doc = w.VszDoc()
        w.time_pair_pages(doc, "pair", [w.Trace("mjd", "v")], "mjd", "local", axis,
                          mjd_breaks=[61349.50001, 61349.50002], local_breaks=[1.0, 2.0])
        text = doc.text()
        self.assertEqual(text.count("axis-broken"), 2)
        self.assertIn("[61349.50001, 61349.50002]", text)


class VeuszRenderTests(unittest.TestCase):
    """Veusz loads and renders the broken axes (skipped without Veusz)."""

    def test_render(self):
        """Every page with a broken x axis loads and exports to PNG."""
        try:
            from veusz import document, qtall, widgets, dataimport  # noqa: F401  (registers)
        except Exception as exc:                  # Veusz is optional
            self.skipTest("Veusz not installed: %s" % exc)
        app = qtall.QApplication.instance() or qtall.QApplication([])
        self.assertIsNotNone(app)
        t = np.r_[np.arange(100.0), 400.0 + np.arange(100.0)]
        v = np.sin(t / 10.0)
        positions, breaks = w.gap_plan([t], w.GapRule(n=1))
        doc = w.VszDoc()
        w.xy_page(doc, "xy", [w.Trace("t_gaps", "v_gaps", "v")], x_breaks=breaks)
        w.grid_page(doc, "grid", [w.Panel([w.Trace("t_gaps", "v_gaps")], x_breaks=breaks)])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "breaks.vszh5")
            with w.VeuszWriter(path) as out:
                out.dataset_1d("t_gaps", w.insert_gaps(t, positions[0]))
                out.dataset_1d("v_gaps", w.insert_gaps(v, positions[0]))
                out.finish(doc.text())
            vdoc = document.Document()
            vdoc.load(path, mode="hdf5")
            kinds = []

            def walk(widget):
                """Every widget below ``widget``."""
                for child in widget.children:
                    yield child
                    yield from walk(child)
            for widget in walk(vdoc.basewidget):
                if widget.name == "x":
                    kinds.append(widget.typename)
            self.assertEqual(kinds, ["axis-broken", "axis-broken"])
            cmd = document.CommandInterface(vdoc)
            for page in (0, 1):
                target = os.path.join(folder, "page%d.png" % page)
                cmd.Export(target, page=page, dpi=40)
                self.assertGreater(os.path.getsize(target), 500)


if __name__ == "__main__":
    unittest.main()
