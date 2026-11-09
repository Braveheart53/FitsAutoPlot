# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/validate_plugin_host.py

Optional real Veusz plugin-registration test, separate from file generation.

Run using a Veusz-equipped Python: python tests/validate_plugin_host.py

Created on 2026-10-11

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.1.1: Header template and Spyder cells (no code change).
Date: 2026-11-09
# %%%%% Function Descriptions
        (none: a script without functions)
# %%%%% Variable Descriptions
    Define all utilized variables
        (no module constants)
# %%%%% More Info
    Needs a Python with Veusz installed (the suite itself does not).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.1.0: Exercise Document.loadPlugins and registered class globals.
Date: 2026-10-11
=============================================================================
"""
# %% Imports
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from veusz import document, qtall, widgets, dataimport, plugins, utils

# %% Script
app = qtall.QApplication.instance() or qtall.QApplication([])
folder = Path(__file__).resolve().parents[1]
expected = ("FITS_AutoPlot_plugin.py", "Franks_AutoPlot_plugin.py")
before = len(plugins.toolspluginregistry)
document.Document.loadPlugins([str(folder / name) for name in expected])
registered = plugins.toolspluginregistry[before:]
assert len(registered) == 2, len(registered)
for cls in registered:
    obj = cls()
    fields = {field.name for field in obj.fields}
    assert {"output_file", "minimized", "also_full"} <= fields
    assert "begin_native_plugin_export" in cls._apply.__globals__
    module = sys.modules[cls.__module__]
    dialog = module._PluginBatchDialog()
    dialog.close()
    print("PASS Veusz %s plugin loader: %s" % (utils.version(), cls.__name__))
    # Exercise the registered callback in the actual host Qt environment.
    import numpy as np
    import h5py
    from astropy.io import fits
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        if cls.__name__ == "FITSAutoPlotPlugin":
            input_file = directory / "synthetic.fits"
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="DMJD", format="D", array=[60805., 60806., 60807.]),
                fits.Column(name="DELTAT", format="D", array=[1., np.nan, 3.]),
            ], name="Timing")
            fits.HDUList([fits.PrimaryHDU(), table]).writeto(input_file)
        else:
            input_file = directory / "time_gbt.dat"
            input_file.write_text(
                "MJD A B C D\n========\n60805 1 2 3 4\n60806 2 nan 4 5\n",
                encoding="utf-8")
        output_file = directory / "native.vszh5"
        host = document.Document()
        before_text = host.basewidget.getSaveText()
        before_datasets = set(host.data)
        fields = dict(preselected_files=str(input_file),
                      output_file=str(output_file), minimized=True,
                      datetime_duplicate=True, hide_keys=True)
        with patch.object(module._PluginBatchDialog, "exec_", return_value=1), \
             patch.object(module.QMessageBox, "information"):
            obj.apply(document.CommandInterface(host), fields)
        assert host.basewidget.getSaveText() == before_text
        assert set(host.data) == before_datasets
        with h5py.File(output_file) as h:
            assert len(h["Veusz/Data"]) > 0
            assert h["Veusz"].attrs["vsz_format"] == 1
        print("PASS Veusz %s registered plugin export: %s; host unchanged"
              % (utils.version(), cls.__name__))
