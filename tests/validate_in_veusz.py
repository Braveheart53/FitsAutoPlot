# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/validate_in_veusz.py

Optional independent consumer test; not imported by any AutoPlot module.

Usage: python validate_in_veusz.py OUTPUT_DIRECTORY input1.vszh5 ...
This validation-only utility intentionally imports Veusz; generation does not.

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
# %%%% 0.1.0: Load and render native files using a separately installed Veusz.
Date: 2026-10-11
=============================================================================
"""
# %% Imports
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from veusz import document, qtall, utils, widgets, dataimport  # register classes/commands

# %% Script
app = qtall.QApplication.instance() or qtall.QApplication([])
folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
for filename in sys.argv[2:]:
    doc = document.Document()
    doc.load(filename, mode="hdf5")
    command = document.CommandInterface(doc)
    pages = doc.basewidget.children
    for index, page in enumerate(pages):
        target = folder / ("%s_%02d.svg" % (Path(filename).stem, index))
        command.Export(str(target), page=index)
        if not target.exists() or target.stat().st_size < 100:
            raise AssertionError("No rendered page: %s" % target.name)
    print("PASS Veusz %s: %s, %d datasets, %d rendered pages"
          % (utils.version(), Path(filename).name, len(doc.data), len(pages)))
