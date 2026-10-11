# -*- coding: utf-8 -*-
"""Optional independent consumer test; not imported by any AutoPlot module.
# %%%% 0.1.0: Load and render native files using a separately installed Veusz.
# Date: 2026-10-11
Usage: python validate_in_veusz.py OUTPUT_DIRECTORY input1.vszh5 ...
This validation-only utility intentionally imports Veusz; generation does not.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from veusz import document, qtall, utils, widgets, dataimport  # register classes/commands

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
