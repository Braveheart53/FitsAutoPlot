# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/make_demo_projects.py

Create synthetic examples; never reads the user's measurement files.

Usage: python tests/make_demo_projects.py [output_directory]

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
    Run:  python tests/make_demo_projects.py [output_directory]
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.1.0: Reproducible native writer examples.
Date: 2026-10-11
=============================================================================
"""
# %% Imports
import sys
from pathlib import Path
from test_native_export import (
    NativeTests, NativeDocument, MemoryAwareCache, MemoryMonitorConfig,
    fitstool, frankstool,
)

# %% Script
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples")
folder.mkdir(parents=True, exist_ok=True)
fixtures = NativeTests()
fixtures.setUp()
try:
    for tagged in (True, False):
        cache = MemoryAwareCache(MemoryMonitorConfig())
        path = fixtures.write_fits(tagged)
        data = fitstool.FITSProcessor("astropy", cache).read(str(path))
        doc = NativeDocument()
        fitstool.push_to_veusz(
            doc, str(path), data, "astropy", datetime_duplicate=True,
            hide_keys=True, gap_absolute=1.0)
        fitstool.build_unit_overlay_pages(
            doc, [dict(data, base=data["base_name"])], datetime_duplicate=True,
            hide_keys=True, gap_absolute=1.0)
        stem = "synthetic_fits_" + ("tagged" if tagged else "untagged")
        doc.save(folder / (stem + ".vszh5"), minimized=True)
        doc.save(folder / (stem + "_full.vszh5"))
        cache.cleanup()
    cache = MemoryAwareCache(MemoryMonitorConfig())
    path = fixtures.write_franks()
    data = frankstool.parse_franks_file(str(path), cache)
    doc = NativeDocument()
    frankstool.push_franks_to_veusz(
        doc, data, datetime_duplicate=True, hide_keys=True, gap_absolute=1.0)
    frankstool.build_unit_overlay_pages_franks(
        doc, [dict(data, base=data["base_name"])], datetime_duplicate=True,
        hide_keys=True, gap_absolute=1.0)
    doc.save(folder / "synthetic_franks.vszh5", minimized=True)
    doc.save(folder / "synthetic_franks_full.vszh5")
    cache.cleanup()
finally:
    fixtures.tearDown()
print("Wrote six synthetic full/minimized example projects.")
