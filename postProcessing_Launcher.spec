# -*- mode: python ; coding: utf-8 -*-
# %% Header Info
# postProcessing_Launcher.spec - PyInstaller build of the launcher (bundles the tools as data)
# %%% Author Information
# @author: William W. Wallace
# Author Email: wwallace@nrao.edu
# Author Secondary Email: naval.antennas@gmail.com
# Author Business Phone: +1 (304) 456-2216
# %%% Revisions
# Utilizing Semantic Schema as External Release.Internal Release.Working version
# %%%% 0.2.1: Header template; assets/ now holds real icons (bundled as before).
# Date: 2026-11-09
#             The whole assets/ folder is bundled, so the new PNG icons
#             (tools/make_suite_icons.py) ship with the frozen launcher.
# %%%% 0.2.0: (2026-11-05) see the notes below.
# PyInstaller spec for postProcessing_Launcher.py 0.2.0
# 2026-11-05: also bundles the shared modules the tools import (_suite_gui,
# _autoplot_pyqtgraph, vszh5_writer).
#
# The launcher starts each plotting tool as a separate Python process, so the
# tool scripts are bundled as plain data files (collect_data_files() on a
# single-module script returns nothing, which is why 0.1.4 bundled none).
# The launcher looks for a tool next to the .exe first, then in the bundle.
# A frozen launcher needs a Python interpreter for the tools: set
# GBO_POSTPROC_PYTHON to that python.exe, or have python on PATH.

datas = [('assets', 'assets')]
datas += [(script, '.') for script in (
    'ATR_AutoPlot.py',
    'FITS_AutoPlot.py',
    'FITS_AutoPlot_plugin.py',
    'Franks_AutoPlot.py',
    'Franks_AutoPlot_plugin.py',
    'CSV_TSV_AutoPlot.py',
    'RAndS_FSW_ASCII_Plotter.py',
    'Touchstone_AutoPlot.py',
    '_autoplot_common.py',
    '_autoplot_native.py',
    '_autoplot_pyqtgraph.py',
    '_suite_gui.py',
    'vszh5_writer.py',
    'fastest_ascii_import.py',
    'Hanging_Threads.py',
)]


a = Analysis(
    ['postProcessing_Launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='postProcessing_Launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\GBT_1.ico'],
)
