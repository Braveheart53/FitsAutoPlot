# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('assets', 'assets')]
datas += collect_data_files('RAndS_FSW_ASCII_Plotter')
datas += collect_data_files('Hanging_Threads')
datas += collect_data_files('FITS_AutoPlot')
datas += collect_data_files('fastest_ascii_import')
datas += collect_data_files('CSV_TSV_AutoPlot')
datas += collect_data_files('ATR_AutoPlot')


a = Analysis(
    ['postProcessing_Launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['veusz', 'scikit-rf', 'veusz.plugins'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
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
