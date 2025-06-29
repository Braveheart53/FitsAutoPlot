[app]
# Title of your application
title = ScientificPlottingSuite
# Project directory - the general assumption is that project_dir is the parent directory of input_file
project_dir = .
# Source file path
input_file = postProcessing_Launcher.py
# Directory where executable is stored
exec_directory = dist
# Path to .pyproject project file (if exists)
project_file =
# Icon file path (converted to appropriate format for platform)
icon = assets/GBT_2.ico

[python]
# Python path - use current environment
python_path = python
# Python packages to install for deployment
# nuitka version can be specified, ordered-set improves compile performance, zstandard optimizes size
packages = nuitka==2.6.8,ordered-set,zstandard,veusz,qtpy,numpy,astropy
# Android packages (not needed for your use case)
android_packages = buildozer==1.5.*,cython==0.29.*

[qt]
# Comma separated path to qml files (empty for your widget-based app)
qml_files =
# Excluded qml plugin binaries to reduce size
excluded_qml_plugins = QtQuick,QtQuick3D,QtCharts,QtWebEngine,QtTest,QtSensors
# Qt modules used by the application
modules = Core,Gui,Widgets
# Qt plugins to include
plugins = platforms,styles,imageformats
# Path to pyside wheel (empty to use installed version)
wheel_pyside =
# Path to shiboken wheel (empty to use installed version)
wheel_shiboken =

[nuitka]
# Deployment mode: onefile or standalone
mode = onefile
# macOS permissions (not applicable for Windows)
macos.permissions =
# Extra Nuitka arguments - this is where your complex Nuitka options go
extra_args = 
    --mingw64
    --follow-imports
    --enable-plugin=pylint-warnings
    --nofollow-import-to=pyqt5
    --nofollow-import-to=PyQt5
    --nofollow-import-to=tkinter
    --nofollow-import-to=tk-inter
    --nofollow-import-to=Tkinter
    --include-package=veusz
    --include-package-data=veusz
    --include-package=qtpy
    --include-package=numpy
    --include-package=astropy
    --include-package=multiprocessing
    --include-package=subprocess
    --include-data-dir=assets=assets
    --include-data-files=ATR_AutoPlot.py=./
    --include-data-files=FITS_AutoPlot.py=./
    --include-data-files=RAndS_FSW_ASCII_Plotter.py=./
    --include-data-files=*.py=./
    --lto=yes
    --remove-output
    --show-progress
    --show-memory
    --show-modules
    --windows-console-mode=disable
    --assume-yes-for-downloads
    --quiet
    --noinclude-qt-translations=True

[buildozer]
# Build mode for Android (not applicable for your use case)
mode = debug
# Recipe directory path
recipe_dir =
# Path to extra qt android jars
jars_dir =
# NDK path (empty uses default)
ndk_path =
# SDK path (empty uses default)
sdk_path =
# Modules used (comma separated)
modules =
# Other libraries to be loaded (comma separated)
local_libs =
# Architecture of deployed platform
arch =
