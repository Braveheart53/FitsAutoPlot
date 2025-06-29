[app]
# title of your application
title = ScientificPlottingSuite
# project directory. the general assumption is that project_dir is the parent directory
# of input_file
project_dir = .
# source file path
input_file = postProcessing_Launcher.py
# directory where exec is stored
exec_directory = ./dist
# path to .pyproject project file
project_file = 
# application icon
icon = C:\Users\wwallace\Documents\GitHub\FitsAutoPlot\assets\GBT_2.ico

[python]
# python path
python_path = C:\Users\wwallace\miniforge3\envs\Py3p8\python.exe
# python packages to install
# ordered-set = increase compile time performance of nuitka packaging
# zstandard = provides final executable size optimization
packages = nuitka==2.7.11,ordered_set,zstandard,veusz,qtpy,numpy,astropy,multiprocessing,subprocess
# buildozer = for deploying Android application
android_packages = buildozer==1.5.0,cython==0.29.33

[qt]
# comma separated path to qml files required
# normally all the qml files required by the project are added automatically
qml_files = 
# excluded qml plugin binaries
excluded_qml_plugins = QtQuick,QtQuick3D,QtCharts,QtWebEngine,QtTest,QtSensors
# qt modules used by the application
modules = Core,Gui,Widgets
# qt plugins to include
plugins = platforms,styles,imageformats
# path to pyside wheel (empty to use installed version)
wheel_pyside = 
# path to shiboken wheel (empty to use installed version)
wheel_shiboken = 

[android]
# path to pyside wheel
wheel_pyside = 
# path to shiboken wheel
wheel_shiboken = 
# plugins to be copied to libs folder of the packaged application. comma separated
plugins = platforms_qtforandroid

[nuitka]
# deployment mode = onefile or standalone
mode = standalone
# macos permissions (not applicable for windows)
macos.permissions = 
# (str) specify any extra nuitka arguments
# eg = extra_args = --show-modules --follow-stdlib
extra_args = 
	--quiet
	--noinclude-qt-translations
	--lto=yes
	--show-progress
	--assume-yes-for-downloads
	--include-data-dir=assets=assets
	--include-data-files=ATR_AutoPlot.py=./
	--include-data-files=FITS_AutoPlot.py=./
	--include-data-files=RAndS_FSW_ASCII_Plotter.py=./
	--include-data-files=*.py=./

[buildozer]
# build mode
# possible options = [release, debug]
# release creates an aab, while debug creates an apk
mode = debug
# contrains path to pyside6 and shiboken6 recipe dir
recipe_dir = 
# path to extra qt android jars to be loaded by the application
jars_dir = 
# if empty uses default ndk path downloaded by buildozer
ndk_path = 
# if empty uses default sdk path downloaded by buildozer
sdk_path = 
# modules used. comma separated
modules = 
# other libraries to be loaded. comma separated.
# loaded at app startup
local_libs = plugins_platforms_qtforandroid
# architecture of deployed platform
# possible values = ["aarch64", "armv7a", "i686", "x86_64"]
arch = 

