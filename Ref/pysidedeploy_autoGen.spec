[app]
# title of your application
title = pyside_app_demo
# project directory. the general assumption is that project_dir is the parent directory
# of input_file
project_dir = .
# source file path
input_file = postProcessing_Launcher.py
# directory where exec is stored
exec_directory = .
# path to .pyproject project file
project_file = 
# application icon
icon = C:\Users\wwallace\miniforge3\envs\Py3p8\Library\bin\deploy_lib\pyside_icon.ico

[python]
# python path
python_path = C:\Users\wwallace\miniforge3\envs\Py3p8\python.exe
# python packages to install
# ordered-set = increase compile time performance of nuitka packaging
# zstandard = provides final executable size optimization
packages = nuitka==1.8.0,ordered_set,zstandard
# buildozer = for deploying Android application
android_packages = buildozer==1.5.0,cython==0.29.33

[qt]
# comma separated path to qml files required
# normally all the qml files required by the project are added automatically
qml_files = 
# excluded qml plugin binaries
excluded_qml_plugins = 

[android]
# path to pyside wheel
wheel_pyside = 
# path to shiboken wheel
wheel_shiboken = 
# plugins to be copied to libs folder of the packaged application. comma separated
plugins = platforms_qtforandroid

[nuitka]
# (str) specify any extra nuitka arguments
# eg = extra_args = --show-modules --follow-stdlib
extra_args = --quiet --noinclude-qt-translations

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

