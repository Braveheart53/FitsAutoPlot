# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
postProcessing_Launcher.py

GBO Electronics Scientific Plotting Suite - central launcher for the
FitsAutoPlot post-processing GUIs.  One button (and one Applications menu
entry) per tool:

    * ATR_AutoPlot.py             GBO Outdoor Antenna Range (.atr) files
    * FITS_AutoPlot.py            NRAO / GBO FITS files -> native .vszh5
    * Franks_AutoPlot.py          "FranksProcessed" GB timing data files
    * CSV_TSV_AutoPlot.py         comma / tab / custom-delimited files
    * RAndS_FSW_ASCII_Plotter.py  Rohde & Schwarz FSW ASCII (.sft) files
    * Touchstone_AutoPlot.py      Touchstone (.sNp) S-parameter files
    * ORBIT_FR/ORBITFR_AutoPlot.py  ORBIT/FR (MVG) data; shown automatically
                                    once that script exists (see
                                    ORBIT_FR/ORBIT_FR_API_REQUIREMENTS.md)

Each tool is started as its own Python process so the launcher stays
responsive and a crash in one tool cannot take the others down.  Output of
every launch is written to a log file (File > Open Launch Log Folder); a tool
that exits with an error within a few seconds is reported with the end of
its log.

Created on 2025-06-28

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later (GNU General Public License v3.0 or later, see LICENSE)
Compatible with: Python 3.8+, qtpy 2.x (PySide6 preferred; PyQt5/PyQt6 work)

# %%% How to edit or update
Background and splash images:
    MAIN_BACKGROUND_IMAGE_PATH   = "assets/load_fall_colors.jpg"
    SPLASH_BACKGROUND_IMAGE_PATH = "assets/load_building.jpg"
Paths are resolved against the folder holding this script (or the frozen
executable), not the current working directory.

To add another plotting tool append one ``LaunchTarget`` to
``LAUNCH_TARGETS``.  The button, the Applications menu entry, the keyboard
shortcut, the About dialog's status list and the icon placeholder are all
generated from that single entry.  ``optional=True`` hides the entry until
its script exists.

Command line:
    python postProcessing_Launcher.py [--splash | --no-splash] [--list]
                                      [--version]
Environment:
    GBO_POSTPROC_PYTHON   Python interpreter used to start the tools
                          (default: the interpreter running the launcher;
                          required for a frozen launcher unless python is on
                          PATH).
    QT_API                Qt binding for qtpy (default here: pyside6).

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.2.0: All secondary GUIs, Help > About, View theme menu, launch fixes
Date: 2026-11-05
    Added:
      * Franks_AutoPlot.py button (FranksProcessed GB timing data).
      * One registry (LAUNCH_TARGETS) drives buttons, the new Applications
        menu (Ctrl+1 ... Ctrl+7), the About status list and the icon
        placeholders; optional entries (ORBIT/FR) appear once their script
        exists.
      * Menu bar: File (scripts folder, launch logs, Exit), Applications,
        View (Light Mode / Dark Mode, remembered between sessions), Help
        (About, Veusz plugin files, project repository, About Qt).
      * Help > About with author, NRAO/GBO, e-mail, licence, copyright and
        runtime details (Show Details), as in the author's other tools.
      * Status bar; per-launch log files; early-exit error report.
    Changed:
      * Removed every "under construction" label; "Rhode" -> "Rohde".
      * Background image and button style kept unchanged; in Dark Mode only
        the translucent panels and window chrome change tint.
      * The modal "Application Launched" pop-up is now a status-bar message.
    Fixed:
      * Splash screen never closed: the worker thread emitted
        progressUpdated/splashFinished while the application listened to
        progress_updated/splash_finished, so the main window never opened
        whenever the splash was shown (always, for a frozen build).
      * Splash image path pointed to load_building.png (only .jpg exists);
        image is now scaled to fit the screen.
      * Button text after launching was restored from a table that did not
        match the button labels (e.g. Touchstone showed the file name);
        each button now keeps its own label.
      * Scripts, icons and images were looked up relative to the current
        working directory, so starting the launcher from another folder
        failed; everything is now relative to the launcher's folder.
      * QT_API was set after the Qt imports (no effect); now set first and
        only as a default.
      * A frozen launcher started the tools with sys.executable (itself);
        it now uses GBO_POSTPROC_PYTHON or python on PATH.
      * Invalid "\\&" escape sequence warning.

# %%%% 0.1.4: Five launch buttons (ATR, FITS, CSV/TSV, R&S FSW, Touchstone)
Date: 2025-06-28
    Splash screen, configurable background image, Nuitka/PyInstaller
    support.
=============================================================================
"""

# %% Import Modules
from __future__ import annotations

import argparse
import datetime
import html
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, IO, List, Optional

# Prefer PySide6 for qtpy, but let an explicit QT_API from the user win.
# Must be set BEFORE qtpy is imported (it was set afterwards in 0.1.4).
os.environ.setdefault("QT_API", "pyside6")

# %%% Qt modules
if getattr(sys, "frozen", False):
    # Compiled executable (PyInstaller / Nuitka): PySide6 directly.
    from PySide6 import __version__ as QT_BINDING_VERSION
    from PySide6.QtCore import (QSettings, QSize, Qt, QThread, QTimer, QUrl,
                                Signal, qVersion)
    from PySide6.QtGui import (QAction, QActionGroup, QColor,
                               QDesktopServices, QFont, QIcon, QKeySequence,
                               QPalette, QPixmap)
    from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
                                   QMessageBox, QProgressBar, QPushButton,
                                   QSizePolicy, QSpacerItem, QSplashScreen,
                                   QVBoxLayout, QWidget)
    QT_BINDING = "PySide6"
else:
    # Development environment: qtpy abstraction layer.
    import qtpy
    from qtpy.QtCore import (QSettings, QSize, Qt, QThread, QTimer, QUrl,
                             Signal, qVersion)
    from qtpy.QtGui import (QColor, QDesktopServices, QFont, QIcon,
                            QKeySequence, QPalette, QPixmap)
    try:  # qtpy >= 2 exposes QAction in QtGui for every binding
        from qtpy.QtGui import QAction, QActionGroup
    except ImportError:  # pragma: no cover - qtpy 1.x with Qt5
        from qtpy.QtWidgets import QAction, QActionGroup
    from qtpy.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
                                QMessageBox, QProgressBar, QPushButton,
                                QSizePolicy, QSpacerItem, QSplashScreen,
                                QVBoxLayout, QWidget)
    QT_BINDING = qtpy.API_NAME
    QT_BINDING_VERSION = getattr(qtpy, "PYSIDE_VERSION", None) or \
        getattr(qtpy, "PYQT_VERSION", None) or "unknown"


# %% Application identity (shown in the title bar, footer and Help > About)
APP_NAME = "GBO Electronics Scientific Plotting Suite"
APP_VERSION = "0.2.0"
__version__ = APP_VERSION
__author__ = "William W. Wallace"
__email__ = "wwallace@nrao.edu"
__company__ = "NRAO / Green Bank Observatory (GBO)"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (C) 2025-2026 William W. Wallace"
REPOSITORY_URL = "https://github.com/Braveheart53/FitsAutoPlot"
ORGANIZATION_NAME = "NRAO-GBO"
ORGANIZATION_DOMAIN = "nrao.edu"
FACILITY_NOTE = ("The National Radio Astronomy Observatory and the Green Bank "
                 "Observatory are facilities of the U.S. National Science "
                 "Foundation operated under cooperative agreement by "
                 "Associated Universities, Inc.")

# %% Configuration Variables
# Background image configuration - relative to the launcher folder.
MAIN_BACKGROUND_IMAGE_PATH = "assets/load_fall_colors.jpg"
SPLASH_BACKGROUND_IMAGE_PATH = "assets/load_building.jpg"
WINDOW_ICON_CANDIDATES = ("assets/app_icon.png", "assets/GBT_1.ico")

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 760          # grows with optional entries; capped to screen
SPLASH_MAX_SIZE = QSize(720, 480)
SPLASH_STEP_SECONDS = 0.25   # five steps -> ~1.25 s total
BUTTON_RESTORE_MS = 2000     # button disabled this long after a launch
EARLY_EXIT_CHECK_MS = 4000   # a tool that dies before this is reported
REAP_INTERVAL_MS = 2000      # finished processes are reaped this often
LOG_TAIL_LINES = 25

# GPL notice shown in Help > About (same wording as the author's other tools)
GPL_URL = "https://www.gnu.org/licenses/"
GPL_NOTICE_PARAGRAPHS = (
    "This program is free software: you can redistribute it and/or modify it "
    "under the terms of the GNU General Public License as published by the "
    "Free Software Foundation, either version 3 of the License, or (at your "
    "option) any later version.",
    "This program is distributed in the hope that it will be useful, but "
    "WITHOUT ANY WARRANTY; without even the implied warranty of "
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU "
    "General Public License for more details.",
    "You should have received a copy of the GNU General Public License along "
    "with this program.  If not, see <https://www.gnu.org/licenses/>.",
)


# %% Paths
def _app_dir() -> Path:
    """Folder holding the launcher script, or the frozen executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = _app_dir()
# PyInstaller one-file builds unpack bundled data to sys._MEIPASS.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))


def resource_path(relative: str) -> Path:
    """
    Resolve a launcher-relative path.

    The folder next to the launcher wins over the PyInstaller bundle so a
    user can drop in updated scripts or images without rebuilding.

    Parameters
    ----------
    relative : str
        Path relative to the launcher folder (forward slashes are fine).

    Returns
    -------
    Path
        First existing candidate, or the launcher-folder candidate if none
        exists (so error messages show the expected location).
    """
    candidates = [APP_DIR / relative, BUNDLE_DIR / relative]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _log_dir() -> Path:
    """Per-user folder for the launch logs (created on demand)."""
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    folder = Path(base) / "GBO_PostProcessing" / "launch_logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# %% Launch registry - one entry per plotting tool
@dataclass(frozen=True)
class LaunchTarget:
    """
    One plotting tool the launcher can start.

    Attributes
    ----------
    key : str
        Short identifier (also used in log file names).
    label : str
        Button and menu text.
    tooltip : str
        Hover text.
    script : str
        Script path relative to the launcher folder.
    icon : str
        Optional button icon relative to the launcher folder.
    file_types : str
        File types handled (shown in Help > About details).
    optional : bool
        Hide the entry until its script exists.
    """

    key: str
    label: str
    tooltip: str
    script: str
    icon: str
    file_types: str
    optional: bool = False

    @property
    def script_path(self) -> Path:
        """Absolute path of the tool's script."""
        return resource_path(self.script)

    @property
    def available(self) -> bool:
        """True when the script exists."""
        return self.script_path.is_file()


LAUNCH_TARGETS: List[LaunchTarget] = [
    LaunchTarget(
        "atr", "Plot GBO Outdoor ATR Files",
        "Launch the GBO Outdoor Antenna Range file plotter\n"
        "for processing and visualizing ATR measurement data "
        "(rectangular and polar pattern plots).",
        "ATR_AutoPlot.py", "assets/atr_icon.png", ".atr"),
    LaunchTarget(
        "fits", "Plot FITS Files",
        "Launch the FITS file plotter for NRAO Green Bank FITS files\n"
        "(BinTable and image HDUs); writes a native Veusz .vszh5 project.",
        "FITS_AutoPlot.py", "assets/fits_icon.png", ".fits, .fit, .fits.gz"),
    LaunchTarget(
        "franks", "Plot Franks Timing Data Files",
        "Launch the Franks AutoPlot for the FranksProcessed GB timing data\n"
        "(YYYY_MM.t00new and time_gbt.dat); writes a native Veusz .vszh5 "
        "project.",
        "Franks_AutoPlot.py", "assets/franks_icon.png",
        ".t00new, time_gbt.dat"),
    LaunchTarget(
        "csv", "Plot Delimited Files",
        "Launch the CSV/TSV file plotter for comma, tab, or\n"
        "custom-separated data files with arbitrary delimiters.",
        "CSV_TSV_AutoPlot.py", "assets/csvicon.png", ".csv, .tsv, .txt"),
    LaunchTarget(
        "rands", "Plot Rohde & Schwarz FSW Files",
        "Launch the Rohde & Schwarz FSW ASCII plotter\n"
        "for spectrum analyzer trace data (.sft).",
        "RAndS_FSW_ASCII_Plotter.py", "assets/rands_icon.png", ".sft"),
    LaunchTarget(
        "snp", "Plot Touchstone Files",
        "Launch the Touchstone AutoPlot for S-parameter files of any port\n"
        "count: magnitude/phase, time domain, time gating, Smith charts.",
        "Touchstone_AutoPlot.py", "assets/snp_icon.png", ".s1p ... .sNp"),
    LaunchTarget(
        "orbitfr", "Plot ORBIT/FR Measurement Data",
        "Launch the ORBIT/FR (MVG 959 Spectrum) data plotter.",
        "ORBIT_FR/ORBITFR_AutoPlot.py", "assets/orbitfr_icon.png",
        "959 Spectrum data", optional=True),
]

# Veusz Tools-menu plugins shipped in the repository (Help menu lists them).
VEUSZ_PLUGIN_FILES = ("FITS_AutoPlot_plugin.py", "Franks_AutoPlot_plugin.py")


def visible_targets() -> List[LaunchTarget]:
    """Targets to show: all required ones, optional ones once present."""
    return [t for t in LAUNCH_TARGETS if not t.optional or t.available]


def qt_text(label: str) -> str:
    """Escape '&' so Qt shows it instead of making a keyboard mnemonic."""
    return label.replace("&", "&&")


# %% Theme handling (View > Light Mode / Dark Mode)
# Only the translucent panels and the window chrome change with the theme;
# the background image and the button gradient are identical in both.
PANEL_STYLES: Dict[str, Dict[str, str]] = {
    "light": {
        "title": """
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
                background-color: rgba(255, 255, 255, 200);
                border-radius: 10px;
                padding: 15px;
            }""",
        "subtitle": """
            QLabel {
                font-size: 16px;
                color: #34495e;
                margin-bottom: 20px;
                background-color: rgba(255, 255, 255, 150);
                border-radius: 8px;
                padding: 10px;
            }""",
        "frame": """
            QFrame#ButtonsFrame {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 10px;
                padding: 0px;
            }""",
        "footer": """
            QLabel {
                font-size: 12px;
                color: #072911;
                background-color: rgba(255, 255, 255, 100);
                border-radius: 5px;
                padding: 8px;
            }""",
    },
    "dark": {
        "title": """
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #ecf0f1;
                margin-bottom: 10px;
                background-color: rgba(28, 33, 40, 210);
                border-radius: 10px;
                padding: 15px;
            }""",
        "subtitle": """
            QLabel {
                font-size: 16px;
                color: #d5dde5;
                margin-bottom: 20px;
                background-color: rgba(28, 33, 40, 170);
                border-radius: 8px;
                padding: 10px;
            }""",
        "frame": """
            QFrame#ButtonsFrame {
                background-color: rgba(28, 33, 40, 180);
                border-radius: 10px;
                padding: 0px;
            }""",
        "footer": """
            QLabel {
                font-size: 12px;
                color: #c8e6c9;
                background-color: rgba(28, 33, 40, 160);
                border-radius: 5px;
                padding: 8px;
            }""",
    },
}

# Menu bar / menus / status bar.  The main window carries a style sheet (the
# background image), and widgets under a style sheet do not pick up a later
# QApplication palette change, so the chrome is styled explicitly here.
CHROME_STYLES: Dict[str, str] = {
    "light": "",
    "dark": """
        QMenuBar { background-color: #2d3238; color: #ecf0f1; }
        QMenuBar::item { background: transparent; padding: 4px 10px; }
        QMenuBar::item:selected { background-color: #3daee9; color: #000000; }
        QMenu { background-color: #2d3238; color: #ecf0f1;
                border: 1px solid #4a525b; }
        QMenu::item:selected { background-color: #3daee9; color: #000000; }
        QMenu::item:disabled { color: #7f8c8d; }
        QMenu::separator { height: 1px; background: #4a525b; margin: 4px 8px; }
        QStatusBar { background-color: #2d3238; color: #ecf0f1; }
        QStatusBar QLabel { color: #ecf0f1; }
        QMessageBox { background-color: #2d3238; }
        QMessageBox QLabel { color: #ecf0f1; }
    """,
}

# Unchanged from 0.1.4 (kept on purpose in both themes).
BUTTON_STYLE = """
    QPushButton {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 #3daee9, stop: 1 #2980b9);
        border: none;
        border-radius: 15px;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 5px 5px;
        text-align: left;
    }
    QPushButton:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 #5dade2, stop: 1 #3498db);
    }
    QPushButton:pressed {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 #2980b9, stop: 1 #1f618d);
        padding-top: 17px;
        padding-left: 27px;
    }
    QPushButton:disabled {
        background: #bdc3c7;
        color: #7f8c8d;
    }
"""


def build_palette(mode: str) -> QPalette:
    """
    Fusion palette for the requested theme.

    Parameters
    ----------
    mode : str
        "light" or "dark".

    Returns
    -------
    QPalette
        Palette to install with ``QApplication.setPalette``.
    """
    app = QApplication.instance()
    if mode != "dark":
        return app.style().standardPalette()
    roles = QPalette.ColorRole
    pal = QPalette()
    window, base, text = QColor(45, 50, 56), QColor(30, 34, 40), QColor(236, 240, 241)
    pal.setColor(roles.Window, window)
    pal.setColor(roles.WindowText, text)
    pal.setColor(roles.Base, base)
    pal.setColor(roles.AlternateBase, window)
    pal.setColor(roles.ToolTipBase, QColor(44, 62, 80))
    pal.setColor(roles.ToolTipText, text)
    pal.setColor(roles.Text, text)
    pal.setColor(roles.Button, window)
    pal.setColor(roles.ButtonText, text)
    pal.setColor(roles.BrightText, QColor(255, 85, 85))
    pal.setColor(roles.Link, QColor(93, 173, 226))
    pal.setColor(roles.Highlight, QColor(61, 174, 233))
    pal.setColor(roles.HighlightedText, QColor(0, 0, 0))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        pal.setColor(disabled, role, QColor(127, 140, 141))
    return pal


# %% Help > About content (kept free of widgets so it can be checked headless)
def build_about_content() -> Dict[str, str]:
    """
    Build the three text blocks of the About dialog.

    Returns
    -------
    dict
        ``text``    rich text: name, version, purpose, author, company,
                    e-mail link, licence, copyright, facility note.
        ``notice``  rich text: the GPL warranty and redistribution notice.
        ``runtime`` plain text for "Show Details": interpreter, Qt and the
                    status of every launch target.
    """
    esc = html.escape
    mail = esc(__email__, quote=True)
    text = (
        "<h3>{name}</h3>"
        "<p>Version <b>{ver}</b> (Semantic Versioning 2.0.0)</p>"
        "<p>Central launcher for the GBO Electronics post-processing and "
        "plotting tools (ATR, FITS, Franks timing data, delimited text, "
        "R&amp;S FSW, Touchstone). Each tool runs in its own process and "
        "writes Veusz projects.</p>"
        "<table cellspacing='4'>"
        "<tr><td><b>Author</b></td><td>{author}</td></tr>"
        "<tr><td><b>Company</b></td><td>{company}</td></tr>"
        "<tr><td><b>Email</b></td><td><a href='mailto:{mail}'>{mail}</a>"
        "</td></tr>"
        "<tr><td><b>Licence</b></td><td>GNU General Public License v3.0 or "
        "later ({lic})</td></tr>"
        "<tr><td><b>Copyright</b></td><td>{copy}</td></tr>"
        "<tr><td><b>Repository</b></td><td><a href='{repo}'>{repo}</a>"
        "</td></tr>"
        "</table>"
        "<p><small>{facility}</small></p>"
    ).format(name=esc(APP_NAME), ver=esc(__version__), author=esc(__author__),
             company=esc(__company__), mail=mail, lic=esc(__license__),
             copy=esc(__copyright__), repo=esc(REPOSITORY_URL, quote=True),
             facility=esc(FACILITY_NOTE))

    paragraphs = []
    for para in GPL_NOTICE_PARAGRAPHS:
        safe = esc(para, quote=False)
        safe = safe.replace("&lt;%s&gt;" % GPL_URL,
                            "&lt;<a href='%s'>%s</a>&gt;" % (GPL_URL, GPL_URL))
        paragraphs.append(safe)
    notice = "<br><br>".join(paragraphs)

    lines = [
        "%s %s" % (APP_NAME, __version__),
        "Python      : %s (%s)" % (platform.python_version(), sys.executable),
        "Qt binding  : %s %s, Qt %s" % (QT_BINDING, QT_BINDING_VERSION,
                                        qVersion()),
        "Platform    : %s" % platform.platform(),
        "Frozen      : %s" % bool(getattr(sys, "frozen", False)),
        "App folder  : %s" % APP_DIR,
        "Launch logs : %s" % _log_dir(),
        "",
        "Tools:",
    ]
    for target in LAUNCH_TARGETS:
        state = "found" if target.available else (
            "not installed (optional)" if target.optional else "MISSING")
        lines.append("  %-8s %-34s %s" % (target.key, target.script, state))
    lines.append("")
    lines.append("Veusz Tools-menu plugins:")
    for name in VEUSZ_PLUGIN_FILES:
        path = resource_path(name)
        lines.append("  %s (%s)" % (path, "found" if path.is_file() else "MISSING"))
    return {"text": text, "notice": notice, "runtime": "\n".join(lines)}


# %% Splash Screen Classes
class SplashScreenThread(QThread):
    """
    Thread that steps the splash-screen progress bar.

    Signals
    -------
    progress_updated : Signal(int, str)
        Progress percentage (0-100) and the status message for that step.
    splash_finished : Signal()
        Emitted when the splash screen should close.
    """

    progress_updated = Signal(int, str)
    splash_finished = Signal()

    def __init__(self, step_seconds: float = SPLASH_STEP_SECONDS):
        """
        Initialize the splash screen thread.

        Parameters
        ----------
        step_seconds : float
            Pause between progress steps, in seconds.
        """
        super().__init__()
        self.step_seconds = step_seconds
        self.progress_steps = [
            ("Initializing application...", 20),
            ("Loading plotting modules...", 40),
            ("Configuring GUI components...", 60),
            ("Preparing user interface...", 80),
            ("Finalizing startup...", 100),
        ]

    def run(self):
        """Emit each progress step, then signal completion."""
        for message, progress in self.progress_steps:
            time.sleep(self.step_seconds)
            self.progress_updated.emit(progress, message)
        self.splash_finished.emit()


class CustomSplashScreen(QSplashScreen):
    """Splash screen with a progress bar and a status label."""

    def __init__(self, pixmap: QPixmap):
        """
        Initialize the custom splash screen.

        Parameters
        ----------
        pixmap : QPixmap
            Background image for the splash screen (already scaled).
        """
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)

        # Progress bar along the bottom edge
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, pixmap.height() - 50,
                                      pixmap.width() - 100, 20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: rgba(255, 255, 255, 180);
                text-align: center;
                font-weight: bold;
                color: black;
            }
            QProgressBar::chunk {
                background-color: #3daee9;
                border-radius: 3px;
            }
        """)

        # Status label just above it
        self.status_label = QLabel(self)
        self.status_label.setGeometry(50, pixmap.height() - 80,
                                      pixmap.width() - 100, 25)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 3px;
                padding: 5px;
                font-weight: bold;
                color: black;
            }
        """)
        self.status_label.setText("Initializing application...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def update_progress(self, value: int, message: str = ""):
        """
        Update the progress bar and status text.

        Parameters
        ----------
        value : int
            Progress percentage (0-100).
        message : str
            Status text for this step.
        """
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)


# %% Main Application Classes
@dataclass
class LaunchRecord:
    """A running (or finished) tool process and its log file."""

    target: LaunchTarget
    process: subprocess.Popen
    log_path: Path
    log_handle: IO
    started: float = field(default_factory=time.monotonic)


class LaunchError(RuntimeError):
    """Raised when a tool cannot be started."""


class ScriptLauncher:
    """Start plotting tools as separate Python processes and track them."""

    def __init__(self):
        """Create an empty process table."""
        self.records: List[LaunchRecord] = []

    @staticmethod
    def interpreter() -> str:
        """
        Python interpreter used for the tools.

        Order: ``GBO_POSTPROC_PYTHON``; the launcher's own interpreter when
        not frozen; otherwise python on PATH.

        Returns
        -------
        str
            Interpreter path.

        Raises
        ------
        LaunchError
            When no interpreter can be found (frozen launcher only).
        """
        configured = os.environ.get("GBO_POSTPROC_PYTHON", "").strip()
        if configured:
            if Path(configured).is_file():
                return configured
            raise LaunchError("GBO_POSTPROC_PYTHON points to a missing file:\n"
                              "%s" % configured)
        if not getattr(sys, "frozen", False):
            return sys.executable
        for name in ("python", "python3", "pythonw"):
            found = shutil.which(name)
            if found:
                return found
        raise LaunchError("No Python interpreter found. Set the environment "
                          "variable GBO_POSTPROC_PYTHON to the python.exe of "
                          "the environment that has the plotting packages.")

    def launch(self, target: LaunchTarget) -> LaunchRecord:
        """
        Start one tool.

        Parameters
        ----------
        target : LaunchTarget
            Tool to start.

        Returns
        -------
        LaunchRecord
            Process handle and log file.

        Raises
        ------
        LaunchError
            Missing script, missing interpreter or a failed spawn.
        """
        script = target.script_path
        if not script.is_file():
            raise LaunchError("Could not find script:\n%s\n\nPlease make sure "
                              "the script is in the launcher folder." % script)
        python = self.interpreter()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = _log_dir() / ("%s_%s.log" % (target.key, stamp))
        handle = open(log_path, "w", encoding="utf-8", errors="replace")
        handle.write("# %s %s launching %s\n# python: %s\n# cwd: %s\n\n" % (
            APP_NAME, __version__, script, python, script.parent))
        handle.flush()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")   # log lines appear promptly
        try:
            process = subprocess.Popen(
                [python, str(script)], cwd=str(script.parent), env=env,
                stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL)
        except Exception as exc:
            handle.close()
            raise LaunchError("Failed to launch %s\n\nError: %s" % (script, exc))
        record = LaunchRecord(target, process, log_path, handle)
        self.records.append(record)
        return record

    def reap(self) -> List[LaunchRecord]:
        """
        Close the logs of finished processes and drop them from the table.

        Returns
        -------
        list of LaunchRecord
            Records that finished since the previous call.
        """
        finished = [r for r in self.records if r.process.poll() is not None]
        for record in finished:
            try:
                record.log_handle.close()
            except Exception:
                pass
            self.records.remove(record)
        return finished

    @property
    def running_count(self) -> int:
        """Number of tool processes still running."""
        return sum(1 for r in self.records if r.process.poll() is None)


def read_log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """Last ``lines`` lines of a launch log (empty string if unreadable)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return ""


class MainWindow(QMainWindow):
    """
    Main application window: one launch button per plotting tool.

    The background image and the button style are those of 0.1.4; the menu
    bar, status bar and theme switch were added in 0.2.0.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.script_launcher = ScriptLauncher()
        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)
        self.theme_mode = str(self.settings.value("theme", "light"))
        if self.theme_mode not in PANEL_STYLES:
            self.theme_mode = "light"
        self.buttons: Dict[str, QPushButton] = {}
        self.launch_actions: Dict[str, QAction] = {}
        self.targets = visible_targets()

        self._setup_window()
        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()
        self._setup_background()
        self._apply_styling()
        self._apply_theme(self.theme_mode, persist=False)

        # Reap finished tools and keep the status bar's count current.
        self._reap_timer = QTimer(self)
        self._reap_timer.timeout.connect(self._reap_processes)
        self._reap_timer.start(REAP_INTERVAL_MS)

    # ----- window chrome -------------------------------------------------
    def _setup_window(self):
        """Configure title, size, position and icon."""
        self.setWindowTitle("%s v%s" % (APP_NAME, APP_VERSION))
        self.setMinimumWidth(WINDOW_WIDTH)
        screen = QApplication.primaryScreen()
        # 60 px per button beyond the six standard tools (optional entries)
        height = WINDOW_HEIGHT + 60 * max(0, len(self.targets) - 6)
        if screen is not None:
            height = min(WINDOW_HEIGHT, screen.availableGeometry().height() - 40)
        self.resize(WINDOW_WIDTH, height)
        self._center_window()
        for candidate in WINDOW_ICON_CANDIDATES:
            icon_path = resource_path(candidate)
            if icon_path.is_file():
                self.setWindowIcon(QIcon(str(icon_path)))
                break

    def _center_window(self):
        """Center the window on the primary screen's available area."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _setup_menu(self):
        """Build File, Applications, View and Help menus."""
        bar = self.menuBar()

        # File
        file_menu = bar.addMenu("&File")
        act = QAction("Open &Scripts Folder", self)
        act.triggered.connect(lambda: self._open_folder(APP_DIR))
        file_menu.addAction(act)
        act = QAction("Open Launch &Log Folder", self)
        act.triggered.connect(lambda: self._open_folder(_log_dir()))
        file_menu.addAction(act)
        file_menu.addSeparator()
        act = QAction("E&xit", self)
        act.setShortcut(QKeySequence("Ctrl+Q"))
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # Applications - mirrors the buttons, with Ctrl+1..9 shortcuts
        apps_menu = bar.addMenu("&Applications")
        for index, target in enumerate(self.targets, start=1):
            act = QAction(qt_text(target.label), self)
            if index <= 9:
                act.setShortcut(QKeySequence("Ctrl+%d" % index))
            act.setStatusTip(target.tooltip.replace("\n", " "))
            act.setEnabled(target.available)
            act.triggered.connect(
                lambda checked=False, key=target.key: self._launch_application(key))
            apps_menu.addAction(act)
            self.launch_actions[target.key] = act

        # View - light / dark display
        view_menu = bar.addMenu("&View")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.act_light = QAction("&Light Mode", self, checkable=True)
        self.act_dark = QAction("&Dark Mode", self, checkable=True)
        for action, mode in ((self.act_light, "light"), (self.act_dark, "dark")):
            group.addAction(action)
            action.triggered.connect(
                lambda checked=False, m=mode: self._apply_theme(m))
            view_menu.addAction(action)

        # Help
        help_menu = bar.addMenu("&Help")
        act = QAction("&About %s..." % APP_NAME, self)
        act.triggered.connect(self.show_about)
        help_menu.addAction(act)
        act = QAction("&Veusz Plugin Files...", self)
        act.triggered.connect(self._show_plugin_info)
        help_menu.addAction(act)
        act = QAction("Project &Repository", self)
        act.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(REPOSITORY_URL)))
        help_menu.addAction(act)
        help_menu.addSeparator()
        act = QAction("About &Qt", self)
        act.triggered.connect(QApplication.aboutQt)
        help_menu.addAction(act)

    def _setup_statusbar(self):
        """Status bar with a permanent running-tools counter."""
        self.running_label = QLabel("Running tools: 0")
        self.statusBar().addPermanentWidget(self.running_label)
        missing = [t.script for t in self.targets if not t.available]
        if missing:
            self.statusBar().showMessage("Missing scripts: %s" % ", ".join(missing))
        else:
            self.statusBar().showMessage("Ready", 5000)

    # ----- central widget ---------------------------------------------------
    def _setup_ui(self):
        """Set up the user interface layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(50, 30, 50, 20)

        self._create_title_section(main_layout)
        self._create_buttons_section(main_layout)
        self._create_footer_section(main_layout)

    def _create_title_section(self, parent_layout: QVBoxLayout):
        """
        Create the title and subtitle labels.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Parent layout to add the title section to.
        """
        self.title_label = QLabel(APP_NAME)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label = QLabel(
            "Professional Scientific Data Visualization Suite")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parent_layout.addWidget(self.title_label)
        parent_layout.addWidget(self.subtitle_label)

    def _create_buttons_section(self, parent_layout: QVBoxLayout):
        """
        Create one styled launch button per visible target.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Parent layout to add the buttons section to.
        """
        self.buttons_frame = QFrame()
        self.buttons_frame.setObjectName("ButtonsFrame")
        buttons_layout = QVBoxLayout(self.buttons_frame)
        buttons_layout.setSpacing(10)

        for target in self.targets:
            tooltip = target.tooltip
            if not target.available:
                tooltip += "\n\nScript not found: %s" % target.script_path
            button = self._create_styled_button(qt_text(target.label),
                                                tooltip, target.icon)
            button.setEnabled(target.available)
            button.clicked.connect(
                lambda checked=False, key=target.key: self._launch_application(key))
            self.buttons[target.key] = button
            buttons_layout.addWidget(button)

        parent_layout.addWidget(self.buttons_frame)

    @staticmethod
    def _create_styled_button(text: str, tooltip: str,
                              icon_path: str) -> QPushButton:
        """
        Create a consistently styled button (style unchanged from 0.1.4).

        Parameters
        ----------
        text : str
            Button text.
        tooltip : str
            Button tooltip text.
        icon_path : str
            Path to the button icon, relative to the launcher folder.

        Returns
        -------
        QPushButton
            Configured button widget.
        """
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.setMinimumHeight(50)
        button.setMaximumHeight(70)
        icon_file = resource_path(icon_path)
        if icon_file.is_file():
            button.setIcon(QIcon(str(icon_file)))
            button.setIconSize(QSize(32, 32))
        button.setStyleSheet(BUTTON_STYLE)
        return button

    def _create_footer_section(self, parent_layout: QVBoxLayout):
        """
        Create the footer with version and author information.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Parent layout to add the footer section to.
        """
        spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum,
                             QSizePolicy.Policy.Expanding)
        parent_layout.addItem(spacer)
        self.footer_label = QLabel(
            "Version %s | Scientific Computing Suite | NRAO / Green Bank "
            "Observatory\nAuthor: %s" % (APP_VERSION, __author__))
        self.footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parent_layout.addWidget(self.footer_label)

    def _setup_background(self):
        """Set the background image (unchanged look; absolute path now)."""
        background = resource_path(MAIN_BACKGROUND_IMAGE_PATH)
        self.setObjectName("MainWindow")
        self._base_style = ""
        if background.is_file():
            # Object-name selector so child widgets do not inherit it.
            self._base_style = """
                #MainWindow {
                    border-image: url("%s") 0 0 0 0 stretch stretch;
                }
            """ % background.as_posix()
        else:
            print("Warning: background image not found: %s" % background)

    def _apply_styling(self):
        """Application font, window colour and tooltip style."""
        self.setFont(QFont("Segoe UI", 10))
        self._base_style += """
            QMainWindow {
                background-color: #ecf0f1;
            }
            QToolTip {
                background-color: #2c3e50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
        """
        self.setStyleSheet(self._base_style)

    def _apply_theme(self, mode: str, persist: bool = True):
        """
        Switch between light and dark display.

        Parameters
        ----------
        mode : str
            "light" or "dark".
        persist : bool
            Remember the choice for the next start.
        """
        mode = mode if mode in PANEL_STYLES else "light"
        self.theme_mode = mode
        QApplication.instance().setPalette(build_palette(mode))
        self.setStyleSheet(self._base_style + CHROME_STYLES[mode])
        styles = PANEL_STYLES[mode]
        self.title_label.setStyleSheet(styles["title"])
        self.subtitle_label.setStyleSheet(styles["subtitle"])
        self.buttons_frame.setStyleSheet(styles["frame"])
        self.footer_label.setStyleSheet(styles["footer"])
        self.act_light.setChecked(mode == "light")
        self.act_dark.setChecked(mode == "dark")
        if persist:
            try:
                self.settings.setValue("theme", mode)
            except Exception:
                pass

    # ----- actions -------------------------------------------------------------
    def _target(self, key: str) -> Optional[LaunchTarget]:
        """Look up a launch target by key."""
        return next((t for t in LAUNCH_TARGETS if t.key == key), None)

    def _launch_application(self, script_key: str):
        """
        Launch the tool identified by ``script_key``.

        Parameters
        ----------
        script_key : str
            Key of a ``LaunchTarget`` (e.g. 'atr', 'fits', 'franks').
        """
        target = self._target(script_key)
        if target is None:
            QMessageBox.warning(self, "Configuration Error",
                                "No script configured for: %s" % script_key)
            return
        try:
            record = self.script_launcher.launch(target)
        except LaunchError as exc:
            QMessageBox.critical(self, "Launch Error", str(exc))
            return

        # Brief visual feedback; the label itself never changes.
        button = self.buttons.get(script_key)
        if button is not None:
            button.setEnabled(False)
            QTimer.singleShot(BUTTON_RESTORE_MS,
                              lambda b=button: b.setEnabled(True))
        self.statusBar().showMessage(
            "Started %s (PID %d) - it should appear shortly. Log: %s" % (
                target.label, record.process.pid, record.log_path.name), 10000)
        self._update_running_label()
        QTimer.singleShot(EARLY_EXIT_CHECK_MS,
                          lambda r=record: self._check_early_exit(r))

    def _check_early_exit(self, record: LaunchRecord):
        """Report a tool that already exited with an error."""
        code = record.process.poll()
        if code is None or code == 0:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Tool Exited")
        box.setText("%s exited with code %d shortly after starting." % (
            record.target.label, code))
        box.setInformativeText("The end of its log is under Show Details.\n"
                               "Full log: %s" % record.log_path)
        box.setDetailedText(read_log_tail(record.log_path))
        box.exec()

    def _reap_processes(self):
        """Close the logs of finished tools; refresh the counter."""
        self.script_launcher.reap()
        self._update_running_label()

    def _update_running_label(self):
        """Show how many tools are running."""
        self.running_label.setText(
            "Running tools: %d" % self.script_launcher.running_count)

    def _open_folder(self, folder: Path):
        """Open a folder in the system file browser."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _show_plugin_info(self):
        """Explain how to load the Veusz Tools-menu plugins."""
        rows = []
        for name in VEUSZ_PLUGIN_FILES:
            path = resource_path(name)
            rows.append("<li><code>%s</code>%s</li>" % (
                html.escape(str(path)),
                "" if path.is_file() else " <b>(missing)</b>"))
        box = QMessageBox(self)
        box.setWindowTitle("Veusz Plugin Files")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            "<p>FITS and Franks AutoPlot can also run inside Veusz from "
            "<b>Tools &gt; NRAO / FitsAutoPlot</b>. In Veusz choose "
            "<b>Edit &gt; Preferences &gt; Plugins &gt; Add</b> and select:</p>"
            "<ul>%s</ul><p>Restart Veusz afterwards. The plugins write a "
            "separate native .vszh5 project and leave the open Veusz "
            "document untouched.</p>" % "".join(rows))
        box.exec()

    def build_about_box(self) -> QMessageBox:
        """The About dialog: author, version, GPL notice; runtime details."""
        content = build_about_content()
        box = QMessageBox(self)
        box.setWindowTitle("About %s" % APP_NAME)
        box.setIcon(QMessageBox.Icon.Information)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(content["text"])
        box.setInformativeText(content["notice"])
        box.setDetailedText(content["runtime"])
        box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        return box

    def show_about(self):
        """Show the About dialog modally."""
        box = self.build_about_box()
        box.exec()

    def closeEvent(self, event):  # noqa: N802 - Qt API
        """Leave running tools alone; just close their log handles."""
        self._reap_timer.stop()
        for record in list(self.script_launcher.records):
            try:
                record.log_handle.flush()
            except Exception:
                pass
        super().closeEvent(event)


# %% Application Class
class PlottingSuiteApplication(QApplication):
    """
    Application object with optional splash screen.

    The splash is shown when the splash image exists (and always for a
    frozen build); ``--no-splash`` skips it.
    """

    def __init__(self, argv):
        """
        Initialize the application.

        Parameters
        ----------
        argv : list
            Command line arguments.
        """
        super().__init__(argv)
        self.splash = None
        self.main_window = None
        self.splash_thread = None

        self.setApplicationName(APP_NAME)
        self.setApplicationVersion(APP_VERSION)
        self.setOrganizationName(ORGANIZATION_NAME)
        self.setOrganizationDomain(ORGANIZATION_DOMAIN)
        self.setStyle("Fusion")

    def show_splash_screen(self) -> bool:
        """
        Display the splash screen and start its progress thread.

        Returns
        -------
        bool
            True once the splash screen is showing.
        """
        splash_file = resource_path(SPLASH_BACKGROUND_IMAGE_PATH)
        pixmap = QPixmap(str(splash_file)) if splash_file.is_file() else QPixmap()
        if pixmap.isNull():
            print("Warning: splash image not found at %s; using a plain "
                  "splash screen" % splash_file)
            pixmap = QPixmap(400, 300)
            pixmap.fill(QColor(0, 0, 139))
        else:
            pixmap = pixmap.scaled(SPLASH_MAX_SIZE,
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)

        self.splash = CustomSplashScreen(pixmap)
        self.splash.show()
        self.processEvents()

        self.splash_thread = SplashScreenThread()
        self.splash_thread.progress_updated.connect(self.splash.update_progress)
        self.splash_thread.splash_finished.connect(self._on_splash_finished)
        self.splash_thread.start()
        return True

    def _on_splash_finished(self):
        """Open the main window and close the splash screen."""
        self.main_window = MainWindow()
        self.main_window.show()
        if self.splash:
            self.splash.finish(self.main_window)
            self.splash = None
        if self.splash_thread:
            self.splash_thread.quit()
            self.splash_thread.wait()
            self.splash_thread = None

    def run_without_splash(self):
        """Open the main window directly."""
        self.main_window = MainWindow()
        self.main_window.show()


# %% Utility Functions
def setup_qt_plugins():
    """Point Qt at PySide6's platform plugins (frozen builds)."""
    try:
        import PySide6
        plugin_path = os.path.join(os.path.dirname(PySide6.__file__),
                                   "plugins", "platforms")
        if os.path.exists(plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    except ImportError:
        pass


def ensure_assets_directory():
    """
    Create the assets folder and a placeholder for every missing image.

    Placeholders tell a developer which file name each button, the window
    and the splash screen look for.  Skipped for frozen builds.
    """
    if getattr(sys, "frozen", False):
        return
    assets_dir = APP_DIR / "assets"
    try:
        assets_dir.mkdir(exist_ok=True)
    except OSError:
        return
    wanted = [(Path(MAIN_BACKGROUND_IMAGE_PATH).name,
               "Place main window background image here"),
              (Path(SPLASH_BACKGROUND_IMAGE_PATH).name,
               "Place splash screen background image here"),
              ("app_icon.png", "Place application icon here")]
    wanted += [(Path(t.icon).name, "Place %s button icon here" % t.label)
               for t in LAUNCH_TARGETS]
    for filename, description in wanted:
        placeholder = assets_dir / ("%s.placeholder" % filename)
        if (assets_dir / filename).exists() or placeholder.exists():
            continue
        try:
            with open(placeholder, "w", encoding="utf-8") as fh:
                fh.write("%s\nRename this file to '%s' after adding your "
                         "image.\n" % (description, filename))
        except OSError:
            pass


def check_script_dependencies() -> bool:
    """
    Print any required tool script that is missing.

    Returns
    -------
    bool
        True if every required script exists.
    """
    missing = [t for t in LAUNCH_TARGETS if not t.optional and not t.available]
    if missing:
        print("Warning: missing script files:")
        for target in missing:
            print("  - %s: %s" % (target.key, target.script_path))
        print("\nPlease ensure all script files are in the launcher folder.")
        return False
    return True


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    """Command-line options (unknown Qt options are passed through)."""
    parser = argparse.ArgumentParser(description=APP_NAME, add_help=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--splash", action="store_true",
                       help="always show the splash screen")
    group.add_argument("--no-splash", action="store_true",
                       help="never show the splash screen")
    parser.add_argument("--list", action="store_true",
                        help="print the launch targets and exit")
    parser.add_argument("--version", action="store_true",
                        help="print the version and exit")
    args, _unknown = parser.parse_known_args(argv[1:])
    return args


# %% Main Execution
def main(argv: Optional[List[str]] = None) -> int:
    """
    Main application entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command line (default ``sys.argv``).

    Returns
    -------
    int
        Process exit code.
    """
    argv = list(sys.argv if argv is None else argv)
    args = parse_arguments(argv)
    if args.version:
        print("%s %s\n%s\nAuthor: %s <%s>, %s\nLicence: %s" % (
            APP_NAME, APP_VERSION, __copyright__, __author__, __email__,
            __company__, __license__))
        return 0
    if args.list:
        for target in LAUNCH_TARGETS:
            state = "found" if target.available else (
                "optional, not installed" if target.optional else "MISSING")
            print("%-8s %-36s %s" % (target.key, target.script, state))
        return 0

    if getattr(sys, "frozen", False):
        setup_qt_plugins()

    app = PlottingSuiteApplication(argv)
    ensure_assets_directory()
    check_script_dependencies()

    show_splash = not args.no_splash and (
        getattr(sys, "frozen", False) or args.splash
        or resource_path(SPLASH_BACKGROUND_IMAGE_PATH).is_file())
    if show_splash:
        app.show_splash_screen()
    else:
        app.run_without_splash()

    run = getattr(app, "exec", None) or getattr(app, "exec_")
    return run()


# %% Entry Point
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - last-resort report
        print("Application failed to start: %s" % exc)
        sys.exit(1)
