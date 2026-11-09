# -*- coding: utf-8 -*-
"""
=============================================================================
_suite_gui.py
-----------------------------------------------------------------------------
# %% Header Info
--------

Shared window chrome for every GUI of the GBO Electronics Scientific Plotting
Suite (ATR, CSV/TSV, R&S FSW, Touchstone, FITS and Franks AutoPlot):

    * one light / dark theme for the whole suite, stored where the launcher
      stores it (QSettings "NRAO-GBO" / "GBO Electronics Scientific Plotting
      Suite", key "theme"), so a theme chosen in any tool or in the launcher
      is used by every tool started afterwards;
    * View > Light Mode / Dark Mode and Help > About (author, NRAO / GBO,
      e-mail, licence, library versions) menus;
    * a dark style sheet for dialogs that live inside another program (the
      Veusz plugins), so they never re-theme their host;
    * the pyqtgraph "Plots" tab, "Open in Veusz" and "Export PDF / PNG
      (Veusz)" helpers.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later
Compatible with: Python 3.8+, qtpy (PySide6 preferred; PyQt5/PyQt6 work)

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.1.1: Header template and docstrings (no functional change).
Date: 2026-11-09
#              * Header in the author's template: author contact lines and,
#                under this entry, Function Descriptions, Variable
#                Descriptions and More Info.
#              * Docstrings for every function, class and method.  No change
#                to what the program does.
# %%%%% Function Descriptions
        saved_theme: The suite's stored theme ('light' or 'dark').
        save_theme: Store the suite theme (shared with the launcher).
        build_palette: Fusion palette for 'light' or 'dark' (the launcher's colours).
        apply_theme: Fusion style and the theme's palette for the whole application.
        dialog_stylesheet: Style sheet that themes ONE dialog and its children (plugins in Veusz).
        add_view_menu: View > Light Mode / Dark Mode (exclusive); returns (light, dark) actions.
        add_file_menu: File menu: the given (text, slot, shortcut) entries, then Exit (Ctrl+Q).
        _module_version: ``__version__`` of module ``name``; 'installed' when it has none, 'not installed' when it cannot be imported.
        about_details: Plain-text runtime details for the About dialog.
        show_about: Help > About in the suite's common layout.
        add_help_menu: Help > About <name> and About Qt.
        make_plot_panel: (widget, panel): the pyqtgraph ProjectPlotPanel, or an explanation label (panel None) when pyqtgraph is missing; processing and saving still work.
        find_veusz: Veusz program: VEUSZ_EXECUTABLE, then vszh5_writer.find_veusz_exe().
        export_with_veusz: PDF (one bookmark per page) and per-page PNG files of a saved project, rendered by Veusz through vszh5_writer.render_pages; returns the files.
        open_in_veusz: Open a saved project in the Veusz program (optional viewer).
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION = "0.1.1"
        SUITE_ORG = "NRAO-GBO"
        SUITE_APP = "GBO Electronics Scientific Plotting Suite"
        THEME_KEY = "theme"
        AUTHOR = "William W. Wallace"
        EMAIL = "wwallace@nrao.edu"
        COMPANY = "NRAO / Green Bank Observatory (GBO)"
        LICENCE = "GPL-3.0-or-later"
        COPYRIGHT = "Copyright (C) 2025-2026 William W. Wallace"
        FACILITY_NOTE: NRAO / GBO statement in the About box
        GPL_NOTICE: licence notice in the About box
# %%%%% More Info
    Shared by the launcher and every program (theme, menus, About box, Veusz helpers).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.1.0: First version - suite-wide theme, menus, About, Plots tab.
Date: 2026-11-05
=============================================================================
"""
# %% Import Modules
from __future__ import annotations

import html
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

os.environ.setdefault("QT_API", "pyside6")

from qtpy.QtCore import QSettings, Qt  # noqa: E402
from qtpy.QtGui import QColor, QKeySequence, QPalette  # noqa: E402
try:
    from qtpy.QtGui import QAction, QActionGroup  # noqa: E402
except ImportError:  # pragma: no cover - qtpy 1.x with Qt5
    from qtpy.QtWidgets import QAction, QActionGroup  # type: ignore  # noqa: E402
from qtpy.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

VERSION = "0.1.1"
__version__ = VERSION

# %% Suite identity (same values as postProcessing_Launcher.py)
SUITE_ORG = "NRAO-GBO"
SUITE_APP = "GBO Electronics Scientific Plotting Suite"
THEME_KEY = "theme"
AUTHOR = "William W. Wallace"
EMAIL = "wwallace@nrao.edu"
COMPANY = "NRAO / Green Bank Observatory (GBO)"
LICENCE = "GPL-3.0-or-later"
COPYRIGHT = "Copyright (C) 2025-2026 William W. Wallace"
FACILITY_NOTE = ("The National Radio Astronomy Observatory and the Green Bank "
                 "Observatory are facilities of the U.S. National Science "
                 "Foundation operated under cooperative agreement by "
                 "Associated Universities, Inc.")
GPL_NOTICE = ("This program is free software: you can redistribute it and/or "
              "modify it under the terms of the GNU General Public License as "
              "published by the Free Software Foundation, either version 3 of "
              "the License, or (at your option) any later version. It comes "
              "with ABSOLUTELY NO WARRANTY. See "
              "<a href='https://www.gnu.org/licenses/'>https://www.gnu.org/licenses/</a>.")


# %% Theme
def saved_theme(default: str = "dark") -> str:
    """The suite's stored theme ('light' or 'dark')."""
    try:
        value = str(QSettings(SUITE_ORG, SUITE_APP).value(THEME_KEY, default))
    except Exception:
        value = default
    return value if value in ("light", "dark") else default


def save_theme(mode: str) -> None:
    """Store the suite theme (shared with the launcher)."""
    try:
        QSettings(SUITE_ORG, SUITE_APP).setValue(THEME_KEY, mode)
    except Exception:
        pass


def build_palette(mode: str) -> QPalette:
    """Fusion palette for 'light' or 'dark' (the launcher's colours)."""
    app = QApplication.instance()
    if mode != "dark":
        return app.style().standardPalette()
    roles = QPalette.ColorRole
    pal = QPalette()
    window, base, text = QColor(45, 50, 56), QColor(30, 34, 40), QColor(236, 240, 241)
    for role, colour in ((roles.Window, window), (roles.WindowText, text),
                         (roles.Base, base), (roles.AlternateBase, window),
                         (roles.ToolTipBase, QColor(44, 62, 80)),
                         (roles.ToolTipText, text), (roles.Text, text),
                         (roles.Button, window), (roles.ButtonText, text),
                         (roles.BrightText, QColor(255, 85, 85)),
                         (roles.Link, QColor(93, 173, 226)),
                         (roles.Highlight, QColor(61, 174, 233)),
                         (roles.HighlightedText, QColor(0, 0, 0))):
        pal.setColor(role, colour)
    try:
        pal.setColor(roles.PlaceholderText, QColor(150, 155, 160))
    except AttributeError:  # pragma: no cover - Qt < 5.12
        pass
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        pal.setColor(disabled, role, QColor(127, 140, 141))
    return pal


def apply_theme(mode: str, persist: bool = True) -> str:
    """Fusion style and the theme's palette for the whole application."""
    mode = "dark" if mode == "dark" else "light"
    app = QApplication.instance()
    if app is not None:
        app.setStyle("Fusion")
        app.setPalette(build_palette(mode))
    if persist:
        save_theme(mode)
    return mode


def dialog_stylesheet(mode: str) -> str:
    """Style sheet that themes ONE dialog and its children (plugins in Veusz).

    A palette alone is not enough there: the host's style may ignore it, and
    changing the application palette would re-theme Veusz itself.
    """
    if mode != "dark":
        return ""
    return """
        QWidget { background-color: #2d3238; color: #ecf0f1; }
        QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QComboBox,
        QSpinBox, QDoubleSpinBox, QTableWidget {
            background-color: #1e2228; color: #ecf0f1;
            border: 1px solid #4a525b; selection-background-color: #3daee9; }
        QPushButton { background-color: #3a4048; border: 1px solid #4a525b;
                      padding: 4px 10px; }
        QPushButton:hover { background-color: #46505a; }
        QPushButton:disabled, QLabel:disabled, QCheckBox:disabled { color: #7f8c8d; }
        QGroupBox { border: 1px solid #4a525b; margin-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        QProgressBar { border: 1px solid #4a525b; text-align: center; }
        QProgressBar::chunk { background-color: #3daee9; }
        QHeaderView::section { background-color: #3a4048; color: #ecf0f1; }
    """


# %% Menus
def add_view_menu(window: Any, on_change: Callable[[str], None],
                  current: str) -> Tuple[Any, Any]:
    """View > Light Mode / Dark Mode (exclusive); returns (light, dark) actions."""
    menu = window.menuBar().addMenu("&View")
    group = QActionGroup(window)
    group.setExclusive(True)
    light = QAction("&Light Mode", window, checkable=True)
    dark = QAction("&Dark Mode", window, checkable=True)
    for action, mode in ((light, "light"), (dark, "dark")):
        group.addAction(action)
        action.setChecked(mode == current)
        action.triggered.connect(lambda checked=False, m=mode: on_change(m))
        menu.addAction(action)
    return light, dark


def add_file_menu(window: Any, actions: Iterable[Tuple[str, Callable[[], Any], str]] = ()
                  ) -> Any:
    """File menu: the given (text, slot, shortcut) entries, then Exit (Ctrl+Q)."""
    menu = window.menuBar().addMenu("&File")
    for text, slot, shortcut in actions:
        action = QAction(text, window)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
    if actions:
        menu.addSeparator()
    exit_action = QAction("E&xit", window)
    exit_action.setShortcut(QKeySequence("Ctrl+Q"))
    exit_action.triggered.connect(window.close)
    menu.addAction(exit_action)
    return menu


def _module_version(name: str) -> str:
    """``__version__`` of module ``name``; 'installed' when it has none,
    'not installed' when it cannot be imported.
    """
    try:
        module = sys.modules.get(name) or __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return "not installed"


def about_details(name: str, version: str, modules: Sequence[str] = ()) -> str:
    """Plain-text runtime details for the About dialog."""
    try:
        import qtpy
        binding = "%s %s" % (qtpy.API_NAME, qtpy.QT_VERSION)
    except Exception:
        binding = "unknown"
    lines = ["%s %s" % (name, version),
             "Python     : %s (%s)" % (platform.python_version(), sys.executable),
             "Qt         : %s" % binding]
    for module in ("numpy", "pyqtgraph", "h5py", "vszh5_writer", "_autoplot_native") + tuple(modules):
        lines.append("%-11s: %s" % (module, _module_version(module)
                                     if module != "_autoplot_native" else
                                     getattr(sys.modules.get(module), "VERSION", "not loaded")))
    lines.append("Platform   : %s" % platform.platform())
    lines.append("Theme      : %s (shared with the launcher)" % saved_theme())
    return "\n".join(lines)


def show_about(parent: Any, name: str, version: str, description: str,
               modules: Sequence[str] = ()) -> None:
    """Help > About in the suite's common layout."""
    esc = html.escape
    text = ("<h3>%s</h3><p>Version <b>%s</b> (Semantic Versioning 2.0.0)</p><p>%s</p>"
            "<table cellspacing='4'>"
            "<tr><td><b>Author</b></td><td>%s</td></tr>"
            "<tr><td><b>Company</b></td><td>%s</td></tr>"
            "<tr><td><b>Email</b></td><td><a href='mailto:%s'>%s</a></td></tr>"
            "<tr><td><b>Licence</b></td><td>GNU General Public License v3.0 or later "
            "(%s)</td></tr><tr><td><b>Copyright</b></td><td>%s</td></tr></table>"
            "<p><small>%s</small></p>" % (
                esc(name), esc(version), esc(description), esc(AUTHOR), esc(COMPANY),
                EMAIL, EMAIL, esc(LICENCE), esc(COPYRIGHT), esc(FACILITY_NOTE)))
    box = QMessageBox(parent)
    box.setWindowTitle("About %s" % name)
    box.setIcon(QMessageBox.Icon.Information)
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setText(text)
    box.setInformativeText(GPL_NOTICE)
    box.setDetailedText(about_details(name, version, modules))
    box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    box.exec()


def add_help_menu(window: Any, name: str, version: str, description: str,
                  modules: Sequence[str] = ()) -> Any:
    """Help > About <name> and About Qt."""
    menu = window.menuBar().addMenu("&Help")
    about = QAction("&About %s..." % name, window)
    about.triggered.connect(lambda: show_about(window, name, version, description, modules))
    menu.addAction(about)
    about_qt = QAction("About &Qt", window)
    about_qt.triggered.connect(QApplication.aboutQt)
    menu.addAction(about_qt)
    return menu


# %% Plots tab and Veusz
def make_plot_panel(dark: bool) -> Tuple[Any, Optional[Any]]:
    """(widget, panel): the pyqtgraph ProjectPlotPanel, or an explanation label
    (panel None) when pyqtgraph is missing; processing and saving still work."""
    try:
        import _autoplot_pyqtgraph as autoplot_pg
        autoplot_pg.configure(antialias=True, wheel_needs_ctrl=True)
        panel = autoplot_pg.ProjectPlotPanel(dark=dark)
        return panel, panel
    except Exception as exc:  # pragma: no cover - pyqtgraph missing or broken
        label = QLabel("Interactive plots need pyqtgraph:\n\n    mamba install pyqtgraph\n\n"
                       "(%s: %s)\n\nProcessing and saving the Veusz project work "
                       "without it." % (type(exc).__name__, exc))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label, None


def find_veusz() -> str:
    """Veusz program: VEUSZ_EXECUTABLE, then vszh5_writer.find_veusz_exe()."""
    configured = os.environ.get("VEUSZ_EXECUTABLE", "").strip().strip('"')
    if configured and os.path.isfile(configured):
        return configured
    try:
        from vszh5_writer import find_veusz_exe
        found = find_veusz_exe(env_vars=("VEUSZ_EXE", "QUICKSYN_VEUSZ"))
        if found:
            return found
    except Exception:
        pass
    return shutil.which("veusz") or ""


def export_with_veusz(parent: Any, path: str, log: Optional[Callable[[str], None]] = None
                      ) -> list:
    """
    PDF (one bookmark per page) and per-page PNG files of a saved project,
    rendered by Veusz through vszh5_writer.render_pages; returns the files.

    Veusz is needed for this step only (not for writing the .vszh5).
    """
    from qtpy.QtGui import QCursor
    try:
        from vszh5_writer import render_pages
    except Exception as exc:
        QMessageBox.warning(parent, "Export", "vszh5_writer is not available (%s)." % exc)
        return []
    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
    try:
        files = render_pages(path, dpi=150, pdf=True, png_pages="all",
                             veusz_exe=find_veusz(), png_background="white")
    except Exception as exc:
        QApplication.restoreOverrideCursor()
        QMessageBox.warning(parent, "Export",
                            "Veusz could not render %s:\n%s\n\nInstall Veusz or set "
                            "VEUSZ_EXECUTABLE." % (os.path.basename(path), exc))
        return []
    QApplication.restoreOverrideCursor()
    if log is not None:
        log("Exported %d file(s) next to %s" % (len(files), path))
    return list(files)


def open_in_veusz(parent: Any, path: str) -> bool:
    """Open a saved project in the Veusz program (optional viewer)."""
    exe = find_veusz()
    command = [exe, path] if exe else [sys.executable, "-m", "veusz", path]
    try:
        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {"start_new_session": True}
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **kwargs)
        return True
    except Exception as exc:
        QMessageBox.warning(parent, "Open in Veusz",
                            "Veusz could not be started (%s).\nInstall Veusz or set "
                            "VEUSZ_EXECUTABLE. Veusz is not needed to create the file." % exc)
        return False
