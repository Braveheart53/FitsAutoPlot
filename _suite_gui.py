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

# %%%% 0.1.3: release_after_clear() and the Clear tool tip for every program.
Date: 2026-11-16
#              * NEW release_after_clear(log): deletes the widgets and threads
#                given to deleteLater(), then _autoplot_native.release_memory.
#              * NEW CLEAR_TOOLTIP: what Clear removes, and that added files
#                join the ones already read.
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
        release_after_clear: Finish a Clear (0.1.3): delete the widgets and threads already given to deleteLater() (the old plots) now, then give the freed memory back to the o...
        GapOptionsBox (class): "Gaps" options for a tool's plots (0.1.2), the same controls as FITS and Franks AutoPlot.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION = "0.1.3"
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
        CLEAR_TOOLTIP: tool tip of every program's Clear button
# %%%%% More Info
    Shared by the launcher and every program (theme, menus, About box, Veusz
    helpers, GapOptionsBox, release_after_clear and CLEAR_TOOLTIP).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.1.2: GapOptionsBox for every plot generator.
Date: 2026-11-16
#              * NEW GapOptionsBox (QGroupBox "Gaps"): the FITS / Franks gap
#                controls for ATR, CSV/TSV, R&S and Touchstone - line breaks
#                (K, fixed small gap), axis breaks (n, fixed large gap), the
#                non-time-axis option; every control has a tool tip;
#                time_layout() gives the _autoplot_native TimeLayout.
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
        GapOptionsBox (class): "Gaps" options for a tool's plots (0.1.2), the same controls as FITS and Franks AutoPlot.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION = "0.1.2"
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
    Shared by the launcher and every program (theme, menus, About box, Veusz
    helpers, GapOptionsBox: the gap options of ATR, CSV/TSV, R&S, Touchstone).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
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

from qtpy.QtCore import QCoreApplication, QEvent, QSettings, Qt  # noqa: E402
from qtpy.QtGui import QColor, QKeySequence, QPalette  # noqa: E402
try:
    from qtpy.QtGui import QAction, QActionGroup  # noqa: E402
except ImportError:  # pragma: no cover - qtpy 1.x with Qt5
    from qtpy.QtWidgets import QAction, QActionGroup  # type: ignore  # noqa: E402
from qtpy.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFormLayout,  # noqa: E402
                            QGroupBox, QLabel, QLineEdit, QMessageBox)

VERSION = "0.1.3"
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


# %% Gap options box (0.1.2): every plot generator
def release_after_clear(log: Optional[Callable[[str], None]] = None) -> dict:
    """Finish a Clear (0.1.3): delete the widgets and threads already given
    to deleteLater() (the old plots) now, then give the freed memory back to
    the operating system (_autoplot_native.release_memory).  ``log`` gets the
    RSS before and after; returns release_memory()'s result."""
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    from _autoplot_native import release_memory
    return release_memory(log)


CLEAR_TOOLTIP = ("Remove every file from the list together with its data: the project "
                 "(nothing cleared is saved later), the Plots tab and the previews; the "
                 "memory goes back to the operating system.  The files on disk are not "
                 "touched.  Without Clear, added files are processed together with the "
                 "ones already read.")


class GapOptionsBox(QGroupBox):
    """
    "Gaps" options for a tool's plots (0.1.2), the same controls as FITS and
    Franks AutoPlot.

    A small gap (a trace's step > K x its median step, or a fixed time) can
    break the line / trace; a large gap (no trace has data for > n x the
    small-gap threshold, or a fixed time) can break the x axis. Each is a
    check box, so the plots get line breaks, axis breaks, both or neither.
    The rule also works on sorted x data that are not time (frequency,
    angle, index), with the step also large on a log scale so a log-spaced
    sweep is not cut. time_layout() gives the _autoplot_native TimeLayout
    that the project (and its Plots tab) uses.
    """

    def __init__(self, parent: Any = None) -> None:
        """Build the check boxes, spin boxes and text boxes (defaults: both
        breaks on, K = 10, n = 100, non-time axes included)."""
        super().__init__("Gaps", parent)
        from _autoplot_native import GAP_STRING_HELP
        form = QFormLayout(self)
        self.lines_cb = QCheckBox("Break the line / trace at small gaps")
        self.lines_cb.setChecked(True)
        self.lines_cb.setToolTip(
            "On: a trace's line stops and restarts where its step > K x its median "
            "step (or the fixed small gap); NaN rows go into plot copies (<y>_gaps), "
            "the data are not changed.  Off: every line is drawn across its gaps.")
        form.addRow(self.lines_cb)
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(1.0, 1000.0)
        self.k_spin.setDecimals(2)
        self.k_spin.setValue(10.0)
        self.k_spin.setToolTip("Small gap: a step longer than K x the median step of that trace "
                               "(also the scale of the large gap, n x small gap).")
        form.addRow(QLabel("Small gap K (step > K × median step):"), self.k_spin)
        self.small_text = QLineEdit()
        self.small_text.setPlaceholderText("e.g. 90s, 15min (time axes; overrides K)")
        self.small_text.setToolTip(GAP_STRING_HELP + "  Time axes only.")
        form.addRow(QLabel("Small gap fixed (time):"), self.small_text)
        self.axis_cb = QCheckBox("Break the x axis at large gaps")
        self.axis_cb.setChecked(True)
        self.axis_cb.setToolTip(
            "On: the x axis is cut (a Veusz broken axis; shaded in the Plots tab) "
            "where no trace has data for > n x the small-gap threshold (or the fixed "
            "large gap).  Off: one continuous x axis.  Independent of the line breaks.")
        form.addRow(self.axis_cb)
        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(0.0, 1.0e6)
        self.n_spin.setDecimals(1)
        self.n_spin.setSingleStep(10.0)
        self.n_spin.setValue(100.0)
        self.n_spin.setSpecialValueText("0 = no axis breaks")
        self.n_spin.setToolTip("Large gap: no data for longer than n x the small-gap threshold.")
        form.addRow(QLabel("Large gap n (no data > n × small gap):"), self.n_spin)
        self.large_text = QLineEdit()
        self.large_text.setPlaceholderText("e.g. 12h, 3d (time axes; overrides n)")
        self.large_text.setToolTip(GAP_STRING_HELP + "  Time axes only; never shorter than "
                                   "the small gap.")
        form.addRow(QLabel("Large gap fixed (time):"), self.large_text)
        self.numeric_cb = QCheckBox("Also on x axes that are not time (frequency, angle, index)")
        self.numeric_cb.setChecked(True)
        self.numeric_cb.setToolTip(
            "On: sorted non-time x data get the same K / n rule, with the step also "
            "large on a log scale (a log-spaced sweep is not cut).  Off: time axes only.")
        form.addRow(self.numeric_cb)
        # K and the fixed small gap also set the scale of the large gap
        # (n x small gap), so they stay enabled while either break is on
        self.lines_cb.toggled.connect(self._enable_values)
        self.axis_cb.toggled.connect(self._enable_values)

    def _enable_values(self, _checked: bool = False) -> None:
        """Enable the small-gap values while either break is on and the
        large-gap values while axis breaks are on."""
        small = self.lines_cb.isChecked() or self.axis_cb.isChecked()
        self.k_spin.setEnabled(small)
        self.small_text.setEnabled(small)
        self.n_spin.setEnabled(self.axis_cb.isChecked())
        self.large_text.setEnabled(self.axis_cb.isChecked())

    def time_layout(self) -> Any:
        """The _autoplot_native TimeLayout of the current settings."""
        from _autoplot_native import TimeLayout, parse_gap_string
        return TimeLayout(k=float(self.k_spin.value()), n=float(self.n_spin.value()),
                          manual_s=parse_gap_string(self.small_text.text()) * 86400.0,
                          axis_manual_s=parse_gap_string(self.large_text.text()) * 86400.0,
                          break_lines=self.lines_cb.isChecked(),
                          break_axis=self.axis_cb.isChecked(),
                          numeric=self.numeric_cb.isChecked())
