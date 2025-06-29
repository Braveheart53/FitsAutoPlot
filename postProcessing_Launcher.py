# -*- coding: utf-8 -*-
"""
Enhanced Qt GUI Launcher for Multiple Plotting Applications.

# %% Header Info
This application provides a centralized interface to launch different plotting
tools including ATR file processing, FITS file visualization, and R&S FSW
SFT file analysis.

# %%% Author Info
Author: William W. Wallace
Last updated: 2025-06-28
Compatible with: Python 3.8+, PySide6, QtPy

# %%% Features
- Three main function buttons for different plotting applications
- Configurable background image support
- Splash screen with progress bar for Nuitka deployment
- Extensible design for additional plotting tools
- Autopep8 compliant code formatting

# %%% How to edit or update
Modify these variables to point to your images
- MAIN_BACKGROUND_IMAGE_PATH = "assets/your_background.jpg"
- SPLASH_BACKGROUND_IMAGE_PATH = "assets/your_splash.png"

To add additional plotting tools, simply:
- Add the script path to SCRIPT_PATHS dictionary
- Add button configuration to button_configs list in _create_buttons_section
- The system will automatically handle the new application
"""

# %% Import Modules

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

# Qt modules - using qtpy for cross-platform compatibility
from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize
from qtpy.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSplashScreen, QProgressBar, QMessageBox,
    QFrame, QSizePolicy, QSpacerItem
)

# %% Configuration Variables

# Background image configuration - easily configurable by developers
MAIN_BACKGROUND_IMAGE_PATH = "assets/load_fall_colors.jpg"  # Relative path
SPLASH_BACKGROUND_IMAGE_PATH = "assets/load_building.png"  # Splash screen image

# Application configuration
APP_NAME = "GBO Electronics Scientific Plotting Suite"
APP_VERSION = "0.1.3"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Script paths - adjust these to your script locations
# TODO: Add Paths and Script Keys Here
SCRIPT_PATHS = {
    "atr": "ATR_AutoPlot.py",
    "fits": "FITS_AutoPlot.py",
    "rands": "RAndS_FSW_ASCII_Plotter.py",
    "snp": "Touchstone_AutoPlot.py"
}

# %% Splash Screen Classes


class SplashScreenThread(QThread):
    """Thread for handling splash screen progress simulation."""

    progress_updated = Signal(int)
    splash_finished = Signal()

    def __init__(self):
        """Initialize the splash screen thread."""
        super().__init__()
        self.progress_steps = [
            ("Initializing application...", 20),
            ("Loading plotting modules...", 40),
            ("Configuring GUI components...", 60),
            ("Preparing user interface...", 80),
            ("Finalizing startup...", 100)
        ]

    def run(self):
        """Execute splash screen progress simulation."""
        import time

        for message, progress in self.progress_steps:
            time.sleep(0.5)  # Simulate loading time
            self.progress_updated.emit(progress)

        self.splash_finished.emit()


class CustomSplashScreen(QSplashScreen):
    """Enhanced splash screen with progress bar and custom styling."""

    def __init__(self, pixmap: QPixmap):
        """
        Initialize the custom splash screen.

        Parameters
        ----------
        pixmap : QPixmap
            Background image for the splash screen.
        """
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)

        # Configure splash screen appearance
        self.setMask(pixmap.mask())

        # Create progress bar
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

        # Create status label
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
        self.status_label.setAlignment(Qt.AlignCenter)

    def update_progress(self, value: int):
        """
        Update the progress bar value.

        Parameters
        ----------
        value : int
            Progress percentage (0-100).
        """
        self.progress_bar.setValue(value)

        # Update status message based on progress
        if value <= 20:
            self.status_label.setText("Initializing application...")
        elif value <= 40:
            self.status_label.setText("Loading plotting modules...")
        elif value <= 60:
            self.status_label.setText("Configuring GUI components...")
        elif value <= 80:
            self.status_label.setText("Preparing user interface...")
        else:
            self.status_label.setText("Finalizing startup...")


# %% Main Application Classes

class ScriptLauncher:
    """Utility class for launching external Python scripts."""

    @staticmethod
    def launch_script(script_path: str) -> bool:
        """
        Launch an external Python script using subprocess.

        This method uses subprocess.Popen to run the script in a separate
        process, allowing the main GUI to remain responsive.

        Parameters
        ----------
        script_path : str
            Path to the Python script to execute.

        Returns
        -------
        bool
            True if script was launched successfully, False otherwise.
        """
        try:
            # Verify script exists
            if not os.path.exists(script_path):
                QMessageBox.critical(
                    None, "Script Not Found",
                    f"Could not find script: {script_path}\n\n"
                    f"Please ensure the script exists and the path is correct."
                )
                return False

            # Launch script using subprocess
            # Using sys.executable ensures we use the same Python interpreter
            subprocess.Popen([
                sys.executable, script_path
            ], cwd=os.path.dirname(script_path) or ".")

            return True

        except Exception as e:
            QMessageBox.critical(
                None, "Launch Error",
                f"Failed to launch script: {script_path}\n\n"
                f"Error: {str(e)}"
            )
            return False


class MainWindow(QMainWindow):
    """
    Main application window with three plotting application launcher buttons.

    This class creates the primary interface featuring three main buttons
    for launching different plotting applications, with support for
    background images and modern styling.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.script_launcher = ScriptLauncher()

        # Configure main window
        self._setup_window()

        # Set up the user interface
        self._setup_ui()

        # Apply background image if available
        self._setup_background()

        # Apply modern styling
        self._apply_styling()

    def _setup_window(self):
        """Configure basic window properties."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        # self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Center window on screen
        self._center_window()

        # Set window icon if available
        icon_path = "assets/app_icon.png"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _center_window(self):
        """Center the window on the screen."""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.geometry()

        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2

        self.move(x, y)

    def _setup_ui(self):
        """Set up the user interface layout and widgets."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # Add title label
        self._create_title_section(main_layout)

        # Add main buttons section
        self._create_buttons_section(main_layout)

        # Add footer section
        self._create_footer_section(main_layout)

    def _create_title_section(self, parent_layout: QVBoxLayout):
        """
        Create the title section of the interface.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Parent layout to add the title section to.
        """
        title_label = QLabel(APP_NAME)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
                background-color: rgba(255, 255, 255, 200);
                border-radius: 10px;
                padding: 15px;
            }
        """)

        subtitle_label = QLabel(
            "Professional Scientific Data Visualization Suite"
        )
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #34495e;
                margin-bottom: 20px;
                background-color: rgba(255, 255, 255, 150);
                border-radius: 8px;
                padding: 10px;
            }
        """)

        parent_layout.addWidget(title_label)
        parent_layout.addWidget(subtitle_label)

    def _create_buttons_section(self, parent_layout: QVBoxLayout):
        """
        Create the main buttons section.

        This section contains the three primary function buttons for
        launching different plotting applications.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Parent layout to add the buttons section to.
        """
        # Create buttons frame for better organization
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 10px;
                padding: 0px;
            }
        """)

        buttons_layout = QVBoxLayout(buttons_frame)
        buttons_layout.setSpacing(10)

        # Button configurations: (text, tooltip, script_key, icon_path)
        # TODO: Add to this list when adding a new button
        button_configs = [
            (
                "Plot GBO Outdoor ATR Files",
                "Launch the GBO Outdoor Antenna Range file plotter\n"
                "for processing and visualizing ATR measurement data",
                "atr",
                "assets/atr_icon.png"
            ),
            (
                "Plot FITS Files (under Construction)",
                "Launch the FITS file visualization tool\n"
                "for astronomical and scientific image data",
                "fits",
                "assets/fits_icon.png"
            ),
            (
                "Plot R&S FSW SFT Files",
                "Launch the Rohde & Schwarz FSW ASCII plotter\n"
                "for spectrum analyzer data visualization",
                "rands",
                "assets/rands_icon.png"
            ),
            (
                "Plot Touchfiles with Time Domain Processing "
                " (under construction)",
                "Launch the GBO Touchstone File AutoPlot GU \nI"
                "for processing and visualizing S-parameter measurement data",
                "snp",
                "assets/snp_icon.png"
            )
        ]

        # Create buttons with enhanced styling
        self.buttons = {}
        for text, tooltip, script_key, icon_path in button_configs:
            button = self._create_styled_button(text, tooltip, icon_path)

            # Connect button to appropriate launch method
            button.clicked.connect(
                lambda checked, key=script_key: self._launch_application(key)
            )

            self.buttons[script_key] = button
            buttons_layout.addWidget(button)

        parent_layout.addWidget(buttons_frame)

    def _create_styled_button(self, text: str, tooltip: str,
                              icon_path: str) -> QPushButton:
        """
        Create a consistently styled button.

        Parameters
        ----------
        text : str
            Button text.
        tooltip : str
            Button tooltip text.
        icon_path : str
            Path to button icon image.

        Returns
        -------
        QPushButton
            Configured button widget.
        """
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.setMinimumHeight(50)
        button.setMaximumHeight(70)

        # Set icon if available
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            button.setIcon(icon)
            button.setIconSize(QSize(32, 32))

        # Apply modern button styling
        button.setStyleSheet("""
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
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
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
        """)

        return button

    def _create_footer_section(self, parent_layout: QVBoxLayout):
        """
        Create the footer section with version and author information.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Parent layout to add the footer section to.
        """
        # Add spacer to push footer to bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum,
                             QSizePolicy.Expanding)
        parent_layout.addItem(spacer)

        footer_label = QLabel(
            f"Version {APP_VERSION} | Scientific Computing Suite \n Author: William W. Wallace")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #072911;
                background-color: rgba(255, 255, 255, 100);
                border-radius: 5px;
                padding: 8px;
            }
        """)

        parent_layout.addWidget(footer_label)

    def _setup_background(self):
        """Set up the background image if available."""
        if os.path.exists(MAIN_BACKGROUND_IMAGE_PATH):
            try:
                # Set background using stylesheet with object name specification
                # This prevents child widgets from inheriting the background
                self.setObjectName("MainWindow")
                background_style = f"""
                    #MainWindow {{
                        border-image: url({MAIN_BACKGROUND_IMAGE_PATH}) 0 0 0 0 stretch stretch;
                    }}
                """
                self.setStyleSheet(background_style)
            except Exception as e:
                print(f"Warning: Could not set background image: {e}")

    def _apply_styling(self):
        """Apply additional modern styling to the application."""
        # Set application-wide font
        font = QFont("Segoe UI", 10)  # Modern font choice
        self.setFont(font)

        # Configure window properties for modern appearance
        self.setStyleSheet(self.styleSheet() + """
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
        """)

    def _launch_application(self, script_key: str):
        """
        Launch the specified application.

        This method handles the launching of external Python scripts
        and provides user feedback on the operation status.

        Parameters
        ----------
        script_key : str
            Key identifying which script to launch ('atr', 'fits', 'rands').
        """
        script_path = SCRIPT_PATHS.get(script_key)

        if not script_path:
            QMessageBox.warning(
                self, "Configuration Error",
                f"No script path configured for: {script_key}"
            )
            return

        # Provide visual feedback
        # TODO: Find out why the launch button Touchston files does not update
        button = self.buttons.get(script_key)
        if button:
            button.setEnabled(False)
            button.setText(button.text() + " (Launching...)")

        # Launch the script
        success = self.script_launcher.launch_script(script_path)

        if success:
            QMessageBox.information(
                self, "Application Launched",
                f"Successfully launched {script_path}\n\n"
                f"The application should appear shortly."
            )

        # Restore button state
        if button:
            QTimer.singleShot(
                2000, lambda: self._restore_button_state(button, script_key))

    def _restore_button_state(self, button: QPushButton, script_key: str):
        """
        Restore button to normal state after launching.

        Parameters
        ----------
        button : QPushButton
            Button to restore.
        script_key : str
            Script key for determining original text.
        """
        button.setEnabled(True)

        # Restore original button text
        original_texts = {
            "atr": "Plot GBO Outdoor ATR Files",
            "fits": "Plot FITS Files",
            "rands": "Plot R&S FSW SFT Files"
        }

        if script_key in original_texts:
            button.setText(original_texts[script_key])


# %% Application Class

class PlottingSuiteApplication(QApplication):
    """
    Main application class with splash screen support.

    This class manages the application lifecycle, including the optional
    splash screen for deployment scenarios (particularly useful when
    compiled with Nuitka).
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

        # Configure application properties
        self.setApplicationName(APP_NAME)
        self.setApplicationVersion(APP_VERSION)
        self.setOrganizationName("Scientific Computing")

        # Set application style
        self.setStyle('Fusion')  # Modern cross-platform style

    def show_splash_screen(self) -> bool:
        """
        Display splash screen if background image is available.

        Returns
        -------
        bool
            True if splash screen was displayed, False otherwise.
        """
        # Check if splash screen image exists
        if not os.path.exists(SPLASH_BACKGROUND_IMAGE_PATH):
            # Create a default splash screen pixmap if image not found
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.darkBlue)
            print(
                f"Warning: Splash image not found at {SPLASH_BACKGROUND_IMAGE_PATH}")
            print("Using default splash screen")
        else:
            pixmap = QPixmap(SPLASH_BACKGROUND_IMAGE_PATH)

        # Create and show splash screen
        self.splash = CustomSplashScreen(pixmap)
        self.splash.show()

        # Process events to ensure splash screen is visible
        self.processEvents()

        # Start splash screen thread
        self.splash_thread = SplashScreenThread()
        self.splash_thread.progress_updated.connect(
            self.splash.update_progress)
        self.splash_thread.splash_finished.connect(self._on_splash_finished)
        self.splash_thread.start()

        return True

    def _on_splash_finished(self):
        """Handle splash screen completion."""
        # Create and show main window
        self.main_window = MainWindow()
        self.main_window.show()

        # Close splash screen
        if self.splash:
            self.splash.finish(self.main_window)
            self.splash = None

        # Clean up splash thread
        if self.splash_thread:
            self.splash_thread.quit()
            self.splash_thread.wait()
            self.splash_thread = None

    def run_without_splash(self):
        """Run application without splash screen."""
        self.main_window = MainWindow()
        self.main_window.show()


# %% Utility Functions

def setup_qt_plugins():
    """
    Setup Qt platform plugin paths for compiled applications.
    """
    try:
        import PySide6
        dirname = os.path.dirname(PySide6.__file__)
        plugin_path = os.path.join(dirname, 'plugins', 'platforms')
        if os.path.exists(plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    except ImportError:
        pass


def ensure_assets_directory():
    """
    Ensure assets directory exists and create placeholder files if needed.

    This function helps developers set up the basic directory structure
    for background images and icons.
    """
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    # Create placeholder text files to guide developers
    placeholder_files = [
        ("main_background.jpg", "Place main window background image here"),
        ("splash_background.png", "Place splash screen background image here"),
        ("app_icon.png", "Place application icon here"),
        ("atr_icon.png", "Place ATR plotter icon here"),
        ("fits_icon.png", "Place FITS plotter icon here"),
        ("rands_icon.png", "Place R&S plotter icon here"),
    ]

    for filename, description in placeholder_files:
        placeholder_path = assets_dir / f"{filename}.placeholder"
        if not placeholder_path.exists() and not (assets_dir / filename).exists():
            with open(placeholder_path, "w") as f:
                f.write(f"{description}\n")
                f.write(
                    f"Rename this file to '{filename}' after adding your image.\n")


def check_script_dependencies():
    """
    Check if required script files exist and provide user guidance.

    Returns
    -------
    bool
        True if all scripts exist, False otherwise.
    """
    missing_scripts = []

    for script_name, script_path in SCRIPT_PATHS.items():
        if not os.path.exists(script_path):
            missing_scripts.append((script_name, script_path))

    if missing_scripts:
        print("Warning: Missing script files:")
        for name, path in missing_scripts:
            print(f"  - {name}: {path}")
        print("\nPlease ensure all script files are in the correct location.")
        return False

    return True


# %% Main Execution

def main():
    """
    Main application entry point.

    This function handles application initialization, splash screen display,
    and main window creation based on deployment context.
    """
    # Call this before any Qt imports
    if getattr(sys, 'frozen', False):  # Check if running as compiled executable
        setup_qt_plugins()

    # Create application instance
    app = PlottingSuiteApplication(sys.argv)

    # Ensure assets directory exists
    ensure_assets_directory()

    # Check script dependencies
    check_script_dependencies()

    # Determine if we should show splash screen
    # Show splash screen if compiled with Nuitka or if explicitly requested
    show_splash = (
        getattr(sys, 'frozen', False) or  # Nuitka compiled
        '--splash' in sys.argv or         # Explicit request
        os.path.exists(SPLASH_BACKGROUND_IMAGE_PATH)  # Image available
    )

    if show_splash:
        app.show_splash_screen()
    else:
        app.run_without_splash()

    # Execute application event loop
    return app.exec_()


# %% Entry Point

if __name__ == "__main__":
    """
    Script entry point.

    This section ensures proper execution when the script is run directly
    and handles any initialization errors gracefully.
    """
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)
