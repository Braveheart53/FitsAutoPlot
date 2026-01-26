#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Qt GUI Launcher for Multiple Plotting Applications

TITLE: Enhanced Qt GUI Launcher
        GBO Electronics Scientific Plotting Suite
        
DESCRIPTION: This application provides a centralized interface to launch 
different plotting tools including ATR file processing, FITS file visualization,
Touchstone S-parameter analysis, CSV/TSV data visualization, and Rohde & Schwarz 
FSW spectrum analyzer data processing.

AUTHOR: William W. Wallace
LAST UPDATED: 2026-01-26

COMPATIBLE WITH: Python 3.8+, PySide6, QtPy

FEATURES:
    - Multiple plotting application launcher buttons
    - Configurable background image support
    - Splash screen with progress bar for Nuitka deployment
    - Extensible design for additional plotting tools
    - Autopep8 compliant code formatting

HOW TO EDIT OR UPDATE:
    Modify these variables to point to your images:
    - MAINBACKGROUNDIMAGEPATH = 'assets/yourbackground.jpg' (Relative path)
    - SPLASHBACKGROUNDIMAGEPATH = 'assets/yoursplash.png'
    
    To add additional plotting tools, simply:
    1. Add the script path to SCRIPTPATHS dictionary
    2. Add button configuration to buttonconfigs list in createbuttonsSection
    3. The system will automatically handle the new application
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

# Import modules based on execution context
if getattr(sys, 'frozen', False):
    # Force direct PySide6 usage for compiled builds
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
    from PySide6.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSplashScreen, QProgressBar, QMessageBox,
        QFrame, QSizePolicy, QSpacerItem
    )
    # Running as compiled executable - use PySide6 directly
else:
    # Development environment - use QtPy
    os.environ['QT_API'] = 'pyside6'
    from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize
    from qtpy.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSplashScreen, QProgressBar, QMessageBox,
        QFrame, QSizePolicy, QSpacerItem
    )

# ============================================================================
# CONFIGURATION SECTION - Easy-to-modify variables
# ============================================================================

MAINBACKGROUNDIMAGEPATH = 'assets/loadfallcolors.jpg'  # Relative path
SPLASHBACKGROUNDIMAGEPATH = 'assets/loadbuilding.png'  # Splash screen image

APPNAME = 'GBO Electronics Scientific Plotting Suite'
APPVERSION = '0.1.4'
WINDOWWIDTH = 900
WINDOWHEIGHT = 700

# Script paths - Add your script filenames here
SCRIPTPATHS = {
    'atr': 'ATRAutoPlot.py',
    'fits': 'FITSAutoPlot.py',
    'rands': 'RAndSFSWASCIIPlotter.py',
    'snp': 'TouchstoneAutoPlot.py',
    'csv': 'CSV_TSV_AutoPlot.py',
    'rands_large': 'RAndS_FSW_ASCII_Plotter_lrgFiles.py'
}

# ============================================================================
# SPLASH SCREEN THREAD - For handling progress during startup
# ============================================================================

class SplashScreenThread(QThread):
    """
    Thread for handling splash screen progress simulation.
    
    Signals
    -------
    progressUpdated : Signal(int)
        Emitted with progress percentage (0-100).
    splashFinished : Signal()
        Emitted when splash screen should close.
    """

    progressUpdated = Signal(int)
    splashFinished = Signal()

    def __init__(self):
        """
        Initialize the splash screen thread.
        """
        super().__init__()
        self.progressSteps = [
            ('Initializing application...', 20),
            ('Loading plotting modules...', 40),
            ('Configuring GUI components...', 60),
            ('Preparing user interface...', 80),
            ('Finalizing startup...', 100)
        ]

    def run(self):
        """
        Execute splash screen progress simulation.
        """
        import time
        for message, progress in self.progressSteps:
            time.sleep(0.5)  # Simulate loading time
            self.progressUpdated.emit(progress)
        self.splashFinished.emit()


class CustomSplashScreen(QSplashScreen):
    """
    Enhanced splash screen with progress bar and custom styling.
    """

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
        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(50, pixmap.height() - 50, pixmap.width() - 100, 20)
        self.progressBar.setStyleSheet("""
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
        self.statusLabel = QLabel(self)
        self.statusLabel.setGeometry(50, pixmap.height() - 80, pixmap.width() - 100, 25)
        self.statusLabel.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 3px;
                padding: 5px;
                font-weight: bold;
                color: black;
            }
        """)
        self.statusLabel.setText('Initializing application...')
        self.statusLabel.setAlignment(Qt.AlignCenter)

    def updateProgress(self, value: int):
        """
        Update the progress bar value.
        
        Parameters
        ----------
        value : int
            Progress percentage (0-100).
        """
        self.progressBar.setValue(value)
        
        # Update status message based on progress
        if value <= 20:
            self.statusLabel.setText('Initializing application...')
        elif value <= 40:
            self.statusLabel.setText('Loading plotting modules...')
        elif value <= 60:
            self.statusLabel.setText('Configuring GUI components...')
        elif value <= 80:
            self.statusLabel.setText('Preparing user interface...')
        else:
            self.statusLabel.setText('Finalizing startup...')


# ============================================================================
# MAIN APPLICATION CLASSES
# ============================================================================

class ScriptLauncher:
    """
    Utility class for launching external Python scripts.
    """

    @staticmethod
    def launchScript(scriptPath: str) -> bool:
        """
        Launch an external Python script using subprocess.
        
        This method uses subprocess.Popen to run the script in a separate
        process, allowing the main GUI to remain responsive.
        
        Parameters
        ----------
        scriptPath : str
            Path to the Python script to execute.
        
        Returns
        -------
        bool
            True if script was launched successfully, False otherwise.
        """
        try:
            # Verify script exists
            if not os.path.exists(scriptPath):
                QMessageBox.critical(
                    None,
                    'Script Not Found',
                    f'Could not find script: {scriptPath}\n\n'
                    f'Please ensure the script exists and the path is correct.'
                )
                return False
            
            # Launch the script in a separate process
            subprocess.Popen(
                [sys.executable, scriptPath],
                cwd=os.path.dirname(scriptPath) or '.'
            )
            return True
        except Exception as e:
            QMessageBox.critical(
                None,
                'Launch Error',
                f'Failed to launch script: {scriptPath}\n\n'
                f'Error: {str(e)}'
            )
            return False


class MainWindow(QMainWindow):
    """
    Main application window with plotting application launcher buttons.
    
    This class creates the primary interface featuring multiple buttons for
    launching different plotting applications, with support for background
    images and modern styling.
    """

    def __init__(self):
        """
        Initialize the main window.
        """
        super().__init__()
        self.scriptLauncher = ScriptLauncher()
        
        # Using sys.executable ensures we use the same Python interpreter
        self.setupWindow()      # Configure main window
        self.setupUI()          # Set up the user interface
        self.setupBackground()  # Apply background image if available
        self.applyStyling()     # Apply modern styling

    def setupWindow(self):
        """
        Configure basic window properties.
        """
        self.setWindowTitle(f'{APPNAME} v{APPVERSION}')
        self.setMinimumSize(WINDOWWIDTH, WINDOWHEIGHT)
        self.setFixedSize(WINDOWWIDTH, WINDOWHEIGHT)
        
        self.centerWindow()  # Center window on screen
        
        # Set window icon if available
        iconPath = 'assets/appicon.png'
        if os.path.exists(iconPath):
            self.setWindowIcon(QIcon(iconPath))

    def centerWindow(self):
        """
        Center the window on the screen.
        """
        screenGeometry = QApplication.primaryScreen().geometry()
        windowGeometry = self.geometry()
        x = (screenGeometry.width() - windowGeometry.width()) // 2
        y = (screenGeometry.height() - windowGeometry.height()) // 2
        self.move(x, y)

    def setupUI(self):
        """
        Set up the user interface layout and widgets.
        """
        # Create central widget
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        # Create main layout
        mainLayout = QVBoxLayout(centralWidget)
        mainLayout.setSpacing(30)
        mainLayout.setContentsMargins(50, 50, 50, 50)
        
        # Add title label
        self.createTitleSection(mainLayout)
        
        # Add main buttons section
        self.createButtonsSection(mainLayout)
        
        # Add footer section
        self.createFooterSection(mainLayout)

    def createTitleSection(self, parentLayout: QVBoxLayout):
        """
        Create the title section of the interface.
        
        Parameters
        ----------
        parentLayout : QVBoxLayout
            Parent layout to add the title section to.
        """
        titleLabel = QLabel(APPNAME)
        titleLabel.setAlignment(Qt.AlignCenter)
        titleLabel.setStyleSheet("""
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
        
        subtitleLabel = QLabel('Professional Scientific Data Visualization Suite')
        subtitleLabel.setAlignment(Qt.AlignCenter)
        subtitleLabel.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #34495e;
                margin-bottom: 20px;
                background-color: rgba(255, 255, 255, 150);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        parentLayout.addWidget(titleLabel)
        parentLayout.addWidget(subtitleLabel)

    def createButtonsSection(self, parentLayout: QVBoxLayout):
        """
        Create the main buttons section.
        
        This section contains the primary function buttons for launching
        different plotting applications.
        
        Parameters
        ----------
        parentLayout : QVBoxLayout
            Parent layout to add the buttons section to.
        """
        # Create buttons frame for better organization
        buttonsFrame = QFrame()
        buttonsFrame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 180);
                border-radius: 10px;
                padding: 0px;
            }
        """)
        
        buttonsLayout = QVBoxLayout(buttonsFrame)
        buttonsLayout.setSpacing(10)
        
        # Define button configurations: (text, tooltip, scriptkey, iconpath)
        buttonConfigs = [
            (
                'Plot GBO Outdoor ATR Files',
                'Launch the GBO Outdoor Antenna Range file plotter for processing '
                'and visualizing ATR measurement data',
                'atr',
                'assets/atricon.png'
            ),
            (
                'Plot FITS Files',
                'Launch the FITS file visualization tool for astronomical and '
                'scientific image data',
                'fits',
                'assets/fitsicon.png'
            ),
            (
                'Plot Delimited Files',
                'Launch the CSV/TSV file plotter for visualizing comma, tab, or '
                'custom-separated data files with arbitrary delimiters',
                'csv',
                'assets/csvicon.png'
            ),
            (
                'Plot Rhode and Schwarz FSW Files',
                'Launch the Rohde & Schwarz FSW ASCII plotter for spectrum analyzer '
                'data visualization from Rhode and Schwarz test equipment',
                'rands',
                'assets/randsicon.png'
            ),
            (
                'Rhode and Schwarz FSW LARGE Files',
                'Launch the Rohde & Schwarz FSW ASCII plotter optimized for large files '
                'for high-volume spectrum analyzer data visualization',
                'rands_large',
                'assets/randslrgicon.png'
            ),
            (
                'Plot Touchstone Files with Smith Charts',
                'Launch the Touchstone File AutoPlot tool for processing and '
                'visualizing S-parameter measurement data with Smith chart generation',
                'snp',
                'assets/snpicon.png'
            )
        ]
        
        # Create buttons with enhanced styling
        self.buttons = {}
        for text, tooltip, scriptKey, iconPath in buttonConfigs:
            button = self.createStyledButton(text, tooltip, iconPath)
            
            # Connect button to appropriate launch method
            button.clicked.connect(
                lambda checked, key=scriptKey: self.launchApplication(key)
            )
            
            self.buttons[scriptKey] = button
            buttonsLayout.addWidget(button)
        
        parentLayout.addWidget(buttonsFrame)

    def createStyledButton(self, text: str, tooltip: str, iconPath: str) -> QPushButton:
        """
        Create a consistently styled button.
        
        Parameters
        ----------
        text : str
            Button text.
        tooltip : str
            Button tooltip text.
        iconPath : str
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
        if os.path.exists(iconPath):
            icon = QIcon(iconPath)
            button.setIcon(icon)
            button.setIconSize(QSize(32, 32))
        
        # Apply modern button styling
        button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3daee9, stop: 1 #2980b9
                );
                border: none;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 5px 5px;
                text-align: left;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5dade2, stop: 1 #3498db
                );
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2980b9, stop: 1 #1f618d
                );
                padding-top: 17px;
                padding-left: 27px;
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        
        return button

    def createFooterSection(self, parentLayout: QVBoxLayout):
        """
        Create the footer section with version and author information.
        
        Parameters
        ----------
        parentLayout : QVBoxLayout
            Parent layout to add the footer section to.
        """
        # Add spacer to push footer to bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        parentLayout.addItem(spacer)
        
        # Create footer label
        footerLabel = QLabel(
            f'Version {APPVERSION} | Scientific Computing Suite\n'
            f'Author: William W. Wallace'
        )
        footerLabel.setAlignment(Qt.AlignCenter)
        footerLabel.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #072911;
                background-color: rgba(255, 255, 255, 100);
                border-radius: 5px;
                padding: 8px;
            }
        """)
        parentLayout.addWidget(footerLabel)

    def setupBackground(self):
        """
        Set up the background image if available.
        """
        if os.path.exists(MAINBACKGROUNDIMAGEPATH):
            try:
                # This prevents child widgets from inheriting the background
                backgroundStyle = f'MainWindow {{ border-image: url({MAINBACKGROUNDIMAGEPATH}) 0 0 0 0 stretch stretch; }}'
                self.setStyleSheet(backgroundStyle)
            except Exception as e:
                print(f'Warning: Could not set background image: {e}')

    def applyStyling(self):
        """
        Apply additional modern styling to the application.
        """
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
        
        # Configure window properties for modern appearance
        font = QFont('Segoe UI', 10)  # Modern font choice
        self.setFont(font)

    def launchApplication(self, scriptKey: str):
        """
        Launch the specified application.
        
        This method handles the launching of external Python scripts and
        provides user feedback on the operation status.
        
        Parameters
        ----------
        scriptKey : str
            Key identifying which script to launch ('atr', 'fits', 'csv', 
            'rands', 'rands_large', 'snp').
        """
        scriptPath = SCRIPTPATHS.get(scriptKey)
        
        if not scriptPath:
            QMessageBox.warning(
                self,
                'Configuration Error',
                f'No script path configured for: {scriptKey}'
            )
            return
        
        # Launch the script
        success = self.scriptLauncher.launchScript(scriptPath)
        
        if success:
            QMessageBox.information(
                self,
                'Application Launched',
                f'Successfully launched: {scriptPath}\n\n'
                f'The application should appear shortly.'
            )
            
            # Disable button temporarily
            if scriptKey in self.buttons:
                button = self.buttons[scriptKey]
                button.setEnabled(False)
                
                # Restore button state after 2 seconds
                QTimer.singleShot(
                    2000,
                    lambda b=button, k=scriptKey: self.restoreButtonState(b, k)
                )

    def restoreButtonState(self, button: QPushButton, scriptKey: str):
        """
        Restore button to normal state after launching.
        
        Parameters
        ----------
        button : QPushButton
            Button to restore.
        scriptKey : str
            Script key for determining original text.
        """
        button.setEnabled(True)


# ============================================================================
# APPLICATION CLASS
# ============================================================================

class PlottingSuiteApplication(QApplication):
    """
    Main application class with splash screen support.
    
    This class manages the application lifecycle, including the optional
    splash screen for deployment scenarios particularly useful when compiled
    with Nuitka.
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
        self.mainWindow = None
        self.splashThread = None

        # Configure application properties
        self.setApplicationName(APPNAME)
        self.setApplicationVersion(APPVERSION)
        self.setOrganizationName('Scientific Computing')
        
        # Set application style
        self.setStyle('Fusion')  # Modern cross-platform style

    def showSplashScreen(self) -> bool:
        """
        Display splash screen if background image is available.
        
        Returns
        -------
        bool
            True if splash screen was displayed, False otherwise.
        """
        if os.path.exists(SPLASHBACKGROUNDIMAGEPATH):
            pixmap = QPixmap(SPLASHBACKGROUNDIMAGEPATH)
        else:
            # Create a default splash screen pixmap if image not found
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.darkBlue)
            print(f'Warning: Splash image not found at {SPLASHBACKGROUNDIMAGEPATH}')
            print('Using default splash screen')
        
        self.splash = CustomSplashScreen(pixmap)
        self.splash.show()
        
        # Process events to ensure splash screen is visible
        self.processEvents()
        
        # Start splash screen thread
        self.splashThread = SplashScreenThread()
        self.splashThread.progressUpdated.connect(self.splash.updateProgress)
        self.splashThread.splashFinished.connect(self.onSplashFinished)
        self.splashThread.start()
        
        return True

    def onSplashFinished(self):
        """
        Handle splash screen completion.
        """
        if self.splash:
            self.splash.finish(self.mainWindow)
            self.splash = None

    def runWithoutSplash(self):
        """
        Run application without splash screen.
        """
        self.mainWindow = MainWindow()
        self.mainWindow.show()

    def runSplashMode(self):
        """
        Execute application event loop.
        """
        return self.exec()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setupQtPlugins():
    """
    Setup Qt platform plugin paths for compiled applications.
    """
    try:
        import PySide6
        dirname = os.path.dirname(PySide6.__file__)
        pluginPath = os.path.join(dirname, 'plugins', 'platforms')
        if os.path.exists(pluginPath):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pluginPath
    except ImportError:
        pass


def ensureAssetsDirectory():
    """
    Ensure assets directory exists and create placeholder files if needed.
    
    This function helps developers set up the basic directory structure
    for background images and icons.
    """
    assetsDir = Path('assets')
    assetsDir.mkdir(exist_ok=True)
    
    # Create placeholder text files to guide developers
    placeholderFiles = [
        ('mainbackground.jpg', 'Place main window background image here'),
        ('splashbackground.png', 'Place splash screen background image here'),
        ('appicon.png', 'Place application icon here'),
        ('atricon.png', 'Place ATR plotter icon here'),
        ('fitsicon.png', 'Place FITS plotter icon here'),
        ('csvicon.png', 'Place CSV/TSV plotter icon here'),
        ('randsicon.png', 'Place Rhode and Schwarz plotter icon here'),
        ('randslrgicon.png', 'Place Rhode and Schwarz large files icon here'),
        ('snpicon.png', 'Place Touchstone plotter icon here'),
    ]
    
    for filename, description in placeholderFiles:
        placeholderPath = assetsDir / f'{filename}.placeholder'
        if not placeholderPath.exists() and not (assetsDir / filename).exists():
            with open(placeholderPath, 'w') as f:
                f.write(description)
                f.write(f'\n\nRename this file to {filename} after adding your image.')


def checkScriptDependencies() -> bool:
    """
    Check if required script files exist and provide user guidance.
    
    Returns
    -------
    bool
        True if all scripts exist, False otherwise.
    """
    missingScripts = []
    for scriptName, scriptPath in SCRIPTPATHS.items():
        if not os.path.exists(scriptPath):
            missingScripts.append((scriptName, scriptPath))
    
    if missingScripts:
        print('Warning: Missing script files:')
        for name, path in missingScripts:
            print(f'  - {name}: {path}')
        print('\nPlease ensure all script files are in the correct location.')
        return False
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main application entry point.
    
    This function handles application initialization, splash screen display,
    and main window creation based on deployment context.
    """
    # Call this before any Qt imports if running as compiled executable
    if getattr(sys, 'frozen', False):
        setupQtPlugins()
    
    # Ensure assets directory exists
    ensureAssetsDirectory()
    
    # Create application instance
    app = PlottingSuiteApplication(sys.argv)
    
    # Show splash screen if compiled with Nuitka or if explicitly requested
    showSplash = (
        getattr(sys, 'frozen', False) or
        '--splash' in sys.argv or  # Nuitka compiled
        os.path.exists(SPLASHBACKGROUNDIMAGEPATH)  # Image available
    )
    
    if showSplash:
        app.showSplashScreen()
        app.mainWindow = MainWindow()
        return app.runSplashMode()
    else:
        app.runWithoutSplash()
        return app.runSplashMode()


if __name__ == '__main__':
    """
    Script entry point.
    
    This section ensures proper execution when the script is run directly
    and handles any initialization errors gracefully.
    """
    try:
        sys.exit(main())
    except Exception as e:
        print(f'Application failed to start: {e}')
        sys.exit(1)
