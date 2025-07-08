# -*- coding: utf-8 -*-
"""
=============================================================================
Enhanced R&S FSW ASCII Plotter with Averaging, Multiprocessing and GPU Support
Created on 2025-06-28 – revised 2025-07-08
Author: William W. Wallace
Email : wwallace@nrao.edu
Version: 1.1.0
=============================================================================
"""
# %% Imports (unchanged unless noted)
from dataclasses import dataclass
import os, re, sys, subprocess, psutil, math
import numpy as np
from operator import itemgetter
# … (all other original imports remain)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
# … (rest of original header code remains unchanged)

# %% Configuration dataclass (unchanged)

# %% GPUProcessor class (unchanged)

# %% Multiprocessing helpers (unchanged)

# %% Enhanced Qt GUI classes
class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with modern Qt interface."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced R&S SFT File Plotter")
        self.setGeometry(100, 100, 800, 600)

        # Configuration
        self.config = ProcessingConfig()

        # ────► CHANGED: pass config into VZPlotRnS ◄───
        self.vzplot = VZPlotRnS(self.config)

        self._setup_ui()
        self.selected_files = []

    # … (rest of class unchanged) …

# %% Veusz Auto-Plotter class
class VZPlotRnS:
    """Enhanced Veusz plotting class with multiprocessing support."""
    # ────► CHANGED SIGNATURE: receive config ◄───
    def __init__(self, config: ProcessingConfig):
        self.config = config                         # NEW
        self.doc = embed.Embedded('Enhanced R&S SFT File Plotter')
        self.first_1d = True
        self.doc.EnableToolbar(enable=True)

        # … (search strings, sft_lines and plotInfo definitions unchanged) …

    # ──────────────────────────────────────────────────────────────────────
    # NEW METHOD: create linear & dB averages for all datasets in a tag
    # ──────────────────────────────────────────────────────────────────────
    def _create_average_datasets(self, tag: str):
        """
        Create linear and dB averages from all datasets carrying *tag*.
        Excludes datasets whose names contain 'freq'. Uses GPU and/or
        multiprocessing when enabled in self.config.
        """
        # Collect candidate dataset names
        datasets = [ds for ds in self.doc.Tags(tag)
                    if ('freq' not in ds.lower()
                        and not ds.endswith(('_avg_dB', '_avg_lin')))]

        if len(datasets) < 2:        # nothing to average
            return

        # Choose backend
        use_gpu = self.config.enable_gpu_processing and CUPY_AVAILABLE
        xp = cp if use_gpu else np

        # Accumulate linear-domain data
        running_sum = None
        for name in datasets:
            arr = xp.asarray(self.doc.GetData(name))
            lin = xp.power(10.0, arr / 10.0)        # dB → linear
            running_sum = lin if running_sum is None else running_sum + lin

        avg_lin = running_sum / len(datasets)
        avg_db  = 10.0 * xp.log10(avg_lin)

        # Bring back to numpy if processed on GPU
        if use_gpu:
            avg_lin = cp.asnumpy(avg_lin)
            avg_db  = cp.asnumpy(avg_db)

        # Register datasets in Veusz
        lin_name = f"{tag}_avg_lin"
        db_name  = f"{tag}_avg_dB"
        self.doc.SetData(name=lin_name, val=avg_lin)
        self.doc.SetData(name=db_name,  val=avg_db)
        self.doc.TagDatasets(tag, [lin_name, db_name])

        # Create plot titled “average”
        prev_title = self.plotInfo.graph_title       # preserve
        self.plotInfo.graph_title = f"{tag} average"
        self._plot_1d(db_name)                       # plot average (dB)
        self.plotInfo.graph_title = prev_title

    # ──────────────────────────────────────────────────────────────────────
    # (original _process_file_data remains the same **except** we call the
    #   new averaging method at the end)
    # ──────────────────────────────────────────────────────────────────────
    def _process_file_data(self, filename, data_returned):
        # … original body up to final loop unchanged …

        # After all individual datasets are added & plotted:
        self._create_average_datasets(base_name)     # NEW CALL

    # ──────────────────────────────────────────────────────────────────────
    # _create_page, _plot_1d, save, open_veusz_gui – unchanged
    # ──────────────────────────────────────────────────────────────────────

# %% All remaining code (Qt helpers, main(), etc.) unchanged.
