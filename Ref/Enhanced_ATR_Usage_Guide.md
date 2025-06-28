# Enhanced ATR AutoPlot - Quick Start Guide

## Installation Requirements
Same as original ATR_AutoPlot:
- Python 3.7+
- qtpy (with PySide6 or PyQt5)
- numpy
- veusz
- Optional: cupy (NVIDIA GPU), pyopencl (cross-platform GPU), taichi (GPU)

## Launch Application
```bash
python ATR_AutoPlot_Enhanced_GUI.py
```

## GUI Interface Overview

### File Selection Section
- **Browse ATR Files**: Select multiple .atr files at once
- **File List**: Visual display of selected files
- **Clear Files**: Remove all selected files

### Plot Configuration Section  
- **Plot Title**: Custom title for your plots
- **Dataset Name**: Base name for datasets

### Processing Options Section
- **Enable Multiprocessing**: Toggle parallel file processing
- **CPU Cores**: Set number of worker processes (1 to max cores)
- **Enable GPU Processing**: Toggle GPU acceleration
- **GPU Backend**: Shows detected GPU backend (CuPy/OpenCL/Taichi)

### Status Section
- **Progress Bar**: Visual progress during processing
- **Status Messages**: Timestamped log of all operations

### Control Buttons
- **Process and Create Plots**: Start processing (green button)
- **Save Veusz Project**: Save results to .vszh5 file (blue button)  
- **Close**: Exit application (red button)

## Typical Workflow

1. **Select Files**: Click "Browse ATR Files" and select multiple .atr files
2. **Configure Options**: 
   - Set plot title if desired
   - Enable/disable multiprocessing based on file count
   - Enable GPU if available and desired
   - Adjust CPU cores if needed
3. **Process**: Click "Process and Create Plots"
   - Watch progress bar for completion status
   - Monitor status messages for detailed information
4. **Save**: Click "Save Veusz Project" when processing completes
   - Choose save location for .vszh5 file
   - Optionally launch Veusz GUI automatically

## Features

### Multiprocessing
- Automatically recommended for multiple files
- Configurable worker count
- Significant speedup for large datasets

### GPU Acceleration  
- Supports CuPy (NVIDIA), OpenCL (cross-platform), Taichi
- Automatically detected and configured
- Fallback to CPU if GPU unavailable

### Real-time Feedback
- Progress bar shows completion percentage  
- Status log shows detailed processing information
- Non-blocking GUI allows interaction during processing

### File Management
- Visual file list with multi-selection support
- Easy add/remove files capability
- Duplicate file prevention

## Plot Outputs (Same as Original)
- Individual magnitude and phase plots
- Polar coordinate plots  
- Overlay plots for comparison
- High-precision .vszh5 format
- Legacy .vsz format backup

## Error Handling
- Detailed error messages with specific file information
- Graceful handling of corrupted files
- User-friendly error dialogs
- Comprehensive status logging
