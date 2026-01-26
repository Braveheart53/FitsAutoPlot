# Quick Reference Guide

## Installation (5 minutes)

```bash
# Step 1: Create virtual environment
python -m venv autoplot
source autoplot/bin/activate

# Step 2: Install all dependencies
pip install scikit-rf numpy scipy pandas matplotlib pyside6 PyPDF2 veusz

# Step 3: Verify
python -c "import skrf; print('Ready!')"

# Step 4: Run either script
python Touchstone_AutoPlot.py        # Smith chart enhancement
python CSV_TSV_AutoPlot.py           # CSV/TSV visualizer
```

---

## TouchstoneSmithMpl.py - Quick Start

### File → Browse Files
- Select .s1p, .s2p, .s3p, .s4p Touchstone files

### Smith Chart Analysis Tab
```
Configuration:
  Chart Type:              z (Impedance) or y (Admittance)
  Reference Impedance:     50.0 Ω (or your standard)
  Draw Labels:             ☑ (show impedance values)
  Draw VSWR Circles:       ☑ (show 1.5, 2.0, 3.0)
  
Output Format:            PNG | PDF | TIFF | BMP | SVG | JPG

PDF Options:
  ☑ Combine all plots into single PDF (with bookmarks)

Generate → Select Output Directory → Done
```

### Output
- Individual image files OR
- Single multi-page bookmarked PDF (if selected)
- All S-parameters (S11, S12, S21, S22, etc.)

---

## CSV_TSV_AutoPlot.py - Quick Start

### File → Browse Files
- Select .csv, .tsv, or .txt files

### Configure Delimiter
```
Option 1 (Recommended):
  Click "Auto-Detect Delimiter"

Option 2 (Presets):
  Click one of:  Comma | Tab | Semicolon | Pipe

Option 3 (Custom):
  Type your delimiter in the text box
```

### Select Columns for Plotting
```
X-Axis Column:        [Dropdown - select one column]
Y-Axis Columns:       [List - select one or more columns]
                      (Hold Ctrl/Cmd for multi-select)
```

### Configure Plot
```
Plot Title:           [Enter title]
X-Axis Label:         [Enter X label]
Y-Axis Label:         [Enter Y label]
☑ Show Legend:        [Toggle to show/hide]
```

### Generate
```
Live Preview:         [Updates in real-time as you change settings]
Generate Plots in Veusz:  [Creates final plot in Veusz]
  → Select output directory → Option to open in Veusz GUI
```

---

## Output Formats Explained

### Smith Chart Formats (TouchstoneSmithMpl.py)

| Format | Best For | Quality | File Size | Editable |
|--------|----------|---------|-----------|----------|
| **PNG** | Web, slides | Raster (good) | Medium | No |
| **PDF** | Documents, printing | Vector (best) | Large | Limited |
| **TIFF** | Archival | Raster (best) | Largest | No |
| **BMP** | Legacy systems | Raster (good) | Large | No |
| **SVG** | Web, infinite zoom | Vector | Small | Yes |
| **JPG** | Compression needed | Raster (lossy) | Smallest | No |

**Recommendation:** PNG for general use, PDF for printing/documents

### Veusz Format (CSV_TSV_AutoPlot.py)

| Format | Best For | Editable | Notes |
|--------|----------|----------|-------|
| **.vszh5** | Final result | Yes | Open in Veusz GUI to modify |

---

## Delimiter Examples

```
Comma (CSV):
  value1,value2,value3
  10,20,30

Tab (TSV):
  value1  value2  value3
  10      20      30

Semicolon:
  value1;value2;value3
  10;20;30

Pipe:
  value1|value2|value3
  10|20|30

Custom (colon):
  value1:value2:value3
  10:20:30
```

---

## Keyboard Shortcuts

### Both Applications
| Key | Action |
|-----|--------|
| Ctrl+O | Open/Browse files |
| Ctrl+Q | Quit application |
| Ctrl+C | Copy (in tables) |

### CSV_TSV_AutoPlot
| Key | Action |
|-----|--------|
| Ctrl+Click | Multi-select Y columns |
| Shift+Click | Range select columns |

---

## Configuration Reference

### Smith Chart (TouchstoneSmithMpl.py)

```python
SmithChartMatplotlibConfig:
  figure_size = (10, 10)              # Figure size in inches
  dpi = 150                           # Resolution (100-300)
  title_fontsize = 14                 # Title font size
  label_fontsize = 12                 # Axis label size
  line_width = 2.0                    # Trace line width
  marker_size = 6.0                   # Point marker size
  grid_alpha = 0.3                    # Circle visibility (0-1)
```

### Plot Styling (CSV_TSV_AutoPlot.py)

```python
PlotConfig:
  x_scale = "linear" or "log"         # X-axis type
  y_scale = "linear" or "log"         # Y-axis type
  line_style = "-"                    # - (solid), -- (dashed), etc.
  line_width = 2.0                    # Line thickness
  marker_style = "o"                  # o, s, ^, v, D, *, +, x
  marker_size = 6.0                   # Marker size in points
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "No module named..." | `pip install [module_name]` |
| Slow file loading | Close other applications, reduce file size |
| Matplotlib not displaying | `pip install --upgrade matplotlib` |
| Veusz not launching | Install: `pip install veusz` |
| PDF merge fails | Install: `pip install PyPDF2` |
| Delimiter not detected | Use preset or manual entry |
| Data preview empty | Check delimiter setting |

---

## Data Requirements

### For Smith Charts (TouchstoneSmithMpl.py)

**Input:** Touchstone files
- Extensions: .s1p, .s2p, .s3p, .s4p, .sp
- Format: Touchstone 1.0 or 2.0
- S-parameter data required
- Frequency range: any

**Output:** Image files + optional PDF
- Each S-parameter gets its own plot
- 2×2 network = 4 S-parameters = 4 plots

### For CSV Visualization (CSV_TSV_AutoPlot.py)

**Input:** Delimited text files
- Extensions: .csv, .tsv, .txt, or any text file
- Content: Numeric data with header row
- Format: Any delimiter character
- Rows: Unlimited

**Output:** Veusz project file
- Format: .vszh5 (can be edited in Veusz GUI)
- Editable: Yes
- Re-plottable: Yes

---

## Example Workflow

### Workflow 1: Smith Chart Documentation

```
Step 1: Run TouchstoneSmithMpl.py
Step 2: Browse → Select network_design.S2P
Step 3: Smith Chart tab → Set chart type to "z"
Step 4: Output Format → Select "PDF"
Step 5: Check "Combine all plots into single PDF"
Step 6: Click Generate
Step 7: Select Documents folder
Step 8: Get smith_charts.pdf with all S-parameters
Step 9: Include in technical report
```

### Workflow 2: Lab Measurement Analysis

```
Step 1: Export measurement data as CSV from equipment
Step 2: Run CSV_TSV_AutoPlot.py
Step 3: Browse → Select measurement.csv
Step 4: Click "Auto-Detect Delimiter" (usually comma)
Step 5: X-axis column → time (or frequency)
Step 6: Y-axis columns → select all measurements
Step 7: Set title: "Lab Measurement Results"
Step 8: View live preview (blue trace shows data)
Step 9: Generate Plots in Veusz
Step 10: View in Veusz GUI, export as PDF
```

### Workflow 3: Multi-file Comparison

```
Step 1: Run CSV_TSV_AutoPlot.py
Step 2: Browse → Select test1.csv, test2.csv, test3.csv
Step 3: Delimiter auto-detected
Step 4: Select same X and Y columns for all
Step 5: Generate Veusz plots for each
Step 6: Compare PDFs or Veusz documents side-by-side
```

---

## Tips & Tricks

### Smith Charts
- **Zoom:** Use matplotlib's zoom tools in interactive window
- **Colors:** Trace is blue, start is green, end is red
- **Resolution:** Use TIFF for printing, PNG for screen
- **Bookmarking:** PDF bookmarks help with navigation

### CSV/TSV Data
- **Large files:** Pre-split files >1GB before loading
- **Missing data:** Handled automatically (skipped)
- **Multiple formats:** One delimiter works for all loaded files
- **Preview:** Export preview as image before generating final plot
- **Scaling:** Toggle linear/log for different data ranges

---

## Default Settings

### TouchstoneSmithMpl.py
```
Reference Impedance:        50.0 Ω
Chart Type:                 z (Impedance)
Draw Labels:                ON
Draw VSWR Circles:          ON
Output Format:              PNG
Combine PDF:                OFF
Figure DPI:                 150
Figure Size:                10×10 inches
```

### CSV_TSV_AutoPlot.py
```
Delimiter:                  , (comma)
Encoding:                   utf-8
Skip Header:                OFF
Skip Blank Lines:           ON
X Scale:                    linear
Y Scale:                    linear
Show Legend:                ON
Show Grid:                  ON
Marker Style:               circle (o)
Line Style:                 solid (-)
```

---

## Performance Benchmarks

### Typical Processing Times

| Task | Time | System |
|------|------|--------|
| Load Touchstone (.S2P) | <1s | Modern PC |
| Generate 1 Smith chart | <2s | Modern PC |
| Generate 4 Smith charts | <5s | Modern PC |
| Generate + Combine PDF (4 charts) | <10s | Modern PC |
| Load CSV (10,000 rows) | <1s | Modern PC |
| Update preview | <500ms | Modern PC |
| Generate Veusz plot | <3s | Modern PC |

---

## System Requirements

### Minimum
- OS: Windows 7+, macOS 10.13+, Linux (any)
- Python: 3.8 or higher
- RAM: 2 GB
- Disk: 500 MB free

### Recommended
- OS: Windows 10+, macOS 12+, Linux (recent)
- Python: 3.10 or higher
- RAM: 4 GB
- Disk: 2 GB free
- GPU: Optional (for acceleration in Touchstone processing)

---

## Getting Help

### Within the Application
- Status Log shows all operations
- Error messages indicate what went wrong
- Hover tooltips explain settings

### External Help
- Check Implementation_Guide.md for detailed docs
- Python errors usually indicate missing dependencies
- Veusz errors indicate installation issues

### Common Help Topics
- **Installation errors:** Reinstall Python + pip
- **Module not found:** Use `pip install [module]`
- **Display issues:** Update matplotlib
- **File not found:** Check file path and permissions
- **Permission denied:** Run as administrator

---

## Version Information

**Current Version:** 1.0 (2026-01-26)

**Components:**
- TouchstoneSmithMpl.py v1.0
- CSV_TSV_AutoPlot.py v1.0
- Implementation_Guide.md v1.0
- Quick_Reference.md (this file) v1.0

---

## Support Resources

| Resource | Purpose |
|----------|---------|
| Implementation_Guide.md | Complete documentation |
| README_First.md | Overview and use cases |
| Inline code comments | Understanding code |
| Docstrings | Function documentation |
| Status log | Real-time operation tracking |

---

**Ready to use. Happy plotting!**

