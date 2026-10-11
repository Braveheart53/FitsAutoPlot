# AutoPlot native Veusz export

Version: 0.1.0  
Date: 2026-10-11

The FITS and Franks applications now generate native `.vszh5` projects with NumPy and h5py. Creating a project no longer imports `veusz.embed`, starts an embedded Veusz process, calls `ImportFileFITS`, or asks Veusz to save it.

## What to install

Keep all six Python modules together: `_autoplot_native.py`, `_autoplot_common.py`, `FITS_AutoPlot.py`, `Franks_AutoPlot.py`, and both corresponding `_plugin.py` files. Do not combine the new engines with the older common helper.

For the tested Python 3.8 environment:

```sh
python -m pip install -r requirements-py38.txt
python FITS_AutoPlot.py
# or
python Franks_AutoPlot.py
```

The pinned requirements are for the standalone applications. When installing the plugins into an existing Veusz environment, retain the Qt binding used by that host rather than replacing it; add the missing NumPy, Astropy, h5py, qtpy and psutil dependencies there.

The supplied `setup_py3p8_nrao.sh` now installs h5py instead of requiring Veusz. Its default source directory is the directory containing the installer, not a hard-coded user repository path; the installer was syntax-checked, not executed to create a complete conda environment.

## Workflow changes

- **Standalone GUIs:** Parse files and build an in-memory native document, then use the existing Save control. Full, minimized, and optional full-companion outputs are supported.
- **Plugins:** Still register in Veusz's Tools menu, but now export a separate `.vszh5` project. They deliberately leave the currently open Veusz document untouched. New plugin fields are `output_file`, `minimized` and `also_full`; a blank output filename opens a save dialog.
- **Viewer:** Live embedded preview has been removed. After saving, “Open in Veusz” can launch an optional external viewer; `VEUSZ_EXECUTABLE` can identify that executable, or it can be found on `PATH`.
- **Veusz imports:** The writer and standalone engines do not import Veusz. The two plugin entry points use `veusz.plugins` only for host menu registration. Optional validation utilities in `tests/` intentionally use Veusz as an independent consumer, never as part of generation.
- **FITS backend:** Astropy is the only reader. Historical `both` and `veusz` arguments remain accepted as aliases, but they do not invoke a second reader or create live source links.
- **Raw naming:** FITS raw column names are now consistently `base__HDU__column__raw`. Sorted and overlay naming conventions are retained.

## ISP GUI implementation reused

The recovered original is `isp_speed_monitor.py` version **1.0.0**, dated **2026-09-04**, in the section titled `Veusz native HDF5 writer`. Its native writer already existed before the later numbered packaging plan: the later “Step 1” delivery was script 1.3.0 / bundle 1.7.0, not the initial introduction of native export. Only the original 1.0.0 source was recovered for this migration; the line numbers below refer to that verified source, not to unrecovered later versions.

| ISP source location | Purpose | AutoPlot implementation |
|---|---|---|
| `epoch_to_veusz()`, lines 205–208 | Seconds relative to 2009-01-01 | Existing `mjd_to_veusz_seconds()` retained |
| `VszDoc`, lines 1032–1054 | Builds `Add`, `To`, `Set` command text | `NativeWidget._write()` and `NativeDocument.SaveToString()` |
| `_escape_hdf_name()`, lines 1057–1058 | Escapes `/` and backticks in flat HDF5 names | `_escape_name()` |
| `VeuszWriter`, lines 1061–1108 | Writes HDF5 dataset groups and document script | `NativeDocument.save()` |
| `_add_time_graph()`, lines 1111–1169 | Constructs graphs and datetime axes | Existing AutoPlot page builders retained |
| `export_veusz()`, lines 1172–1324 | Combines datasets and plot command text | `push_to_veusz()` / `push_franks_to_veusz()`, overlay builders, and native save |
| `export_all()`, lines 1327–1342 | Connects export to the GUI workflow | Existing Save handlers and shared plugin export helpers |

The central ISP code was:

```python
# isp_speed_monitor.py v1.0.0, VeuszWriter.__init__ / finish
self.f = h5py.File(path, "w")
self.root = self.f.create_group("Veusz")
self.root.attrs["vsz_format"] = 1
self.data = self.root.create_group("Data")
self.docgrp = self.root.create_group("Document")
# ...write typed arrays and tags...
self.docgrp["document"] = [document_text.encode("utf-8")]
self.f.close()
```

This is the export step requested: write typed arrays and the plot-building command script directly into one HDF5 file. AutoPlot adapts the methodology rather than importing the ISP application or copying its unrelated database, networking or GUI logic.

## Native output structure

The format was cross-checked against `Document.saveToHDF5File`, the dataset serializers, and `loadHDF5Doc` in the tagged Veusz 3.4 and 4.1 source trees [1] ([Veusz repository](https://github.com/veusz/veusz)). The writer uses the following structure:

```text
/Veusz
    attributes: vsz_format=1, vsz_version="3.4", vsz_saved_at,
                autoplot_writer_version="0.1.0"
    /Data/<escaped-dataset-name>
        attribute: vsz_datatype="1d" | "2d" | "date" | "text"
        data
        serr / nerr / perr                  when present
        xrange / yrange / edge / centre data when present
    /Document
        document                           one UTF-8 byte string in a 1-element array
        /Tags/<tag>                        retained logical dataset names
```

`vsz_version="3.4"` identifies the compatibility baseline, not a Veusz runtime used for generation. Flat dataset-group names use backtick escaping; original names remain in `vsz_name` attributes. The native command model emits literal `Add`, `To` and `Set` statements plus `SetCompatLevel(0)`, with explicit graph axes and safe Python string representations.

## Behavior retained and fixes verified

- **Date views:** `_dt` retains the original numeric x values plus date annotations. `_dt_labels` uses numeric seconds since 2009-01-01 and `mode='datetime'`; no unintended conversion of raw MJD to seconds has been introduced.
- **Data integrity:** Numeric NaNs, numeric error arrays, Unicode text, tags, 2-D image data, channel-pair grouping, sorting, gap-string parsing and datetime label density are retained.
- **Minimized save:** An exact whitelist comes from dataset references in the in-memory plot model. Only those datasets and their applicable tags are serialized; the full model is never pruned or restored. Missing references fail before the destination is touched.
- **Atomic writes:** Output is written to a same-directory temporary file and replaced only after a successful close. An interrupted or failed write does not deliberately truncate an existing project.
- **Compression:** Nonempty numeric arrays use lossless gzip plus shuffle. Full and minimized output share the same writer.
- **Legend fix:** `hide_keys` now sets the key widget's `hide` property, rather than hiding only its border.
- **Tagged-page fix:** Internal `__datelabels_*__` bucket metadata is excluded from the measurement-column list. Previously its MJD array could trigger an ambiguous NumPy truth-value exception or create spurious plots.
- **Plugin-loader fix:** Both Veusz loader styles, including execution without `__file__`, are handled. Registered classes are imported into real module namespaces so their callbacks retain the correct globals.
- **Plain-text captions:** Widget labels and legend captions escape `_`, `^`, `{` and `}`. Dataset names and raw text contents are not rewritten.
- **Datetime layout:** Native datetime graphs default to a 3 cm bottom margin and 8 pt tick labels unless explicitly overridden, so the long rotated dates fit within the page.
- **Parallelism:** File-level parsing remains concurrent. This migration is not a claim of maximum parallelism or a measured speedup; no timing benchmark was used to select thread counts.

## Tests and limitations

Run the native tests without Veusz:

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
python tests/make_demo_projects.py examples
```

The test suite deliberately blocks Veusz imports. It exercises both engines, both GUI constructors, both plugin export workflows, full/minimized output, typed datasets, NaNs, tags, errors, name escaping, literal serialization, missing-reference errors, atomic-write failure cleanup and Python 3.8 grammar.

For optional consumer validation in a separate Veusz-equipped environment:

```sh
python tests/validate_plugin_host.py
python tests/validate_in_veusz.py rendered examples/*.vszh5
```

See `VALIDATION_v0.1.0.md` for the exact environments and observed results. The examples contain only synthetic measurements; no recovered ISP campaign data or original user measurement files are included in the release archive.

The native model intentionally implements the AutoPlot subset of widget and dataset operations, not the entire Veusz API. Dataset expressions must be materialized before use; a general-purpose `.vszh5` importer, arbitrary Veusz-document editing, live embedded rendering, and source-linked data are outside this migration.

## Reference details

| Reference | Full bibliographic entry | DOI or URL |
|---|---|---|
| [1] | J. S. Sanders and Veusz contributors, “Veusz scientific plotting application,” GitHub, source tags `veusz-3.4` and `veusz-4.1`, accessed Oct. 11, 2026. Format source: `veusz/document/doc.py`, `veusz/document/loader.py`, `veusz/datasets/{oned,twod,text,date}.py`, and `veusz/utils/utilfuncs.py`. | [Veusz source repository](https://github.com/veusz/veusz) |
