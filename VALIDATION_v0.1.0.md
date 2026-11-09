# Native AutoPlot validation

Version: 0.1.0  
Date: 2026-10-11

These results come from executing the updated code, inspecting the resulting HDF5 files, and loading/rendering them in separate Veusz consumer environments. Veusz was not used to generate the projects.

## Native generation

The final **15-test regression suite passed on CPython 3.8.20**. During those tests, an import hook deliberately made every `veusz` import unavailable; the standalone engines, GUI constructors and both plugin export workflows still completed. The optional plugin-registration import was caught as designed.

Tested generation dependencies:

| Package | Version |
|---|---:|
| NumPy | 1.24.4 |
| Astropy | 5.2.2 |
| h5py | 3.11.0 |
| qtpy | 2.4.3 |
| PyQt5 | 5.15.11 |
| psutil | 7.2.2 |

The suite covers typed arrays, symmetric errors, NaNs, Unicode and empty text, escaped dataset names, date datasets, 2-D images, tags, exact reference-based minimization, atomic-write failures, missing references, safe literal serialization, plain-text caption escaping, widget-name collisions, broken-axis replacement, the gap parser, tagged/untagged FITS, Franks data, both GUI constructors and both plugin workflows. All six production Python modules also passed Python 3.8 grammar parsing and descending semantic-version-header checks.

Both plugin workflow tests supplied a host object that raises on every attempted access. Both exports succeeded, demonstrating that they do not mutate the host document to create their output.

## Independent Veusz consumer checks

| Consumer environment | Native HDF5 load and SVG rendering | Actual plugin loading, dialog construction and registered export |
|---|---|---|
| Veusz 4.1, built from tagged source, CPython 3.13 / Qt6 | Passed | Both plugins passed |
| Veusz 3.6.2, distribution package, CPython 3.13 / Qt5 | Passed | Both plugins passed |
| Veusz 3.4 Python source, CPython 3.13 / Qt5, with 3.6.2 compiled helper extensions | Passed | Both plugins passed |

The third environment is a **3.4-source compatibility harness**, not an unmodified Veusz 3.4 binary installation. It exercises the 3.4 HDF5 loader, widget/settings code and plugin loader with compatible compiled helper extensions; it does not replace a final smoke test on the user's exact 3.4 installation.

Both registered plugin callbacks also exported synthetic input files successfully in all three consumer environments. The host document's widget text and dataset-name set were compared before and after each invocation and remained unchanged.

The final six packaged synthetic projects were regenerated after the datetime-margin adjustment. Each consumer loaded and rendered all **58 pages** across those six files:

| Synthetic project | Minimized datasets | Full datasets | Pages per file |
|---|---:|---:|---:|
| Tagged FITS with image and datetime variants | 9 | 15 | 7 |
| Untagged FITS with image and datetime variants | 9 | 13 | 7 |
| Franks numeric, annotation and datetime variants | 23 | 33 | 15 |

The packaged `examples/datetime_axis_preview.png` is a rasterized SVG exported by the independent Veusz 4.1 consumer, not a Matplotlib reconstruction. It shows the synthetic datetime overlay with hidden legend and a deliberate large time gap; date labels were visually checked after increasing the bottom margin.

## Checks using attached measurement data

Two FITS files and one nonempty Franks file from the attached archive were processed into additional full/minimized test projects. All four resulting files loaded and rendered in the three consumer environments above: nine pages per FITS project and 33 pages per Franks project.

| Input subset | Full datasets | Minimized datasets | Full bytes | Minimized bytes |
|---|---:|---:|---:|---:|
| Two FITS files, with per-file and overlay variants | 620 | 490 | 4,258,501 | 2,561,949 |
| One Franks file, with per-file and overlay variants | 79 | 53 | 1,231,192 | 574,000 |

These are measurements of the newly generated test projects, not a claim that the older attached 1,858-dataset documents were converted or repaired. The original attachments were left unchanged, and private measurement outputs are not included in the release archive.

## Boundaries

- **Installer:** `bash -n` passed. A full conda installation was not run.
- **Operating system:** Runtime tests were performed on Linux. Windows launch behavior and the user's exact Veusz installations still warrant a smoke test.
- **Performance:** No timing benchmark or claim of maximum parallelization is included. File-parsing concurrency remains in place.
- **Coverage:** The native model supports the operations used by these AutoPlot builders, not every Veusz widget, dataset expression or command.
- **Dependencies:** Native generation needs h5py and the stated GUI/scientific dependencies. It does not need Veusz; actual plugin registration and optional viewing naturally require a Veusz host/viewer.

References: None; this validation report uses local execution results, not external publications.
