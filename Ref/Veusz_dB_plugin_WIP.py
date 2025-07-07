# -*- coding: utf-8 -*-
"""
dB Averaging Dataset Plugin for Veusz.
This plugin converts dB datasets to linear magnitude, computes their average,
and creates both linear and dB averaged datasets.
Based on the original Veusz_AvgdB.py script.
Author: William W. Wallace
"""
# %% Module Import
import numpy as np
# import veusz.plugins as plugins
from veusz.plugins.datasetplugin import (
    DatasetPlugin, DatasetPluginException, Dataset1D, datasetpluginregistry,
    DatasetText
)
# DatasetPluginManager,
# TODO: see if loading DatasetPlugin gives access to helper and if this is needed
from veusz.plugins.datasetplugin import DatasetPluginHelper as hlpr
from veusz.plugins import (field)
import re
# %% Class Definitions
# %%% User Input Fields utilized


class dBLinearAvgPlugin(DatasetPlugin):
    """Dataset plugin to average dB datasets in linear domain."""
    # Plugin metadata
    menu = ("Signal Processing", "dB Average")
    name = "dB_Linear_Average"
    author = "William W. Wallace"
    description_short = "Average dB datasets in linear domain"
    description_full = (
        "Converts multiple dB datasets to linear magnitude, "
        "computes their average in linear domain, then provides "
        "both linear magnitude and dB averaged outputs. "
        "This is the correct way to average magnitude data "
        "expressed in dB."
    )

    def __init__(self):
        """Define input fields for the plugin."""
        self.fields = [
            field.FieldDatasetMulti(
                'input_datasets',
                'Input dB datasets'
            ),
            field.FieldText(
                'output_prefix',
                'Output dataset prefix',
                default='avg_'
            ),
            field.FieldText(
                'base_name',
                'Output dataset base name',
                default='avg'
            ),
            field.FieldText(
                'output_suffix',
                'Output dataset suffix',
                default='_avg'
            )
        ]

    def getDatasets(self, fields):
        """Define output datasets."""
        prefix = fields['output_prefix']
        suffix = fields['output_suffix']
        base_name = fields['base_name']
        if not prefix.strip() and not suffix.strip() and not base_name.strip():
            raise DatasetPluginException(
                "Output prefix, base name, AND suffix cannot be empty")
        # Create two output datasets: linear magnitude and dB
        self.linear_output = Dataset1D(
            f"{prefix}{base_name}_linear_mag{suffix}")
        self.db_output = Dataset1D(f"{prefix}{base_name}_dB{suffix}")
        return [self.linear_output, self.db_output]

    def updateDatasets(self, fields, helper):
        """Compute averaged datasets."""
        input_names = fields['input_datasets']
        if not input_names:
            raise DatasetPluginException("No input datasets selected")
        # Get input datasets
        try:
            input_datasets = helper.getDatasets(input_names, dimensions=1)
        except Exception as e:
            raise DatasetPluginException(
                f"Error getting input datasets: {str(e)}")
        if len(input_datasets) == 0:
            raise DatasetPluginException("No valid input datasets found")
        # Convert all dB datasets to linear magnitude
        linear_data = []
        for dataset in input_datasets:
            if dataset.data is None or len(dataset.data) == 0:
                continue
            # Convert dB to linear magnitude: linear = 10^(dB/20)
            linear_mag = 10.0 ** (dataset.data / 20.0)
            linear_data.append(linear_mag)
        if not linear_data:
            raise DatasetPluginException("No valid data to process")
        # Find the maximum length among all datasets
        max_length = max(len(data) for data in linear_data)
        # Pad shorter datasets with NaN to handle different lengths
        # required to create a data_matrix by np.array
        padded_data = []
        for data in linear_data:
            if len(data) < max_length:
                padded = np.full(max_length, np.nan)
                padded[:len(data)] = data
                padded_data.append(padded)
            else:
                padded_data.append(data)
        # Convert to matrix for averaging
        data_matrix = np.array(padded_data)
        # Compute average in linear domain (ignoring NaN values)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_linear = np.nanmean(data_matrix, axis=0)
        # Convert back to dB: dB = 20*log10(linear)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_db = 20.0 * np.log10(avg_linear)
        # Handle any infinite or invalid values

        avg_linear = np.where(np.isfinite(avg_linear), avg_linear, np.nan)
        avg_db = np.where(np.isfinite(avg_db), avg_db, np.nan)
        # Update output datasets
        self.linear_output.update(data=avg_linear)
        self.db_output.update(data=avg_db)


class dBToLinearPlugin(DatasetPlugin):
    """Dataset plugin to convert dB to linear magnitude."""
    menu = ("Signal Processing", "dB to Linear")
    name = "dB2Linear"
    author = "William W. Wallace"
    description_short = "Convert dB dataset to linear magnitude"
    description_full = (
        "Converts a dB magnitude dataset to linear magnitude "
        "using the formula: linear = 10^(dB/20)"
    )

    def __init__(self):
        """Define input fields."""
        self.fields = [
            field.FieldDataset(
                'input_dataset',
                'Input dB dataset'
            ),
            field.FieldDataset(
                'output_dataset',
                'Output dataset name'
            ),
        ]

    def getDatasets(self, fields):
        """Define output dataset."""
        output_name = fields['output_dataset']
        if not output_name.strip():
            raise DatasetPluginException("Output dataset name cannot be empty")
        self.output = Dataset1D(output_name)
        return [self.output]

    def updateDatasets(self, fields, helper):
        """Convert dB to linear magnitude."""
        input_name = fields['input_dataset']
        try:
            input_dataset = helper.getDataset(input_name, dimensions=1)
        except Exception as e:
            raise DatasetPluginException(
                f"Error getting input dataset: {str(e)}")
        if input_dataset.data is None or len(input_dataset.data) == 0:
            raise DatasetPluginException("Input dataset is empty")
        # Convert dB to linear magnitude: linear = 10^(dB/20)
        linear_data = 10.0 ** (input_dataset.data / 20.0)
        # Handle error bars if present
        linear_serr = None
        linear_perr = None
        linear_nerr = None
        if input_dataset.serr is not None:
            # For symmetric errors in dB, convert to linear domain
            # Error propagation: if y = 10^(x/20), then dy = y * ln(10)/20 * dx
            conversion_factor = linear_data * np.log(10) / 20.0
            linear_serr = conversion_factor * input_dataset.serr
        if input_dataset.perr is not None:
            conversion_factor = linear_data * np.log(10) / 20.0
            linear_perr = conversion_factor * input_dataset.perr
        if input_dataset.nerr is not None:
            conversion_factor = linear_data * np.log(10) / 20.0
            linear_nerr = conversion_factor * input_dataset.nerr
        self.output.update(
            data=linear_data,
            serr=linear_serr,
            perr=linear_perr,
            nerr=linear_nerr
        )


class LinearTodBPlugin(DatasetPlugin):
    """Dataset plugin to convert linear magnitude to dB."""
    menu = ("Signal Processing", "Linear to dB")
    name = "Linear2dB"
    author = "William W. Wallace"
    description_short = "Convert linear magnitude to dB"
    description_full = (
        "Converts a linear magnitude dataset to dB "
        "using the formula: dB = 20*log10(linear)"
    )

    def __init__(self):
        """Define input fields."""
        self.fields = [
            field.FieldDataset(
                'input_dataset',
                'Input linear dataset'
            ),
            field.FieldDataset(
                'output_dataset',
                'Output dataset name'
            ),
        ]

    def getDatasets(self, fields):
        """Define output dataset."""
        output_name = fields['output_dataset']
        if not output_name.strip():
            raise DatasetPluginException("Output dataset name cannot be empty")
        self.output = Dataset1D(output_name)
        return [self.output]

    def updateDatasets(self, fields, helper):
        """Convert linear magnitude to dB."""
        input_name = fields['input_dataset']
        try:
            input_dataset = helper.getDataset(input_name, dimensions=1)
        except Exception as e:
            raise DatasetPluginException(
                f"Error getting input dataset: {str(e)}")
        if input_dataset.data is None or len(input_dataset.data) == 0:
            raise DatasetPluginException("Input dataset is empty")
        # Convert linear to dB: dB = 20*log10(linear)
        # Handle zero and negative values
        with np.errstate(invalid='ignore', divide='ignore'):
            db_data = 20.0 * np.log10(np.abs(input_dataset.data))
        # Replace infinite values with a NaN
        db_data = np.where(np.isfinite(db_data), db_data, np.nan)
        # Handle error bars if present
        db_serr = None
        db_perr = None
        db_nerr = None

        if input_dataset.serr is not None:
            # Error propagation: if y = 20*log10(x), then dy = 20/(x*ln(10)) * dx
            conversion_factor = 20.0 / (input_dataset.data * np.log(10))
            conversion_factor = np.where(np.isfinite(
                conversion_factor), conversion_factor, np.nan)
            db_serr = np.abs(conversion_factor * input_dataset.serr)
        if input_dataset.perr is not None:
            conversion_factor = 20.0 / (input_dataset.data * np.log(10))
            conversion_factor = np.where(np.isfinite(
                conversion_factor), conversion_factor, np.nan)
            db_perr = np.abs(conversion_factor * input_dataset.perr)
        if input_dataset.nerr is not None:
            conversion_factor = 20.0 / (input_dataset.data * np.log(10))
            conversion_factor = np.where(np.isfinite(
                conversion_factor), conversion_factor, np.nan)
            db_nerr = -np.abs(conversion_factor * input_dataset.nerr)
        self.output.update(
            data=db_data,
            serr=db_serr,
            perr=db_perr,
            nerr=db_nerr
        )
# %%% Processing
# ----------------------------------------------------------------------
# INTERNAL HELPERS
# ----------------------------------------------------------------------


class _ConsoleMixin:
    """Safely write messages to Veusz' console or fall back to stdout."""
    @staticmethod
    def _log(helper, text):
        try:
            helper.doc.log(text)
        except Exception:  # pragma: no cover
            print(text)


class _MathHelpers:
    """Static helpers for dB / linear conversions and averaging."""
    @staticmethod
    def lin_from_db(arr):
        return 10.0 ** (arr / 20.0)

    @staticmethod
    def db_from_lin(arr):
        with np.errstate(invalid='ignore', divide='ignore'):
            out = 20.0 * np.log10(arr)
        return np.where(np.isfinite(out), out, np.nan)

    @staticmethod
    def pad(arr, N):
        if len(arr) < N:
            tmp = np.full(N, np.nan)
            tmp[: len(arr)] = arr
            return tmp
        return arr

    @staticmethod
    def average(arrs):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.nanmean(arrs, axis=0)


class _pluginUtilities:
    """Utilies that are common to multiple processes."""

    # def __init__(self, helper):
    #     """Set it up"""
    #     # self.tag_map = {}
    #     self._log(helper, f"Utilizing Wally Plugin Utilities!")

    @staticmethod
    def tagProcessing():
        """Get all tags and create a dict with datasets as values."""
        # if not hasattr(self, 'tag_map'):
        #     self.tag_map = {}
        from veusz.plugins.datasetplugin import DatasetPluginHelper as hlpr
        # TODO: Replace this with something meaningful
        inputHelper = []
        tag_map = {}
        # TODO: States this is not interatble, find out why, and fix
        # replace helper with something
        # helper is an object passed by veusz when the plugin is run, it is not
        # the import as defined in the comments in that section
        # but see if we can get the datasets and tags in a different way
        for ds_name in hlpr.datasets1d:
            if "freq" in ds_name.lower():
                # _ConsoleMixin._log(hlpr, f"[ByTag] excluded '{ds_name}'")
                print(f"[ByTag] excluded '{ds_name}'")
            else:
                # if it's not a frequency data container, process it
                ds = hlpr.getDataset(ds_name, dimensions=1)
                # without helper this would be Root. something,
                # TODO: Correct this... need to find access method to tags
                tags = inputHelper._doc.data[ds_name].tags
                # we have the tag, now we build a map where we have a dict of tags
                # with the dataset names as values
                for tag in tags:
                    tag_map.setdefault(tag, []).append(ds_name)
        # Check after loop completion
        if not tag_map:
            raise DatasetPluginException("No tagged datasets found.")
        return tag_map

# %%% Processing by Tag
# # ----------------------------------------------------------------------
# # "PROCESS BY TAG" - one result per tag
# # -----
class dBLinearAvgByTagPlugin(_ConsoleMixin, _MathHelpers, _pluginUtilities,
                             DatasetPlugin):
    """Process all 1D data by tag, treat it as dB, avg in linear."""

    """Create data sets of linear and dB resutls of the average. """
    # Plugin metadata
    menu = ("Signal Processing", "Process by Tag", "Average By Tag")
    name = "dB_Linear_Average"
    author = "William W. Wallace"
    description_short = "Average dB datasets in linear domain"
    description_full = (
        "Converts all 1D datasets by tage to linear magnitude unless `freq` is in the name, "
        "computes their average in linear domain, then provides "
        "both linear magnitude and dB averaged outputs. "
        "This is the correct way to average magnitude data "
        "expressed in dB."
    )

    def __init__(self):
        """Initialize the plugin - only define fields here."""
        # super().__init__()
        # Don't create datasets or call updateDatasets() here
        self.fields = [
            field.FieldText(
                'meaningless_field',
                'Just a Field, Everything is Automated.',
                default='Meaningless'
            )
        ]
    # helper is an object passed at run time by veusz, getDatasets does not
    # get his helper....
    def getDatasets(self, fields):
        """Return empty list - datasets are created dynamically."""
        # TODO: see if you can use the DatasetPlugin to get all the tags
        # and file names
        self.tag_map = self.tagProcessing()  # Call as static method
        # TODO: create the datasets by tags
        self.dBAvgData = {}
        self.linAvgData = {}
        for tag in self.tag_map.items():
            try:
                # output_name = fields['output_dataset']
                # if not output_name.strip():
                #     raise DatasetPluginException("Output dataset name cannot be empty")
                # self.output = Dataset1D(output_name)
                # return [self.output]
                output_name_dB = str(tag) + '_avg_dB'
                output_name_lin = str(tag) + '_avg_lin'

                self.dBAvgData[tag] = Dataset1D(output_name_dB)

                self.linAvgData[tag] = Dataset1D(output_name_lin)
                # Don't Forget to Set tags of these sets

            except Exception as exc:
                self._log(helper, f"[ByTag] tag '{tag}' datasets created.")
        return [self.dBAvgData, self.linAvgData]

    def updateDatasets(self, fields, helper):
        """Process datasets and create outputs."""
        # Build tag map as an instance variable
        # TODO: See if calling external function is the issue
        self.tag_map = self.tagProcessing(helper)  # Call as static method
        # self.tag_map = {}
        # for ds_name in helper.datasets1d:
        #     if "freq" in ds_name.lower():
        #         _ConsoleMixin._log(helper, f"[ByTag] excluded '{ds_name}'")
        #     else:
        #         # if it's not a frequency data container, process it
        #         ds = helper.getDataset(ds_name, dimensions=1)
        #         tags = helper._doc.data[ds_name].tags
        #         # we have the tag, now we build a map where we have a dict of tags
        #         # with the dataset names as values
        #         for tag in tags:
        #             self.tag_map.setdefault(tag, []).append(ds_name)
        # # Check after loop completion
        # if not self.tag_map:
        #     raise DatasetPluginException("No tagged datasets found.")
        # Process each tag group
        for tag, group in self.tag_map.items():
            try:
                # Convert dataset names to dataset objects
                datasets = [helper.getDataset(
                    ds_name, dimensions=1) for ds_name in group]
                # Process dataset objects
                lin = [self.lin_from_db(ds.data) for
                       ds in datasets if ds.data is not None]
                if not lin:  # Skip if no valid data
                    self._log(helper, f"[ByTag] tag '{tag}' has no valid data")
                    continue
                N = max(map(len, lin))
                avg_lin = self.average([self.pad(a, N) for a in lin])
                avg_db = self.db_from_lin(avg_lin)

                # self.linear_output.update(data=avg_linear)
                # self.db_output.update(data=avg_db)
                # Create The Data Sets and then update them
                # helper.createDataset does not exist!
                # dB
                self.dBAvgData[tag].update(data=avg_db)
                # dBOut = helper.createDataset(f"{tag}_dB_avg", create_new=True)
                # dBOut.update(data=avg_db)

                # Linaer Data set
                self.linAvgData[tag].update(data=avg_lin)
                # linOut = helper.createDataset(
                #     f"{tag}_lin_avg", create_new=True)
                # linOut.update(data=avg_lin)
                # self._log(
                #     helper, f"[ByTag] wrote '{tag}_dB_avg' & '{tag}_lin_avg'")
            except Exception as exc:
                self._log(helper, f"[ByTag] tag '{tag}' -> {exc}")


# class dBToLinearByTagPlugin(_ConsoleMixin, _MathHelpers, DatasetPlugin):
#     menu = ("Signal Processing", "Process By Tag", "dB to Linear")
#     name = "dB2Linear_By_Tag"
#     author = "William W. Wallace"
#     description_short = "Convert dB->linear by tag."
#     description_full = (
#         "Each tag group is converted from dB to linear magnitude and "
#         "averaged. Result stored as _avg (linear units)."
#     )
#     def __init__(self):
#         # TODO: make a field for tag drop downs or at least user entry
#         # then apply to all data in that tag
#         pass
#     def getDatasets(self, fields):
#         """Return empty list - datasets are created dynamically."""
#         return []
#     def updateDatasets(self, fields, helper):
#         """Process datasets and create outputs."""
#         # Build tag map as an instance variable
#         pluginUtilities.tagProcessing(self, helper)
# class LinearTodBByTagPlugin(_ConsoleMixin, _MathHelpers, DatasetPlugin):
#     menu = ("Signal Processing", "Process By Tag", "Linear to dB")
#     name = "Linear2dB_By_Tag"
#     author = "William W. Wallace"
#     description_short = "Convert linear->dB by tag."
#     description_full = (
#         "Each tag group is converted from linear magnitude to dB, then "
#         "averaged. Result stored as _avg (dB)."
#     )
#     # TODO: make a field for tag drop downs or at least user entry
#     # then apply to all data in that tag
#     def getDatasets(self, fields):
#         """Return empty list - datasets are created dynamically."""
#         return []
#     def updateDatasets(self, fields, helper):
#         """Process datasets and create outputs."""
#         # Build tag map as an instance variable
#         pluginUtilities.tagProcessing(self, helper)
# ----------------------------------------------------------------------
# REGISTER PLUGINS
# ----------------------------------------------------------------------
# By Tag
datasetpluginregistry.append(dBLinearAvgByTagPlugin)
# datasetpluginregistry.append(dBToLinearByTagPlugin)
# datasetpluginregistry.append(LinearTodBByTagPlugin)
# By Selected Data
# datasetpluginregistry.append(dBLinearAvgSelectedPlugin)
# datasetpluginregistry.append(dBToLinearSelectedPlugin)
# datasetpluginregistry.append(LinearTodBSelectedPlugin)
# By User Input
datasetpluginregistry.append(dBLinearAvgPlugin)
datasetpluginregistry.append(LinearTodBPlugin)
datasetpluginregistry.append(dBToLinearPlugin)
