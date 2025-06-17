# -*- coding: utf-8 -*-
# =============================================================================
# Author: William W. Wallace
# This is a set of post processing classes and functions utilized for post
# processing data originating from Green Bank Observatory Antenna Ranges.
#
# Options will be included to export in hdf5 format, csv, excel, or matlab .mat
# file format. These are optional, the default method will be to initialize
# Veusz, populate the data, and create plots therin. All aftorementioned
# formats
# will be options to export all selected files into single files of multi-
# dimensional arrays given polarization information.
#
# A GUI will be created using these functions for ease of use in a different
# file
#
# --- Revision History----
# 0.0.1: Initial Draft and structure, learn and use parallel processing and
#           or cuda for this
# =============================================================================

# Import all required modules
import skrf as rf
import veusz.embed as veusz
import time
import numpy as np
import h5py
import os as os
import pprint
# Cupy is numpy implementation in CUDA for GPU use, need to learn more
import cupy
# Parallel Processing Modules
from multiprocessing import Pool  # udpate when you learn it!
from multiprocessing import Process
# Enable saving to .mat files
from scipy.io import savemat
# Cuda for parallel processing
from cuda import cuda, nvrtc  # need to learn this as well, see class below
# tkinter, which is a tcl wrapper just for dialogs
from tkinter.filedialog import askopenfilenames


# Begin Class definitions based upon use cases for range and data
class GBOutDoor:
    """
    All functions for GBO outdoor range
    """

    def import_data(self, tuple data):
        """
        Parameters
        ----------
        tuple data : TYPE
            DESCRIPTION.

        Returns
        -------
        data_imported : TYPE
            DESCRIPTION.


        Notes
        -------
        Import and parse data from GBO outdoor range, original script name
        was import_ATR_file_inVeusz.py looking at using Pandas or Xarray to
        organize data fields.

        """
        pass
        print("import_data")
        # return data_imported

    def parse_data(self, tuple data):
        """

        Parameters
        ----------
        tuple data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass


class GBOAnechoic:
    """
    All functions specifically for anechoic range:

        calc_FF:: calculate far field separation
        calc_Fresnel:: calculate fresnel zones

        essentially rewrite all code previously used for chamber measurements
        and design and implement herein as functions
    """
    pass


class GBOExport:
    """
    All functions for data formatting and export:

        export_Mat:: function for exporting data into a matlab compatible file
        export_Vuesz:: function to export into Veusz and create some basic
            plots

    """
    pass


class GPUFun:
    """
    GPU functions for use of Cuda
    See: https://nvidia.github.io/cuda-python/overview.html

    It does take setup of the kernal
    """
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError('Unknown error type: {}'.format(error))

    def checkCudaErrors(result):
        if result[0].value:
            raise RuntimeError("CUDA error code={}({})".format(
                result[0].value, _cudaGetErrorEnum(result[0])))
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]


class ParaProcessing:
    """
    Parallel processing class if need, may remove in the future
    """
    pass


def main():
    """
    execution of main function
    """
    # =============================================================================
    # Description
    # =============================================================================

    # file selection dialog, need to update for any files other than current
    # outdoor range files
    # =============================================================================
    #     filename = askopenfilenames(filetypes=[("Antenna Range Files",
    #                                             ".ATR")])
    #
    #     # start the main loop for processing the selected files
    #     for mainLooper in range(len(filename)):
    #         # this loop processes the files selected one at a time,
    #           while combining
    #         # the data as it progresses
    #
    #         # get the file parts for further use of the current file.
    #         fileParts= os.path.split(filename[mainLooper])
    #
    #     # After the mainloop, I need to combine all the data into a
    #       multi-dimensional
    #     # array. Then call Veusz and parse the data into that gui.
    # =============================================================================


if __name__ == "__main__":
    main()
