# -*- coding: utf-8 -*-
# %% Header
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

# event loop QApplication.exec_()
# %% Import all required modules
# %%% GUI Uses
# %%%% Tkinter
import tkinter as tk
from tkinter.filedialog import askopenfilenames
from tkinter import filedialog
from tkinter import messagebox
# %%%% Py Qt imports
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
# from PyQt6.QtCore import Qt, QSize
# %%%% PySide6 Imports
# from PySide6.QtGui import *
# from PySide6.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
# from PySide6.QtCore import Qt, QSize
# %%%% qtpy imports
from qtpy.QtGui import *
from qtpy.QtWidgets import (QApplication, QPushButton, QMainWindow, QLabel,
                            QWidget)
from qtpy.QtCore import Qt, QSize

# %%% Math Modules
# import pandas as pd
import xarray as xr
import numpy as np
import skrf as rf
import mpmath as mpm
import scipy.constants as cons
from astropy.time import Time as tme

# %%% Unit Conversion
# import pint as pint
# ureg = pint.UnitRegistry()

# %%% System Interface Modules
import os
import os.path as pathCheck
import time as time
import sys
# add something to the python sys path
# sys.path.append(os.path.abspath("something"))


# %%% Plotting Environment
import veusz.embed as vz
import pprint

# %%% File type Export/Import
import h5py as h5
from scipy.io import savemat

# %%% Parallelization
# from multiprocessing import Pool  # udpate when you learn it!
# from multiprocessing import Process
# from multiprocess import Process
# from multiprocess import Pool

# %%% GPU Implementation
# %%%% Cuda
# import cupy as cp
# from cuda import cuda, nvrtc  # need to learn this as well, see class below
# %%%% Rocm Hips
# from hip import hip
# from hip import hiprtc

# %% Add to Python Sysytem Path for calling in scripts
# add_subdirs_to_path("somedir")

# %% Locally Global constants
c_0 = cons.speed_of_light  # in meters
pi = mpm.pi  # using mpmath versus numpy, there is a reason for this, just
# not sure what that reason is at the moment

# %% Class Definitions
# Begin Class definitions based upon use cases for range and data


class AntennaFun:
    """
    All Antenna Functions from a previous life...
    """

    def __init__(self):
        pass

    def DolphChebby_layout():
        # See jasik 2-20
        print('Not yet implemented')
        return ()
        # end dolphchecby function

    def SGH_BWApprox_ByDim(Horn_Ap_Eplane, Horn_Ap_Hplane, freq_tune, Horn_Body_len=0,
                           Horn_WG_a=0, Horn_WG_b=0, antenna_efficiency=0.522,
                           units_O_len='m', units_O_freq='Hz', units_O_angle='deg'):
        # typicallay Horn_WG_a = 2*horn_wg_b
        # units are Hz, m, s for input
        # HPBW_Eplane = 15
        # See notes dates 2019-04-29 and Jasik 2-14 as well as NAVAIR "EW and RADAR
        # Systems Handbook" section 3-1.4
        # example of calling the function after importing to Ant
        # (Eplane,Hplane,Gain,floater) = Ant.SGH_BWApprox_ByDim(16.25,
        #   21.94,1e9,units_O_len='in',Horn_Body_len=21,Horn_WG_a=,Horn_WG_b=0)
        UnitConversions = Antenna_unit_conversion(
            units_O_len, units_O_freq, units_O_angle)
        freq_tune_Hz = freq_tune * UnitConversions[1]
        lambda_tune_Hz = c_0/freq_tune_Hz
        Dimension_array = np.array([Horn_Body_len, Horn_Ap_Eplane, Horn_Ap_Hplane, Horn_WG_a,
                                    Horn_WG_b])
        Dimension_array = Dimension_array * UnitConversions[0]

        Physical_Aperture = Dimension_array[1] * Dimension_array[2]
        Affective_Aperture = antenna_efficiency * Physical_Aperture

        Antenna_Gain_linear = 4*pi*Affective_Aperture/(lambda_tune_Hz**2)
        Antenna_Gain_dB = 10*mpm.log(Antenna_Gain_linear, 10)
        Antenna_Gain_dB_float = 10*np.log10(float(Antenna_Gain_linear))

        # Gain = 4*pi /(HPBW_az*HPBW_el), here this is 4pi square steradians
        # therefore, to convert it would be 4*pi*(180/pi)**2 ~= 4*pi*57.3**2 ~=
        # 41,253 degrees
        # see also
        # http://www.cs.binghamton.edu/~vinkolar/directional/Direct_antennas.pdf
        # as well as the NAVAIR EW and Radar Handbook section 3-3.12
        # note the difference in the definiton of H-plone HPBW.

        # need to calculate 10 dB BW from sinc^2 function
        # can use expansion technique of the sinc^2 fuction to approximate.
        HPBW_Hplane = 70*lambda_tune_Hz/Dimension_array[2]
        HPBW_Eplane = 56*lambda_tune_Hz/Dimension_array[1]

        return (HPBW_Eplane, HPBW_Hplane, Antenna_Gain_dB, Antenna_Gain_dB_float)
        # End of SGH_BWApprox

    def tapered_design():
        # place holder for completed tapered design
        return ()
        # end tapered_design function

    def plot_RF_data(dataToPlot, plotType, XAxisMin='calc', XAxisMax='calc',
                     YAxisMin='calc', YAxisMax='calc',
                     ZAxisMin='calc', ZAxisMax='calc', YAxisLabel=None,
                     XAxisLabel=None, ZAxisLabel=None, spherical_R=None,
                     spherical_Theta=None, spherical_Phi=None, smithType='z',
                     VSWRCircle=True, refZ=50, smithLabels=True, smithRad=1,
                     PlotTitle=None, plotLegendLocation=5, savePlot=True,
                     saveLocation='E:/Product_Data/Raw_Net/Non-Production-testing/PythonOutput',
                     savePlotType=['pdf', 'png', 'eps', 'jpg', 'tiff'], units_O_len='m',
                     units_O_freq='Hz', units_O_angle='deg'):
        # dataToPlot == can be array or file name, need to check for which)
        # plotType == smith, polar, rectangular, scatter, line etc. Need to ensure
        # VSWRCircle == can be list of numbers or boolean for default
        # checks for correct data format

        # good reference:
        # https://scikit-rf.readthedocs.io/en/latest/tutorials/Plotting.html
        from skrf import plotting
        from plotting import save_all_figs
        from matplotlib import pyplot as plt
        from matplotlib import style

        UnitConversions = Antenna_unit_conversion(units_O_len, units_O_freq,
                                                  units_O_angle)
        # figure out what type of data is in the passed information
        # look at error plotting skrf.plotting.plot_calibration_errors
        if pathCheck.isfile(dataToPlot):
            fileTypeExtension = pathCheck.splitext(dataToPlot)[-1].lower()
            fileType = fileTypeExtension[-3:]
            if fileTypeExtension == ".s2p":
                formattedDataPlotted = rf(dataToPlot)
                formattedDataPlotted.plot_s_smith(chart_type=smithType,
                                                  draw_vswr=VSWRCircle, ref_imm=refZ,
                                                  draw_labels=smithLabels,
                                                  smithR=smithRad)
                save_all_figs(saveLocation, savePlotType)
                # stop development here for this function 2019-10-17 need to continue
                # with file type and saving of plotted data as well as custome
                # plot formats and arrangements
            elif fileTypeExtension == ".s1p":
                formattedDataPlotted = None
                print('not yet implemented')

            elif fileTypeExtension == ".csv":
                formattedDataPlotted = None
                print('not yet implemented')
        else:
            # then dataToPlot is a data array
            # need to check for validity of it as well.
            formattedDataPlotted = None
            print('not yet implemented')
        # if dataToPlot.lower().endswith(('.png', '.jpg', '.jpeg'))
        return (formattedDataPlotted)

    def specular_calc(Range_length, Tx_ht, Rcv_ht, Fresnel_zones, units_O_len,
                      units_O_freq, Tuned_Freq, Low_freq, High_freq):
        # User Input Required
        # =============================================================================
        #     Range_length = 14.5
        #     Tx_ht = 6
        #     Rcv_ht = 6
        #     Fresnel_Zones = 6
        #     units_O_len = 'ft'
        #     units_O_freq = 'GHz'
        #     Tuned_freq = 2.45
        #     Low_freq = 0.5
        #     High_freq = 8
        # =============================================================================

        UnitsUsed = Antenna_unit_conversion(units_O_len, units_O_freq)

        # Should replace with use of unit conver function
        # =============================================================================
        #     # Need to put in a case statedment for the unit_O_freq, right now treated it as
        #     # GHz
        #     def Hz():
        #         return 1
        #     def kHz():
        #         return 1e3
        #     def MHz():
        #         return 1e6
        #     def GHz():
        #         return 1e9
        #
        #     # All calculations completed in Hz
        #     freq_unit = {'Hz' : Hz,
        #                'kHz' : kHz,
        #                'MHz' : MHz,
        #                'GHz' : GHz
        #                }
        #
        #     def meter():
        #         return 1
        #     def foot():
        #         return 100/2.54/12
        #     def inch():
        #         return 100/2.54
        #     def centimeter():
        #         return 100
        #     def kilometer():
        #         return 0.001
        #     def millimeter():
        #         return 1000
        #
        #     # all calculations will be completed in meters
        #     length_unit = {'m' : meter,
        #                    'ft' : foot,
        #                    'in' : inch,
        #                    'cm' : centimeter,
        #                    'km' : kilometer,
        #                    'mm' : millimeter
        #                    }
        #
        #     Len_unit = length_unit[units_O_len]()   # get the unit conversion factor
        # =============================================================================

        # convert lengths to meters
        Len_unit = UnitsUsed[0]
        freq_unit = UnitsUsed[1]
        Range_length_U = Range_length / Len_unit
        Rcv_ht_U = Rcv_ht / Len_unit
        Tx_ht_U = Tx_ht / Len_unit

        # tuned wavelength in terms of chosen length unit U denotes in meters
        Tuned_lambda_U = (c_0 / (Tuned_freq * freq_unit[units_O_freq]()))
        Tuned_lambda = Tuned_lambda_U / Len_unit  # wavelength in chosen units
        Low_lambda_U = c_0 / (Low_freq * freq_unit)
        High_lambda_U = c_0 / (High_freq * freq_unit)

        # an indirect pathlength may be calculated here, but is not need at the moment.

        # present data in chosen length unit
        print("At Lowest Frequency")
        specular_patch_dims = fresnel_zone_calc(Fresnel_Zones, Low_lambda_U,
                                                Range_length_U, Rcv_ht_U, Tx_ht_U)
        Fresnel_Cn_units = specular_patch_dims['Cn'] * Len_unit
        Fresnel_Ln_units = specular_patch_dims['Ln'] * Len_unit
        Fresnel_Wn_units = specular_patch_dims['Wn'] * Len_unit
        print('Center of Specular Area:', Fresnel_Cn_units, units_O_len)
        print('Length of Specular Area Ellipse:',
              Fresnel_Ln_units, units_O_len)
        print('Width of Specular Area Ellipse:', Fresnel_Wn_units, units_O_len)
        del specular_patch_dims, Fresnel_Cn_units, Fresnel_Ln_units, Fresnel_Wn_units

        print()

        print("At Tuned Frequency")
        specular_patch_dims = fresnel_zone_calc(Fresnel_Zones, Tuned_lambda_U,
                                                Range_length_U, Rcv_ht_U, Tx_ht_U)
        Fresnel_Cn_units = specular_patch_dims['Cn'] * Len_unit
        Fresnel_Ln_units = specular_patch_dims['Ln'] * Len_unit
        Fresnel_Wn_units = specular_patch_dims['Wn'] * Len_unit
        print('Center of Specular Area:', Fresnel_Cn_units, units_O_len)
        print('Length of Specular Area Ellipse:',
              Fresnel_Ln_units, units_O_len)
        print('Width of Specular Area Ellipse:', Fresnel_Wn_units, units_O_len)
        del specular_patch_dims, Fresnel_Cn_units, Fresnel_Ln_units, Fresnel_Wn_units

        print()

        print("At Highest Frequency")
        specular_patch_dims = fresnel_zone_calc(Fresnel_Zones, High_lambda_U,
                                                Range_length_U, Rcv_ht_U, Tx_ht_U)
        Fresnel_Cn_units = specular_patch_dims['Cn'] * Len_unit
        Fresnel_Ln_units = specular_patch_dims['Ln'] * Len_unit
        Fresnel_Wn_units = specular_patch_dims['Wn'] * Len_unit
        print('Center of Specular Area:', Fresnel_Cn_units, units_O_len)
        print('Length of Specular Area Ellipse:',
              Fresnel_Ln_units, units_O_len)
        print('Width of Specular Area Ellipse:', Fresnel_Wn_units, units_O_len)

        # Now do a piece count based on this and it turned 45 degrees

        # should most likely
        # end function for specular patch calc

    def Antenna_unit_conversion(units_O_len, units_O_freq, units_O_angle='deg'):
        # =============================================================================
        #     units_O_len = 'ft'
        #     units_O_freq = 'GHz'
        #
        # =============================================================================
        # GHz
        def Hz():
            return 1

        def kHz():
            return 1e3

        def MHz():
            return 1e6

        def GHz():
            return 1e9

        # All calculations completed in Hz
        freq_unit = {'Hz': Hz,
                     'kHz': kHz,
                     'MHz': MHz,
                     'GHz': GHz
                     }

        def meter():
            return 1

        def foot():
            return 12*2.54/100

        def inch():
            return 2.54/100

        def centimeter():
            return 100

        def kilometer():
            return 1e3

        def millimeter():
            return 1e-3

        # all calculations will be completed in meters
        length_unit = {'m': meter,
                       'ft': foot,
                       'in': inch,
                       'cm': centimeter,
                       'km': kilometer,
                       'mm': millimeter
                       }

        def deg():
            return 1

        def rad():
            return cons.pi/180

        angle_unit = {'deg': deg,
                      'rad': rad}

        # get the unit conversion factor
        Len_unit = length_unit[units_O_len]()
        freq_unit = freq_unit[units_O_freq]()
        angle_unit = angle_unit[units_O_angle]()

        return (Len_unit, freq_unit, angle_unit)
        # better to return a dict?
        # end antenna units function

    def fresnel_zone_calc(Fresnel_Zones, Tuned_lambda_U, Range_length_U, Rcv_ht_U, Tx_ht_U):
        # Script Calculations
        # may have to use mpm.atan
        Psi = mpm.atan((Tx_ht_U + Rcv_ht_U) / Range_length_U)
        Psi_degrees = (180/cons.pi)*Psi
        pathLength_direct = mpm.sqrt(
            Range_length_U**2 + (Rcv_ht_U + Tx_ht_U)**2)
        F1 = ((Fresnel_Zones * Tuned_lambda_U) /
              (2 * Range_length_U)) + mpm.sec(Psi)
        F2 = (Rcv_ht_U**2 - Tx_ht_U**2) / ((F1**2 - 1) * Range_length_U**2)
        F3 = (Rcv_ht_U**2 + Tx_ht_U**2) / ((F1**2 - 1) * Range_length_U**2)

        Fresnel_Cn = Range_length_U * (1-F2)/2
        Fresnel_Ln = Range_length_U * F1 * mpm.sqrt(1 + F2**2 - 2*F3)
        Fresnel_Wn = Range_length_U * \
            mpm.sqrt((F1**2 - 1) * (1 + F2**2 - 2*F3))

        # define a way to calculate piece count, remember patch is 45 degrees so its 3' instead of 2
        # most likely a different def / function
        # a different function is not required. Divide or mod by 24*sqr(2)/2 and use 45
        # degrees off extremeties
        # return round numbers and piece count, total and width with length
        # OR return all three exact, rounded or 45 layout, and number of pieces
        # with length width piece number

        return {'Cn': Fresnel_Cn, 'Ln': Fresnel_Ln, 'Wn': Fresnel_Wn}
        # end of fresnel_zone_calc

    def illumination_O_Range(Antenna_physical_tilt_angle, HPBW, TendB_BW,
                             Height_of_aperture, units_O_len='ft',
                             units_O_freq='GHz', units_O_angle='degrees'):
        # need to populate with NRL_arch_illumination_code
        # @@@ Left of here, need to copy over and adjust code for NRL_arch_illumination @@@
        # constants
        c_0 = cons.speed_of_light  # in meters

        # Variables
        # H-plane for DH6500 at 6.5 GHz 18,24
        # E-plane for DH6500 at 6.5 GHz 18,24
        # H-plane for DH6500 at 12 GHz 10,20
        # E-plane for DH6500 at 12 GHz 10,20
        # =============================================================================
        #     Antenna_physical_tilt_angle = 10
        #     HPBW = 16
        #     TendB_BW = 26
        #     Height_of_aperture = 5 #vertical height of the aperture center
        #     units_O_len = 'ft'
        #     units_O_freq = 'GHz'
        #     units_O_angle = 'degress'
        # =============================================================================
        Dimensional_units = Antenna_unit_conversion(
            units_O_len, units_O_freq, units_O_angle)
        # get the unit conversion factor, divide by his to get meters
        Len_unit = Dimensional_units[0]
        # get the unit conversion factor, multiple freq by this to get Hz
        Freq_unit = Dimensional_units[1]
        Angle_unit = Dimensional_units[2]

        Height_of_aperture = Height_of_aperture/Len_unit    # conver to meters

        # tangent of the required angles
        if units_O_angle == 'deg':
            tanTilt = mpm.tan(mpm.radians(Antenna_physical_tilt_angle))
            tanHPBW = mpm.tan(mpm.radians(HPBW/2))
            tanTendB_BW = mpm.tan(mpm.radians(TendB_BW/2))
        elif units_O_angle == 'rad':
            tanTilt = mpm.tan(mpm.degrees(Antenna_physical_tilt_angle))
            tanHPBW = mpm.tan(mpm.degrees(HPBW/2))
            tanTendB_BW = mpm.tan(mpm.degrees(TendB_BW/2))
        else:
            messagebox.showerror('Illumination Error',
                                 'Unit for angle is neither rad nor deg.')
            sys.exit()

        # Calculated lead and trailing distance from direct ray
        trial1_HPBW = Height_of_aperture*((tanTilt)+(tanHPBW))
        trial2_HPBW = Height_of_aperture*(((tanTilt + tanHPBW) /
                                           (1-tanTilt*tanHPBW))-tanTilt)
        trial1_10dBBW = Height_of_aperture*((tanTilt)+(tanTendB_BW))
        trial2_10dBBW = Height_of_aperture*(((tanTilt + tanTendB_BW) /
                                            (1-tanTilt*tanTendB_BW))-tanTilt)
        # Calc HP illumination distance
        HP_Illumination_distance = Height_of_aperture*(
            (((tanTilt + tanHPBW)/(1-tanTilt*tanHPBW))-tanTilt) +
            (tanTilt)+(tanHPBW))
        # Calc distance to 10 dB BW
        TendB_Illumination_distance = Height_of_aperture*(
            (((tanTilt + tanTendB_BW)/(1-tanTilt*tanTendB_BW))-tanTilt) +
            (tanTilt)+(tanTendB_BW))

        # Text Output
        # =============================================================================
        #     print("Half Power BW Illumination_distance: ",HP_Illumination_distance,
        #           " m")
        #     print("Half Power BW Illumination_distance: ",HP_Illumination_distance*
        #           Len_unit,units_O_len)
        #     print("10 dB BW Illumination_distance: ",TendB_Illumination_distance," m")
        #     print("10 dB BW Illumination_distance: ",TendB_Illumination_distance*
        #           Len_unit,units_O_len)
        #
        #     # Leading and trailing distances
        #     print("--")
        #     print("Half Power BW Illumination_distance 1: ",trial1_HPBW," m")
        #     print("Half Power BW Illumination_distance 1: ",trial1_HPBW*
        #           Len_unit,units_O_len)
        #     print("10 dB BW Illumination_distance 1: ",trial1_10dBBW," m")
        #     print("10 dB BW Illumination_distance 1: ",trial1_10dBBW*
        #           Len_unit,units_O_len)
        #     print("--")
        #     print("Half Power BW Illumination_distance 2: ",trial2_HPBW," m")
        #     print("Half Power BW Illumination_distance 2: ",trial2_HPBW*
        #           Len_unit,units_O_len)
        #     print("10 dB BW Illumination_distance 2: ",trial2_10dBBW," m")
        #     print("10 dB BW Illumination_distance 2: ",trial2_10dBBW*
        #           Len_unit,units_O_len)
        # =============================================================================
        return (HP_Illumination_distance*Len_unit,
                TendB_Illumination_distance*Len_unit, trial1_HPBW*Len_unit,
                trial2_HPBW*Len_unit, trial1_10dBBW*Len_unit, trial2_10dBBW*Len_unit)
        # end of illuminationORange


class GBOutDoor:
    """
    All functions for GBO outdoor range
    """

    def __init__(self):
        pass

    def import_data(self, data: tuple):
        """
        Parameters
        ----------
        tuple data : TYPE
            DESCRIPTION.

        Returns
        -------
        data_imported : TYPE
            DESCRIPTION.

        """
        print("Test data import")
        pass

    def parse_data(self, data: tuple):
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

    def select_atr_files():
        # file selection dialog, need to update for any files other than current
        # outdoor range files
        filename = askopenfilenames(filetypes=[("Antenna Range Files",
                                                ".ATR")])

        # start the main loop for processing the selected files
        for mainLooper in range(len(filename)):
            # this loop processes the files selected one at a time, while combining
            # the data as it progresses

            # get the file parts for further use of the current file.
            fileParts = os.path.split(filename[mainLooper])

        # After the mainloop, I need to combine all the data into a multi-dimensional
        # array. Then call Veusz and parse the data into that gui.

    class ATR_Plot_App:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("ATR Plotter")

            # File selection
            self.file_label = tk.Label(self.root, text="Select ATR file:")
            self.file_label.pack()
            self.file_entry = tk.Entry(self.root, width=50)
            self.file_entry.pack()
            self.file_button = tk.Button(self.root, text="Browse",
                                         command=GBOutDoor.ATR_Plot_App.select_atr_file(self))
            self.file_button.pack()

            # Plot name input
            self.plot_name_label = tk.Label(self.root, text="Enter plot name:")
            self.plot_name_label.pack()
            self.plot_name_entry = tk.Entry(self.root, width=50)
            self.plot_name_entry.pack()

            # Dataset name input
            self.dataset_name_label = tk.Label(
                self.root, text="Enter dataset name:")
            self.dataset_name_label.pack()
            self.dataset_name_entry = tk.Entry(self.root, width=50)
            self.dataset_name_entry.pack()

            # Create plot button
            create_button = tk.Button(
                self.root, text="Create Plot", command=self.create_plot)
            create_button.pack()

            # Status label
            self.status_label = tk.Label(self.root, text="")
            self.status_label.pack()

        def select_atr_file(self):
            """
            Select A single ATR file for processing

            Returns
            -------
            file_path : TYPE
                DESCRIPTION.
            file_entry : TYPE
                DESCRIPTION.

            """
            file_path = filedialog.askopenfilename(
                filetypes=[("GBO Outdoor Test Range files", "*.atr")])
            if file_path:
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, file_path)
            # return file_path, file_entry

        def create_plot(self):
            file_path = self.file_entry.get()
            plot_name = self.plot_name_entry.get()
            dataset_name = self.dataset_name_entry.get()

            if not file_path or not plot_name or not dataset_name:
                self.status_label.config(text="Please fill in all fields")
                return

            try:
                # Read ATR file
                line_number = 13  # remeber it is zero indexed
                header_info, selected_data = (
                    GBOutDoor.ATR_Plot_App.process_data(file_path,
                                                        line_number, self))

                # take the numpy and call it df
                df = selected_data

                # partse header info for future use, remember if the ATR
                # format changes this MUST be changed
                header_lines = header_info.splitlines()
                # freqs are ion MHz in the ATR file
                freq_max = float(header_lines[8].split(":")[-1].strip())
                freq_min = float(header_lines[9].split(":")[-1].strip())
                freq_step = float(header_lines[10].split(":")[-1].strip())

                if freq_min != freq_max:
                    print("This script is not yet capable of reducing " +
                          "data that consists of multiple frequencies. " +
                          "Contact William Wallace at x2216 and ask him " +
                          "to implement this feature.")
                    return
                else:
                    # once plotting of multiple files at once is made, this
                    # array will have to index with the various data sets
                    # by length
                    freq_array = [freq_max]

                Polarization_plane = float(
                    header_lines[5].split(":")[-1].split()[0])  # in deg

                # As of 2025-02-21 The outdoor range is only capable of single
                # elevation scans (at 0 el)
                Az_min = float(header_lines[6].split(":")[-1].strip())
                Az_max = float(header_lines[7].split(":")[-1].strip())
                # All steps in az for the GBO outdoor range are by default
                # 1 degrees in a continual scan
                Az_angles = np.arange(Az_min, Az_max + 1, 1, dtype=float)

                # Create Veusz embedded window
                embed = vz.Embedded('Veusz', hidden=False)
                embed.EnableToolbar()

                # Create a new document
                # doc = embed.Root.AddDocument()

                # Create a new page
                page = embed.Root.Add('page')

                # Create a new graph
                graph = page.Add('graph', name='graph1', autoadd=False)

                # Set plot title
                # incorrect graph.title.val = plot_name

                # First add an dimension to the 2D array in the first axis
                # then parse out the data
                # df = np.expand_dims(df, axis=3)

                # add frequency to the matrix
                data_freq = freq_array
                data_mag = df[:, 0]
                data_phase = df[:, 1]
                data_Az_angle = Az_angles

                # iloc is a pandas function....
                # freq_data = df.iloc[:, 0].tolist()
                # mag_data = df.iloc[:, 1].tolist()
                # phase_data = df.iloc[:, 3].tolist()

                data_col_stack = np.column_stack(
                    (data_Az_angle, data_mag, data_phase)
                )

                # trying xr.dataarray
                data_full_2 = xr.DataArray(data_col_stack,
                                           coords=[data_Az_angle, data_freq],
                                           dims=["Azimuth", "frequnecy"]

                                           attrs=dict(description=(
                                               "Full Dataset for the associated " +
                                               "outdoor " +
                                               "range measurement.")
                                           )
                                           )

                # Use Xarray to create a dataset
                # a dataset in xarray may not be needed here, a dataarray
                # may be best, but also astropy tables may work better here
                # looking into this, leaving any items in work
                data_full = xr.Dataset(
                    data_vars=dict(magnitude=(["freq", "Az"], data_mag),
                                   phase=(["freq", "Az"], data_phase)
                                   ),
                    coords=dict(
                        frequency=("freq", data_freq),
                        Azimuth=("Az", data_Az_angle)
                    ),
                    attrs=dict(description=(
                        "Full Dataset for the associated " + "outdoor " +
                        "range measurement.")
                    )
                )

                # Create datasets
                embed.SetData(dataset_name + '_freq', data_freq)
                embed.SetData(dataset_name + '_mag', data_mag)
                embed.SetData(dataset_name + '_phase', data_phase)

                # Create xy plot
                xy = graph.Add('xy', name=dataset_name)
                xy.xData.val = dataset_name + '_x'
                xy.yData.val = dataset_name + '_y'

                # Create a new Axis
                # note that in xy.xAxis, this can be changed to match the give name
                x_axis = graph.Add('axis', name='frequency')
                y_axis = graph.Add('axis', name='counts')
                x_axis.label.val = 'frequency (MHz)'
                y_axis.label.val = 'Counts'
                y_axis.direction.val = 'vertical'
                # x_axis.childnames gives you all the settable parameters
                x_axis.autoRange.val = 'Auto'
                xy.xAxis.val = 'frequency'
                xy.yAxis.val = 'counts'
                xy.marker.val = 'none'
                xy.PlotLine.color.val = 'red'
                xy.PlotLine.width.val = '1pt'

                # Add Page or Graph Title

                # Save The Veusz file
                saveDir = os.path.dirname(file_path)
                embed.Save(saveDir + '/' + dataset_name + '.vsz')

                # Show the plot
                embed.WaitForClose()

                self.status_label.config(text="Plot created successfully")
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")

        def process_data(file_path, line_number, self):
            # Read the entire file into a list of lines
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Extract header information
            header_lines = lines[:line_number]
            header_info = ''.join(header_lines).strip()

            # Extract the specified line of numerical data
            data_line = lines[line_number].strip()

            # Remove the last character if it's '#'
            if data_line.endswith('#'):
                data_line = data_line[:-1]

            # Convert the line of numerical data into a list of numbers
            data_numbers = list(map(float, data_line.split()))

            # Select every other item from the list of numbers
            selected_phase_data = data_numbers[::2]
            selected_magnitude_data = data_numbers[1::2]
            # selected_data = np.array([[selected_magnitude_data],
            #                           [selected_phase_data]])
            selected_data = np.array([selected_magnitude_data,
                                     selected_phase_data])
            selected_data_transpose = selected_data.transpose()
            return header_info, selected_data_transpose

        def run(self):

            # run the GUI!
            self.root.mainloop()


class GBOAnechoic:
    """
    All functions specifically for anechoic range

    """

    def __init__(self):
        pass
    # Place Walters Python Code for plotting phase centers, ensure it
    # uses GUI elements to choose files


class GBOExport:
    """
    All functions for data formatting and export
    """

    def __init__(self):
        pass


class GPUFun:
    """
    GPU functions for use of Cuda
    See: https://nvidia.github.io/cuda-python/overview.html

    It does take setup of the kernal
    """

    def __init__(self):
        pass

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

    def __init__(self):
        pass


class Astronomy:
    """
    Used at the moment to create various functions from the AstroPy module.
    """

    def __init__(self):
        pass

    def mjdconvert_time(mjd_time, *args):
        """


        Parameters
        ----------
        mjd_time : TYPE
            Take in 'mjd'

        *args : Varargin
            Various variables or arguments taken for format and scales

        Returns
        -------
        iso_time : String
            return iso time in a string or object

        """

        # =============================================================================
        #         Python 3.8 does not have a built-in switch or case statement
        # like some other programming languages. However, the functionality of
        # a case statement can be achieved through several alternative methods:
        # 1. Dictionary Mapping
        # This approach uses a dictionary to map cases to their corresponding
        # actions.
        # def case_statement(argument):
        #     options = {
        #         'case1': lambda: "Action for case 1",
        #         'case2': lambda: "Action for case 2",
        #         'case3': lambda: "Action for case 3"
        #     }
        #     return options.get(argument, lambda: "Default action")()
        #
        # print(case_statement('case2'))
        # # Expected output: Action for case 2
        #
        # print(case_statement('case4'))
        # # Expected output: Default action
        #
        # 2. If-elif-else Statements
        # This is the most straightforward way to mimic a case statement.
        # def case_statement(argument):
        #     if argument == 'case1':
        #         return "Action for case 1"
        #     elif argument == 'case2':
        #         return "Action for case 2"
        #     elif argument == 'case3':
        #         return "Action for case 3"
        #     else:
        #         return "Default action"
        #
        # 3. Class-based approach
        # For more complex scenarios, a class can be used to handle cases.
        # class CaseStatement:
        #     def case1(self):
        #         return "Action for case 1"
        #
        #     def case2(self):
        #         return "Action for case 2"
        #
        #     def default(self):
        #         return "Default action"
        #
        #     def execute(self, argument):
        #         method_name = 'case' + argument[4:] if argument.startswith('case') else 'default'
        #         method = getattr(self, method_name, self.default)
        #         return method()
        #
        # case_executor = CaseStatement()
        # print(case_executor.execute('case1'))
        # # Expected output: Action for case 1
        #
        # print(case_executor.execute('case4'))
        # # Expected output: Default action
        #
        # Python 3.10 and later: match-case statement
        # It's worth noting that Python 3.10 introduced the match-case
        # statement, which provides a more direct way to implement case-like
        # logic. However, this is not available in Python 3.8.
        # def case_statement(argument):
        #     match argument:
        #         case 'case1':
        #             return "Action for case 1"
        #         case 'case2':
        #             return "Action for case 2"
        #         case 'case3':
        #             return "Action for case 3"
        #         case _:
        #             return "Default action"
        #
        #
        # Generative AI is experimental.
        #
        # =============================================================================

        # --> if MJD not in UTC, can add scale flag: scale='tdb'
        t = tme(mjd_time, format="mjd")
        t_out = []
        if len(args) > 0:
            for arg in args:
                # case statement, as match and case is not introduced until Python
                # 3.10
                # go through the args and create a t_out for each
                # match arg:
                #     case 'iso':
                #         t_out.append(t.iso)
                #     case _:
                #         t_out.append(t.iso)
                #         return t_out
                t_out.append(Astronomy.case_4_mjd_timeFormat(t, t_out, arg))
                print(t_out)
        else:
            t_out.append(Astronomy.case_4_mjd_timeFormat(t, t_out, "iso"))

        # print the returned format(s)
        print(t_out)
        return t_out

    def case_4_mjd_timeFormat(timeObjIn, timeObjOut, optionString):
        """
        Case statement for changing time FORMATs.

        Annother case statement for changing time scales is required.Subformats
        are not yet implemented, but should be as **kwargs for ease of use.

        Parameters
        ----------
        timeObjIn : TYPE
            DESCRIPTION.
        timeObjIn : TYPE
            DESCRIPTION.
        optionString : TYPE
            DESCRIPTION.

        Returns
        -------
        Case option for given input optionString
            DESCRIPTION.

        References
        -------
        https://docs.astropy.org/en/stable/time/index.html
        """
        timeList = []
        options = {
            'iso': lambda: timeList.append(timeObjIn.iso),
            'fits': lambda: timeList.append(timeObjIn.fits),
            'isot': lambda: timeList.append(timeObjIn.isot),
            'gps': lambda: timeList.append(timeObjIn.gps),
            'cxcsec': lambda: timeList.append(timeObjIn.cxcsec),
            'decimalyear': lambda: timeList.append(timeObjIn.decimalyear),
            'jd': lambda: timeList.append(timeObjIn.jd),
            'julian': lambda: timeList.append(timeObjIn.jd),
            'jyear': lambda: timeList.append(timeObjIn.jyear),
            'jyear_str': lambda: timeList.append(timeObjIn.jyear_str),
            'mjd': lambda: timeList.append(timeObjIn.mjd),
            'plot_date': lambda: timeList.append(timeObjIn.plot_date),
            'unix': lambda: timeList.append(timeObjIn.unix),
            'unix_tai': lambda: timeList.append(timeObjIn.unix_tai),
            'yday': lambda: timeList.append(timeObjIn.yday),
            'ymdhms': lambda: timeList.append(timeObjIn.ymdhms),
            'yyymmddHHmmss': lambda: timeList.append(timeObjIn.ymdhms),
            'datetime64': lambda: timeList.append(timeObjIn.datetime64)
        }
        # options.setdefault('iso')
        # print(options.get(optionString,
        # lambda: timeObjOut.append(timeObjIn.iso))())
        timeObjOut = options.get(optionString,
                                 lambda: timeList.append(timeObjIn.iso))()
        return timeList


class CSVFun:
    """
    Used for processing CSV Data
    """

    def __init__(self):
        pass

    def select_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)


def add_subdirs_to_path(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        sys.path.append(dirpath)


def main():
    """
    execution of main function.
    """
    # =============================================================================
    # The main script is below. First we import the files through a dialog. Then
    # utilize the classes and their functions to process the data.
    # =============================================================================

    # Just showing use of the mjd conversion function.
    mjd = 60686.3539
    newTime = Astronomy.mjdconvert_time(mjd, "iso", "ymdhms", "fits")
    # newTime = Astronomy.mjdconvert_time(mjd)
    print(newTime)

    ATR_Example = GBOutDoor.ATR_Plot_App()
    ATR_Example.run()

    # sys.path.append(
    #     os.path.abspath("C:\\Users\\wwallace\\OneDrive - National Radio Astronomy Observatory" +
    #                     "\\Documents\\Python_Vault")
    # )


if __name__ == "__main__":
    main()
