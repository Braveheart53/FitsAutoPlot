"""Import Data into Vuesz from 2port Touchstone File."""
# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul 11 16:32:12 2022
#
# basic Scikit-rf info
# https://scikit-rf.readthedocs.io/en/latest/tutorials/Networks.html#Basic-Properties
#
# Scikir-RF network interface info
# https://scikit-rf.readthedocs.io/en/latest/api/network.html
#
# @author: wwallace
# William W. Wallace
# Sandbox for scikit-RF play
# Rev G
# created file pre and append strings  at the beginning of  the script
# added in S12 and S21 processing in both frequency and time domain
# need to alter all this to be an import plugion for Veusz or call Veusz
# directly
# F: changed the time domain options.
# G: added auto data tagging with file name
#  : created a function to setdata rather than multiple iterations
#  : added file name filter for s1p and s2p, need to generalize the entire
#    script to do snp files and update with wild card in filter
#  : made it import multiple files at once
#
#
# a little lesson for cmath and scikit-rf
# below, currentNet[1,2,1] is [freq zero index, port num 1 indx,
#                              port num 1 indx]
# can also be accessed by currentNet.s21[1].s
# other options such as scikit-rf complex_to_degree
# https://scikit-rf.readthedocs.io/en/latest/api/mathFunctions.html
# to extract real: currentNet[1,2,1].s.real
# image: currentNet[1,2,1].s.imag
# phase: cmath.phase(currentNet[1,2,1].s) in rad
# phase: np.rad2deg(cmath.phase(currentNet[1,2,1].s))
# magnitude: abs(currentNet[1,2,1].s)
# =============================================================================
import os as os
import skrf as rf  # sci-kit RF module
import numpy as np

# import veusz --for direct invocation

# from matplotlib import pyplot as plt  # use for plotting figures
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator)
# from pylab import *
# from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilenames  # for file import dialog

# from tkinter.filedialog import askdirectory


def SDataVuesz(
    DataSetStringRoot, sub_Nets, SubNet2Write, string1, string2, idx
):
    """Create a veusz data set."""
    """idx == loop index"""
    if not string2:
        dataSetName = DataSetStringRoot + string1
        SetData(dataSetName, SubNet2Write)
        TagDatasets(DataSetStringRoot, [dataSetName])

    else:
        dataSetName = (
            DataSetStringRoot
            + string1
            + string2
            + str(sub_Nets[idx].frequency.start / 1e9)
            + " to "
            + str(sub_Nets[idx].frequency.stop / 1e9)
        )
        SetData(dataSetName, SubNet2Write[idx])
        TagDatasets(DataSetStringRoot, [dataSetName])


# import veusz.embed as veusz  # use for Vuesz visualization
# see https://veusz.github.io/docs/manual/api.html#non-qt-python-programs
# from skrf.plotting import func_on_all_figs as foaf  # act on all figures

# we don't want a full GUI, so keep the root window from appearing
# Tk().withdraw()

# show an "Open" dialog box and return the path to the selected file
filename = askopenfilenames(filetypes=[("Touchstone Files", ".s1p .s2p")])
# print(filename)
# multifiles = Tk.askopenfilename()
for mainLooper in range(len(filename)):
    # Define the current network
    currentNet = rf.Network(filename[mainLooper])

    # break out single S parameters ports
    S11_Net = currentNet.s11

    # subdivide for passband of filter for time domain overlay
    lowerValues = [
        currentNet.s11.frequency.start / 1e9,
        25.5,
        25.5,
        25.5,
        25.5,
        25.5,
        27,
        27,
        26.1,
    ]
    upperValues = [
        currentNet.s11.frequency.stop / 1e9,
        27,
        29,
        30,
        33,
        26,
        30,
        33,
        27.04,
    ]
    prepend2fileName = ""
    append2fileName = ""
    numberOfNets2Use = len(lowerValues)

    # preallocate the list size
    # del sub_Nets
    sub_Nets = [float("nan")] * numberOfNets2Use  # create a list of nan

    for i in range(len(sub_Nets)):
        sub_Nets[i] = currentNet[
            str(lowerValues[i]) + "-" + str(upperValues[i]) + "GHz"
        ]

    print("Now We Start to define data sets in Veusz")

    # create arrays for data to import into Veusz
    # can also use .tolist() instead of resize if needed.
    freqList = np.arange(
        currentNet.frequency.start,
        currentNet.frequency.stop + currentNet.frequency.step,
        currentNet.frequency.step,
    )
    timeList = np.asarray(currentNet.frequency.t_ns) * 1e-9
    S11_dB = np.asarray(currentNet.s11.s_db)
    S11_dB_time = np.asarray(currentNet.s11.s_time_db)
    S12_dB = np.asarray(currentNet.s12.s_db)
    S12_dB_time = np.asarray(currentNet.s12.s_time_db)
    S21_dB = np.asarray(currentNet.s21.s_db)
    S21_dB_time = np.asarray(currentNet.s21.s_time_db)
    S22_dB = np.asarray(currentNet.s22.s_db)
    S22_dB_time = np.asarray(currentNet.s22.s_time_db)

    # for some reason, an array of dims 801,1,1 is not the same as 801....
    S11_dB.resize(currentNet.frequency.npoints)
    S12_dB.resize(currentNet.frequency.npoints)
    S21_dB.resize(currentNet.frequency.npoints)
    S22_dB.resize(currentNet.frequency.npoints)

    S11_dB_time.resize(currentNet.frequency.npoints)
    S12_dB_time.resize(currentNet.frequency.npoints)
    S21_dB_time.resize(currentNet.frequency.npoints)
    S22_dB_time.resize(currentNet.frequency.npoints)
    # freqList.resize(currentNet.frequency.npoints)
    # timeList.resize(currentNet.frequency.npoints)

    # preallocate arrays
    sub_Nets_4Veusz_time = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_freq = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S11_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S12_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S21_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S22_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S11_time_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S12_time_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S21_time_dB = [float("nan")] * numberOfNets2Use
    sub_Nets_4Veusz_S22_time_dB = [float("nan")] * numberOfNets2Use

    for i in range(len(sub_Nets)):
        sub_Nets_4Veusz_time[i] = np.asarray(sub_Nets[i].frequency.t_ns) * 1e-9
        sub_Nets_4Veusz_freq[i] = np.arange(
            sub_Nets[i].frequency.start,
            sub_Nets[i].frequency.stop + sub_Nets[i].frequency.step,
            sub_Nets[i].frequency.step,
        )
        sub_Nets_4Veusz_S11_dB[i] = np.asarray(sub_Nets[i].s11.s_db)
        sub_Nets_4Veusz_S12_dB[i] = np.asarray(sub_Nets[i].s12.s_db)
        sub_Nets_4Veusz_S21_dB[i] = np.asarray(sub_Nets[i].s21.s_db)
        sub_Nets_4Veusz_S22_dB[i] = np.asarray(sub_Nets[i].s22.s_db)
        sub_Nets_4Veusz_S11_time_dB[i] = np.asarray(sub_Nets[i].s11.s_time_db)
        sub_Nets_4Veusz_S12_time_dB[i] = np.asarray(sub_Nets[i].s12.s_time_db)
        sub_Nets_4Veusz_S21_time_dB[i] = np.asarray(sub_Nets[i].s21.s_time_db)
        sub_Nets_4Veusz_S22_time_dB[i] = np.asarray(sub_Nets[i].s22.s_time_db)
        # resize the arrays
        sub_Nets_4Veusz_S11_dB[i].resize(sub_Nets[i].frequency.npoints)
        sub_Nets_4Veusz_S12_dB[i].resize(sub_Nets[i].frequency.npoints)
        sub_Nets_4Veusz_S21_dB[i].resize(sub_Nets[i].frequency.npoints)
        sub_Nets_4Veusz_S22_dB[i].resize(sub_Nets[i].frequency.npoints)

        sub_Nets_4Veusz_S11_time_dB[i].resize(sub_Nets[i].frequency.npoints)
        sub_Nets_4Veusz_S12_time_dB[i].resize(sub_Nets[i].frequency.npoints)
        sub_Nets_4Veusz_S21_time_dB[i].resize(sub_Nets[i].frequency.npoints)
        sub_Nets_4Veusz_S22_time_dB[i].resize(sub_Nets[i].frequency.npoints)

    # start data set import / defintion in Veusz
    # =============================================================================
    # How to add data point labels:
    # SetDataText('NAME_OF_XYlabel', ["({}, {})".format(str(x), str(y)) for x, y in
    # zip(GetData("NAME_X")[0], GetData("NAME_Y")[0])])
    # https://github.com/veusz/veusz/issues/541
    # =============================================================================
    fileParts = os.path.split(filename[mainLooper])
    DataSetStringRoot = prepend2fileName + fileParts[1] + append2fileName
    # magnitude in dB
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S11_dB,
        " S11 dB",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S12_dB,
        " S12 dB",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S21_dB,
        " S21 dB",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S22_dB,
        " S22 dB",
        "",
        0,
    )

    # dB time response
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S11_dB_time,
        " S11 time dB",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S12_dB_time,
        " S12 time dB",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S21_dB_time,
        " S21 time dB",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        S22_dB_time,
        " S22 time dB",
        "",
        0,
    )

    # freq and time scales
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        freqList,
        " Frequency",
        "",
        0,
    )
    SDataVuesz(
        DataSetStringRoot,
        sub_Nets,
        timeList,
        " Time",
        "",
        0,
    )

    # now create the data sets for all the subnets
    for i in range(len(sub_Nets)):
        # magnitude wrt to freq data
        # SetData(
        #     DataSetStringRoot
        #     + " S11 dB: "
        #     + "Freq "
        #     + str(sub_Nets[i].frequency.start / 1e9)
        #     + " to "
        #     + str(sub_Nets[i].frequency.stop / 1e9),
        #     sub_Nets_4Veusz_S11_dB[i],
        # )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S11_dB,
            " S11 dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S12_dB,
            " S12 dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S21_dB,
            " S21 dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S22_dB,
            " S22 dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S21_dB,
            " S21 dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S21_dB,
            " S21 dB: ",
            "Freq ",
            i,
        )

        # magnitude wrt to time domain
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S11_time_dB,
            " S11 Time dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S12_time_dB,
            " S12 Time dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S21_time_dB,
            " S21 Time dB: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_S22_time_dB,
            " S22 Time dB: ",
            "Freq ",
            i,
        )

        # freq and time scales
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_freq,
            " Frequncy: ",
            "Freq ",
            i,
        )
        SDataVuesz(
            DataSetStringRoot,
            sub_Nets,
            sub_Nets_4Veusz_time,
            " Time: ",
            "Freq ",
            i,
        )

    print("Set Definition Completed for file name." + str(filename[mainLooper]))
