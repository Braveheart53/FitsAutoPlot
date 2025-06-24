"""Plot 2 Port Touchstone Files."""
# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul 11 16:32:12 2022
# python 3.10 required
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
# Rev C
# added a location for time domain plot limits
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
# import os as os
import skrf as rf  # sci-kit RF module
import cmath

import numpy as np
from matplotlib import pyplot as plt  # use for plotting figures

# Many of these are commented out but meant for future use
from matplotlib.ticker import (
    MultipleLocator,
    FormatStrFormatter,
    AutoMinorLocator,
)

# from pylab import *
# from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename  # for file import dialog

# import veusz.embed as veusz  # use for Vuesz visualization
# see https://veusz.github.io/docs/manual/api.html#non-qt-python-programs
# from skrf.plotting import func_on_all_figs as foaf  # act on all figures

# we don't want a full GUI, so keep the root window from appearing
# Tk().withdraw()

# Plotting Options and User Inputs
TD_ylims = [-40, 0]  # dB in time7
TD_xlims = [0, 1]  # in nano-seconds
VSWR_ylim = [1, 1.2]
spec_FreqLims = [26.5e9, 40e9]  # limits of the spec line in Hz
spec_MagLims = [-35, -35]  # limits of the magnitude of the spec per freq
# show an "Open" dialog box and return the path to the selected file


# function definition space
# currently inwork!
# will replace definiting each by hand and will just call these functions
# to parse data
def parseSparamData(
    NetUsed, FirstPort, SecondPort, typeOData, freqStart, freqStop
):
    """Docstring."""
    match typeOData:
        case "dB":
            print("Use findMyPorts with type O Data case")
        case _:
            return "error with typeOData passed to parsing functioin"


def findMyPorts(NetUsed, FirstPort, SecondPort):
    """Docstring."""
    match [FirstPort, SecondPort]:
        case ([1, 1]):
            print("This is S11")
        case _:
            return "Something went wrong in the s-param parsing function."


filename = askopenfilename()
# print(filename)
# multifiles = Tk.askopenfilename()

# Define the current network
currentNet = rf.Network(filename)

# break out single S parameters ports
S11_Net = currentNet.s11

fig1 = plt.figure(1)
subplt1 = plt.subplot(211)
# plot s11 and plot something with it
S11_Net.plot_s_db(m=0, n=0, label="S11")
currentNet.s22.plot_s_db(m=0, n=0, label="S22")
plt.title("S11 Time Domain from 7/08 S-param Data SN004 10002022")
plt.xlim([S11_Net.frequency.start, S11_Net.frequency.stop])
plt.ylim([-40, 0])
# add spec line
# need to also add an option for step line that creates step line for
# the array, axhiline is just easy for now
plt.axhline(
    y=spec_MagLims[0],
    xmin=(min(spec_FreqLims) - S11_Net.frequency.start)
    / (S11_Net.frequency.stop - S11_Net.frequency.start),
    xmax=(max(spec_FreqLims) - S11_Net.frequency.start)
    / (S11_Net.frequency.stop - S11_Net.frequency.start),
    color="r",
    linewidth=3,
)
plt.grid(
    visible=True,
    which="major",
    axis="both",
    color="#DADADA",
    linestyle="solid",
    linewidth=1,
)
plt.grid(
    visible=True,
    which="minor",
    axis="both",
    color="#DADADA",
    linestyle="dashed",
    linewidth=1,
)
# minor axis ticks and grid enable
fig1.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
# change freq scale to GHz of currect axis
rf.plotting.scale_frequency_ticks(fig1.gca(), "GHz")
plt.xlabel("Frequency (GHz)")
# S11_someOtherNet.plot_s_db(m=0, n=0, label='somethingElse')

S11_sub_Net = currentNet.s11["24.5-29GHz"]

subplt2 = plt.subplot(212)
# plot s11 time domain
S11_Net.plot_s_db_time(m=0, n=0, label="Full Measured Band")
S11_sub_Net.plot_s_db_time(m=0, n=0, label="Passband")
plt.xlim(TD_xlims)
plt.ylim(TD_ylims)
plt.grid(
    visible=True,
    which="major",
    axis="both",
    color="#DADADA",
    linestyle="solid",
    linewidth=1,
)
plt.grid(
    visible=True,
    which="minor",
    axis="both",
    color="#DADADA",
    linestyle="dashed",
    linewidth=1,
)

# subdivide for passband of filter for time domain overlay
lowerValues = [
    currentNet.s11.frequency.start / 1e9,
    24.5,
    25,
    26,
    25.3,
    25.5,
    26.5,
]
upperValues = [
    currentNet.s11.frequency.stop / 1e9,
    29,
    32.5,
    32.5,
    27.35,
    27,
    32.5,
]
numberOfNets2Use = len(lowerValues)

# create a numpy array with all the filtered subsets, must be a list as
# network is not value
# =============================================================================
# del sub_Nets
# sub_Nets = np.empty((numberOfNets2Use, 1))
# sub_Nets[:] = np.nan
# =============================================================================

# preallocate the list size
# del sub_Nets
sub_Nets = [float("nan")] * numberOfNets2Use  # create a list of nan

for i in range(len(sub_Nets)):
    sub_Nets[i] = currentNet.s11[
        str(lowerValues[i]) + "-" + str(upperValues[i]) + "GHz"
    ]

fig2 = plt.figure(2)  # create the figure and loop to plot the time domains
for i in range(len(sub_Nets)):
    label2use = (
        "Pass Band of "
        + str(lowerValues[i])
        + "-"
        + str(upperValues[i])
        + " GHz"
    )
    sub_Nets[i].plot_s_db_time(m=0, n=0, label=label2use)
plt.xlim(TD_xlims)
plt.ylim(TD_ylims)
plt.grid(
    visible=True,
    which="major",
    axis="both",
    color="#DADADA",
    linestyle="solid",
    linewidth=1,
)
plt.grid(
    visible=True,
    which="minor",
    axis="both",
    color="#DADADA",
    linestyle="dashed",
    linewidth=1,
)
# minor axis ticks and grid enable
fig2.gca().xaxis.set_minor_locator(AutoMinorLocator(5))

# Plot group delay and change title
# calc group delay
groupDelay_S11 = abs(currentNet.s11.group_delay)
fig3 = plt.figure(3)
currentNet.plot(groupDelay_S11)
plt.ylabel("Group Delay (units)")
plt.title("Group Delay Play")
plt.grid(
    visible=True,
    which="major",
    axis="both",
    color="#DADADA",
    linestyle="solid",
    linewidth=1,
)
plt.grid(
    visible=True,
    which="minor",
    axis="both",
    color="#DADADA",
    linestyle="dashed",
    linewidth=1,
)
# minor axis ticks and grid enable
fig3.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
# change freq scale to GHz of currect axis
rf.plotting.scale_frequency_ticks(fig3.gca(), "GHz")
plt.xlabel("Frequency (GHz)")

fig4 = rf.plotting.subplot_params(currentNet, add_titles=True, newfig=True)
# rf.plotting.scale_frequency_ticks("GHz")
plt.grid(
    visible=True,
    which="major",
    axis="both",
    color="#DADADA",
    linestyle="solid",
    linewidth=1,
)
plt.grid(
    visible=True,
    which="minor",
    axis="both",
    color="#DADADA",
    linestyle="dashed",
    linewidth=1,
)
for k in range(4):
    rf.plotting.scale_frequency_ticks(fig4[0].get_axes()[k], "GHz")
    fig4[0].get_axes()[k].set_xlabel("Frequency (GHz)")

fig5 = plt.figure(5)
currentNet.s11.plot_s_vswr()
# currentNet.frequency(unit='GHz'), does not work
plt.grid(
    visible=True,
    which="major",
    axis="both",
    color="#DADADA",
    linestyle="solid",
    linewidth=1,
)
plt.grid(
    visible=True,
    which="minor",
    axis="both",
    color="#DADADA",
    linestyle="dashed",
    linewidth=1,
)
plt.xlim([S11_Net.frequency.start, S11_Net.frequency.stop])
plt.ylim(VSWR_ylim)
# minor axis ticks and grid enable
fig5.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
# change freq scale to GHz of currect axis
rf.plotting.scale_frequency_ticks(fig5.gca(), "GHz")
plt.xlabel("Frequency (GHz)")

fig6 = plt.figure(6)
currentNet.plot_it_all()

fig7 = plt.figure(7)
# sub_Nets[-1].plot_s_db_time(m=0, n=0,
#                             label=currentNet.s11[str(lowerValues[-1]) +
#                                                  "-" + str(upperValues[-1]) +
#                                                  "GHz"])
# sub_Nets[-2].plot_s_db_time(m=0, n=0,
#                             label=currentNet.s11[str(lowerValues[-2]) +
#                                                  "-" + str(upperValues[-2]) +
#                                                  "GHz"])
sub_Nets[-1].plot_s_db_time(
    m=0,
    n=0,
    label=(
        "Pass Band of "
        + str(lowerValues[-1])
        + "-"
        + str(upperValues[-1])
        + " GHz"
    ),
)
sub_Nets[-2].plot_s_db_time(
    m=0,
    n=0,
    label=(
        "Pass Band of "
        + str(lowerValues[-2])
        + "-"
        + str(upperValues[-2])
        + " GHz"
    ),
)
plt.xlim([0, 1])
plt.ylim([-70, -30])

print("WTF")
