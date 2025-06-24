"""Just some quick functions for use in Veusz consol."""
# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul 11 16:32:12 2022
#
# @author: wwallace
# William W. Wallace
# Sandbox for Veusz commands
# =============================================================================

# =============================================================================
# Some notesfor Veusz console
# numpy is loaded as *
# full python interface otherwise
#
# GetData(DataSetName) for getting a tupple of the form (data, symerr, negerr,
#  poserr)
#
# SetData(DataSetName,NumpyArray) for 'setting' a new dataset per
#  SetData(name, val, ymerr=None, negerr=None, poserr=None)
#
# SetDataExpression(name, val, symerr=None, negerr=None, poserr=None,
# linked=False, parametric=None) ==>
#  Create a new dataset based on the
#  expressions given. The expressions are Python syntax expressions based
#  on existing datasets.If linked is True, the dataset will change as the
#  datasets in the expressions change. Parametric can be set to a tuple of
#  (minval, maxval, numitems). t in the expression will iterate from minval
#  to maxval in numitems values.
# =============================================================================

# the following can be used to launch Veusz directly, for now we are copying
# and pasting these  scripts
# =============================================================================
# import veusz.embed as veusz  # use for Vuesz visualization
# =============================================================================

# Take an sparam 1D Array in mag dB and create an array that is linear mag
# =============================================================================
# initialize whatever required packages
import tkinter as tk
from tkinter import simpledialog
import numpy as np

# from tkinter.ttk import tk  # from tkinter import Tk for Python 3.x
# from tkinter.filedialog import askopenfilename


class Demo1:
    """
    Creates a new window with a button, the size of the window.

    for creating new windows. The new window defined by class Demo2
    """

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(
            self.frame, text="New Window", width=25, command=self.new_window
        )
        self.button1.pack()
        self.frame.pack()

    def new_window(self):
        """Def new window."""
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)


class Demo2:
    """
    Window Demo2.

    New windows with a quit button, window the size of the button
    """

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.quitButton = tk.Button(
            self.frame, text="Quit", width=25, command=self.close_windows
        )
        self.quitButton.pack()
        self.frame.pack()

    def close_windows(self):
        """Close  windows function."""
        self.master.destroy()


class MultiSelectBox:
    """
    Creates a tkinter multiselection pane.

    Utilizing all available data  sets
    """

    def __init__(self, master):
        self.master = master
        self.master.title("Data Set Selection")
        # self.master.wm_title("Data Set Pick")

        # for scrolling vertically
        self.yscrollbar = tk.Scrollbar(self.master)
        self.yscrollbar.pack(side="right", fill="y")

        self.label = tk.Label(
            self.master,
            text="Select the datasets to convert "
            + "to average (in linear domain):  ",
            font=("Times New Roman", 10),
            padx=10,
            pady=10,
        )
        self.label.pack()

        self.frame = tk.Frame(self.master)
        # self.frame.
        # self.framedir
        self.list = tk.Listbox(
            self.master, selectmode="extended", yscrollcommand="yscrollbar.set"
        )

        # Widget expands horizontally and
        # vertically by assigning both to
        # fill option
        self.list.pack(padx=10, pady=10, expand="yes", fill="both")
        # we now have an empty list box with a window title
        # of self.master.title
        # now get the list of all dataset names from Veusz
        # dataNames = ["dBone", "dbtwo", "three dB", "four", "five dB"]
        dataNames = GetDatasets()  # gets all data set names from Veusz console

        # now when we populate, we only populate with data expressed in dB
        # and magnitude
        idxNow = 0
        dBNames = [np.nan] * len(dataNames)
        for VzDataSetNameIdx in range(len(dataNames)):
            if "dB" in dataNames[VzDataSetNameIdx]:
                if idxNow == 0:
                    dBNames[idxNow] = dataNames[VzDataSetNameIdx]
                    self.list.insert("end", dataNames[VzDataSetNameIdx])
                    self.list.itemconfig(idxNow, bg="lime")
                    idxNow = idxNow + 1
                else:
                    dBNames[idxNow] = dataNames[VzDataSetNameIdx]
                    self.list.insert("end", dataNames[VzDataSetNameIdx])
                    self.list.itemconfig(idxNow, bg="lime")
                    idxNow = idxNow + 1

        # Attach listbox to vertical scrollbar
        self.yscrollbar.config(command=self.list.yview)

        # Add Button to close window
        self.winCls = tk.Button(
            self.frame,
            text="Close Window",
            width=42,
            command=self.close_windows,
        )
        self.winCls.pack(side="bottom")

        # Add button to process selection
        self.AvgButton = tk.Button(
            self.frame,
            text="Average Selection",
            width=42,
            command=self.AvgdBData,
        )
        self.AvgButton.pack(side="bottom")

        # pack it up Mister!
        # self.list.pack()
        self.frame.pack()

    def Process_dB2lin(self, dataOInterest):
        """Processess All Selected Data in the listbox."""
        # selectedInList = [np.nan] * len(self.list.curselection())
        # newNames = [np.nan] * len(self.list.curselection())
        i = 0
        # for idx in self.list.curselection():
        #     selectedInList[i] = self.list.get(idx)

        #     # now we know what was selected and will step through the
        #     # selected data and create linear data from it
        #     # linked by use of SetDataExpression() in Veusz
        #     print(i)
        #     print(idx)
        #     print(selectedInList)

        #     # make the new dataset name
        #     newNames[i] = selectedInList[i].replace("dB", "linMag")
        #     print(newNames[i])

        #     # make the new expression  based data set
        #     # could not quickly ge this to work
        #     # SetDataExpression(
        #     #     newNames[i],
        #     #     10**(selectedInList[i]/20),
        #     #     linked=True,
        #     # )
        #     SetData(newNames[i], 10 ** (GetData(selectedInList[i])[0] / 20))
        #     i = i + 1

    def Process_lin2dB(self, dataOInterest):
        """Processess All Selected Data in the listbox."""
        # selectedInList = [np.nan] * len(self.list.curselection())
        # newNames = [np.nan] * len(self.list.curselection())
        i = 0
        # for idx in self.list.curselection():
        #     selectedInList[i] = self.list.get(idx)

        #     # now we know what was selected and will step through the
        #     # selected data and create linear data from it
        #     # linked by use of SetDataExpression() in Veusz
        #     print(i)
        #     print(idx)
        #     print(selectedInList)

        #     # make the new dataset name
        #     newNames[i] = selectedInList[i].replace("linMag", "dB_calc")
        #     print(newNames[i])

        #     # make the new expression  based data set
        #     # could not quickly ge this to work
        #     # SetDataExpression(
        #     #     newNames[i],
        #     #     10**(selectedInList[i]/20),
        #     #     linked=True,
        #     # )
        #     20 * log10((GetData(selectedInList[i])[0])))
        #     i = i + 1

    def close_windows(self):
        """Close  windows function."""
        self.master.destroy()

    def AvgdBData(self):
        """Processess All Selected Data in the listbox as a single average."""
        self.selectedInList = [None] * 2
        # self.selectedInList will be two dim with first being list of names,
        # and second the index in list
        self.selectedInList[0] = [np.nan] * len(self.list.curselection())
        self.selectedInList[1] = [np.nan] * len(self.list.curselection())
        self.datalinMag = [np.nan] * len(self.list.curselection())
        self.newNames = [np.nan] * len(self.list.curselection())
        i = 0

        # build the linear data in self.data.linMag and the name
        # in self.selectedInList[0], with indicies at self.selectedInList[1]
        for idx in self.list.curselection():
            self.selectedInList[0][i] = self.list.get(idx)
            self.selectedInList[1][i] = idx
            # now we know what was selected and will step through the
            # selected data and create linear data from it
            # linked by use of SetDataExpression() in Veusz

            #  some output to track in Veusz command window
            print(i)
            print(idx)
            # print(self.selectedInList[0])
            # print(self.selectedInList[1])

            # make the new dataset name
            self.newNames[i] = self.selectedInList[0][i].replace("dB", "linMag")
            # print(self.newNames[i])

            # store all the linear magnitude data in self object at
            # self.datalinMag
            # also storing initial dB  data if you one wants
            # self.datadBMag[i] = GetData(self.selectedInList[0][i])[0]
            self.datalinMag[i] = 10 ** (
                GetData(self.selectedInList[0][i])[0] / 20
            )

            i = i + 1

        print("Out of loop")
        print("Selected List")
        print(self.selectedInList)
        # print(self.selectedInList[0])
        # print(self.selectedInList[1])
        print("Generated Names")
        print(self.newNames)
        # now loop and create an np.matrix with all data to average across a
        # given dimension, as np.average(matrix,axis=0 or 1)
        self.dataMatrix_linMag = np.matrix(self.datalinMag, dtype="float64")
        # print(self.dataMatrix_linMag)
        print("Shape of np.matrix of linear Magnitudes of All Selected Data")
        print(self.dataMatrix_linMag.shape)
        print("***********That was the NP matrix**********")

        self.dataAvg_linMag = np.average(self.dataMatrix_linMag, axis=0)
        self.dataAvg_dBMag = 20 * np.log10(self.dataAvg_linMag)
        print("Shape of Avg_linMag")
        print(self.dataAvg_linMag.shape)
        print("Shape of Avg_dBMag")
        print(self.dataAvg_dBMag.shape)

        # Get the name for the Avg data, will append Avg
        # does not check for existing data sets
        # application_window = tk.Toplevel(self.master)
        userInput = simpledialog.askstring(
            "Input",
            [
                "Enter a string to signitify"
                + " the name of the "
                + "averaged dataset."
                + " The string '_Avg_*' where '*' is linMag or"
                + " dBmag, will be appended."
            ],
            parent=self.master,
        )

        if userInput is not None:
            self.exportName_linMag = userInput + "_Avg_linMag"
            self.exportName_dBMag = userInput + "_Avg_dBMag"
            print(self.exportName_linMag)
            # print(self.dataAvg_linMag.tolist())
            # print(len(np.transpose(self.dataAvg_linMag).tolist()))
            # A = np.squeeze(np.asarray(M))
            SetData(
                self.exportName_linMag,
                np.squeeze(np.asarray(self.dataAvg_linMag)),
            )
            SetData(
                self.exportName_dBMag,
                np.squeeze(np.array(self.dataAvg_dBMag)),
            )
            print(
                "Data Set to be Created after closing  window: "
                + str(self.exportName_linMag)
            )
            print(
                "Data Set to be Created after closing  window: "
                + str(self.exportName_dBMag)
            )
        else:
            print("Nothing was entered for dataset name")

        # used to debug with Veusz command terminal
        # =====================================================================
        #         k = 0
        #         for dataOInterset in self.datalinMag:
        #             print(dataOInterset)
        #             print(
        #                 [
        #                     "*********That was all data set: "
        #                     + str(k)
        #                     + " *************"
        #                 ]
        #             )
        #
        #             k = k + 1
        # =====================================================================

        # will need a pop up dailog  for data set name entry for avg
        print("Out of loop")
        # average all the data sets together
        # sum(linear_Data_sets) / len(selectedInList)
        # maybe a sum function?

    def inputdlg(self, dlkASk, dlgInput):
        """Dialog Window in construction."""
        self.newWindow = tk.Toplevel(self.master)
        self.app = InputDlg(self.newWindow)
        # needed to make the whole item look cleaner and close with the master
        # window set.


class InputDlg:
    """A class to use an input dialog."""


def main():
    """Fuction for tkinter min window display."""
    root = tk.Tk()
    # app = Demo1(root)
    window_width = 300
    window_height = 680

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    myApp = MultiSelectBox(root)
    root.mainloop()


if __name__ == "__main__":
    main()

main()
# import numpy as *

# =============================================================================
# A select the dataset(s)
# =============================================================================

# =============================================================================
#  A.1. start using tkinter multi select
# =============================================================================
# =============================================================================
# masterWindow = tk()
# screen_width = masterWindow.winfo_screenwidth()
# screen_height = masterWindow.winfo_screenheight()
#
# self.window_width = screen_width * 0.01
# self.window_height = screen_height * 0.04
#
# # Window startst in center of screen
# self.window_start_x = screen_width / 2
# self.window_start_y = screen_height / 2
# masterWindow.geometry("")  # tkinter computes window size
# masterWindow.geometry("+%d+%d" % (self.window_start_x, self.window_start_y))
# self.buttonsFrame.pack(side=TOP)
# button_width = 13
# button_height = 2
# masterWindow.mainloop()
# # masterWindow.geometry('100x150')
# =============================================================================


# dataSet = "test"

# =============================================================================
