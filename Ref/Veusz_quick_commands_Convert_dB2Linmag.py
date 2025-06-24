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
            text="Select the datasets to convert to linear:  ",
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
        self.ProcessButton = tk.Button(
            self.frame,
            text="Process Selection",
            width=42,
            command=self.Process_dB2lin,
        )
        self.ProcessButton.pack(side="bottom")

        # pack it up Mister!
        # self.list.pack()
        self.frame.pack()

    def Process_dB2lin(self):
        """Processess All Selected Data in the listbox."""
        selectedInList = [np.nan] * len(self.list.curselection())
        newNames = [np.nan] * len(self.list.curselection())
        i = 0
        for idx in self.list.curselection():
            selectedInList[i] = self.list.get(idx)

            # now we know what was selected and will step through the
            # selected data and create linear data from it
            # linked by use of SetDataExpression() in Veusz
            print(i)
            print(idx)
            print(selectedInList)

            # make the new dataset name
            newNames[i] = selectedInList[i].replace("dB", "linMag")
            print(newNames[i])

            # make the new expression  based data set
            # could not quickly ge this to work
            # SetDataExpression(
            #     newNames[i],
            #     10**(selectedInList[i]/20),
            #     linked=True,
            # )
            SetData(newNames[i], 10 ** (GetData(selectedInList[i])[0] / 20))
            i = i + 1

    def close_windows(self):
        """Close  windows function."""
        self.master.destroy()


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
