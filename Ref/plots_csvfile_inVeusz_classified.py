import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import veusz.embed as vz


class csv_autoPlot():

    def __init(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("CSV Plotter")

        # File selection
        self.file_label = tk.Label(self.root, text="Select CSV file:")
        self.file_label.pack()
        self.file_entry = tk.Entry(self.root, width=50)
        self.file_entry.pack()
        self.file_button = tk.Button(
            self.root, text="Browse", command=self.select_file
        )
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
        self.create_button = tk.Button(
            self.root, text="Create Plot", command=self.create_plot)
        self.create_button.pack()

        # Status label
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack()

        self.root.mainloop()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def create_plot(self):
        file_path = self.file_entry.get()
        plot_name = self.plot_name_entry.get()
        dataset_name = self.dataset_name_entry.get()

        if not file_path or not plot_name or not dataset_name:
            self.status_label.config(text="Please fill in all fields")
            return

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Create Veusz embedded window
            embed = vz.Embedded('Veusz')
            embed.EnableToolbar()

            # Create a new document
            # doc = embed.Root.AddDocument()

            # Create a new page
            page = embed.Root.Add('page')

            # Create a new graph
            graph = page.Add('graph', name='graph1', autoadd=False)

            # Set plot title
            # incorrect graph.title.val = plot_name

            # Assume the first column is x and the second column is y
            x_data = df.iloc[:, 0].tolist()
            y_data = df.iloc[:, 1].tolist()

            # Create datasets
            embed.SetData(dataset_name + '_x', x_data)
            embed.SetData(dataset_name + '_y', y_data)

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
