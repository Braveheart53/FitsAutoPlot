import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import veusz.embed as vz


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)


def create_plot():
    file_path = file_entry.get()
    plot_name = plot_name_entry.get()
    dataset_name = dataset_name_entry.get()

    if not file_path or not plot_name or not dataset_name:
        status_label.config(text="Please fill in all fields")
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

        status_label.config(text="Plot created successfully")
    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")


# Create main window
root = tk.Tk()
root.title("CSV Plotter")

# File selection
file_label = tk.Label(root, text="Select CSV file:")
file_label.pack()
file_entry = tk.Entry(root, width=50)
file_entry.pack()
file_button = tk.Button(root, text="Browse", command=select_file)
file_button.pack()

# Plot name input
plot_name_label = tk.Label(root, text="Enter plot name:")
plot_name_label.pack()
plot_name_entry = tk.Entry(root, width=50)
plot_name_entry.pack()

# Dataset name input
dataset_name_label = tk.Label(root, text="Enter dataset name:")
dataset_name_label.pack()
dataset_name_entry = tk.Entry(root, width=50)
dataset_name_entry.pack()

# Create plot button
create_button = tk.Button(root, text="Create Plot", command=create_plot)
create_button.pack()

# Status label
status_label = tk.Label(root, text="")
status_label.pack()

root.mainloop()
