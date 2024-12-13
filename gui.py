from ddm_functions import *
from ddm_models_2 import *
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# GUI setup
def create_gui():
    def select_file():
        file_path_var.set(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]))

    def select_folder():
        proj_path_var.set(filedialog.askdirectory())

    def run_function():
        file_path = file_path_var.get()
        title = title_var.get()
        proj_path = proj_path_var.get()

        if not file_path or not proj_path or not title:
            messagebox.showerror("Error", "All fields must be filled out.")
        else:
            root.destroy()  # Close the GUI
            run_ddm_on_csv(file_path, title, proj_path)

    root = tk.Tk()
    root.title("DDM Analysis GUI")

    # File selector
    tk.Label(root, text="CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    file_path_var = tk.StringVar()
    tk.Entry(root, textvariable=file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=select_file).grid(row=0, column=2, padx=5, pady=5)

    # Folder selector
    tk.Label(root, text="Project Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    proj_path_var = tk.StringVar()
    tk.Entry(root, textvariable=proj_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=select_folder).grid(row=1, column=2, padx=5, pady=5)

    # Title entry
    tk.Label(root, text="Model Title:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    title_var = tk.StringVar()
    tk.Entry(root, textvariable=title_var, width=50).grid(row=2, column=1, padx=5, pady=5)

    # Run button
    tk.Button(root, text="Run DDM Analysis", command=run_function, bg="green", fg="white").grid(row=3, column=0, columnspan=3, pady=10)

    root.mainloop()

# Run the GUI
create_gui()