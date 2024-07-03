#! C:\Users\manue\Desktop\GA\Proiect\venv\Scripts\python.exe
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import threading
import os
from genetic_routing import run_genetic_algorithm
from tktooltip import ToolTip

# Initialize global variables
filename = None
processing_thread = None
event = threading.Event()


# Function to update progress bar
def update_progress():
    global processing_thread

    if os.path.exists('done.txt'):
        processing_label.config(text="Procesare finalizată!", justify='center')

    if (processing_thread and processing_thread.is_alive() or event or not event.is_set()) and os.path.isfile("progress.txt"):
        progress_bar.grid()
        try:
            with open("progress.txt", "r") as f:
                data = f.read().strip().split(',')
                if len(data) == 2:
                    progress = int(data[1])
                    max_rounds = int(data[0])
                    progress_bar['value'] = progress
                    processing_label.config(text=f"Progres {int(progress*max_rounds/100)} / {max_rounds}", justify='center')
                    if progress >= 100:
                        processing_label.config(text="Procesare finalizată!", justify='center')
                else:
                    progress_bar['value'] = 0
                    processing_label.config(text="", justify='center')

        except Exception:
            pass
    else:
        try:
            # Hide progress bar if no active process
            progress_bar.grid_remove()

        except Exception:
            pass
    app.after(1000, update_progress)


# Function performed on a thread
def task(event, params):
    try:
        global filename
        run_genetic_algorithm(filename=filename, event=event, **params)
    finally:
        event.clear()
        global processing_thread, stop_button
        stop_button.grid_remove()
        processing_thread = None
        app.after(0, process_complete_cleanup)


def process_complete_cleanup():
    global processing_thread, event, open_button, stop_button
    if os.path.exists('progress.txt'):
        os.remove('progress.txt')
    stop_button.grid_remove()
    open_button = ttk.Button(app, text='Selectează fișier', command=lambda: choose_file() if processing_thread is None else None)
    open_button.grid(column=0, row=0, padx=20, pady=10, rowspan=2)
    processing_thread = None
    event = threading.Event()


# Forcefully stops the processing thread and cleans up resources
def force_thread_stop():
    global processing_thread, event, stop_button
    event.set()
    stop_button.grid_remove()
    if processing_thread is not None:
        processing_label.config(text="Oprire căutare")
        # Wait for the processing thread to finish
        while processing_thread.is_alive():
            processing_thread.join(timeout=0.1)  # Check every 0.1 seconds

        processing_thread = None


# Function to open the file dialog
def choose_file():
    filetypes = (("Fișiere KiCad PCB", "*.kicad_pcb"), ("Toate fișierele", "*.*"))
    text.delete('1.0', 'end')
    processing_label.config(text="")

    global processing_thread, event
    processing_thread = None
    event.clear()
    try:
        global filename
        f = fd.askopenfile(filetypes=filetypes)
        filename = f.name
        text.insert('1.0', f"filename:\t{filename}\n\ncontent:\n{f.read()}")
        f.close()

        params = {
            "all": var1.get(),
            "power": var2.get(),
            "ground": var3.get(),
            "width_all": validate_value(e1.get(), var1.get()),
            "clearance_all": validate_value(e2.get(), var1.get()),
            "width_power": validate_value(e3.get(), var2.get()),
            "clearance_power": validate_value(e4.get(), var2.get()),
            "width_ground": validate_value(e5.get(), var3.get()),
            "clearance_ground": validate_value(e6.get(), var3.get()),
            "keep_values": keep_var.get(),
            "layer": 1 if layer_var.get() == "BOTTOM" else 2
        }

        global open_button, stop_button
        open_button.grid_remove()
        stop_button = ttk.Button(app, text='Oprire', command=lambda: force_thread_stop() if processing_thread is not None else None)
        stop_button.grid(column=0, row=7, pady=10)
        processing_label.config(text="Preprocesare...", justify='center')
        
        # Delete progress file
        if os.path.exists("progress.txt"):
            os.remove("progress.txt")

        if os.path.exists("done.txt"):
            os.remove("done.txt")
        

        event = threading.Event()
        processing_thread = threading.Thread(target=task, args=(event, params))
        processing_thread.start()

    except Exception as e:
        processing_label.config(text=f"Error: {str(e)}", justify='center')


def validate_value(value, validator):
    if validator:
        return int(value) if value.isdigit() and int(value) >= 10000 else None
    return None


# Validation for entries
def validate_positive_integer(P):
    return P == "" or (P.isdigit() and int(P) > 0)


# Create a GUI app
app = tk.Tk()
app.title('Rutare cu Algoritmi Genetici')
app.geometry('1050x475')
app.minsize(width=1050, height=475)


# Define widgets
text = tk.Text(app, height=20, width=60)
processing_label = tk.Label(app, text="", font=("Arial", 12))
open_button = ttk.Button(app, text='Selectează fișier', command=lambda: choose_file() if processing_thread is None else None)
stop_button = ttk.Button(app, text='Oprire', command=lambda: force_thread_stop() if processing_thread is not None else None)
stop_button.grid_remove()  # Hide stop button initially
keep_var = tk.BooleanVar()
keep_ck = tk.Checkbutton(app, text="Păstrarea rutelor existente", variable=keep_var)
layer_var = tk.StringVar(value="BOTTOM")
layer_dropdown = ttk.OptionMenu(app, layer_var, "BOTTOM", "BOTTOM", "TOP/BOTTOM")
ToolTip(layer_dropdown, msg="Selectarea modului de rutare")

# Layout widgets
open_button.grid(column=0, row=0, padx=20, pady=10, rowspan=2)
keep_ck.grid(column=0, row=2, padx=20, pady=10)
layer_dropdown.grid(column=0, row=4, padx=20, pady=10)
text.grid(column=1, row=0, rowspan=9, padx=20, pady=10)
processing_label.grid(column=1, row=9, pady=10)


# Progress bar widget
progress_bar = ttk.Progressbar(app, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(column=1, row=10, pady=20)
progress_bar.grid_remove()


# Right column (Checkboxes and entries)
var1 = tk.BooleanVar()
all_ck = tk.Checkbutton(app, text="Toate traseele", variable=var1, command=lambda: update_entries()).grid(row=0, column=2, sticky='w')
tk.Label(app, text='Lățime').grid(row=1, column=2, padx=30, sticky='w')
tk.Label(app, text='Distanța min.').grid(row=2, column=2, padx=30, pady=(0, 20), sticky='w')
e1 = tk.Entry(app)
e2 = tk.Entry(app)
e1.grid(row=1, column=3)
e2.grid(row=2, column=3, pady=(0, 20))

var2 = tk.BooleanVar()
power_ck = tk.Checkbutton(app, text="Trasee de alimentare", variable=var2, command=lambda: update_entries()).grid(row=3, column=2, sticky='w')
tk.Label(app, text='Lățime').grid(row=4, column=2, padx=30, sticky='w')
tk.Label(app, text='Distanța min.').grid(row=5, column=2, padx=30, pady=(0, 20), sticky='w')
e3 = tk.Entry(app)
e4 = tk.Entry(app)
e3.grid(row=4, column=3)
e4.grid(row=5, column=3, pady=(0, 20))

var3 = tk.BooleanVar()
ground_ck = tk.Checkbutton(app, text="Trasee de masă", variable=var3, command=lambda: update_entries()).grid(row=6, column=2, sticky="w")
tk.Label(app, text='Lățime').grid(row=7, column=2, padx=30, sticky='w')
tk.Label(app, text='Distanța min.').grid(row=8, column=2, padx=30, pady=(0, 20), sticky='w')
e5 = tk.Entry(app)
e6 = tk.Entry(app)
e5.grid(row=7, column=3)
e6.grid(row=8, column=3, pady=(0, 20))

tk.Label(app, text='Dimensiunile sunt exprimate în')

# Update entries based on checkboxes
def update_entries():
    if processing_thread is None:
        e1.config(state='normal' if var1.get() else 'disabled')
        e2.config(state='normal' if var1.get() else 'disabled')
        e3.config(state='normal' if var2.get() else 'disabled')
        e4.config(state='normal' if var2.get() else 'disabled')
        e5.config(state='normal' if var3.get() else 'disabled')
        e6.config(state='normal' if var3.get() else 'disabled')
        ToolTip(e1, msg="Valorile exprimate în nm")
        ToolTip(e2, msg="Valorile exprimate în nm")
        ToolTip(e3, msg="Valorile exprimate în nm")
        ToolTip(e4, msg="Valorile exprimate în nm")
        ToolTip(e5, msg="Valorile exprimate în nm")
        ToolTip(e6, msg="Valorile exprimate în nm")
    else:
        # If processing is in progress, disable all entries and checkboxes
        e1.config(state='disabled')
        e2.config(state='disabled')
        e3.config(state='disabled')
        e4.config(state='disabled')
        e5.config(state='disabled')
        e6.config(state='disabled')
        keep_ck.config(state='disabled')
        all_ck.config(state='disabled')
        power_ck.config(state='disabled')
        ground_ck.config(state='disabled')


var1.trace_add("write", lambda *args: update_entries())
var2.trace_add("write", lambda *args: update_entries())
var3.trace_add("write", lambda *args: update_entries())

validate_command = (app.register(validate_positive_integer), '%P')

e1.config(validate="key", validatecommand=validate_command)
e2.config(validate="key", validatecommand=validate_command)
e3.config(validate="key", validatecommand=validate_command)
e4.config(validate="key", validatecommand=validate_command)
e5.config(validate="key", validatecommand=validate_command)
e6.config(validate="key", validatecommand=validate_command)


# Default values dict
default_values = {"e1": "500000", "e2": "200000", "e3": "1000000", "e4": "500000", "e5": "500000", "e6": "200000"}

def set_default_input_values():
    for entry_name, default_value in default_values.items():
        entry_widget = globals()[entry_name]  # Obțineți widget-ul câmpului de intrare după numele său
        entry_widget.insert(0, default_value)  # Inserați valoarea implicită în câmpul de intrare

# Default values
set_default_input_values()

if __name__ == "__main__":
    # Delete progress file
    if os.path.exists("progress.txt"):
        os.remove("progress.txt")

    if os.path.exists("done.txt"):
        os.remove("done.txt")

    app.after(1000, update_progress)  # Apelează funcția inițial
    app.mainloop()