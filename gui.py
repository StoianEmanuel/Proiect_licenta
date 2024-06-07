#! C:\Users\manue\Desktop\GA\Proiect\venv\Scripts\python.exe
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import threading
from genetic_routing import run_genetic_algorithm

# Initialize global variables
filename = None
output_file = None
processing_thread = None
process_callback = None
event = threading.Event()


# Function performed on a thread
def task(event, params):
    try:
        global filename, process_callback
        run_genetic_algorithm(filename=filename, event=event, **params)

        processing_label.config(text="Processing complete!", justify='center')
        stop_button.grid_remove()
    finally:
        event.clear()


# Forcefully stops the processing thread and cleans up resources
def force_thread_stop():
    global processing_thread, event
    event.set()
    stop_button.grid_remove()
    if processing_thread is not None:
        processing_label.config(text="Stopping thread")
        stop_thread = threading.Thread(target=wait_for_thread_completion)
        stop_thread.start()


# Waits for the processing thread to complete and updates the interface
def wait_for_thread_completion():
    global processing_thread
    if processing_thread is not None:
        processing_thread.join()  # Wait for the thread to finish
        processing_thread = None
        app.after(0, update_interface)  # Update the interface after the thread ends


# Updates the interface after the processing thread has stopped
def update_interface(text="Thread stopped"):
    processing_label.config(text=text)


# Function to open the file dialog
def choose_file():
    filetypes = (("KiCad PCB files", "*.kicad_pcb"), ("All files", "*.*"))
    text.delete('1.0', 'end')
    processing_label.config(text="")

    global processing_thread, event
    processing_thread = None
    event.clear()
    try:
        global filename, output_file, process_callback
        f = fd.askopenfile(filetypes=filetypes)
        text.insert('1.0', f.read())
        filename = f.name
        f.close()

        dot_index = filename.rfind(".")
        if dot_index != -1:
            output_file = filename[:dot_index] + ".ses"
        else:
            output_file = filename + ".ses"

        params = {
            "all": var1.get(),
            "power": var2.get(),
            "ground": var3.get(),
            "width_all": int(e1.get()) if e1.get().isdigit() else None,
            "clearance_all": int(e2.get()) if e2.get().isdigit() else None,
            "width_power": int(e3.get()) if e3.get().isdigit() else None,
            "clearance_power": int(e4.get()) if e4.get().isdigit() else None,
            "width_ground": int(e5.get()) if e5.get().isdigit() else None,
            "clearance_ground": int(e6.get()) if e6.get().isdigit() else None,
            "keep_values": keep_var.get(),
            "layer": 1 if layer_var.get() == "BOTTOM" else 2
        }

        event = threading.Event()
        processing_thread = threading.Thread(target=task, args=(event, params))
        processing_thread.start()
        stop_button.grid(column=0, row=2, pady=10)
        processing_label.config(text="Processing...", justify='center')

    except Exception as e:
        processing_label.config(text=f"Error: {str(e)}", justify='center')


# Validation for entries
def validate_positive_integer(P):
    return P == "" or (P.isdigit() and int(P) > 0)


# Create a GUI app
app = tk.Tk()
app.title('Routing with Genetic Algorithm')
app.geometry('900x450')
app.minsize(width=900, height=450)


# Define widgets
text = tk.Text(app, height=20, width=60)
processing_label = tk.Label(app, text="", font=("Arial", 12))
open_button = ttk.Button(app, text='Choose file', command=lambda: choose_file() if processing_thread is None else None)
stop_button = ttk.Button(app, text='Stop processing', command=lambda: force_thread_stop() if processing_thread is not None else None)
stop_button.grid_remove()  # Hide stop button initially
keep_var = tk.BooleanVar()
keep_ck = tk.Checkbutton(app, text="Keep", variable=keep_var)
layer_var = tk.StringVar(value="BOTTOM")
layer_dropdown = ttk.OptionMenu(app, layer_var, "BOTTOM", "BOTTOM", "TOP/BOTTOM")

# Layout widgets
open_button.grid(column=0, row=0, padx=20, pady=10)
keep_ck.grid(column=0, row=1, padx=20, pady=10)
layer_dropdown.grid(column=0, row=3, padx=20, pady=10)
text.grid(column=1, row=0, rowspan=9, padx=20, pady=10)
processing_label.grid(column=1, row=9, pady=10)


# Right column (Checkboxes and entries)
var1 = tk.BooleanVar()
all_ck = tk.Checkbutton(app, text="All", variable=var1, command=lambda: update_entries()).grid(row=0, column=2, padx=20)
tk.Label(app, text='Width').grid(row=1, column=2, padx=20)
tk.Label(app, text='Clearance').grid(row=2, column=2, padx=20, pady=(0, 20))
e1 = tk.Entry(app)
e2 = tk.Entry(app)
e1.grid(row=1, column=3)
e2.grid(row=2, column=3, pady=(0, 20))

var2 = tk.BooleanVar()
power_ck = tk.Checkbutton(app, text="Power", variable=var2, command=lambda: update_entries()).grid(row=3, column=2, padx=20)
tk.Label(app, text='Width').grid(row=4, column=2, padx=20)
tk.Label(app, text='Clearance').grid(row=5, column=2, padx=20, pady=(0, 20))
e3 = tk.Entry(app)
e4 = tk.Entry(app)
e3.grid(row=4, column=3)
e4.grid(row=5, column=3, pady=(0, 20))

var3 = tk.BooleanVar()
ground_ck = tk.Checkbutton(app, text="Ground", variable=var3, command=lambda: update_entries()).grid(row=6, column=2, padx=20, pady=10)
tk.Label(app, text='Width').grid(row=7, column=2, padx=20)
tk.Label(app, text='Clearance').grid(row=8, column=2, padx=20, pady=(0, 20))
e5 = tk.Entry(app)
e6 = tk.Entry(app)
e5.grid(row=7, column=3)
e6.grid(row=8, column=3, pady=(0, 20))


# Update entries based on checkboxes
def update_entries():
    if processing_thread is None:
        e1.config(state='normal' if var1.get() else 'disabled')
        e2.config(state='normal' if var1.get() else 'disabled')
        e3.config(state='normal' if var2.get() else 'disabled')
        e4.config(state='normal' if var2.get() else 'disabled')
        e5.config(state='normal' if var3.get() else 'disabled')
        e6.config(state='normal' if var3.get() else 'disabled')
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


app.mainloop()
