import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import threading
from genetic_routing import run_genetic_algorithm
from utils import delete_file

# Initialize global variables
selected_file = None
output_file = None
processing_thread = None
event = threading.Event()


# Define a callback function to update the progress bar
def update_progress(progress_tuple):
    if progress_tuple:
        max_dimension, current_iteration = progress_tuple
        progress_value = (current_iteration / max_dimension) * 100
        progress_var.set(progress_value)
    else:
        progress_var.set(0)


# Function performed on a thread
def task(event, params):
    try:
        global output_file, selected_file
        run_genetic_algorithm(save_file=output_file, read_file=selected_file, event=event, process_callback = None, **params)
        # ------- de adaugat **kwargs pentru run_genetic_algorithm
        processing_label.config(text="Processing complete!", justify='center')
        stop_button.grid_remove()
        progress_bar.grid_remove()
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
    delete_file(output_file)


# Function to open the file dialog
def open_text_file():
    filetypes = (("DSN files", "*.dsn"), ("Text files", "*.txt"), ("All files", "*.*"))
    text.delete('1.0', 'end')
    processing_label.config(text="")
    
    global processing_thread, event
    processing_thread = None
    event.clear()
    try:
        global selected_file, output_file
        f = fd.askopenfile(filetypes=filetypes)
        text.insert('1.0', f.read())
        selected_file = f.name
        print(selected_file)
        f.close()

        dot_index = selected_file.rfind(".")
        if dot_index != -1:
            output_file = selected_file[:dot_index] + ".ses"
        else:
            output_file = selected_file + ".ses"

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
            "keep_values": keep_var.get()
        }

        event = threading.Event()
        processing_thread = threading.Thread(target=task, args=(event, params))
        processing_thread.start()
        stop_button.grid(column=0, row=2, pady=10)
        processing_label.config(text="Processing...", justify='center')
        progress_bar.grid(row=9, column=0, columnspan=4, pady=20)  # Show the progress bar
    
    except Exception as e:
        processing_label.config(text=f"Error: {str(e)}", justify='center')


# Validation for entries
def validate_positive_integer(P):
    if P.isdigit() and int(P) > 0:
        return True
    return False


# Create a GUI app
app = tk.Tk()
app.title('Routing with Genetic Algorithm')
app.geometry('800x700')
app.minsize(width=800, height=700)


# Define widgets
text = tk.Text(app, height=20, width=60)
processing_label = tk.Label(app, text="", font=("Arial", 12))
open_button = ttk.Button(app, text='Choose file', command=lambda: open_text_file() if processing_thread is None else None)
stop_button = ttk.Button(app, text='Stop processing', command=lambda: force_thread_stop() if processing_thread is not None else None)
stop_button.grid_remove()  # Hide stop button initially
keep_var = tk.BooleanVar()
keep_checkbox = tk.Checkbutton(app, text="Keep", variable=keep_var)


# Layout widgets
open_button.grid(column=0, row=0, padx=20, pady=10)
keep_checkbox.grid(column=0, row=1, padx=20, pady=10)
text.grid(column=1, row=0, rowspan=9, padx=20, pady=10)
processing_label.grid(column=1, row=9, pady=10)


# Right column (Checkboxes and entries)
var1 = tk.BooleanVar()
tk.Checkbutton(app, text="All", variable=var1).grid(row=0, column=2, padx=20)
tk.Label(app, text='Width').grid(row=1, column=2, padx=20)
tk.Label(app, text='Clearance').grid(row=2, column=2, padx=20, pady=(0,20))
e1 = tk.Entry(app)
e2 = tk.Entry(app)
e1.grid(row=1, column=3)
e2.grid(row=2, column=3, pady=(0,20))

var2 = tk.BooleanVar()
tk.Checkbutton(app, text="Power", variable=var2).grid(row=3, column=2, padx=20)
tk.Label(app, text='Width').grid(row=4, column=2, padx=20)
tk.Label(app, text='Clearance').grid(row=5, column=2, padx=20, pady=(0,20))
e3 = tk.Entry(app)
e4 = tk.Entry(app)
e3.grid(row=4, column=3)
e4.grid(row=5, column=3, pady=(0,20))

var3 = tk.BooleanVar()
tk.Checkbutton(app, text="Ground", variable=var3).grid(row=6, column=2, padx=20, pady=10)
tk.Label(app, text='Width').grid(row=7, column=2, padx=20)
tk.Label(app, text='Clearance').grid(row=8, column=2, padx=20, pady=(0,20))
e5 = tk.Entry(app)
e6 = tk.Entry(app)
e5.grid(row=7, column=3)
e6.grid(row=8, column=3, pady=(0,20))


# Progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(app, orient="horizontal", length=300, mode="determinate", variable=progress_var)
progress_bar.grid(row=9, column=0, columnspan=4, pady=20)
progress_bar.grid_remove()  # Hide progress bar initially


# Update entries based on checkboxes
def update_entries():
    e1.config(state='normal' if var1.get() else 'disabled')
    e2.config(state='normal' if var1.get() else 'disabled')
    e3.config(state='normal' if var2.get() else 'disabled')
    e4.config(state='normal' if var2.get() else 'disabled')
    e5.config(state='normal' if var3.get() else 'disabled')
    e6.config(state='normal' if var3.get() else 'disabled')

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