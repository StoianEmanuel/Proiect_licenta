import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import threading
from genetic_routing import run_genetic_algorithm
from utils import delete_file


selected_file = None    # = read_file           (.txt | .dsn | .csv)
output_file = None      # = solution_file       (.ses)
processing_thread = None    # thread for genetic_routing
event = threading.Event()   # used to stop threads


# Function performed on a thread
def task(event):
    try:
        global output_file, selected_file
        run_genetic_algorithm(save_file = output_file, read_file = selected_file, event = event)
        processing_label.config(text="Processing complete!", justify='center')
        stop_button.grid_forget()
    finally:
        event.clear()


# Forcefully stops the processing thread and cleans up resources
def force_thread_stop():
    global processing_thread, event
    event.set()
    stop_button.grid_forget()
    if processing_thread is not None:
        processing_label.config(text="Stopping thread")
        stop_thread = threading.Thread(target=wait_for_thread_completion)
        stop_thread.start()


# Waits for the processing thread to complete and updates the interface.
def wait_for_thread_completion():
    global processing_thread
    if processing_thread is not None:
        processing_thread.join()  # Așteaptă terminarea firului de execuție
        processing_thread = None
        app.after(0, update_interface)  # Actualizează interfața după ce firul de execuție s-a încheiat


# Updates the interface after the processing thread has stopped.
def update_interface(text = "Thread stopped"):
    processing_label.config(text = text)
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
        f = fd.askopenfile(filetypes = filetypes)
        text.insert('1.0', f.read())
        selected_file = f.name
        print(selected_file)
        f.close()

        dot_index = selected_file.rfind(".") # find last "." in selected (read) file path to replace output's file extension 
        if dot_index != -1:
            output_file = selected_file[:dot_index] + ".ses"
        else:
            output_file = selected_file + ".ses"

        event = threading.Event()
        processing_thread = threading.Thread(target = task, args=(event,))
        processing_thread.start()
        stop_button.grid(column=0, row=2, pady=10)
        processing_label.config(text="Processing...", justify='center')
    
    except Exception as e:
        processing_label.config(text=f"Error: {str(e)}", justify='center')


# Create a GUI app 
app = tk.Tk() 
app.title('Routing with Genetic Algorithm') 
app.geometry('600x350') 
app.minsize(width=600, height=350)
app.columnconfigure(0, weight=1)

text = tk.Text(app, height=12) 
processing_label = tk.Label(app, text="", font=("Arial", 12))
open_button = ttk.Button(app, text = 'Open file', command = lambda: open_text_file() if processing_thread is None else None)
stop_button = ttk.Button(app, text = 'Stop processing', command = lambda: force_thread_stop() if processing_thread is not None else None)

text.grid(column = 0, row = 0, pady = (20, 10))
open_button.grid(column = 0, row = 1, pady = 10)
processing_label.grid(column = 0, row = 2, pady = 10)


app.mainloop()