import tkinter as tk 
from tkinter import ttk 
from tkinter import filedialog as fd 
import time
import threading


# File used for processing
selected_file = None


# Function to simulate processing
def processing():
    # Simulate some processing time (3 seconds)
    time.sleep(3)
    # Update label with processing complete message
    processing_label.config(text="Processing complete!", justify='center')
    stop_button.grid_forget()


# Function to stop the processing
def stop_processing():
    global processing_thread
    if processing_thread and processing_thread.is_alive():
        # Oprim thread-ul de procesare dacă este în desfășurare
        processing_thread.join()
        processing_label.config(text="Processing stopped by user!", justify='center')
        stop_button.grid_forget()


# Function to open the file dialog 
def open_text_file():
    # Specify the file types 
    filetypes = (("DSN files", "*.dsn"), ("Text files", "*.txt"), ("All files", "*.*"))
    
    # Clear textfield area
    text.delete('1.0', 'end')
    processing_label.config(text="")

    global processing_thread
    processing_thread = None

    # Show the open file dialog by specifying path 
    try:
        global selected_file
        f = fd.askopenfile(filetypes=filetypes) 

        # Insert the text extracted from file in a textfield 
        text.insert('1.0', f.read())

        selected_file = f.name
        print(selected_file)

        # Close the file
        f.close()

        # Înființăm un thread separat pentru a apela funcția care simulează procesarea
        processing_thread = threading.Thread(target=processing)
        processing_thread.start()

        # Înființăm butonul pentru oprirea procesării și îl afișăm
        stop_button.grid(column=0, row=2, pady=10)

        # Display processing message
        processing_label.config(text="Processing...", justify='center')

        # Call function to simulate processing after a delay
        #app.after(1000, processing)
    
    except Exception as e:
        processing_label.config(text=f"Error: {str(e)}", justify='center')


# Create a GUI app 
app = tk.Tk() 

# Specify the title to app 
app.title('Routing with Genetic Algorithm') 
app.geometry('600x350') 
# Set the minimum size of the window
app.minsize(width=600, height=350)

# Configure the resizing behavior
app.columnconfigure(0, weight=1)


# Create a textfield for putting the text extracted from file 
text = tk.Text(app, height=12) 

# Create a label for displaying processing message
processing_label = tk.Label(app, text="", font=("Arial", 12))

# Create an open file button 
open_button = ttk.Button(app, text='Open file', command=open_text_file)

# Creăm butonul pentru oprirea procesării
stop_button = ttk.Button(app, text='Stop processing', command=stop_processing)

# Specify the location of elements
text.grid(column=0, row=0, pady=(20, 10))
open_button.grid(column=0, row=1, pady=10)
processing_label.grid(column=0, row=2, pady=10)

# Make infinite loop for displaying app on the screen 
app.mainloop()
