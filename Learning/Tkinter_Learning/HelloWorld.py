import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Hello World App")

# Add a label
label = tk.Label(root, text="Hello, World!")
label.pack()  # Add the label to the window

def on_click():
    print("Button clicked!")

button = tk.Button(root, text="Click Me", command=on_click)
button.pack()

entry = tk.Entry(root, width=20)
entry.pack()

text = tk.Text(root, height=5, width=30)
text.pack()

frame = tk.Frame(root, bg="lightblue", padx=10, pady=10)
frame.pack()

label_in_frame = tk.Label(frame, text="Inside a Frame")
label_in_frame.pack()

var = tk.IntVar()  # Variable to store checkbox state
checkbox = tk.Checkbutton(root, text="Check Me", variable=var)
checkbox.pack()

# Start the main event loop
root.mainloop()