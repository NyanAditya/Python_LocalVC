import tkinter as tk

def greet():
    name = entry.get()
    label.config(text=f"Hello, {name}!")

# Create window
root = tk.Tk()
root.title("Greeting App")

# Widgets
label = tk.Label(root, text="Enter your name:")
label.pack()

entry = tk.Entry(root)
entry.pack()

button = tk.Button(root, text="Greet", command=greet)
button.pack()

root.mainloop()
