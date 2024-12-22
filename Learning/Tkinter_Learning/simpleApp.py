import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Counter App")

# Initialize counter variable
counter = 1

# Function to update the counter
def update_counter(value):
    global counter
    counter *= value
    label.config(text=f"Counter: {counter}")

# Add label to display counter
label = tk.Label(root, text="Counter: 0", font=("Arial", 24))
label.pack()

# Add buttons to increase and decrease the counter
btn_increase = tk.Button(root, text="Increase", command=lambda: update_counter(100000000000))
btn_decrease = tk.Button(root, text="Decrease", command=lambda: update_counter(-100000000000))

btn_increase.pack(side="left", padx=10)
btn_decrease.pack(side="right", padx=10)

# Start the main event loop
root.mainloop()
