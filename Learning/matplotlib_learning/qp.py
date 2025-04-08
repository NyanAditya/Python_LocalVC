import matplotlib.pyplot as plt
import numpy as np

# Data
Food = ["Meat", "Banana", "Avocados", "Sweet Potatoes", "Spinach", "Watermelon", 
        "Coconut water", "Beans", "Legumes", "Tomato"]
Calories = [250, 130, 140, 120, 20, 20, 10, 50, 40, 19]
Potassium = [40, 55, 20, 30, 40, 32, 10, 26, 25, 20]
Fat = [8, 5, 3, 6, 1, 1.5, 0, 2, 1.5, 2.5]

# Set position of bars on X axis
x = np.arange(len(Food))
width = 0.3  # Width of bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, Calories, width, label="Calories", color="red")
ax.bar(x, Potassium, width, label="Potassium", color="blue")
ax.bar(x + width, Fat, width, label="Fat", color="green")

# Add labels
ax.set_xlabel("Food Items")
ax.set_ylabel("Nutritional Value")
ax.set_title("Nutritional Comparison of Foods")
ax.set_xticks(x)
ax.set_xticklabels(Food, rotation=45, ha="right")
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
