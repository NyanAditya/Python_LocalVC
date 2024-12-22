import matplotlib.pyplot as plt

x = [5, 7, 8, 7, 2]
y = [99, 86, 87, 88, 100]

plt.scatter(x, y, color='red', label='Data Points')
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
