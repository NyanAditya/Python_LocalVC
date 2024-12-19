import matplotlib.pyplot as plt

# Simple Line Plot
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y, label="Line")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')
plt.legend()
plt.show()
