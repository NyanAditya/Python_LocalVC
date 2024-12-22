import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6]

plt.hist(data, bins=5, color='green', edgecolor='black')
plt.title("Histogram Example")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.show()
