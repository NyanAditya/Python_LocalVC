import matplotlib.pyplot as plt

labels = ['A', 'B', 'C', 'D']
sizes = [40, 30, 20, 10]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Pie Chart Example")
plt.show()
