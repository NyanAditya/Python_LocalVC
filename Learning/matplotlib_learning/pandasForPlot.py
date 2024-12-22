import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame
data = {'X': [1, 2, 3, 4], 'Y': [2, 4, 6, 8]}
df = pd.DataFrame(data)

# Plot the DataFrame
df.plot(x='X', y='Y', kind='line')  # Other kinds: 'bar', 'scatter', etc.
plt.title("Plot from Pandas")
plt.show()
