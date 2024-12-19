import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Access data
print("DataFrame:\n", df)
print("Average Age:", df['Age'].mean())

# Filter rows
print("People older than 28:\n", df[df['Age'] > 28])
