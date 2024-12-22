import numpy as np

arr = np.arange(1, 7)  # Array: [1, 2, 3, 4, 5, 6]
reshaped = arr.reshape((2, 3))  # Reshape to 2 rows, 3 columns
transposed = reshaped.T  # Transpose the array

print("Original Array:\n", arr)
print("Reshaped Array:\n", reshaped)
print("Transposed Array:\n", transposed)