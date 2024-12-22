import numpy as np

# Generate random numbers
rand_num = np.random.rand(3)
print("Random numbers:", rand_num)

# Generate random integers
rand_int = np.random.randint(1, 10, 3)
print("Random integers:", rand_int)

# Generate random numbers from a normal distribution
rand_norm = np.random.randn(3)
print("Random numbers from normal distribution:", rand_norm)

# Generate random numbers from a uniform distribution
rand_uniform = np.random.uniform(1, 10, 3)
print("Random numbers from uniform distribution:", rand_uniform)

# Generate random numbers for a matrix
rand_matrix = np.random.rand(2, 3)
print("Random matrix:\n", rand_matrix)

# Generate random numbers for an array of integers between 1 and 10
randint_arr = np.random.randint(0, 10, size=(2, 2))
print("Random integer array:\n", randint_arr)
