# import numpy as np
#
# out = np.linspace(0, 1, 50)
# print(out)

import numpy as np

# Define two simple 1D arrays (vectors)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Vector a: {a}")
print(f"Vector b: {b}\n")

# --- SCENARIO 1: WITH np.outer ---
# This creates a matrix where every element of 'a' is multiplied by every element of 'b'
outer_result = np.outer(a, b)

print("--- 1. With np.outer(a, b) ---")
print("Result is a Matrix (Shape 3x3):")
print(outer_result)
print("\nExplanation: \nRow 1 is 1 * [4, 5, 6]")
print("Row 2 is 2 * [4, 5, 6]")
print("Row 3 is 3 * [4, 5, 6]\n")


# --- SCENARIO 2: WITHOUT np.outer (Standard Multiplication) ---
# This does element-wise multiplication: a[0]*b[0], a[1]*b[1], etc.
standard_result = a * b

print("--- 2. Without np.outer (Just 'a * b') ---")
print("Result is a Vector (Shape 3,):")
print(standard_result)
print("Explanation: [1*4, 2*5, 3*6]\n")


# --- SCENARIO 3: HOW TO REPLICATE OUTER WITHOUT THE FUNCTION ---
# If you didn't have np.outer, you would have to reshape 'a' into a column
# to force NumPy to 'broadcast' it against 'b'.
manual_outer = a[:, np.newaxis] * b

print("--- 3. Replicating Outer Manually (Broadcasting) ---")
print("Result is the same Matrix:")
print(manual_outer)