import numpy as np

list1 = ([[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
          [6, 7, 8, 9, 10],
          [6, 7, 8, 9, 10]])

np_list1 = np.array(list1)
print(type(list1))
arr3 = np.array(list1)
print(type(arr3))
print(arr3.astype(float))
print("\n")
print(np_list1[1:, 4:])