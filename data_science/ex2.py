import numpy as np
import random

arr1 = np.array([i for i in range(50)])
arr2 = np.array([random.randint(0, i) for i in range(50)])
arr4 = arr1 + arr2
arr5 = arr4.reshape(5, 10)
print(arr5)
print(arr5.reshape(-1, 2))
print(arr5.dtype)


