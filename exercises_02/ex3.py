import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification


ds1 = make_moons(n_samples=100, shuffle=True, noise=0.3, random_state=15)
ds2 = make_circles(n_samples=100, shuffle=True, noise=0.3, random_state=15)
ds3 = []
x, y = ds1
# print(x.size)
# print(x.shape)
# print(y.size)
# print(y.shape)
x1 = x.reshape(2, 100)
print(x1.size)
print(x1.shape)
ds4 = np.add(x1, y)
print(ds4)
print(ds4.size)
print(ds4.shape)

