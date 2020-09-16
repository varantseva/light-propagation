import numpy as np
import math as m
import matplotlib.pyplot as plt

lvl_1 = 1 / np.exp(2.)
a1 = 100
a0 = a1 / np.sqrt(np.log(10))

x = np.linspace(-500, 500, 100)

E = np.exp(- x ** 2 / a0 ** 2)
plt.plot(x , E)

c = np.zeros(100)
for i in range(100):
    c[i] = 0.10

plt.plot(x , c)
plt.show()
