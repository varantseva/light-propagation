import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as sc


C0 = sc.c
n0 = 1.          # [-] refractive index
n2 = 1e-23       # [m**2/W]
lam0 = 744e-9    # [m] wavelength
a0 = 1e-3        # [m] radius
w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0   # wave number


Pmax = 3.72 * lam0 ** 2 / (8 * np.pi * n2)


Pin = np.linspace(Pmax * 1.1, 10 * Pmax, 100)
Lth = np.zeros(len(Pin))
Pth = np.zeros(len(Pin))

Ld = k0 * a0 ** 2

for i in range(100):
    Pth[i] = Pin[i] / Pmax
    Lth[i] = (0.367 * Ld) / np.sqrt((np.sqrt(Pth[i]) - 0.852) ** 2 - 0.0219)

P = [1, 1.5, 2, 3, 4, 5, 6, 7.5, 10]
L = [24, 18, 8.02, 4.55, 3.38, 2.76, 2.37, 1.99, 1.61]

# fig, axes = plt.subplots()
# axes.plot(Pth, Lth)
# axes.plot(P, L)
# axes.set_xlabel('Pin / Pcr', color='blue')
# axes.set_ylabel('z', color='blue')

f = np.linspace(0.01, 10., len(Pin))
Lf = np.zeros(len(Pin))

for i in range(len(Pin)):
    Lf[i] = 1 / (1 / f[i] + 1 / 3.5726)

fig, axes = plt.subplots()
axes.plot(f, Lf)
axes.set_xlabel('f ', color='blue')
axes.set_ylabel('z', color='blue')


plt.show()
