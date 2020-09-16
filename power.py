import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.constants as sc

C0 = sc.c
EPS0 = sc.epsilon_0
xmin, xmax, Nx = -10e-3, 10e-3, 1024
ymin, ymax, Ny = -10e-3, 10e-3, 1024
n0 = 1.          # [-] refractive index
n2 = 1e-23       # [m**2/W]
lam0 = 744e-9    # [m] wavelength

a0 = 2e-3        # [m] radius
I0 = 0.045e16   # [W/m**2]  initial intensity
w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0   # wave number

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

Y, X = np.meshgrid(y, x)
KY, KX = np.meshgrid(ky, kx)

ksi = 0.5 * n0 * EPS0 * C0

A = np.sqrt(I0 / ksi)

E = A * np.exp(-0.5 * (X ** 2 + Y ** 2) / a0 ** 2) + 0j
I = ksi * np.abs(E) ** 2
P = 0
for i in range(Nx):
    for j in range(Ny):
        P += I[i, j] * dx * dy
print('alph=', P * 4 * np.pi * n0 * n2 / lam0 ** 2)
