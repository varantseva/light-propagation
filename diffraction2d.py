import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

C0 = 3e8  # [m/s] speed of light in vacuum

# ********************************************************************
# input
# ********************************************************************

# im = mpl.image.imread("8.png")
#
# im = im[:, :, 1]
# E0 = 1 - im

# Nx, Ny = E0.shape
# ration = Nx/Ny
# xarea = 30e-3
# yarea = xarea / ration

# xmin, xmax = -0.5 * xarea, 0.5 * xarea
# ymin, ymax = -0.5 * yarea, 0.5 * yarea
xmin, xmax, Nx = -15e-3, 15e-3, 2048
ymin, ymax, Ny = -15e-3, 15e-3, 2048
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

n0 = 1.  # [-] refractive index
lam0 = 1000e-9  # [m] wavelength
a0 = 0.5e-3  # [m] radius

zmin, zmax = 0., 2
dz = zmax

# ********************************
# Grid
# *******************************

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

Y, X = np.meshgrid(y, x)
KY, KX = np.meshgrid(ky, kx)

# **********************************
# Instal comdition
# ********************************


w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0  # wave number
E0 = np.zeros((Nx, Ny), np.complex)

for i in range(Nx):
    print(abs(x[i]))
    for j in range(Ny):
        # if abs(x[i]) < 0.5e-3 and abs(y[j]) < 0.5e-2:  # square
        # if x[i] ** 2 + y[j] ** 2 <= 1e-6:              # circel
        if -5.5e-4 < x[i] < -5e-4 or 5e-4 < x[i] < 5.5e-4:    # triangle
        # if -2e-3 < y[j] < np.sqrt(3) * x[i] + 2e-3 and y[j] < np.sqrt(3) * -x[i] + 2e-3:  # Young
            E0[i, j] = 1

# *****
# Main cycle
# *******

z = zmin
S0 = np.fft.fft2(E0)

while z < zmax:
    z += dz
    S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * z)  # 3. compute spectrum at desired distance
    E = np.fft.ifft2(S)  # 4. compute field at desired distance

# *********
# Ananlis
# *************

I0 = np.abs(E0) ** 2
I = np.abs(E) ** 2

T = zmax/1e-3 * lam0
print(T)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

norm = mpl.colors.LogNorm(vmin=1e-3, vmax=1.)

cmap = "plasma"

p0 = axes[0].pcolormesh(x / 1e-3, y / 1e-3, I0.transpose(), norm=norm, cmap=cmap)
p1 = axes[1].pcolormesh(x / 1e-3, y / 1e-3, I.transpose(), norm=norm, cmap=cmap)

plt.colorbar(p0, ax=axes[0], orientation="horizontal")
plt.colorbar(p1, ax=axes[1], orientation="horizontal")

plt.tight_layout()
plt.show()
