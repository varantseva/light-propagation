import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

xmin, xmax, Nx = -10e-3, 10e-3, 2048
ymin, ymax, Ny = -10e-3, 10e-3, 2048

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(kx, ky)

C0 = 3e8  # [m/s] speed of light in vacuum
n0 = 1.  # [-] refractive index
lam0 = 650e-9  # [m] wavelength

w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0  # wave number

a0 = 2e-3  # [m] radius
E0 = np.exp(-0.5 * (X ** 2 + Y ** 2) / a0 ** 2)

S0 = np.fft.fft2(E0)

z = 100.
S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * z)
E = np.fft.ifft2(S)
I = np.abs(E) ** 2
I0 = np.abs(E0) ** 2

Ix0 = I0[:, int(Ny/2)]
Iy0 = I0[int(Nx/2), :]
Ix = I[:, int(Ny/2)]
Iy = I[int(Nx/2), :]

print(k0*a0**2)

norm = mpl.colors.Normalize(vmin=0., vmax=1.)
plt.pcolormesh(x / 1e-3, y / 1e-3, I.transpose(), norm=norm)
plt.xlabel("x, [mm]")
plt.ylabel("y, [mm]")
plt.colorbar()
plt.tight_layout()
plt.show()
