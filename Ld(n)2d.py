import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


def radius(x, I, lvl: object = np.exp(1.)) -> object:
    Nx = len(I)
    Imax = np.max(I)
    imax = np.argmax(I)
    rad = -1.
    for i in range(imax, Nx):
        if I[i] <= Imax / lvl:
            rad = x[i] - x[imax]
            break
    return rad


xmin, xmax, Nx = -10e-3, 10e-3, 512
ymin, ymax, Ny = -10e-3, 10e-3, 512

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
#
w0 = 2. * np.pi * C0 / lam0  # field frequency
# k0 = n0 * w0 / C0  # wave number

a0 = 2e-3  # [m] radius
E0 = np.exp(-0.5 * (X ** 2 + Y ** 2) / a0 ** 2)

S0 = np.fft.fft2(E0)

zmin, zmax, Nz = 0., 100., 100  # [m] propagation distance
z = np.linspace(zmin, zmax, Nz)

# S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * z)
# E = np.fft.ifft2(S)
# I = np.abs(E) ** 2
I0 = np.abs(E0) ** 2

n = np.linspace(1, 3, 30)
Ld = np.zeros(30)

for j in range(30):
    k0 = n[j] * w0 / C0
    for i in range(Nz):
        S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * z[i])  # 3. compute spectrum at desired distance
        E = np.fft.ifft2(S)  # 4. compute field at desired distance
        I = np.abs(E)**2
        Imax = np.max(I)
        print(Imax, np.max(I0))
        if Imax <= np.max(I0) / 2:
            print(z[i])
            Ld[j] = z[i]
            break

# Ix0 = I0[:, int(Ny/2)]
# Iy0 = I0[int(Nx/2), :]
# Ix = I[:, int(Ny/2)]
# Iy = I[int(Nx/2), :]

k0 = n * 2. * np.pi / lam0
Ldth = k0 * a0 ** 2

# fig, axes = plt.subplots(1, 1)

plt.plot(n, Ld)
plt.plot(n, Ldth)
# axes.set_xlabel('wavelength', color='blue')
# axes.set_ylabel('Ld', color='blue')

plt.show()