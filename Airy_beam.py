import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.special as spec
import scipy.constants as sc


C0 = 3e8  # [m/s] speed of light in vacuum
xmin, xmax, Nx = -10e-3, 10e-3, 1024
ymin, ymax, Ny = -10e-3, 10e-3, 1024

n0 = 1.                      # [-] refractive index
n2 = 1e-23                   # [m**2/W]
lam0 = 744e-9                # [m] wavelength
a0 = 1e-3                    # [m] radius
I0 = 1.5e16                # [W/m**2]  initial intensity
w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0            # wave number
C0 = sc.c
EPS0 = sc.epsilon_0

zmin, zmax = 0., 3.
dz = zmax / 100

fname = "results/airy.h5"

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(kx, ky)

ksi = 0.5 * n0 * EPS0 * C0

x0 = 0.2e-3
y0 = 0.2e-3
a = 0.05

theta = -np.pi / 4
XR = X * np.cos(theta) - Y * np.sin(theta)
YR = X * np.sin(theta) + Y * np.cos(theta)
X = XR
Y = YR


Aix, Aip, Bi, Bip = spec.airy(X / x0)
Aiy, Aip, Bi, Bip = spec.airy(Y / y0)

E = Aix * Aiy * np.exp(a * (X / x0 + Y / y0))
E = E / np.max(E)

for i in range(Nx):
    for j in range(Ny):
        if i + 32 > j and i - 30 < j:
            if 1024 - i - 33 < j and 1024 - i + 30 > j:
                E[i][j] = 0


iwrite = 0
fp = h5py.File(fname, "w")
dset = fp.create_dataset("grid/x", data=x)
dset = fp.create_dataset("grid/y", data=y)
dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
dset.attrs["z"] = zmin


z = zmin
S = np.fft.fft2(E)

while z < zmax:
    I = ksi * np.abs(E) ** 2
    Imax = np.max(I)
    z += dz
    S = np.fft.fft2(E)
    S = S * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * dz)  # 3. compute spectrum at desired distance
    E = np.fft.ifft2(S)  # 4. compute field at desired distance

    iwrite += 1
    fp = h5py.File(fname, "a")
    dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
    dset.attrs["z"] = z
    fp.close()



