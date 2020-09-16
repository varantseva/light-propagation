import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

fname = "results/puasson.h5"

C0 = 3e8  # [m/s] speed of light in vacuum
xmin, xmax, Nx = -15e-3, 15e-3, 1024
ymin, ymax, Ny = -15e-3, 15e-3, 1024

n0 = 1.  # [-] refractive index
lam0 = 1300e-9  # [m] wavelength
a0 = 0.5e-3  # [m] radius

zmin, zmax = 0., 1.
dz = zmax
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(kx, ky)

E0 = np.zeros((Nx, Ny), np.complex)
w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0  # wave number

iwrite = 0
fp = h5py.File(fname, "w")
dset = fp.create_dataset("grid/x", data=x)
dset = fp.create_dataset("grid/y", data=y)
dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
dset.attrs["z"] = zmin
fp.close()

for i in range(Nx):
    for j in range(Ny):
        if x[i] ** 2 + y[j] ** 2 > 3e-6:
            E0[i, j] = 1


z = zmin
S0 = np.fft.fft2(E0)

while z < zmax:
    z += dz
    S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * z)  # 3. compute spectrum at desired distance
    E = np.fft.ifft2(S)  # 4. compute field at desired distance

I0 = np.abs(E0) ** 2
I = np.abs(E) ** 2

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

p0 = axes[0].pcolormesh(x / 1e-3, y / 1e-3, I0.transpose())
p1 = axes[1].pcolormesh(x / 1e-3, y / 1e-3, I.transpose(),)

plt.colorbar(p0, ax=axes[0], orientation="horizontal")
plt.colorbar(p1, ax=axes[1], orientation="horizontal")

plt.tight_layout()
plt.show()
