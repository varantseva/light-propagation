import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

C0 = 3e8  # [m/s] speed of light in vacuum

# ********************************************************************
# input
# ********************************************************************

im = mpl.image.imread("111.png")

im = im[:, :, 1]
E = 1 - im

Nx, Ny = E.shape
ration = Nx/Ny
xarea = 30e-3
yarea = xarea / ration

xmin, xmax = -0.5 * xarea, 0.5 * xarea
ymin, ymax = -0.5 * yarea, 0.5 * yarea

n0 = 1.  # [-] refractive index
lam0 = 1000e-9  # [m] wavelength
a0 = 0.5e-3  # [m] radius

zmin, zmax = 0., 1
dz = zmax / 100

fname = "results/outPNG.h5"

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


iwrite = 0
fp = h5py.File(fname, "w")
dset = fp.create_dataset("grid/x", data=x)
dset = fp.create_dataset("grid/y", data=y)
dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
dset.attrs["z"] = zmin
fp.close()

# *****
# Main cycle
# *******

I0 = np.abs(E) ** 2

z = zmin

while z < zmax:
    z += dz
    S = np.fft.fft2(E)
    S = S * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * dz)  # 3. compute spectrum at desired distance
    E = np.fft.ifft2(S)  # 4. compute field at desired distance

    iwrite += 1
    fp = h5py.File(fname, "a")
    dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
    dset.attrs["z"] = z
    fp.close()


I = np.abs(E) ** 2

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

norm = mpl.colors.LogNorm(vmin=1e-3, vmax=1.)

cmap = "viridis"

p0 = axes[0].pcolormesh(x / 1e-3, y / 1e-3, I0.transpose(), norm=norm, cmap=cmap)
p1 = axes[1].pcolormesh(x / 1e-3, y / 1e-3, I.transpose(), norm=norm, cmap=cmap)

plt.colorbar(p0, ax=axes[0], orientation="horizontal")
plt.colorbar(p1, ax=axes[1], orientation="horizontal")

plt.tight_layout()
plt.show()
