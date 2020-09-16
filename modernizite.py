import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.constants as sc
import hankel


def radius(x, I, lvl: object = np.exp(1.)):
    Nx = len(I)
    Imax = np.max(I)
    imax = np.argmax(I)
    rad = -1.
    for i in range(imax, Nx):
        if I[i] <= Imax / lvl:
            rad = x[i] - x[imax]
            break
    return rad


C0 = sc.c
EPS0 = sc.epsilon_0

# ********************************************************************
# input
# ********************************************************************
rmax, Nr = 10e-3, 1024

xmin, xmax, Nx = -10e-3, 10e-3, 1024
ymin, ymax, Ny = -10e-3, 10e-3, 1024

n0 = 1.                      # [-] refractive index
n2 = 1e-23                   # [m**2/W]
lam0 = 744e-9                # [m] wavelength
a0 = 1e-3                    # [m] radius
I0 = 1.5e16                # [W/m**2]  initial intensity
w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0            # wave number
Pmax = 3.72 * lam0 ** 2 / (8 * np.pi * n2)
# print(Pmax / 1e9, Pmax / (np.pi * a0 ** 2) / 1e16)


f = 0.5   # fokal

zmin, zmax = 0., 20
dz0 = zmax / 100.

NONLINEARITY = False

phimax = np.pi / 100.
Istop = 100. * I0

fname = "results/hankel.h5"

# ********************************
# Grid
# *******************************

ht = hankel.dht(rmax, Nr)
r = ht.rcoord()
kr = 2. * np.pi * ht.vcoord()

# **********************************
# Instal comdition
# **********************************
ksi = 0.5 * n0 * EPS0 * C0

A = np.sqrt(I0 / ksi)
E = A * np.exp(-0.5 * r ** 2 / a0 ** 2) # * np.exp(-1j * k0 * r ** 2 / (2. * f))

# **********************************
# Write
# **********************************

iwrite = 0
fp = h5py.File(fname, "w")
dset = fp.create_dataset("grid/r", data=r)
dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
dset.attrs["z"] = zmin
fp.close()

# *****
# Main cycle
# *******

print(I0/1e16)

z = zmin
k = 0
m = 0
I = ksi * np.abs(E) ** 2

while z < zmax:
    I = ksi * np.abs(E) ** 2
    Imax = np.max(I)
    if NONLINEARITY:
        dzk = phimax / (w0 / C0 * n2 * Imax)
    else:
        dzk = dz0
    dz = min(dzk, dz0)
    z += dz
    S = ht.dht(E)
    S = S * np.exp(-1j * kr ** 2 / (2. * k0) * dz)  # 3. compute spectrum at desired distance
    E = ht.idht(S)  # 4. compute field at desired distance

    if NONLINEARITY:
        E = E * np.exp(1j * w0 / C0 * n2 * I * dz)

    print(Imax / 1e16)

    iwrite += 1
    fp = h5py.File(fname, "a")
    dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
    dset.attrs["z"] = z

    fp.close()

    if Imax > Istop:
        print("Too high!")
        print('z=', z)
        break


print("Pmax=0.2")
# ******************************
# Ananlis
# ********************************
# I = np.abs(E) ** 2
# plt.plot(z, Imax)
# cmap = "viridis"
# norm = mpl.colors.Normalize(vmin=1e-3, vmax=1.)
# plt.pcolormesh(x, y, I.transpose(), cmap=cmap, norm=norm)
# plt.show()

# T = zmax/1e-3 * lam0
# print(T)
#
# fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
#
# norm = mpl.colors.LogNorm(vmin=1e-3, vmax=1.)
#
# cmap = "inferno"
#
# p0 = axes[0].pcolormesh(x / 1e-3, y / 1e-3, I0.transpose(), norm=norm, cmap=cmap)
# p1 = axes[1].pcolormesh(x / 1e-3, y / 1e-3, I.transpose(), norm=norm, cmap=cmap)
#
# plt.colorbar(p0, ax=axes[0], orientation="horizontal")
# plt.colorbar(p1, ax=axes[1], orientation="horizontal")
#
# plt.tight_layout()
# plt.show()
