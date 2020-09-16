import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os

fname = "../results/airy.h5"

fp = h5py.File(fname, "r")
x = fp["grid/x"][:]
y = fp["grid/y"][:]
Nx = len(x)
Ny = len(y)

dsets = list(fp["field"])
Nz = len(dsets)

z = np.zeros(Nz)
Izx = np.zeros((Nz, Nx))
for i in range(Nz):
    dset = dsets[i]
    data = fp["field/" + dset]
    E = data[:]
    z[i] = data.attrs["z"]

    I = np.abs(E) ** 2
    Ix = I[:, int(Ny / 2)]
    Izx[i, :] = Ix

cmap = "viridis"
norm = mpl.colors.Normalize(vmin=1e-3, vmax=1.)
plt.pcolormesh(z, x, Izx.transpose(), cmap=cmap, norm=norm)
plt.colorbar()
plt.tight_layout()
plt.show()

fp.close()
