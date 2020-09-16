import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os

fname = "../results/pulse_sol.h5"

fp = h5py.File(fname, "r")
t = fp["grid/t"][:]
Nt = len(t)

dsets = list(fp["field"])
Nz = len(dsets)

z = np.zeros(Nz)
Izt = np.zeros((Nz, Nt))

for i in range(Nz):
    dset = dsets[i]
    data = fp["field/" + dset]
    E = data[:]
    z[i] = data.attrs["z"]
    I = np.abs(E) ** 2
    Izt[i] = I[:]

cmap = "viridis"
plt.pcolormesh(z, t, Izt.transpose(), cmap=cmap)
plt.colorbar()
plt.tight_layout()
plt.show()

fp.close()
