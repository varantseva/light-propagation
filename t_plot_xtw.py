import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os

fname = "../results/pulseNL.h5"

fp = h5py.File(fname, "r")
t = fp["grid/t"][:]
Nt = len(t)
w = 2. * np.pi * np.fft.fftfreq(Nt, t[1] - t[0])
w = np.fft.ifftshift(w)

dsets = list(fp["field"])
Nz = len(dsets)

z = np.zeros(Nz)
Szt = np.zeros((Nz, Nt))
for i in range(Nz):
    dset = dsets[i]
    data = fp["field/" + dset]
    E = data[:]
    z[i] = data.attrs["z"]

    S = np.fft.ifft(E)
    S = np.fft.ifftshift(S)
    Szt[i, :] = np.abs(S) ** 2

cmap = "viridis"
norm = mpl.colors.Normalize(vmin=1e-3, vmax=1.)
plt.pcolormesh(z, t, Szt.transpose() / np.max(Szt[0, :]), cmap=cmap, norm=norm)
plt.colorbar()
plt.ylim(-2e14, 2e14)
plt.tight_layout()
plt.show()

fp.close()
