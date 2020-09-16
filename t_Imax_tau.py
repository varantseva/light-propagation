import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os


def radius(x, I, lvl=np.exp(1.)):
    Nx = len(I)
    Imax = np.max(I)
    imax = np.argmax(I)
    rad = -1.
    for i in range(imax, Nx):
        if I[i] <= Imax / lvl:
            rad = x[i] - x[imax]
            break
    return rad


fnames = [
    "../results/pulse_linear.h5",
    "../results/plasma.h5",
    ]

for fname in fnames:
    fp = h5py.File(fname, "r")

    t = fp["grid/t"][:]

    dsets = list(fp["field"])
    Nz = len(dsets)
    z = np.zeros(Nz)
    Imax = np.zeros(Nz)
    # tau = np.zeros(Nz)

    for i in range(Nz):
        dset = dsets[i]
        data = fp["field/" + dset]
        E = data[:]
        print(E)
        I = np.abs(E) ** 2
        Imax[i] = np.max(I)
        z[i] = data.attrs["z"]
        # if Imax[i] <= data[0][0] ** 2 / 2:
        #     tau[i] = z[i]

    fp.close()

    plt.plot(z, Imax / Imax[0], label=os.path.basename(fname))

plt.legend()
plt.tight_layout()
plt.show()
