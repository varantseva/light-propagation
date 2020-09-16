import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

fname = "../results/grating.h5"

dset = "000"

fp = h5py.File(fname, "r")

x = fp["grid/x"][:]
y = fp["grid/y"][:]

data = fp["field/" + dset]
E = data[:]
z = data.attrs["z"]

I = np.abs(E)**2
# I = I / np.max(I)
cmap = "inferno"
norm = mpl.colors.Normalize(vmin=1e-3, vmax=1.)

plt.pcolormesh(x / 1e-3, y / 1e-3, I.transpose(), cmap=cmap, norm=norm)
plt.xlabel("x, [мм]")
plt.ylabel("y, [мм]")
plt.colorbar()
plt.title("z={0:f}".format(z))
plt.tight_layout()
plt.show()

fp.close()
