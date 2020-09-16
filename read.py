import h5py
import numpy as np

fp = h5py.File("test.h5", "r")

dset = fp["pi"]
pi = dset[()]

a = fp["arrays/a"]
a = dset[10:50]


dset = fp["arrays/a2d"]
z = dset.attrs["z"]

print(a)

fp.close()
