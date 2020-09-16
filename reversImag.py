import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

fname = "../results/outsq.h5"

fp = h5py.File(fname, "r")

xmin, xmax, Nx = -10e-3, 10e-3, 1024
ymin, ymax, Ny = -10e-3, 10e-3, 1024

C0 = 3e8
n0 = 1.  # [-] refractive index
lam0 = 650e-9  # [m] wavelength
a0 = 0.5e-3  # [m] radius
w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0  # wave number

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

Y, X = np.meshgrid(y, x)
KY, KX = np.meshgrid(ky, kx)

zmin, zmax = 1., 0.
dz = zmin / 100

dset = "100"
data = fp["field/" + dset]
E = data[:]
fp.close()


z = zmin
while z > zmax:
    print(1)
    z -= dz
    S = np.fft.fft2(E)
    S = S * np.exp(1j * (KX ** 2 + KY ** 2) / (2. * k0) * dz)  # 3. compute spectrum at desired distance
    E = np.fft.ifft2(S)  # 4. compute field at desired distance

I = np.abs(E) ** 2

cmap = "viridis"
norm = mpl.colors.Normalize(vmin=1e-3, vmax=1.)
plt.pcolormesh(x, y, I.transpose(), cmap=cmap, norm=norm)
plt.show()
