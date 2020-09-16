import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.constants as sc


C0 = sc.c
EPS0 = sc.epsilon_0
MU0 = sc.mu_0
QE = sc.e
ME = sc.m_e

# ********************************************************************
# input
# ********************************************************************

tmin, tmax, Nt = -700e-15, 700e-15, 1024

n0 = 1.
sigmaK = 1.2e-41              #[(m ** 2 / W) ** K / s] ionizations
K = 3
rhont = 2.5e25                #[1 / m **3] density of neutral molecules
k2 = -2e-29                   # GVD cof [m / c^2]
n2 = 1e-23                   # [m**2/W]
lam0 = 744e-9                # [m] wavelength
tau0 = 100e-15 / (2. * np.sqrt(np.log(2.)))
I0 = 1.e16                # [W/m**2]  initial intensity

zmin, zmax = 0., 100.
dz0 = zmax / 100.

NONLINEARITY = True
PLASMA = False

fname = "results/nonl.h5"
dzh5 = dz0
# ********************************
# Grid
# *******************************

t = np.linspace(tmin, tmax, Nt)
dt = t[1] - t[0]
w = 2. * np.pi * np.fft.fftfreq(Nt, dt)

# **********************************
# Instal comdition
# **********************************

w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0            # wave number
ksi = 0.5 * n0 * EPS0 * C0

A = np.sqrt(I0 / ksi)
E = A * np.exp((-0.5 * t ** 2) / (tau0 ** 2))

# gamma = n0 * EPS0 * n2 * w0 / 2
#
# I0 = -k2 * C0 / (n2 * w0 * tau0 ** 2)

# I0 = -k2 / (gamma * tau0**2) * ksi

phimax = np.pi / 100.
Istop = 100. * I0

# As = I0 / ksi
# As = -k2 / (gamma * tau0**2)
# A = np.sqrt(As)
# E = A * 2 / (np.exp(t / tau0) + np.exp(-t / tau0))
# E = A * np.exp(-1j * k0 * (X ** 2 + Y ** 2) / (2. * f))

I = ksi * np.abs(E) ** 2

rho = np.zeros(Nt)

mu = 1.0
Rp = MU0 * mu / (2. * k0) * QE ** 2 / ME

# **********************************
# Write
# **********************************

iwrite = 0
fp = h5py.File(fname, "w")
dset = fp.create_dataset("grid/t", data=t)
dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
dset.attrs["z"] = zmin
dzh5next = zmin + dzh5
fp.close()

# *****
# Main cycle
# *******

print(I0/1e16)

z = zmin
while z < zmax:

    I = ksi * np.abs(E) ** 2
    Imax = np.max(I)
    rhomax = np.max(rho)

    if NONLINEARITY:
        dzk = phimax / (w0 / C0 * np.abs(n2) * Imax)
    else:
        dzk = dz0

    if PLASMA:
        if rhomax == 0:
            dzp = dz0
        else:
            dzp = phimax / (Rp * rhomax)
    else:
        dzp = dz0

    dz = min(dzk, dz0, dzp)
    print(z, Imax / I0, rhomax / rhont)
    z += dz

    if NONLINEARITY:
        E = E * np.exp(1j * w0 / C0 * n2 * I * dz)

    if PLASMA:
        rho[0] = 0.
        for i in range(1, Nt):
            R = sigmaK * I[i - 1] ** K
            rho[i] = rhont - (rhont - rho[i - 1]) * np.exp(-R * dt)
        E = E * np.exp(-1j * Rp * rho * dz)

    S = np.fft.ifft(E)
    S = S * np.exp(1j * 0.5 * k2 * w ** 2 * dz)  # 3. compute spectrum at desired distance
    E = np.fft.fft(S)  # 4. compute field at desired distance

    print(Imax / 1e16)

    if z > dzh5next:
        iwrite += 1
        fp = h5py.File(fname, "a")
        dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
        dzh5next += dzh5
        dset.attrs["z"] = z

    fp.close()

    if Imax > Istop:
        print("Too high!")
        print('z=', z)
        break

# *********
# Ananlis
# *************
# I = np.abs(E) ** 2
# plt.plot(z, Imax)
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
