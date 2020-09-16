import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.constants as sc
import hankel

C0 = sc.c
EPS0 = sc.epsilon_0
MU0 = sc.mu_0
QE = sc.e
ME = sc.m_e

# ********************************************************************
# input
# ********************************************************************
rmax, Nr = 10e-3, 500
tmin, tmax, Nt = -200e-15, 200e-15, 512

n0 = 1.
sigmaK = 1.2e-41  # [(m ** 2 / W) ** K / s] ionizations
K = 3
rhont = 2.5e25  # [1 / m **3] density of neutral molecules
k2 = 2e-29  # GVD cof [m / c^2]
n2 = 1e-23  # [m**2/W]
lam0 = 248e-9  # [m] wavelength
a0 = 2e-3
tau0 = 35e-15
     # 100e-15 / (2. * np.sqrt(np.log(2.)))
I0 = 1.e16  # [W/m**2]  initial intensity

zmin, zmax = 0., 10.
dz0 = zmax / 100.

NONLINEARITY = True
PLASMA = True

fname = "results/pulse_rt_nl_248.h5"
dzh5 = dz0
# ********************************
# Grid
# *******************************
ht = hankel.dht(rmax, Nr)
r = ht.rcoord()
kr = 2. * np.pi * ht.vcoord()

t = np.linspace(tmin, tmax, Nt)
dt = t[1] - t[0]
w = 2. * np.pi * np.fft.fftfreq(Nt, dt)

# **********************************
# Instal comdition
# **********************************

w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0  # wave number
ksi = 0.5 * n0 * EPS0 * C0

A = np.sqrt(I0 / ksi)
E = np.zeros((Nr, Nt), np.complex)
for i in range(Nr):
    E[i, :] = A * np.exp((-0.5 * t ** 2) / (tau0 ** 2)) * \
              np.exp((-0.5 * r[i] ** 2) / (a0 ** 2))

# gamma = n0 * EPS0 * n2 * w0 / 2
# I0 = -k2 * C0 / (n2 * w0 * tau0 ** 2)
# I0 = -k2 / (gamma * tau0**2) * ksi

phimax = np.pi / 100.
Istop = 1000. * I0

# As = I0 / ksi
# As = -k2 / (gamma * tau0**2)
# A = np.sqrt(As)
# E = A * 2 / (np.exp(t / tau0) + np.exp(-t / tau0))
# E = A * np.exp(-1j * k0 * (X ** 2 + Y ** 2) / (2. * f))

# I = ksi * np.abs(E) ** 2
# plt.pcolormesh(t, r, I)
# plt.show()
# quit()

rho = np.zeros((Nr, Nt))

mu = 1.0
Rp = MU0 * mu / (2. * k0) * QE ** 2 / ME
# **********************************
# Write
# **********************************

iwrite = 0
fp = h5py.File(fname, "w")
dset = fp.create_dataset("grid/t", data=t)
dset = fp.create_dataset("grid/r", data=r)
dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
dset.attrs["z"] = zmin
dset = fp.create_dataset("plasma/{0:03d}".format(iwrite), data=rho)
dset.attrs["z"] = zmin
dzh5next = zmin + dzh5
fp.close()

# *****
# Main cycle
# *******
print(I0 / 1e16)

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
        for i in range(Nr):
            rho[i, 0] = 0.
            for j in range(1, Nt):
                R = sigmaK * I[i, j - 1] ** K
                rho[i, j] = rhont - (rhont - rho[i, j - 1]) * np.exp(-R * dt)
        E = E * np.exp(-1j * Rp * rho * dz)

    # diffraction
    for j in range(Nt):
        Er = E[:, j]
        S = ht.dht(Er)
        S = S * np.exp(-1j * kr ** 2 / (2. * k0) * dz)  # 3. compute spectrum at desired distance
        Er = ht.idht(S)
        E[:, j] = Er

    # dispersion
    for i in range(Nr):
        Et = E[i, :]
        S = np.fft.ifft(Et)
        S = S * np.exp(1j * 0.5 * k2 * w ** 2 * dz)  # 3. compute spectrum at desired distance
        Et = np.fft.fft(S)
        E[i, :] = Et


    S = np.fft.ifft(E)
    S = S * np.exp(1j * 0.5 * k2 * w ** 2 * dz)  # 3. compute spectrum at desired distance
    E = np.fft.fft(S)  # 4. compute field at desired distance

    print(Imax / 1e16)

    if z > dzh5next:
        iwrite += 1
        fp = h5py.File(fname, "a")
        dset = fp.create_dataset("field/{0:03d}".format(iwrite), data=E)
        dset.attrs["z"] = z
        dset = fp.create_dataset("plasma/{0:03d}".format(iwrite), data=rho)
        dset.attrs["z"] = z
        dzh5next += dzh5


    fp.close()

    if Imax > Istop:
        print("Too high!")
        print('z=', z)
        break

# *********
# Ananlis
# *************
# # I = np.abs(E) ** 2
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
