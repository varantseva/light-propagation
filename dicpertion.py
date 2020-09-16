import numpy as np
import math as m
import matplotlib.pyplot as plt


TAUfwhm = 100e-15
TAU = TAUfwhm / (2 * m.sqrt(2))

tmin, tmax, Nt = -700e-15, 700e-15, 1024
t = np.linspace(tmin, tmax, Nt)
dt = t[1] - t[0]
omega = 2. * np.pi * np.fft.fftfreq(Nt, dt)
k2 = 2e-29

E0 = np.exp(- t ** 2 / (2 * TAU ** 2))
S0 = np.fft.ifft(E0)

zmin, zmax, Nz = 0., 500., 20
z = np.linspace(zmin, zmax, Nz)

for i in range(Nz):
    S = S0 * np.exp(1j * k2 * omega ** 2 / 2. * z[i])
    E = np.fft.fft(S)
    I = np.abs(E) ** 2

plt.plot(t, I)
plt.show()
