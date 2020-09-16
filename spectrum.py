import numpy as np
import matplotlib.pyplot as plt
import random

tau0 = 1.
w0 = 2. * 1.


tmin, tmax = -4. * np.pi, 4. * np.pi
Nt = 1024
t = np.linspace(tmin, tmax, Nt)
w = 2. * np.pi * np.fft.fftfreq(Nt, t[1] - t[0])


noise = np.zeros(Nt)
for i in range(Nt):
    noise[i] = random.random() - 0.5

# E = np.cos(w0 * t) + np.cos(5 * w0 * t + np.pi / 2)
E = np.exp(-0.5 * t ** 2 / tau0 ** 2) + noise
S = np.fft.ifft(E)

dw = 2.
F = np.exp(-w**20 / dw**20)
S = S * F
E2 = np.fft.fft(S)
S2 = np.abs(S) ** 2

print(w)

w = np.fft.ifftshift(w)
S2 = np.fft.ifftshift(S2)
F = np.fft.ifftshift(F)

# plt.plot(t, E)
plt.plot(w, S2)
plt.xlim(-10., 10.)
plt.plot(w, F * 0.01)

plt.show()
