import numpy as np
import matplotlib.pyplot as plt

tau0 = 100e-15 / (2 * np.sqrt(np.log(2.)))

tmin, tmax, Nt = -5. * tau0, 5 * tau0, 1024
t = np.linspace(tmin, tmax, Nt)
dt = t[1] - t[0]

I = 1e40 * 1e4 * np.exp(-t ** 2 / tau0 ** 2)

rhont = 2.5e25
K = 1
sigmaK = 1.2e-41


rho = np.zeros(Nt)
rho[0] = 0.
for i in range(1, Nt):
    R = sigmaK * I[i -1] ** K
    rho[i] = rhont - (rhont-rho[i-1]) * np.exp(-R * dt)


plt.plot(t, I / max(I) * max(rho) / rhont)
plt.plot(t, rho/rhont)

plt.show()