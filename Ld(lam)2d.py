import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


def radius(x, I, lvl: object = np.exp(1.)) -> object:
    Nx = len(I)
    Imax = np.max(I)
    imax = np.argmax(I)
    rad = -1.
    for i in range(imax, Nx):
        if I[i] <= Imax / lvl:
            rad = x[i] - x[imax]
            break
    return rad


xmin, xmax, Nx = -10e-3, 10e-3, 512
ymin, ymax, Ny = -10e-3, 10e-3, 512

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2. * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2. * np.pi * np.fft.fftfreq(Ny, dy)

X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(kx, ky)

C0 = 3e8  # [m/s] speed of light in vacuum
n0 = 1.  # [-] refractive index


a0 = 2e-3  # [m] radius
lam0 = 650e-9  # [m] wavelength

w0 = 2. * np.pi * C0 / lam0  # field frequency
k0 = n0 * w0 / C0  # wave number
E0 = np.exp(-0.5 * (X ** 2 + Y ** 2) / a0 ** 2)

S0 = np.fft.fft2(E0)
Ldth = k0 * a0 ** 2
print(Ldth)
quit()

zmin, zmax, Nz = 0., 100., 1000  # [m] propagation distance
z = np.linspace(zmin, zmax, Nz)
I0 = np.abs(E0) ** 2

lammin, lammax, Nlam = 400e-9, 750e-9, 30
lam = np.linspace(lammin, lammax, Nlam)
LdLam = np.zeros(Nlam)

for j in range(Nlam):
    w = 2. * np.pi * C0 / lam[j]  # field frequency
    k = n0 * w / C0
    for i in range(Nz):
        S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k) * z[i])  # 3. compute spectrum at desired distance
        E = np.fft.ifft2(S)  # 4. compute field at desired distance
        I = np.abs(E) ** 2
        Imax = np.max(I)
        print(Imax, np.max(I0))
        if Imax <= np.max(I0) / 2:
            print(z[i])
            LdLam[j] = z[i]
            break


nmin, nmax, Nn = 1., 2., 30
n = np.linspace(nmin, nmax, Nn)
Ldn = np.zeros(Nn)

for j in range(30):
    k = n[j] * w0 / C0
    for i in range(Nz):
        S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k) * z[i])  # 3. compute spectrum at desired distance
        E = np.fft.ifft2(S)  # 4. compute field at desired distance
        I = np.abs(E)**2
        Imax = np.max(I)
        print(Imax, np.max(I0))
        if Imax <= np.max(I0) / 2:
            print(z[i])
            Ldn[j] = z[i]
            break


amin, amax, Na = 1e-3, 2e-3, 100
a = np.linspace(amin, amax, Na)
Lda = np.zeros(Na)

for j in range(Na):
    E0 = np.exp(-0.5 * (X ** 2 + Y ** 2) / a[j] ** 2)
    S0 = np.fft.fft2(E0)
    I0 = np.abs(E0) ** 2
    for i in range(Nz):
        S = S0 * np.exp(-1j * (KX ** 2 + KY ** 2) / (2. * k0) * z[i])  # 3. compute spectrum at desired distance
        E = np.fft.ifft2(S)  # 4. compute field at desired distance
        I = np.abs(E)**2
        Imax = np.max(I)
        print(Imax, np.max(I0))
        if Imax <= np.max(I0) / 2:
            print(z[i])
            Lda[j] = z[i]
            break

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].plot(LdLam, lam / 1e-9)
axes[1].plot(Ldn, n)
axes[2].plot(Lda, a / 1e-3)
axes[2].set_xlabel('Дифракционная длинна', color='blue')
axes[0].set_ylabel('Длина волны, нм', color='blue')
axes[1].set_ylabel('n', color='blue')
axes[2].set_ylabel('Радиус ручка', color='blue')
# k0 = n0 * 2. * np.pi / lam
Ldth = k0 * a0 ** 2



plt.show()
