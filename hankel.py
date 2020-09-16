"""
Implementation of the pth-order disrete Hankel transform.

Method:
    M. Guizar-Sicairos and J.C. Gutierrez-Vega, JOSA A, 21, 53 (2004).
    http://www.opticsinfobase.org/josaa/abstract.cfm?uri=JOSAA-21-1-53

Additional info:
    http://mathematica.stackexchange.com/questions/26233/computation-of-hankel-transform-using-fft-fourier
"""
import numpy as np
import scipy.special as spec


class dht:
    def __init__(self, R, Nr, p=0):
        jn_zeros = spec.jn_zeros(p, Nr + 1)
        a = jn_zeros[:-1]
        aNp1 = jn_zeros[-1]

        V = aNp1 / (2. * np.pi * R)
        J = abs(spec.jn(p + 1, a)) / R

        S = 2. * np.pi * R * V

        T = np.zeros((Nr, Nr))
        for m in range(Nr):
            T[m, :] = (2. * spec.jn(p, a[:] * a[m] / S) /
                       abs(spec.jn(p + 1, a[:])) /
                       abs(spec.jn(p + 1, a[m])) / S)

        self.R = R
        self.Nr = Nr
        self.a = a
        self.V = V
        self.T = T
        self.J = J

        self.RdivJ = self.R / self.J
        self.JdivV = self.J / self.V
        self.VdivJ = self.V / self.J
        self.JdivR = self.J / self.R

    def rcoord(self):
        r = self.a / (2. * np.pi * self.V)
        return r

    def vcoord(self):
        v = self.a / (2. * np.pi * self.R)
        return v

    def dht(self, f1):
        assert len(f1) == self.Nr
        F1 = f1 * self.RdivJ
        F2 = np.dot(self.T, F1)   # matrix product
        f2 = F2 * self.JdivV
        return f2

    def idht(self, f2):
        assert len(f2) == self.Nr
        F2 = f2 * self.VdivJ
        F1 = np.dot(self.T, F2)
        f1 = F1 * self.JdivR
        return f1
