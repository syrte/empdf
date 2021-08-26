"""
author: 
    Zhaozhou Li (lizz.astro@gmail.com)

history:
    This code is taken from my project mw_mass3.
"""

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ['NFWProf']


def cspl_interp(x, xp, yp):
    return CubicSpline(xp, yp)(x)


class NFWProf:
    def __init__(self, **kwargs):
        """Possible initialization
            NFWProf(rs=rs, rhos=rhos, G=G)
            NFWProf(rs=rs, vs=vs, G=G)
            NFWProf(Mh=Mh, Rh=Rh, c=c, G=G)
            NFWProf(Mh=Mh, rhoh=rhoh, c=c, G=G)

        Value of G depends on adopted units
            G = 43007.1     # kpc, 1e10 Msun, km/s
            G = 43.0071     # Mpc, 1e10 Msun, km/s
            G = 43007.1e-10 # kpc,      Msun, km/s
            G = 1           #  rs, rs*vs^2/G,   vs
        G for kpc/h, Msun/h is the same as for kpc, Msun
        """
        G = kwargs.pop('G', 1)

        if 'rs' in kwargs:
            if 'rhos' in kwargs:
                rs, rhos = kwargs['rs'], kwargs['rhos']
                vs = (4 * np.pi * G * rhos * rs**2)**0.5
            elif 'vs' in kwargs:
                rs, vs = kwargs['rs'], kwargs['vs']
                rhos = (vs / rs)**2 / (4 * np.pi * G)

            if 'rhoh' in kwargs:
                rhoh = kwargs['rhoh']
                c = np.logspace(-3, 3, 901)
                rhos_rhoh = c**3 / (np.log(1 + c) - c / (1 + c)) / 3
                c = np.exp(cspl_interp(np.log(rhos / rhoh), np.log(rhos_rhoh), np.log(c)))
                Rh = rs * c
                Mh = 4 * np.pi / 3 * rhoh * Rh**3

        elif 'c' in kwargs:
            if 'Rh' in kwargs:
                Mh, Rh, c = kwargs['Mh'], kwargs['Rh'], kwargs['c']
            elif 'rhoh' in kwargs:
                Mh, rhoh, c = kwargs['Mh'], kwargs['rhoh'], kwargs['c']
                Rh = (Mh / (4 * np.pi / 3 * rhoh))**(1 / 3)
            rs = Rh / c
            rhos = Mh / (4 * np.pi * rs**3) / (np.log(1 + c) - c / (1 + c))
            vs = (4 * np.pi * G * rhos * rs**2)**0.5

        self.__dict__.update(locals())

    def rho(self, r):
        x = r / self.rs
        return self.rhos / x / (1 + x)**2

    def potential(self, r, zp=np.inf):
        x = r / self.rs
        if np.isposinf(zp):
            return self.vs**2 * (- np.log(1 + x) / x)
        elif zp == 0:
            return self.vs**2 * (1 - np.log(1 + x) / x)
        else:
            return self.potential(r) - self.potential(zp)

    def mass(self, r):
        x = r / self.rs
        return 4 * np.pi * self.rhos * self.rs**3 * (np.log(1 + x) - x / (1 + x))

    def rho_mean(self, r):
        "enclosed mean density"
        return self.mass(r) / (4 * np.pi / 3 * r**3)

    def to_agama(self):
        import agama
        pot = agama.Potential(type='Spheroid',
                              densityNorm=self.rhos,
                              scaleRadius=self.rs,
                              alpha=1, beta=3, gamma=1
                              )
        return pot


class PowProf:
    pass
