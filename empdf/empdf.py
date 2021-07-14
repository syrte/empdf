"""
Author: Zhaozhou Li (lizz.astro@gmail.com)
"""

# import os
# os.environ["OMP_NUM_THREADS"] = "20"
# os.environ["MKL_NUM_THREADS"] = "20"

import numpy as np
from oPDF import oPDF

from handy import EqualGridInterpolator
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from scipy.stats import gaussian_kde

# -----------------------------------------------
import cyper
import cython_gsl

pyx = cyper.inline(
    open('./integrator.pyx').read(),
    include_dirs=[cython_gsl.get_include(), np.get_include()],
    library_dirs=[cython_gsl.get_library_dir()],
    cimport_dirs=[cython_gsl.get_cython_include_dir()],
    cythonize_args={'language_level': '3'},
    libraries=['gsl', 'gslcblas', 'm'],
    openmp='-fopenmp',
    fast_indexing=True,
)
Integrator = pyx.Integrator
Particle_dtype = pyx.Particle_dtype
# -----------------------------------------------


G = 43007.1
# -----------------------------------------------


def decompose_r_v(r, v):
    rr = np.sqrt((r**2).sum(-1))
    vvsq = (v**2).sum(-1)
    vv = np.sqrt(vvsq)
    vr = (v * r).sum(-1) / rr
    vrsq = np.fmin(vr**2, vvsq)
    vt = np.sqrt(vvsq - vrsq)
    return rr, vv, vr, vt


# -----------------------------------------------
def interp(x, xp, yp):
    return CubicSpline(xp, yp)(x)


def interp_logy(x, xp, yp):
    return np.exp(CubicSpline(xp, np.log(yp))(x))


def interp_loglog(x, xp, yp):
    return np.exp(CubicSpline(np.log(xp), np.log(yp))(np.log(x)))


def compute_neff(weights):
    return weights.sum()**2 / (weights**2).sum()


def MADN(x):
    """
    the normalized median absolute deviation (MADN),
    a robust measure of scale that is more robust than IQR

    see https://en.wikipedia.org/wiki/Robust_measures_of_scale
    """
    return np.median(np.abs(x - np.median(x))) * 1.4826


class KDE2D:
    N_BIN_FFT = 100

    def __init__(self, xp, yp, weights=None, backend='sklearn', bw=None, **args):
        """
        backend:
            'scipy', 'sklearn', 'KDEpy.FFTKDE', 'KDEpy.TreeKDE'
        args:
            sklearn: kernel=epanechnikov
        """
        self.backend = backend

        if self.backend != 'scipy':
            xscale, yscale = MADN(xp), MADN(yp)
            xp = xp / xscale
            yp = yp / yscale
            xy = np.stack([xp, yp], axis=-1)

            self.scale = xscale, yscale
            self.pdf_scale = xscale * yscale
            self.xy = xy

        if self.backend == 'scipy':
            from scipy.stats import gaussian_kde
            self.kde = gaussian_kde((xp, yp), weights=weights, **args)

        elif self.backend == 'sklearn':
            from sklearn.neighbors import KernelDensity
            args = {**dict(kernel='gaussian', rtol=1e-5), **args}
            self.kde = KernelDensity(bandwidth=bw, **args).fit(xy)

        elif self.backend == 'KDEpy.FFTKDE':
            from KDEpy import FFTKDE
            bin, dens = FFTKDE(bw=bw, kernel='epa').fit(xy).evaluate(self.NBIN_FFT)
            self.kde = EqualGridInterpolator(bin, dens/self.pdf_scale)

        elif self.backend == 'KDEpy.TreeKDE':
            from KDEpy import TreeKDE
            bin, dens = TreeKDE(bw=bw, kernel='epa').fit(xy).evaluate(self.NBIN_FFT)
            self.kde = EqualGridInterpolator(bin, dens/self.pdf_scale)

        self.xp = xp
        self.yp = yp

    def autopdf(self):
        return self.pdf(self.xy, scaled=True)

    def pdf(self, x, y=None, scaled=False):
        if not scaled and self.backend != 'scipy':
            xscale, yscale = self.scale
            x = x / xscale
            y = y / yscale

        if self.backend == 'scipy':
            return self.kde((x, y))

        elif self.backend == 'sklearn':
            if y is None:
                xy = x
            else:
                xy = np.stack([x, y], axis=-1)
            return np.exp(self.kde.score_samples(xy))/self.pdf_scale


class Potential:
    N_GRID = 1001

    def __init__(self, rmin, rmax, pot):
        r_cir = np.linspace(rmin, rmax, self.N_GRID)
        m_cir = self.pot.mass(r_cir)
        U_cir = self.pot.potential(r_cir)

        E_cir = 0.5 * G * m_cir / r_cir + U_cir
        L2_cir = G * m_cir * r_cir

        self.rmin = rmin
        self.rmax = rmax
        self.r_cir = r_cir
        self.E_cir = E_cir
        self.U_cir = U_cir
        self.L2_cir = L2_cir

    def L2_max(self, E, return_r=False):
        L2_max = interp_logy(E, self.E_cir, self.L2_cir)

        ix1 = E < self.E_cir[0]
        L2_max[ix1] = 2 * self.rmin**2 * (E[ix1] - self.U_cir[0])

        ix2 = E > self.E_cir[-1]
        L2_max[ix2] = 2 * self.rmax**2 * (E[ix2] - self.U_cir[-1])

        if return_r:
            r = interp_logy(E, self.E_cir, self.r_cir)
            r[ix1] = self.rmin
            r[ix2] = self.rmax
            return L2_max, r
        else:
            return L2_max


class DFInterpolator:
    N_EBIN_R = 50
    N_JBIN_R = 50
    N_RBIN = 50

    x1, w1 = roots_legendre(N_EBIN_R)
    x2, w2 = roots_legendre(N_JBIN_R)  # x2 in [-1, 1]
    x2 = 0.5 * x2 + 0.5  # x2 in [0, 1]
    w2 = 0.5 * w2

    def __init__(self, Emin, Emax, rmin, rmax, N_EBIN, N_JBIN, pot, N_Ej2_interp, integrator):
        E = np.linspace(Emin, Emax, N_EBIN)
        j2 = np.linspace(0, 1, N_JBIN)
        E2d, j22d = np.meshgrid(E, j2, sparse=True, indexing='ij')

        r_pin = pot.r_pin(E2d)
        L2_max = pot.L2_max(E2d)
        L2 = L2_max * j2

        parr = np.zeros((N_EBIN, N_JBIN), dtype=Particle_dtype)
        parr['r'] = r_pin
        parr['E'] = E2d
        parr['L2'] = L2

        integrator.update_data(parr.ravel(), rmin, rmax)
        integrator.solve_radial_limits()
        integrator.integrate_radial_period(set_tcur=False, set_tobs=False)

        Tr = parr['Tr']
        p_Ej2 = N_Ej2_interp(E, j2)
        f = p_Ej2 / (4 * np.pi**2 * Tr * L2_max)

        self.f_E_j2 = EqualGridInterpolator(E, j2, f)
        self.p_Ej2 = EqualGridInterpolator(E, j2, p_Ej2)
        self.Tr = EqualGridInterpolator(E, j2, Tr)

    def f_E_L2(self, E, L2):
        L2_max = self.pot.L2_max(E)
        j2 = L2 / L2_max
        return self.f_E_j2(E, j2)

    def prepare_pdf(self):
        r = np.linspace(self.rmin, self.rmax, self.N_RBIN)
        U = self.pot(r)
        vmax = (2 * (self.E_max - U))**0.5

        v = 0.5 * vmax * self.x1 + 0.5 * vmax
        dv = 0.5 * vmax * self.w1
        c = self.x2  # cos(theta)
        dc = self.w2

        v2 = v**2
        E = U + 0.5 * v2
        L2 = v2 * (1 - c**2)
        f = self.f_E_L2(E, L2)
        pdf_r = 4 * np.pi * (f * v2 * dv * dc).reshape(-1, 1).sum(-1)
        cdf_r = CubicSpline(r, pdf_r * r**2).antiderivative(1)

        self.pdf_r = pdf_r
        self.cdf_r = cdf_r

    def pdf_r(self, r):
        "rmin <= r <= rmax"
        if not hasattr(self, 'pdf_r_'):
            self.prepare_pdf()
        return interp(r, self.r, self.pdf_r_)

    def cdf_r(self, r):
        "rmin <= r <= rmax"
        pass


class Estimator:
    N_RBIN_INTERP = 501
    N_EBIN = 51
    N_JBIN = 51

    def __init__(self, r, v, pot_factory, rmin=None, rmax=None, rmin_obs=None, rmax_obs=None):
        """
        pot_factory: a function of potential
        """
        ntracer = len(r)
        assert r.shape == v.shape == (ntracer, 3), "r and v should have the same shape (n, 3)."

        rr, vv, vr, vt = decompose_r_v(self.r, self.v)
        K = 0.5 * vv**2
        L = vr * rr
        L2 = L**2

        parr = np.zeros(ntracer, dtype=Particle_dtype)
        parr['r'] = rr
        parr['L2'] = L2

        if rmin is None:
            rmin = rr.min()
        else:
            assert rmin <= rr.min(), "rmin <= min{|r|} is expected."
        if rmax is None:
            rmax = rr.max()
        else:
            assert rmax >= rr.max(), "rmax <= max{|r|} is expected."

        if rmin_obs is not None:
            assert len(rmin_obs) == ntracer
            parr['rmin_obs'] = rmin_obs
        if rmax_obs is not None:
            assert len(rmax_obs) == ntracer
            parr['rmax_obs'] = rmax_obs

        self.r = r
        self.v = v
        self.rmin = rmin
        self.rmax = rmax
        self.ntracer = ntracer
        self._pot_factory = pot_factory
        self._parr = parr
        self.integrator = Integrator(parr, rmin, rmax)
        self.param = None

        ldict = locals()
        for key in ['rr', 'vv', 'vr', 'vt', 'K', 'L', 'L2']:
            setattr(self, key, ldict[key])

    def mass(self, r):
        return r**2 * self.dpot_dr(r) / G

    def _update_param(self, param):
        if np.array_equal(param, self.param):
            return

        self.param = param
        self.pot = self._pot_factory(*param)

        r = np.linspace(self.rmin, self.rmax, self.N_RBIN_INTERP)
        p = self.pot.potential(r, param)
        self.pot = CubicSpline(r, p, bc_type='natural', extrapolate=False)
        self.dpot_dr = self.pot.derivative(1)

        self.integrator.update_potential(r, p)
        self.integrator.solve_radial_limits()
        self.integrator.integrate_radial_period(set_tcur=True, set_tobs=False)

        U = self.pot.potential(self.rr, param)
        E = self.K + U
        self._parr['E'] = E

        L2_max = self.pot.L2_max(E, return_r=False)
        j2 = L2 / L2_max

        T = tracer.data['Tr']
        if self.obs_correct:
            w = tracer.data['Tr']/tracer.data['Tr_obs']

        self.U = U
        self.E = E
        self.j2 = j2

        res, loc = {}, locals()
        for key in ['T', 'am_max',
                    'E', 'j2', 'wi',
                    'x', 'y', 'w', 'xx', 'yy', 'ww']:
            res[key] = loc[key]

        self.data_orb = res


        # preparing points
        # ----------------------------
        x, y, w = E, j2, wi
        xx = np.concatenate([x, x, x])
        yy = np.concatenate([y, 2 - y, -y])  # to make a symmetric boundary
        ww = np.concatenate([w, w, w])


    def _make_orbit_grid(self, ):
        Emin, Emax = self.E.min(), self.E.max()


    def lnp_emdf(self, param=None):
        if param is not None:
            self._update_param(param)
            self._prepare_emdf()

        orb = self.data_orb
        x, y, xx, yy = orb['x'], orb['y'], orb['xx'], orb['yy']
        T, am_max = orb['T'], orb['am_max']

        # XXX: need a weighted version
        p_Ej = gaussian_kde((xx, yy))((x, y))

        return np.log(p_Ej / T / am_max**2).sum()

    def lnp_adap(self, param=None):
        if param is not None:
            self._update_param(param)
            self._prepare_emdf()

        orb = self.data_orb
        x, y, xx, yy = orb['x'], orb['y'], orb['xx'], orb['yy']
        T, am_max = orb['T'], orb['am_max']

        n_ngb = np.floor(self.ntracer**0.5 * 2).astype('i')
        p_Ej = AdapKDE2D(
            xx, yy, n_eps=3, n_ngb=n_ngb,
            scale=y.std() / x.std()
        ).density(x, y)

        return np.log(p_Ej / T / am_max**2).sum()

    def lnp_opdf(self, param=None):
        self._update_param(param)
        oPDF.Estimators.RBinLike.nbin = int(max(round(np.log(self.ntracer)), 2))
        return self.integrator.likelihood(param, oPDF.Estimators.RBinLike)

    def lnp_AD(self, param=None):
        self._update_param(param)
        return -self.integrator.likelihood(param, oPDF.Estimators.AD)

    def lnp_MeanPhase(self, param=None):
        self._update_param(param)
        return -self.integrator.likelihood(param, oPDF.Estimators.MeanPhase)
