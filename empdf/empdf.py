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
    """
    r, v: shape (n, 3)
    """
    rr = np.sqrt((r**2).sum(-1))
    vv2 = (v**2).sum(-1)
    vv = np.sqrt(vv2)
    vr = (v * r).sum(-1) / rr
    vr2 = np.fmin(vr**2, vv2)
    vt = np.sqrt(vv2 - vr2)
    return rr, vv, vr, vt


# -----------------------------------------------
# def interp(x, xp, yp):
#     return CubicSpline(xp, yp)(x)


# def interp_logy(x, xp, yp):
#     return np.exp(CubicSpline(xp, np.log(yp))(x))


# def interp_loglog(x, xp, yp):
#     return np.exp(CubicSpline(np.log(xp), np.log(yp))(np.log(x)))


# -----------------------------------------------

# -----------------------------------------------

class L2maxMapper:
    N_GRID = 1001

    def __init__(self, rmin, rmax, pot):
        """
        pot: potential object with callable '.potential' and (optional) '.mass'
        """
        r_cir = np.linspace(rmin, rmax, self.N_GRID)
        U_cir = pot.potential(r_cir)
        if hasattr(pot, 'mass'):
            m_cir = pot.mass(r_cir)
        else:
            m_cir = CubicSpline(r_cir, U_cir).derivative()(r_cir) * r_cir**2 / G
            # dU/dr=GM/r^2 => M = r^2 dU/dr / G

        E_cir = 0.5 * G * m_cir / r_cir + U_cir
        L2_cir = G * m_cir * r_cir

        interp_lnL2_cir = CubicSpline(E_cir, np.log(L2_cir), extrapolate=False)
        interp_lnr_cir = CubicSpline(E_cir, np.log(r_cir), extrapolate=False)

        self.__dict__.update(locals())
        del self.self

    def L2_max(self, E, return_r=False):
        """
        return_r:
            return the circular orbit give E, clipped to [rmin, rmax]
        """
        L2_max = np.exp(self.interp_lnL2_cir(E))

        ix1 = E <= self.E_cir[0]
        L2_max[ix1] = self.rmin**2 * 2 * (E[ix1] - self.U_cir[0])

        ix2 = E >= self.E_cir[-1]
        L2_max[ix2] = self.rmax**2 * 2 * (E[ix2] - self.U_cir[-1])

        if return_r:
            r = np.exp(self.interp_lnr_cir(E))
            r[ix1] = self.rmin
            r[ix2] = self.rmax
            return L2_max, r
        else:
            return L2_max


class DFInterpolator:
    N_EBIN_I = 100  # interpolator for DF(E, j2)
    N_JBIN_I = 100  # interpolator for DF(E, j2)

    N_RBIN_R = 50
    N_VBIN_R = 50  # quadrature grids for rho(r)
    N_CBIN_R = 50  # quadrature grids for rho(r)

    x1, w1 = roots_legendre(N_VBIN_R)
    x2, w2 = roots_legendre(N_CBIN_R)
    x1 = 0.5 * x1 + 0.5  # x1 in [0, 1]
    w1 = 0.5 * w1
    x2 = 0.5 * x2 + 0.5  # x2 in [0, 1]
    w2 = 0.5 * w2

    # @classmethod
    # def set_params(cls, N_RBIN_R):

    def __init__(self, Emin, Emax, L2mapper, N_Ej2_interp, integrator):

        E = np.linspace(Emin, Emax, self.N_EBIN_I)
        j2 = np.linspace(0, 1, self.N_JBIN_I)
        L2_max, r_pin = L2mapper.L2_max(E, return_r=True)

        parr = np.zeros((self.N_EBIN_I, self.N_JBIN_I), dtype=Particle_dtype)
        parr['r'] = r_pin.reshape(-1, 1)
        parr['E'] = E.reshape(-1, 1)
        parr['L2'] = L2_max.reshape(-1, 1) * j2

        integrator.update_data(parr.reshape(-1), L2mapper.rmin, L2mapper.rmax)
        integrator.solve_radial_limits()
        integrator.integrate_radial_period(set_tcur=False, set_tobs=False)

        Tr = parr['Tr']
        p_Ej2 = N_Ej2_interp(E, j2)
        f_Ej2 = p_Ej2 / (4 * np.pi**2 * Tr * L2_max)

        self.f_Ej2 = EqualGridInterpolator([E, j2], f_Ej2)
        self.p_Ej2 = EqualGridInterpolator([E, j2], p_Ej2)
        self.Tr_Ej2 = EqualGridInterpolator([E, j2], Tr)

        self.parr = parr
        self.L2mapper = L2mapper
        self.rmin = L2mapper.rmin
        self.rmax = L2mapper.rmax

        self._prepare_pdf()

    def f_EL2(self, E, L2):
        L2_max = self.L2mapper.L2_max(E)
        j2 = L2 / L2_max
        return self.f_Ej2([E, j2])

    def _prepare_pdf(self):
        r = np.linspace(self.rmin, self.rmax, self.N_RBIN_R).reshape(-1, 1, 1)
        U = self.L2mapper(r)
        vmax = (2 * (self.E_max - U))**0.5  # shape (nr, 1, 1)

        v = self.x1.reshape(-1, 1) * vmax  # shape (nr, nv, 1)
        dv = self.w1.reshape(-1, 1) * vmax
        c = self.x2  # cos(theta), shape (nc)
        dc = self.w2

        v2 = v**2
        E = U + 0.5 * v2
        L2 = r**2 * v2 * (1 - c**2)  # shape (nr, nv, nc)
        f = self.f_EL2(E, L2)  # shape (nr, nv, nc)
        p_r = 4 * np.pi * (f * v2 * dv * dc).reshape(-1, 1).sum(-1)

        r = r.reshape(-1, 1)
        self.pdf_r = CubicSpline(r, p_r)
        self.cdf_r = CubicSpline(r, p_r * r**2).antiderivative(1)


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
            w = tracer.data['Tr'] / tracer.data['Tr_obs']

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
