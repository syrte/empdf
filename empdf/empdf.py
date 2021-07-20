"""
Author: Zhaozhou Li (lizz.astro@gmail.com)
"""

# import os
# os.environ["OMP_NUM_THREADS"] = "20"
# os.environ["MKL_NUM_THREADS"] = "20"

import numpy as np
from handy import EqualGridInterpolator
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from scipy.stats import multinomial
from scipy.stats import norm as normal

from .kde import KDE

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
def _setattr_dict(obj, var_dict, var_list=None):
    """
    Set attrs of obj by given dict, skipping 'self'
    """
    if var_list is None:
        var_list = var_dict.keys()

    for key in var_list:
        if key != 'self':
            setattr(obj, key, var_dict[key])


class Status:
    pass


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
class PotUtility:
    N_RBIN_I = 1001

    def __init__(self, rmin, rmax, pot):
        """
        pot: potential object with callable '.potential' and (optional) '.mass'
        """
        r = np.linspace(rmin, rmax, self.N_RBIN_I)
        U = pot.potential(r)

        if hasattr(pot, 'mass'):
            GM = pot.mass(r)
        else:
            GM = CubicSpline(r, U).derivative()(r) * r**2
            # dU/dr=GM/r^2 => GM = r^2 dU/dr

        E_cir = 0.5 * GM / r + U
        L2_cir = GM * r

        interp_lnL2_cir = CubicSpline(E_cir, np.log(L2_cir), extrapolate=False)
        interp_lnr_cir = CubicSpline(E_cir, np.log(r), extrapolate=False)

        integrator = Integrator()
        integrator.set_potential(r, U)

        _setattr_dict(self, locals())

    def L2_max(self, E, return_r=False):
        """
        return_r:
            return the circular orbit give E, clipped to [rmin, rmax]
        """
        L2_max = np.exp(self.interp_lnL2_cir(E))

        ix1 = E <= self.E_cir[0]
        L2_max[ix1] = self.rmin**2 * 2 * (E[ix1] - self.U[0])  # .clip(1e-20)  # avoid zero?

        ix2 = E >= self.E_cir[-1]
        L2_max[ix2] = self.rmax**2 * 2 * (E[ix2] - self.U[-1])  # .clip(1e-20)  # avoid zero?

        if return_r:
            r = np.exp(self.interp_lnr_cir(E))
            r[ix1] = self.rmin
            r[ix2] = self.rmax
            return L2_max, r
        else:
            return L2_max

    # self.pot = CubicSpline(r, p, bc_type='natural', extrapolate=False)
    # self.dpot_dr = self.pot.derivative(1)
    # def mass(self, r):
    #     return r**2 * self.dpot_dr(r) / G


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

    def __init__(self, Emin, Emax, pot_util, N_Ej2_interp):

        E = np.linspace(Emin, Emax, self.N_EBIN_I)
        j2 = np.linspace(0, 1, self.N_JBIN_I)
        L2_max, r_pin = pot_util.L2_max(E, return_r=True)

        buffer = np.zeros((self.N_EBIN_I, self.N_JBIN_I), dtype=Particle_dtype)
        buffer['r'] = r_pin.reshape(-1, 1)
        buffer['E'] = E.reshape(-1, 1)
        buffer['L2'] = L2_max.reshape(-1, 1) * j2

        pot_util.integrator.set_data(buffer.reshape(-1), pot_util.rmin, pot_util.rmax)
        pot_util.integrator.solve_radial_limits()
        pot_util.integrator.compute_radial_period(set_tcur=False, set_tobs=False)

        Tr = buffer['Tr'].copy()
        p_Ej2 = N_Ej2_interp(E, j2)
        f_Ej2 = p_Ej2 / (4 * np.pi**2 * Tr * L2_max)

        self.f_Ej2 = EqualGridInterpolator([E, j2], f_Ej2)
        self.p_Ej2 = EqualGridInterpolator([E, j2], p_Ej2)
        self.Tr_Ej2 = EqualGridInterpolator([E, j2], Tr)

        self.buffer = buffer
        self.pot_util = pot_util
        self.rmin = pot_util.rmin
        self.rmax = pot_util.rmax

        self._prepare_pdf()

    def f_EL2(self, E, L2):
        L2_max = self.pot_util.L2_max(E)
        j2 = L2 / L2_max
        return self.f_Ej2([E, j2])

    def _prepare_pdf(self):
        r = np.linspace(self.rmin, self.rmax, self.N_RBIN_R).reshape(-1, 1, 1)
        U = self.pot_util(r)
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


class Tracer:
    def __init__(self, r, v, rlim=None, rlim_obs=None):
        """
        r, v: array shape (n, 3)
        rlim: None or 2-tuple
        rlim_obs: None or array of shape (n, 2)
            Should be sub-interval of rlim.
        """
        rr, vv, vr, vt = decompose_r_v(self.r, self.v)
        K = 0.5 * vv**2
        L = vt * rr
        L2 = L**2
        n = len(rr)

        if rlim is None:
            rlim = rr.min(), rr.max()
        else:
            assert rlim[0] <= rr.min(), "rlim[0] <= min{|r|} is expected."
            assert rlim[1] >= rr.max(), "rlim[1] >= max{|r|} is expected."

        if rlim_obs is None:
            rlim_obs = rlim

        rmin, rmax = rlim
        rmin_obs, rmax_obs = rlim_obs
        assert rmin < rmax
        assert np.all(rmin_obs < rmax_obs)

        # prepare buffer or orbit integration
        buffer = np.zeros(len(rr), dtype=Particle_dtype)
        buffer['r'] = rr
        buffer['L2'] = L2
        buffer['rmin_obs'] = rmin_obs
        buffer['rmax_obs'] = rmax_obs

        stale = Status()  # status flags

        _setattr_dict(self, locals())

    def update_potential(self, pot_util):
        # pot_util = PotUtility(self.rmin, self.rmax, pot)

        U = pot_util.pot.potential(self.rr)
        E = self.U + self.K
        L2_max = pot_util.L2_max(E)
        j2 = self.L2 / L2_max
        Emin, Emax = E.min(), E.max()

        _setattr_dict(self, locals())

        self.buffer['E'] = E

        self.stale.rlim = True
        self.stale.Tr = True
        self.stale.phase = True
        self.stale.wobs = True
        self.stale.N_Ej2 = True
        self.stale.grid = True

    def integrate(self, set_Tr=True, set_phase=False, set_wobs=False):
        # orbit integration: Tr and wgt_obs

        stale = self.stale
        integrator = self.pot_util.integrator
        integrator.set_data(self.buffer, self.rmin, self.rmax)

        if stale.rlim:
            integrator.solve_radial_limits()
            stale.rlim = False

        if set_phase or set_wobs:
            set_Tr = True

        set_Tr = stale.Tr and set_Tr
        set_phase = stale.phase and set_phase
        set_wobs = stale.wobs and set_wobs

        if (set_Tr or set_phase or set_wobs):
            integrator.compute_radial_period(set_t=set_Tr, set_tcur=set_phase, set_tobs=set_wobs)

            if set_Tr:
                self.Tr = self.buffer['Tr'].copy()
                stale.Tr = False
            if set_phase:
                self.theta = 0.5 * np.sign(self.vr) * self.buffer['Tr_cur'] / self.buffer['Tr']
                stale.phase = False

            if set_wobs:
                self.wgt_obs = self.buffer['Tr'] / self.buffer['Tr_obs']
                self.buffer['wgt_obs'] = self.wgt_obs
                stale.wgt_obs = False

    def count_raidal_bin(self, rbin):
        integrator = self.pot_util.integrator
        integrator.set_data(self.buffer, self.rmin, self.rmax)
        bincount = integrator.count_raidal_bin()
        return bincount

    def make_N_Ej2_interp(self, **kde_opt):
        """
        kde_opt:
            e.g., backend='sklearn', bw_factor=1, kernel='epanechnikov'
        """
        if not self.stale.N_Ej2:
            return

        tracer = self.tracer

        data = np.stack([tracer.E, tracer.j2], axis=-1)
        weights = tracer.wobs

        self.N_Ej2_interp = KDE(
            data, weights=weights,
            boundary=[[tracer.Emin, None], [0, 1]], **kde_opt)
        self.stale.N_Ej2 = False

    def make_orbit_grid(self):
        if not self.stale.grid:
            return

        tracer = self.tracer
        N_Ej2_interp = self.N_Ej2_interp

        Emin = tracer.Emin
        Emax = tracer.Emax + N_Ej2_interp.bandwidth * N_Ej2_interp.scale[0] * 2  # Emax + 2 * bandwidth

        self.df_interp = DFInterpolator(Emin, Emax, self.pot_util, self.N_Ej2_interp)
        self.stale.grid = False


class Estimator:
    def __init__(self, r, v, pot_factory, pot_param=None,
                 rlim=None, rlim_obs=None, pobs_fun=None):
        """
        r, v: array shape (n, 3)
            Tracer kinematics
        pot_factory: callable
            A function of potential
        pot_param: 
            Initial parameters
        rlim: None or 2-tuple
            Radius cut of the sample
        rlim_obs: None or array of shape (n, 2)     
            Observation limit for each tracer   
        pobs_fun: callable
            Radial completeness function, not implemented yet.
        """
        n = len(r)
        assert r.shape == v.shape == (n, 3), "r and v should have the same shape (n, 3)."

        if rlim_obs is not None and pobs_fun is not None:
            raise ValueError("Only one of rlim_obs and pobs_fun can be specified.")
        if pobs_fun is not None:
            raise NotImplementedError

        self.tracer = Tracer(r, v, rlim=None, rlim_obs=None)
        self.ntracer = self.tracer.n
        self.rmin = self.tracer.rmin
        self.rmax = self.tracer.rmax

        self.rlim_obs = rlim_obs
        self.pobs_fun = pobs_fun

        self.pot_factory = pot_factory
        self.pot_param = None  # initialize with None

        if pot_param is not None:
            self.update_param(pot_param)

    def update_param(self, param=None,
                     set_Tr=False,
                     set_phase=False,
                     set_wobs=False,
                     set_p_Ej2=False,
                     set_grid=False):
        if param is None:
            if self.pot_param is None:
                raise ValueError('The current param was not set.')
        elif np.array_equal(param, self.pot_param):
            pass
        else:
            self.pot_param = param
            self.pot = self.pot_factory(*param)
            self.pot_util = PotUtility(self.rmin, self.rmax, self.pot)
            self.tracer.update_potential(self.pot_util)

        if set_grid:
            set_p_Ej2 = True
        if set_p_Ej2:
            set_Tr = True

        self.tracer.integrate(set_Tr=set_Tr, set_phase=set_phase, set_wobs=set_wobs)

        if set_p_Ej2:
            self.tracer.make_N_Ej2_interp()
        if set_grid:
            self.tracer.make_orbit_grid()

    def lnp_emdf(self, param=None):
        self.update_param(param, set_Tr=True, set_p_Ej2=True)

        tracer = self.tracer
        Tr, L2_max = tracer.Tr, tracer.L2_max
        p_Ej = self.N_Ej2_interp.autopdf()

        lnp = np.log(p_Ej / Tr / L2_max).sum()
        return lnp

    def lnp_opdf(self, param=None, rbin=None):
        """
        rbin: None, int, or 1D float array
            If None, log(ntracer) will be used.
        """
        self.update_param(param, set_phase=True)

        if rbin is None:
            rbin = int(max(round(np.log(self.tracer.n)), 2))

        if np.isscalar(rbin):
            rbin = np.linspace(self.rmin, self.rmax, rbin + 1)

        rbin_current = getattr(self, '_opdf_rbin', None)
        if not np.array_equal(rbin, rbin_current):
            self._opdf_rbin = rbin
            self._opdf_rcnt = np.histogram(np.self.tracer.rr, rbin)
        rcnt = self._opdf_rcnt

        bincount = self.tracer.count_raidal_bin(rbin)
        bincount /= np.sum(bincount)

        lnp = multinomial.logpmf(rcnt, n=self.ntracer, p=bincount)
        return lnp

    def lnp_opdf_smooth(self, param=None):
        raise NotImplementedError

    def stats_AD(self, param=None):
        """
        Negative statistic returned for optimization procedure.
        """
        self._update_param(param, set_phase=True)

        phase_abs = np.abs(self.tracer.phase) * 2
        return -AndersonDarling_stat(phase_abs)

    def stats_MeanPhase(self, param=None):
        """
        Negative statistic returned for optimization procedure.
        """
        self._update_param(param)

        phase_abs = np.abs(self.tracer.phase) * 2
        return -MeanPhase_stat(phase_abs)


def MeanPhase_stat(x):
    "Normalized mean phase statistic"
    n = x.size
    return (np.mean(x, axis=1) - 0.5) * (12 * n)**0.5


def AndersonDarling_stat(x, axis=None):
    "Anderson Darling statistic"
    n = len(x)
    i = np.arange(n)
    x = np.sort(x, axis=axis)
    D = - n - np.mean((2 * i + 1) * np.log(x) + (2 * (n - i) - 1) * np.log(1 - x), axis=axis)
    return D


def AndersonDarling_prob(x):
    "Approximate pdf for AD statistic (esp. for n >= 5 and p > 1e-3), see Han et al. 2016"
    w, m1, s1, m2, s2 = 0.569, -0.570, 0.511, 0.227, 0.569
    return normal(m1, s1).pdf(x) * w + normal(m2, s2).pdf(x) * (1 - w)
