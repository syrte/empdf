"""
Author: Zhaozhou Li (lizz.astro@gmail.com)



Code structure

Tracer:
    Workhorse object

Estimator:
    Likelihoods

"""

# import os
# os.environ["OMP_NUM_THREADS"] = "20"
# os.environ["MKL_NUM_THREADS"] = "20"

import numpy as np
from handy import EqualGridInterpolator
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from scipy import stats

from .kde import KDE

# -----------------------------------------------
import cyper
import cython_gsl

pyx = cyper.inline(
    './integrator.pyx',
    include_dirs=[cython_gsl.get_include(), np.get_include()],
    library_dirs=[cython_gsl.get_library_dir()],
    cimport_dirs=[cython_gsl.get_cython_include_dir()],
    directives={'language_level': '3', 'cdivision': True},
    libraries=['gsl', 'gslcblas', 'm'],
    openmp='-fopenmp',
    fast_indexing=True,
)
Integrator = pyx.Integrator
Particle_dtype = pyx.Particle_dtype


# -----------------------------------------------
def set_attrs(obj, var_dict, var_list=None):
    """
    Set attrs of obj by given dict, skipping 'self'
    """
    if var_list is None:
        var_list = var_dict.keys()

    for key in var_list:
        if key != 'self':
            setattr(obj, key, var_dict[key])


# -----------------------------------------------
# TODO

def set_opt(**args):
    "See options_default for available options"
    global options
    options.update(**args)


options_default = dict(
    N_RBIN_I=1001,
    N_EBIN_I=100,  # interpolator for DF(E, j2)
    N_JBIN_I=100,  # interpolator for DF(E, j2)
    N_RBIN_R=50,
    N_VBIN_R=50,   # quadrature grids for rho(r)
    N_CBIN_R=50,   # quadrature grids for rho(r)
    KDE_opt={},    # KDE options for N_Ej2
)


options = dict()
set_opt(**options_default)


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


class Status:
    """
    status flag for updating parameters
    """
    pass


def make_grid(rmin, rmax, n, dr=None):
    """
    make interpolating grids
    """
    if dr is None:
        dr = 0.5 * (1 - np.cos(0.5 * np.pi / n)) * (rmax - rmin)

    r = np.logspace(np.log10(rmin + dr), np.log10(rmax + dr), n - 2) - dr
    dr1 = r[1] - r[0]
    dr2 = r[-1] - r[-2]

    r = np.hstack([rmin, rmin + 1e-2 * dr1, r[1:-1], rmax - 1e-2 * dr2, rmax])
    return r

# -----------------------------------------------


class PotUtility:
    N_RBIN_I = 501

    def __init__(self, rmin, rmax, pot, mass=None):
        """
        pot: potential object with callable '.potential' and (optional) '.mass'
        """
        # r = np.linspace(rmin, rmax, self.N_RBIN_I)
        r = make_grid(rmin, rmax, self.N_RBIN_I)
        U = pot(r)
        interp_pot = CubicSpline(r, U, extrapolate=False)

        if mass is None:
            GM = interp_pot.derivative()(r) * r**2
            # dU/dr=GM/r^2 => GM = r^2 dU/dr
        else:
            GM = mass(r)

        E_cir = 0.5 * GM / r + U
        L2_cir = GM * r

        # do not interp log(r) over E!
        interp_L2_cir = CubicSpline(E_cir, L2_cir, extrapolate=False)
        interp_r_cir = CubicSpline(E_cir, r, extrapolate=False)

        integrator = Integrator()
        integrator.set_potential(r, U, interp_pot.c)

        set_attrs(self, locals())

    def __call__(self, r):
        return self.interp_pot(r)

    def L2_max(self, E, return_r=False):
        """
        return_r:
            return the circular orbit give E, clipped to [rmin, rmax]
        """
        L2_max = self.interp_L2_cir(E)

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

    def __init__(self, Emin, Emax, pot_util, N_Ej2_interp, func_obs=None):

        E = np.linspace(Emin, Emax, self.N_EBIN_I)
        j2 = np.linspace(0, 1, self.N_JBIN_I)
        L2_max, r_pin = pot_util.L2_max(E, return_r=True)

        buffer = np.zeros((self.N_EBIN_I, self.N_JBIN_I), dtype=Particle_dtype)
        buffer['r'] = r_pin.reshape(-1, 1)
        buffer['E'] = E.reshape(-1, 1)
        buffer['L2'] = L2_max.reshape(-1, 1) * j2

        pot_util.integrator.set_data(buffer.reshape(-1), pot_util.rmin, pot_util.rmax)
        pot_util.integrator.solve_radial_limits()
        pot_util.integrator.compute_radial_period(set_t=True, set_tcur=False, set_tobs=False)

        Tr = buffer['Tr'].copy()
        p_Ej2 = N_Ej2_interp.pdf(np.stack([E, j2], axis=-1))
        f_Ej2 = p_Ej2 / (4 * np.pi**2 * Tr * L2_max)

        self.f_Ej2 = EqualGridInterpolator([E, j2], f_Ej2)
        self.p_Ej2 = EqualGridInterpolator([E, j2], p_Ej2)
        # self.Tr_Ej2 = EqualGridInterpolator([E, j2], Tr)

        self.buffer = buffer
        self.pot_util = pot_util
        self.rmin = pot_util.rmin
        self.rmax = pot_util.rmax

        self.func_obs = func_obs

        self._prepare_pdf()

    def f_EL2(self, E, L2):
        L2_max = self.pot_util.L2_max(E)
        j2 = L2 / L2_max
        return self.f_Ej2([E, j2])

    def __call__(self, E, j2):
        return self.f_Ej2([E, j2])

    def _prepare_pdf(self):
        # r = np.linspace(self.rmin, self.rmax, self.N_RBIN_R).reshape(-1, 1, 1)
        r = make_grid(self.rmin, self.rmax, self.N_RBIN_R).reshape(-1, 1, 1)
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

        r = r.reshape(-1)
        self.pdf_r = CubicSpline(r, p_r)
        self.cdf_r = CubicSpline(r, p_r * r**2).antiderivative(1)
        # XXX, Normalized or not?

        if self.func_obs is not None:
            pobs = self.func_obs(r)
            self.cdf_r_obs = CubicSpline(r, p_r * r**2 * pobs).antiderivative(1)


class Tracer:
    def __init__(self, r, v, rlim=None, rlim_obs=None, func_obs=None, pot=None):
        """
        r, v: array shape (n, 3)
        rlim: None or 2-tuple
        rlim_obs: None or array of shape (n, 2)
            Should be sub-interval of rlim.
        func_obs: callable
            Can be set though update_func_obs later.
        pot: callable
            Can be set though update_potential later.
        """
        rr, vv, vr, vt = decompose_r_v(r, v)
        K = 0.5 * vv**2
        L = vt * rr
        L2 = L**2
        n = len(rr)

        if rlim is None:
            rlim = rr.min(), rr.max()
        else:
            assert rlim[0] <= rr.min(), "rlim[0] <= min{|r|} is expected."
            assert rlim[1] >= rr.max(), "rlim[1] >= max{|r|} is expected."
        assert rlim[0] < rlim[1], "rlim[0] < rlim[1] is expected."

        stale = Status()  # status flags

        set_attrs(self, locals())

        # prepare buffer or orbit integration
        self.buffer = np.zeros(len(rr), dtype=Particle_dtype)
        self.buffer['r'] = rr
        self.buffer['L2'] = L2

        if rlim_obs is not None:
            assert np.all(rlim_obs[0] < rlim_obs[1])
            self.buffer['rmin_obs'] = rlim_obs[0]
            self.buffer['rmax_obs'] = rlim_obs[1]
        elif func_obs is not None:
            self.update_func_obs(func_obs)
        else:
            self._wobs = None

        if pot is not None:
            self.update_potential(pot)

    def update_func_obs(self, func_obs):
        self.func_obs = func_obs
        self._pobs = func_obs(self.rr)  # need normalization later!
        self._wobs = 1 / self._pobs

        self.stale.wobs = False
        self.stale.fobs = True
        self.stale.N_Ej2 = True
        self.stale.f_Ej2 = True

    def update_potential(self, pot):
        pot_util = PotUtility(self.rmin, self.rmax, pot)
        U = pot_util.interp_pot(self.rr)
        E = self.U + self.K
        L2_max = pot_util.L2_max(E)
        j2 = (self.L2 / L2_max).clip(max=1)
        Emin, Emax = E.min(), E.max()

        set_attrs(self, locals())

        self.buffer['E'] = E

        self.stale.rlim = True
        self.stale.Tr = True
        self.stale.phase = True

        if self.rlim_obs is None:
            self.stale.wobs = False  # no need of wobs
        else:
            self.stale.wobs = True

        self.stale.fobs = True
        self.stale.N_Ej2 = True
        self.stale.f_Ej2 = True

    def integrate(self, set_Tr=False, set_phase=False, set_wobs=False):
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
                self.phase = 0.5 * np.sign(self.vr) * self.buffer['Tr_cur'] / self.buffer['Tr']
                stale.phase = False

            if set_wobs:
                self._wobs = self.buffer['Tr'] / self.buffer['Tr_obs']
                stale.wobs = False

    def count_raidal_bin(self, rbin):
        integrator = self.pot_util.integrator
        if self._wobs is None:
            self.buffer['wgt'] = 1
        else:
            self.buffer['wgt'] = self._wobs
        integrator.set_data(self.buffer, self.rmin, self.rmax)
        bincount = integrator.count_raidal_bin()
        return bincount

    def build_N_Ej2(self, **kde_opt):
        """
        kde_opt:
            e.g., backend='sklearn', bw_factor=1, kernel='epanechnikov'
        """
        if not self.stale.N_Ej2:
            return

        data = np.stack([self.E, self.j2], axis=-1)
        weights = self._wobs
        boundary = [[self.Emin, None], [0, 1]]

        self.N_Ej2_interp = KDE(data, weights=weights, boundary=boundary, **kde_opt)
        self.stale.N_Ej2 = False

    def build_f_Ej2(self):
        if not self.stale.f_Ej2:
            return

        Emin = self.Emin
        Emax = self.Emax + self.N_Ej2_interp.bandwidth * self.N_Ej2_interp.scale[0] * 2  # Emax + 2 * bandwidth

        self.df_interp = DFInterpolator(Emin, Emax, self.pot_util, self.N_Ej2_interp, self.func_obs)
        self.stale.f_Ej2 = False

    def compute_fobs(self):
        "Normalized observation probability."
        if not self.stale.fobs:
            return

        if self.rlim_obs is not None:
            pobs_cdf = self.df_interp.cdf_r(self.rlim_obs)
            self.fobs = 1 / (pobs_cdf[:, 1] - pobs_cdf[:, 0])

        elif self.func_obs is not None:
            pobs_cdf = self.df_interp.cdf_r_obs([self.rmax, self.rmin])
            self.fobs = self._pobs / (pobs_cdf[:, 1] - pobs_cdf[:, 0])

        else:
            self.fobs = 1

        self.stale.fobs = False


class Estimator:
    """
    Examples
    --------
    estimator =  Estimator(r, v, rlim=[rmin, rmax], rlim_obs=rlim_obs)
    pot = pot_factory(param)
    lnp = estimator.lnp_emdf(pot)
    """

    def __init__(self, r, v, rlim=None, rlim_obs=None, func_obs=None, pot=None):
        """
        r, v: array shape (n, 3)
            Tracer kinematics
        rlim: None or 2-tuple
            Radius cut of the sample
        rlim_obs: None or array of shape (n, 2)
            Observation limit for each tracer
        func_obs: callable
            Radial completeness function.
        """
        n = len(r)
        assert r.shape == v.shape == (n, 3), "r and v should have the same shape (n, 3)."

        if rlim_obs is not None and func_obs is not None:
            raise ValueError("Only one of rlim_obs and func_obs can be specified.")

        self.tracer = Tracer(r, v, rlim=rlim, rlim_obs=rlim_obs, func_obs=func_obs, pot=pot)
        self.ntracer = self.tracer.n
        self.rmin = self.tracer.rmin
        self.rmax = self.tracer.rmax

    def update_pot(self,
                   pot=None,
                   func_obs=None,
                   set_Tr=False,
                   set_phase=False,
                   set_wobs=False,
                   set_fobs=False,
                   set_N_Ej2=False,
                   set_f_Ej2=False,
                   ):
        """
        pot: None or function of r
            Potential, unchanged if None.
        func_obs: None or function of r
            Radial completeness function, unchanged if None.
        """
        if pot is not None:
            self.tracer.update_potential(pot)
        elif self.tracer.pot is None:
            raise ValueError('Please set a `pot` first.')

        if func_obs is not None:
            if self.tracer.lim_obs is None:
                self.tracer.update_func_obs(func_obs)
            else:
                raise ValueError("Only one of rlim_obs and func_obs can be specified.")

        # solve the dependence
        if set_fobs:
            set_f_Ej2 = True
        if set_f_Ej2:
            set_N_Ej2 = True
        if set_N_Ej2:
            set_wobs = True
        if set_wobs or set_phase:
            set_Tr = True

        # calculations
        if set_Tr:
            self.tracer.integrate(set_Tr=set_Tr, set_phase=set_phase, set_wobs=set_wobs)
        if set_N_Ej2:
            self.tracer.build_N_Ej2()
        if set_f_Ej2:
            self.tracer.build_f_Ej2()
        if set_fobs:
            self.tracer.compute_fobs()

    def lnp_emdf(self, pot=None, mass=None, func_obs=None):
        self.update_pot(pot, mass, func_obs, set_Tr=True, set_wobs=True, set_pobs=True, set_N_Ej2=True)

        tracer = self.tracer
        Tr, L2_max = tracer.Tr, tracer.L2_max
        lnp_Ej = tracer.N_Ej2_interp.autopdf(log=True)

        lnp = lnp_Ej - np.log(Tr * L2_max).sum()

        if self.tracer.rlim_obs is None and self.tracer.func_obs is None:
            return lnp
        else:
            return lnp + np.log(self.tracer.fobs).sum()

    def lnp_opdf(self, pot=None, rbin=None):
        """
        rbin: None, int, or 1D float array
            If None, log(ntracer) will be used.
        """
        self.update_pot(pot)

        if rbin is None:
            rbin = max(round(np.log(self.tracer.n)), 2)

        if np.isscalar(rbin):
            rbin = np.linspace(self.rmin, self.rmax, int(rbin) + 1)

        # observation
        rbin_old = getattr(self, '_opdf_rbin', None)
        if not np.array_equal(rbin, rbin_old):
            self._opdf_rbin = rbin
            self._opdf_rcnt = np.histogram(self.tracer.rr, rbin)
        rcnt = self._opdf_rcnt

        # time average
        bincount = self.tracer.count_raidal_bin(rbin)
        bincount /= np.sum(bincount)
        pbin = bincount / bincount.sum()

        # log probability
        lnp = stats.multinomial.logpmf(rcnt, n=self.ntracer, p=pbin)
        return lnp

    def lnp_opdf_smooth(self, pot=None):
        raise NotImplementedError

    def stats_AD(self, pot=None):
        """
        Negative statistic returned for maximization procedure.
        """
        self._update_pot(pot, set_phase=True)

        phase_abs = np.abs(self.tracer.phase) * 2
        return -AndersonDarling_stat(phase_abs)

    def stats_MeanPhase(self, pot=None):
        """
        Negative statistic returned for maximization procedure.
        """
        self._update_pot(pot, set_phase=True)

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
    return stats.norm(m1, s1).pdf(x) * w + stats.norm(m2, s2).pdf(x) * (1 - w)
