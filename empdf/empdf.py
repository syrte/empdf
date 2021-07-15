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
from itertools import product as iter_product

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
def interp(x, xp, yp):
    return CubicSpline(xp, yp)(x)


def interp_logy(x, xp, yp):
    return np.exp(CubicSpline(xp, np.log(yp))(x))


def interp_loglog(x, xp, yp):
    return np.exp(CubicSpline(np.log(xp), np.log(yp))(np.log(x)))


# -----------------------------------------------
def compute_neff(weights, normed=False):
    if normed:
        return 1 / np.sum(weights**2)
    else:
        return np.sum(weights)**2 / np.sum(weights**2)


def scotts_factor(neff, d):
    """
    Scott's rule of thumb, adopted as default in scipy.stats.gaussian_kde

    neff: 
        Effective number of data points
    d: 
        Number of dimensions

    See Scott 2015, "Multivariate Density Estimation: Theory, Practice, and Visualization"
    """
    return neff**(-1. / (d + 4))


def MADN(x):
    """
    The normalized median absolute deviation (MADN),
    a robust measure of scale that is more robust than IQR.
    See https://en.wikipedia.org/wiki/Robust_measures_of_scale

    shape (n,) --> scalar
    shape (n, d) --> shape (d,)
    """
    return np.median(np.abs(x - np.median(x, axis=0)), axis=0) * 1.4826


def boundary_reflex(x, boundary=None, c_contiguous=False):
    """
    Parameters
    ----------
    x: shape(n, d)
    boundary: None or shape (d, 2)
    c_contiguous:
        Make no difference for building KDTree and queries.

    Returns
    -------
    out: shape(m*n, d)
        m is the number of combinations: m = prod_i (len(boundary[i]) + 1)
        out always satisfies out[:n] == x

    Examples
    --------
    n = 1000
    x = np.random.rand(n, 3)
    a = boundary_reflex(x, [[0, 1], [0], None])  # bounds: [0, 1]x[0, inf]x[-inf, inf]
    assert a.shape == (6*n, 3)
    assert np.array_equal(a[:n], x)
    plt.scatter(x.T[0], x.T[1])
    plt.scatter(a.T[0], a.T[1], s=5)
    """
    n, d = x.shape
    if boundary is None:
        out = x
    else:
        arr_list = [[] for i in range(d)]
        for i in range(d):
            xi = x[:, i]
            arr_list[i].append(xi)  # original
            if boundary[i] is None:
                continue
            else:
                for bound in boundary[i]:
                    if bound is None:
                        continue
                    else:
                        arr_list[i].append(2 * bound - xi)  # reflected
        arr_join = np.array(list(iter_product(*arr_list)))  # shape (m, d, n)
        out = arr_join.transpose(0, 2, 1).reshape(-1, d)  # correct shape (m*n, d)

    if c_contiguous:
        out = np.ascontiguousarray(out)
    return out


# See Table 6.2, Scott 2015
# and https://scikit-learn.org/stable/modules/density.html
kernel_var_dict = dict(
    tophat=1 / 3,
    linear=1 / 6,
    epanechnikov=1 / 5,
    normal=1,
    cosine=1 - 8 / np.pi**2
)


class KDE:
    N_BIN_FFT = 100

    def __init__(self, data, weights=None, boundary=None,
                 backend='sklearn', bw_factor=1, kernel='epanechnikov', **options):
        """
        data: shape (n, d)
        weights: shape (n,)
        boundary: None or shape (d, 2)
        backend: ['scipy', 'sklearn', 'KDEpy.FFTKDE', 'KDEpy.TreeKDE']
        bw_factor:
            bandwidth = Scott's rule * bw_factor
        options:
            sklearn: kernel='epanechnikov'

        data = np.stack([E, j2], axis=-1)
        kde = KDE2D(data, weights=w, boundary=[[Emin, None], [0, 1]],
                    backend='sklearn', kernel='epanechnikov')
        """
        n, d = data.shape

        # calculate neff before reflex
        if weights is None:
            neff = n
        else:
            assert weights.shape == (n,), "incorrect `weights` shape!"
            weights = weights / np.sum(weights)
            neff = compute_neff(weights, normed=True)
        bandwidth = scotts_factor(neff, d=d) * bw_factor

        # reflect to reduce bias at boundary
        if boundary is not None:
            assert len(boundary) == d, "incorrect `boundary` shape!"
            data = boundary_reflex(data, boundary=boundary)
            nrep = len(data) // n  # number of duplicates

            if weights is not None:
                weights = np.tile(weights, nrep)  # duplicate weights as well
        else:
            nrep = 1

        # calculate scale; scipy has its own scale
        if backend != 'scipy':
            # use the original n points for scale
            scale = MADN(data[:n])
            pdf_scale = np.prod(scale)
            data_scaled = data / scale

        # initialize KDE estimator
        if backend == 'scipy':
            from scipy.stats import gaussian_kde

            # initialize with the original n points for covariance matrix
            # this modified kde object is only designed for calling kde.pdf and kde.logpdf
            kde = gaussian_kde(data[:n].T, weights=weights[:n], bw_method=bandwidth)
            kde.dataset = data.T
            kde.n = len(data)
            if weights is None:
                kde._weights = np.ones(kde.n) / kde.n
            else:
                kde._weights = weights / nrep  # normalized to 1

        if backend == 'sklearn':
            from sklearn.neighbors import KernelDensity

            options.setdefault('rtol', 1e-6)
            bw_normed = bandwidth / kernel_var_dict[kernel]**0.5  # normalize kernels to var = 1
            kde = KernelDensity(kernel=kernel, bandwidth=bw_normed, **options)
            kde.fit(data_scaled, sample_weight=weights)

        elif backend == 'KDEpy.FFTKDE' or backend == 'KDEpy.TreeKDE':
            raise NotImplementedError

        ldict = locals()
        for key in ['kde', 'data', 'weights', 'boundary', 'backend', 'n', 'nrep',
                    'scale', 'pdf_scale', 'data_scaled', 'bandwidth']:
            if key in ldict:
                setattr(self, key, ldict[key])

    def pdf(self, data, scaled=False):
        if self.backend == 'scipy':
            return self.nrep * self.kde.pdf(data.T)

        elif self.backend == 'sklearn':
            if scaled:
                lnp = self.kde.score_samples(data)
            else:
                lnp = self.kde.score_samples(data / self.scale)
            return self.nrep / self.pdf_scale * np.exp(lnp)

        else:
            raise NotImplementedError

    def autopdf(self):
        if self.backend == 'scipy':
            return self.pdf(self.data[:self.n])
        else:
            return self.pdf(self.data_scaled[:self.n], scaled=True)


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
