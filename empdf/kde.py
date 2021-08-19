"""
Author: Zhaozhou Li (lizz.astro@gmail.com)

KDE module.

Notes.
sklearn is about 3x faster than scipy.stats.gaussian_kde
For very large dataset, FFTKDE might be useful, not implemented yet. Check trial/empdf.py later.
"""

import numpy as np
from itertools import product as iter_product
from handy import EqualGridInterpolator


def compute_neff(weights, normed=False):
    if normed:
        return 1 / np.sum(weights**2)
    else:
        return np.sum(weights)**2 / np.sum(weights**2)


def scotts_factor(neff, d):
    """
    Scott's rule of thumb, adopted as default in scipy.stats.gaussian_kde

    Parameters
    ----------
    neff: 
        Effective number of data points
    d: 
        Number of dimensions

    See Scott 2015, "Multivariate Density Estimation: Theory, Practice, and Visualization"
    """
    return neff**(-1. / (d + 4))


def MADN(x, weights=None, axis=None, keepdims=False):
    """
    The normalized median absolute deviation (MADN),
    a robust measure of scale that is more robust than IQR.
    See https://en.wikipedia.org/wiki/Robust_measures_of_scale

    It is equivalent to scipy.stats.median_abs_deviation(x, axis=axis, scale='normal')
    which is only available for scipy>=1.5.0
    """
    scale = 0.6744897501960817  # special.ndtri(0.75)
    if weights is None:
        med = np.median(x, axis=axis, keepdims=True)
        madn = np.median(np.abs(x - med), axis=axis, keepdims=keepdims) / scale
    else:
        raise NotImplementedError
        # see https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
        # what if any(weights > 0.5*sum(weights))? => zero madn!!!
    return madn


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
    assert a.shape == (6 * n, 3)
    assert np.array_equal(a[:n], x)
    plt.scatter(x.T[0], x.T[1])
    plt.scatter(a.T[0], a.T[1], s=5)
    """
    if boundary is None:
        out = x
    else:
        n, d = x.shape
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


def boundary_reflex_grid(x, boundary=None):
    """
    helper function for FFTKDE
    If not None, boundary must be in grid points.

    >>> boundary_reflex_grid([np.linspace(0, 5, 6)])
    (array([0., 1., 2., 3., 4., 5.]),)
    >>> boundary_reflex_grid([np.linspace(0, 5, 6)], [[0, None]])
    (array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.]),)
    >>> boundary_reflex_grid([np.linspace(0, 5, 6), np.linspace(0, 2, 5)], [[0, 5], [0, None]])
    (array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
             8.,  9., 10.]),
     array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ]))
    """
    if boundary is None:
        out = x
    else:
        out = [[] for _ in range(len(x))]
        for i, xi in enumerate(x):
            out[i].append(xi)  # original
            if boundary[i] is None:
                continue
            else:
                for bound in boundary[i]:
                    if bound is None:
                        continue
                    else:
                        out[i].append(2 * bound - xi)  # reflected
            out[i] = np.sort(np.unique(np.concatenate(out[i])))
    return tuple(out)


# See Table 6.2, Scott 2015
# and https://scikit-learn.org/stable/modules/density.html
kernel_var_dict = dict(
    tophat=1 / 3,
    linear=1 / 6,
    epanechnikov=1 / 5,
    gaussian=1,
    cosine=1 - 8 / np.pi**2,
    exponential=2,
)

kernel_kdepy_dict = dict(
    tophap='box',
    linear='tri',
    epanechnikov='epa',
    gaussian='gaussian',
    cosine='cosine',
    exponential='exponential',
)


class KDE:
    N_BIN_FFT = 100

    def __init__(self, data, weights=None, bw_factor=1, boundary=None,
                 backend='sklearn', kernel='epanechnikov', **options):
        """
        Parameters
        ----------
        data: shape (n, d)
        weights: shape (n,)
            Data and weights.
        bw_factor:
            bandwidth = Scott's rule * bw_factor for scaled data
        boundary: None or shape (d, 2)
            If not None, reflex boundary correction is used for each axis.
        backend: ['scipy', 'sklearn', 'KDEpy.FFTKDE', 'KDEpy.TreeKDE']
            Backend to use.
        kernel: ['tophat', 'linear', 'epanechnikov', 'gaussian', 'cosine', 'exponential']
            scipy backend always uses gaussian kernel, hence ignoring this option.
        options:
            See docs of corresponding backend.
            KDEpy.FFTKDE, KDEpy.TreeKDE: 'grids'

        Examples
        --------
        data = np.stack([E, j2], axis=-1)
        kde = KDE(data, weights=w, boundary=[[Emin, None], [0, 1]],
                  backend='sklearn', kernel='epanechnikov')

        The original data is stored as kde.data[:kde.n].
        """
        n, d = data.shape

        # calculate neff and bandwidth
        if weights is None:
            neff = n
        else:
            assert weights.shape == (n,), "incorrect `weights` shape!"
            weights = weights / np.sum(weights)
            neff = compute_neff(weights, normed=True)

        bandwidth = scotts_factor(neff, d=d) * bw_factor

        # reflect data for boundary correction
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
            scale = MADN(data[:n], axis=0)  # use the original n points for scale
            pdf_scale = np.prod(scale)
            data_scaled = data / scale

        # initialize KDE estimator
        if backend == 'scipy':
            self._init_kde_scipy(data, weights, bandwidth, n, nrep, **options)
        elif backend == 'sklearn':
            self._init_kde_sklearn(data_scaled, weights, bandwidth, kernel, **options)
        elif backend == 'KDEpy.FFTKDE' or backend == 'KDEpy.TreeKDE':
            self._init_kde_kdepy(data_scaled, weights, bandwidth, kernel, backend, boundary, nrep, scale, pdf_scale, **options)

        ldict = locals()
        for key in ['data', 'weights', 'boundary', 'backend', 'kernel', 'bandwidth',
                    'n', 'd', 'nrep', 'scale', 'pdf_scale', 'data_scaled']:
            if key in ldict:
                setattr(self, key, ldict[key])

    def _init_kde_scipy(self, data, weights, bandwidth, n, nrep, **options):
        from scipy.stats import gaussian_kde

        # initialize with the original n points for covariance matrix
        # this modified kde object is only designed for calling kde.pdf and kde.logpdf
        if weights is None:
            kde = gaussian_kde(data[:n].T, bw_method=bandwidth, **options)
        else:
            kde = gaussian_kde(data[:n].T, bw_method=bandwidth, weights=weights[:n], **options)

        kde.dataset = data.T
        kde.n = len(data)
        if weights is None:
            kde._weights = np.ones(kde.n) / kde.n
        else:
            kde._weights = weights / nrep  # normalized to 1
        self.kde = kde

    def _init_kde_sklearn(self, data_scaled, weights, bandwidth, kernel, **options):
        from sklearn.neighbors import KernelDensity

        options.setdefault('rtol', 1e-6)
        bw_normed = bandwidth / kernel_var_dict[kernel]**0.5  # normalize kernels to var = 1

        self.kde = KernelDensity(kernel=kernel, bandwidth=bw_normed, **options).fit(data_scaled, sample_weight=weights)

    def _init_kde_kdepy(self, data_scaled, weights, bandwidth, kernel, backend, boundary, nrep, scale, pdf_scale, **options):
        kernel = kernel_kdepy_dict[kernel]

        grids = options.get('grids', None)
        log = options.setdefault('log', False)
        if grids is not None and hasattr(grids, '__len__') and not np.isscalar(grids[0]):
            grids = boundary_reflex_grid(grids, boundary)
            options['grids'] = [g / s for g, s in zip(grids, scale)]  # scale the grids

        x_grid, y_grid = kdepy_grid(data_scaled, weights=weights, kernel=kernel, bw=bandwidth, 
                                    backend=backend, return_grid=True, **options)
        x_grid = [g * s for g, s in zip(x_grid, scale)]

        if log:
            fill_value = -np.inf
            y_grid = np.log(nrep / pdf_scale) + y_grid
        else:
            y_grid = nrep / pdf_scale * y_grid
            fill_value = 0
        self.kde = EqualGridInterpolator(x_grid, y_grid, padding='constant', fill_value=fill_value)
        self.kde.log = bool(log)

    def _pdf_scipy(self, data, log=False):
        p = self.nrep * self.kde.pdf(data.T)

        if log:
            return np.log(p)
        else:
            return p

    def _pdf_sklearn(self, data, log=False):
        factor = self.nrep / self.pdf_scale
        lnp = np.log(factor) + self.kde.score_samples(data / self.scale)

        if log:
            return lnp
        else:
            return np.exp(lnp)

    def _pdf_kdepy(self, data, log=False):
        if log:
            if self.kde.log:
                return self.kde(data.T)
            else:
                return np.log(self.kde(data.T))
        else:
            if self.kde.log:
                return np.exp(self.kde(data.T))
            else:
                return self.kde(data.T)

    def __call__(self, data, log=False):
        return self.pdf(data, log=log)

    def pdf(self, data, log=False):
        """
        data: shape (n, d)
        log: bool
            If return p or lnp.
        """
        if self.backend == 'scipy':
            return self._pdf_scipy(data, log=log)

        elif self.backend == 'sklearn':
            return self._pdf_sklearn(data, log=log)

        elif self.backend == 'KDEpy.FFTKDE' or self.backend == 'KDEpy.TreeKDE':
            return self._pdf_kdepy(data, log=log)

    def autopdf(self, log=False):
        """
        Calculate the pdf of the underlying data points.
        """
        return self.pdf(self.data[:self.n], log=log)


def unfold_grid(x_fold):
    "function for kdepy_grid"
    ndim = x_fold.shape[-1]

    x_grid = [None] * ndim
    for i in range(ndim):
        ix = [0] * (ndim + 1)
        ix[i] = slice(None)
        ix[-1] = i
        x_grid[i] = x_fold[tuple(ix)]
    return tuple(x_grid)


def kdepy_grid(X, weights=None, grids=200, kernel='gaussian', bw=1, grids_tol=3,
               log=False, backend='KDEpy.FFTKDE', return_grid=False):
    """
    X: array
    weights: array
        Data and weights
    grids: int, tuple of int, tuple of array
        Grids to evaluate on, must cover all data points.
    kernel: str {'box', 'tri', 'epa', 'gaussian', 'cosine', 'exponential'}
        The kernel function.
    bw: float or str
        Bandwidth.
    grids_tol: float
        Tolerance used for generating auto grids.
    log: bool
        If true, log(prob) returned.
    backend: ['FFTKDE', 'TreeKDE']

    Example
    -------
    from scipy import stats

    ndat = 1000
    x = np.random.randn(ndat, 1)
    s = ndat**-0.2
    weights = np.ones(ndat)

    y1 = kdepy_grid(x, weights, grids=100, bw=s)(x.T)
    y2 = kdepy_grid(x, weights, grids=[np.linspace(-5, 5, 201)], bw=s)(x.T)
    y3 = stats.gaussian_kde(x.T, s, weights)(x.T)

    plt.scatter(x, y1, s=1)
    plt.scatter(x, y2, s=1)
    plt.scatter(x, y3, s=1)

    xx = np.linspace(-4, 4, 500)
    plt.plot(xx, stats.norm.pdf(xx), 'k--')

    plt.yscale('log')
    """

    if X.ndim == 1:
        X = X.reshape(-1, 1)
        grids = [grids]
    ndim = X.shape[-1]

    if np.isscalar(grids):
        grids = [grids] * ndim

    import KDEpy
    if backend == 'KDEpy.FFTKDE':
        kde = KDEpy.FFTKDE(kernel=kernel, bw=bw).fit(X, weights)
    elif backend == 'KDEpy.TreeKDE':
        kde = KDEpy.TreeKDE(kernel=kernel, bw=bw).fit(X, weights)

    # if np.isscalar(grids[0]):
    #     n_grid = tuple(grids)
    #     xx, yy = kde.evaluate(n_grid)
    #     x_fold = xx.reshape(*n_grid, ndim)
    #     x_grid = unfold_grid(x_fold)
    #     y_grid = yy.reshape(*n_grid)
    if np.isscalar(grids[0]):
        n_grid = grids
        xmin, xmax = X.min(axis=0) - bw * grids_tol, X.max(axis=0) + bw * grids_tol
        x_grid = tuple([np.linspace(xmin[i], xmax[i], n_grid[i]) for i in range(ndim)])
    else:
        n_grid = tuple([len(bin) for bin in grids])
        x_grid = tuple(grids)

    xx = np.stack(np.meshgrid(*x_grid, indexing='ij'), axis=-1).reshape(-1, ndim)
    yy = kde.evaluate(xx)
    y_grid = yy.reshape(*n_grid)

    if log:
        y_grid = np.log(y_grid)

    if return_grid:
        return x_grid, y_grid
    else:
        if log:
            fill_value = -np.inf
        else:
            fill_value = 0

        interp = EqualGridInterpolator(x_grid, y_grid, padding='constant', fill_value=fill_value)
        interp.log = bool(log)
        return interp


# -----------------------------
def test_compute_neff():
    assert compute_neff(np.ones(100)) == 100
    assert compute_neff(np.array([1, 1, 0])) == 2
    assert compute_neff(np.array([0.5, 0.5, 0]), normed=True) == 2


def test_scotts_factor():
    assert np.allclose(scotts_factor(100, 1), 0.3981071705534972)
    assert np.allclose(scotts_factor(100, 4), 0.5623413251903491)


def test_MADN():
    from numpy.random import default_rng
    rng = default_rng(10)
    vals = rng.standard_normal(1000)
    assert np.allclose(MADN(vals), 0.997429417584344)
    assert MADN(np.arange(50) / 50) == 0.3706505546264005
