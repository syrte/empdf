"""
Author: Zhaozhou Li (lizz.astro@gmail.com)

KDE module.

Notes.
sklearn is about 3x faster than scipy.stats.gaussian_kde
For very large dataset, FFTKDE might be useful, not implemented yet. Check trial/empdf.py later.
"""

import numpy as np
from itertools import product as iter_product


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


def MADN(x, axis=None, keepdims=False):
    """
    The normalized median absolute deviation (MADN),
    a robust measure of scale that is more robust than IQR.
    See https://en.wikipedia.org/wiki/Robust_measures_of_scale

    It is equivalent to scipy.stats.median_abs_deviation(x, axis=axis, scale='normal')
    which is only available for scipy>=1.5.0
    """
    scale = 0.6744897501960817  # special.ndtri(0.75)
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis, keepdims=keepdims) / scale


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


# See Table 6.2, Scott 2015
# and https://scikit-learn.org/stable/modules/density.html
kernel_var_dict = dict(
    tophat=1 / 3,
    linear=1 / 6,
    epanechnikov=1 / 5,
    normal=1,
    gaussian=1,
    cosine=1 - 8 / np.pi**2
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
        kernel:
            Supported kernels:
            sklearn: epanechnikov, gaussian, cosine, linear, tophat
            KDEpy: epa, gaussian, cosine, ...
            scipy backend always uses gaussian kernel, hence ignoring this option.
        options:
            See docs of corresponding backend.

        Examples
        --------
        data = np.stack([E, j2], axis=-1)
        kde = KDE(data, weights=w, boundary=[[Emin, None], [0, 1]],
                  backend='sklearn', kernel='epanechnikov')

        The original data is stored as kde.data[:kde.n].
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
            from scipy.stats import gaussian_kde

            # initialize with the original n points for covariance matrix
            # this modified _kde object is only designed for calling _kde.pdf and _kde.logpdf
            if weights is None:
                _kde = gaussian_kde(data[:n].T, bw_method=bandwidth)
            else:
                _kde = gaussian_kde(data[:n].T, bw_method=bandwidth, weights=weights[:n])

            _kde.dataset = data.T
            _kde.n = len(data)
            if weights is None:
                _kde._weights = np.ones(_kde.n) / _kde.n
            else:
                _kde._weights = weights / nrep  # normalized to 1

            kernel = 'gaussian'

        elif backend == 'sklearn':
            from sklearn.neighbors import KernelDensity

            options.setdefault('rtol', 1e-6)
            bw_normed = bandwidth / kernel_var_dict[kernel]**0.5  # normalize kernels to var = 1

            _kde = KernelDensity(kernel=kernel, bandwidth=bw_normed, **options)
            _kde.fit(data_scaled, sample_weight=weights)

        elif backend == 'KDEpy.FFTKDE' or backend == 'KDEpy.TreeKDE':
            raise NotImplementedError

        ldict = locals()
        for key in ['_kde', 'data', 'weights', 'boundary', 'backend', 'kernel', 'bandwidth',
                    'n', 'd', 'nrep', 'scale', 'pdf_scale', 'data_scaled']:
            if key in ldict:
                setattr(self, key, ldict[key])

    def pdf(self, data, log=False, scaled=False):
        """
        data: shape (n, d)
        log: bool
            If return p or lnp.
        scaled: bool
            Option for autopdf only.
        """
        if self.backend == 'scipy':
            if scaled:
                raise ValueError('input data is not supposed to be scaled')
            p = self.nrep * self._kde.pdf(data.T)

            if log:
                return np.log(p)
            else:
                return p

        elif self.backend == 'sklearn':
            if scaled:
                lnp = self._kde.score_samples(data)
            else:
                lnp = self._kde.score_samples(data / self.scale)

            if log:
                return np.log(self.nrep / self.pdf_scale) + lnp
            else:
                return (self.nrep / self.pdf_scale) * np.exp(lnp)

        elif self.backend == 'KDEpy.FFTKDE' or self.backend == 'KDEpy.TreeKDE':
            raise NotImplementedError

    def autopdf(self, log=False):
        """
        Calculate the pdf of the underlying data points.
        """
        if self.backend == 'scipy':
            return self.pdf(self.data[:self.n], log=log, scaled=False)
        else:
            return self.pdf(self.data_scaled[:self.n], log=log, scaled=True)


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
