"""
Add exponentiated variance (EV) acquisition function to skopt.

Author:
    Zhaozhou Li (lizz.astro@gmail.com)
"""

import numpy as np
import warnings
import skopt.acquisition
from skopt.acquisition import _gaussian_acquisition as _gaussian_acquisition_original
from skopt import Optimizer, gp_minimize


__all__ = ['Optimizer', 'gp_minimize']


def _gaussian_acquisition_wrapper(
        X, model, y_opt=None, acq_func="LCB",
        return_grad=False, acq_func_kwargs=None):
    """
    Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    if acq_func == 'EV':
        # Check inputs
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X is {}-dimensional, however,"
                             " it must be 2-dimensional.".format(X.ndim))

        func_and_grad = gaussian_ev(X, model, y_opt, return_grad)

        if return_grad:
            return -func_and_grad[0], -func_and_grad[1]  # reverse for minimization
        else:
            return -func_and_grad

    else:
        return _gaussian_acquisition_original(X, model, y_opt, acq_func,
                                              return_grad, acq_func_kwargs)


def gaussian_ev(X, model, y_opt, return_grad=False):
    """
    Use the exponentiated variance [BAPE] to estimate 
    the acquisition values. The exploration is maximized.
    Note that the model describes -lnL as function of X.

    [BAPE] Kandasamy et al. 2015, Bayesian active learning for posterior estimation

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Values where the acquisition function should be computed.
    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.
    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns
    -------
    values : array-like, shape (X.shape[0],)
        Acquisition function values computed at X.
    grad : array-like, shape (n_samples, n_features)
        Gradient at X.
    """
    # Compute posterior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)
            mu = y_opt - mu  # note y=-lnL

            expvar = np.exp(std**2)
            ev = np.exp(2 * mu) * expvar * (expvar - 1)
            ev_grad = 2 * ev * mu_grad + 4 * np.exp(2 * mu) * expvar * (expvar - 0.5) * std * std_grad
            # wolframalpha: D[E^(2 x + y^2) (E^y^2 - 1), y]

            return ev, ev_grad

        else:
            mu, std = model.predict(X, return_std=True)
            mu = y_opt - mu  # note y=-lnL

            expvar = np.exp(std**2)
            ev = np.exp(2 * mu) * expvar * (expvar - 1)

            return ev


skopt.acquisition._gaussian_acquisition = _gaussian_acquisition_wrapper
skopt.optimizer.optimizer._gaussian_acquisition = _gaussian_acquisition_wrapper
skopt.acquisition.gaussian_ev = gaussian_ev
