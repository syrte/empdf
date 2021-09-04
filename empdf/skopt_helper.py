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

        # Evaluate acquisition function
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

            std = std + 1e-16  # prevent log(0) in ev
            var = std**2

            # log variance of lognormal
            ev = -2 * mu + var + logexpm1(var)  # note lnL=-mu
            ev_grad = -2 * mu_grad + 2 * (1 + expexpm1(var)) * std * std_grad

            return ev, ev_grad

        else:
            mu, std = model.predict(X, return_std=True)

            std = std + 1e-16  # prevent log(0) in ev
            var = std**2

            # log variance of lognormal
            ev = -2 * mu + var + logexpm1(var)  # note lnL=-mu

            return ev


def logexpm1(x):
    "robust log(exp(x)-1)"
    return np.where(x < 37, np.log(np.expm1(x)), x)


def expexpm1(x):
    "robust exp(x)/(exp(x)-1), differential of logexpm1"
    return np.where(x < 37, np.exp(x) / (np.expm1(x)), 1)


skopt.acquisition._gaussian_acquisition = _gaussian_acquisition_wrapper
skopt.optimizer.optimizer._gaussian_acquisition = _gaussian_acquisition_wrapper
skopt.acquisition.gaussian_ev = gaussian_ev
