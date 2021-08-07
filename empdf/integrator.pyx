"""
Author: Zhaozhou Li (lizz.astro@gmail.com)
"""

import numpy as np
from numpy cimport ndarray
from cython_gsl cimport *

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from libc.math cimport pi, sqrt, isnan, isinf, isfinite, fmax, fmin
from libc.time cimport time

from cython.parallel cimport parallel, prange
cimport openmp as omp


cdef:
    int MAX_ITER = 100
    int MAX_INTV = 1000
    double TOL_CIRC = 1 - 1e-3  # treat rmin/rmax > TOL_CIRC as circular orbit

Particle_dtype = np.dtype({
    'formats': ['f8'] * 11,
    'names': ['E', 'L2', 'r', 'rmin', 'rmax', 'Tr', 'Tr_cur', 'rmin_obs', 'rmax_obs', 'Tr_obs']
})


cdef packed struct Particle_t:
    double E         # E = 0.5 v^2 + U
    double L2        # L^2
    double r         # current position
    double rmin      # orbit limits
    double rmax      # orbit limits
    double Tr        # rmin->rmax_val, half radial period
    double Tr_cur    # [opt] rmin->r
    double rmin_obs  # [opt] observable range
    double rmax_obs  # [opt] specified by observation
    double Tr_obs    # [opt] rmin_obs->rmax_obs


cdef packed struct Orbit_t:
    double E
    double L2
    gsl_spline * potential
    gsl_interp_accel * spl_acc  # workspace for interpolation lookup


cdef double f_vr2(double r, void * params) nogil:
    "vr^2"
    cdef:
        Orbit_t * orbit = <Orbit_t *> params
        double vr2 = 2 * (orbit.E - gsl_spline_eval(orbit.potential, r, orbit.spl_acc)) - orbit.L2 / (r * r)

    return vr2


cdef double f_vr_inv(double r, void * params) nogil:
    "1/|vr|"
    cdef:
        Orbit_t * orbit = <Orbit_t *> params
        double vr2 = 2 * (orbit.E - gsl_spline_eval(orbit.potential, r, orbit.spl_acc)) - orbit.L2 / (r * r)

    if vr2 > 0.:
        return 1. / sqrt(vr2)
    else:
        return 0.


cdef void solve_radial_limits(Particle_t[:] parr, gsl_spline * potential, double rmin, double rmax,
                              double epsrel=1e-6, double epsabs=0.) nogil:
    cdef:
        int i, j, n = len(parr)
        Particle_t * p
        double rt, x_lo, x_hi

        gsl_interp_accel * spl_acc
        gsl_root_fsolver * solver
        gsl_function * func
        Orbit_t * orbit

    with nogil, parallel():
        spl_acc = gsl_interp_accel_alloc()  # workspace for interpolation lookup
        solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent)
        func = <gsl_function *> malloc(sizeof(gsl_function))
        orbit = <Orbit_t *> malloc(sizeof(Orbit_t))

        func.function = &f_vr2
        func.params = orbit
        orbit.potential = potential
        orbit.spl_acc = spl_acc

        for i in prange(n):
            p = &parr[i]
            orbit.E = p.E
            orbit.L2 = p.L2

            if f_vr2(p.r, orbit) <= 0:
                p.rmin = p.r
                p.rmax = p.r
                continue

            if f_vr2(rmin, orbit) < 0:
                gsl_root_fsolver_set(solver, func, rmin, p.r)
                for j in range(MAX_ITER):
                    gsl_root_fsolver_iterate(solver)
                    rt = gsl_root_fsolver_root(solver)
                    x_lo = gsl_root_fsolver_x_lower(solver)
                    x_hi = gsl_root_fsolver_x_upper(solver)
                    if (x_hi - x_lo) <= epsrel * x_hi + epsabs:
                        break
                p.rmin = x_hi  # instead of rt to avoid negative vr2
            else:
                p.rmin = rmin

            if f_vr2(rmax, orbit) < 0:
                gsl_root_fsolver_set(solver, func, p.r, rmax)
                for j in range(MAX_ITER):
                    gsl_root_fsolver_iterate(solver)
                    rt = gsl_root_fsolver_root(solver)
                    x_lo = gsl_root_fsolver_x_lower(solver)
                    x_hi = gsl_root_fsolver_x_upper(solver)
                    if (x_hi - x_lo) <= epsrel * x_lo + epsabs:
                        break
                p.rmax = x_lo  # instead of rt to avoid negative vr2
            else:
                p.rmax = rmax

        gsl_root_fsolver_free(solver)
        gsl_interp_accel_free(spl_acc)
        free(func)
        free(orbit)


cdef void compute_radial_period(Particle_t[:] parr, gsl_spline * potential,
                                bint set_t, bint set_tcur, bint set_tobs,
                                double epsrel=1e-6, double epsabs=0.) nogil:
    cdef:
        int i, j, n = len(parr)
        Particle_t * p
        double rmin, rmax, t
        double pderiv, pderiv2  # 1st, 2nd derivatives of potential

        gsl_interp_accel * spl_acc
        gsl_integration_cquad_workspace * workspace
        gsl_function * func
        Orbit_t * orbit
        gsl_rng * rng

    with nogil, parallel():
        spl_acc = gsl_interp_accel_alloc()  # workspace for interp lookups
        workspace = gsl_integration_cquad_workspace_alloc(MAX_INTV)
        func = <gsl_function *> malloc(sizeof(gsl_function))
        orbit = <Orbit_t *> malloc(sizeof(Orbit_t))
        rng = gsl_rng_alloc(gsl_rng_default)

        func.function = &f_vr_inv
        func.params = orbit
        orbit.potential = potential
        orbit.spl_acc = spl_acc
        gsl_rng_set(rng, omp.omp_get_thread_num() << 16 + time(NULL))

        for i in prange(n):
            p = &parr[i]
            orbit.E = p.E
            orbit.L2 = p.L2

            if p.rmin < p.rmax * TOL_CIRC:
                if set_t:
                    gsl_integration_cquad(func, p.rmin, p.rmax, epsabs, epsrel, workspace, &p.Tr, NULL, NULL)
                if set_tcur:
                    gsl_integration_cquad(func, p.rmin, p.r, epsabs, epsrel, workspace, &p.Tr_cur, NULL, NULL)
                if set_tobs:
                    rmin = fmax(p.rmin, p.rmin_obs)
                    rmax = fmin(p.rmax, p.rmax_obs)
                    if rmin < rmax:
                        gsl_integration_cquad(func, rmin, rmax, epsabs, epsrel, workspace, &p.Tr_obs, NULL, NULL)
                    else:
                        p.Tr_obs = 0.  # this should never happen! no overlap between rlim and rlim_obs

            else:
                # it seems very close to circular orbit
                pderiv = gsl_spline_eval_deriv(potential, p.r, spl_acc)
                pderiv2 = gsl_spline_eval_deriv2(potential, p.r, spl_acc)
                t = pi * sqrt(p.r / (3 * pderiv + p.r * pderiv2))  # half radial period!

                if set_t:
                    p.Tr = t
                if set_tcur:
                    p.Tr_cur = t * gsl_rng_uniform(rng)  # a random number
                if set_tobs:
                    p.Tr_obs = t  # the whole circular orbit is observable

        gsl_interp_accel_free(spl_acc)
        gsl_integration_cquad_workspace_free(workspace)
        free(func)
        free(orbit)
        gsl_rng_free(rng)

    # Notes
    # we know Δvr^2 ~ Δr at rlims=[rmin, rmax], which allows more efficient algorithm?
    # e.g. gsl_integration_qagp(func, rlims, 2, epsabs, epsrel, ...)


cdef ndarray count_raidal_bin(Particle_t[:] parr, gsl_spline * potential, double[:] rbin,
                              double epsrel=1e-6, double epsabs=0.):
    cdef:
        int i, j, j0, j1, n = len(parr), nbin = len(rbin)
        Particle_t * p
        double rmin, rmax
        double[:] bincount = np.zeros(nbin - 1, dtype='f8')

        gsl_interp_accel * bin_acc
        gsl_interp_accel * spl_acc
        gsl_integration_cquad_workspace * workspace
        gsl_function * func
        Orbit_t * orbit
        double * t

    cdef omp.omp_lock_t lock
    omp.omp_init_lock(&lock)

    with nogil, parallel():
        bin_acc = gsl_interp_accel_alloc()  # workspace for binary search
        spl_acc = gsl_interp_accel_alloc()  # workspace for interp lookups
        workspace = gsl_integration_cquad_workspace_alloc(MAX_INTV)
        func = <gsl_function *> malloc(sizeof(gsl_function))
        orbit = <Orbit_t *> malloc(sizeof(Orbit_t))
        t = <double *> malloc(sizeof(double))

        func.function = &f_vr_inv
        func.params = orbit
        orbit.potential = potential
        orbit.spl_acc = spl_acc

        for i in prange(n):
            p = &parr[i]
            orbit.E = p.E
            orbit.L2 = p.L2

            rmin = fmax(p.rmin, rbin[0])
            rmax = fmin(p.rmax, rbin[nbin - 1])

            if rmin < rmax * TOL_CIRC:
                j0 = gsl_interp_accel_find(bin_acc, &rbin[0], nbin, rmin)
                j1 = gsl_interp_accel_find(bin_acc, &rbin[0], nbin, rmax)  # rbin[j1+1] is secured

                for j in range(j0, j1 + 1):
                    rmin = fmax(p.rmin, rbin[j])
                    rmax = fmin(p.rmax, rbin[j + 1])
                    if rmin < rmax:
                        gsl_integration_cquad(func, rmin, rmax, epsabs, epsrel, workspace, t, NULL, NULL)

                        omp.omp_set_lock(&lock)
                        bincount[j] += t[0] / p.Tr
                        omp.omp_unset_lock(&lock)

            elif rmin <= rmax:
                # circular orbit
                j = gsl_interp_accel_find(bin_acc, &rbin[0], nbin, rmin)

                omp.omp_set_lock(&lock)
                bincount[j] += 1
                omp.omp_unset_lock(&lock)

            else:
                pass  # this may happen! no overlap between rlim and rbin

        gsl_interp_accel_free(bin_acc)
        gsl_interp_accel_free(spl_acc)
        gsl_integration_cquad_workspace_free(workspace)
        free(func)
        free(orbit)
        free(t)

    omp.omp_destroy_lock(&lock)

    return bincount.base


cdef class Integrator:
    cdef:
        public Particle_t[:] parr  # particle data array
        public double rmin
        public double rmax
        public double[:] r
        public double[:] U
        gsl_spline * potential

    def __cinit__(self):
        self.potential = NULL

    def __dealloc__(self):
        if self.potential != NULL:
            gsl_spline_free(self.potential)

    def set_data(self, Particle_t[:] parr, double rmin, double rmax):
        self.parr = parr
        self.rmin = rmin
        self.rmax = rmax

    def set_potential(self, double[:] r, double[:] U, double[:, ::1] c=None):
        """
        update potential with radius and potential array
        c = scipy.interpolate.CubicSpline(r, U).c[1]
        """
        cdef:
            gsl_spline * potential

        assert r.size == U.size
        if c is not None:
            assert c.shape[0] == 4 and c.shape[1] == r.size - 1

        potential = cspline_sp2gsl(r, U, c)

        if self.potential != NULL:
            gsl_spline_free(self.potential)
        self.potential = potential

        self.r = r
        self.U = U

    def solve_radial_limits(self, double epsrel=1e-6, double epsabs=0.):
        """
        solve rmin and rmax and store them in parr
        rmin = max(rmin, rper)
        rmax = min(rmax, rapo)
        """
        solve_radial_limits(self.parr, self.potential, self.rmin, self.rmax, epsrel, epsabs)

    def compute_radial_period(self, bint set_t=True, bint set_tcur=False, bint set_tobs=False, 
                              double epsrel=1e-6, double epsabs=0.):
        """
        set_tcur: calculate the time from rmin to r
        set_tobs: calculate the time from rmin_obs, rmax_obs
        """
        compute_radial_period(self.parr, self.potential, set_t, set_tcur, set_tobs, epsrel, epsabs)

    def count_raidal_bin(self, double[:] rbin, double epsrel=1e-6, double epsabs=0.):
        """
        calculate the time-average radial bin count
        """
        bincount = count_raidal_bin(self.parr, self.potential, rbin, epsrel, epsabs)
        return bincount


# -----------------
cdef struct gsl_spline_t:
    gsl_interp_t *interp
    double *x
    double *y
    size_t  size

cdef struct gsl_interp_t:
    const gsl_interp_type *type
    double  xmin
    double  xmax
    size_t  size
    cspline_state_t *state

cdef struct cspline_state_t:
    double *c
    double *g
    double *diag
    double *offdiag


cdef gsl_spline *cspline_sp2gsl(double[:] x, double[:] y, double[:, ::1] c=None) nogil:
    '''
    c = scipy.interpolate.CubicSpline(x, y).c
    '''
    cdef:
        int n = len(x)
        gsl_spline *spline
        gsl_spline_t *spl

    with gil:
        assert (y.shape[0] == n)
        if c is not None:
            assert (c.shape[0] == 4) and (c.shape[1] == n - 1)

    spline = gsl_spline_alloc(gsl_interp_cspline, n)

    if c is not None:
        spl = <gsl_spline_t *> spline
        # need this conversion because cython_gsl does not provide complete struct member info

        spl.interp.xmin = x[0]
        spl.interp.xmax = x[n - 1]
        memcpy(spl.x, &x[0], n * sizeof(double))
        memcpy(spl.y, &y[0], n * sizeof(double))

        memcpy(spl.interp.state.c, &c[1, 0], (n - 1) * sizeof(double))
        spl.interp.state.c[n - 1] = c[1, n - 2] + 3 * c[0, n - 2] * (x[n - 1] - x[n - 2])

    else:
        gsl_spline_init(spline, &x[0], &y[0], n)

    return spline