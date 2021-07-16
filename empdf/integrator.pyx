"""
Author: Zhaozhou Li (lizz.astro@gmail.com)
"""

import numpy as np
from numpy cimport ndarray
from cython_gsl cimport *

from libc.stdlib cimport malloc, free
from libc.math cimport pi, sqrt, isnan, isinf, isfinite, fmax, fmin
from libc.time cimport time

from cython.parallel cimport parallel, prange
from openmp cimport omp_get_thread_num


# cdef:
#     struct Precision_t:
#         double REL_EPS_rlim
#         double ABS_EPS_rlim
#         double REL_EPS_Tr
#         double ABS_EPS_Tr
#     Precision_t eps


Particle_dtype = np.dtype({
    'formats': ['f8'] * 11,
    'names': ['E', 'L2', 'r', 'rmin', 'rmax', 'Tr', 'Tr_cur', 'rmin_obs', 'rmax_obs', 'Tr_obs', 'wgt']
})


cdef packed struct Particle_t:
    double E         # E = 0.5 v^2 + U
    double L2        # L^2
    double r         # current position
    double rmin      # orbit limits
    double rmax      # orbit limits
    double Tr        # rmin->rmax_val
    double Tr_cur    # [opt] rmin->r
    double rmin_obs  # [opt] observable range
    double rmax_obs  # [opt] specified by observation
    double Tr_obs    # [opt] rmin_obs->rmax_obs
    double wgt       # [opt] weight of the orbit, should be 1 by default


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

    if vr2 <= 0.:
        return 0.
    else:
        return 1. / sqrt(vr2)


cdef void solve_radial_limits(Particle_t[:] parr, double rmin, double rmax, gsl_spline * potential) nogil:
    cdef:
        int i, j, n = len(parr)
        Particle_t * p
        double rt, x_lo, x_hi

        int MAX_ITER = 100
        double REL_EPS = 1e-6, ABS_EPS = 0

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
                    if (x_hi - x_lo) <= REL_EPS * x_lo + ABS_EPS:
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
                    if (x_hi - x_lo) <= REL_EPS * x_lo + ABS_EPS:
                        break
                p.rmax = x_lo  # instead of rt to avoid negative vr2
            else:
                p.rmax = rmax

        gsl_root_fsolver_free(solver)
        gsl_interp_accel_free(spl_acc)
        free(func)
        free(orbit)


cdef void compute_radial_period(Particle_t[:] parr, gsl_spline * potential, bint set_tcur, bint set_tobs) nogil:
    cdef:
        int i, j, n = len(parr)
        Particle_t * p
        double rmin, rmax
        double pderiv, pderiv2  # 1st, 2nd derivatives of potential

        int MAX_INTVAL = 1000
        double REL_EPS = 1e-6, ABS_EPS = 0

        gsl_interp_accel * spl_acc
        gsl_integration_cquad_workspace * workspace
        gsl_function * func
        Orbit_t * orbit
        gsl_rng * rng

    with nogil, parallel():
        spl_acc = gsl_interp_accel_alloc()  # workspace for interp lookups
        workspace = gsl_integration_cquad_workspace_alloc(MAX_INTVAL)
        func = <gsl_function *> malloc(sizeof(gsl_function))
        orbit = <Orbit_t *> malloc(sizeof(Orbit_t))
        rng = gsl_rng_alloc(gsl_rng_default)

        func.function = &f_vr_inv
        func.params = orbit
        orbit.potential = potential
        orbit.spl_acc = spl_acc
        gsl_rng_set(rng, omp_get_thread_num() << 16 + time(NULL))

        for i in prange(n):
            p = &parr[i]
            orbit.E = p.E
            orbit.L2 = p.L2

            if p.rmin < p.rmax:
                gsl_integration_cquad(func, p.rmin, p.rmax, ABS_EPS, REL_EPS, workspace, &p.Tr, NULL, NULL)

                if set_tcur:
                    gsl_integration_cquad(func, p.rmin, p.r, ABS_EPS, REL_EPS, workspace, &p.Tr_cur, NULL, NULL)

                if set_tobs:
                    rmin = fmax(p.rmin, p.rmin_obs)
                    rmax = fmin(p.rmax, p.rmax_obs)
                    if rmin < rmax:
                        gsl_integration_cquad(func, rmin, rmax, ABS_EPS, REL_EPS, workspace, &p.Tr_obs, NULL, NULL)
                    else:
                        p.Tr_obs = 0.  # this should never happen!

            else:
                # it seems very close to circular orbit
                pderiv = gsl_spline_eval_deriv(potential, p.r, spl_acc)
                pderiv2 = gsl_spline_eval_deriv2(potential, p.r, spl_acc)
                p.Tr = 2 * pi * sqrt(p.r / (3 * pderiv + p.r * pderiv2))

                if set_tcur:
                    p.Tr_cur = p.Tr * gsl_rng_uniform(rng)  # a random number

                if set_tobs:
                    p.Tr_obs = p.Tr  # the whole circular orbit is observable

        gsl_interp_accel_free(spl_acc)
        gsl_integration_cquad_workspace_free(workspace)
        free(func)
        free(orbit)
        gsl_rng_free(rng)

    # Notes
    # we know Δvr^2 ~ Δr at rlims=[rmin, rmax], which allows more efficient algorithm?
    # e.g. gsl_integration_qagp(func, rlims, 2, ABS_EPS, REL_EPS, ...)


cdef ndarray count_raidal_bin(Particle_t[:] parr, double[:] rbin, gsl_spline * potential):
    cdef:
        int i, j, j0, j1, n = len(parr), nbin = len(rbin) - 1
        Particle_t * p
        double rmin, rmax
        double[:] bincount = np.zeros(nbin, dtype='f8')

        int MAX_INTVAL = 1000
        double REL_EPS = 1e-6, ABS_EPS = 0

        gsl_interp_accel * spl_acc
        gsl_integration_cquad_workspace * workspace
        gsl_function * func
        Orbit_t * orbit
        double * t

    with nogil, parallel():
        spl_acc = gsl_interp_accel_alloc()  # workspace for interp lookups
        workspace = gsl_integration_cquad_workspace_alloc(MAX_INTVAL)
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
            rmax = fmin(p.rmax, rbin[nbin])

            if rmin < rmax:
                j0 = gsl_interp_accel_find(spl_acc, &rbin[0], nbin + 1, rmin)
                j1 = gsl_interp_accel_find(spl_acc, &rbin[0], nbin + 1, rmax)

                for j in range(j0, j1):
                    rmin = fmax(p.rmin, rbin[j])
                    rmax = fmin(p.rmax, rbin[j + 1])
                    if rmin < rmax:
                        gsl_integration_cquad(func, rmin, rmax, ABS_EPS, REL_EPS, workspace, t, NULL, NULL)
                        bincount[j] += p.wgt * t[0] / p.Tr

            elif rmin == rmax:
                # circular orbit
                j = gsl_interp_accel_find(spl_acc, &rbin[0], nbin + 1, rmin)
                bincount[j] += p.wgt

        gsl_interp_accel_free(spl_acc)
        gsl_integration_cquad_workspace_free(workspace)
        free(func)
        free(orbit)
        free(t)

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

    def set_potential(self, double[:] r, double[:] U):
        """
        update potential with radius and potential array
        """
        cdef:
            int n = len(r)
            gsl_spline * potential = gsl_spline_alloc(gsl_interp_cspline, n)
            # Cubic spline with natural boundary conditions

        gsl_spline_init(potential, &r[0], &U[0], n)

        if self.potential != NULL:
            gsl_spline_free(self.potential)
        self.potential = potential

        self.r = r
        self.U = U

    def solve_radial_limits(self):
        """
        solve rmin and rmax and store them in parr
        rmin = max(rmin, rper)
        rmax = min(rmax, rapo)
        """
        solve_radial_limits(self.parr, self.rmin, self.rmax, self.potential)

    def compute_radial_period(self, bint set_tcur=False, bint set_tobs=False):
        """
        set_tcur: calculate the time from rmin to r
        set_tobs: calculate the time from rmin_obs, rmax_obs
        """
        compute_radial_period(self.parr, self.potential, set_tcur=set_tcur, set_tobs=set_tobs)

    def count_raidal_bin(self, double[:] rbin):
        """
        calculate the expected radial bin count
        """
        bincount = count_raidal_bin(self.parr, rbin, self.potential)
        return bincount
