"""
Finite-source point-lens helpers.

This module implements a small differentiable uniform-source quadrature for
single-lens finite-source magnification.  It is intentionally independent of
external microlensing packages so that FSPL/FSPL+Parallax remain cheap and JAX
compatible.

The source disk is averaged in polar coordinates with equal-area radial rings:

    A_FSPL(u, rho) = < A_PSPL(|u_vec + r_vec|) >_disk

For rho -> 0 the function smoothly falls back to PSPL.
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# Defaults are intentionally moderate.  Increase for final single-event checks.
FSPL_N_R = max(int(os.environ.get("FSPL_N_R", "4")), 1)
FSPL_N_THETA = max(int(os.environ.get("FSPL_N_THETA", "12")), 4)


def pspl_from_u(u):
    """Point-source point-lens magnification as a function of separation u."""
    u_safe = jnp.maximum(u, 1.0e-12)
    return (u_safe**2 + 2.0) / (u_safe * jnp.sqrt(u_safe**2 + 4.0))


def fspl_uniform_quadrature_from_tau_beta(
    tau,
    beta,
    rho,
    *,
    n_r: int = FSPL_N_R,
    n_theta: int = FSPL_N_THETA,
):
    """
    Uniform finite-source point-lens magnification.

    Parameters
    ----------
    tau, beta : array-like
        Rectilinear coordinates of the source center in Einstein-radius units.
        For no parallax, beta is just u0.  For parallax, beta can be an array.
    rho : float
        Source angular radius normalized by theta_E.
    n_r, n_theta : int
        Equal-area radial rings and azimuthal samples per ring.

    Notes
    -----
    This is a numerical disk average, not the analytic elliptic-integral
    formula.  It is robust, differentiable and fast enough for FSPL screening.
    For publication-quality final values, increase FSPL_N_R/FSPL_N_THETA or
    compare selected events with MulensModel/VB/pyLIMA.
    """
    tau = jnp.asarray(tau)
    beta = jnp.asarray(beta)
    rho = jnp.asarray(rho)

    u_center = jnp.sqrt(tau**2 + beta**2)
    A_point = pspl_from_u(u_center)

    # If rho is effectively zero, avoid unnecessary quadrature and singular
    # behavior.  The branch is JAX-traceable because both paths are defined.
    rho_safe = jnp.maximum(rho, 1.0e-10)

    # Unit vectors parallel and perpendicular to the center-source vector.
    # For u=0 choose an arbitrary orientation; the disk average is rotationally
    # symmetric, so this choice does not affect the result.
    ux = jnp.where(u_center > 1.0e-12, tau / jnp.maximum(u_center, 1.0e-12), 1.0)
    uy = jnp.where(u_center > 1.0e-12, beta / jnp.maximum(u_center, 1.0e-12), 0.0)
    px = -uy
    py = ux

    # Equal-area annuli: r_i/rho = sqrt((i + 1/2)/n_r).
    rr = jnp.sqrt((jnp.arange(n_r, dtype=tau.dtype) + 0.5) / float(n_r))
    theta = 2.0 * jnp.pi * (jnp.arange(n_theta, dtype=tau.dtype) + 0.5) / float(n_theta)

    # Shape: (..., n_r, n_theta)
    dr_parallel = rho_safe * rr[:, None] * jnp.cos(theta)[None, :]
    dr_perp = rho_safe * rr[:, None] * jnp.sin(theta)[None, :]

    x = u_center[..., None, None] + dr_parallel
    y = dr_perp
    # Because we aligned the coordinate system with the source-center vector,
    # the separation from the lens is simply sqrt(x^2 + y^2).
    u_samples = jnp.sqrt(x**2 + y**2)
    A_fs = jnp.mean(pspl_from_u(u_samples), axis=(-2, -1))

    return jnp.where(rho > 1.0e-9, A_fs, A_point)


fspl_uniform_quadrature_from_tau_beta_jit = jax.jit(
    fspl_uniform_quadrature_from_tau_beta,
    static_argnames=("n_r", "n_theta"),
)
