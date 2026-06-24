"""
Binary-lens helpers for the Roman microlensing fitting pipeline.

This module intentionally uses the public microJAX API directly:

    from microjax.inverse_ray.lightcurve import mag_binary

microJAX is installed from PyPI as ``microjaxx`` but imported as ``microjax``.
The finite-source binary-lens function expects complex source positions
``w = x + i y`` in Einstein-radius units, source radius ``rho``, projected
separation ``s`` and mass ratio ``q = m2/m1``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

try:
    from microjax.inverse_ray.lightcurve import mag_binary as _microjax_mag_binary
except Exception as exc:  # pragma: no cover - runtime dependency check
    _microjax_mag_binary = None
    _MICROJAX_IMPORT_ERROR = exc
else:
    _MICROJAX_IMPORT_ERROR = None


def require_microjax():
    """Raise a clear error if the microJAX finite-source backend is unavailable."""
    if _microjax_mag_binary is None:
        raise ImportError(
            "Finite-source binary-lens modelling requires microJAX. "
            "Install it with `pip install microjaxx`. The PyPI package is "
            "called `microjaxx`, but the import name is `microjax`. "
            f"Original import error: {_MICROJAX_IMPORT_ERROR!r}"
        )
    return _microjax_mag_binary


def rectilinear_trajectory(t, t_0, u_0, t_E, alpha_deg, d_tau=0.0, d_beta=0.0):
    """
    Return complex source positions for a standard binary-lens trajectory.

    Without parallax this follows the convention used in the microJAX docs:

        tau = (t - t0)/tE
        x = -u0 sin(alpha) + tau cos(alpha)
        y =  u0 cos(alpha) + tau sin(alpha)
        w = x + i y

    With parallax, ``d_tau`` and ``d_beta`` are the usual geocentric/satellite
    trajectory perturbations parallel and perpendicular to the rectilinear
    motion.  They are added before the rotation into the binary-lens frame:

        tau  -> tau + d_tau
        beta -> u0  + d_beta

    This is the same convention as the PSPL parallax implementation in
    ``magnification_model.py``.
    """
    alpha = jnp.deg2rad(alpha_deg)
    tau = (t - t_0) / t_E + d_tau
    beta = u_0 + d_beta
    x = -beta * jnp.sin(alpha) + tau * jnp.cos(alpha)
    y = beta * jnp.cos(alpha) + tau * jnp.sin(alpha)
    return jnp.asarray(x + 1j * y, dtype=jnp.complex128)


def fsbl_magnification_microjax(
    t,
    t_0,
    u_0,
    t_E,
    rho,
    s,
    q,
    alpha_deg,
    d_tau=0.0,
    d_beta=0.0,
    *,
    r_resolution=12,
    th_resolution=24,
    MAX_FULL_CALLS=128,
    chunk_size=256,
    Nlimb=20,
):
    """
    Finite-source binary-lens magnification using microJAX.

    ``d_tau`` and ``d_beta`` are optional parallax perturbations.  For the
    non-parallax FSBL model they are zero scalars; for FSBL+Parallax they are
    arrays with the same shape as ``t``.

    The keyword arguments are passed through to microJAX when supported by the
    installed version. If an older microJAX version does not accept some of
    these tuning parameters, we fall back to the minimal public call.
    """
    mag_binary = require_microjax()
    w_points = rectilinear_trajectory(
        t, t_0, u_0, t_E, alpha_deg, d_tau=d_tau, d_beta=d_beta
    )

    # microJAX 0.1.x accepts these accuracy/runtime knobs in current docs.
    # The fallback keeps the code usable if the local installed version has a
    # smaller signature.
    try:
        return mag_binary(
            w_points,
            rho,
            s=s,
            q=q,
            r_resolution=r_resolution,
            th_resolution=th_resolution,
            MAX_FULL_CALLS=MAX_FULL_CALLS,
            chunk_size=chunk_size,
            Nlimb=Nlimb,
        )
    except TypeError:
        return mag_binary(w_points, rho, s=s, q=q)


# JIT the thin wrapper. Keep static accuracy knobs outside the traced argument
# list by using default constants above.
fsbl_magnification_microjax_jit = jax.jit(fsbl_magnification_microjax)


def wrap_angle_deg(alpha_deg):
    """Map any angle to the interval [0, 360). Useful for readable output."""
    return jnp.mod(alpha_deg, 360.0)


def soft_box_penalty(x, lo, hi, sigma):
    """
    Smooth quadratic penalty outside [lo, hi].

    This is not a physical prior; it only prevents Adam from walking into
    numerically absurd binary-lens regions where the finite-source backend can
    become painfully slow or meaningless for a blind search.
    """
    below = jnp.maximum(lo - x, 0.0)
    above = jnp.maximum(x - hi, 0.0)
    return 0.5 * (below / sigma) ** 2 + 0.5 * (above / sigma) ** 2
