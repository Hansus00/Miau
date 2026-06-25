from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import os
import jax.numpy as jnp
from microjax.likelihood import linear_chi2

from binary_lens import fsbl_magnification_microjax_jit, soft_box_penalty
from finite_source import fspl_uniform_quadrature_from_tau_beta_jit

# Parallax imports are intentionally lazy. Non-parallax models must not fail at
# import time just because Roman_ephemeris_jax.txt is not present.
#
# Important compatibility note:
#   microjaxx==0.1.1 exposes the public API set_parallax()/compute_parallax().
#   Newer development versions may also expose set_parallax_ephem()/
#   compute_parallax_ephem() plus HeliocentricEphemeris.  The fitting code below
#   supports both.  If the ephemeris API or Roman ephemeris file is unavailable,
#   it falls back to the Keplerian annual-parallax approximation instead of
#   crashing.
_EPH_OBJECT = None
_EPH_LOAD_ERROR = None


def _to_jd_minus_2450000(t):
    """Accept either absolute JD/BJD or JD-2450000 and return JD-2450000."""
    return jnp.where(t > 2_450_000.0, t - 2_450_000.0, t)


def _load_ephemeris():
    """Load Roman/JPL-Horizons ephemeris if the installed microJAX supports it."""
    global _EPH_OBJECT, _EPH_LOAD_ERROR
    if _EPH_OBJECT is not None:
        return _EPH_OBJECT
    if _EPH_LOAD_ERROR is not None:
        raise _EPH_LOAD_ERROR

    try:
        from microjax.trajectory import (
            HeliocentricEphemeris,
            load_horizons_vectors_file,
        )

        table = load_horizons_vectors_file("data/Roman_ephemeris_jax.txt")
        _EPH_OBJECT = HeliocentricEphemeris.from_horizons_vectors_table(table)
        return _EPH_OBJECT
    except Exception as exc:  # pragma: no cover - depends on package version/file
        _EPH_LOAD_ERROR = exc
        raise


def _parallax_offsets(t, params):
    """Return d_tau, d_beta for parallax.

    Preference order:
      1. ephemeris-based parallax, if the installed microJAX version exposes
         set_parallax_ephem()/compute_parallax_ephem() and
         data/Roman_ephemeris_jax.txt is present;
      2. public microjaxx==0.1.1 annual-parallax API:
         set_parallax()/compute_parallax().

    The fallback is still physically meaningful annual parallax, but it is not a
    Roman-spacecraft ephemeris correction.  Use a newer microJAX development
    install plus the Roman Horizons file if you specifically need the spacecraft
    ephemeris version.
    """
    t_rel = _to_jd_minus_2450000(t)
    t_ref = _to_jd_minus_2450000(params["t_0_par"])
    ra = params["coords"][0]
    dec = params["coords"][1]

    # Newer microJAX development API: ephemeris-driven parallax.
    try:
        from microjax.trajectory import set_parallax_ephem, compute_parallax_ephem

        eph_object = _load_ephemeris()
        proj = set_parallax_ephem(tref=t_ref, RA=ra, Dec=dec, eph=eph_object)
        return compute_parallax_ephem(
            t_rel, params["pi_E_N"], params["pi_E_E"], proj
        )
    except Exception:
        # microjaxx==0.1.1 public API: Keplerian annual parallax.
        from microjax.trajectory import set_parallax, compute_parallax

        parallax_params = set_parallax(
            tref=t_ref,
            tperi=0.0,
            tvernal=0.0,
            RA=ra,
            Dec=dec,
        )
        return compute_parallax(
            t_rel, params["pi_E_N"], params["pi_E_E"], parallax_params
        )


def _pspl_from_tau_beta(tau, beta):
    u = jnp.sqrt(beta**2 + tau**2)
    return (u**2 + 2.0) / (u * jnp.sqrt(u**2 + 4.0))


def _pspl_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    return _pspl_from_tau_beta(tau, params["u_0"])


def _parallax_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    d_tau, d_beta = _parallax_offsets(t, params)
    return _pspl_from_tau_beta(tau + d_tau, params["u_0"] + d_beta)


def _fspl_magnification(t, params):
    """Finite-source point-lens magnification with a uniform source."""
    tau = (t - params["t_0"]) / params["t_E"]
    return fspl_uniform_quadrature_from_tau_beta_jit(tau, params["u_0"], params["rho"])


def _fspl_parallax_magnification(t, params):
    """Finite-source point-lens magnification with parallax."""
    tau = (t - params["t_0"]) / params["t_E"]
    d_tau, d_beta = _parallax_offsets(t, params)
    return fspl_uniform_quadrature_from_tau_beta_jit(
        tau + d_tau, params["u_0"] + d_beta, params["rho"]
    )


def _bspl_magnification(t, params):
    tau_1 = (t - params["t_0_1"]) / params["t_E"]
    tau_2 = (t - params["t_0_2"]) / params["t_E"]
    A_1 = _pspl_from_tau_beta(tau_1, params["u_0_1"])
    A_2 = _pspl_from_tau_beta(tau_2, params["u_0_2"])
    return (A_1 + params["q_f"] * A_2) / (1.0 + params["q_f"])


def _bspl_parallax_magnification(t, params):
    """
    Binary-source point-lens model with parallax.

    The same observer-induced displacement is applied to both source components.
    This is the natural 1L2S extension when both sources share the same lens and
    the same relative proper-motion frame.
    """
    d_tau, d_beta = _parallax_offsets(t, params)
    tau_1 = (t - params["t_0_1"]) / params["t_E"] + d_tau
    tau_2 = (t - params["t_0_2"]) / params["t_E"] + d_tau
    A_1 = _pspl_from_tau_beta(tau_1, params["u_0_1"] + d_beta)
    A_2 = _pspl_from_tau_beta(tau_2, params["u_0_2"] + d_beta)
    return (A_1 + params["q_f"] * A_2) / (1.0 + params["q_f"])


def _fsbl_magnification(t, params):
    """Finite-source binary-lens magnification with microJAX."""
    return fsbl_magnification_microjax_jit(
        t,
        params["t_0"],
        params["u_0"],
        params["t_E"],
        params["rho"],
        params["s"],
        params["q"],
        params["alpha_deg"],
    )


def _fsbl_parallax_magnification(t, params):
    """
    Finite-source binary-lens model with parallax.

    We compute the standard parallax offsets d_tau and d_beta, then pass them
    into the same binary-lens trajectory builder used by FSBL.  Inside
    ``binary_lens.rectilinear_trajectory`` the perturbed coordinates are rotated
    into the binary-lens frame by alpha_deg.
    """
    d_tau, d_beta = _parallax_offsets(t, params)
    return fsbl_magnification_microjax_jit(
        t,
        params["t_0"],
        params["u_0"],
        params["t_E"],
        params["rho"],
        params["s"],
        params["q"],
        params["alpha_deg"],
        d_tau,
        d_beta,
    )


_MAGNIFICATION_FUNCS = {
    "pspl": _pspl_magnification,
    "parallax": _parallax_magnification,
    "fspl": _fspl_magnification,
    "fspl_parallax": _fspl_parallax_magnification,
    "bspl": _bspl_magnification,
    "bspl_parallax": _bspl_parallax_magnification,
    "fsbl": _fsbl_magnification,
    "fsbl_parallax": _fsbl_parallax_magnification,
}


def _negative_flux_prior(F, sigma=10.0):
    return 0.5 * (jnp.maximum(-F, 0.0) / sigma) ** 2


def _prior_parallax(params):
    return 0.5 * (params["pi_E_N"] ** 2 + params["pi_E_E"] ** 2) / (0.15**2)


def _prior_bspl(params):
    prior = 0.5 * (params["q_f"] - 1.0) ** 2 / (10.0**2)
    return prior + 0.5 * (jnp.maximum(-params["q_f"], 0.0) / 3.0) ** 2


def _prior_fspl(params):
    """Weak guardrails for finite-source point-lens fits."""
    log_rho = jnp.log(params["rho"])
    log_tE = jnp.log(params["t_E"])
    return (
        soft_box_penalty(log_rho, jnp.log(1.0e-6), jnp.log(1.0), 0.35)
        + soft_box_penalty(log_tE, jnp.log(0.005), jnp.log(2000.0), 0.5)
        + soft_box_penalty(jnp.abs(params["u_0"]), 0.0, 5.0, 0.5)
    )


def _prior_fspl_parallax(params):
    return _prior_fspl(params) + _prior_parallax(params)


def _prior_fsbl(params):
    """
    Weak numerical guardrails for blind FSBL optimization.

    These are deliberately broad; they do not encode a strong astrophysical
    population prior. They mainly keep gradient descent away from regions where
    the finite-source binary-lens call is uninformative or extremely slow.
    """
    log_s = jnp.log(params["s"])
    log_q = jnp.log(params["q"])
    log_rho = jnp.log(params["rho"])
    log_tE = jnp.log(params["t_E"])

    rho_max = float(os.environ.get("FSBL_RHO_MAX", "0.05"))

    return (
        soft_box_penalty(log_s, jnp.log(0.05), jnp.log(20.0), 0.25)
        + soft_box_penalty(log_q, jnp.log(1.0e-6), jnp.log(1.0), 0.35)
        + soft_box_penalty(log_rho, jnp.log(1.0e-5), jnp.log(rho_max), 0.10)
        + soft_box_penalty(log_tE, jnp.log(0.05), jnp.log(2000.0), 0.5)
        + soft_box_penalty(jnp.abs(params["u_0"]), 0.0, 5.0, 0.5)
    )


def _prior_bspl_parallax(params):
    return _prior_bspl(params) + _prior_parallax(params)


def _prior_fsbl_parallax(params):
    return _prior_fsbl(params) + _prior_parallax(params)


_PRIOR_FUNCS = {
    "parallax": _prior_parallax,
    "fspl": _prior_fspl,
    "fspl_parallax": _prior_fspl_parallax,
    "bspl": _prior_bspl,
    "bspl_parallax": _prior_bspl_parallax,
    "fsbl": _prior_fsbl,
    "fsbl_parallax": _prior_fsbl_parallax,
}


def magnification(t, params):
    model = params.get("model")
    if model not in _MAGNIFICATION_FUNCS:
        raise NotImplementedError(f"Magnification model '{model}' is not implemented.")
    return _MAGNIFICATION_FUNCS[model](t, params)


def neg_lnprob(t, params, flux, flux_err):
    A = magnification(t, params)
    Fs, _, Fb, _, chi2 = linear_chi2(A, flux, flux_err)

    prior_term = _negative_flux_prior(Fs) + _negative_flux_prior(Fb)

    prior_fn = _PRIOR_FUNCS.get(params.get("model"))
    if prior_fn is not None:
        prior_term = prior_term + prior_fn(params)

    return 0.5 * chi2 + prior_term
