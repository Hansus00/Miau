import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from microjax.likelihood import linear_chi2
from microjax.trajectory import (
    set_parallax_ephem,
    compute_parallax_ephem,
    load_horizons_vectors_file,
    HeliocentricEphemeris,
)
from microlux import binary_mag
from fastlens.mag_fft_jax import fspl_disk

_ephemeris_table = load_horizons_vectors_file("data/Roman_ephemeris_jax.txt")
# _ephemeris_table = load_horizons_vectors_file("data/earth_orbital_parallax_table.txt")
_eph_object = HeliocentricEphemeris.from_horizons_vectors_table(_ephemeris_table)
_fspl_obj = fspl_disk()


def _pspl_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    u = jnp.sqrt(params["u_0"] ** 2 + tau**2)
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))


def _fspl_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    u = jnp.sqrt(params["u_0"] ** 2 + tau**2)
    return _fspl_obj.A(u, params["rho"])


def _parallax_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    t_ephem = t - 2_450_000.0
    t_0_par_ephem = params["t_0_par"] - 2_450_000.0
    proj = set_parallax_ephem(
        tref=t_0_par_ephem,
        RA=params["coords"][0],
        Dec=params["coords"][1],
        eph=_eph_object,
    )
    d_tau, d_beta = compute_parallax_ephem(
        t_ephem, params["pi_E_N"], params["pi_E_E"], proj
    )

    tau += d_tau
    u_0 = params["u_0"] + d_beta
    u = jnp.sqrt(u_0**2 + tau**2)
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))


def _bspl_magnification(t, params):
    tau_1 = (t - params["t_0_1"]) / params["t_E"]
    tau_2 = (t - params["t_0_2"]) / params["t_E"]
    u_1 = jnp.sqrt(params["u_0_1"] ** 2 + tau_1**2)
    u_2 = jnp.sqrt(params["u_0_2"] ** 2 + tau_2**2)
    A_1 = (u_1**2 + 2) / (u_1 * jnp.sqrt(u_1**2 + 4))
    A_2 = (u_2**2 + 2) / (u_2 * jnp.sqrt(u_2**2 + 4))
    return (A_1 + params["q_f"] * A_2) / (1 + params["q_f"])


def _fsbl_magnification_core(t, t_0, u_0, t_E, rho, q, s, alpha_deg):
    return binary_mag(
        t_0,
        u_0,
        t_E,
        rho,
        q,
        s,
        alpha_deg,
        t,
    )


_fsbl_magnification_core = jax.jit(_fsbl_magnification_core, backend="cpu")


def _fsbl_magnification(t, params):
    return _fsbl_magnification_core(
        t,
        params["t_0"],
        params["u_0"],
        params["t_E"],
        params["rho"],
        params["q"],
        params["s"],
        params["alpha_deg"],
    )


def _negative_flux_prior(Fb, sigma=10.0):
    return 0.5 * (jnp.maximum(-Fb, 0.0)) ** 2 / (sigma**2)


_MAGNIFICATION_FUNCS = {
    "pspl": _pspl_magnification,
    "parallax": _parallax_magnification,
    "fspl": _fspl_magnification,
    "bspl": _bspl_magnification,
    "fsbl": _fsbl_magnification,
}


def _prior_parallax(params):
    return 0.5 * (params["pi_E_N"] ** 2 + params["pi_E_E"] ** 2) / (0.15**2)


def _prior_bspl(params):
    prior = 0.5 * (params["q_f"] - 1.0) ** 2 / (10.0**2)
    return prior + 0.5 * (jnp.maximum(-params["q_f"], 0.0)) ** 2 / (3.0**2)


_PRIOR_FUNCS = {
    "parallax": _prior_parallax,
    "bspl": _prior_bspl,
}


def magnification(t, params):
    model = params.get("model")
    if model not in _MAGNIFICATION_FUNCS:
        raise NotImplementedError(f"Magnification model '{model}' is not implemented.")
    return _MAGNIFICATION_FUNCS[model](t, params)


def neg_lnprob(t, params, mag, mag_err):
    A = magnification(t, params)
    Fs, _, Fb, _, chi2 = linear_chi2(A, mag, mag_err)

    prior_fb = _negative_flux_prior(Fb)
    prior_fs = _negative_flux_prior(Fs)
    prior_term = prior_fs + prior_fb

    model = params.get("model")

    prior_fn = _PRIOR_FUNCS.get(model)
    if prior_fn is not None:
        prior_term += prior_fn(params)

    return 0.5 * chi2 + prior_term
