import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from microjax.likelihood import linear_chi2
from microlux import binary_mag
from fastlens.mag_fft_jax import fspl_disk

try:
    from microjax.trajectory import (
        set_parallax_ephem,
        compute_parallax_ephem,
        load_horizons_vectors_file,
        HeliocentricEphemeris,
    )
except Exception:  # microjaxx==0.1.1 compatibility fallback
    set_parallax_ephem = None
    compute_parallax_ephem = None
    load_horizons_vectors_file = None
    HeliocentricEphemeris = None

_ephemeris_table = None
_eph_object = None
_fspl_obj = fspl_disk(
    N_fft=int(os.environ.get("FASTLENS_N_FFT", "1024")),
    rho_switch=float(os.environ.get("FASTLENS_RHO_SWITCH", "1e-4")),
)
_twinkle_obj = None
_twinkle_n_srcs = 0


def ensure_ephemeris_loaded():
    """Load ephemeris before any JAX trace (must not run inside jit/vmap/grad)."""
    global _ephemeris_table, _eph_object
    if _eph_object is not None:
        return _eph_object
    if load_horizons_vectors_file is None or HeliocentricEphemeris is None:
        raise ImportError(
            "This installed microjax version does not expose ephemeris parallax API. "
            "Install a compatible microjax or skip parallax models."
        )
    eph_file = os.environ.get("ROMAN_EPHEMERIS_FILE", "data/Roman_ephemeris_jax.txt")
    _ephemeris_table = load_horizons_vectors_file(eph_file)
    _eph_object = HeliocentricEphemeris.from_horizons_vectors_table(_ephemeris_table)
    return _eph_object


def _pspl_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    u = jnp.sqrt(params["u_0"] ** 2 + tau**2)
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))


def _fspl_magnification(t, params):
    """Finite-source point-lens magnification from local fastlens FFTLog code."""
    tau = (t - params["t_0"]) / params["t_E"]
    u = jnp.sqrt(params["u_0"] ** 2 + tau**2)
    return _fspl_obj.A(u, params["rho"])


def _parallax_offsets(t, params):
    if set_parallax_ephem is None or compute_parallax_ephem is None:
        raise ImportError(
            "Parallax ephemeris functions are missing in this microjax installation."
        )
    t_ephem = t - 2_450_000.0
    t_0_par_ephem = params["t_0_par"] - 2_450_000.0
    proj = set_parallax_ephem(
        tref=t_0_par_ephem,
        RA=params["coords"][0],
        Dec=params["coords"][1],
        eph=_eph_object,
    )
    return compute_parallax_ephem(t_ephem, params["pi_E_N"], params["pi_E_E"], proj)


def _parallax_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    d_tau, d_beta = _parallax_offsets(t, params)

    tau = tau + d_tau
    u_0 = params["u_0"] + d_beta
    u = jnp.sqrt(u_0**2 + tau**2)
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))


def _fspl_parallax_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    d_tau, d_beta = _parallax_offsets(t, params)
    tau = tau + d_tau
    u_0 = params["u_0"] + d_beta
    u = jnp.sqrt(u_0**2 + tau**2)
    return _fspl_obj.A(u, params["rho"])


def _point_lens_from_tau_beta(tau, beta):
    u = jnp.sqrt(beta**2 + tau**2)
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))


def _bspl_magnification(t, params):
    tau_1 = (t - params["t_0_1"]) / params["t_E"]
    tau_2 = (t - params["t_0_2"]) / params["t_E"]
    A_1 = _point_lens_from_tau_beta(tau_1, params["u_0_1"])
    A_2 = _point_lens_from_tau_beta(tau_2, params["u_0_2"])
    return (A_1 + params["q_f"] * A_2) / (1 + params["q_f"])


def _bspl_parallax_magnification(t, params):
    """Binary-source point-lens model with Roman ephemeris parallax.

    The same parallax displacement is applied to both source trajectories,
    because the lens-observer geometry is common to the two source components.
    """
    d_tau, d_beta = _parallax_offsets(t, params)
    tau_1 = (t - params["t_0_1"]) / params["t_E"] + d_tau
    tau_2 = (t - params["t_0_2"]) / params["t_E"] + d_tau
    A_1 = _point_lens_from_tau_beta(tau_1, params["u_0_1"] + d_beta)
    A_2 = _point_lens_from_tau_beta(tau_2, params["u_0_2"] + d_beta)
    return (A_1 + params["q_f"] * A_2) / (1 + params["q_f"])


def _fsbl_magnification_core(t, t_0, u_0, t_E, rho, q, s, alpha_deg):
    return binary_mag(t_0, u_0, t_E, rho, q, s, alpha_deg, t)


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


def _create_twinkle_obj(n_srcs, device_num=0, n_stream=1, RelTol=1e-4):
    """Legacy helper for FSBLGrid; real Twinkle grid lives in twinkle_grid_search.py."""
    global _twinkle_obj, _twinkle_n_srcs
    import importlib

    twinkle = importlib.import_module("twinkle")
    if not hasattr(twinkle, "Twinkle"):
        raise AttributeError(
            "Imported module 'twinkle' has no Twinkle class. Set TWINKLE_PYTHON_DIR "
            "to the compiled AsterLight0626/Twinkle/python directory."
        )
    _twinkle_obj = twinkle.Twinkle(n_srcs, device_num, n_stream, RelTol)
    _twinkle_n_srcs = n_srcs


def _fsbl_grid_magnification(t, params):
    tau = (t - params["t_0"]) / params["t_E"]
    n_srcs = len(t)
    u_vec = params["u_0"] * np.ones(n_srcs)
    sin_alpha = np.sin(np.radians(params["alpha_deg"]))
    cos_alpha = np.cos(np.radians(params["alpha_deg"]))
    x = tau * cos_alpha - u_vec * sin_alpha
    y = tau * sin_alpha + u_vec * cos_alpha

    if _twinkle_obj is None or _twinkle_n_srcs != n_srcs:
        _create_twinkle_obj(n_srcs)

    mag = np.empty(n_srcs)
    _twinkle_obj.set_params(params["s"], params["q"], params["rho"], x, y)
    _twinkle_obj.run()
    _twinkle_obj.return_mag_to(mag)
    return mag


_MAGNIFICATION_FUNCS = {
    "pspl": _pspl_magnification,
    "parallax": _parallax_magnification,
    "fspl": _fspl_magnification,
    "fspl_parallax": _fspl_parallax_magnification,
    "bspl": _bspl_magnification,
    "bspl_parallax": _bspl_parallax_magnification,
    "fsbl": _fsbl_magnification,
    "fsbl_grid": _fsbl_grid_magnification,
}


def _negative_flux_prior(Fb, sigma=10.0):
    return 0.5 * (jnp.maximum(-Fb, 0.0)) ** 2 / (sigma**2)


def _prior_parallax(params):
    return 0.5 * (params["pi_E_N"] ** 2 + params["pi_E_E"] ** 2) / (0.15**2)


def _prior_bspl(params):
    prior = 0.5 * (params["q_f"] - 1.0) ** 2 / (10.0**2)
    return prior + 0.5 * (jnp.maximum(-params["q_f"], 0.0)) ** 2 / (3.0**2)


def _prior_fspl(params):
    rho = params["rho"]
    return 0.5 * (jnp.maximum(jnp.log(1.0e-6) - jnp.log(rho), 0.0) / 0.5) ** 2 + 0.5 * (
        jnp.maximum(jnp.log(rho) - jnp.log(1.0), 0.0) / 0.5
    ) ** 2


def _prior_fsbl(params):
    return (
        0.5 * (jnp.maximum(jnp.log(1e-5) - jnp.log(params["rho"]), 0.0) / 0.5) ** 2
        + 0.5 * (jnp.maximum(jnp.log(params["rho"]) - jnp.log(0.2), 0.0) / 0.5) ** 2
        + 0.5 * (jnp.maximum(jnp.log(1e-6) - jnp.log(params["q"]), 0.0) / 0.5) ** 2
        + 0.5 * (jnp.maximum(jnp.log(params["q"]) - jnp.log(1.0), 0.0) / 0.5) ** 2
    )


_PRIOR_FUNCS = {
    "parallax": _prior_parallax,
    "fspl": _prior_fspl,
    "fspl_parallax": lambda p: _prior_fspl(p) + _prior_parallax(p),
    "bspl": _prior_bspl,
    "bspl_parallax": lambda p: _prior_bspl(p) + _prior_parallax(p),
    "fsbl": _prior_fsbl,
}


def magnification(t, params):
    model = params.get("model")
    if model not in _MAGNIFICATION_FUNCS:
        raise NotImplementedError(f"Magnification model '{model}' is not implemented.")
    return _MAGNIFICATION_FUNCS[model](t, params)


def neg_lnprob(t, params, mag, mag_err):
    A = magnification(t, params)
    Fs, _, Fb, _, chi2 = linear_chi2(A, mag, mag_err)

    prior_fb = _negative_flux_prior(Fb, sigma=2.0)
    prior_fs = _negative_flux_prior(Fs, sigma=2.0)
    prior_term = prior_fs + prior_fb

    model = params.get("model")
    prior_fn = _PRIOR_FUNCS.get(model)
    if prior_fn is not None:
        prior_term += prior_fn(params)

    return 0.5 * chi2 + prior_term
