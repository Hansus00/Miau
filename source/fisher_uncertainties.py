"""Fisher-information posterior approximations for submission uncertainties."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special as jsp

from magnification_model import (
    _PRIOR_FUNCS,
    _negative_flux_prior,
    ensure_ephemeris_loaded,
    magnification,
)


@dataclass(frozen=True)
class FisherParamMap:
    submission_key: str
    theta_index: int
    scale_sigma: Callable[[float, float], float]


def _sigma_linear(_value: float, cov_ii: float) -> float:
    return math.sqrt(max(cov_ii, 0.0))


def _sigma_log(value: float, cov_ii: float) -> float:
    return abs(value) * math.sqrt(max(cov_ii, 0.0))


def _sigma_deg_to_rad(_value_deg: float, cov_ii: float) -> float:
    return math.radians(math.sqrt(max(cov_ii, 0.0)))


def _safe_log(x: float, floor: float = 1e-300) -> float:
    return math.log(max(x, floor))


def neg_lnprob_explicit(t, param_dict: Dict[str, Any], mag, mag_err) -> jnp.ndarray:
    """Same objective as the optimizer, with explicit Fs/Fb (not profiled)."""
    A = magnification(t, param_dict)
    Fs = param_dict["Fs"]
    Fb = param_dict["Fb"]
    inv_var = jnp.where(jnp.isfinite(mag_err) & (mag_err > 0), 1.0 / mag_err**2, 0.0)
    resid = mag - Fs * A - Fb
    chi2 = jnp.sum(inv_var * resid**2)

    prior_term = _negative_flux_prior(Fs, 2.0) + _negative_flux_prior(Fb, 2.0)
    model = param_dict.get("model")
    prior_fn = _PRIOR_FUNCS.get(model)
    if prior_fn is not None:
        prior_term += prior_fn(param_dict)
    return 0.5 * chi2 + prior_term


def gaussian_sigma_scale(confidence_level: float) -> float:
    cl = min(max(float(confidence_level), 1e-6), 1.0 - 1e-6)
    return float(math.sqrt(2.0) * jsp.erfinv(jnp.asarray(cl)))


def _parameter_scales(theta: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return jnp.maximum(jnp.abs(theta), 1.0)


def fisher_covariance(
    theta: jnp.ndarray,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    scales = _parameter_scales(theta)

    def scaled_loss(z: jnp.ndarray) -> jnp.ndarray:
        return loss_fn(theta + scales * z)

    H = jax.hessian(scaled_loss)(jnp.zeros_like(theta))
    H = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(H)
    evals = jnp.maximum(evals, 1e-8)
    H_reg = (evecs * evals) @ evecs.T
    cov_z = jnp.linalg.inv(H_reg)
    return (scales[:, None] * cov_z) * scales[None, :]


def _build_maps(model_name: str) -> Tuple[
    Callable[[Dict[str, Any], Dict[str, Any]], jnp.ndarray],
    Callable[[jnp.ndarray, Dict[str, Any]], Dict[str, Any]],
    List[FisherParamMap],
]:
    if model_name == "PSPL":
        def pack(section, _ctx):
            return jnp.array(
                [
                    section["t_0"],
                    _safe_log(section["t_E"]),
                    section["u_0"],
                    section["Fs"],
                    section["Fb"],
                ],
                dtype=jnp.float64,
            )

        def unpack(theta, _ctx):
            return {
                "model": "pspl",
                "t_0": theta[0],
                "t_E": jnp.exp(theta[1]),
                "u_0": theta[2],
                "Fs": theta[3],
                "Fb": theta[4],
            }

        maps = [
            FisherParamMap("t0", 0, _sigma_linear),
            FisherParamMap("tE", 1, _sigma_log),
            FisherParamMap("u0", 2, _sigma_linear),
            FisherParamMap("F0_S", 3, _sigma_linear),
            FisherParamMap("F0_B", 4, _sigma_linear),
        ]
        return pack, unpack, maps

    if model_name == "FSPL":
        def pack(section, _ctx):
            return jnp.array(
                [
                    section["t_0"],
                    _safe_log(section["t_E"]),
                    section["u_0"],
                    _safe_log(section["rho"]),
                    section["Fs"],
                    section["Fb"],
                ],
                dtype=jnp.float64,
            )

        def unpack(theta, _ctx):
            return {
                "model": "fspl",
                "t_0": theta[0],
                "t_E": jnp.exp(theta[1]),
                "u_0": theta[2],
                "rho": jnp.exp(theta[3]),
                "Fs": theta[4],
                "Fb": theta[5],
            }

        maps = [
            FisherParamMap("t0", 0, _sigma_linear),
            FisherParamMap("tE", 1, _sigma_log),
            FisherParamMap("u0", 2, _sigma_linear),
            FisherParamMap("rho", 3, _sigma_log),
            FisherParamMap("F0_S", 4, _sigma_linear),
            FisherParamMap("F0_B", 5, _sigma_linear),
        ]
        return pack, unpack, maps

    if model_name == "PSPL+Parallax":
        def pack(section, _ctx):
            return jnp.array(
                [
                    section["t_0"],
                    _safe_log(section["t_E"]),
                    section["u_0"],
                    section["pi_E_N"],
                    section["pi_E_E"],
                    section["Fs"],
                    section["Fb"],
                ],
                dtype=jnp.float64,
            )

        def unpack(theta, ctx):
            return {
                "model": "parallax",
                "t_0": theta[0],
                "t_E": jnp.exp(theta[1]),
                "u_0": theta[2],
                "pi_E_N": theta[3],
                "pi_E_E": theta[4],
                "Fs": theta[5],
                "Fb": theta[6],
                "t_0_par": ctx["t_0_par"],
                "coords": ctx["coords"],
            }

        maps = [
            FisherParamMap("t0", 0, _sigma_linear),
            FisherParamMap("tE", 1, _sigma_log),
            FisherParamMap("u0", 2, _sigma_linear),
            FisherParamMap("piEN", 3, _sigma_linear),
            FisherParamMap("piEE", 4, _sigma_linear),
            FisherParamMap("F0_S", 5, _sigma_linear),
            FisherParamMap("F0_B", 6, _sigma_linear),
        ]
        return pack, unpack, maps

    if model_name == "FSPL+Parallax":
        def pack(section, _ctx):
            return jnp.array(
                [
                    section["t_0"],
                    _safe_log(section["t_E"]),
                    section["u_0"],
                    _safe_log(section["rho"]),
                    section["pi_E_N"],
                    section["pi_E_E"],
                    section["Fs"],
                    section["Fb"],
                ],
                dtype=jnp.float64,
            )

        def unpack(theta, ctx):
            return {
                "model": "fspl_parallax",
                "t_0": theta[0],
                "t_E": jnp.exp(theta[1]),
                "u_0": theta[2],
                "rho": jnp.exp(theta[3]),
                "pi_E_N": theta[4],
                "pi_E_E": theta[5],
                "Fs": theta[6],
                "Fb": theta[7],
                "t_0_par": ctx["t_0_par"],
                "coords": ctx["coords"],
            }

        maps = [
            FisherParamMap("t0", 0, _sigma_linear),
            FisherParamMap("tE", 1, _sigma_log),
            FisherParamMap("u0", 2, _sigma_linear),
            FisherParamMap("rho", 3, _sigma_log),
            FisherParamMap("piEN", 4, _sigma_linear),
            FisherParamMap("piEE", 5, _sigma_linear),
            FisherParamMap("F0_S", 6, _sigma_linear),
            FisherParamMap("F0_B", 7, _sigma_linear),
        ]
        return pack, unpack, maps

    if model_name == "BSPL":
        def pack(section, _ctx):
            return jnp.array(
                [
                    section["t_0_1"],
                    section["t_0_2"],
                    _safe_log(section["t_E"]),
                    section["u_0_1"],
                    section["u_0_2"],
                    _safe_log(section["q_f"]),
                    section["Fs"],
                    section["Fb"],
                ],
                dtype=jnp.float64,
            )

        def unpack(theta, _ctx):
            return {
                "model": "bspl",
                "t_0_1": theta[0],
                "t_0_2": theta[1],
                "t_E": jnp.exp(theta[2]),
                "u_0_1": theta[3],
                "u_0_2": theta[4],
                "q_f": jnp.exp(theta[5]),
                "Fs": theta[6],
                "Fb": theta[7],
            }

        maps = [
            FisherParamMap("t0", 0, _sigma_linear),
            FisherParamMap("t0_source2", 1, _sigma_linear),
            FisherParamMap("tE", 2, _sigma_log),
            FisherParamMap("u0", 3, _sigma_linear),
            FisherParamMap("u0_source2", 4, _sigma_linear),
            FisherParamMap("flux_ratio", 5, _sigma_log),
            FisherParamMap("F0_B", 7, _sigma_linear),
        ]
        return pack, unpack, maps

    if model_name == "FSBL":
        def pack(section, _ctx):
            return jnp.array(
                [
                    section["t_0"],
                    _safe_log(section["t_E"]),
                    section["u_0"],
                    _safe_log(section["s"]),
                    _safe_log(section["q"]),
                    _safe_log(section["rho"]),
                    section["alpha_deg"],
                    section["Fs"],
                    section["Fb"],
                ],
                dtype=jnp.float64,
            )

        def unpack(theta, _ctx):
            return {
                "model": "fsbl",
                "t_0": theta[0],
                "t_E": jnp.exp(theta[1]),
                "u_0": theta[2],
                "s": jnp.exp(theta[3]),
                "q": jnp.exp(theta[4]),
                "rho": jnp.exp(theta[5]),
                "alpha_deg": theta[6],
                "Fs": theta[7],
                "Fb": theta[8],
            }

        maps = [
            FisherParamMap("t0", 0, _sigma_linear),
            FisherParamMap("tE", 1, _sigma_log),
            FisherParamMap("u0", 2, _sigma_linear),
            FisherParamMap("s", 3, _sigma_log),
            FisherParamMap("q", 4, _sigma_log),
            FisherParamMap("rho", 5, _sigma_log),
            FisherParamMap("alpha", 6, _sigma_deg_to_rad),
            FisherParamMap("F0_S", 7, _sigma_linear),
            FisherParamMap("F0_B", 8, _sigma_linear),
        ]
        return pack, unpack, maps

    raise ValueError(f"Fisher uncertainties are not implemented for model {model_name!r}")


def _select_time_indices(
    t: jnp.ndarray,
    n_valid: int,
    max_points: int,
    start_boundary: Optional[float] = None,
    end_boundary: Optional[float] = None,
) -> jnp.ndarray:
    idx = jnp.arange(n_valid, dtype=jnp.int32)
    if start_boundary is not None and end_boundary is not None:
        valid_t = t[:n_valid]
        window_mask = (valid_t >= start_boundary) & (valid_t <= end_boundary)
        window_idx = jnp.flatnonzero(window_mask)
        if int(window_idx.size) > 0:
            idx = window_idx.astype(jnp.int32)

    if max_points <= 0 or int(idx.size) <= max_points:
        return idx

    sample_pos = jnp.linspace(0, idx.size - 1, max_points)
    return jnp.unique(idx[sample_pos.astype(jnp.int32)])


def fisher_submission_uncertainties(
    model_name: str,
    section: Dict[str, Any],
    data: Dict[str, Any],
    *,
    confidence_level: float = 0.68,
    max_points: int = 2048,
) -> Optional[Dict[str, float]]:
    """
    Estimate 1-D marginal uncertainties from the Fisher information matrix
    (inverse Hessian of the explicit negative log-posterior).
    """
    if model_name not in {
        "PSPL",
        "FSPL",
        "PSPL+Parallax",
        "FSPL+Parallax",
        "BSPL",
        "FSBL",
    }:
        return None

    if "parallax" in model_name.lower() or model_name.endswith("+Parallax"):
        ensure_ephemeris_loaded()

    pack, unpack, maps = _build_maps(model_name)
    ctx = {
        "coords": data.get("coords", jnp.array([0.0, 0.0])),
        "t_0_par": section.get("t_0_par", section.get("t_0", section.get("t_0_1"))),
    }

    try:
        theta0 = pack(section, ctx)
    except (KeyError, TypeError, ValueError):
        return None

    n_valid = int(data.get("n_valid", len(data["t"])))
    idx = _select_time_indices(
        data["t"],
        n_valid,
        max_points,
        data.get("start_boundary"),
        data.get("end_boundary"),
    )
    t = data["t"][idx]
    mag = data["mag"][idx]
    mag_err = data["mag_err"][idx]

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        p = unpack(theta, ctx)
        return neg_lnprob_explicit(t, p, mag, mag_err)

    try:
        cov = fisher_covariance(theta0, loss)
    except Exception:
        return None

    cov_host = jnp.asarray(cov)
    scale = gaussian_sigma_scale(confidence_level)
    out: Dict[str, float] = {}
    for spec in maps:
        ii = spec.theta_index
        cov_ii = float(cov_host[ii, ii])
        if not math.isfinite(cov_ii):
            continue
        center = float(theta0[ii])
        if spec.submission_key in {"tE", "rho", "s", "q", "flux_ratio"}:
            phys_center = math.exp(center)
        elif spec.submission_key == "alpha":
            phys_center = math.radians(center)
        else:
            phys_center = center
        sigma = spec.scale_sigma(phys_center, cov_ii) * scale
        if math.isfinite(sigma) and sigma > 0.0:
            out[spec.submission_key] = sigma

    if model_name == "BSPL" and "flux_ratio" in section:
        qf = float(section["q_f"])
        fs = float(section["Fs"])
        if qf > -0.999999 and fs > 0.0:
            cov_fs = float(cov_host[6, 6])
            cov_lq = float(cov_host[5, 5])
            cov_fq = float(cov_host[6, 5])
            denom = (1.0 + qf) ** 2
            d_s1_fs = 1.0 / (1.0 + qf)
            d_s1_lq = -fs * qf / denom
            var_s1 = (
                d_s1_fs**2 * cov_fs
                + d_s1_lq**2 * cov_lq
                + 2.0 * d_s1_fs * d_s1_lq * cov_fq
            )
            d_s2_fs = qf / (1.0 + qf)
            d_s2_lq = fs / denom
            var_s2 = (
                d_s2_fs**2 * cov_fs
                + d_s2_lq**2 * cov_lq
                + 2.0 * d_s2_fs * d_s2_lq * cov_fq
            )
            if var_s1 > 0.0:
                out["F0_S1"] = math.sqrt(var_s1) * scale
            if var_s2 > 0.0:
                out["F0_S2"] = math.sqrt(var_s2) * scale
    return out or None
