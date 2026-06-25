from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from magnification_model import magnification, neg_lnprob


class ModelBase:
    """Base class for all microlensing models."""

    def __init__(
        self,
        name,
        param_names,
        *,
        learning_rate=1.0e-2,
        n_steps=10_000,
        min_improvement=1.0e-5,
        patience=20,
    ):
        self.name = name
        self.param_names = param_names
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.min_improvement = min_improvement
        self.patience = patience

    def setup_data(self, data, prev_results):
        """Allows expanding data required by some models."""
        return data

    def to_dict(self, params, data):
        """Converts a parameter array to a physical-parameter dictionary."""
        raise NotImplementedError

    def neg_lnprob_fn(self, params, data):
        """Computes negative log posterior/objective for optimization."""
        param_dict = self.to_dict(params, data)
        return neg_lnprob(data["t"], param_dict, data["mag"], data["mag_err"])


class PSPL(ModelBase):
    """Standard point-source point-lens model."""

    def __init__(self):
        super().__init__("PSPL", ["t_0", "t_E", "u_0"], patience=40)

    def to_dict(self, params, data):
        return {
            "model": "pspl",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
        }


class Parallax(ModelBase):
    """PSPL model with satellite/orbital parallax."""

    def __init__(self):
        super().__init__(
            "PSPL+Parallax",
            ["t_0", "t_E", "u_0", "pi_E_N", "pi_E_E"],
            learning_rate=3.0e-3,
            patience=40,
        )

    def setup_data(self, data, prev_results):
        data["t_0_par"] = prev_results["PSPL"]["dict"]["t_0"]
        return data

    def to_dict(self, params, data):
        return {
            "model": "parallax",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
            "pi_E_N": params[3],
            "pi_E_E": params[4],
            "t_0_par": data["t_0_par"],
            "coords": data["coords"],
        }


class FSPL(ModelBase):
    """Finite-source point-lens model for short/FFP-like events."""

    def __init__(self):
        super().__init__(
            "FSPL",
            ["t_0", "t_E", "u_0", "rho"],
            learning_rate=float(os.environ.get("FSPL_LEARNING_RATE", "5.0e-3")),
            n_steps=int(os.environ.get("FSPL_N_STEPS", "5000")),
            min_improvement=float(os.environ.get("FSPL_MIN_IMPROVEMENT", "1.0e-5")),
            patience=int(os.environ.get("FSPL_PATIENCE", "40")),
        )

    def setup_data(self, data, prev_results):
        if "PSPL" not in prev_results:
            raise ValueError("FSPL initialization requires PSPL to be run first.")
        return data

    def to_dict(self, params, data):
        return {
            "model": "fspl",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
            "rho": jnp.exp(params[3]),
        }


class FSPLParallax(ModelBase):
    """Finite-source point-lens model with parallax."""

    def __init__(self):
        super().__init__(
            "FSPL+Parallax",
            ["t_0", "t_E", "u_0", "rho", "pi_E_N", "pi_E_E"],
            learning_rate=float(os.environ.get("FSPL_PARALLAX_LEARNING_RATE", "3.0e-3")),
            n_steps=int(os.environ.get("FSPL_PARALLAX_N_STEPS", os.environ.get("FSPL_N_STEPS", "5000"))),
            min_improvement=float(os.environ.get("FSPL_PARALLAX_MIN_IMPROVEMENT", os.environ.get("FSPL_MIN_IMPROVEMENT", "1.0e-5"))),
            patience=int(os.environ.get("FSPL_PARALLAX_PATIENCE", os.environ.get("FSPL_PATIENCE", "40"))),
        )

    def setup_data(self, data, prev_results):
        if "FSPL" not in prev_results:
            raise ValueError("FSPL+Parallax initialization requires FSPL to be run first.")
        data["t_0_par"] = prev_results["FSPL"]["dict"]["t_0"]
        return data

    def to_dict(self, params, data):
        return {
            "model": "fspl_parallax",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
            "rho": jnp.exp(params[3]),
            "pi_E_N": params[4],
            "pi_E_E": params[5],
            "t_0_par": data["t_0_par"],
            "coords": data["coords"],
        }


def _bspl_second_peak_time(t, flux, n_valid, t_0, t_E, u_0, Fs, Fb):
    """Estimate a second-source peak from positive residuals to PSPL."""
    A_pspl = magnification(t, {"model": "pspl", "t_0": t_0, "t_E": t_E, "u_0": u_0})
    model_flux = Fs * A_pspl + Fb
    residual = flux - model_flux

    valid = jnp.arange(t.shape[0]) < n_valid
    masked_residual = jnp.where(valid, residual, -jnp.inf)
    top_idx = jnp.argsort(masked_residual)[-10:]
    return jnp.mean(t[top_idx])


class BSPL(ModelBase):
    """Binary-source point-lens model, useful as a false-positive competitor."""

    def __init__(self):
        super().__init__(
            "BSPL",
            ["t_0_1", "t_0_2", "t_E", "u_0_1", "u_0_2", "q_f"],
            learning_rate=5.0e-3,
            patience=40,
        )

    def setup_data(self, data, prev_results):
        pspl_dict = prev_results["PSPL"]["dict"]
        Fs = prev_results["PSPL"]["Fs"]
        Fb = prev_results["PSPL"]["Fb"]

        data["t_0_2_guess"] = jax.vmap(_bspl_second_peak_time)(
            data["t"],
            data["mag"],
            data["n_valid"],
            pspl_dict["t_0"],
            pspl_dict["t_E"],
            pspl_dict["u_0"],
            Fs,
            Fb,
        )
        return data

    def to_dict(self, params, data):
        return {
            "model": "bspl",
            "t_0_1": params[0] + data["t_0_shift"],
            "t_0_2": params[1] + data["t_0_shift"],
            "t_E": jnp.exp(params[2]),
            "u_0_1": params[3],
            "u_0_2": params[4],
            "q_f": jnp.exp(params[5]),
        }


class BSPLParallax(ModelBase):
    """Binary-source point-lens model with satellite/orbital parallax."""

    def __init__(self):
        super().__init__(
            "BSPL+Parallax",
            [
                "t_0_1",
                "t_0_2",
                "t_E",
                "u_0_1",
                "u_0_2",
                "q_f",
                "pi_E_N",
                "pi_E_E",
            ],
            learning_rate=2.0e-3,
            patience=50,
        )

    def setup_data(self, data, prev_results):
        if "BSPL" not in prev_results:
            raise ValueError("BSPL+Parallax initialization requires BSPL to be run first.")
        data["t_0_par"] = prev_results["PSPL"]["dict"]["t_0"]
        return data

    def to_dict(self, params, data):
        return {
            "model": "bspl_parallax",
            "t_0_1": params[0] + data["t_0_shift"],
            "t_0_2": params[1] + data["t_0_shift"],
            "t_E": jnp.exp(params[2]),
            "u_0_1": params[3],
            "u_0_2": params[4],
            "q_f": jnp.exp(params[5]),
            "pi_E_N": params[6],
            "pi_E_E": params[7],
            "t_0_par": data["t_0_par"],
            "coords": data["coords"],
        }


class FSBL(ModelBase):
    """
    Finite-source binary-lens model optimized from many starting points.

    Physical parameters:
        t_0, t_E, u_0, s, q, rho, alpha_deg

    Optimized coordinates:
        t_0_shifted, log(t_E), u_0, log(s), log(q), log(rho), alpha_deg

    The model itself is evaluated through microJAX using a complex source
    trajectory and ``mag_binary``.
    """

    def __init__(self):
        super().__init__(
            "FSBL",
            ["t_0", "t_E", "u_0", "s", "q", "rho", "alpha_deg"],
            learning_rate=float(os.environ.get("FSBL_LEARNING_RATE", "2.0e-3")),
            n_steps=int(os.environ.get("FSBL_N_STEPS", "2000")),
            min_improvement=float(os.environ.get("FSBL_MIN_IMPROVEMENT", "1.0e-5")),
            patience=int(os.environ.get("FSBL_PATIENCE", "25")),
        )

    def setup_data(self, data, prev_results):
        if "PSPL" not in prev_results:
            raise ValueError("FSBL initialization requires PSPL to be run first.")
        return data

    def to_dict(self, params, data):
        return {
            "model": "fsbl",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
            "s": jnp.exp(params[3]),
            "q": jnp.exp(params[4]),
            "rho": jnp.exp(params[5]),
            "alpha_deg": params[6],
        }


class FSBLParallax(ModelBase):
    """
    Finite-source binary-lens model with satellite/orbital parallax.

    Optimized coordinates:
        t_0_shifted, log(t_E), u_0, log(s), log(q), log(rho), alpha_deg,
        pi_E_N, pi_E_E

    By default this is initialized from the already optimized FSBL solution plus
    pi_E_N=pi_E_E=0.  This keeps the search memory-safe and avoids repeating the
    full 320-start binary-lens grid unless you explicitly change the initializer.
    """

    def __init__(self):
        super().__init__(
            "FSBL+Parallax",
            [
                "t_0",
                "t_E",
                "u_0",
                "s",
                "q",
                "rho",
                "alpha_deg",
                "pi_E_N",
                "pi_E_E",
            ],
            learning_rate=float(os.environ.get("FSBL_PARALLAX_LEARNING_RATE", "1.0e-3")),
            n_steps=int(os.environ.get("FSBL_PARALLAX_N_STEPS", os.environ.get("FSBL_N_STEPS", "2000"))),
            min_improvement=float(os.environ.get("FSBL_PARALLAX_MIN_IMPROVEMENT", os.environ.get("FSBL_MIN_IMPROVEMENT", "1.0e-5"))),
            patience=int(os.environ.get("FSBL_PARALLAX_PATIENCE", os.environ.get("FSBL_PATIENCE", "25"))),
        )

    def setup_data(self, data, prev_results):
        if "FSBL" not in prev_results:
            raise ValueError("FSBL+Parallax initialization requires FSBL to be run first.")
        data["t_0_par"] = prev_results["FSBL"]["dict"]["t_0"]
        return data

    def to_dict(self, params, data):
        return {
            "model": "fsbl_parallax",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
            "s": jnp.exp(params[3]),
            "q": jnp.exp(params[4]),
            "rho": jnp.exp(params[5]),
            "alpha_deg": params[6],
            "pi_E_N": params[7],
            "pi_E_E": params[8],
            "t_0_par": data["t_0_par"],
            "coords": data["coords"],
        }
