import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from magnification_model import magnification, neg_lnprob


class ModelBase:
    """Base class for all microlensing models."""

    def __init__(self, name, param_names):
        self.name = name
        self.param_names = param_names

    def setup_data(self, data, prev_results):
        """Allows expanding data required by some models."""
        return data

    def to_dict(self, params, data):
        """Converts parameter array to dictionary format."""
        raise NotImplementedError

    def neg_lnprob_fn(self, params, data):
        """Computes negative log probability for optimization."""
        param_dict = self.to_dict(params, data)
        return neg_lnprob(data["t"], param_dict, data["mag"], data["mag_err"])


class PSPL(ModelBase):
    """Standard Point Source Point Lens Model.
    Reparametrization:
    - t_0: t_0 - t_0_shift
    - t_E: log(t_E)
    - u_0: u_0
    """

    def __init__(self):
        super().__init__("PSPL", ["t_0", "t_E", "u_0"])

    def to_dict(self, params, data):
        return {
            "model": "pspl",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
        }


class FSPL(ModelBase):
    """
    Finite Source Point Lens Model.
    Reparametrization:
    - t_0: t_0 - t_0_shift
    - t_E: log(t_E)
    - u_0: u_0
    - rho: log(rho)
    """

    def __init__(self):
        super().__init__("FSPL", ["t_0", "t_E", "u_0", "rho"])

    def to_dict(self, params, data):
        return {
            "model": "fspl",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": jnp.exp(params[1]),
            "u_0": params[2],
            "rho": jnp.exp(params[3]),
        }


class Parallax(ModelBase):
    """
    PSPL Model with Satellite Parallax.
    Reparametrization:
    - t_0: t_0 - t_0_shift
    - t_E: log(t_E)
    - u_0: u_0
    - pi_E_N: pi_E_N
    - pi_E_E: pi_E_E
    """

    def __init__(self):
        super().__init__("PSPL+Parallax", ["t_0", "t_E", "u_0", "pi_E_N", "pi_E_E"])

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


def _bspl_second_peak_time(t, mag, n_valid, t_0, t_E, u_0, Fs, Fb):
    """
    helper meant to be jax.vmap'd over a batch axis
    """
    A_pspl = magnification(t, {"model": "pspl", "t_0": t_0, "t_E": t_E, "u_0": u_0})
    model_flux = Fs * A_pspl + Fb
    residual = mag - model_flux

    valid = jnp.arange(t.shape[0]) < n_valid
    masked_residual = jnp.where(valid, residual, -jnp.inf)
    top_idx = jnp.argsort(masked_residual)[-10:]
    return jnp.mean(t[top_idx])


class BSPL(ModelBase):
    """
    Binary Source Point Lens Model.
    Reparametrization:
    - t_0_1: t_0_1 - t_0_shift
    - t_0_2: t_0_2 - t_0_shift
    - t_E: log(t_E)
    - u_0_1: u_0_1
    - u_0_2: u_0_2
    - q_f: log(q_f)
    """

    def __init__(self):
        super().__init__("BSPL", ["t_0_1", "t_0_2", "t_E", "u_0_1", "u_0_2", "q_f"])

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


class FSBL(ModelBase):
    """
    Finite Source Binary Lens Model.
    Reparametrization:
    - t_0: t_0 - t_0_shift
    - t_E: log(t_E)
    - u_0: u_0
    - s: log(s)
    - q: log(q)
    - rho: log(rho)
    - alpha_deg: alpha_deg
    """

    def __init__(self):
        super().__init__("FSBL", ["t_0", "t_E", "u_0", "s", "q", "rho", "alpha_deg"])

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


class FSBLGrid(ModelBase):
    """
    Finite Source Binary Lens initial grid search model.
    No reparametrization other than t_0_shift.
    """

    def __init__(self):
        super().__init__(
            "FSBLGrid", ["t_0", "t_E", "u_0", "s", "q", "rho", "alpha_deg"]
        )

    def to_dict(self, params, data):
        return {
            "model": "fsbl_grid",
            "t_0": params[0] + data["t_0_shift"],
            "t_E": params[1],
            "u_0": params[2],
            "s": params[3],
            "q": params[4],
            "rho": params[5],
            "alpha_deg": params[6],
        }
