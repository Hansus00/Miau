import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class InitialConditions:
    """Class for estimating initial conditions for optimization."""

    def __init__(self, data, threshold=2.0, min_consecutive=5, window_size=15):
        self.data = data
        self.threshold = threshold
        self.min_consecutive = min_consecutive
        self.window_size = window_size

        b, p, s, e = self._compute_ic_stats(data["t"], data["mag"], data["mag_err"])
        self.baseline = b
        self.main_peak_time = p
        self.t_0_shift = p
        self.start_boundary = s
        self.end_boundary = e

    def _compute_ic_stats(self, t, mag, mag_err):
        baseline = jnp.median(mag)
        main_peak_time = jnp.mean(t[jnp.argsort(mag)[-10:]])

        above_threshold = (mag > baseline + self.threshold * mag_err).astype(jnp.int32)
        consecutive = jax.lax.reduce_window(
            above_threshold,
            0,
            jax.lax.add,
            (self.window_size,),
            (1,),
        )

        mask = consecutive >= self.min_consecutive
        any_valid = jnp.any(mask)

        first_idx = jnp.argmax(mask)
        last_idx = (len(mask) - 1) - jnp.argmax(mask[::-1])

        start_idx = jnp.where(any_valid, first_idx, 0)
        end_idx = jnp.where(any_valid, last_idx + self.window_size - 1, len(t) - 1)

        return baseline, main_peak_time, t[start_idx], t[end_idx]

    def get_processed_data(self):
        """Returns data augmented with initial conditions."""
        processed = self.data.copy()
        processed["t_0_shift"] = jnp.asarray(self.t_0_shift, dtype=jnp.float64)
        processed["start_boundary"] = jnp.asarray(
            self.start_boundary, dtype=jnp.float64
        )
        processed["end_boundary"] = jnp.asarray(self.end_boundary, dtype=jnp.float64)
        processed["baseline"] = jnp.asarray(self.baseline, dtype=jnp.float64)
        return processed

    def get_model_init_params(self, model_name, prev_results):
        """Returns the initial parameters for a given model."""
        init_methods = {
            "PSPL": self._init_pspl,
            "PSPL+Parallax": self._init_pspl_parallax,
            "BSPL": self._init_bspl,
            "FSBL": self._init_fsbl,
        }

        if model_name not in init_methods:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {list(init_methods.keys())}"
            )

        return init_methods[model_name](prev_results)

    def _init_pspl(self, prev_results):
        return jnp.array([0.0, jnp.log(20.0), 0.1], dtype=jnp.float64)

    def _init_pspl_parallax(self, prev_results):
        pspl = prev_results["PSPL"]["raw_params"]
        return jnp.concatenate([pspl, jnp.array([0.0, 0.0], dtype=jnp.float64)])

    def _init_bspl(self, prev_results):
        pspl = prev_results["PSPL"]["raw_params"]
        return jnp.array([0.0, 0.0, pspl[1], pspl[2], pspl[2], -1.0], dtype=jnp.float64)

    def _init_fsbl(self, prev_results):
        pspl = prev_results["PSPL"]["raw_params"]
        return jnp.concatenate(
            [
                pspl,
                jnp.array(
                    [jnp.log(1.0), jnp.log(1.0), jnp.log(1.0e-3), 0.0],
                    dtype=jnp.float64,
                ),
            ]
        )
