import optax

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from microjax.likelihood import linear_chi2


def build_optimize_loop(neg_lnprob_fn, optimizer, n_steps, min_improvement, patience):
    """Creates a JIT-compiled optimization loop."""
    max_steps = jnp.array(n_steps, dtype=jnp.int32)
    patience_limit = jnp.array(patience, dtype=jnp.int32)
    min_imp = jnp.array(min_improvement, dtype=jnp.float64)

    @jax.jit
    def optimize_step(params, opt_state, data):
        neg_lnprob_val, grads = jax.value_and_grad(neg_lnprob_fn)(params, data)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
            value=neg_lnprob_val,
            grad=grads,
            value_fn=lambda p: neg_lnprob_fn(p, data),
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, neg_lnprob_val

    @jax.jit
    def optimize_loop(init_params, data):
        opt_state = optimizer.init(init_params)
        init_neg_lnprob = neg_lnprob_fn(init_params, data)

        state = {
            "step": jnp.array(0, dtype=jnp.int32),
            "params": init_params,
            "opt_state": opt_state,
            "prev_chi2": 2.0 * init_neg_lnprob,
            "low_improvement_count": jnp.array(0, dtype=jnp.int32),
        }

        def cond_fn(s):
            return (s["step"] < max_steps) & (
                s["low_improvement_count"] < patience_limit
            )

        def body_fn(s):
            next_params, next_opt_state, next_neg_lnprob = optimize_step(
                s["params"], s["opt_state"], data
            )
            next_chi2 = 2.0 * next_neg_lnprob
            chi2_improvement = s["prev_chi2"] - next_chi2

            low_imp_count = jnp.where(
                chi2_improvement < min_imp,
                s["low_improvement_count"] + 1,
                jnp.array(0, dtype=jnp.int32),
            )

            return {
                "step": s["step"] + 1,
                "params": next_params,
                "opt_state": next_opt_state,
                "prev_chi2": next_chi2,
                "low_improvement_count": low_imp_count,
            }

        return jax.lax.while_loop(cond_fn, body_fn, state)

    return optimize_loop


@jax.jit
def get_eval_metrics(A_model, mag, mag_err):
    """Helper to compute chi2 for file writing."""
    Fs, _, Fb, _, chi2 = linear_chi2(A_model, mag, mag_err)
    return Fs, Fb, chi2
