from __future__ import annotations

import os
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from initial_conditions import InitialConditions
from magnification_model import magnification
from optimization import build_optimize_loop, get_eval_metrics


def _stack_numeric_dict(batch_data_dict):
    out = {}
    for k, v in batch_data_dict.items():
        if len(v) == 0:
            continue
        if isinstance(v[0], (jnp.ndarray, np.ndarray, float, int)):
            out[k] = jnp.stack(v)
    return out


def _as_multistart_params(p):
    """Normalize init params to shape (n_starts, n_params)."""
    p = jnp.asarray(p, dtype=jnp.float64)
    if p.ndim == 1:
        return p[None, :]
    if p.ndim == 2:
        return p
    raise ValueError(f"Initial parameters must be 1D or 2D, got shape {p.shape}.")


def _select_best_start(opt_result):
    """Select best optimized parameters for each event from batched multistart output."""
    objective = opt_result["prev_chi2"]
    best_idx = jnp.argmin(objective, axis=1)
    event_idx = jnp.arange(objective.shape[0])
    best_params = opt_result["params"][event_idx, best_idx]
    best_objective = objective[event_idx, best_idx]
    return best_params, best_idx, best_objective


def _single_event_data(batched_data, event_index, *, trim_to_valid=False):
    """Extract one event from the batched dictionary."""
    single = {k: v[event_index] for k, v in batched_data.items()}

    if trim_to_valid:
        n_valid = int(np.asarray(single["n_valid"]))
        for key in ("t", "mag", "mag_err"):
            if key in single:
                single[key] = single[key][:n_valid]
        single["n_valid"] = jnp.asarray(n_valid, dtype=jnp.int32)

    return single


def _thin_event_data(single_data, *, max_points, peak_fraction=0.70, fixed_length=True):
    """
    Build a compact, information-rich data subset for expensive FSBL calls.

    microJAX finite-source binary-lens magnification is far too expensive to
    evaluate for every survey-cadence point during a blind grid search.  This
    routine keeps a mixture of:

      1. the highest signal-to-noise/excursion points, which are most likely to
         contain caustic/anomaly information;
      2. a uniform-in-time skeleton, which keeps baseline and wings constrained.

    If ``fixed_length=True`` the returned arrays always have length
    ``max_points``.  Short light curves are padded with infinite errors so JAX
    compiles one reusable shape instead of recompiling for many lengths.
    """
    if max_points <= 0:
        return single_data

    t = np.asarray(single_data["t"])
    flux = np.asarray(single_data["mag"])
    ferr = np.asarray(single_data["mag_err"])
    n = len(t)

    if n == 0:
        return single_data

    n_keep = min(int(max_points), n)
    if n <= n_keep:
        idx = np.arange(n, dtype=np.int64)
    else:
        baseline = np.nanmedian(flux)
        safe_err = np.where(np.isfinite(ferr) & (ferr > 0.0), ferr, np.inf)
        signal = np.abs(flux - baseline) / safe_err
        signal = np.where(np.isfinite(signal), signal, -np.inf)

        n_peak = int(round(n_keep * peak_fraction))
        n_peak = max(1, min(n_peak, n_keep))
        n_uniform = n_keep - n_peak

        peak_idx = np.argpartition(-signal, n_peak - 1)[:n_peak]
        if n_uniform > 0:
            uniform_idx = np.linspace(0, n - 1, n_uniform, dtype=np.int64)
            idx = np.unique(np.concatenate([peak_idx, uniform_idx]))
            # Unique may slightly reduce count; top up with next-best signal points.
            if len(idx) < n_keep:
                extra = np.argsort(-signal)
                extra = extra[~np.isin(extra, idx)][: n_keep - len(idx)]
                idx = np.concatenate([idx, extra])
        else:
            idx = peak_idx

        idx = np.sort(idx[:n_keep])

    out = dict(single_data)
    out["t"] = jnp.asarray(t[idx], dtype=single_data["t"].dtype)
    out["mag"] = jnp.asarray(flux[idx], dtype=single_data["mag"].dtype)
    out["mag_err"] = jnp.asarray(ferr[idx], dtype=single_data["mag_err"].dtype)
    out["n_valid"] = jnp.asarray(len(idx), dtype=jnp.int32)

    if fixed_length and len(idx) < max_points:
        pad = int(max_points) - len(idx)
        t_pad_val = float(out["t"][-1]) if len(idx) > 0 else 0.0
        out["t"] = jnp.pad(out["t"], (0, pad), constant_values=t_pad_val)
        out["mag"] = jnp.pad(out["mag"], (0, pad), constant_values=0.0)
        out["mag_err"] = jnp.pad(out["mag_err"], (0, pad), constant_values=jnp.inf)
        # n_valid intentionally stays as the number of real points, but the
        # likelihood ignores the padded points through mag_err=inf.

    return out


def _append_model_summary(
    *,
    m,
    event_index,
    single_data,
    best_params,
    best_start_idx,
    best_objective,
    n_starts,
    event_model_results,
    batched_dict_lists,
    Fs_list,
    Fb_list,
    chi2_list,
    eval_n_points=None,
):
    """Compute final pure chi2/Fs/Fb for one event and save bookkeeping."""
    p_dict = m.to_dict(best_params, single_data)
    A = magnification(single_data["t"], p_dict)
    Fs, Fb, chi2 = get_eval_metrics(A, single_data["mag"], single_data["mag_err"])

    # Two linear nuisance parameters Fs and Fb are solved analytically.
    dof = int(np.asarray(single_data["n_valid"])) - len(m.param_names) - 2
    dof = max(dof, 1)

    event_model_results[event_index][m.name] = {
        "param_dict": p_dict,
        "chi2": chi2,
        "dof": dof,
        "Fs": Fs,
        "Fb": Fb,
        "best_start_idx": best_start_idx,
        "best_objective": best_objective,
        "n_starts": n_starts,
        "eval_n_points": int(eval_n_points) if eval_n_points is not None else int(np.asarray(single_data["n_valid"])),
    }

    for k, v in p_dict.items():
        batched_dict_lists[k].append(v)
    Fs_list.append(Fs)
    Fb_list.append(Fb)
    chi2_list.append(chi2)


def _run_batched_model(
    *,
    m,
    batched_data,
    batched_init_params,
    num_events,
    n_starts,
    event_model_results,
):
    """Fast path for cheap models such as PSPL and BSPL."""
    optimizer = optax.adam(learning_rate=m.learning_rate)
    opt_loop = build_optimize_loop(
        m.neg_lnprob_fn,
        optimizer,
        m.n_steps,
        m.min_improvement,
        m.patience,
    )

    run_one_model = jax.jit(
        jax.vmap(jax.vmap(opt_loop, in_axes=(0, None)), in_axes=(0, 0))
    )

    opt_result = run_one_model(batched_init_params, batched_data)
    batched_opt_params, best_start_idx, best_objective = _select_best_start(opt_result)

    batched_dict_lists = defaultdict(list)
    Fs_list, Fb_list, chi2_list = [], [], []

    for i in range(num_events):
        single_data = _single_event_data(batched_data, i, trim_to_valid=False)
        _append_model_summary(
            m=m,
            event_index=i,
            single_data=single_data,
            best_params=batched_opt_params[i],
            best_start_idx=best_start_idx[i],
            best_objective=best_objective[i],
            n_starts=n_starts,
            event_model_results=event_model_results,
            batched_dict_lists=batched_dict_lists,
            Fs_list=Fs_list,
            Fb_list=Fb_list,
            chi2_list=chi2_list,
        )

    return {
        "raw_params": batched_opt_params,
        "dict": {
            k: (jnp.stack(v) if isinstance(v[0], (jnp.ndarray, float, int)) else v)
            for k, v in batched_dict_lists.items()
        },
        "Fs": jnp.stack(Fs_list),
        "Fb": jnp.stack(Fb_list),
        "chi2": jnp.stack(chi2_list),
        "best_start_idx": best_start_idx,
        "best_objective": best_objective,
        "n_starts": n_starts,
    }


def _run_memory_heavy_multistart_model(
    *,
    m,
    batched_data,
    init_params_per_event,
    num_events,
    event_model_results,
    start_chunk_size,
    top_k,
    coarse_max_points,
    opt_max_points,
    final_full_eval,
    skip_delta_chi2,
):
    """
    Fast and safe path for FSBL/microJAX.

    Key design:
      * never vmap over all events;
      * never coarse-rank on the full light curve;
      * optimize only top_k starts;
      * optionally evaluate final chi2 on the full curve after the best fit.
    """
    optimizer = optax.adam(learning_rate=m.learning_rate)
    opt_loop = build_optimize_loop(
        m.neg_lnprob_fn,
        optimizer,
        m.n_steps,
        m.min_improvement,
        m.patience,
    )

    eval_start_chunk = jax.jit(jax.vmap(m.neg_lnprob_fn, in_axes=(0, None)))
    run_start_chunk = jax.jit(jax.vmap(opt_loop, in_axes=(0, None)))

    batched_dict_lists = defaultdict(list)
    Fs_list, Fb_list, chi2_list = [], [], []
    best_params_all = []
    best_start_idx_all = []
    best_objective_all = []

    for i in range(num_events):
        full_data = _single_event_data(batched_data, i, trim_to_valid=True)
        starts = init_params_per_event[i]
        n_starts = int(starts.shape[0])

        # Optional screening: if the best previous cheap model already fits well,
        # do not spend microJAX time on a binary-lens search. Disabled by default.
        if skip_delta_chi2 is not None:
            cheap_chi2_values = []
            for model_name, res in event_model_results[i].items():
                if "FSBL" not in model_name.upper():
                    cheap_chi2_values.append(float(np.asarray(res["chi2"])))
            if cheap_chi2_values:
                best_cheap = min(cheap_chi2_values)
                pspl_chi2 = float(np.asarray(event_model_results[i].get("PSPL", {"chi2": best_cheap})["chi2"]))
                if pspl_chi2 - best_cheap < skip_delta_chi2:
                    # Still write a valid entry by evaluating the first FSBL start.
                    best_params_i = starts[0]
                    best_objective_i = jnp.asarray(np.nan, dtype=jnp.float64)
                    best_start_idx_i = 0
                    eval_data = _thin_event_data(full_data, max_points=opt_max_points, fixed_length=True)
                    _append_model_summary(
                        m=m,
                        event_index=i,
                        single_data=eval_data,
                        best_params=best_params_i,
                        best_start_idx=jnp.asarray(best_start_idx_i, dtype=jnp.int32),
                        best_objective=best_objective_i,
                        n_starts=n_starts,
                        event_model_results=event_model_results,
                        batched_dict_lists=batched_dict_lists,
                        Fs_list=Fs_list,
                        Fb_list=Fb_list,
                        chi2_list=chi2_list,
                        eval_n_points=int(np.asarray(eval_data["n_valid"])),
                    )
                    best_params_all.append(best_params_i)
                    best_start_idx_all.append(jnp.asarray(best_start_idx_i, dtype=jnp.int32))
                    best_objective_all.append(best_objective_i)
                    print(f"  {m.name} event {i + 1}/{num_events}: skipped by cheap-model screen")
                    continue

        coarse_data = _thin_event_data(full_data, max_points=coarse_max_points, fixed_length=True)
        opt_data = _thin_event_data(full_data, max_points=opt_max_points, fixed_length=True)

        print(
            f"  {m.name} event {i + 1}/{num_events}: "
            f"n_valid={int(np.asarray(full_data['n_valid']))}, "
            f"coarse_points={len(coarse_data['t'])}, opt_points={len(opt_data['t'])}, "
            f"grid_starts={n_starts}, top_k={min(top_k, n_starts)}, "
            f"chunk={start_chunk_size}"
        )

        # Stage 1: coarse ranking on a deliberately thinned light curve.
        coarse_scores = []
        for start0 in range(0, n_starts, start_chunk_size):
            start1 = min(start0 + start_chunk_size, n_starts)
            start_chunk = starts[start0:start1]
            scores = 2.0 * eval_start_chunk(start_chunk, coarse_data)
            coarse_scores.append(scores)

        coarse_scores = jnp.concatenate(coarse_scores)
        n_to_optimize = min(top_k, n_starts)
        candidate_idx = jnp.argsort(coarse_scores)[:n_to_optimize]
        candidate_starts = starts[candidate_idx]

        best_params_i = None
        best_objective_i = None
        best_start_idx_i = None

        # Stage 2: gradient optimization on a larger, but still bounded subset.
        for cand0 in range(0, n_to_optimize, start_chunk_size):
            cand1 = min(cand0 + start_chunk_size, n_to_optimize)
            start_chunk = candidate_starts[cand0:cand1]
            original_indices = candidate_idx[cand0:cand1]

            opt_result = run_start_chunk(start_chunk, opt_data)
            objective = opt_result["prev_chi2"]
            local_best = int(np.asarray(jnp.argmin(objective)))
            local_objective = objective[local_best]
            local_params = opt_result["params"][local_best]
            local_original_idx = int(np.asarray(original_indices[local_best]))

            if best_objective_i is None or float(np.asarray(local_objective)) < float(np.asarray(best_objective_i)):
                best_objective_i = local_objective
                best_params_i = local_params
                best_start_idx_i = local_original_idx

        # Stage 3: final reporting. Full-curve finite-source evaluation is useful
        # but can dominate runtime; for blind triage the subset chi2 is enough.
        eval_data = full_data if final_full_eval else opt_data
        _append_model_summary(
            m=m,
            event_index=i,
            single_data=eval_data,
            best_params=best_params_i,
            best_start_idx=jnp.asarray(best_start_idx_i, dtype=jnp.int32),
            best_objective=best_objective_i,
            n_starts=n_starts,
            event_model_results=event_model_results,
            batched_dict_lists=batched_dict_lists,
            Fs_list=Fs_list,
            Fb_list=Fb_list,
            chi2_list=chi2_list,
            eval_n_points=int(np.asarray(eval_data["n_valid"])),
        )

        best_params_all.append(best_params_i)
        best_start_idx_all.append(jnp.asarray(best_start_idx_i, dtype=jnp.int32))
        best_objective_all.append(best_objective_i)

    return {
        "raw_params": jnp.stack(best_params_all),
        "dict": {
            k: (jnp.stack(v) if isinstance(v[0], (jnp.ndarray, float, int)) else v)
            for k, v in batched_dict_lists.items()
        },
        "Fs": jnp.stack(Fs_list),
        "Fb": jnp.stack(Fb_list),
        "chi2": jnp.stack(chi2_list),
        "best_start_idx": jnp.stack(best_start_idx_all),
        "best_objective": jnp.stack(best_objective_all),
        "n_starts": int(init_params_per_event[0].shape[0]),
    }


def run_pipeline(files, out_dir, data_loader, models, max_len):
    """Main pipeline execution for a batch of light curve files."""
    os.makedirs(out_dir, exist_ok=True)

    fsbl_start_chunk_size = max(int(os.environ.get("FSBL_START_CHUNK", "4")), 1)
    fsbl_top_k = max(int(os.environ.get("FSBL_TOPK", "4")), 1)
    fsbl_coarse_max_points = max(int(os.environ.get("FSBL_COARSE_MAX_POINTS", "256")), 16)
    fsbl_opt_max_points = max(int(os.environ.get("FSBL_OPT_MAX_POINTS", "768")), 32)
    fsbl_final_full_eval = os.environ.get("FSBL_FINAL_FULL_EVAL", "0") == "1"

    skip_raw = os.environ.get("FSBL_SKIP_DELTA_CHI2", "")
    fsbl_skip_delta_chi2 = float(skip_raw) if skip_raw.strip() else None

    batch_data_dict = defaultdict(list)
    valid_files = []
    event_objects = []

    for file in files:
        try:
            raw_data = data_loader.load_event(file)

            event_name = os.path.splitext(os.path.basename(file))[0]
            if hasattr(data_loader, "event_coords") and event_name in data_loader.event_coords:
                raw_data["coords"] = data_loader.event_coords[event_name]
            elif "coords" not in raw_data:
                raw_data["coords"] = jnp.array([0.0, 0.0], dtype=jnp.float64)

            init_conds = InitialConditions(raw_data)
            pd_data = init_conds.get_processed_data(max_len=max_len)

            for k, v in pd_data.items():
                batch_data_dict[k].append(v)

            valid_files.append(file)
            event_objects.append(init_conds)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if len(valid_files) == 0:
        print("No valid light curves to fit.")
        return

    batched_data = _stack_numeric_dict(batch_data_dict)
    num_events = len(valid_files)

    prev_results = {}
    event_model_results = {i: {} for i in range(num_events)}

    for m in models:
        print(f"\n=== Fitting {m.name} for {num_events} event(s) ===")

        batched_data = m.setup_data(batched_data.copy(), prev_results)

        init_params_per_event = []
        for i in range(num_events):
            single_prev = {pm: {"raw_params": prev_results[pm]["raw_params"][i]} for pm in prev_results}
            single_data_for_init = _single_event_data(batched_data, i, trim_to_valid=False)
            p = event_objects[i].get_model_init_params(m.name, single_prev, single_data_for_init)
            init_params_per_event.append(_as_multistart_params(p))

        n_starts_set = {int(p.shape[0]) for p in init_params_per_event}
        if len(n_starts_set) != 1:
            raise ValueError(
                f"All events must have the same number of starts for {m.name}; got {sorted(n_starts_set)}."
            )
        n_starts = next(iter(n_starts_set))
        print(f"Using {n_starts} start(s) per event for {m.name}.")

        is_fsbl_family = "FSBL" in m.name.upper()

        if is_fsbl_family:
            print(
                f"{m.name} uses fast finite-source mode: event-by-event, "
                f"coarse subset={fsbl_coarse_max_points}, opt subset={fsbl_opt_max_points}, "
                f"chunk={fsbl_start_chunk_size}, top_k={fsbl_top_k}, "
                f"final_full_eval={fsbl_final_full_eval}."
            )
            prev_results[m.name] = _run_memory_heavy_multistart_model(
                m=m,
                batched_data=batched_data,
                init_params_per_event=init_params_per_event,
                num_events=num_events,
                event_model_results=event_model_results,
                start_chunk_size=fsbl_start_chunk_size,
                top_k=fsbl_top_k,
                coarse_max_points=fsbl_coarse_max_points,
                opt_max_points=fsbl_opt_max_points,
                final_full_eval=fsbl_final_full_eval,
                skip_delta_chi2=fsbl_skip_delta_chi2,
            )
        else:
            batched_init_params = jnp.stack(init_params_per_event)
            prev_results[m.name] = _run_batched_model(
                m=m,
                batched_data=batched_data,
                batched_init_params=batched_init_params,
                num_events=num_events,
                n_starts=n_starts,
                event_model_results=event_model_results,
            )

    for i, file in enumerate(valid_files):
        out_file = os.path.join(out_dir, os.path.basename(file).replace(".csv", "_params.txt"))
        with open(out_file, "w", encoding="utf-8") as f:
            for m in models:
                res = event_model_results[i][m.name]
                f.write(f"[{m.name}]\n")
                f.write(f"n_starts: {res['n_starts']}\n")
                f.write(f"best_start_idx: {res['best_start_idx']}\n")
                f.write(f"best_objective_2lnpost: {res['best_objective']}\n")
                f.write(f"eval_n_points: {res['eval_n_points']}\n")
                for key in m.param_names:
                    f.write(f"{key}: {res['param_dict'][key]}\n")
                f.write(f"Chi2: {res['chi2']}\n")
                f.write(f"chi2/dof: {res['chi2'] / res['dof']}\n")
                f.write(f"Fs: {res['Fs']}\n")
                f.write(f"Fb: {res['Fb']}\n\n")
