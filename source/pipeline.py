import os
import optax
import numpy as np
from collections import defaultdict

import jax
import jax.numpy as jnp

from initial_conditions import InitialConditions
from magnification_model import ensure_ephemeris_loaded, magnification
from optimization import build_optimize_loop, get_eval_metrics


_PARALLAX_MODEL_NAMES = frozenset({"PSPL+Parallax", "FSPL+Parallax"})


def run_pipeline(files, out_dir, data_loader, models, max_len):
    """Main pipeline execution for a batch of light curve files."""
    if any(m.name in _PARALLAX_MODEL_NAMES for m in models):
        ensure_ephemeris_loaded()

    optimizer = optax.adam(learning_rate=1e-2)
    n_steps = 10_000
    min_chi2_improvement = 1e-5
    patience = 20

    vmap_opt_loops = {
        m.name: jax.jit(
            jax.vmap(
                build_optimize_loop(
                    m.neg_lnprob_fn, optimizer, n_steps, min_chi2_improvement, patience
                )
            )
        )
        for m in models
    }

    os.makedirs(out_dir, exist_ok=True)

    batch_data_dict = defaultdict(list)
    valid_files = []
    event_objects = []

    for file in files:
        try:
            raw_data = data_loader.load_event(file)

            event_name = os.path.splitext(os.path.basename(file))[0]
            if (
                hasattr(data_loader, "event_coords")
                and event_name in data_loader.event_coords
            ):
                raw_data["coords"] = data_loader.event_coords[event_name]
            elif "coords" not in raw_data:
                raw_data["coords"] = jnp.array([0.0, 0.0])

            init_conds = InitialConditions(raw_data)
            pd_data = init_conds.get_processed_data(max_len=max_len)

            for k, v in pd_data.items():
                batch_data_dict[k].append(v)

            valid_files.append(file)
            event_objects.append(init_conds)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    batched_data = {
        k: jnp.stack(v)
        for k, v in batch_data_dict.items()
        if isinstance(v[0], (jnp.ndarray, float, int, np.ndarray))
    }
    num_events = len(valid_files)

    prev_results = {}
    event_model_results = {i: {} for i in range(num_events)}

    for m in models:

        batched_data = m.setup_data(batched_data.copy(), prev_results)

        batched_init_params = []
        for i in range(num_events):
            single_prev = {
                pm: {"raw_params": prev_results[pm]["raw_params"][i]}
                for pm in prev_results
            }
            single_data_for_init = {k: v[i] for k, v in batched_data.items()}
            p = event_objects[i].get_model_init_params(
                m.name, single_prev, single_data_for_init
            )
            batched_init_params.append(p)
        batched_init_params = jnp.stack(batched_init_params)

        res = vmap_opt_loops[m.name](batched_init_params, batched_data)
        batched_opt_params = res["params"]

        batched_dict_lists = defaultdict(list)
        Fs_list, Fb_list = [], []
        for i in range(num_events):
            single_data = {k: v[i] for k, v in batched_data.items()}
            single_params = batched_opt_params[i]
            p_dict = m.to_dict(single_params, single_data)

            A = magnification(single_data["t"], p_dict)
            Fs, Fb, chi2 = get_eval_metrics(
                A, single_data["mag"], single_data["mag_err"]
            )
            dof = int(single_data["n_valid"]) - len(m.param_names)

            event_model_results[i][m.name] = {
                "param_dict": p_dict,
                "chi2": chi2,
                "dof": dof,
                "Fs": Fs,
                "Fb": Fb,
            }
            for k, v in p_dict.items():
                batched_dict_lists[k].append(v)
            Fs_list.append(Fs)
            Fb_list.append(Fb)

        prev_results[m.name] = {
            "raw_params": batched_opt_params,
            "dict": {
                k: (jnp.stack(v) if isinstance(v[0], (jnp.ndarray, float, int)) else v)
                for k, v in batched_dict_lists.items()
            },
            "Fs": jnp.stack(Fs_list),
            "Fb": jnp.stack(Fb_list),
        }

    for i, file in enumerate(valid_files):
        out_file = os.path.join(
            out_dir, os.path.basename(file).replace(".csv", "_params.txt")
        )
        with open(out_file, "w") as f:
            for m in models:
                res = event_model_results[i][m.name]
                f.write(f"[{m.name}]\n")
                for key in m.param_names:
                    f.write(f"{key}: {res['param_dict'][key]}\n")
                f.write(f"Chi2: {res['chi2']}\n")
                f.write(f"chi2/dof: {res['chi2'] / res['dof']}\n")
                f.write(f"Fs: {res['Fs']}\n")
                f.write(f"Fb: {res['Fb']}\n\n")