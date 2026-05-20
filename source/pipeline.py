import os
import optax

from initial_conditions import InitialConditions
from magnification_model import magnification
from optimization import build_optimize_loop, get_eval_metrics


def run_pipeline(files, out_dir, data_loader, models):
    """Main pipeline execution for a list of light curve files."""
    optimizer = optax.adam(learning_rate=1e-2)
    n_steps = 10_000
    min_chi2_improvement = 1e-5
    patience = 20

    opt_loops = {
        m.name: build_optimize_loop(
            m.neg_lnprob_fn, optimizer, n_steps, min_chi2_improvement, patience
        )
        for m in models
    }

    os.makedirs(out_dir, exist_ok=True)

    for file in files:
        raw_data = data_loader.load_event(file)

        init_conds = InitialConditions(raw_data)
        base_data = init_conds.get_processed_data()

        prev_results = {}
        out_file = os.path.join(
            out_dir, os.path.basename(file).replace(".csv", "_params.txt")
        )

        with open(out_file, "w") as f:
            for m in models:
                data = m.setup_data(base_data.copy(), prev_results)

                init_params = init_conds.get_model_init_params(m.name, prev_results)

                res = opt_loops[m.name](init_params, data)
                opt_params = res["params"]

                param_dict = m.to_dict(opt_params, data)
                A = magnification(data["t"], param_dict)
                Fs, Fb, chi2 = get_eval_metrics(A, data["mag"], data["mag_err"])
                dof = len(data["mag"]) - len(m.param_names)

                prev_results[m.name] = {
                    "raw_params": opt_params,
                    "dict": param_dict,
                }

                f.write(f"[{m.name}]\n")
                for key in m.param_names:
                    f.write(f"{key}: {param_dict[key]}\n")
                f.write(f"Chi2: {chi2}\n")
                f.write(f"chi2/dof: {chi2 / dof}\n")
                f.write(f"Fs: {Fs}\n")
                f.write(f"Fb: {Fb}\n\n")
