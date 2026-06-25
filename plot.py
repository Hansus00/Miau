#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Make imports from ./source work when this script is in project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "source"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    import jax.numpy as jnp
    from magnification_model import magnification
except Exception as exc:
    raise RuntimeError(
        "Could not import project magnification_model. "
        "Run this script from the project root, or place it next to the source/ directory."
    ) from exc


# ============================================================
# Model names
# ============================================================

MODEL_TO_INTERNAL = {
    "PSPL": "pspl",
    "PSPL+Parallax": "parallax",
    "FSPL": "fspl",
    "FSPL+Parallax": "fspl_parallax",
    "BSPL": "bspl",
    "BSPL+Parallax": "bspl_parallax",
    "FSBL": "fsbl",
    "FSBL+Parallax": "fsbl_parallax",
}

DEFAULT_MODEL_ORDER = [
    "PSPL",
    "FSPL",
    "PSPL+Parallax",
    "FSPL+Parallax",
    "BSPL",
    "BSPL+Parallax",
    "FSBL",
    "FSBL+Parallax",
]


# ============================================================
# Small utilities
# ============================================================

def extract_first_float(text: str) -> float:
    """
    Extract first float from strings like:
        '123.45'
        'Array(123.45, dtype=float64)'
        'nan'
    """
    text = str(text).strip()

    if text.lower() in {"nan", "+nan", "-nan"}:
        return np.nan

    match = re.search(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot parse float from: {text!r}")

    return float(match.group(0))


def sanitize_event_name(event: str) -> str:
    event = Path(event).name
    if event.endswith(".csv"):
        event = event[:-4]
    if event.endswith("_params.txt"):
        event = event[:-11]
    return event


def find_data_file(event: str, data_dir: Path) -> Path:
    """
    Find event CSV. First tries exact EVENT.csv, then fuzzy search.
    """
    event = sanitize_event_name(event)

    direct = data_dir / f"{event}.csv"
    if direct.exists():
        return direct

    matches = sorted(data_dir.glob(f"*{event}*.csv"))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print("Found multiple possible data files:")
        for m in matches[:20]:
            print(f"  {m}")
        raise RuntimeError(
            "Please give a more specific --event name or use --data-file."
        )

    raise FileNotFoundError(f"Could not find data file for event {event!r} in {data_dir}")


def find_result_file(event: str, results_dir: Path) -> Path:
    """
    Find result params file. First tries exact EVENT_params.txt, then fuzzy search.
    """
    event = sanitize_event_name(event)

    direct = results_dir / f"{event}_params.txt"
    if direct.exists():
        return direct

    matches = sorted(results_dir.glob(f"*{event}*_params.txt"))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print("Found multiple possible result files:")
        for m in matches[:20]:
            print(f"  {m}")
        raise RuntimeError(
            "Please give a more specific --event name or use --result-file."
        )

    raise FileNotFoundError(
        f"Could not find result file for event {event!r} in {results_dir}"
    )


# ============================================================
# Loading data/results
# ============================================================

def load_lightcurve_csv(path: Path) -> dict:
    """
    Loads original Roman challenge CSV in columns:
        bjd, mag, mag_err

    Also computes flux using the same convention as DataLoader:
        flux = 10^(-0.4 * (mag - 22))
    """
    df = pd.read_csv(path, header=None, names=["bjd", "mag", "mag_err"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("bjd")

    if len(df) == 0:
        raise RuntimeError(f"No valid rows in {path}")

    t = df["bjd"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    mag_err = df["mag_err"].to_numpy(dtype=float)

    flux = 10.0 ** (-0.4 * (mag - 22.0))
    flux_err = flux * (0.4 * np.log(10.0)) * mag_err

    finite = (
        np.isfinite(t)
        & np.isfinite(mag)
        & np.isfinite(mag_err)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
        & (flux > 0.0)
        & (flux_err > 0.0)
    )

    return {
        "t": t[finite],
        "mag": mag[finite],
        "mag_err": mag_err[finite],
        "flux": flux[finite],
        "flux_err": flux_err[finite],
    }


def load_coords(event_name: str, coord_file: Path) -> np.ndarray:
    """
    Load [ra_deg, dec_deg] from coords.csv if available.
    If missing, return [0, 0]. Non-parallax models do not care.
    """
    if not coord_file.exists():
        return np.array([0.0, 0.0], dtype=float)

    df = pd.read_csv(coord_file)
    event_name = sanitize_event_name(event_name)

    if "name" not in df.columns:
        return np.array([0.0, 0.0], dtype=float)

    row = df[df["name"].astype(str) == event_name]
    if len(row) == 0:
        # Sometimes the data filename has filter suffixes etc.
        row = df[df["name"].astype(str).str.contains(event_name, regex=False)]

    if len(row) == 0:
        return np.array([0.0, 0.0], dtype=float)

    r = row.iloc[0]
    return np.array([float(r["ra_deg"]), float(r["dec_deg"])], dtype=float)


def parse_params_file(path: Path) -> dict:
    """
    Parse results/optax_results/EVENT_params.txt.

    Expected format:
        [PSPL]
        t_0: ...
        t_E: ...
        ...
        Chi2: ...
        Fs: ...
        Fb: ...
    """
    results = {}
    current_model = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_model = line[1:-1]
                results[current_model] = {}
                continue

            if current_model is None:
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                results[current_model][key] = extract_first_float(value)
            except ValueError:
                results[current_model][key] = value

    return results


# ============================================================
# Building model dictionaries
# ============================================================

def attach_hidden_parallax_fields(params_by_model: dict, event_name: str, coords: np.ndarray) -> None:
    """
    Param files contain fitted parameters, Fs, Fb, Chi2 etc.,
    but not hidden bookkeeping fields used by parallax:
        t_0_par
        coords

    Reconstruct them here.
    """
    # PSPL+Parallax was fitted with t_0_par from PSPL.
    if "PSPL+Parallax" in params_by_model:
        if "PSPL" in params_by_model and "t_0" in params_by_model["PSPL"]:
            params_by_model["PSPL+Parallax"]["t_0_par"] = params_by_model["PSPL"]["t_0"]
        else:
            params_by_model["PSPL+Parallax"]["t_0_par"] = params_by_model["PSPL+Parallax"].get("t_0", 0.0)
        params_by_model["PSPL+Parallax"]["coords"] = coords

    # FSPL+Parallax was fitted with t_0_par from FSPL.
    if "FSPL+Parallax" in params_by_model:
        if "FSPL" in params_by_model and "t_0" in params_by_model["FSPL"]:
            params_by_model["FSPL+Parallax"]["t_0_par"] = params_by_model["FSPL"]["t_0"]
        else:
            params_by_model["FSPL+Parallax"]["t_0_par"] = params_by_model["FSPL+Parallax"].get("t_0", 0.0)
        params_by_model["FSPL+Parallax"]["coords"] = coords

    # BSPL+Parallax was fitted with t_0_par from PSPL.
    if "BSPL+Parallax" in params_by_model:
        if "PSPL" in params_by_model and "t_0" in params_by_model["PSPL"]:
            params_by_model["BSPL+Parallax"]["t_0_par"] = params_by_model["PSPL"]["t_0"]
        else:
            params_by_model["BSPL+Parallax"]["t_0_par"] = params_by_model["BSPL+Parallax"].get("t_0_1", 0.0)
        params_by_model["BSPL+Parallax"]["coords"] = coords

    # FSBL+Parallax was fitted with t_0_par from FSBL.
    if "FSBL+Parallax" in params_by_model:
        if "FSBL" in params_by_model and "t_0" in params_by_model["FSBL"]:
            params_by_model["FSBL+Parallax"]["t_0_par"] = params_by_model["FSBL"]["t_0"]
        else:
            params_by_model["FSBL+Parallax"]["t_0_par"] = params_by_model["FSBL+Parallax"].get("t_0", 0.0)
        params_by_model["FSBL+Parallax"]["coords"] = coords


def build_magnification_params(model_name: str, parsed: dict) -> dict:
    """
    Convert section from params file into dictionary accepted by magnification().
    """
    if model_name not in MODEL_TO_INTERNAL:
        raise ValueError(f"Unknown model name: {model_name}")

    d = dict(parsed)
    d["model"] = MODEL_TO_INTERNAL[model_name]

    # Remove non-physical bookkeeping keys if present.
    for key in [
        "Chi2",
        "chi2/dof",
        "Fs",
        "Fb",
        "n_starts",
        "best_start_idx",
        "best_objective_2lnpost",
        "eval_n_points",
    ]:
        d.pop(key, None)

    if "coords" in d:
        d["coords"] = jnp.asarray(d["coords"], dtype=jnp.float64)

    return d


# ============================================================
# Model evaluation
# ============================================================

def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    return 22.0 - 2.5 * np.log10(np.maximum(flux, 1e-300))


def evaluate_model_flux(t: np.ndarray, model_name: str, section: dict) -> np.ndarray:
    """
    Evaluate fitted flux:
        F_model(t) = Fs * A(t) + Fb
    """
    if "Fs" not in section or "Fb" not in section:
        raise ValueError(f"Model {model_name} has no Fs/Fb in params file.")

    params = build_magnification_params(model_name, section)

    t_jax = jnp.asarray(t, dtype=jnp.float64)
    A = magnification(t_jax, params)
    A = np.asarray(A, dtype=float)

    Fs = float(section["Fs"])
    Fb = float(section["Fb"])

    return Fs * A + Fb


# ============================================================
# Plotting
# ============================================================

def plot_event_fit(
    *,
    event_name: str,
    lc: dict,
    params_by_model: dict,
    models_to_plot: list[str],
    y_mode: str,
    n_grid: int,
    show_residuals: bool,
    save_path: Path | None,
    title: str | None,
):
    t = lc["t"]

    # Use exactly the observed data times for model evaluation.
    # This is faster and avoids drawing an artificial smooth curve,
    # especially important for FSBL/microJAX.
    t_model = t.copy()

    if show_residuals:
        fig, (ax, ax_res) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax_res = None

    # Plot data.
    if y_mode == "mag":
        ax.errorbar(
            lc["t"],
            lc["mag"],
            yerr=lc["mag_err"],
            fmt=".",
            ms=3,
            alpha=0.45,
            capsize=0,
            label="data",
        )
        ax.set_ylabel("Magnitude")
        ax.invert_yaxis()
    else:
        ax.errorbar(
            lc["t"],
            lc["flux"],
            yerr=lc["flux_err"],
            fmt=".",
            ms=3,
            alpha=0.45,
            capsize=0,
            label="data",
        )
        ax.set_ylabel("Flux, ZP=22")

    # Plot models at exactly the data time points.
    for model_name in models_to_plot:
        if model_name not in params_by_model:
            print(f"Skipping {model_name}: not found in params file.")
            continue

        section = params_by_model[model_name]

        try:
            model_flux_data = evaluate_model_flux(t_model, model_name, section)
        except Exception as exc:
            print(f"Skipping {model_name}: could not evaluate model: {exc}")
            continue

        if y_mode == "mag":
            y_model = flux_to_mag(model_flux_data)
        else:
            y_model = model_flux_data

        chi2 = section.get("Chi2", np.nan)
        chi2dof = section.get("chi2/dof", np.nan)

        label = f"{model_name}"
        if np.isfinite(chi2dof):
            label += f"  χ²/dof={chi2dof:.3g}"

        # Sort by time before plotting the line.
        order = np.argsort(t_model)
        ax.plot(
            t_model[order],
            y_model[order],
            lw=1.8,
            marker=".",
            ms=2,
            alpha=0.9,
            label=label,
        )

        if show_residuals and ax_res is not None:
            if y_mode == "mag":
                residual = lc["mag"] - y_model
                residual_err = lc["mag_err"]
                ax_res.set_ylabel("Data - model [mag]")
            else:
                residual = lc["flux"] - y_model
                residual_err = lc["flux_err"]
                ax_res.set_ylabel("Data - model [flux]")

            ax_res.axhline(0.0, lw=1)
            ax_res.errorbar(
                lc["t"],
                residual,
                yerr=residual_err,
                fmt=".",
                ms=3,
                alpha=0.45,
                capsize=0,
                label=model_name,
            )

    ax.set_title(title or f"Microlensing fit: {event_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    if ax_res is not None:
        ax_res.set_xlabel("BJD")
        ax_res.grid(alpha=0.25)
    else:
        ax.set_xlabel("BJD")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    plt.show()

# ============================================================
# CLI
# ============================================================

def parse_model_list(text: str, available: dict) -> list[str]:
    if text.lower() == "all":
        return [m for m in DEFAULT_MODEL_ORDER if m in available]

    requested = [m.strip() for m in text.split(",") if m.strip()]

    # Allow some lowercase aliases.
    alias = {
        "pspl": "PSPL",
        "parallax": "PSPL+Parallax",
        "pspl+parallax": "PSPL+Parallax",
        "fspl": "FSPL",
        "fspl+parallax": "FSPL+Parallax",
        "bspl": "BSPL",
        "bspl+parallax": "BSPL+Parallax",
        "fsbl": "FSBL",
        "fsbl+parallax": "FSBL+Parallax",
    }

    out = []
    for m in requested:
        canonical = alias.get(m.lower(), m)
        if canonical not in available:
            print(f"Warning: requested model {canonical!r} not found in results.")
            continue
        out.append(canonical)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot Roman microlensing event data and fitted model curves."
    )

    parser.add_argument(
        "--event",
        required=False,
        help="Event name, e.g. ulwdc1_1234. Can be given with or without .csv.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Direct path to event CSV. Overrides --event lookup.",
    )
    parser.add_argument(
        "--result-file",
        default=None,
        help="Direct path to EVENT_params.txt. Overrides --event lookup.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/data_F146",
        help="Directory with event CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/optax_results",
        help="Directory with *_params.txt files.",
    )
    parser.add_argument(
        "--coord-file",
        default="data/coords.csv",
        help="Coordinates CSV with columns name, ra_deg, dec_deg.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Models to plot: all, or comma-separated list, e.g. "
            "PSPL,PSPL+Parallax,BSPL,FSBL"
        ),
    )
    parser.add_argument(
        "--y",
        choices=["mag", "flux"],
        default="mag",
        help="Plot in magnitudes or flux. Default: mag.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=800,
        help="Number of model points for smooth curves. FSBL can be slow; reduce if needed.",
    )
    parser.add_argument(
        "--no-residuals",
        action="store_true",
        help="Disable residual panel.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path, e.g. plots/event_fit.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    coord_file = Path(args.coord_file)

    if args.data_file is not None:
        data_file = Path(args.data_file)
        event_name = sanitize_event_name(data_file.name)
    else:
        if args.event is None:
            raise RuntimeError("Give either --event or --data-file.")
        data_file = find_data_file(args.event, data_dir)
        event_name = sanitize_event_name(data_file.name)

    if args.result_file is not None:
        result_file = Path(args.result_file)
    else:
        result_file = find_result_file(event_name, results_dir)

    print(f"Data file:   {data_file}")
    print(f"Result file: {result_file}")

    lc = load_lightcurve_csv(data_file)
    params_by_model = parse_params_file(result_file)

    coords = load_coords(event_name, coord_file)
    attach_hidden_parallax_fields(params_by_model, event_name, coords)

    models_to_plot = parse_model_list(args.models, params_by_model)

    if len(models_to_plot) == 0:
        raise RuntimeError("No valid models selected/found.")

    print("Models to plot:")
    for m in models_to_plot:
        chi2 = params_by_model[m].get("Chi2", np.nan)
        chi2dof = params_by_model[m].get("chi2/dof", np.nan)
        print(f"  {m:16s} chi2={chi2:.6g}, chi2/dof={chi2dof:.6g}")

    save_path = Path(args.save) if args.save else None

    plot_event_fit(
        event_name=event_name,
        lc=lc,
        params_by_model=params_by_model,
        models_to_plot=models_to_plot,
        y_mode=args.y,
        n_grid=args.n_grid,
        show_residuals=not args.no_residuals,
        save_path=save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()