"""CPU MultiNest posterior/refinement for FSBL candidates.

This script is designed to be launched as a separate process while Twinkle keeps
using the GPU in the parent process.  It forces Twinkle/JAX work onto CPU before
any JAX import.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Tuple


import numpy as np
import pandas as pd

try:
    from twinkle_grid_search import (
        TwinkleGridConfig,
        import_twinkle_module,
        make_twinkle_engine,
        trajectory_xy,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Could not import Twinkle grid search helpers. Check requirements.") from exc


_TWINKLE_ENGINE = None


def _get_twinkle_engine(n_srcs: int):
    global _TWINKLE_ENGINE
    if _TWINKLE_ENGINE is None or getattr(_TWINKLE_ENGINE, "_miau_n_srcs", None) != n_srcs:
        cfg = TwinkleGridConfig.from_env()
        twinkle = import_twinkle_module()
        _TWINKLE_ENGINE = make_twinkle_engine(twinkle, n_srcs, cfg)
        _TWINKLE_ENGINE._miau_n_srcs = n_srcs
    return _TWINKLE_ENGINE


def load_lightcurve_csv(path: str | Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path, header=None, names=["bjd", "mag", "mag_err"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("bjd")
    t = df["bjd"].to_numpy(float)
    flux = 10.0 ** (-0.4 * (df["mag"].to_numpy(float) - 22.0))
    flux_err = flux * (0.4 * np.log(10.0)) * df["mag_err"].to_numpy(float)
    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    return {"t": t[good], "flux": flux[good], "flux_err": flux_err[good]}


def load_seed(path: str | Path, rank: int = 1) -> dict:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) for k, v in row.items() if k != "rank"})
    if not rows:
        raise RuntimeError(f"No rows in seed file: {path}")
    rows = sorted(rows, key=lambda x: x.get("chi2", np.inf))
    return rows[max(0, rank - 1)]


def _first_float(text: str) -> float:
    """Parse floats from params files, including Array(..., dtype=...)."""
    match = re.search(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
        str(text),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot parse float from {text!r}")
    return float(match.group(0))


def load_model_params_file(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Read EVENT_params.txt written by source/run.py."""
    sections: Dict[str, Dict[str, float]] = {}
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections[current] = {}
                continue
            if current is None or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            try:
                sections[current][key] = _first_float(val)
            except ValueError:
                pass
    return sections


def seed_from_pspl_fspl(params_file: str | Path, prefer: str = "FSPL") -> dict:
    """
    Construct an FSBL seed from PSPL/FSPL results.

    PSPL/FSPL can constrain t0, tE, u0 and, for FSPL, rho.  They do not
    constrain binary-lens topology, so s, q, alpha are deliberately broad
    default centers used only to build broad priors.
    """
    sections = load_model_params_file(params_file)
    prefer = prefer.upper()

    base_name = None
    if prefer == "FSPL" and "FSPL" in sections:
        base_name = "FSPL"
    elif "PSPL" in sections:
        base_name = "PSPL"
    elif "FSPL" in sections:
        base_name = "FSPL"
    else:
        raise RuntimeError(f"No PSPL/FSPL section found in params file: {params_file}")

    base = sections[base_name]
    pspl = sections.get("PSPL", {})
    fspl = sections.get("FSPL", {})

    def pick_float(*values: float | None, default: float) -> float:
        for value in values:
            if value is not None:
                return float(value)
        return float(default)

    t0 = pick_float(base.get("t_0"), pspl.get("t_0"), fspl.get("t_0"), default=0.0)
    tE = pick_float(base.get("t_E"), pspl.get("t_E"), fspl.get("t_E"), default=1.0)
    u0 = pick_float(base.get("u_0"), pspl.get("u_0"), fspl.get("u_0"), default=0.1)
    rho = float(fspl.get("rho", 1.0e-3))
    rho = min(max(rho, 1.0e-5), 0.1)

    return {
        "chi2": pick_float(base.get("Chi2"), default=np.inf),
        "Fs": float(base.get("Fs", 1.0)),
        "Fb": float(base.get("Fb", 0.0)),
        "t0": t0,
        "tE": max(tE, 1.0e-3),
        "u0": u0,
        "s": float(os.environ.get("MN_INIT_S", "1.0")),
        "q": float(os.environ.get("MN_INIT_Q", "0.1")),
        "rho": rho,
        "alpha_deg": float(os.environ.get("MN_INIT_ALPHA_DEG", "90.0")),
        "seed_source": base_name,
    }


def weighted_linear_fit(A, y, sig) -> Tuple[float, float, float]:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    sig = np.asarray(sig, dtype=float)
    w = np.where(np.isfinite(sig) & (sig > 0), 1.0 / sig**2, 0.0)
    good = np.isfinite(A) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if good.sum() < 3:
        return np.nan, np.nan, np.inf
    A = A[good]
    y = y[good]
    w = w[good]
    S = np.sum(w)
    SA = np.sum(w * A)
    SAA = np.sum(w * A * A)
    Sy = np.sum(w * y)
    SAy = np.sum(w * A * y)
    det = SAA * S - SA * SA
    if abs(det) < 1e-300 or not np.isfinite(det):
        return np.nan, np.nan, np.inf
    Fs = (SAy * S - Sy * SA) / det
    Fb = (SAA * Sy - SA * SAy) / det
    chi2 = np.sum(w * (y - Fs * A - Fb) ** 2)
    return float(Fs), float(Fb), float(chi2)


def fsbl_mag_cpu(t, theta):
    t0, log_tE, u0, log_s, log_q, log_rho, alpha = theta
    tt = np.asarray(t, dtype=np.float64)
    alpha_deg = float(alpha)
    x, y = trajectory_xy(tt, float(t0), float(np.exp(log_tE)), float(u0), alpha_deg)
    engine = _get_twinkle_engine(len(tt))
    mag = np.empty(len(tt), dtype=np.float64)
    engine.set_params(float(np.exp(log_s)), float(np.exp(log_q)), float(np.exp(log_rho)), x, y)
    engine.run()
    engine.return_mag_to(mag)
    return mag


def make_prior_bounds(seed: dict, t: np.ndarray, broad_binary: bool = False) -> np.ndarray:
    t_span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    tE = max(seed["tE"], 1e-3)

    if broad_binary:
        # Priors when MultiNest is seeded from PSPL/FSPL, as requested:
        #   t0    : PSPL/FSPL t0 +/- 5 tE
        #   tE    : from ~0 to 3 * tE_PSPL/FSPL
        #   u0    : positive, 0 to 4
        #   alpha : unchanged, 0 to 360 deg
        #   s, q  : unchanged broad binary-lens ranges
        #
        # tE cannot be exactly zero because MultiNest samples log(tE),
        # so MN_TE_MIN is the numerical replacement for zero.
        t0_half_width = float(os.environ.get("MN_PSPL_T0_WIDTH_TE", "5.0")) * tE
        te_min = float(os.environ.get("MN_TE_MIN", "1.0e-3"))
        te_hi_factor = float(os.environ.get("MN_PSPL_TE_HI_FACTOR", "3.0"))
        rho_center = max(float(seed.get("rho", 1e-3)), 1e-6)
        rho_hi = float(os.environ.get("MN_RHO_MAX", "0.2"))
        bounds = np.asarray(
            [
                [seed["t0"] - t0_half_width, seed["t0"] + t0_half_width],
                [np.log(te_min), np.log(max(seed["tE"] * te_hi_factor, te_min * 1.01))],
                [float(os.environ.get("MN_U0_MIN", "0.0")), float(os.environ.get("MN_U0_MAX", "4.0"))],
                [np.log(float(os.environ.get("MN_S_MIN", "0.1"))), np.log(float(os.environ.get("MN_S_MAX", "10.0")))],
                [np.log(float(os.environ.get("MN_Q_MIN", "1e-5"))), np.log(float(os.environ.get("MN_Q_MAX", "1.0")))],
                [np.log(max(rho_center / 30.0, float(os.environ.get("MN_RHO_MIN", "1e-6")))), np.log(rho_hi)],
                [0.0, 360.0],
            ],
            dtype=float,
        )
    else:
        bounds = np.asarray(
            [
                [seed["t0"] - 0.5 * tE, seed["t0"] + 0.5 * tE],
                [np.log(max(seed["tE"] / 3.0, 1e-3)), np.log(min(seed["tE"] * 3.0, max(t_span * 5, 1.0)))],
                [seed["u0"] - 1.0, seed["u0"] + 1.0],
                [np.log(max(seed["s"] / 3.0, 0.03)), np.log(min(seed["s"] * 3.0, 30.0))],
                [np.log(max(seed["q"] / 10.0, 1e-7)), np.log(min(seed["q"] * 10.0, 1.5))],
                [np.log(max(seed["rho"] / 10.0, 1e-7)), np.log(min(seed["rho"] * 10.0, 0.5))],
                [0.0, 360.0],
            ],
            dtype=float,
        )
    return bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--seed-file", default=None, help="Twinkle seed CSV. Optional if --params-file is given.")
    parser.add_argument("--params-file", default=None, help="EVENT_params.txt from source/run.py; uses PSPL/FSPL as broad FSBL prior center.")
    parser.add_argument("--prefer-single-lens", choices=["PSPL", "FSPL"], default="FSPL")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--n-live", type=int, default=int(os.environ.get("MULTINEST_N_LIVE", "300")))
    parser.add_argument("--evidence-tolerance", type=float, default=float(os.environ.get("MULTINEST_EVIDENCE_TOL", "0.5")))
    parser.add_argument("--sampling-efficiency", type=float, default=float(os.environ.get("MULTINEST_SAMPLING_EFF", "0.3")))
    parser.add_argument("--max-points", type=int, default=int(os.environ.get("MULTINEST_MAX_POINTS", "0")))
    args = parser.parse_args()

    try:
        import pymultinest
    except Exception as exc:
        raise RuntimeError(
            "pymultinest is not installed or cannot find the MultiNest shared library. "
            "Install PyMultiNest/MultiNest for posterior sampling, or run only Twinkle grid search."
        ) from exc

    lc = load_lightcurve_csv(args.data_file)
    if args.max_points and len(lc["t"]) > args.max_points:
        # deterministic thinning over time for CPU posterior smoke tests
        take = np.linspace(0, len(lc["t"]) - 1, args.max_points, dtype=int)
        lc = {k: v[take] for k, v in lc.items()}

    if args.seed_file is not None:
        seed = load_seed(args.seed_file, args.rank)
        broad_binary = False
    elif args.params_file is not None:
        seed = seed_from_pspl_fspl(args.params_file, prefer=args.prefer_single_lens)
        broad_binary = True
    else:
        raise RuntimeError("Give either --seed-file from Twinkle or --params-file from PSPL/FSPL results.")

    bounds = make_prior_bounds(seed, lc["t"], broad_binary=broad_binary)
    ndim = 7
    t = lc["t"]
    y = lc["flux"]
    e = lc["flux_err"]

    def prior(cube, ndim_, nparams_):
        for i in range(ndim):
            lo, hi = bounds[i]
            cube[i] = lo + cube[i] * (hi - lo)

    def loglike(cube, ndim_, nparams_):
        theta = np.asarray([cube[i] for i in range(ndim)], dtype=float)
        try:
            A = fsbl_mag_cpu(t, theta)
            Fs, Fb, chi2 = weighted_linear_fit(A, y, e)
            if not np.isfinite(chi2):
                return -1e300
            # Weakly reject strongly negative source flux.
            penalty = 0.0
            if Fs < 0:
                penalty += (Fs / max(np.nanmedian(y), 1e-12)) ** 2 * 100.0
            return float(-0.5 * chi2 - penalty)
        except Exception:
            return -1e300

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = str(out_dir / "mn_")
    with open(out_dir / "seed_used.txt", "w") as f:
        for k, v in seed.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBounds:\n")
        names = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg"]
        for name, (lo, hi) in zip(names, bounds):
            f.write(f"{name}: {lo} {hi}\n")

    pymultinest.run(
        loglike,
        prior,
        ndim,
        outputfiles_basename=basename,
        resume=True,
        verbose=True,
        n_live_points=args.n_live,
        evidence_tolerance=args.evidence_tolerance,
        sampling_efficiency=args.sampling_efficiency,
    )

    analyzer = pymultinest.Analyzer(n_params=ndim, outputfiles_basename=basename)
    stats = analyzer.get_stats()
    best = analyzer.get_best_fit()["parameters"]
    A = fsbl_mag_cpu(t, np.asarray(best))
    Fs, Fb, chi2 = weighted_linear_fit(A, y, e)
    names = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg"]
    with open(out_dir / "best_fit.txt", "w") as f:
        f.write(f"chi2: {chi2}\nFs: {Fs}\nFb: {Fb}\n")
        for name, val in zip(names, best):
            f.write(f"{name}: {val}\n")
        f.write(f"tE: {np.exp(best[1])}\n")
        f.write(f"s: {np.exp(best[3])}\n")
        f.write(f"q: {np.exp(best[4])}\n")
        f.write(f"rho: {np.exp(best[5])}\n")
        f.write("\nStats:\n")
        f.write(str(stats))


if __name__ == "__main__":
    main()
