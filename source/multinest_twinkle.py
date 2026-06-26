"""GPU/Twinkle MultiNest posterior/refinement for FSBL candidates.

This is a drop-in replacement for multinest_cpu.py, but it evaluates the
finite-source binary-lens magnification with AsterLight0626/Twinkle instead of
microlux/JAX. It keeps the same PSPL/FSPL-seeded priors:

  t0 = t0_PSPL/FSPL +/- 5 tE
  tE = 0..3 tE_PSPL/FSPL  (implemented in log tE with MN_TE_MIN)
  u0 = 0..4
  s, q, alpha broad

Important: do NOT set CUDA_VISIBLE_DEVICES="" for this script, because Twinkle
uses the GPU. Set TWINKLE_PYTHON_DIR to the compiled Twinkle/python directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def import_twinkle_module():
    """Import the compiled AsterLight0626/Twinkle Python module."""
    extra_dir = os.environ.get("TWINKLE_PYTHON_DIR")
    if extra_dir:
        sys.path.insert(0, extra_dir)
    try:
        twinkle = importlib.import_module("twinkle")
    except Exception as exc:
        raise RuntimeError(
            "Could not import Twinkle. Set TWINKLE_PYTHON_DIR to the compiled "
            "AsterLight0626/Twinkle/python directory, e.g. "
            "export TWINKLE_PYTHON_DIR=$HOME/Twinkle/python"
        ) from exc
    if not hasattr(twinkle, "Twinkle"):
        raise RuntimeError(
            "Imported module 'twinkle' has no Twinkle class. You may be importing "
            "a different package named twinkle. Set TWINKLE_PYTHON_DIR to the "
            "compiled AsterLight0626/Twinkle/python directory. Imported from: "
            f"{getattr(twinkle, '__file__', '<unknown>')}"
        )
    return twinkle


def make_twinkle_engine(twinkle, n_srcs: int, device_num: int = 0, n_stream: int = 1, reltol: float = 1e-4, astrometry: bool = False):
    """Construct Twinkle engine, compatible with both 4- and 5-argument APIs."""
    try:
        return twinkle.Twinkle(int(n_srcs), int(device_num), int(n_stream), float(reltol), bool(astrometry))
    except TypeError:
        return twinkle.Twinkle(int(n_srcs), int(device_num), int(n_stream), float(reltol))


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

    PSPL/FSPL constrain t0, tE, u0 and, for FSPL, rho. They do not constrain
    binary-lens topology, so s, q, alpha are broad default centers used only to
    build broad priors.
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

    t0 = float(base.get("t_0", pspl.get("t_0", fspl.get("t_0"))))
    tE = float(base.get("t_E", pspl.get("t_E", fspl.get("t_E"))))
    u0 = float(base.get("u_0", pspl.get("u_0", fspl.get("u_0", 0.1))))
    rho = float(fspl.get("rho", 1.0e-3))
    rho = min(max(rho, 1.0e-5), 0.1)

    return {
        "chi2": float(base.get("Chi2", np.inf)),
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


def trajectory_xy(t: np.ndarray, t0: float, tE: float, u0: float, alpha_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    tau = (t - t0) / tE
    alpha = np.deg2rad(alpha_deg)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    # Same convention as the Twinkle grid-search wrapper used before.
    x = tau * ca - u0 * sa
    y = tau * sa + u0 * ca
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def make_fsbl_mag_twinkle(engine, t: np.ndarray):
    """Return a fast magnification function using one persistent Twinkle engine."""
    t = np.asarray(t, dtype=np.float64)
    mag = np.empty(len(t), dtype=np.float64)

    def fsbl_mag_twinkle(theta: np.ndarray) -> np.ndarray:
        t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg = theta
        tE = float(np.exp(log_tE))
        s = float(np.exp(log_s))
        q = float(np.exp(log_q))
        rho = float(np.exp(log_rho))
        x, y = trajectory_xy(t, float(t0), tE, float(u0), float(alpha_deg))
        engine.set_params(s, q, rho, x, y)
        engine.run()
        engine.return_mag_to(mag)
        return mag

    return fsbl_mag_twinkle


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
    parser.add_argument("--twinkle-device", type=int, default=int(os.environ.get("TWINKLE_DEVICE", "0")))
    parser.add_argument("--twinkle-n-stream", type=int, default=int(os.environ.get("TWINKLE_N_STREAM", "1")))
    parser.add_argument("--twinkle-reltol", type=float, default=float(os.environ.get("TWINKLE_RELTOL", "1e-4")))
    parser.add_argument("--twinkle-astrometry", action="store_true", default=os.environ.get("TWINKLE_ASTROMETRY", "0") == "1")
    args = parser.parse_args()

    try:
        import pymultinest
    except Exception as exc:
        raise RuntimeError(
            "pymultinest is not installed or cannot find the MultiNest shared library. "
            "Install PyMultiNest/MultiNest first."
        ) from exc

    lc = load_lightcurve_csv(args.data_file)
    if args.max_points and len(lc["t"]) > args.max_points:
        # Deterministic thinning over time for fast posterior/smoke tests.
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

    twinkle = import_twinkle_module()
    engine = make_twinkle_engine(
        twinkle,
        n_srcs=len(t),
        device_num=args.twinkle_device,
        n_stream=args.twinkle_n_stream,
        reltol=args.twinkle_reltol,
        astrometry=args.twinkle_astrometry,
    )
    fsbl_mag = make_fsbl_mag_twinkle(engine, t)
    print(
        f"Using Twinkle MultiNest likelihood: points={len(t)}, device={args.twinkle_device}, "
        f"reltol={args.twinkle_reltol}, n_live={args.n_live}",
        flush=True,
    )

    def prior(cube, ndim_, nparams_):
        for i in range(ndim):
            lo, hi = bounds[i]
            cube[i] = lo + cube[i] * (hi - lo)

    def loglike(cube, ndim_, nparams_):
        theta = np.asarray([cube[i] for i in range(ndim)], dtype=float)
        try:
            A = fsbl_mag(theta)
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
    names = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg"]
    with open(out_dir / "seed_used.txt", "w") as f:
        f.write("backend: twinkle\n")
        f.write(f"twinkle_python_dir: {os.environ.get('TWINKLE_PYTHON_DIR', '')}\n")
        f.write(f"twinkle_device: {args.twinkle_device}\n")
        f.write(f"twinkle_reltol: {args.twinkle_reltol}\n")
        f.write(f"n_points: {len(t)}\n\n")
        for k, v in seed.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBounds:\n")
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
    A = fsbl_mag(np.asarray(best, dtype=float))
    Fs, Fb, chi2 = weighted_linear_fit(A, y, e)
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
