"""Twinkle/GPU MultiNest refinement for FSBL+Parallax, seeded from an existing FSBL run.

Use this after you already have a directory like
    results/RMDC26_000005_multinest_FSBL_from_FSPL/best_fit.txt

The nonlinear parameters are:
    t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg, pi_E_N, pi_E_E

Parallax is implemented by changing the source trajectory before passing it to
Twinkle:
    tau  -> tau  + d_tau(t; pi_E_N, pi_E_E)
    beta -> beta + d_beta(t; pi_E_N, pi_E_E)

The ephemeris parser supports simple whitespace/CSV files with columns
    JD x y z
and basic JPL-HORIZONS vector lines containing JD plus X=, Y=, Z=.
Positions are assumed to be in AU.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Reuse the tested Twinkle/MultiNest helpers from the current project.
from multinest_twinkle import (
    import_twinkle_module,
    make_twinkle_engine,
    load_lightcurve_csv,
    weighted_linear_fit,
)

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def first_float(value: str, default: float | None = None) -> float | None:
    m = FLOAT_RE.search(str(value))
    if m is None:
        return default
    try:
        out = float(m.group(0))
    except ValueError:
        return default
    return out if np.isfinite(out) else default


def parse_key_value_file(path: str | Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            x = first_float(val)
            if x is not None:
                out[key.strip()] = float(x)
    return out


def load_fsbl_seed(fsbl_dir: str | Path) -> Dict[str, float]:
    """Load existing FSBL maximum-likelihood result from best_fit.txt."""
    fsbl_dir = Path(fsbl_dir)
    best_path = fsbl_dir / "best_fit.txt" if fsbl_dir.is_dir() else fsbl_dir
    if not best_path.exists():
        raise FileNotFoundError(f"Cannot find FSBL best_fit.txt: {best_path}")

    raw = parse_key_value_file(best_path)
    required = ["t0", "u0", "alpha_deg"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise RuntimeError(f"Missing {missing} in {best_path}")

    # Prefer physical values written at the bottom of best_fit.txt. Fall back to logs.
    tE = raw.get("tE", math.exp(raw["log_tE"]) if "log_tE" in raw else None)
    s = raw.get("s", math.exp(raw["log_s"]) if "log_s" in raw else None)
    q = raw.get("q", math.exp(raw["log_q"]) if "log_q" in raw else None)
    rho = raw.get("rho", math.exp(raw["log_rho"]) if "log_rho" in raw else None)
    if tE is None or s is None or q is None or rho is None:
        raise RuntimeError(f"Missing tE/s/q/rho or their logs in {best_path}")

    return {
        "chi2": raw.get("chi2", np.inf),
        "Fs": raw.get("Fs", 1.0),
        "Fb": raw.get("Fb", 0.0),
        "t0": raw["t0"],
        "tE": max(float(tE), 1e-6),
        "u0": raw["u0"],
        "s": max(float(s), 1e-8),
        "q": max(float(q), 1e-12),
        "rho": max(float(rho), 1e-12),
        "alpha_deg": raw["alpha_deg"],
        "pi_E_N": raw.get("pi_E_N", 0.0),
        "pi_E_E": raw.get("pi_E_E", 0.0),
    }


def infer_event_id_from_data_file(path: str | Path) -> str:
    return Path(path).stem


def load_coords(coord_file: str | Path, event_id: str) -> Tuple[float, float]:
    import pandas as pd

    df = pd.read_csv(coord_file)
    if "name" not in df.columns:
        raise RuntimeError(f"coord file {coord_file} has no 'name' column")
    row = df.loc[df["name"] == event_id]
    if row.empty:
        raise RuntimeError(f"No coordinates for {event_id} in {coord_file}")
    r0 = row.iloc[0]
    return float(r0["ra_deg"]), float(r0["dec_deg"])


def _parse_horizons_xyz_line(line: str) -> Tuple[float, float, float, float] | None:
    """Parse lines like '2450000.5 ... X= ... Y= ... Z= ...'."""
    jd_match = re.search(r"\b(24\d{5,}(?:\.\d+)?)\b", line)
    if jd_match is None:
        return None
    x_match = re.search(r"\bX\s*=\s*(%s)" % FLOAT_RE.pattern, line, flags=re.IGNORECASE)
    y_match = re.search(r"\bY\s*=\s*(%s)" % FLOAT_RE.pattern, line, flags=re.IGNORECASE)
    z_match = re.search(r"\bZ\s*=\s*(%s)" % FLOAT_RE.pattern, line, flags=re.IGNORECASE)
    if x_match and y_match and z_match:
        return float(jd_match.group(1)), float(x_match.group(1)), float(y_match.group(1)), float(z_match.group(1))
    return None


def load_ephemeris_xyz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return times JD and heliocentric observer positions [x,y,z] in AU."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("$$"):
                continue
            parsed = _parse_horizons_xyz_line(line)
            if parsed is not None:
                rows.append(parsed)
                continue
            # Generic numeric line: take the first four floats as JD, x, y, z.
            vals = [float(m.group(0)) for m in FLOAT_RE.finditer(line)]
            if len(vals) >= 4 and 2_000_000.0 < vals[0] < 3_000_000.0:
                rows.append((vals[0], vals[1], vals[2], vals[3]))

    if len(rows) < 3:
        raise RuntimeError(
            f"Could not parse enough ephemeris rows from {path}. Expected JD x y z in AU, "
            "or JPL-HORIZONS vector lines with X=, Y=, Z=."
        )

    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    # Drop duplicate times.
    _, idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[np.sort(idx)]
    return arr[:, 0], arr[:, 1:4]


def sky_basis(ra_deg: float, dec_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    n = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], dtype=float)
    east = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=float)
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], dtype=float)
    return n, east, north


def interp_position(times_jd: np.ndarray, pos_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(t, times_jd, pos_xyz[:, i]) for i in range(3)]).T


def make_parallax_projector(
    t: np.ndarray,
    *,
    ra_deg: float,
    dec_deg: float,
    ephem_t: np.ndarray,
    ephem_xyz: np.ndarray,
    t_ref: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build geocentric projected ephemeris offsets at observation times.

    We subtract position and constant velocity at t_ref, so the parallax offset is
    zero to first order at the reference epoch. This is the usual stable
    geocentric parameterization.
    """
    _, east, north = sky_basis(ra_deg, dec_deg)
    r = interp_position(ephem_t, ephem_xyz, t)
    r0 = interp_position(ephem_t, ephem_xyz, np.asarray([t_ref], dtype=float))[0]

    # Numerical velocity in AU/day near t_ref.
    dt = min(1.0, max(0.01, 0.05 * (ephem_t[-1] - ephem_t[0]) / max(len(ephem_t), 2)))
    rp = interp_position(ephem_t, ephem_xyz, np.asarray([t_ref + dt], dtype=float))[0]
    rm = interp_position(ephem_t, ephem_xyz, np.asarray([t_ref - dt], dtype=float))[0]
    v0 = (rp - rm) / (2.0 * dt)

    delta = r - (r0[None, :] + (t - t_ref)[:, None] * v0[None, :])
    d_e = delta @ east
    d_n = delta @ north
    return d_e.astype(np.float64), d_n.astype(np.float64)


def trajectory_xy_parallax(
    t: np.ndarray,
    t0: float,
    tE: float,
    u0: float,
    alpha_deg: float,
    pi_E_N: float,
    pi_E_E: float,
    d_e: np.ndarray,
    d_n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tau = (t - t0) / tE

    # Convention matching the usual north/east microlens parallax vector.
    # If you compare against a trusted PSPL+Parallax implementation and the sign
    # is flipped, change signs here rather than in the fitted parameters.
    d_tau = pi_E_N * d_n + pi_E_E * d_e
    d_beta = pi_E_N * d_e - pi_E_E * d_n

    tau_eff = tau + d_tau
    beta_eff = u0 + d_beta

    alpha = np.deg2rad(alpha_deg)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    x = tau_eff * ca - beta_eff * sa
    y = tau_eff * sa + beta_eff * ca
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def make_fsbl_parallax_mag_twinkle(engine, t: np.ndarray, d_e: np.ndarray, d_n: np.ndarray):
    t = np.asarray(t, dtype=np.float64)
    mag = np.empty(len(t), dtype=np.float64)

    def mag_fn(theta: np.ndarray) -> np.ndarray:
        t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg, pi_E_N, pi_E_E = theta
        tE = float(np.exp(log_tE))
        s = float(np.exp(log_s))
        q = float(np.exp(log_q))
        rho = float(np.exp(log_rho))
        x, y = trajectory_xy_parallax(
            t, float(t0), tE, float(u0), float(alpha_deg), float(pi_E_N), float(pi_E_E), d_e, d_n
        )
        engine.set_params(s, q, rho, x, y)
        engine.run()
        engine.return_mag_to(mag)
        return mag

    return mag_fn


def make_prior_bounds(seed: Dict[str, float]) -> np.ndarray:
    tE = max(seed["tE"], 1e-6)
    t0_w = float(os.environ.get("PLX_T0_WIDTH_TE", "0.5")) * tE
    te_lo_fac = float(os.environ.get("PLX_TE_LO_FACTOR", "0.5"))
    te_hi_fac = float(os.environ.get("PLX_TE_HI_FACTOR", "2.0"))
    u0_w = float(os.environ.get("PLX_U0_WIDTH", "1.0"))
    factor = float(os.environ.get("PLX_BINARY_FACTOR", "3.0"))
    rho_factor = float(os.environ.get("PLX_RHO_FACTOR", "10.0"))
    pi_max = float(os.environ.get("MN_PI_E_MAX", "2.0"))

    return np.asarray(
        [
            [seed["t0"] - t0_w, seed["t0"] + t0_w],
            [np.log(max(seed["tE"] * te_lo_fac, 1e-3)), np.log(max(seed["tE"] * te_hi_fac, 1.1e-3))],
            [max(0.0, seed["u0"] - u0_w), min(4.0, seed["u0"] + u0_w)],
            [np.log(max(seed["s"] / factor, 0.03)), np.log(min(seed["s"] * factor, 30.0))],
            [np.log(max(seed["q"] / factor, 1e-7)), np.log(min(seed["q"] * factor, 1.5))],
            [np.log(max(seed["rho"] / rho_factor, 1e-7)), np.log(min(seed["rho"] * rho_factor, 0.5))],
            [0.0, 360.0],
            [-pi_max, pi_max],
            [-pi_max, pi_max],
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--fsbl-dir", required=True, help="Existing FSBL MultiNest output dir containing best_fit.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--coord-file", default="data/coords.csv")
    parser.add_argument("--ephemeris-file", default="data/Roman_ephemeris_jax.txt")
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--n-live", type=int, default=int(os.environ.get("MULTINEST_N_LIVE", "200")))
    parser.add_argument("--max-iter", type=int, default=200000, help="Maximum number of MultiNest iterations/samples")
    parser.add_argument("--evidence-tolerance", type=float, default=float(os.environ.get("MULTINEST_EVIDENCE_TOL", "0.5")))
    parser.add_argument("--sampling-efficiency", type=float, default=float(os.environ.get("MULTINEST_SAMPLING_EFF", "0.3")))
    parser.add_argument("--max-points", type=int, default=int(os.environ.get("MULTINEST_MAX_POINTS", "0")))
    parser.add_argument("--twinkle-device", type=int, default=int(os.environ.get("TWINKLE_DEVICE", "0")))
    parser.add_argument("--twinkle-n-stream", type=int, default=int(os.environ.get("TWINKLE_N_STREAM", "1")))
    parser.add_argument("--twinkle-reltol", type=float, default=float(os.environ.get("TWINKLE_RELTOL", "1e-4")))
    args = parser.parse_args()

    try:
        import pymultinest
    except Exception as exc:
        raise RuntimeError("pymultinest is not installed or cannot find libmultinest.so") from exc

    lc = load_lightcurve_csv(args.data_file)
    if args.max_points and len(lc["t"]) > args.max_points:
        take = np.linspace(0, len(lc["t"]) - 1, args.max_points, dtype=int)
        lc = {k: v[take] for k, v in lc.items()}

    event_id = args.event_id or infer_event_id_from_data_file(args.data_file)
    ra_deg, dec_deg = load_coords(args.coord_file, event_id)
    eph_t, eph_xyz = load_ephemeris_xyz(args.ephemeris_file)

    seed = load_fsbl_seed(args.fsbl_dir)
    t_ref = float(seed["t0"])
    d_e, d_n = make_parallax_projector(
        lc["t"], ra_deg=ra_deg, dec_deg=dec_deg, ephem_t=eph_t, ephem_xyz=eph_xyz, t_ref=t_ref
    )

    bounds = make_prior_bounds(seed)
    ndim = 9
    names = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg", "pi_E_N", "pi_E_E"]
    t, y, e = lc["t"], lc["flux"], lc["flux_err"]

    twinkle = import_twinkle_module()
    engine = make_twinkle_engine(twinkle, len(t), args.twinkle_device, args.twinkle_n_stream, args.twinkle_reltol, False)
    fsbl_plx_mag = make_fsbl_parallax_mag_twinkle(engine, t, d_e, d_n)
    print(
        f"Using Twinkle FSBL+Parallax MultiNest: event={event_id}, points={len(t)}, "
        f"device={args.twinkle_device}, n_live={args.n_live}",
        flush=True,
    )

    def prior(cube, ndim_, nparams_):
        for i in range(ndim):
            lo, hi = bounds[i]
            cube[i] = lo + cube[i] * (hi - lo)

    parallax_prior_sigma = float(os.environ.get("MN_PARALLAX_PRIOR_SIGMA", "0.15"))

    def loglike(cube, ndim_, nparams_):
        theta = np.asarray([cube[i] for i in range(ndim)], dtype=float)
        try:
            A = fsbl_plx_mag(theta)
            Fs, Fb, chi2 = weighted_linear_fit(A, y, e)
            if not np.isfinite(chi2):
                return -1e300
            penalty = 0.0
            if Fs < 0:
                penalty += (Fs / max(np.nanmedian(y), 1e-12)) ** 2 * 100.0
            if parallax_prior_sigma > 0:
                penalty += 0.5 * (theta[7] ** 2 + theta[8] ** 2) / (parallax_prior_sigma**2)
            return float(-0.5 * chi2 - penalty)
        except Exception:
            return -1e300

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = str(out_dir / "mn_")
    with open(out_dir / "seed_used.txt", "w") as f:
        f.write("backend: twinkle_fsbl_parallax\n")
        f.write(f"event_id: {event_id}\n")
        f.write(f"ra_deg: {ra_deg}\n")
        f.write(f"dec_deg: {dec_deg}\n")
        f.write(f"ephemeris_file: {args.ephemeris_file}\n")
        f.write(f"t_0_par: {t_ref}\n")
        f.write(f"n_points: {len(t)}\n\n")
        f.write("Seed FSBL:\n")
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
    best = np.asarray(analyzer.get_best_fit()["parameters"], dtype=float)
    A = fsbl_plx_mag(best)
    Fs, Fb, chi2 = weighted_linear_fit(A, y, e)

    with open(out_dir / "best_fit.txt", "w") as f:
        f.write(f"model: FSBL+Parallax\n")
        f.write(f"chi2: {chi2}\nFs: {Fs}\nFb: {Fb}\n")
        for name, val in zip(names, best):
            f.write(f"{name}: {val}\n")
        f.write(f"tE: {np.exp(best[1])}\n")
        f.write(f"s: {np.exp(best[3])}\n")
        f.write(f"q: {np.exp(best[4])}\n")
        f.write(f"rho: {np.exp(best[5])}\n")
        f.write(f"t_0_par: {t_ref}\n")
        f.write("\nStats:\n")
        f.write(str(stats))


if __name__ == "__main__":
    main()
