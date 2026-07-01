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
    """Load an FSBL seed from either a CSV seed table or our best_fit.txt.

    Earlier versions only accepted CSV files.  The normal stage-2 workflow,
    however, uses ``--seed-file results/.../best_fit.txt``.  This parser accepts
    that text format and returns the same seed keys used by ``make_prior_bounds``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Native best_fit.txt format written by this script.
    if ":" in text and "t0" in text and ("log_tE" in text or "tE" in text):
        raw: Dict[str, float] = {}
        for line in text.splitlines():
            if not line.strip() or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            if key == "Stats":
                break
            try:
                raw[key] = _first_float(val)
            except Exception:
                pass

        def get_pos(name, log_name=None, default=None):
            if name in raw and np.isfinite(raw[name]) and raw[name] > 0:
                return float(raw[name])
            if log_name and log_name in raw and np.isfinite(raw[log_name]):
                return float(np.exp(raw[log_name]))
            return default

        seed = {
            "chi2": _finite_float(raw.get("chi2"), np.inf),
            "Fs": _finite_float(raw.get("Fs"), 1.0),
            "Fb": _finite_float(raw.get("Fb"), 0.0),
            "t0": _finite_float(raw.get("t0"), _finite_float(raw.get("t_0"))),
            "tE": get_pos("tE", "log_tE", 1.0),
            "u0": _finite_float(raw.get("u0"), _finite_float(raw.get("u_0"), 0.1)),
            "s": get_pos("s", "log_s", 1.0),
            "q": get_pos("q", "log_q", 0.1),
            "rho": get_pos("rho", "log_rho", 1.0e-3),
            "alpha_deg": _finite_float(raw.get("alpha_deg"), _finite_float(raw.get("alpha"), 90.0)),
            "seed_source": "best_fit.txt",
        }
        if seed["t0"] is None or seed["tE"] is None or seed["u0"] is None:
            raise RuntimeError(f"Could not parse finite t0/tE/u0 from seed file: {path}")
        return seed

    # CSV seed table fallback.
    rows = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            parsed = {}
            for k, v in row.items():
                if k == "rank" or v is None or str(v).strip() == "":
                    continue
                parsed[k] = float(v)
            rows.append(parsed)
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


def _finite_positive(x, default: float | None = None) -> float | None:
    try:
        y = float(x)
    except Exception:
        return default
    if not np.isfinite(y) or y <= 0.0:
        return default
    return y


def _finite_float(x, default: float | None = None) -> float | None:
    try:
        y = float(x)
    except Exception:
        return default
    if not np.isfinite(y):
        return default
    return y


def _valid_single_lens_seed(section: dict) -> bool:
    return (
        _finite_float(section.get("t_0")) is not None
        and _finite_positive(section.get("t_E")) is not None
        and _finite_float(section.get("u_0")) is not None
    )


def seed_from_pspl_fspl(params_file: str | Path, prefer: str = "FSPL") -> dict:
    """
    Construct an FSBL seed from PSPL/FSPL results.

    PSPL/FSPL constrain t0, tE, u0 and, for FSPL, rho. They do not constrain
    binary-lens topology, so s, q, alpha are broad default centers used only to
    build broad priors.

    Important robustness detail: do not use an FSPL section just because it
    exists. Short/failed events can write NaN parameters. In that case this
    function falls back to the first finite PSPL/FSPL seed instead of producing
    NaN prior bounds for MultiNest.
    """
    sections = load_model_params_file(params_file)
    prefer = prefer.upper()

    candidates = []
    if prefer in {"PSPL", "FSPL"}:
        candidates.append(prefer)
    candidates.extend(["FSPL", "PSPL"])

    base_name = None
    for name in candidates:
        if name in sections and _valid_single_lens_seed(sections[name]):
            base_name = name
            break
    if base_name is None:
        raise RuntimeError(f"No finite PSPL/FSPL seed found in params file: {params_file}")

    base = sections[base_name]
    fspl = sections.get("FSPL", {})

    t0 = _finite_float(base.get("t_0"))
    tE = _finite_positive(base.get("t_E"))
    u0 = _finite_float(base.get("u_0"), 0.1)
    if t0 is None or tE is None or u0 is None:
        raise RuntimeError(f"Internal error: selected non-finite seed {base_name} from {params_file}")

    rho = _finite_positive(fspl.get("rho"), 1.0e-3)
    rho = min(max(float(rho), 1.0e-5), 0.1)

    return {
        "chi2": _finite_float(base.get("Chi2"), np.inf),
        "Fs": _finite_float(base.get("Fs"), 1.0),
        "Fb": _finite_float(base.get("Fb"), 0.0),
        "t0": float(t0),
        "tE": max(float(tE), 1.0e-3),
        "u0": float(u0),
        "s": float(os.environ.get("MN_INIT_S", "1.0")),
        "q": float(os.environ.get("MN_INIT_Q", "0.1")),
        "rho": float(rho),
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


def make_fsbl_mag_twinkle_batched(
    twinkle,
    t: np.ndarray,
    *,
    batch_size: int,
    device_num: int = 0,
    n_stream: int = 1,
    reltol: float = 1e-4,
    astrometry: bool = False,
):
    """Return a Twinkle magnification function that evaluates points in batches.

    This keeps the likelihood on many selected points, e.g. 15000, but never
    creates a single huge Twinkle engine.  Instead it reuses one engine per
    batch length, normally one engine for ``batch_size`` and one for the final
    shorter batch.
    """
    t = np.asarray(t, dtype=np.float64)
    n_points = len(t)
    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size >= n_points:
        engine = make_twinkle_engine(
            twinkle,
            n_srcs=n_points,
            device_num=device_num,
            n_stream=n_stream,
            reltol=reltol,
            astrometry=astrometry,
        )
        return make_fsbl_mag_twinkle(engine, t), {
            "batch_size": n_points,
            "n_batches": 1,
            "unique_engine_sizes": [n_points],
        }

    slices = [slice(i, min(i + batch_size, n_points)) for i in range(0, n_points, batch_size)]
    engines: Dict[int, object] = {}
    mag_buffers: Dict[int, np.ndarray] = {}
    out = np.empty(n_points, dtype=np.float64)

    def _get_engine(n_srcs: int):
        if n_srcs not in engines:
            engines[n_srcs] = make_twinkle_engine(
                twinkle,
                n_srcs=n_srcs,
                device_num=device_num,
                n_stream=n_stream,
                reltol=reltol,
                astrometry=astrometry,
            )
            mag_buffers[n_srcs] = np.empty(n_srcs, dtype=np.float64)
        return engines[n_srcs], mag_buffers[n_srcs]

    def fsbl_mag_twinkle_batched(theta: np.ndarray) -> np.ndarray:
        t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg = theta
        tE = float(np.exp(log_tE))
        s = float(np.exp(log_s))
        q = float(np.exp(log_q))
        rho = float(np.exp(log_rho))
        x, y = trajectory_xy(t, float(t0), tE, float(u0), float(alpha_deg))
        for sl in slices:
            n_srcs = sl.stop - sl.start
            engine, mag = _get_engine(n_srcs)
            engine.set_params(s, q, rho, x[sl], y[sl])
            engine.run()
            engine.return_mag_to(mag)
            out[sl] = mag
        return out

    info = {
        "batch_size": int(batch_size),
        "n_batches": len(slices),
        "unique_engine_sizes": sorted({sl.stop - sl.start for sl in slices}),
    }
    return fsbl_mag_twinkle_batched, info


def _pspl_window_indices(lc: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, object]]:
    """Use the same time-window logic as PSPL/FSPL in source/run.py.

    The pipeline PSPL stage constructs InitialConditions(data) and then fits
    only points satisfying start_boundary <= t <= end_boundary.  This helper
    imports and calls the same InitialConditions class.  Therefore, if you
    change the PSPL crop logic in source/initial_conditions.py, MultiNest FSBL
    gets the same crop automatically.
    """
    try:
        import jax.numpy as jnp
        from initial_conditions import InitialConditions

        data = {
            "t": jnp.asarray(lc["t"], dtype=jnp.float64),
            # In the main pipeline this key is called "mag", but it is already
            # flux.  Use the same convention here.
            "mag": jnp.asarray(lc["flux"], dtype=jnp.float64),
            "mag_err": jnp.asarray(lc["flux_err"], dtype=jnp.float64),
        }
        init = InitialConditions(data)
        start = float(init.start_boundary)
        end = float(init.end_boundary)
        peak = float(init.main_peak_time)
        baseline = float(init.baseline)
        meta: Dict[str, object] = {}
    except Exception as exc:
        # Last-resort fallback: do not crash if this file is run standalone.
        t = np.asarray(lc["t"], dtype=float)
        flux = np.asarray(lc["flux"], dtype=float)
        peak = float(t[int(np.nanargmax(flux))])
        start = peak - 10.0
        end = peak + 10.0
        baseline = float(np.nanmedian(flux))
        meta = {"pspl_window_error": repr(exc)}

    t = np.asarray(lc["t"], dtype=float)
    idx = np.where(np.isfinite(t) & (t >= start) & (t <= end))[0]

    # Safety fallback for extremely short/pathological events.
    if len(idx) == 0:
        flux = np.asarray(lc["flux"], dtype=float)
        peak = float(t[int(np.nanargmax(flux))])
        start = max(float(np.nanmin(t)), peak - 10.0)
        end = min(float(np.nanmax(t)), peak + 10.0)
        idx = np.where(np.isfinite(t) & (t >= start) & (t <= end))[0]
        meta["pspl_window_empty_fallback"] = True

    meta.update(
        {
            "pspl_start_boundary": start,
            "pspl_end_boundary": end,
            "pspl_main_peak_time": peak,
            "pspl_baseline": baseline,
            "pspl_window_points": int(len(idx)),
        }
    )
    return idx[np.argsort(t[idx])], meta


def select_multinest_subset(lc: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Select exactly the same time window as PSPL/FSPL, with no thinning.

    This intentionally has no ``max_points`` cap, no uniform fill/cap, and no
    event_focus mode.  The only point selection is the PSPL/FSPL
    InitialConditions time window from source/initial_conditions.py.
    """
    n_full = len(lc["t"])
    t = np.asarray(lc["t"], dtype=float)

    idx, meta = _pspl_window_indices(lc)
    idx = np.asarray(idx, dtype=int)

    sub = {k: np.asarray(v)[idx] for k, v in lc.items()}
    meta.update(
        {
            "method": "pspl_window_no_thinning",
            "n_input": int(n_full),
            "n_selected": int(len(idx)),
            "thinning": False,
            "max_points_cap": None,
            "selected_t_min": float(np.nanmin(t[idx])) if len(idx) else None,
            "selected_t_max": float(np.nanmax(t[idx])) if len(idx) else None,
        }
    )
    return sub, meta

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


def _remove_old_multinest_outputs(out_dir: Path) -> None:
    """Remove stale PyMultiNest files before a fresh run.

    MultiNest can resume from old ``mn_*`` files. That is dangerous while
    changing batching or priors because the old samples were generated for a
    different likelihood.
    """
    for p in out_dir.glob("mn_*"):
        if p.is_file():
            p.unlink()
    for name in ["best_fit.txt", "seed_used.txt"]:
        p = out_dir / name
        if p.exists() and p.is_file():
            p.unlink()


def _theta_from_seed(seed: dict) -> np.ndarray:
    return np.asarray(
        [
            seed["t0"],
            np.log(max(seed["tE"], 1.0e-6)),
            seed["u0"],
            np.log(max(seed.get("s", 1.0), 1.0e-6)),
            np.log(max(seed.get("q", 0.1), 1.0e-8)),
            np.log(max(seed.get("rho", 1.0e-3), 1.0e-8)),
            seed.get("alpha_deg", 90.0),
        ],
        dtype=float,
    )


def check_batching_consistency(twinkle, t: np.ndarray, seed: dict, *, device_num: int, n_stream: int, reltol: float, astrometry: bool) -> None:
    """Quick sanity check that batched and unbatched Twinkle agree.

    This uses only a small prefix of the selected light curve, so it should not
    be expensive. It verifies the batching wrapper, not the scientific fit.
    """
    n = min(512, len(t))
    if n < 4:
        print("Batching check skipped: too few points.", flush=True)
        return
    t_check = np.asarray(t[:n], dtype=np.float64)
    theta = _theta_from_seed(seed)
    single_engine = make_twinkle_engine(twinkle, n, device_num=device_num, n_stream=n_stream, reltol=reltol, astrometry=astrometry)
    mag_single = make_fsbl_mag_twinkle(single_engine, t_check)(theta).copy()
    mag_batch, info = make_fsbl_mag_twinkle_batched(
        twinkle,
        t_check,
        batch_size=max(1, min(128, n // 2)),
        device_num=device_num,
        n_stream=n_stream,
        reltol=reltol,
        astrometry=astrometry,
    )
    mag_batched = mag_batch(theta).copy()
    diff = np.nanmax(np.abs(mag_single - mag_batched))
    rel = diff / max(np.nanmax(np.abs(mag_single)), 1.0)
    print(
        f"Batching self-check: n={n}, info={info}, max_abs_diff={diff:.3e}, rel={rel:.3e}",
        flush=True,
    )
    if not np.isfinite(diff) or rel > 1.0e-7:
        raise RuntimeError("Batched Twinkle magnification does not match single-engine magnification.")


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("TWINKLE_BATCH_SIZE", os.environ.get("MULTINEST_BATCH_SIZE", "2048"))),
        help=(
            "Evaluate Twinkle likelihood in batches of this many points; 0 disables batching. "
            "This is not point thinning: all points from the PSPL/FSPL window are still used."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing MultiNest run. Default is a fresh run that removes stale mn_* files.")
    parser.add_argument("--no-multimodal", action="store_true", help="Disable MultiNest multimodal mode. Default keeps multimodal mode on.")
    parser.add_argument("--n-clustering-params", type=int, default=int(os.environ.get("MULTINEST_N_CLUSTERING_PARAMS", "7")), help="Number of parameters used for MultiNest clustering; 7 means all FSBL parameters.")
    parser.add_argument("--mode-tolerance", type=float, default=float(os.environ.get("MULTINEST_MODE_TOL", "-1e90")), help="MultiNest mode_tolerance passed through to PyMultiNest.")
    parser.add_argument("--check-batching", action="store_true", help="Before MultiNest, compare batched vs unbatched Twinkle magnification on a small sample.")
    parser.add_argument("--max-iter", type=int, default=200000, help="Maximum number of MultiNest iterations/samples")
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

    lc_full = load_lightcurve_csv(args.data_file)
    lc, selection_meta = select_multinest_subset(lc_full)
    if selection_meta.get("method") != "full":
        print(
            "Point selection for MultiNest FSBL: "
            f"method={selection_meta.get('method')}, "
            f"input={selection_meta.get('n_input')}, "
            f"selected={selection_meta.get('n_selected')}",
            flush=True,
        )

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
    if args.check_batching:
        check_batching_consistency(
            twinkle,
            t,
            seed,
            device_num=args.twinkle_device,
            n_stream=args.twinkle_n_stream,
            reltol=args.twinkle_reltol,
            astrometry=args.twinkle_astrometry,
        )
    fsbl_mag, batch_info = make_fsbl_mag_twinkle_batched(
        twinkle,
        t,
        batch_size=int(args.batch_size),
        device_num=args.twinkle_device,
        n_stream=args.twinkle_n_stream,
        reltol=args.twinkle_reltol,
        astrometry=args.twinkle_astrometry,
    )
    print(
        f"Using Twinkle MultiNest likelihood: points={len(t)}, device={args.twinkle_device}, "
        f"reltol={args.twinkle_reltol}, n_live={args.n_live}, "
        f"batch_size={batch_info.get('batch_size')}, n_batches={batch_info.get('n_batches')}",
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
    if not args.resume:
        _remove_old_multinest_outputs(out_dir)
    basename = str(out_dir / "mn_")
    names = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg"]
    with open(out_dir / "seed_used.txt", "w") as f:
        f.write("backend: twinkle\n")
        f.write(f"twinkle_python_dir: {os.environ.get('TWINKLE_PYTHON_DIR', '')}\n")
        f.write(f"twinkle_device: {args.twinkle_device}\n")
        f.write(f"twinkle_reltol: {args.twinkle_reltol}\n")
        f.write(f"n_points: {len(t)}\n")
        f.write(f"n_points_full: {len(lc_full['t'])}\n")
        f.write("selection_mode: pspl-window-no-thinning\n")
        f.write(f"selection_meta: {selection_meta}\n")
        f.write(f"batch_info: {batch_info}\n")
        f.write(f"resume: {args.resume}\n")
        f.write(f"multimodal: {not args.no_multimodal}\n")
        f.write(f"n_clustering_params: {args.n_clustering_params}\n")
        f.write("wrapped_params: alpha_deg\n\n")
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
        resume=bool(args.resume),
        verbose=True,
        n_live_points=args.n_live,
        evidence_tolerance=args.evidence_tolerance,
        sampling_efficiency=args.sampling_efficiency,
        max_iter=args.max_iter,
        multimodal=not bool(args.no_multimodal),
        n_clustering_params=max(1, min(int(args.n_clustering_params), ndim)),
        wrapped_params=[6],
        mode_tolerance=float(args.mode_tolerance),
    )

    analyzer = pymultinest.Analyzer(n_params=ndim, outputfiles_basename=basename)
    stats = analyzer.get_stats()
    best = analyzer.get_best_fit()["parameters"]

    # Search chi2 is computed on the selected subset used by MultiNest.
    # For submission/AIC comparisons against PSPL/FSPL, recompute the final
    # best-fit fluxes and chi2 on the FULL light curve.  Otherwise FSBL would
    # be compared using e.g. 8000 points while simple models use all 46208.
    A_search = fsbl_mag(np.asarray(best, dtype=float))
    Fs_search, Fb_search, chi2_search = weighted_linear_fit(A_search, y, e)

    full_mag, full_batch_info = make_fsbl_mag_twinkle_batched(
        twinkle,
        lc_full["t"],
        batch_size=int(os.environ.get("TWINKLE_FULL_BATCH_SIZE", args.batch_size)),
        device_num=args.twinkle_device,
        n_stream=args.twinkle_n_stream,
        reltol=args.twinkle_reltol,
        astrometry=args.twinkle_astrometry,
    )
    A_full = full_mag(np.asarray(best, dtype=float))
    Fs, Fb, chi2 = weighted_linear_fit(A_full, lc_full["flux"], lc_full["flux_err"])

    with open(out_dir / "best_fit.txt", "w") as f:
        f.write(f"chi2: {chi2}\nFs: {Fs}\nFb: {Fb}\n")
        f.write(f"search_chi2: {chi2_search}\nsearch_Fs: {Fs_search}\nsearch_Fb: {Fb_search}\n")
        f.write(f"eval_n_points: {len(lc_full['t'])}\nsearch_n_points: {len(t)}\n")
        f.write(f"full_batch_info: {full_batch_info}\n")
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
