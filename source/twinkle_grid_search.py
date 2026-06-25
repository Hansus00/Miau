"""Twinkle-based global FSBL grid search.

This module is intentionally NumPy/Twinkle-only.  It is meant to run on the GPU
and produce good binary-lens starting points.  It does not do posterior
inference.  The CPU posterior stage is handled by multinest_cpu.py in a separate
process.
"""
from __future__ import annotations

import csv
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from event_focus import FocusConfig, select_focus_indices


@dataclass
class TwinkleGridConfig:
    topn: int = 128
    max_points: int = 512
    max_evals: int = 0
    alpha_n: int = 24
    local_rounds: int = 1
    local_n: int = 10000
    final_topn: int = 32
    device_num: int = 0
    n_stream: int = 1
    reltol: float = 1e-4
    astrometry: bool = False
    final_use_focus: bool = True
    random_seed: int = 12345

    @classmethod
    def from_env(cls) -> "TwinkleGridConfig":
        return cls(
            topn=int(os.environ.get("TWINKLE_TOPN", "128")),
            max_points=int(os.environ.get("TWINKLE_MAX_POINTS", "512")),
            max_evals=int(os.environ.get("TWINKLE_MAX_EVALS", "0")),
            alpha_n=int(os.environ.get("TWINKLE_ALPHA_N", "24")),
            local_rounds=int(os.environ.get("TWINKLE_LOCAL_ROUNDS", "1")),
            local_n=int(os.environ.get("TWINKLE_LOCAL_N", "10000")),
            final_topn=int(os.environ.get("TWINKLE_FINAL_TOPN", "32")),
            device_num=int(os.environ.get("TWINKLE_DEVICE", "0")),
            n_stream=int(os.environ.get("TWINKLE_N_STREAM", "1")),
            reltol=float(os.environ.get("TWINKLE_RELTOL", "1e-4")),
            astrometry=os.environ.get("TWINKLE_ASTROMETRY", "0") == "1",
            final_use_focus=os.environ.get("TWINKLE_FINAL_USE_FOCUS", "1") == "1",
            random_seed=int(os.environ.get("TWINKLE_RANDOM_SEED", "12345")),
        )


def _parse_float_list(name: str, default: str) -> np.ndarray:
    text = os.environ.get(name, default)
    return np.asarray([float(x) for x in text.split(",") if x.strip()], dtype=float)


def import_twinkle_module():
    extra_dir = os.environ.get("TWINKLE_PYTHON_DIR")
    if extra_dir:
        sys.path.insert(0, extra_dir)
    twinkle = importlib.import_module("twinkle")
    if not hasattr(twinkle, "Twinkle"):
        raise AttributeError(
            "Imported module 'twinkle' has no Twinkle class. Set TWINKLE_PYTHON_DIR "
            "to the compiled AsterLight0626/Twinkle/python directory. Imported from: "
            f"{getattr(twinkle, '__file__', '<unknown>')}"
        )
    return twinkle


def make_twinkle_engine(twinkle, n_srcs: int, cfg: TwinkleGridConfig):
    try:
        return twinkle.Twinkle(n_srcs, cfg.device_num, cfg.n_stream, cfg.reltol, cfg.astrometry)
    except TypeError:
        return twinkle.Twinkle(n_srcs, cfg.device_num, cfg.n_stream, cfg.reltol)


def trajectory_xy(t: np.ndarray, t0: float, tE: float, u0: float, alpha_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    tau = (t - t0) / tE
    alpha = np.deg2rad(alpha_deg)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    # Same convention as the microJAX wrapper used in previous versions.
    x = tau * ca - u0 * sa
    y = tau * sa + u0 * ca
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def weighted_linear_flux_fit(A: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Tuple[float, float, float]:
    A = np.asarray(A, dtype=float)
    y = np.asarray(flux, dtype=float)
    sig = np.asarray(flux_err, dtype=float)
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
    if not np.isfinite(det) or abs(det) < 1e-300:
        return np.nan, np.nan, np.inf
    Fs = (SAy * S - Sy * SA) / det
    Fb = (SAA * Sy - SA * SAy) / det
    chi2 = np.sum(w * (y - Fs * A - Fb) ** 2)
    return float(Fs), float(Fb), float(chi2)


class TopKeeper:
    def __init__(self, n: int):
        self.n = max(1, int(n))
        self.rows: List[dict] = []

    def add(self, row: dict):
        if not np.isfinite(row.get("chi2", np.inf)):
            return
        self.rows.append(row)
        if len(self.rows) > 4 * self.n:
            self.rows = sorted(self.rows, key=lambda r: r["chi2"])[: self.n]

    def best(self) -> List[dict]:
        return sorted(self.rows, key=lambda r: r["chi2"])[: self.n]


def _estimate_base(data: dict) -> dict:
    t = np.asarray(data["t"], dtype=float)
    f = np.asarray(data["mag"], dtype=float)
    e = np.asarray(data["mag_err"], dtype=float)
    good = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t = t[good]
    f = f[good]
    e = e[good]
    baseline = np.nanmedian(f)
    snr = (f - baseline) / e
    peak = int(np.nanargmax(snr))
    t0 = float(t[peak])
    # Width proxy: time span above 20% of max positive signal.
    sig = f - baseline
    amp = max(float(np.nanmax(sig)), 0.0)
    if amp > 0:
        m = sig > 0.2 * amp
        if m.sum() >= 2:
            width = float(np.nanmax(t[m]) - np.nanmin(t[m]))
        else:
            width = 5.0
    else:
        width = 5.0
    tE = max(width, 0.5)
    return {"t0": t0, "tE": tE, "u0": 0.2}


def _t0_centers(data_focus: dict, base: dict) -> np.ndarray:
    t = np.asarray(data_focus["t"], dtype=float)
    f = np.asarray(data_focus["mag"], dtype=float)
    e = np.asarray(data_focus["mag_err"], dtype=float)
    baseline = np.nanmedian(f)
    snr = (f - baseline) / e
    idx = np.argsort(snr)[::-1]
    n_centers = int(os.environ.get("TWINKLE_T0_CENTERS", "7"))
    centers = [base["t0"]]
    for i in idx[: max(n_centers * 4, n_centers)]:
        ti = float(t[i])
        if all(abs(ti - c) > 0.1 * max(base["tE"], 1e-3) for c in centers):
            centers.append(ti)
        if len(centers) >= n_centers:
            break
    return np.asarray(centers, dtype=float)


def candidate_generator(data_focus: dict, base: dict, cfg: TwinkleGridConfig) -> Iterable[dict]:
    tE_base = max(float(base["tE"]), 1e-3)
    t0_offsets = _parse_float_list("TWINKLE_T0_FRAC_GRID", "-0.5,-0.25,0,0.25,0.5")
    tE_factors = _parse_float_list("TWINKLE_TE_FACTOR_GRID", "0.4,0.7,1.0,1.6,2.5")
    u0_grid = _parse_float_list("TWINKLE_U0_GRID", "-1.5,-1.0,-0.6,-0.3,-0.1,0.1,0.3,0.6,1.0,1.5")
    s_grid = _parse_float_list("TWINKLE_S_GRID", "0.2,0.3,0.5,0.7,0.9,1.0,1.1,1.4,2.0,3.0,5.0")
    q_grid = _parse_float_list("TWINKLE_Q_GRID", "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0")
    rho_grid = _parse_float_list("TWINKLE_RHO_GRID", "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    alpha_grid = np.linspace(0.0, 360.0, cfg.alpha_n, endpoint=False)
    centers = _t0_centers(data_focus, base)

    count = 0
    for center in centers:
        for t0_frac in t0_offsets:
            t0 = float(center + t0_frac * tE_base)
            for fac in tE_factors:
                tE = float(tE_base * fac)
                for u0 in u0_grid:
                    for s in s_grid:
                        for q in q_grid:
                            for rho in rho_grid:
                                for alpha in alpha_grid:
                                    yield {"t0": t0, "tE": tE, "u0": float(u0), "s": float(s), "q": float(q), "rho": float(rho), "alpha_deg": float(alpha)}
                                    count += 1


def evaluate_candidate(engine, t: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, cand: dict) -> dict:
    x, y = trajectory_xy(t, cand["t0"], cand["tE"], cand["u0"], cand["alpha_deg"])
    mag = np.empty(len(t), dtype=np.float64)
    engine.set_params(cand["s"], cand["q"], cand["rho"], x, y)
    engine.run()
    engine.return_mag_to(mag)
    Fs, Fb, chi2 = weighted_linear_flux_fit(mag, flux, flux_err)
    out = dict(cand)
    out.update({"Fs": Fs, "Fb": Fb, "chi2": chi2})
    return out


def random_local_candidates(best: Sequence[dict], rng: np.random.Generator, n: int) -> Iterable[dict]:
    if len(best) == 0 or n <= 0:
        return []
    # Mixture around best rows.  Perturb in log for positive parameters.
    for _ in range(n):
        b = best[int(rng.integers(0, len(best)))]
        yield {
            "t0": float(b["t0"] + rng.normal(0, 0.12 * max(b["tE"], 1e-3))),
            "tE": float(np.exp(np.log(max(b["tE"], 1e-6)) + rng.normal(0, 0.25))),
            "u0": float(b["u0"] + rng.normal(0, 0.18)),
            "s": float(np.exp(np.log(max(b["s"], 1e-6)) + rng.normal(0, 0.25))),
            "q": float(np.clip(np.exp(np.log(max(b["q"], 1e-8)) + rng.normal(0, 0.45)), 1e-7, 1.5)),
            "rho": float(np.clip(np.exp(np.log(max(b["rho"], 1e-8)) + rng.normal(0, 0.4)), 1e-6, 0.5)),
            "alpha_deg": float((b["alpha_deg"] + rng.normal(0, 15.0)) % 360.0),
        }


def save_rows(rows: Sequence[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "chi2", "Fs", "Fb", "t0", "tE", "u0", "s", "q", "rho", "alpha_deg"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, row in enumerate(sorted(rows, key=lambda r: r["chi2"]), start=1):
            rr = {k: row.get(k, "") for k in fields}
            rr["rank"] = rank
            w.writerow(rr)


def run_twinkle_grid_search(data: dict, event_name: str, out_dir: str | Path, base: Optional[dict] = None, cfg: Optional[TwinkleGridConfig] = None) -> Path:
    cfg = cfg or TwinkleGridConfig.from_env()
    out_dir = Path(out_dir)
    base = base or _estimate_base(data)

    full_t = np.asarray(data["t"], dtype=float)
    full_flux = np.asarray(data["mag"], dtype=float)
    full_err = np.asarray(data["mag_err"], dtype=float)
    good = np.isfinite(full_t) & np.isfinite(full_flux) & np.isfinite(full_err) & (full_err > 0)
    data_good = {"t": full_t[good], "mag": full_flux[good], "mag_err": full_err[good]}

    focus_cfg = FocusConfig.from_env()
    idx, meta = select_focus_indices(data_good["t"], data_good["mag"], data_good["mag_err"], focus_cfg)
    if len(idx) > cfg.max_points:
        idx = idx[: cfg.max_points]
    t = data_good["t"][idx]
    flux = data_good["mag"][idx]
    flux_err = data_good["mag_err"][idx]

    twinkle = import_twinkle_module()
    engine = make_twinkle_engine(twinkle, len(t), cfg)
    keeper = TopKeeper(cfg.topn)
    rng = np.random.default_rng(cfg.random_seed)

    print(
        f"Twinkle grid {event_name}: focus_points={len(t)}/{len(data_good['t'])}, "
        f"topn={cfg.topn}, alpha_n={cfg.alpha_n}, max_evals={cfg.max_evals}",
        flush=True,
    )

    n_eval = 0
    for cand in candidate_generator({"t": t, "mag": flux, "mag_err": flux_err}, base, cfg):
        if cfg.max_evals and n_eval >= cfg.max_evals:
            break
        try:
            row = evaluate_candidate(engine, t, flux, flux_err, cand)
            keeper.add(row)
        except Exception as exc:
            # Skip pathological candidate but keep search alive.
            if os.environ.get("TWINKLE_VERBOSE_ERRORS", "0") == "1":
                print(f"Twinkle candidate failed: {exc}", flush=True)
        n_eval += 1
        if n_eval % int(os.environ.get("TWINKLE_PROGRESS_EVERY", "10000")) == 0:
            b = keeper.best()[0] if keeper.best() else {"chi2": np.inf}
            print(f"  evaluated={n_eval}, best chi2={b['chi2']:.6g}", flush=True)

    # Local random refinement on the focus data.
    for r in range(cfg.local_rounds):
        best = keeper.best()
        print(f"Twinkle local round {r+1}/{cfg.local_rounds}, current best={best[0]['chi2']:.6g}", flush=True)
        for cand in random_local_candidates(best[: min(len(best), cfg.final_topn)], rng, cfg.local_n):
            try:
                row = evaluate_candidate(engine, t, flux, flux_err, cand)
                keeper.add(row)
            except Exception:
                pass
            n_eval += 1

    rows = keeper.best()

    # Optional final re-score on all valid points or focus points.
    if cfg.final_use_focus:
        final_t, final_flux, final_err = t, flux, flux_err
    else:
        final_t, final_flux, final_err = data_good["t"], data_good["mag"], data_good["mag_err"]
    final_engine = make_twinkle_engine(twinkle, len(final_t), cfg)
    rescored = []
    for cand in rows[: cfg.final_topn]:
        try:
            rescored.append(evaluate_candidate(final_engine, final_t, final_flux, final_err, cand))
        except Exception:
            rescored.append(cand)
    rows = sorted(rescored + rows[cfg.final_topn :], key=lambda r: r["chi2"])[: cfg.topn]

    path = out_dir / "twinkle_grid" / f"{event_name}_twinkle_top.csv"
    save_rows(rows, path)
    meta_path = out_dir / "twinkle_grid" / f"{event_name}_focus_meta.txt"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        f.write(f"event: {event_name}\n")
        f.write(f"evaluated: {n_eval}\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write(f"used_points: {len(t)}\n")
        f.write(f"base_t0: {base['t0']}\nbase_tE: {base['tE']}\n")
    print(f"Twinkle search done for {event_name}: best chi2={rows[0]['chi2']:.6g}; saved {path}", flush=True)
    return path
