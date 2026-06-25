"""
Twinkle-based coarse/global search for finite-source binary-lens starts.

This module is meant to solve the hard part of blind binary-lens fitting:
finding the correct caustic topology and a plausible basin of attraction.
It deliberately does NOT rely on microJAX autodiff for the global search.
Instead it uses Twinkle for cheap GPU magnification evaluations, keeps the
best candidates, optionally performs local random refinement around them, and
then either:

  * returns the best Twinkle candidate directly as the FSBL result; or
  * passes the best Twinkle candidates to microJAX/optax for final refinement.

The direct-Twinkle mode is controlled in pipeline.py by FSBL_REFINE=0.  This is
usually the right mode for a broad blind search over many binary events.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import jax.numpy as jnp


@dataclass
class TwinkleGridResult:
    starts: jnp.ndarray
    scores: np.ndarray
    rows: list[dict]
    n_evaluated: int
    n_failed: int
    n_total_grid: int


@dataclass
class TwinkleFinalResult:
    params: jnp.ndarray
    row: dict
    n_evaluated: int
    n_failed: int
    eval_n_points: int


def _parse_float_grid(env_name: str, default: str) -> np.ndarray:
    """Parse comma-separated floats from an environment variable."""
    text = os.environ.get(env_name, default).strip()
    if not text:
        return np.asarray([], dtype=float)
    return np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def _alpha_grid() -> np.ndarray:
    explicit = os.environ.get("TWINKLE_ALPHA_GRID", "").strip()
    if explicit:
        return _parse_float_grid("TWINKLE_ALPHA_GRID", explicit)
    n = int(os.environ.get("TWINKLE_ALPHA_N", "16"))
    return np.linspace(0.0, 360.0, n, endpoint=False, dtype=float)


def _build_grids(base_tE: float) -> dict[str, np.ndarray]:
    """Build the broad grid axes from environment variables.

    Defaults are deliberately broader than the first prototype.  The previous
    grid stopped at q=0.1, which is dangerous for stellar binary events.  This
    default grid includes planetary, brown-dwarf, and stellar-binary q.
    """
    day_grid = os.environ.get("TWINKLE_T0_DAY_GRID", "").strip()
    if day_grid:
        t0_offsets = _parse_float_grid("TWINKLE_T0_DAY_GRID", day_grid)
    else:
        frac = _parse_float_grid("TWINKLE_T0_FRAC_GRID", "-0.8,-0.4,0.0,0.4,0.8")
        t0_offsets = frac * max(base_tE, 1.0e-3)

    return {
        "t0_offsets": t0_offsets,
        "tE_factors": _parse_float_grid("TWINKLE_TE_FACTOR_GRID", "0.35,0.6,1.0,1.8,3.0"),
        "u0": _parse_float_grid("TWINKLE_U0_GRID", "-1.5,-1.0,-0.6,-0.3,-0.1,0.1,0.3,0.6,1.0,1.5"),
        "s": _parse_float_grid("TWINKLE_S_GRID", "0.15,0.25,0.4,0.6,0.8,1.0,1.25,1.6,2.5,4.0,6.0"),
        "q": _parse_float_grid("TWINKLE_Q_GRID", "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0"),
        "rho": _parse_float_grid("TWINKLE_RHO_GRID", "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2"),
        "alpha": _alpha_grid(),
    }


def describe_twinkle_grid(base_tE: float, n_t0_centers: int | None = None) -> str:
    grids = _build_grids(base_tE)
    parts = []
    total = 1
    for name, arr in grids.items():
        total *= max(len(arr), 1)
        parts.append(f"{name}={len(arr)}")
    if n_t0_centers is not None:
        total *= max(n_t0_centers, 1)
        parts.insert(0, f"t0_centers={n_t0_centers}")
    return f"{', '.join(parts)}, total={total}"


def _require_twinkle():
    """Import the compiled AsterLight0626/Twinkle Python extension."""
    import importlib
    import sys

    twinkle_python_dir = os.environ.get("TWINKLE_PYTHON_DIR", "").strip()
    if twinkle_python_dir:
        twinkle_python_dir = os.path.abspath(os.path.expanduser(twinkle_python_dir))
        if twinkle_python_dir not in sys.path:
            sys.path.insert(0, twinkle_python_dir)
        sys.modules.pop("twinkle", None)

    try:
        twinkle = importlib.import_module("twinkle")  # type: ignore
    except Exception as exc:  # pragma: no cover - local install dependent
        raise ImportError(
            "TWINKLE_GRID_SEARCH=1 was requested, but the compiled microlensing "
            "Twinkle module could not be imported. Compile AsterLight0626/Twinkle "
            "in its python/ directory with `python setup.py build_ext --inplace`, "
            "then set TWINKLE_PYTHON_DIR=/path/to/Twinkle/python. Original error: "
            f"{exc!r}"
        ) from exc

    if not hasattr(twinkle, "Twinkle"):
        module_file = getattr(twinkle, "__file__", "<unknown>")
        public_attrs = [a for a in dir(twinkle) if not a.startswith("_")]
        shown_attrs = ", ".join(public_attrs[:30]) if public_attrs else "<no public attrs>"
        raise ImportError(
            "Imported a module named `twinkle`, but it is not the AsterLight0626/Twinkle "
            "microlensing extension because it has no `Twinkle` class.\n"
            f"Imported module path: {module_file}\n"
            f"Visible public attributes: {shown_attrs}\n\n"
            "Fix: compile Twinkle in Linux/WSL, set TWINKLE_PYTHON_DIR to the compiled python/ folder, "
            "and confirm `python -c 'import twinkle; print(hasattr(twinkle, \"Twinkle\"))'` prints True."
        )

    return twinkle


def _numpy_linear_chi2(A: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> tuple[float, float, float]:
    """Weighted linear fit y = Fs*A + Fb, returning Fs, Fb, chi2."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    ok = np.isfinite(A) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.inf

    A = A[ok]
    y = y[ok]
    w = 1.0 / np.square(yerr[ok])

    SAA = np.sum(w * A * A)
    SA = np.sum(w * A)
    S1 = np.sum(w)
    SyA = np.sum(w * y * A)
    Sy = np.sum(w * y)

    det = SAA * S1 - SA * SA
    if not np.isfinite(det) or abs(det) < 1.0e-300:
        return np.nan, np.nan, np.inf

    Fs = (SyA * S1 - Sy * SA) / det
    Fb = (SAA * Sy - SA * SyA) / det
    resid = y - (Fs * A + Fb)
    chi2 = float(np.sum(w * resid * resid))

    flux_scale = max(float(np.nanmedian(np.abs(y))), 1.0e-12)
    if Fs < 0.0:
        chi2 += float((Fs / flux_scale) ** 2) * ok.sum()
    if Fb < -0.5 * flux_scale:
        chi2 += float((Fb / flux_scale) ** 2) * ok.sum()

    return float(Fs), float(Fb), chi2


def _trajectory_numpy(t: np.ndarray, t0: float, tE: float, u0: float, alpha_deg: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.deg2rad(alpha_deg)
    tau = (t - t0) / tE
    x = -u0 * np.sin(alpha) + tau * np.cos(alpha)
    y = u0 * np.cos(alpha) + tau * np.sin(alpha)
    return np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)


def _select_twinkle_points(single_data: dict, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Information-rich subset for the Twinkle grid."""
    t = np.asarray(single_data["t"], dtype=float)
    y = np.asarray(single_data["mag"], dtype=float)
    yerr = np.asarray(single_data["mag_err"], dtype=float)

    n_valid = int(np.asarray(single_data.get("n_valid", len(t))))
    t = t[:n_valid]
    y = y[:n_valid]
    yerr = yerr[:n_valid]

    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
    t, y, yerr = t[ok], y[ok], yerr[ok]
    n = len(t)
    if n == 0 or max_points <= 0 or n <= max_points:
        return t, y, yerr

    baseline = np.nanmedian(y)
    signal = np.abs(y - baseline) / np.where(yerr > 0, yerr, np.inf)
    signal = np.where(np.isfinite(signal), signal, -np.inf)

    peak_fraction = float(os.environ.get("TWINKLE_PEAK_FRACTION", "0.80"))
    n_peak = max(1, min(max_points, int(round(max_points * peak_fraction))))
    n_uniform = max_points - n_peak

    peak_idx = np.argpartition(-signal, n_peak - 1)[:n_peak]
    if n_uniform > 0:
        uniform_idx = np.linspace(0, n - 1, n_uniform, dtype=np.int64)
        idx = np.unique(np.concatenate([peak_idx, uniform_idx]))
        if len(idx) < max_points:
            extra = np.argsort(-signal)
            extra = extra[~np.isin(extra, idx)][: max_points - len(idx)]
            idx = np.concatenate([idx, extra])
    else:
        idx = peak_idx

    idx = np.sort(idx[:max_points])
    return t[idx], y[idx], yerr[idx]


def _full_valid_points(single_data: dict, max_points: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid full data, optionally thinned for final Twinkle evaluation."""
    if max_points and max_points > 0:
        return _select_twinkle_points(single_data, max_points=max_points)

    t = np.asarray(single_data["t"], dtype=float)
    y = np.asarray(single_data["mag"], dtype=float)
    yerr = np.asarray(single_data["mag_err"], dtype=float)
    n_valid = int(np.asarray(single_data.get("n_valid", len(t))))
    t, y, yerr = t[:n_valid], y[:n_valid], yerr[:n_valid]
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
    return t[ok], y[ok], yerr[ok]


def _top_signal_t0_centers(single_data: dict, seed_t0_shifted: float, t0_shift: float, base_tE: float) -> np.ndarray:
    """Build shifted t0 centers from PSPL seed plus high-signal data times."""
    centers = [float(seed_t0_shifted)]
    n_peaks = int(os.environ.get("TWINKLE_T0_PEAK_N", "6"))
    if n_peaks <= 0:
        return np.asarray(centers, dtype=float)

    t, y, yerr = _full_valid_points(single_data, max_points=0)
    if len(t) == 0:
        return np.asarray(centers, dtype=float)

    baseline = np.nanmedian(y)
    signal = np.abs(y - baseline) / np.where(yerr > 0, yerr, np.inf)
    order = np.argsort(-signal)
    min_sep = float(os.environ.get("TWINKLE_T0_PEAK_MIN_SEP_FRAC", "0.15")) * max(base_tE, 1.0e-3)

    for idx in order:
        ti_shifted = float(t[idx] - t0_shift)
        if all(abs(ti_shifted - c) >= min_sep for c in centers):
            centers.append(ti_shifted)
        if len(centers) >= n_peaks + 1:
            break

    return np.asarray(centers, dtype=float)


def _iter_grid_candidates(seed_params: np.ndarray, t0_shift: float, t0_centers_shifted: np.ndarray) -> Iterable[tuple[np.ndarray, dict]]:
    """Yield optimizer-coordinate parameter vectors and physical metadata."""
    log_tE_seed = float(seed_params[1])
    tE_seed = max(float(math.exp(log_tE_seed)), 1.0e-6)
    grids = _build_grids(tE_seed)

    for t0_center, dt0, te_fac, u0, s, q, rho, alpha in product(
        t0_centers_shifted,
        grids["t0_offsets"],
        grids["tE_factors"],
        grids["u0"],
        grids["s"],
        grids["q"],
        grids["rho"],
        grids["alpha"],
    ):
        tE = max(tE_seed * float(te_fac), 1.0e-6)
        t0_shifted = float(t0_center) + float(dt0)
        p = np.asarray(
            [
                t0_shifted,
                math.log(tE),
                float(u0),
                math.log(float(s)),
                math.log(float(q)),
                math.log(float(rho)),
                float(alpha) % 360.0,
            ],
            dtype=np.float64,
        )
        meta = _params_to_row(p, t0_shift)
        yield p, meta


def _params_to_row(params: np.ndarray, t0_shift: float) -> dict:
    p = np.asarray(params, dtype=np.float64)
    return {
        "t0_shifted": float(p[0]),
        "t0_abs": float(p[0] + t0_shift),
        "tE": float(math.exp(p[1])),
        "u0": float(p[2]),
        "s": float(math.exp(p[3])),
        "q": float(math.exp(p[4])),
        "rho": float(math.exp(p[5])),
        "alpha_deg": float(p[6] % 360.0),
    }


def _row_to_params(row: dict, t0_shift: float) -> np.ndarray:
    return np.asarray(
        [
            float(row["t0_abs"]) - float(t0_shift),
            math.log(max(float(row["tE"]), 1e-12)),
            float(row["u0"]),
            math.log(max(float(row["s"]), 1e-12)),
            math.log(max(float(row["q"]), 1e-12)),
            math.log(max(float(row["rho"]), 1e-12)),
            float(row["alpha_deg"]) % 360.0,
        ],
        dtype=np.float64,
    )


def _evaluate_physical_candidate(engine, t, y, yerr, row: dict) -> tuple[float, float, float]:
    xs, ys = _trajectory_numpy(
        t,
        t0=float(row["t0_abs"]),
        tE=float(row["tE"]),
        u0=float(row["u0"]),
        alpha_deg=float(row["alpha_deg"]),
    )
    mag = np.empty(len(t), dtype=np.float64)
    engine.set_params(
        float(row["s"]),
        float(row["q"]),
        float(row["rho"]),
        xs,
        ys,
    )
    engine.run()
    engine.return_mag_to(mag)
    return _numpy_linear_chi2(mag, y, yerr)


def _push_heap(heap, top_n: int, counter: int, params: np.ndarray, row: dict) -> None:
    chi2 = float(row["chi2"])
    item = (-chi2, counter, params.copy(), dict(row))
    if len(heap) < top_n:
        heapq.heappush(heap, item)
    else:
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def _local_random_rows(base_rows: list[dict], n_samples: int, round_index: int, seed: int) -> Iterable[dict]:
    """Generate local random perturbations around the current best Twinkle rows."""
    if n_samples <= 0 or not base_rows:
        return []

    rng = np.random.default_rng(seed + 1009 * round_index)
    shrink = float(os.environ.get("TWINKLE_LOCAL_SHRINK", "0.55")) ** max(round_index, 0)

    t0_sigma_frac = float(os.environ.get("TWINKLE_LOCAL_T0_SIGMA_FRAC", "0.25")) * shrink
    log_tE_sigma = float(os.environ.get("TWINKLE_LOCAL_LOGTE_SIGMA", "0.35")) * shrink
    u0_sigma = float(os.environ.get("TWINKLE_LOCAL_U0_SIGMA", "0.25")) * shrink
    log_s_sigma = float(os.environ.get("TWINKLE_LOCAL_LOGS_SIGMA", "0.35")) * shrink
    log_q_sigma = float(os.environ.get("TWINKLE_LOCAL_LOGQ_SIGMA", "0.70")) * shrink
    log_rho_sigma = float(os.environ.get("TWINKLE_LOCAL_LOGRHO_SIGMA", "0.50")) * shrink
    alpha_sigma = float(os.environ.get("TWINKLE_LOCAL_ALPHA_SIGMA", "20.0")) * shrink

    q_min = float(os.environ.get("TWINKLE_Q_MIN", "1e-6"))
    q_max = float(os.environ.get("TWINKLE_Q_MAX", "1.0"))
    s_min = float(os.environ.get("TWINKLE_S_MIN", "0.05"))
    s_max = float(os.environ.get("TWINKLE_S_MAX", "20.0"))
    rho_min = float(os.environ.get("TWINKLE_RHO_MIN", "1e-5"))
    rho_max = float(os.environ.get("TWINKLE_RHO_MAX", "0.1"))
    u0_max = float(os.environ.get("TWINKLE_U0_MAX", "3.0"))

    for _ in range(n_samples):
        b = base_rows[int(rng.integers(0, len(base_rows)))]
        tE0 = max(float(b["tE"]), 1e-6)
        row = {
            "t0_abs": float(b["t0_abs"]) + rng.normal(0.0, t0_sigma_frac * tE0),
            "tE": tE0 * math.exp(rng.normal(0.0, log_tE_sigma)),
            "u0": float(np.clip(float(b["u0"]) + rng.normal(0.0, u0_sigma), -u0_max, u0_max)),
            "s": float(np.clip(float(b["s"]) * math.exp(rng.normal(0.0, log_s_sigma)), s_min, s_max)),
            "q": float(np.clip(float(b["q"]) * math.exp(rng.normal(0.0, log_q_sigma)), q_min, q_max)),
            "rho": float(np.clip(float(b["rho"]) * math.exp(rng.normal(0.0, log_rho_sigma)), rho_min, rho_max)),
            "alpha_deg": float((float(b["alpha_deg"]) + rng.normal(0.0, alpha_sigma)) % 360.0),
        }
        yield row


def twinkle_grid_search_fsbl(single_data: dict, seed_params: np.ndarray) -> TwinkleGridResult:
    """Run broad + optional local Twinkle search and return good FSBL starts."""
    twinkle = _require_twinkle()

    top_n = max(int(os.environ.get("TWINKLE_TOPN", "128")), 1)
    max_points = max(int(os.environ.get("TWINKLE_MAX_POINTS", "768")), 16)
    max_evals_raw = os.environ.get("TWINKLE_MAX_EVALS", "0").strip()
    max_evals = int(max_evals_raw) if max_evals_raw else 0
    reltol = float(os.environ.get("TWINKLE_RELTOL", "3e-4"))
    device_num = int(os.environ.get("TWINKLE_DEVICE", "0"))
    n_stream = int(os.environ.get("TWINKLE_N_STREAM", "1"))
    astrometry = os.environ.get("TWINKLE_ASTROMETRY", "0") == "1"
    local_rounds = max(int(os.environ.get("TWINKLE_LOCAL_ROUNDS", "2")), 0)
    local_n = max(int(os.environ.get("TWINKLE_LOCAL_N", "20000")), 0)
    local_seeds = max(int(os.environ.get("TWINKLE_LOCAL_SEEDS", "32")), 1)
    rng_seed = int(os.environ.get("TWINKLE_RANDOM_SEED", "12345"))

    t, y, yerr = _select_twinkle_points(single_data, max_points=max_points)
    if len(t) < 3:
        raise ValueError("Not enough valid points for Twinkle grid search.")

    seed_params = np.asarray(seed_params, dtype=np.float64)
    t0_shift = float(np.asarray(single_data["t_0_shift"]))
    base_tE = max(float(math.exp(seed_params[1])), 1.0e-6)
    t0_centers = _top_signal_t0_centers(single_data, float(seed_params[0]), t0_shift, base_tE)
    grid_desc = describe_twinkle_grid(base_tE, n_t0_centers=len(t0_centers))

    grids = _build_grids(base_tE)
    n_total = max(len(t0_centers), 1)
    for arr in grids.values():
        n_total *= max(len(arr), 1)
    if max_evals > 0:
        n_total_to_run = min(n_total, max_evals)
    else:
        n_total_to_run = n_total

    print(
        f"    Twinkle grid: points={len(t)}, top_n={top_n}, reltol={reltol:g}, "
        f"device={device_num}, streams={n_stream}, {grid_desc}, running={n_total_to_run}, "
        f"local_rounds={local_rounds}, local_n={local_n}"
    )

    engine = twinkle.Twinkle(int(len(t)), device_num, n_stream, reltol, astrometry)

    heap: list[tuple[float, int, np.ndarray, dict]] = []
    n_evaluated = 0
    n_failed = 0
    progress_every = int(os.environ.get("TWINKLE_PROGRESS_EVERY", "20000"))

    for counter, (params, meta) in enumerate(_iter_grid_candidates(seed_params, t0_shift, t0_centers)):
        if max_evals > 0 and n_evaluated >= max_evals:
            break
        try:
            Fs, Fb, chi2 = _evaluate_physical_candidate(engine, t, y, yerr, meta)
        except Exception:
            n_failed += 1
            n_evaluated += 1
            continue
        if not np.isfinite(chi2):
            n_failed += 1
            n_evaluated += 1
            continue
        row = dict(meta)
        row.update({"chi2": float(chi2), "Fs": float(Fs), "Fb": float(Fb), "stage": "broad"})
        _push_heap(heap, top_n, counter, params, row)
        n_evaluated += 1
        if progress_every > 0 and n_evaluated % progress_every == 0:
            best_now = -max(h[0] for h in heap) if heap else np.inf
            print(f"    Twinkle broad progress: {n_evaluated}/{n_total_to_run}, best chi2≈{best_now:.6g}")

    if not heap:
        raise RuntimeError("Twinkle broad grid search did not return any finite candidate.")

    # Local random refinement around the best broad-grid candidates.  This is
    # usually much more useful than making every grid axis denser.
    for r in range(local_rounds):
        current_best = sorted(heap, key=lambda item: -item[0])[:local_seeds]
        base_rows = [item[3] for item in current_best]
        print(f"    Twinkle local round {r + 1}/{local_rounds}: seeds={len(base_rows)}, samples={local_n}")
        for row in _local_random_rows(base_rows, local_n, r, rng_seed):
            try:
                Fs, Fb, chi2 = _evaluate_physical_candidate(engine, t, y, yerr, row)
            except Exception:
                n_failed += 1
                n_evaluated += 1
                continue
            if not np.isfinite(chi2):
                n_failed += 1
                n_evaluated += 1
                continue
            row = dict(row)
            row.update({"chi2": float(chi2), "Fs": float(Fs), "Fb": float(Fb), "stage": f"local{r + 1}"})
            params = _row_to_params(row, t0_shift)
            _push_heap(heap, top_n, n_evaluated, params, row)
            n_evaluated += 1
            if progress_every > 0 and n_evaluated % progress_every == 0:
                best_now = -max(h[0] for h in heap) if heap else np.inf
                print(f"    Twinkle total progress: {n_evaluated}, best chi2≈{best_now:.6g}")

    best = sorted(heap, key=lambda item: -item[0])  # smallest chi2 first
    starts = np.stack([item[2] for item in best]).astype(np.float64)
    rows = [item[3] for item in best]
    scores = np.asarray([row["chi2"] for row in rows], dtype=float)

    print(
        f"    Twinkle search done: evaluated={n_evaluated}, failed={n_failed}, "
        f"best chi2={scores[0]:.6g}, worst kept chi2={scores[-1]:.6g}, "
        f"best stage={rows[0].get('stage', '')}"
    )

    return TwinkleGridResult(
        starts=jnp.asarray(starts, dtype=jnp.float64),
        scores=scores,
        rows=rows,
        n_evaluated=n_evaluated,
        n_failed=n_failed,
        n_total_grid=n_total,
    )


def twinkle_evaluate_best_fsbl(single_data: dict, starts, n_check: int | None = None, max_points: int = 0) -> TwinkleFinalResult:
    """Evaluate the best Twinkle starts, usually on the full event, and return best.

    This avoids the very slow microJAX refinement stage when the user only needs
    a robust FSBL solution for diagnosis or candidate selection.
    """
    twinkle = _require_twinkle()
    starts_np = np.asarray(starts, dtype=np.float64)
    if starts_np.ndim == 1:
        starts_np = starts_np[None, :]
    if n_check is None:
        n_check = int(os.environ.get("TWINKLE_FINAL_TOPN", "32"))
    n_check = max(1, min(int(n_check), starts_np.shape[0]))

    final_max_points = int(os.environ.get("TWINKLE_FINAL_MAX_POINTS", str(max_points or 0)))
    t, y, yerr = _full_valid_points(single_data, max_points=final_max_points)
    if len(t) < 3:
        raise ValueError("Not enough valid points for final Twinkle evaluation.")

    reltol = float(os.environ.get("TWINKLE_FINAL_RELTOL", os.environ.get("TWINKLE_RELTOL", "1e-4")))
    device_num = int(os.environ.get("TWINKLE_DEVICE", "0"))
    n_stream = int(os.environ.get("TWINKLE_N_STREAM", "1"))
    astrometry = os.environ.get("TWINKLE_ASTROMETRY", "0") == "1"
    engine = twinkle.Twinkle(int(len(t)), device_num, n_stream, reltol, astrometry)

    t0_shift = float(np.asarray(single_data["t_0_shift"]))
    best_row = None
    best_params = None
    n_failed = 0

    for k in range(n_check):
        params = starts_np[k]
        row = _params_to_row(params, t0_shift)
        try:
            Fs, Fb, chi2 = _evaluate_physical_candidate(engine, t, y, yerr, row)
        except Exception:
            n_failed += 1
            continue
        if not np.isfinite(chi2):
            n_failed += 1
            continue
        row.update({"chi2": float(chi2), "Fs": float(Fs), "Fb": float(Fb), "stage": "final_twinkle"})
        if best_row is None or row["chi2"] < best_row["chi2"]:
            best_row = row
            best_params = params.copy()

    if best_row is None or best_params is None:
        raise RuntimeError("Final Twinkle evaluation did not return any finite candidate.")

    return TwinkleFinalResult(
        params=jnp.asarray(best_params, dtype=jnp.float64),
        row=best_row,
        n_evaluated=n_check,
        n_failed=n_failed,
        eval_n_points=len(t),
    )


def save_twinkle_candidates_csv(path: str, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "rank",
        "stage",
        "chi2",
        "Fs",
        "Fb",
        "t0_abs",
        "t0_shifted",
        "tE",
        "u0",
        "s",
        "q",
        "rho",
        "alpha_deg",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {k: row.get(k, "") for k in fieldnames}
            out["rank"] = i
            writer.writerow(out)
