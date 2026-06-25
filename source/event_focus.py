"""Focused fixed-budget point selection for dense Roman light curves.

The goal is not to replace the final likelihood.  It is a search-stage helper:
Twinkle/global grid search and expensive finite-source calls should see a compact
but informative subset around the microlensing signal instead of thousands of
nearly redundant baseline points.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FocusConfig:
    max_points: int = 512
    peak_points: int = 360
    uniform_points: int = 100
    baseline_points: int = 52
    sigma: float = 2.0
    amp_frac: float = 0.02
    dt_bin: float = 0.02
    per_bin: int = 2
    min_points: int = 64

    @classmethod
    def from_env(cls, prefix: str = "TWINKLE_FOCUS_") -> "FocusConfig":
        return cls(
            max_points=int(os.environ.get(prefix + "MAX_POINTS", os.environ.get("TWINKLE_MAX_POINTS", "512"))),
            peak_points=int(os.environ.get(prefix + "PEAK_POINTS", "360")),
            uniform_points=int(os.environ.get(prefix + "UNIFORM_POINTS", "100")),
            baseline_points=int(os.environ.get(prefix + "BASELINE_POINTS", "52")),
            sigma=float(os.environ.get(prefix + "SIGMA", "2.0")),
            amp_frac=float(os.environ.get(prefix + "AMP_FRAC", "0.02")),
            dt_bin=float(os.environ.get(prefix + "DT_BIN", "0.02")),
            per_bin=int(os.environ.get(prefix + "PER_BIN", "2")),
            min_points=int(os.environ.get(prefix + "MIN_POINTS", "64")),
        )


def _best_per_time_bin(t: np.ndarray, score: np.ndarray, idx: np.ndarray, dt_bin: float, per_bin: int) -> np.ndarray:
    if len(idx) == 0 or dt_bin <= 0:
        return idx
    idx = np.asarray(idx, dtype=int)
    t0 = float(np.nanmin(t[idx]))
    bins = np.floor((t[idx] - t0) / dt_bin).astype(int)
    chosen = []
    for b in np.unique(bins):
        in_bin = idx[bins == b]
        order = np.argsort(score[in_bin])[::-1]
        chosen.extend(in_bin[order[:per_bin]])
    return np.asarray(chosen, dtype=int)


def select_focus_indices(t, flux, flux_err, cfg: FocusConfig | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return sorted indices for a fixed-budget microlensing-search subset."""
    cfg = cfg or FocusConfig.from_env()
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(t) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    idx_all = np.where(valid)[0]
    if len(idx_all) == 0:
        return idx_all, {"n_input": 0, "n_selected": 0}

    baseline = float(np.nanmedian(flux[idx_all]))
    signal = flux - baseline
    snr = signal / flux_err
    amp = float(max(np.nanmax(signal[idx_all]), 0.0))

    # Positive microlensing candidates only.  Do not use abs(); negative outliers
    # around baseline are not informative for a standard brightening event.
    focus_mask = valid & (snr > cfg.sigma)
    if amp > 0:
        focus_mask |= valid & (signal > cfg.amp_frac * amp)

    focus_idx = np.where(focus_mask)[0]
    if len(focus_idx) == 0:
        # Fallback: keep highest positive deviations and baseline skeleton.
        order = np.argsort(snr[idx_all])[::-1]
        focus_idx = idx_all[order[: min(cfg.peak_points, len(idx_all))]]

    focus_idx = _best_per_time_bin(t, snr, focus_idx, cfg.dt_bin, cfg.per_bin)

    chosen = []

    # 1. Highest-SNR focus points.
    if len(focus_idx) > 0 and cfg.peak_points > 0:
        order = np.argsort(snr[focus_idx])[::-1]
        chosen.extend(focus_idx[order[: min(cfg.peak_points, len(focus_idx))]])

    # 2. Uniform coverage through the focus region.
    if len(focus_idx) > 0 and cfg.uniform_points > 0:
        fs = focus_idx[np.argsort(t[focus_idx])]
        take = np.linspace(0, len(fs) - 1, min(cfg.uniform_points, len(fs)), dtype=int)
        chosen.extend(fs[take])

    # 3. Small baseline skeleton, uniformly in time.
    baseline_idx = idx_all[~np.isin(idx_all, focus_idx)]
    if len(baseline_idx) > 0 and cfg.baseline_points > 0:
        bs = baseline_idx[np.argsort(t[baseline_idx])]
        take = np.linspace(0, len(bs) - 1, min(cfg.baseline_points, len(bs)), dtype=int)
        chosen.extend(bs[take])

    chosen = np.unique(np.asarray(chosen, dtype=int))

    # 4. Hard cap: keep a mixture of strongest SNR and uniform-in-time selected points.
    max_points = max(cfg.max_points, cfg.min_points)
    if len(chosen) > max_points:
        n_peak = min(int(0.75 * max_points), len(chosen))
        order = np.argsort(snr[chosen])[::-1]
        keep_peak = chosen[order[:n_peak]]
        remaining = np.setdiff1d(chosen, keep_peak, assume_unique=False)
        n_uniform = max_points - len(keep_peak)
        if len(remaining) > 0 and n_uniform > 0:
            rem = remaining[np.argsort(t[remaining])]
            take = np.linspace(0, len(rem) - 1, min(n_uniform, len(rem)), dtype=int)
            keep = np.unique(np.concatenate([keep_peak, rem[take]]))
        else:
            keep = keep_peak
        chosen = keep

    chosen = chosen[np.argsort(t[chosen])]
    meta = {
        "n_input": int(len(idx_all)),
        "n_focus_candidates": int(len(focus_idx)),
        "n_selected": int(len(chosen)),
        "baseline_flux": float(baseline),
        "max_positive_snr": float(np.nanmax(snr[idx_all])),
    }
    return chosen, meta


def subset_arrays(data: dict, indices: np.ndarray) -> dict:
    return {
        "t": np.asarray(data["t"])[indices],
        "mag": np.asarray(data["mag"])[indices],
        "mag_err": np.asarray(data["mag_err"])[indices],
    }
