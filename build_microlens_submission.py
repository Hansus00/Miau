"""
Build a microlens-submit CSV and optionally create/export a submission project
from this pipeline's *_params.txt result files.

By default, one row per event is not exported: the script scores every fitted
model with AIC = 2k + 2*(0.5*chi2 + priors), using the same priors as the
optimizer in source/magnification_model.py. For each event it keeps every model
from the best result source whose AIC lies within --aic-delta of the minimum
(default 2). Use --all-models to export every fit without selection.

Fisher-matrix uncertainties (uncertainty_method=fisher_matrix) are attached by
default for supported models and written to parameter_uncertainties in the CSV /
microlens-submit solution JSON.

Typical use from project root:

  python build_microlens_submission.py \
      --results results \
      --csv submission_solutions.csv \
      --team-name "microlensing-innovation-aleje-ujazdowskie" \
      --tier beginner \
      --repo-url "https://github.com/Hansus00/Miau" \
      --submission-dir rmdc26_submission \
      --run-microlens-submit

Then inspect validation output and final zip in/near rmdc26_submission.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


# -----------------------------------------------------------------------------
# Parsing pipeline *_params.txt files
# -----------------------------------------------------------------------------

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
    flags=re.IGNORECASE,
)


def parse_float_maybe(value: str) -> Any:
    """Parse floats from lines such as 'Array(1.23, dtype=float64)' safely."""
    s = str(value).strip()
    if not s:
        return ""
    m = FLOAT_RE.search(s)
    if m is None:
        return s
    token = m.group(0)
    try:
        return float(token)
    except ValueError:
        return s


def parse_params_txt(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {section_name: {key: value}} from a pipeline params file."""
    sections: Dict[str, Dict[str, Any]] = {}
    current: Optional[str] = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                sections.setdefault(current, {})
                continue
            if current is None or ":" not in line:
                continue
            key, val = line.split(":", 1)
            sections[current][key.strip()] = parse_float_maybe(val)
    return sections


# -----------------------------------------------------------------------------
# Parsing PyMultiNest FSBL output directories
# -----------------------------------------------------------------------------

MN_THETA_NAMES_FSBL = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg"]
MN_THETA_NAMES_FSBL_PARALLAX = [
    "t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg", "pi_E_N", "pi_E_E"
]


def infer_event_id_from_multinest_dir(path: Path) -> str:
    """
    Infer event id from directories such as:
        RMDC26_000005_multinest_FSBL_from_FSPL
        RMDC26_000005_multinest_from_fspl_deep
    """
    name = path.name
    if "_multinest" in name:
        return name.split("_multinest", 1)[0]
    return name


def infer_multinest_model_name(path: Path, section: Optional[Dict[str, Any]] = None) -> str:
    """Infer whether a MultiNest result directory contains FSBL or FSBL+Parallax."""
    name = path.name.lower()
    if "parallax" in name or "paral" in name:
        return "FSBL+Parallax"
    if section is not None and (
        as_float(section, "pi_E_N") is not None
        or as_float(section, "pi_E_E") is not None
        or as_float(section, "piEN") is not None
        or as_float(section, "piEE") is not None
    ):
        return "FSBL+Parallax"
    return "FSBL"


def parse_multinest_best_fit_txt(path: Path) -> Dict[str, Any]:
    """
    Parse best_fit.txt written by source/multinest_cpu.py.

    This file is more useful than raw mn_stats.dat because it contains Fs/Fb/chi2
    in addition to the maximum-likelihood nonlinear parameters.
    """
    raw: Dict[str, Any] = {}
    if not path.exists():
        return raw

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            raw[key] = parse_float_maybe(val)

    section: Dict[str, Any] = {}
    if as_float(raw, "chi2") is not None:
        section["Chi2"] = raw["chi2"]
    if as_float(raw, "Fs") is not None:
        section["Fs"] = raw["Fs"]
    if as_float(raw, "Fb") is not None:
        section["Fb"] = raw["Fb"]

    if as_float(raw, "t0") is not None:
        section["t_0"] = raw["t0"]
    if as_float(raw, "u0") is not None:
        section["u_0"] = raw["u0"]
    if as_float(raw, "alpha_deg") is not None:
        section["alpha_deg"] = raw["alpha_deg"]

    # Parallax components, if this is an FSBL+Parallax MultiNest run.
    for src_key, dst_key in [
        ("pi_E_N", "pi_E_N"),
        ("pi_E_E", "pi_E_E"),
        ("piEN", "pi_E_N"),
        ("piEE", "pi_E_E"),
    ]:
        if as_float(raw, src_key) is not None:
            section[dst_key] = raw[src_key]

    # Prefer explicitly written physical values. Fall back to exponentiated logs.
    if as_float(raw, "tE") is not None:
        section["t_E"] = raw["tE"]
    elif as_float(raw, "log_tE") is not None:
        section["t_E"] = math.exp(float(raw["log_tE"]))

    if as_float(raw, "s") is not None:
        section["s"] = raw["s"]
    elif as_float(raw, "log_s") is not None:
        section["s"] = math.exp(float(raw["log_s"]))

    if as_float(raw, "q") is not None:
        section["q"] = raw["q"]
    elif as_float(raw, "log_q") is not None:
        section["q"] = math.exp(float(raw["log_q"]))

    if as_float(raw, "rho") is not None:
        section["rho"] = raw["rho"]
    elif as_float(raw, "log_rho") is not None:
        section["rho"] = math.exp(float(raw["log_rho"]))

    section["multinest_source"] = "best_fit.txt"
    return section


def _extract_floats_from_line(line: str) -> List[float]:
    out: List[float] = []
    for m in FLOAT_RE.finditer(line):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def parse_mn_stats_maxlike(path: Path, ndim: int = 7) -> Dict[str, Any]:
    """
    Parse maximum-likelihood nonlinear FSBL/FSBL+Parallax parameters from mn_stats.dat.

    Parameter order is the one used by our MultiNest scripts:
      FSBL:
        t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg
      FSBL+Parallax:
        t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg, pi_E_N, pi_E_E
    """
    if not path.exists():
        return {}

    text_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for i, line in enumerate(text_lines):
        low = line.lower()
        if "maximum" in low and "likelihood" in low and "parameter" in low:
            start = i + 1
            break
    if start is None:
        return {}

    vals: List[float] = []
    expected_index = 1
    for line in text_lines[start:]:
        low = line.strip().lower()
        if not low:
            if vals:
                break
            continue
        if any(marker in low for marker in ["maximum a posteriori", "marginal", "mean", "mode", "evidence", "posterior"]):
            if vals:
                break

        nums = _extract_floats_from_line(line)
        if not nums:
            continue

        # Common MultiNest layout: '<index> <value>'.
        if len(nums) >= 2 and abs(nums[0] - expected_index) < 1e-9:
            vals.append(nums[1])
            expected_index += 1
        else:
            vals.append(nums[0])

        if len(vals) >= ndim:
            break

    if len(vals) < ndim:
        return {}

    theta = vals[:ndim]
    section: Dict[str, Any] = {
        "t_0": theta[0],
        "t_E": math.exp(theta[1]),
        "u_0": theta[2],
        "s": math.exp(theta[3]),
        "q": math.exp(theta[4]),
        "rho": math.exp(theta[5]),
        "alpha_deg": theta[6],
        "multinest_source": "mn_stats.dat maximum-likelihood block",
    }
    if ndim >= 9:
        section["pi_E_N"] = theta[7]
        section["pi_E_E"] = theta[8]
    return section


def _candidate_multinest_sample_files(path: Path) -> List[Path]:
    """Return likely posterior sample files written by PyMultiNest/MultiNest."""
    names = [
        "mn_post_equal_weights.dat",
        "mn_equal_weights.dat",
        "post_equal_weights.dat",
        "equal_weights.dat",
    ]
    candidates: List[Path] = [path / name for name in names]
    candidates.extend(sorted(path.glob("*post_equal_weights*.dat")))
    candidates.extend(sorted(path.glob("*equal_weights*.dat")))
    # Raw MultiNest chain; usually columns are weight, -2logL, parameters...
    candidates.extend([path / "mn_.txt", path / "mn.txt", path / "mn_.dat"])
    out: List[Path] = []
    seen = set()
    for p in candidates:
        if p.exists() and p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def _load_numeric_table(path: Path) -> Optional[np.ndarray]:
    try:
        arr = np.loadtxt(path)
    except Exception:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0 or arr.shape[0] < 2:
        return None
    return arr


def _weighted_quantile(values: np.ndarray, weights: Optional[np.ndarray], qs: List[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        good &= np.isfinite(weights) & (weights > 0)
    values = values[good]
    if values.size == 0:
        return np.full(len(qs), np.nan)
    if weights is None:
        return np.quantile(values, qs)
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return np.full(len(qs), np.nan)
    cdf /= cdf[-1]
    return np.interp(qs, cdf, values)


def _theta_samples_from_multinest_file(path: Path, ndim: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read theta samples from a MultiNest output file.

    Equal-weight files are assumed to contain parameters in the first ndim columns
    and optionally logL in the last column. Raw mn_.txt is assumed to contain
    weight, -2logL, then ndim parameters.
    """
    arr = _load_numeric_table(path)
    if arr is None:
        return None, None

    name = path.name.lower()
    if ("equal" in name or "post" in name) and arr.shape[1] >= ndim:
        return arr[:, :ndim], None

    if arr.shape[1] >= ndim + 2:
        weights = arr[:, 0]
        theta = arr[:, 2 : 2 + ndim]
        return theta, weights

    if arr.shape[1] >= ndim:
        return arr[:, :ndim], None

    return None, None


def _angle_uncertainty_rad(alpha_deg_samples: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    """Compute a robust 68% half-width for angular uncertainty, returned in radians."""
    alpha_rad = np.deg2rad(np.asarray(alpha_deg_samples, dtype=float))
    good = np.isfinite(alpha_rad)
    if weights is not None:
        good &= np.isfinite(weights) & (weights > 0)
        w = weights[good]
    else:
        w = None
    alpha_rad = alpha_rad[good]
    if alpha_rad.size < 2:
        return None

    # Unwrap around the circular mean to avoid artificial jumps at 0/360 deg.
    if w is None:
        center = np.angle(np.mean(np.exp(1j * alpha_rad)))
    else:
        center = np.angle(np.sum(w * np.exp(1j * alpha_rad)) / np.sum(w))
    shifted = np.angle(np.exp(1j * (alpha_rad - center))) + center
    q16, q84 = _weighted_quantile(shifted, w, [0.16, 0.84])
    sig = 0.5 * (q84 - q16)
    if math.isfinite(sig) and sig >= 0:
        return float(sig)
    return None


def _uncertainty_from_samples(values: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    q16, q84 = _weighted_quantile(values, weights, [0.16, 0.84])
    sig = 0.5 * (q84 - q16)
    if math.isfinite(sig) and sig >= 0:
        return float(sig)
    return None


def _parse_stats_dict_from_best_fit(path: Path) -> Optional[Dict[str, Any]]:
    """Parse the Python-dict Stats block written by our MultiNest scripts."""
    best_fit = path / "best_fit.txt"
    if not best_fit.exists():
        return None
    txt = best_fit.read_text(encoding="utf-8", errors="ignore")
    marker = "Stats:"
    idx = txt.find(marker)
    if idx < 0:
        return None
    payload = txt[idx + len(marker):].strip()
    if not payload:
        return None
    try:
        obj = ast.literal_eval(payload)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _marginal_half_width(marginal: Dict[str, Any]) -> Optional[float]:
    """Return a 68%-like half-width in the sampled variable."""
    interval = marginal.get("1sigma")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        try:
            lo = float(interval[0])
            hi = float(interval[1])
            if math.isfinite(lo) and math.isfinite(hi):
                return abs(hi - lo) / 2.0
        except Exception:
            pass
    for key in ["sigma", "std", "stdev"]:
        if key in marginal:
            try:
                sig = abs(float(marginal[key]))
                if math.isfinite(sig):
                    return sig
            except Exception:
                pass
    return None


def _marginal_interval(marginal: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    interval = marginal.get("1sigma")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        try:
            lo = float(interval[0])
            hi = float(interval[1])
            if math.isfinite(lo) and math.isfinite(hi):
                return lo, hi
        except Exception:
            pass
    return None


def multinest_stats_uncertainties(path: Path, model_name: str, section: Dict[str, Any]) -> Dict[str, float]:
    """
    Fallback uncertainty parser from the Stats dict in best_fit.txt.

    This is less ideal than posterior samples, but useful when only best_fit.txt
    and mn_stats.dat are kept. MultiNest marginals are in sampled coordinates,
    so log parameters are transformed back to physical space.
    """
    stats = _parse_stats_dict_from_best_fit(path)
    if not stats:
        return {}
    marginals = stats.get("marginals")
    if not isinstance(marginals, list):
        return {}

    ndim = 9 if model_name == "FSBL+Parallax" else 7
    if len(marginals) < ndim:
        return {}

    unc: Dict[str, float] = {}

    # Direct sampled parameters.
    for idx, key in [(0, "t0"), (2, "u0")]:
        sig = _marginal_half_width(marginals[idx])
        if sig is not None:
            unc[key] = sig

    # Log-sampled positive parameters.
    for idx, key, center_key in [
        (1, "tE", "t_E"),
        (3, "s", "s"),
        (4, "q", "q"),
        (5, "rho", "rho"),
    ]:
        interval = _marginal_interval(marginals[idx])
        if interval is not None:
            lo, hi = interval
            sig = 0.5 * abs(math.exp(hi) - math.exp(lo))
        else:
            sig_log = _marginal_half_width(marginals[idx])
            center = as_float(section, center_key)
            sig = abs(center * sig_log) if (sig_log is not None and center is not None) else None
        if sig is not None and math.isfinite(sig):
            unc[key] = float(sig)

    # alpha is submitted in radians, while MultiNest samples alpha_deg.
    sig_alpha_deg = _marginal_half_width(marginals[6])
    if sig_alpha_deg is not None:
        unc["alpha"] = math.radians(sig_alpha_deg)

    if model_name == "FSBL+Parallax":
        sig_n = _marginal_half_width(marginals[7])
        sig_e = _marginal_half_width(marginals[8])
        if sig_n is not None:
            unc["piEN"] = sig_n
        if sig_e is not None:
            unc["piEE"] = sig_e

    return {k: float(v) for k, v in unc.items() if math.isfinite(float(v)) and float(v) >= 0.0}


def multinest_posterior_uncertainties(path: Path, model_name: str) -> Dict[str, float]:
    """
    Estimate 1-sigma-like parameter_uncertainties from MultiNest posterior samples.

    The output keys match microlens-submit parameter names:
      FSBL: t0, u0, tE, s, q, alpha, rho
      FSBL+Parallax: additionally piEN, piEE

    For log-sampled positive parameters we transform samples to physical space
    before measuring the 16-84 percentile half-width.
    """
    ndim = 9 if model_name == "FSBL+Parallax" else 7
    for sample_file in _candidate_multinest_sample_files(path):
        theta, weights = _theta_samples_from_multinest_file(sample_file, ndim)
        if theta is None or theta.shape[1] < ndim or theta.shape[0] < 2:
            continue

        physical: Dict[str, np.ndarray] = {
            "t0": theta[:, 0],
            "tE": np.exp(theta[:, 1]),
            "u0": theta[:, 2],
            "s": np.exp(theta[:, 3]),
            "q": np.exp(theta[:, 4]),
            "rho": np.exp(theta[:, 5]),
        }
        unc: Dict[str, float] = {}
        for key, vals in physical.items():
            sig = _uncertainty_from_samples(vals, weights)
            if sig is not None:
                unc[key] = sig

        alpha_sig = _angle_uncertainty_rad(theta[:, 6], weights)
        if alpha_sig is not None:
            unc["alpha"] = alpha_sig

        if model_name == "FSBL+Parallax":
            sig_n = _uncertainty_from_samples(theta[:, 7], weights)
            sig_e = _uncertainty_from_samples(theta[:, 8], weights)
            if sig_n is not None:
                unc["piEN"] = sig_n
            if sig_e is not None:
                unc["piEE"] = sig_e

        if unc:
            return unc
    return {}

def parse_multinest_result_dir(path: Path) -> Optional[Dict[str, Any]]:
    """
    Return an FSBL/FSBL+Parallax-like section from a MultiNest output directory.

    Preference order:
      1. best_fit.txt, because it contains Fs/Fb/chi2 written by our code.
      2. mn_stats.dat maximum-likelihood parameters.

    If posterior sample files exist, attach MultiNest posterior uncertainties in
    section["multinest_uncertainties"].
    """
    best = parse_multinest_best_fit_txt(path / "best_fit.txt")
    model_name_guess = infer_multinest_model_name(path, best)
    ndim = 9 if model_name_guess == "FSBL+Parallax" else 7
    stats = parse_mn_stats_maxlike(path / "mn_stats.dat", ndim=ndim)

    # If the directory name did not say parallax but mn_stats.dat/best_fit does,
    # re-infer after merging.
    merged = dict(stats)
    merged.update(best)
    model_name = infer_multinest_model_name(path, merged)
    if model_name == "FSBL+Parallax" and ndim != 9:
        stats = parse_mn_stats_maxlike(path / "mn_stats.dat", ndim=9)
        merged = dict(stats)
        merged.update(best)

    if not merged:
        return None

    required = ["t_0", "u_0", "t_E", "s", "q", "rho", "alpha_deg"]
    if model_name == "FSBL+Parallax":
        required += ["pi_E_N", "pi_E_E"]
    if any(as_float(merged, k) is None for k in required):
        return None

    unc = multinest_posterior_uncertainties(path, model_name)
    unc_method = "nested_sampling"
    if not unc:
        unc = multinest_stats_uncertainties(path, model_name, merged)
        unc_method = "nested_sampling_marginals"
    if unc:
        merged["multinest_uncertainties"] = unc
        merged["multinest_uncertainty_method"] = unc_method
        merged["multinest_confidence_level"] = 0.68

    merged["multinest_model_name"] = model_name
    return merged

def iter_multinest_dirs(results_roots: Iterable[Path]) -> Iterable[Path]:
    """Find MultiNest result dirs containing mn_stats.dat under result roots."""
    seen = set()
    for root in results_roots:
        candidates: List[Path] = []
        if root.is_dir() and (root / "mn_stats.dat").exists():
            candidates.append(root)
        elif root.exists():
            candidates.extend(p.parent for p in root.rglob("mn_stats.dat"))

        for d in sorted(candidates):
            # Keep the intended FSBL-from-FSPL folders, but also accept other
            # explicit MultiNest output dirs if the user passes them directly.
            if "multinest" not in d.name.lower() and d != root:
                continue
            rp = d.resolve()
            if rp not in seen:
                seen.add(rp)
                yield d


def infer_event_id_from_params_file(path: Path) -> str:
    name = path.name
    if name.endswith("_params.txt"):
        return name[: -len("_params.txt")]
    return path.stem


def finite_float(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def as_float(d: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key, default)
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return default
    if not math.isfinite(f):
        return default
    return f


def alpha_to_rad(section: Dict[str, Any]) -> Optional[float]:
    """
    microlens-submit expects alpha in radians.
    Our pipeline usually writes alpha_deg. If only alpha exists, assume it is radians
    unless it looks like degrees.
    """
    if as_float(section, "alpha_deg") is not None:
        return math.radians(float(section["alpha_deg"]))
    if as_float(section, "alpha") is not None:
        a = float(section["alpha"])
        # Heuristic: values outside a few radians are almost certainly degrees.
        if abs(a) > 2.0 * math.pi + 1e-6:
            return math.radians(a)
        return a
    return None


# Nonlinear parameters fitted by the optimizer (Fs/Fb are profiled analytically).
MODEL_N_NONLINEAR: Dict[str, int] = {
    "PSPL": 3,
    "FSPL": 4,
    "PSPL+Parallax": 5,
    "FSPL+Parallax": 6,
    "BSPL": 6,
    "BSPL+Parallax": 8,
    "FSBL": 7,
    "FSBL+Parallax": 9,
}


def model_n_free_params(model_name: str) -> int:
    """Count nonlinear fit parameters plus profiled source/background fluxes."""
    k_nl = MODEL_N_NONLINEAR.get(model_name)
    if k_nl is None:
        return 0
    return k_nl + 2


def _negative_flux_prior(flux: float, sigma: float = 2.0) -> float:
    return 0.5 * (max(-flux, 0.0) ** 2) / (sigma**2)


def _prior_parallax(section: Dict[str, Any]) -> float:
    pi_n = as_float(section, "pi_E_N") or 0.0
    pi_e = as_float(section, "pi_E_E") or 0.0
    return 0.5 * (pi_n**2 + pi_e**2) / (0.15**2)


def _prior_bspl(section: Dict[str, Any]) -> float:
    qf = as_float(section, "q_f")
    if qf is None:
        return 0.0
    prior = 0.5 * (qf - 1.0) ** 2 / (10.0**2)
    return prior + 0.5 * (max(-qf, 0.0) ** 2) / (3.0**2)


def _prior_fspl(section: Dict[str, Any]) -> float:
    rho = as_float(section, "rho")
    if rho is None or rho <= 0.0:
        return float("inf")
    log_rho = math.log(rho)
    return 0.5 * (max(math.log(1.0e-6) - log_rho, 0.0) / 0.5) ** 2 + 0.5 * (
        max(log_rho - math.log(1.0), 0.0) / 0.5
    ) ** 2


def _prior_fsbl(section: Dict[str, Any]) -> float:
    rho = as_float(section, "rho")
    q = as_float(section, "q")
    if rho is None or q is None or rho <= 0.0 or q <= 0.0:
        return float("inf")
    log_rho = math.log(rho)
    log_q = math.log(q)
    return (
        0.5 * (max(math.log(1e-5) - log_rho, 0.0) / 0.5) ** 2
        + 0.5 * (max(log_rho - math.log(0.2), 0.0) / 0.5) ** 2
        + 0.5 * (max(math.log(1e-6) - log_q, 0.0) / 0.5) ** 2
        + 0.5 * (max(log_q - math.log(1.0), 0.0) / 0.5) ** 2
    )


_MODEL_PRIOR_FUNCS = {
    "FSPL": _prior_fspl,
    "PSPL+Parallax": _prior_parallax,
    "FSPL+Parallax": lambda s: _prior_fspl(s) + _prior_parallax(s),
    "BSPL": _prior_bspl,
    "BSPL+Parallax": lambda s: _prior_bspl(s) + _prior_parallax(s),
    "FSBL": _prior_fsbl,
    "FSBL+Parallax": lambda s: _prior_fsbl(s) + _prior_parallax(s),
}


def model_prior_term(model_name: str, section: Dict[str, Any]) -> float:
    """Soft priors used in source/magnification_model.neg_lnprob."""
    prior = 0.0
    fs = as_float(section, "Fs")
    fb = as_float(section, "Fb")
    if fs is not None:
        prior += _negative_flux_prior(fs, sigma=2.0)
    if fb is not None:
        prior += _negative_flux_prior(fb, sigma=2.0)
    prior_fn = _MODEL_PRIOR_FUNCS.get(model_name)
    if prior_fn is not None:
        prior += prior_fn(section)
    return prior


def model_neg_lnprob(model_name: str, section: Dict[str, Any]) -> Optional[float]:
    """Negative log posterior proxy: 0.5*chi2 + priors (matches the optimizer objective)."""
    chi2 = as_float(section, "Chi2")
    if chi2 is None:
        chi2 = as_float(section, "chi2")
    if chi2 is None:
        return None
    prior = model_prior_term(model_name, section)
    if not math.isfinite(prior):
        return None
    return 0.5 * chi2 + prior


def model_aic(model_name: str, section: Dict[str, Any]) -> Optional[float]:
    """AIC = 2k + 2*neg_lnprob, with k counting nonlinear params plus profiled fluxes."""
    neg_lnprob = model_neg_lnprob(model_name, section)
    if neg_lnprob is None:
        return None
    k = model_n_free_params(model_name)
    if k <= 0:
        return None
    return 2.0 * k + 2.0 * neg_lnprob


def neg_lnprob_to_loglike(neg_lnprob: float) -> float:
    return -neg_lnprob


def n_points(section: Dict[str, Any], model_name: Optional[str] = None) -> Optional[int]:
    for key in ["n_data_points", "eval_n_points", "N", "n_valid"]:
        val = as_float(section, key)
        if val is not None:
            return int(round(val))
    chi2 = as_float(section, "Chi2")
    if chi2 is None:
        chi2 = as_float(section, "chi2")
    chi2_dof = as_float(section, "chi2/dof")
    if (
        model_name is not None
        and chi2 is not None
        and chi2_dof is not None
        and chi2_dof > 0.0
        and model_name in MODEL_N_NONLINEAR
    ):
        dof = chi2 / chi2_dof
        return int(round(dof + MODEL_N_NONLINEAR[model_name]))
    return None


def add_uncertainty_metadata(
    row: Dict[str, Any],
    uncertainties: Optional[Dict[str, float]],
    confidence_level: float = 0.68,
    method: str = "fisher_matrix",
    overwrite: bool = False,
) -> None:
    if not uncertainties:
        return
    if row.get("parameter_uncertainties") and not overwrite:
        return
    row["parameter_uncertainties"] = json.dumps(uncertainties)
    row["uncertainty_method"] = method
    row["confidence_level"] = confidence_level


def add_multinest_uncertainties(row: Dict[str, Any], section: Dict[str, Any]) -> None:
    """Attach MultiNest posterior uncertainties stored in a parsed section."""
    unc = section.get("multinest_uncertainties")
    if not isinstance(unc, dict) or not unc:
        return
    method = str(section.get("multinest_uncertainty_method", "nested_sampling"))
    confidence_level = float(section.get("multinest_confidence_level", 0.68))
    add_uncertainty_metadata(row, unc, confidence_level=confidence_level, method=method, overwrite=True)


def load_fisher_event_data(
    event_id: str,
    input_dir: Path,
    coord_file: str,
    max_len: int,
) -> Optional[Dict[str, Any]]:
    from data_loader import DataLoader
    from initial_conditions import InitialConditions

    csv_path = input_dir / f"{event_id}.csv"
    if not csv_path.exists():
        matches = sorted(input_dir.glob(f"*{event_id}*.csv"))
        if not matches:
            return None
        csv_path = matches[0]
    try:
        loader = DataLoader(coord_file=coord_file)
        raw = loader.load_event(str(csv_path))
        init = InitialConditions(raw)
        return init.get_processed_data(max_len=max_len)
    except Exception:
        return None


def attach_fisher_uncertainties(
    rows: List[Dict[str, Any]],
    *,
    input_dir: Path,
    coord_file: str,
    max_len: int,
    uncertainty_max_points: int,
    confidence_level: float,
) -> int:
    from fisher_uncertainties import fisher_submission_uncertainties

    event_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    n_attached = 0
    for row in rows:
        model_name = row.get("_model_name")
        section = row.get("_section")
        if not model_name or not section:
            continue
        event_id = row["event_id"]
        if event_id not in event_cache:
            event_cache[event_id] = load_fisher_event_data(
                event_id, input_dir, coord_file, max_len
            )
        data = event_cache[event_id]
        if data is None:
            continue
        unc = fisher_submission_uncertainties(
            model_name,
            section,
            data,
            confidence_level=confidence_level,
            max_points=uncertainty_max_points,
        )
        if unc:
            add_uncertainty_metadata(row, unc, confidence_level=confidence_level)
            n_attached += 1
    return n_attached


def safe_alias(model_name: str, source_tag: str) -> str:
    alias = model_name.replace("+", "_plus_").replace(" ", "_")
    alias = alias.replace("/", "_").replace("-", "_")
    source_tag = re.sub(r"[^A-Za-z0-9_]+", "_", source_tag)
    return f"{alias}__{source_tag}"[:120]


def add_common_optional(
    row: Dict[str, Any],
    section: Dict[str, Any],
    model_name: str,
    notes: str = "",
) -> None:
    neg_lnprob = model_neg_lnprob(model_name, section)
    if neg_lnprob is not None:
        row["log_likelihood"] = neg_lnprob_to_loglike(neg_lnprob)
        row["_neg_lnprob"] = neg_lnprob
    aic = model_aic(model_name, section)
    if aic is not None:
        row["_aic"] = aic
    n = n_points(section, model_name=model_name)
    if n is not None:
        row["n_data_points"] = n
    if notes:
        row["notes"] = notes


def add_flux_1source(row: Dict[str, Any], section: Dict[str, Any]) -> None:
    fs = as_float(section, "Fs")
    fb = as_float(section, "Fb")
    if fs is not None:
        row["F0_S"] = fs
    if fb is not None:
        row["F0_B"] = fb


def add_flux_2source(row: Dict[str, Any], section: Dict[str, Any]) -> None:
    """Convert our BSPL total source flux + q_f into F0_S1/F0_S2/F0_B."""
    fs = as_float(section, "Fs")
    fb = as_float(section, "Fb")
    qf = as_float(section, "q_f")
    if fs is not None and qf is not None and qf > -0.999999:
        row["F0_S1"] = fs / (1.0 + qf)
        row["F0_S2"] = fs * qf / (1.0 + qf)
    elif fs is not None:
        # Fallback: submit total source flux as source 1 only.
        row["F0_S1"] = fs
        row["F0_S2"] = 0.0
    if fb is not None:
        row["F0_B"] = fb


def convert_section_to_row(
    event_id: str,
    model_name: str,
    section: Dict[str, Any],
    source_tag: str,
    include_inactive: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Convert one model section into one microlens-submit CSV row.

    Model tags follow the official CSV import convention:
      1S1L, 1S2L, 2S1L plus higher-order tags like finite-source/parallax.
    """
    row: Dict[str, Any] = {
        "event_id": event_id,
        "solution_alias": safe_alias(model_name, source_tag),
        "is_active": True if not include_inactive else True,
        "bands": json.dumps(["0"]),
    }

    notes = f"Imported automatically from {source_tag}; original pipeline model: {model_name}."

    # ------------------ 1S1L family ------------------
    if model_name == "PSPL":
        required = ["t_0", "u_0", "t_E"]
        if any(as_float(section, k) is None for k in required):
            return None
        row.update(
            model_tags=json.dumps(["1S1L"]),
            t0=section["t_0"],
            u0=section["u_0"],
            tE=section["t_E"],
        )
        add_flux_1source(row, section)
        add_common_optional(row, section, model_name, notes)
        return row

    if model_name == "FSPL":
        required = ["t_0", "u_0", "t_E", "rho"]
        if any(as_float(section, k) is None for k in required):
            return None
        row.update(
            model_tags=json.dumps(["1S1L", "finite-source"]),
            t0=section["t_0"],
            u0=section["u_0"],
            tE=section["t_E"],
            rho=section["rho"],
        )
        add_flux_1source(row, section)
        add_common_optional(row, section, model_name, notes)
        return row

    if model_name == "PSPL+Parallax":
        required = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
        if any(as_float(section, k) is None for k in required):
            return None
        row.update(
            model_tags=json.dumps(["1S1L", "parallax"]),
            t0=section["t_0"],
            u0=section["u_0"],
            tE=section["t_E"],
            piEN=section["pi_E_N"],
            piEE=section["pi_E_E"],
            t_ref=as_float(section, "t_0_par", as_float(section, "t_0")),
        )
        add_flux_1source(row, section)
        add_common_optional(row, section, model_name, notes)
        return row

    if model_name == "FSPL+Parallax":
        required = ["t_0", "u_0", "t_E", "rho", "pi_E_N", "pi_E_E"]
        if any(as_float(section, k) is None for k in required):
            return None
        row.update(
            model_tags=json.dumps(["1S1L", "finite-source", "parallax"]),
            t0=section["t_0"],
            u0=section["u_0"],
            tE=section["t_E"],
            rho=section["rho"],
            piEN=section["pi_E_N"],
            piEE=section["pi_E_E"],
            t_ref=as_float(section, "t_0_par", as_float(section, "t_0")),
        )
        add_flux_1source(row, section)
        add_common_optional(row, section, model_name, notes)
        return row

    # ------------------ 2S1L / binary source ------------------
    if model_name == "BSPL":
        required = ["t_0_1", "u_0_1", "t_E", "t_0_2", "u_0_2", "q_f"]
        if any(as_float(section, k) is None for k in required):
            return None
        row.update(
            model_tags=json.dumps(["2S1L"]),
            t0=section["t_0_1"],
            u0=section["u_0_1"],
            tE=section["t_E"],
            t0_source2=section["t_0_2"],
            u0_source2=section["u_0_2"],
            flux_ratio=section["q_f"],
        )
        add_flux_2source(row, section)
        add_common_optional(row, section, model_name, notes)
        return row

    if model_name == "BSPL+Parallax":
        required = ["t_0_1", "u_0_1", "t_E", "t_0_2", "u_0_2", "q_f", "pi_E_N", "pi_E_E"]
        if any(as_float(section, k) is None for k in required):
            return None
        row.update(
            model_tags=json.dumps(["2S1L", "parallax"]),
            t0=section["t_0_1"],
            u0=section["u_0_1"],
            tE=section["t_E"],
            t0_source2=section["t_0_2"],
            u0_source2=section["u_0_2"],
            flux_ratio=section["q_f"],
            piEN=section["pi_E_N"],
            piEE=section["pi_E_E"],
            t_ref=as_float(section, "t_0_par", as_float(section, "t_0_1")),
        )
        add_flux_2source(row, section)
        add_common_optional(row, section, model_name, notes)
        return row

    # ------------------ 1S2L / binary lens ------------------
    if model_name == "FSBL":
        required = ["t_0", "u_0", "t_E", "s", "q", "rho"]
        if any(as_float(section, k) is None for k in required) or alpha_to_rad(section) is None:
            return None
        row.update(
            model_tags=json.dumps(["1S2L", "finite-source"]),
            t0=section["t_0"],
            u0=section["u_0"],
            tE=section["t_E"],
            s=section["s"],
            q=section["q"],
            alpha=alpha_to_rad(section),
            rho=section["rho"],
        )
        add_flux_1source(row, section)
        add_common_optional(row, section, model_name, notes)
        add_multinest_uncertainties(row, section)
        return row

    if model_name == "FSBL+Parallax":
        required = ["t_0", "u_0", "t_E", "s", "q", "rho", "pi_E_N", "pi_E_E"]
        if any(as_float(section, k) is None for k in required) or alpha_to_rad(section) is None:
            return None
        row.update(
            model_tags=json.dumps(["1S2L", "finite-source", "parallax"]),
            t0=section["t_0"],
            u0=section["u_0"],
            tE=section["t_E"],
            s=section["s"],
            q=section["q"],
            alpha=alpha_to_rad(section),
            rho=section["rho"],
            piEN=section["pi_E_N"],
            piEE=section["pi_E_E"],
            t_ref=as_float(section, "t_0_par", as_float(section, "t_0")),
        )
        add_flux_1source(row, section)
        add_common_optional(row, section, model_name, notes)
        add_multinest_uncertainties(row, section)
        return row

    return None


def iter_params_files(results_roots: Iterable[Path]) -> Iterable[Path]:
    seen = set()
    for root in results_roots:
        if root.is_file() and root.name.endswith("_params.txt"):
            p = root.resolve()
            if p not in seen:
                seen.add(p)
                yield root
        elif root.exists():
            for p in sorted(root.rglob("*_params.txt")):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield p


def build_rows(
    results_roots: List[Path],
    include_models: Optional[set[str]] = None,
    include_multinest: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Standard pipeline *_params.txt files.
    for params_file in iter_params_files(results_roots):
        event_id = infer_event_id_from_params_file(params_file)
        sections = parse_params_txt(params_file)
        source_tag = str(params_file.parent).replace(os.sep, "_")
        for model_name, section in sections.items():
            if include_models and model_name not in include_models:
                continue
            row = convert_section_to_row(event_id, model_name, section, source_tag=source_tag)
            if row is not None:
                row["_model_name"] = model_name
                row["_source_tag"] = source_tag
                row["_section"] = section
                rows.append(row)

    # PyMultiNest FSBL/FSBL+Parallax outputs, e.g.
    # results/*multinest_FSBL_from_FSPL*/mn_stats.dat and
    # results/*multinest_FSBL_Parallax_from_FSBL*/mn_stats.dat.
    wants_multinest = include_multinest and (
        include_models is None
        or "FSBL" in include_models
        or "FSBL+Parallax" in include_models
    )
    if wants_multinest:
        for mn_dir in iter_multinest_dirs(results_roots):
            event_id = infer_event_id_from_multinest_dir(mn_dir)
            section = parse_multinest_result_dir(mn_dir)
            if section is None:
                print(f"WARNING: Could not parse MultiNest result directory: {mn_dir}")
                continue
            model_name = str(section.get("multinest_model_name", infer_multinest_model_name(mn_dir, section)))
            if include_models and model_name not in include_models:
                continue
            source_tag = str(mn_dir).replace(os.sep, "_") + "_maximum_likelihood"
            row = convert_section_to_row(event_id, model_name, section, source_tag=source_tag)
            if row is not None:
                alias_model = "FSBL_Parallax_MultiNest_ML" if model_name == "FSBL+Parallax" else "FSBL_MultiNest_ML"
                row["solution_alias"] = safe_alias(alias_model, source_tag)
                old_notes = row.get("notes", "")
                unc_note = " MultiNest posterior-sample uncertainties attached." if section.get("multinest_uncertainties") else ""
                row["notes"] = (
                    old_notes
                    + f" Maximum-likelihood {model_name} parameters imported from MultiNest output."
                    + unc_note
                )
                row["_model_name"] = model_name
                row["_source_tag"] = source_tag
                row["_section"] = section
                rows.append(row)

    return rows


def choose_best_per_event(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep lowest -2logL/highest log_likelihood per event. Use carefully."""
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        eid = row["event_id"]
        ll = row.get("log_likelihood")
        if eid not in best:
            best[eid] = row
            continue
        prev_ll = best[eid].get("log_likelihood")
        if ll is not None and (prev_ll is None or float(ll) > float(prev_ll)):
            best[eid] = row
    return list(best.values())


def _row_source_key(row: Dict[str, Any]) -> str:
    return str(row.get("_source_tag", row.get("solution_alias", "")))


def select_by_aic(rows: List[Dict[str, Any]], delta_aic: float = 2.0) -> List[Dict[str, Any]]:
    """
    For each event, pick the result source with the best (lowest) AIC model, then
    keep every model from that source whose AIC is within ``delta_aic`` of the best.

    AIC uses the same objective as the optimizer: 2k + 2*(0.5*chi2 + priors).
    """
    by_event: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("_aic") is None:
            continue
        by_event.setdefault(row["event_id"], []).append(row)

    selected: List[Dict[str, Any]] = []
    skipped_events: List[str] = []

    for event_id, event_rows in sorted(by_event.items()):
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for row in event_rows:
            by_source.setdefault(_row_source_key(row), []).append(row)

        best_source = None
        best_source_min_aic = float("inf")
        for source_tag, source_rows in by_source.items():
            source_min = min(float(r["_aic"]) for r in source_rows)
            if source_min < best_source_min_aic:
                best_source_min_aic = source_min
                best_source = source_tag

        if best_source is None:
            skipped_events.append(event_id)
            continue

        source_rows = by_source[best_source]
        min_aic = min(float(r["_aic"]) for r in source_rows)
        cutoff = min_aic + float(delta_aic)
        competitive = [r for r in source_rows if float(r["_aic"]) <= cutoff + 1e-9]
        competitive.sort(key=lambda r: float(r["_aic"]))

        weight_sum = sum(math.exp(-0.5 * (float(r["_aic"]) - min_aic)) for r in competitive)
        for rank, row in enumerate(competitive, start=1):
            delta = float(row["_aic"]) - min_aic
            rel_prob = math.exp(-0.5 * delta) / weight_sum if weight_sum > 0.0 else 1.0
            row = dict(row)
            row["relative_probability"] = rel_prob
            row["is_active"] = True
            old_notes = row.get("notes", "")
            row["notes"] = (
                f"{old_notes} Selected by AIC (rank {rank}/{len(competitive)}; "
                f"AIC={float(row['_aic']):.6g}, ΔAIC={delta:.6g} vs best in source)."
            ).strip()
            selected.append(row)

    if skipped_events:
        print(
            f"WARNING: skipped {len(skipped_events)} event(s) with no AIC-scorable rows: "
            + ", ".join(skipped_events[:5])
            + (" ..." if len(skipped_events) > 5 else "")
        )

    return selected


def strip_internal_row_keys(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    internal = {"_aic", "_neg_lnprob", "_model_name", "_source_tag", "_section"}
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if k not in internal})
    return cleaned


def _parse_model_tags(model_tags_value: Any) -> Tuple[str, List[str]]:
    tags = json.loads(model_tags_value) if isinstance(model_tags_value, str) else model_tags_value
    if not isinstance(tags, list):
        raise ValueError("model_tags must be a JSON list")
    allowed = {"1S1L", "1S2L", "2S1L", "2S2L", "1S3L", "2S3L", "other"}
    hoe = []
    model_type = None
    for tag in tags:
        if tag in allowed:
            if model_type is not None:
                raise ValueError("multiple model types in model_tags")
            model_type = tag
        else:
            hoe.append(tag)
    if model_type is None:
        raise ValueError("no model type in model_tags")
    return model_type, hoe


def _row_submission_parameters(row: Dict[str, Any]) -> Dict[str, Any]:
    skip = {
        "event_id",
        "solution_id",
        "solution_alias",
        "model_tags",
        "bands",
        "notes",
        "parameters",
        "parameter_uncertainties",
        "physical_parameters",
        "physical_parameter_uncertainties",
        "uncertainty_method",
        "confidence_level",
        "log_likelihood",
        "relative_probability",
        "n_data_points",
        "is_active",
        "creation_timestamp",
    }
    params: Dict[str, Any] = {}
    for key, value in row.items():
        if key in skip or key.startswith("_"):
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, (int, float)):
            params[key] = float(value)
        elif isinstance(value, str):
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
        else:
            params[key] = value
    return params


def import_rows_with_microlens_api(
    rows: List[Dict[str, Any]],
    submission_dir: Path,
    *,
    validate: bool = True,
) -> None:
    """Import rows via microlens-submit Python API (supports uncertainty metadata)."""
    from microlens_submit.utils import load

    submission_dir.mkdir(parents=True, exist_ok=True)
    sub = load(str(submission_dir))

    for row in rows:
        model_type, higher_order = _parse_model_tags(row["model_tags"])
        params = _row_submission_parameters(row)
        event = sub.get_event(row["event_id"])
        sol = event.add_solution(model_type, params)
        if row.get("solution_alias"):
            sol.alias = row["solution_alias"]
        if higher_order:
            sol.higher_order_effects = higher_order
        if row.get("bands"):
            try:
                sol.bands = json.loads(row["bands"]) if isinstance(row["bands"], str) else row["bands"]
            except json.JSONDecodeError:
                pass
        if row.get("notes"):
            sol.set_notes(str(row["notes"]), submission_dir, convert_escapes=True)
        if row.get("log_likelihood") is not None:
            sol.log_likelihood = float(row["log_likelihood"])
        if row.get("relative_probability") is not None:
            sol.relative_probability = float(row["relative_probability"])
        if row.get("n_data_points") is not None:
            sol.n_data_points = int(round(float(row["n_data_points"])))
        if row.get("is_active") is not None:
            sol.is_active = bool(row["is_active"])
        if row.get("parameter_uncertainties"):
            unc = row["parameter_uncertainties"]
            sol.parameter_uncertainties = (
                json.loads(unc) if isinstance(unc, str) else unc
            )
        if row.get("uncertainty_method"):
            sol.uncertainty_method = str(row["uncertainty_method"])
        if row.get("confidence_level") is not None:
            sol.confidence_level = float(row["confidence_level"])
        if row.get("t_ref") is not None:
            sol.t_ref = float(row["t_ref"])
        if validate:
            sol.run_validation()
    sub.save()


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    preferred = [
        "event_id",
        "solution_alias",
        "model_tags",
        "bands",
        "t0",
        "u0",
        "tE",
        "s",
        "q",
        "alpha",
        "rho",
        "piEN",
        "piEE",
        "t_ref",
        "t0_source2",
        "u0_source2",
        "flux_ratio",
        "F0_S",
        "F0_B",
        "F0_S1",
        "F0_S2",
        "log_likelihood",
        "relative_probability",
        "n_data_points",
        "parameter_uncertainties",
        "uncertainty_method",
        "confidence_level",
        "is_active",
        "notes",
    ]
    keys = set().union(*(r.keys() for r in rows)) if rows else set(preferred)
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def maybe_run_microlens_submit(
    *,
    rows: List[Dict[str, Any]],
    csv_path: Path,
    submission_dir: Path,
    team_name: str,
    tier: str,
    repo_url: str,
    git_dir: Optional[str],
    generate_dossier: bool,
    export_zip: Optional[Path],
) -> None:
    exe = shutil.which("microlens-submit")
    if exe is None:
        raise RuntimeError(
            "microlens-submit is not on PATH. Install it first: pip install microlens-submit"
        )

    submission_dir.mkdir(parents=True, exist_ok=True)

    init_cmd = [exe, "init", "--team-name", team_name, "--tier", tier]
    run_cmd(init_cmd, cwd=submission_dir)

    sub_json = submission_dir / "submission.json"
    if sub_json.exists():
        data = json.loads(sub_json.read_text(encoding="utf-8"))
    else:
        data = {}
    data["team_name"] = team_name
    data["tier"] = tier
    data["repo_url"] = repo_url
    if git_dir:
        data["git_dir"] = git_dir
    data.setdefault("hardware_info", {"cpu": "unknown", "ram_gb": None})
    sub_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    import_rows_with_microlens_api(rows, submission_dir, validate=True)
    print(f"Imported {len(rows)} solution(s) via microlens-submit Python API.")

    if generate_dossier:
        run_cmd([exe, "generate-dossier"], cwd=submission_dir, check=False)

    if export_zip is not None:
        export_zip = export_zip.resolve()
        try:
            run_cmd([exe, "export-submission", str(export_zip)], cwd=submission_dir)
        except subprocess.CalledProcessError:
            run_cmd([exe, "export", str(export_zip)], cwd=submission_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert pipeline *_params.txt files into microlens-submit CSV and optionally package submission."
    )
    ap.add_argument("--results", nargs="+", required=True, help="Result roots or individual *_params.txt files.")
    ap.add_argument("--csv", default="submission_solutions.csv", help="Output CSV for microlens-submit import-solutions.")
    ap.add_argument(
        "--models",
        default="PSPL,FSPL,PSPL+Parallax,FSPL+Parallax,BSPL,BSPL+Parallax,FSBL,FSBL+Parallax",
        help="Comma-separated model sections to include.",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="Export every fitted model row without AIC-based selection.",
    )
    ap.add_argument(
        "--aic-delta",
        type=float,
        default=2.0,
        help="When using AIC selection, include models within this ΔAIC of the best (default: 2).",
    )
    ap.add_argument("--best-per-event", action="store_true", help="Only submit one best log-likelihood row per event.")
    ap.add_argument(
        "--no-multinest",
        action="store_true",
        help="Do not scan MultiNest output dirs such as *multinest_FSBL_from_FSPL/mn_stats.dat.",
    )
    ap.add_argument("--run-microlens-submit", action="store_true", help="Run microlens-submit init/import/export.")
    ap.add_argument("--submission-dir", default="rmdc26_submission", help="Submission project directory.")
    ap.add_argument("--team-name", default="Barbara Bialek", help="Team name for submission.json.")
    ap.add_argument("--tier", default="experienced", choices=["beginner", "experienced"], help="Challenge tier.")
    ap.add_argument("--repo-url", default="https://github.com/YOUR/REPO", help="Public code repository URL.")
    ap.add_argument("--git-dir", default=None, help="Optional local git directory.")
    ap.add_argument("--no-dossier", action="store_true", help="Skip dossier generation.")
    ap.add_argument("--export-zip", default="final_submission.zip", help="Output submission zip path, if running microlens-submit.")
    ap.add_argument("--input-dir", default="data/data_F146", help="Light-curve CSV directory for Fisher uncertainties.")
    ap.add_argument("--coord-file", default="data/coords.csv", help="Event coordinates for parallax Fisher errors.")
    ap.add_argument("--max-len", type=int, default=46_208, help="Max points when loading light curves for Fisher errors.")
    ap.add_argument(
        "--no-uncertainties",
        action="store_true",
        help="Skip Fisher-matrix parameter_uncertainties in the submission output.",
    )
    ap.add_argument(
        "--uncertainty-max-points",
        type=int,
        default=2048,
        help="Subsample light-curve points when evaluating the Fisher matrix.",
    )
    ap.add_argument(
        "--confidence-level",
        type=float,
        default=0.68,
        help="Gaussian confidence level for reported Fisher uncertainties (default: 0.68).",
    )
    args = ap.parse_args()

    roots = [Path(x) for x in args.results]
    include_models = {m.strip() for m in args.models.split(",") if m.strip()}
    rows = build_rows(roots, include_models=include_models, include_multinest=not args.no_multinest)

    if args.best_per_event:
        rows = choose_best_per_event(rows)
    elif not args.all_models:
        n_before = len(rows)
        rows = select_by_aic(rows, delta_aic=args.aic_delta)
        n_multi = sum(
            1
            for eid in {r["event_id"] for r in rows}
            if sum(1 for r in rows if r["event_id"] == eid) > 1
        )
        print(
            f"AIC selection (ΔAIC <= {args.aic_delta}): kept {len(rows)} row(s) "
            f"from {n_before} candidate(s); {n_multi} event(s) have multiple solutions."
        )

    if not args.no_uncertainties:
        n_unc = attach_fisher_uncertainties(
            rows,
            input_dir=Path(args.input_dir),
            coord_file=args.coord_file,
            max_len=args.max_len,
            uncertainty_max_points=args.uncertainty_max_points,
            confidence_level=args.confidence_level,
        )
        print(f"Attached Fisher-matrix uncertainties to {n_unc} row(s).")

    rows = strip_internal_row_keys(rows)

    if not rows:
        raise SystemExit("No valid solution rows found. Check --results and model names.")

    out_csv = Path(args.csv)
    write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} solution row(s) to {out_csv}")

    # Simple warnings that often catch mistakes.
    n_alpha = sum(1 for r in rows if "alpha" in r)
    if n_alpha:
        print(f"Converted alpha to radians for {n_alpha} binary-lens solution(s).")
    missing_flux = [r for r in rows if not any(k in r for k in ["F0_S", "F0_S1"])]
    if missing_flux:
        print(f"WARNING: {len(missing_flux)} row(s) have no source flux columns.")

    if args.run_microlens_submit:
        maybe_run_microlens_submit(
            rows=rows,
            csv_path=out_csv,
            submission_dir=Path(args.submission_dir),
            team_name=args.team_name,
            tier=args.tier,
            repo_url=args.repo_url,
            git_dir=args.git_dir,
            generate_dossier=not args.no_dossier,
            export_zip=Path(args.export_zip) if args.export_zip else None,
        )


if __name__ == "__main__":
    main()
