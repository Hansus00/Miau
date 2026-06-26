"""
Build a microlens-submit CSV and optionally create/export a submission project
from this pipeline's *_params.txt result files.

By default, one row per event is not exported: the script scores every fitted
model with AIC = 2k + 2*(0.5*chi2 + priors), using the same priors as the
optimizer in source/magnification_model.py. For each event it keeps every model
from the best result source whose AIC lies within --aic-delta of the minimum
(default 2). Use --all-models to export every fit without selection.

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

MN_THETA_NAMES = ["t0", "log_tE", "u0", "log_s", "log_q", "log_rho", "alpha_deg"]


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
    Parse maximum-likelihood nonlinear FSBL parameters from mn_stats.dat.

    MultiNest/PyMultiNest formats differ slightly. This parser looks for a block
    containing 'Maximum Likelihood Parameters' and then extracts ndim numerical
    parameter values. It handles both formats like:

        1   2461854.8
        2   1.23

    and simple one-value-per-line formats.

    The assumed parameter order is the order used in source/multinest_cpu.py:
        t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg

    Raw mn_stats.dat does not contain Fs/Fb/chi2. If best_fit.txt is present,
    parse_multinest_result_dir() will merge those values from there.
    """
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for i, line in enumerate(text):
        low = line.lower()
        if "maximum" in low and "likelihood" in low and "parameter" in low:
            start = i + 1
            break
    if start is None:
        return {}

    vals: List[float] = []
    expected_index = 1
    for line in text[start:]:
        low = line.strip().lower()
        if not low:
            # allow one blank, but stop if we already found values
            if vals:
                break
            continue
        if any(marker in low for marker in ["maximum a posteriori", "marginal", "mean", "mode", "evidence", "posterior"]):
            if vals:
                break

        nums = _extract_floats_from_line(line)
        if not nums:
            continue

        # Common MultiNest layout: '<index> <value>'
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
    return section


def parse_multinest_result_dir(path: Path) -> Optional[Dict[str, Any]]:
    """
    Return an FSBL-like section from a MultiNest output directory.

    Preference order:
      1. best_fit.txt, because it contains Fs/Fb/chi2 written by our code.
      2. mn_stats.dat maximum-likelihood parameters.

    If both exist, parameters from best_fit.txt win, but missing values are filled
    from mn_stats.dat.
    """
    best = parse_multinest_best_fit_txt(path / "best_fit.txt")
    stats = parse_mn_stats_maxlike(path / "mn_stats.dat")

    if not best and not stats:
        return None

    merged = dict(stats)
    merged.update(best)

    required = ["t_0", "u_0", "t_E", "s", "q", "rho", "alpha_deg"]
    if any(as_float(merged, k) is None for k in required):
        return None

    # If chi2/Fs/Fb are missing, keep the row; the CSV validator may complain
    # about missing fluxes, but this lets the user see and fix incomplete outputs.
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
                rows.append(row)

    # PyMultiNest FSBL outputs, e.g. results/*multinest_FSBL_from_FSPL*/mn_stats.dat.
    if include_multinest and (include_models is None or "FSBL" in include_models):
        for mn_dir in iter_multinest_dirs(results_roots):
            event_id = infer_event_id_from_multinest_dir(mn_dir)
            section = parse_multinest_result_dir(mn_dir)
            if section is None:
                print(f"WARNING: Could not parse MultiNest result directory: {mn_dir}")
                continue
            source_tag = str(mn_dir).replace(os.sep, "_") + "_maximum_likelihood"
            row = convert_section_to_row(event_id, "FSBL", section, source_tag=source_tag)
            if row is not None:
                row["solution_alias"] = safe_alias("FSBL_MultiNest_ML", source_tag)
                old_notes = row.get("notes", "")
                row["notes"] = old_notes + " Maximum-likelihood FSBL parameters imported from MultiNest output."
                row["_model_name"] = "FSBL"
                row["_source_tag"] = source_tag
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
            row["aic"] = float(row["_aic"])
            row["relative_probability"] = rel_prob
            row["is_active"] = True
            old_notes = row.get("notes", "")
            row["notes"] = (
                f"{old_notes} Selected by AIC (rank {rank}/{len(competitive)}; "
                f"AIC={row['aic']:.6g}, ΔAIC={delta:.6g} vs best in source)."
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
    internal = {"_aic", "_neg_lnprob", "_model_name", "_source_tag"}
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if k not in internal})
    return cleaned


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
        "aic",
        "n_data_points",
        "relative_probability",
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
    # CLI options may differ slightly between versions. We do repo metadata by
    # JSON patch below as a robust fallback.
    run_cmd(init_cmd, cwd=submission_dir)

    # Patch top-level metadata robustly.
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

    # Import CSV. Path should be absolute because cwd=submission_dir.
    run_cmd([exe, "import-solutions", str(csv_path.resolve()), "--validate"], cwd=submission_dir)

    if generate_dossier:
        run_cmd([exe, "generate-dossier"], cwd=submission_dir, check=False)

    if export_zip is not None:
        export_zip = export_zip.resolve()
        # Try modern command first; fallback to older style if needed.
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
