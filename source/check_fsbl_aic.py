"""Decide whether the FSBL (binary-lens) model has the best AIC for an event.

Compares AIC = chi2 + 2k across:
  - every model section in the simple-fit EVENT_params.txt (source/run.py output)
  - the FSBL Twinkle/MultiNest result in EVENT_multinest_FSBL_from_FSPL/best_fit.txt

Exits 0 (and prints "RUN") iff FSBL's AIC is the minimum (strictly better than every
other available model). Exits 1 (and prints "SKIP: <reason>") otherwise, including
when required files/sections are missing.

Usage:
    python check_fsbl_aic.py --params-file results/optax_results/EVENT_params.txt \
                              --fsbl-best-fit results/EVENT_multinest_FSBL_from_FSPL/best_fit.txt
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
    flags=re.IGNORECASE,
)

# Number of free parameters (k) per simple-fit model, matching models.py param_names.
SIMPLE_MODEL_K: Dict[str, int] = {
    "PSPL": 3,
    "FSPL": 4,
    "PSPL+Parallax": 5,
    "FSPL+Parallax": 6,
    "BSPL": 6,
    "BSPL+Parallax": 8,
    "FSBLGrid": 7,
}

# FSBL (Twinkle) nonlinear-parameter count: t0, log_tE, u0, log_s, log_q, log_rho, alpha_deg.
FSBL_K = 7


def first_float(text: str) -> Optional[float]:
    m = FLOAT_RE.search(str(text))
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_params_txt(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse EVENT_params.txt into {model_name: {key: value}}."""
    sections: Dict[str, Dict[str, float]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections[current] = {}
            continue
        if current is None or ":" not in line:
            continue
        key, val = line.split(":", 1)
        f = first_float(val)
        if f is not None:
            sections[current][key.strip()] = f
    return sections


def parse_key_value_file(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        f = first_float(val)
        if f is not None:
            out[key.strip()] = f
    return out


def collect_aics(params_file: Path, fsbl_best_fit: Path) -> Dict[str, float]:
    aics: Dict[str, float] = {}

    sections = parse_params_txt(params_file)
    for model_name, k in SIMPLE_MODEL_K.items():
        sec = sections.get(model_name)
        if not sec or "Chi2" not in sec:
            continue
        chi2 = sec["Chi2"]
        if not math.isfinite(chi2):
            continue
        aics[model_name] = chi2 + 2 * k

    fsbl_raw = parse_key_value_file(fsbl_best_fit)
    if "chi2" in fsbl_raw and math.isfinite(fsbl_raw["chi2"]):
        aics["FSBL"] = fsbl_raw["chi2"] + 2 * FSBL_K

    return aics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-file", required=True, type=Path)
    ap.add_argument("--fsbl-best-fit", required=True, type=Path)
    args = ap.parse_args()

    if not args.params_file.exists():
        print(f"SKIP: missing params file {args.params_file}")
        return 1
    if not args.fsbl_best_fit.exists():
        print(f"SKIP: missing FSBL best_fit.txt {args.fsbl_best_fit}")
        return 1

    aics = collect_aics(args.params_file, args.fsbl_best_fit)

    if "FSBL" not in aics:
        print("SKIP: could not compute FSBL AIC (missing/invalid chi2 in best_fit.txt)")
        return 1
    if len(aics) < 2:
        print("SKIP: no other model AICs available to compare against")
        return 1

    best_model = min(aics, key=aics.get)
    for name in sorted(aics, key=aics.get):
        marker = " <-- best" if name == best_model else ""
        print(f"  AIC[{name}] = {aics[name]:.4f}{marker}")

    if best_model == "FSBL":
        print("RUN: FSBL has the best (lowest) AIC")
        return 0

    print(f"SKIP: best AIC model is {best_model}, not FSBL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
