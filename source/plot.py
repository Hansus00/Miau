from __future__ import annotations

import argparse
import importlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Make imports from ./source work when this script is in project root
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent if THIS_DIR.name == "source" else THIS_DIR
SOURCE_DIR = PROJECT_ROOT / "source"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import jax.numpy as jnp
    from magnification_model import magnification
except Exception as exc:
    raise RuntimeError(
        "Could not import project magnification_model. "
        "Run this script from the project root, or place it next to the source/ directory."
    ) from exc


# ============================================================
# Model names
# ============================================================

MODEL_TO_INTERNAL = {
    "PSPL": "pspl",
    "PSPL+Parallax": "parallax",
    "FSPL": "fspl",
    "FSPL+Parallax": "fspl_parallax",
    "BSPL": "bspl",
    "BSPL+Parallax": "bspl_parallax",
    "FSBL": "fsbl",
    # FSBL+Parallax is evaluated directly with Twinkle in this plotting script.
    "FSBL+Parallax": "fsbl_parallax",
}

DEFAULT_MODEL_ORDER = [
    "PSPL",
    "FSPL",
    "PSPL+Parallax",
    "FSPL+Parallax",
    "BSPL",
    "BSPL+Parallax",
    "FSBL",
    "FSBL+Parallax",
]

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
    flags=re.IGNORECASE,
)

_TWINKLE_ENGINE_CACHE: dict[tuple[int, int, int, float], object] = {}
_EPHEMERIS_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
_PROJECTOR_CACHE: dict[tuple[str, str, int, float], tuple[np.ndarray, np.ndarray]] = {}


# ============================================================
# Small utilities
# ============================================================

def extract_first_float(text: str) -> float:
    """
    Extract first float from strings like:
        '123.45'
        'Array(123.45, dtype=float64)'
        'nan'
    """
    text = str(text).strip()

    if text.lower() in {"nan", "+nan", "-nan"}:
        return np.nan

    match = FLOAT_RE.search(text)
    if match is None:
        raise ValueError(f"Cannot parse float from: {text!r}")

    return float(match.group(0))


def parse_value_maybe(text: str):
    """Parse a float if possible; otherwise keep a cleaned string."""
    s = str(text).strip()
    try:
        return extract_first_float(s)
    except ValueError:
        return s


def sanitize_event_name(event: str) -> str:
    event = Path(event).name
    if event.endswith(".csv"):
        event = event[:-4]
    if event.endswith("_params.txt"):
        event = event[:-11]
    return event


def find_data_file(event: str, data_dir: Path) -> Path:
    """Find event CSV. First tries exact EVENT.csv, then fuzzy search."""
    event = sanitize_event_name(event)

    direct = data_dir / f"{event}.csv"
    if direct.exists():
        return direct

    matches = sorted(data_dir.glob(f"*{event}*.csv"))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print("Found multiple possible data files:")
        for m in matches[:20]:
            print(f"  {m}")
        raise RuntimeError("Please give a more specific --event name or use --data-file.")

    raise FileNotFoundError(f"Could not find data file for event {event!r} in {data_dir}")


def find_result_file(event: str, results_dir: Path) -> Path:
    """Find result params file. First tries exact EVENT_params.txt, then fuzzy search."""
    event = sanitize_event_name(event)

    direct = results_dir / f"{event}_params.txt"
    if direct.exists():
        return direct

    matches = sorted(results_dir.glob(f"*{event}*_params.txt"))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print("Found multiple possible result files:")
        for m in matches[:20]:
            print(f"  {m}")
        raise RuntimeError("Please give a more specific --event name or use --result-file.")

    raise FileNotFoundError(f"Could not find result file for event {event!r} in {results_dir}")


def as_float(d: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key, default)
    if v is None:
        return None
    try:
        out = float(v)
    except Exception:
        return default
    return out if np.isfinite(out) else default


# ============================================================
# Loading data/results
# ============================================================

def load_lightcurve_csv(path: Path) -> dict:
    """
    Loads original Roman challenge CSV in columns:
        bjd, mag, mag_err

    Also computes flux using the same convention as DataLoader:
        flux = 10^(-0.4 * (mag - 22))
    """
    df = pd.read_csv(path, header=None, names=["bjd", "mag", "mag_err"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("bjd")

    if len(df) == 0:
        raise RuntimeError(f"No valid rows in {path}")

    t = df["bjd"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    mag_err = df["mag_err"].to_numpy(dtype=float)

    flux = 10.0 ** (-0.4 * (mag - 22.0))
    flux_err = flux * (0.4 * np.log(10.0)) * mag_err

    finite = (
        np.isfinite(t)
        & np.isfinite(mag)
        & np.isfinite(mag_err)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
        & (flux > 0.0)
        & (flux_err > 0.0)
    )

    return {
        "t": t[finite],
        "mag": mag[finite],
        "mag_err": mag_err[finite],
        "flux": flux[finite],
        "flux_err": flux_err[finite],
    }


def load_coords(event_name: str, coord_file: Path) -> np.ndarray:
    """
    Load [ra_deg, dec_deg] from coords.csv if available.
    If missing, return [0, 0]. Non-parallax models do not care.
    """
    if not coord_file.exists():
        return np.array([0.0, 0.0], dtype=float)

    df = pd.read_csv(coord_file)
    event_name = sanitize_event_name(event_name)

    if "name" not in df.columns:
        return np.array([0.0, 0.0], dtype=float)

    row = df[df["name"].astype(str) == event_name]
    if len(row) == 0:
        row = df[df["name"].astype(str).str.contains(event_name, regex=False)]

    if len(row) == 0:
        return np.array([0.0, 0.0], dtype=float)

    r = row.iloc[0]
    return np.array([float(r["ra_deg"]), float(r["dec_deg"])], dtype=float)


def parse_params_file(path: Path) -> dict:
    """
    Parse results/optax_results/EVENT_params.txt.

    Expected format:
        [PSPL]
        t_0: ...
        t_E: ...
        ...
        Chi2: ...
        Fs: ...
        Fb: ...
    """
    results = {}
    current_model = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_model = line[1:-1]
                results[current_model] = {}
                continue

            if current_model is None or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            results[current_model][key] = parse_value_maybe(value)

    return results


# ============================================================
# MultiNest FSBL/best_fit readers
# ============================================================

def parse_best_fit_txt(path: Path) -> tuple[Optional[str], dict]:
    """
    Parse best_fit.txt written by multinest_twinkle.py or multinest_twinkle_parallax.py.

    Returns (model_name, section) where model_name is FSBL or FSBL+Parallax.
    """
    if not path.exists():
        return None, {}

    raw: Dict[str, object] = {}
    model_name: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key.lower() == "model":
                if "parallax" in val.lower():
                    model_name = "FSBL+Parallax"
                elif "fsbl" in val.lower():
                    model_name = "FSBL"
                continue
            raw[key] = parse_value_maybe(val)

    section: dict = {}
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

    if as_float(raw, "pi_E_N") is not None:
        section["pi_E_N"] = raw["pi_E_N"]
    if as_float(raw, "pi_E_E") is not None:
        section["pi_E_E"] = raw["pi_E_E"]
    if as_float(raw, "t_0_par") is not None:
        section["t_0_par"] = raw["t_0_par"]

    required = ["t_0", "u_0", "t_E", "s", "q", "rho", "alpha_deg"]
    if any(as_float(section, k) is None for k in required):
        return None, {}

    if model_name is None:
        model_name = "FSBL+Parallax" if ("pi_E_N" in section and "pi_E_E" in section) else "FSBL"
    section["multinest_source"] = str(path)
    return model_name, section


def load_multinest_fsbl_results(
    *,
    event_name: str,
    params_by_model: dict,
    results_root: Path,
    fsbl_dir: Optional[Path] = None,
    fsbl_parallax_dir: Optional[Path] = None,
    auto: bool = True,
) -> None:
    """Add/overwrite FSBL and FSBL+Parallax sections from MultiNest best_fit.txt files."""
    event = sanitize_event_name(event_name)
    candidates: list[Path] = []

    if fsbl_dir is not None:
        candidates.append(fsbl_dir / "best_fit.txt" if fsbl_dir.is_dir() else fsbl_dir)
    if fsbl_parallax_dir is not None:
        candidates.append(fsbl_parallax_dir / "best_fit.txt" if fsbl_parallax_dir.is_dir() else fsbl_parallax_dir)

    if auto and results_root.exists():
        common = [
            results_root / f"{event}_multinest_FSBL_from_FSPL" / "best_fit.txt",
            results_root / f"{event}_multinest_FSBL_Parallax_from_FSBL" / "best_fit.txt",
            results_root / f"{event}_multinest_FSBL_plus_Parallax_from_FSBL" / "best_fit.txt",
        ]
        candidates.extend(common)
        candidates.extend(sorted(results_root.glob(f"{event}_multinest*FSBL*/*best_fit.txt")))

    seen = set()
    for path in candidates:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        model_name, section = parse_best_fit_txt(path)
        if model_name is None:
            print(f"Warning: could not parse MultiNest best_fit file: {path}")
            continue
        params_by_model[model_name] = section
        print(f"Loaded {model_name} from MultiNest: {path}")


# ============================================================
# Parallax helpers for FSBL+Parallax plotting
# ============================================================

def _parse_horizons_xyz_line(line: str) -> Optional[Tuple[float, float, float, float]]:
    jd_match = re.search(r"\b(24\d{5,}(?:\.\d+)?)\b", line)
    if jd_match is None:
        return None
    x_match = re.search(r"\bX\s*=\s*(%s)" % FLOAT_RE.pattern, line, flags=re.IGNORECASE)
    y_match = re.search(r"\bY\s*=\s*(%s)" % FLOAT_RE.pattern, line, flags=re.IGNORECASE)
    z_match = re.search(r"\bZ\s*=\s*(%s)" % FLOAT_RE.pattern, line, flags=re.IGNORECASE)
    if x_match and y_match and z_match:
        return float(jd_match.group(1)), float(x_match.group(1)), float(y_match.group(1)), float(z_match.group(1))
    return None


def load_ephemeris_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return times JD and heliocentric observer positions [x,y,z] in AU."""
    path = Path(path)
    if path in _EPHEMERIS_CACHE:
        return _EPHEMERIS_CACHE[path]

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("$$"):
                continue
            parsed = _parse_horizons_xyz_line(line)
            if parsed is not None:
                rows.append(parsed)
                continue
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
    _, idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[np.sort(idx)]
    out = (arr[:, 0], arr[:, 1:4])
    _EPHEMERIS_CACHE[path] = out
    return out


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
    """Build geocentric projected ephemeris offsets at observation times."""
    _, east, north = sky_basis(ra_deg, dec_deg)
    r = interp_position(ephem_t, ephem_xyz, t)
    r0 = interp_position(ephem_t, ephem_xyz, np.asarray([t_ref], dtype=float))[0]

    dt = min(1.0, max(0.01, 0.05 * (ephem_t[-1] - ephem_t[0]) / max(len(ephem_t), 2)))
    rp = interp_position(ephem_t, ephem_xyz, np.asarray([t_ref + dt], dtype=float))[0]
    rm = interp_position(ephem_t, ephem_xyz, np.asarray([t_ref - dt], dtype=float))[0]
    v0 = (rp - rm) / (2.0 * dt)

    delta = r - (r0[None, :] + (t - t_ref)[:, None] * v0[None, :])
    d_e = delta @ east
    d_n = delta @ north
    return d_e.astype(np.float64), d_n.astype(np.float64)


# ============================================================
# Building model dictionaries
# ============================================================

def attach_hidden_parallax_fields(params_by_model: dict, event_name: str, coords: np.ndarray) -> None:
    """
    Param files contain fitted parameters, Fs, Fb, Chi2 etc.,
    but not hidden bookkeeping fields used by parallax:
        t_0_par
        coords

    Reconstruct them here.
    """
    # PSPL+Parallax was fitted with t_0_par from PSPL.
    if "PSPL+Parallax" in params_by_model:
        if "PSPL" in params_by_model and "t_0" in params_by_model["PSPL"]:
            params_by_model["PSPL+Parallax"]["t_0_par"] = params_by_model["PSPL"]["t_0"]
        else:
            params_by_model["PSPL+Parallax"]["t_0_par"] = params_by_model["PSPL+Parallax"].get("t_0", 0.0)
        params_by_model["PSPL+Parallax"]["coords"] = coords

    # FSPL+Parallax was fitted with t_0_par from FSPL.
    if "FSPL+Parallax" in params_by_model:
        if "FSPL" in params_by_model and "t_0" in params_by_model["FSPL"]:
            params_by_model["FSPL+Parallax"]["t_0_par"] = params_by_model["FSPL"]["t_0"]
        else:
            params_by_model["FSPL+Parallax"]["t_0_par"] = params_by_model["FSPL+Parallax"].get("t_0", 0.0)
        params_by_model["FSPL+Parallax"]["coords"] = coords

    # BSPL+Parallax was fitted with t_0_par from PSPL.
    if "BSPL+Parallax" in params_by_model:
        if "PSPL" in params_by_model and "t_0" in params_by_model["PSPL"]:
            params_by_model["BSPL+Parallax"]["t_0_par"] = params_by_model["PSPL"]["t_0"]
        else:
            params_by_model["BSPL+Parallax"]["t_0_par"] = params_by_model["BSPL+Parallax"].get("t_0_1", 0.0)
        params_by_model["BSPL+Parallax"]["coords"] = coords

    # FSBL+Parallax was fitted with t_0_par from FSBL.
    if "FSBL+Parallax" in params_by_model:
        if "t_0_par" not in params_by_model["FSBL+Parallax"]:
            if "FSBL" in params_by_model and "t_0" in params_by_model["FSBL"]:
                params_by_model["FSBL+Parallax"]["t_0_par"] = params_by_model["FSBL"]["t_0"]
            else:
                params_by_model["FSBL+Parallax"]["t_0_par"] = params_by_model["FSBL+Parallax"].get("t_0", 0.0)
        params_by_model["FSBL+Parallax"]["coords"] = coords


def build_magnification_params(model_name: str, parsed: dict) -> dict:
    """Convert section from params file into dictionary accepted by magnification()."""
    if model_name not in MODEL_TO_INTERNAL:
        raise ValueError(f"Unknown model name: {model_name}")

    d = dict(parsed)
    d["model"] = MODEL_TO_INTERNAL[model_name]

    # Remove non-physical bookkeeping keys if present.
    for key in [
        "Chi2",
        "chi2/dof",
        "Fs",
        "Fb",
        "n_starts",
        "best_start_idx",
        "best_objective_2lnpost",
        "eval_n_points",
        "multinest_source",
        "_event_name",
        "_coord_file",
        "_ephemeris_file",
        "_twinkle_device",
        "_twinkle_n_stream",
        "_twinkle_reltol",
        "model",
    ]:
        d.pop(key, None)

    # Re-add internal model key after removing possible best_fit 'model' key.
    d["model"] = MODEL_TO_INTERNAL[model_name]

    if "coords" in d:
        d["coords"] = jnp.asarray(d["coords"], dtype=jnp.float64)

    return d


# ============================================================
# Twinkle FSBL evaluation
# ============================================================

def import_twinkle_module(twinkle_python_dir: Optional[str] = None):
    extra_dir = twinkle_python_dir or os.environ.get("TWINKLE_PYTHON_DIR")
    if extra_dir and extra_dir not in sys.path:
        sys.path.insert(0, extra_dir)
    try:
        twinkle = importlib.import_module("twinkle")
    except Exception as exc:
        raise RuntimeError(
            "Could not import Twinkle. Set TWINKLE_PYTHON_DIR to the compiled "
            "AsterLight0626/Twinkle/python directory, e.g. export TWINKLE_PYTHON_DIR=$HOME/twinkle_python"
        ) from exc
    if not hasattr(twinkle, "Twinkle"):
        raise RuntimeError(
            "Imported module 'twinkle' has no Twinkle class. You are probably importing the pip package named twinkle. "
            f"Imported from: {getattr(twinkle, '__file__', '<unknown>')}"
        )
    return twinkle


def make_twinkle_engine(twinkle, n_srcs: int, device_num: int = 0, n_stream: int = 1, reltol: float = 1e-4):
    key = (int(n_srcs), int(device_num), int(n_stream), float(reltol))
    if key in _TWINKLE_ENGINE_CACHE:
        return _TWINKLE_ENGINE_CACHE[key]
    try:
        engine = twinkle.Twinkle(int(n_srcs), int(device_num), int(n_stream), float(reltol), False)
    except TypeError:
        engine = twinkle.Twinkle(int(n_srcs), int(device_num), int(n_stream), float(reltol))
    _TWINKLE_ENGINE_CACHE[key] = engine
    return engine


def trajectory_xy(t: np.ndarray, t0: float, tE: float, u0: float, alpha_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    tau = (t - t0) / tE
    alpha = np.deg2rad(alpha_deg)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    x = tau * ca - u0 * sa
    y = tau * sa + u0 * ca
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


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


def fsbl_magnification_twinkle(
    t: np.ndarray,
    section: dict,
    *,
    parallax: bool,
    event_name: str,
    coords: np.ndarray,
    ephemeris_file: Path,
    twinkle_python_dir: Optional[str] = None,
    twinkle_device: int = 0,
    twinkle_n_stream: int = 1,
    twinkle_reltol: float = 1e-4,
) -> np.ndarray:
    required = ["t_0", "u_0", "t_E", "s", "q", "rho", "alpha_deg"]
    missing = [k for k in required if as_float(section, k) is None]
    if missing:
        raise RuntimeError(f"Missing FSBL parameters {missing}")

    t = np.asarray(t, dtype=np.float64)
    if parallax:
        if as_float(section, "pi_E_N") is None or as_float(section, "pi_E_E") is None:
            raise RuntimeError("FSBL+Parallax section has no pi_E_N/pi_E_E")
        if coords is None or len(coords) != 2:
            raise RuntimeError("FSBL+Parallax plotting needs coordinates")
        t_ref = float(section.get("t_0_par", section["t_0"]))
        cache_key = (sanitize_event_name(event_name), str(Path(ephemeris_file).resolve()), len(t), t_ref)
        if cache_key in _PROJECTOR_CACHE:
            d_e, d_n = _PROJECTOR_CACHE[cache_key]
        else:
            ephem_t, ephem_xyz = load_ephemeris_xyz(Path(ephemeris_file))
            d_e, d_n = make_parallax_projector(
                t,
                ra_deg=float(coords[0]),
                dec_deg=float(coords[1]),
                ephem_t=ephem_t,
                ephem_xyz=ephem_xyz,
                t_ref=t_ref,
            )
            _PROJECTOR_CACHE[cache_key] = (d_e, d_n)
        x, y = trajectory_xy_parallax(
            t,
            float(section["t_0"]),
            float(section["t_E"]),
            float(section["u_0"]),
            float(section["alpha_deg"]),
            float(section["pi_E_N"]),
            float(section["pi_E_E"]),
            d_e,
            d_n,
        )
    else:
        x, y = trajectory_xy(
            t,
            float(section["t_0"]),
            float(section["t_E"]),
            float(section["u_0"]),
            float(section["alpha_deg"]),
        )

    twinkle = import_twinkle_module(twinkle_python_dir)
    engine = make_twinkle_engine(twinkle, len(t), twinkle_device, twinkle_n_stream, twinkle_reltol)
    mag = np.empty(len(t), dtype=np.float64)
    engine.set_params(float(section["s"]), float(section["q"]), float(section["rho"]), x, y)
    engine.run()
    engine.return_mag_to(mag)
    return mag


# ============================================================
# Model evaluation
# ============================================================

def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    return 22.0 - 2.5 * np.log10(np.maximum(flux, 1e-300))


def evaluate_model_flux(
    t: np.ndarray,
    model_name: str,
    section: dict,
    *,
    event_name: str,
    coords: np.ndarray,
    ephemeris_file: Path,
    twinkle_python_dir: Optional[str],
    twinkle_device: int,
    twinkle_n_stream: int,
    twinkle_reltol: float,
) -> np.ndarray:
    """Evaluate fitted flux: F_model(t) = Fs * A(t) + Fb."""
    if "Fs" not in section or "Fb" not in section:
        raise ValueError(f"Model {model_name} has no Fs/Fb in params file.")

    if model_name in {"FSBL", "FSBL+Parallax"}:
        try:
            A = fsbl_magnification_twinkle(
                t,
                section,
                parallax=(model_name == "FSBL+Parallax"),
                event_name=event_name,
                coords=coords,
                ephemeris_file=ephemeris_file,
                twinkle_python_dir=twinkle_python_dir,
                twinkle_device=twinkle_device,
                twinkle_n_stream=twinkle_n_stream,
                twinkle_reltol=twinkle_reltol,
            )
        except Exception as exc:
            if model_name == "FSBL+Parallax":
                raise
            print(f"Warning: Twinkle FSBL evaluation failed ({exc}); falling back to magnification_model/microlux.")
            params = build_magnification_params(model_name, section)
            t_jax = jnp.asarray(t, dtype=jnp.float64)
            A = np.asarray(magnification(t_jax, params), dtype=float)
    else:
        params = build_magnification_params(model_name, section)
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        A = np.asarray(magnification(t_jax, params), dtype=float)

    Fs = float(section["Fs"])
    Fb = float(section["Fb"])
    return Fs * A + Fb


# ============================================================
# Plotting
# ============================================================

def plot_event_fit(
    *,
    event_name: str,
    lc: dict,
    params_by_model: dict,
    models_to_plot: list[str],
    y_mode: str,
    n_grid: int,
    show_residuals: bool,
    save_path: Path | None,
    title: str | None,
    coords: np.ndarray,
    ephemeris_file: Path,
    twinkle_python_dir: Optional[str],
    twinkle_device: int,
    twinkle_n_stream: int,
    twinkle_reltol: float,
):
    t = lc["t"]

    # Use exactly the observed data times for model evaluation.
    # This is faster and avoids drawing an artificial smooth curve,
    # especially important for FSBL/Twinkle.
    t_model = t.copy()

    if show_residuals:
        fig, (ax, ax_res) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax_res = None

    # Plot data.
    if y_mode == "mag":
        ax.errorbar(
            lc["t"],
            lc["mag"],
            yerr=lc["mag_err"],
            fmt=".",
            ms=3,
            alpha=0.45,
            capsize=0,
            label="data",
        )
        ax.set_ylabel("Magnitude")
        ax.invert_yaxis()
    else:
        ax.errorbar(
            lc["t"],
            lc["flux"],
            yerr=lc["flux_err"],
            fmt=".",
            ms=3,
            alpha=0.45,
            capsize=0,
            label="data",
        )
        ax.set_ylabel("Flux, ZP=22")

    # Plot models at exactly the data time points.
    for model_name in models_to_plot:
        if model_name not in params_by_model:
            print(f"Skipping {model_name}: not found in params file.")
            continue

        section = params_by_model[model_name]

        try:
            model_flux_data = evaluate_model_flux(
                t_model,
                model_name,
                section,
                event_name=event_name,
                coords=coords,
                ephemeris_file=ephemeris_file,
                twinkle_python_dir=twinkle_python_dir,
                twinkle_device=twinkle_device,
                twinkle_n_stream=twinkle_n_stream,
                twinkle_reltol=twinkle_reltol,
            )
        except Exception as exc:
            print(f"Skipping {model_name}: could not evaluate model: {exc}")
            continue

        if y_mode == "mag":
            y_model = flux_to_mag(model_flux_data)
        else:
            y_model = model_flux_data

        chi2 = section.get("Chi2", np.nan)
        chi2dof = section.get("chi2/dof", np.nan)

        label = f"{model_name}"
        if np.isfinite(chi2dof):
            label += f"  χ²/dof={chi2dof:.3g}"
        elif np.isfinite(chi2):
            label += f"  χ²={chi2:.3g}"
        if "multinest_source" in section:
            label += "  [MultiNest]"

        # Sort by time before plotting the line.
        order = np.argsort(t_model)
        ax.plot(
            t_model[order],
            y_model[order],
            lw=1.8,
            marker=".",
            ms=2,
            alpha=0.9,
            label=label,
        )

        if show_residuals and ax_res is not None:
            if y_mode == "mag":
                residual = lc["mag"] - y_model
                residual_err = lc["mag_err"]
                ax_res.set_ylabel("Data - model [mag]")
            else:
                residual = lc["flux"] - y_model
                residual_err = lc["flux_err"]
                ax_res.set_ylabel("Data - model [flux]")

            ax_res.axhline(0.0, lw=1)
            ax_res.errorbar(
                lc["t"],
                residual,
                yerr=residual_err,
                fmt=".",
                ms=3,
                alpha=0.45,
                capsize=0,
                label=model_name,
            )

    ax.set_title(title or f"Microlensing fit: {event_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    if ax_res is not None:
        ax_res.set_xlabel("BJD")
        ax_res.grid(alpha=0.25)
    else:
        ax.set_xlabel("BJD")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    plt.show()


# ============================================================
# CLI
# ============================================================

def parse_model_list(text: str, available: dict) -> list[str]:
    if text.lower() == "all":
        return [m for m in DEFAULT_MODEL_ORDER if m in available]

    requested = [m.strip() for m in text.split(",") if m.strip()]

    # Allow some lowercase aliases.
    alias = {
        "pspl": "PSPL",
        "parallax": "PSPL+Parallax",
        "pspl+parallax": "PSPL+Parallax",
        "fspl": "FSPL",
        "fspl+parallax": "FSPL+Parallax",
        "bspl": "BSPL",
        "bspl+parallax": "BSPL+Parallax",
        "fsbl": "FSBL",
        "fsbl+parallax": "FSBL+Parallax",
        "fsblparallax": "FSBL+Parallax",
    }

    out = []
    for m in requested:
        canonical = alias.get(m.lower(), m)
        if canonical not in available:
            print(f"Warning: requested model {canonical!r} not found in results.")
            continue
        out.append(canonical)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot Roman microlensing event data and fitted model curves. Supports optax *_params.txt and MultiNest FSBL best_fit.txt."
    )

    parser.add_argument(
        "--event",
        required=False,
        help="Event name, e.g. RMDC26_000005. Can be given with or without .csv.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Direct path to event CSV. Overrides --event lookup.",
    )
    parser.add_argument(
        "--result-file",
        default=None,
        help="Direct path to EVENT_params.txt. Overrides --event lookup.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/data_F146",
        help="Directory with event CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/optax_results",
        help="Directory with *_params.txt files.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root results directory for auto-loading MultiNest FSBL folders. Default: parent of --results-dir.",
    )
    parser.add_argument(
        "--fsbl-dir",
        default=None,
        help="Optional explicit FSBL MultiNest dir or best_fit.txt file.",
    )
    parser.add_argument(
        "--fsbl-parallax-dir",
        default=None,
        help="Optional explicit FSBL+Parallax MultiNest dir or best_fit.txt file.",
    )
    parser.add_argument(
        "--no-auto-multinest",
        action="store_true",
        help="Do not auto-load results/EVENT_multinest_FSBL*/best_fit.txt.",
    )
    parser.add_argument(
        "--coord-file",
        default="data/coords.csv",
        help="Coordinates CSV with columns name, ra_deg, dec_deg.",
    )
    parser.add_argument(
        "--ephemeris-file",
        default="data/Roman_ephemeris_jax.txt",
        help="Roman ephemeris file for FSBL+Parallax plotting.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Models to plot: all, or comma-separated list, e.g. "
            "PSPL,BSPL,FSBL,FSBL+Parallax"
        ),
    )
    parser.add_argument(
        "--y",
        choices=["mag", "flux"],
        default="mag",
        help="Plot in magnitudes or flux. Default: mag.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=800,
        help="Kept for backward compatibility. Models are evaluated at data times.",
    )
    parser.add_argument(
        "--no-residuals",
        action="store_true",
        help="Disable residual panel.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path, e.g. plots/event_fit.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--twinkle-python-dir",
        default=os.environ.get("TWINKLE_PYTHON_DIR"),
        help="Directory containing compiled twinkle*.so. Default: TWINKLE_PYTHON_DIR env var.",
    )
    parser.add_argument("--twinkle-device", type=int, default=int(os.environ.get("TWINKLE_DEVICE", "0")))
    parser.add_argument("--twinkle-n-stream", type=int, default=int(os.environ.get("TWINKLE_N_STREAM", "1")))
    parser.add_argument("--twinkle-reltol", type=float, default=float(os.environ.get("TWINKLE_RELTOL", "1e-4")))

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    coord_file = Path(args.coord_file)
    ephemeris_file = Path(args.ephemeris_file)
    results_root = Path(args.results_root) if args.results_root else results_dir.parent

    if args.data_file is not None:
        data_file = Path(args.data_file)
        event_name = sanitize_event_name(data_file.name)
    else:
        if args.event is None:
            raise RuntimeError("Give either --event or --data-file.")
        data_file = find_data_file(args.event, data_dir)
        event_name = sanitize_event_name(data_file.name)

    if args.result_file is not None:
        result_file = Path(args.result_file)
    else:
        result_file = find_result_file(event_name, results_dir)

    print(f"Data file:   {data_file}")
    print(f"Result file: {result_file}")

    lc = load_lightcurve_csv(data_file)
    params_by_model = parse_params_file(result_file)

    load_multinest_fsbl_results(
        event_name=event_name,
        params_by_model=params_by_model,
        results_root=results_root,
        fsbl_dir=Path(args.fsbl_dir) if args.fsbl_dir else None,
        fsbl_parallax_dir=Path(args.fsbl_parallax_dir) if args.fsbl_parallax_dir else None,
        auto=not args.no_auto_multinest,
    )

    coords = load_coords(event_name, coord_file)
    attach_hidden_parallax_fields(params_by_model, event_name, coords)

    models_to_plot = parse_model_list(args.models, params_by_model)

    if len(models_to_plot) == 0:
        raise RuntimeError("No valid models selected/found.")

    print("Models to plot:")
    for m in models_to_plot:
        chi2 = params_by_model[m].get("Chi2", np.nan)
        chi2dof = params_by_model[m].get("chi2/dof", np.nan)
        src = ""
        if "multinest_source" in params_by_model[m]:
            src = " [MultiNest]"
        print(f"  {m:16s} chi2={chi2:.6g}, chi2/dof={chi2dof:.6g}{src}")

    save_path = Path(args.save) if args.save else None

    plot_event_fit(
        event_name=event_name,
        lc=lc,
        params_by_model=params_by_model,
        models_to_plot=models_to_plot,
        y_mode=args.y,
        n_grid=args.n_grid,
        show_residuals=not args.no_residuals,
        save_path=save_path,
        title=args.title,
        coords=coords,
        ephemeris_file=ephemeris_file,
        twinkle_python_dir=args.twinkle_python_dir,
        twinkle_device=args.twinkle_device,
        twinkle_n_stream=args.twinkle_n_stream,
        twinkle_reltol=args.twinkle_reltol,
    )


if __name__ == "__main__":
    main()
