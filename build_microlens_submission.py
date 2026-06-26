"""
Build a microlens-submit CSV and optionally create/export a submission project
from this pipeline's *_params.txt result files.

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


def chi2_to_loglike(section: Dict[str, Any]) -> Optional[float]:
    chi2 = as_float(section, "Chi2")
    if chi2 is None:
        chi2 = as_float(section, "chi2")
    if chi2 is None:
        return None
    return -0.5 * chi2


def n_points(section: Dict[str, Any]) -> Optional[int]:
    for key in ["n_data_points", "eval_n_points", "N", "n_valid"]:
        val = as_float(section, key)
        if val is not None:
            return int(round(val))
    return None


def safe_alias(model_name: str, source_tag: str) -> str:
    alias = model_name.replace("+", "_plus_").replace(" ", "_")
    alias = alias.replace("/", "_").replace("-", "_")
    source_tag = re.sub(r"[^A-Za-z0-9_]+", "_", source_tag)
    return f"{alias}__{source_tag}"[:120]


def add_common_optional(row: Dict[str, Any], section: Dict[str, Any], notes: str = "") -> None:
    ll = chi2_to_loglike(section)
    if ll is not None:
        row["log_likelihood"] = ll
    n = n_points(section)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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
        add_common_optional(row, section, notes)
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


def build_rows(results_roots: List[Path], include_models: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for params_file in iter_params_files(results_roots):
        event_id = infer_event_id_from_params_file(params_file)
        sections = parse_params_txt(params_file)
        source_tag = str(params_file.parent).replace(os.sep, "_")
        for model_name, section in sections.items():
            if include_models and model_name not in include_models:
                continue
            row = convert_section_to_row(event_id, model_name, section, source_tag=source_tag)
            if row is not None:
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
    ap.add_argument("--best-per-event", action="store_true", help="Only submit one best log-likelihood row per event.")
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
    rows = build_rows(roots, include_models=include_models)

    if args.best_per_event:
        rows = choose_best_per_event(rows)

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
