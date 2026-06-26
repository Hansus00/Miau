from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from data_loader import DataLoader
from initial_conditions import InitialConditions
from magnification_model import ensure_ephemeris_loaded, magnification
from models import BSPLParallax
from optimization import build_optimize_loop, get_eval_metrics

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan",
    flags=re.IGNORECASE,
)


def first_float(value: str) -> Optional[float]:
    m = FLOAT_RE.search(str(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_params_txt(path: Path) -> Dict[str, Dict[str, Any]]:
    sections: Dict[str, Dict[str, Any]] = {}
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
        k, v = line.split(":", 1)
        f = first_float(v)
        sections[current][k.strip()] = f if f is not None else v.strip()
    return sections


def remove_section(text: str, section_name: str) -> str:
    lines = text.splitlines()
    out = []
    i = 0
    target = f"[{section_name}]"
    while i < len(lines):
        if lines[i].strip() == target:
            i += 1
            while i < len(lines) and not (lines[i].strip().startswith("[") and lines[i].strip().endswith("]")):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip() + "\n"


def build_init_from_bspl(sections: Dict[str, Dict[str, Any]], t0_shift: float) -> jnp.ndarray:
    if "BSPL" not in sections:
        raise RuntimeError("No [BSPL] section found; cannot seed BSPL+Parallax.")
    bspl = sections["BSPL"]
    pspl_plx = sections.get("PSPL+Parallax", {})
    pi_n = float(pspl_plx.get("pi_E_N", 0.0) or 0.0)
    pi_e = float(pspl_plx.get("pi_E_E", 0.0) or 0.0)
    return jnp.array(
        [
            float(bspl["t_0_1"]) - t0_shift,
            float(bspl["t_0_2"]) - t0_shift,
            jnp.log(max(float(bspl["t_E"]), 1.0e-6)),
            float(bspl["u_0_1"]),
            float(bspl["u_0_2"]),
            jnp.log(max(float(bspl["q_f"]), 1.0e-12)),
            pi_n,
            pi_e,
        ],
        dtype=jnp.float64,
    )


def fit_one(
    event_id: str,
    data_file: Path,
    params_file: Path,
    out_file: Path,
    coord_file: str,
    max_len: int,
    learning_rate: float,
    n_steps: int,
    patience: int,
    min_improvement: float,
    overwrite: bool,
) -> bool:
    sections = parse_params_txt(params_file)
    if "BSPL+Parallax" in sections and not overwrite:
        print(f"Skipping {event_id}: [BSPL+Parallax] already exists in {out_file}")
        return False
    if "PSPL" not in sections or "BSPL" not in sections:
        print(f"Skipping {event_id}: missing PSPL or BSPL in {params_file}")
        return False

    loader = DataLoader(coord_file=coord_file)
    raw = loader.load_event(str(data_file))
    init_conds = InitialConditions(raw)
    data = init_conds.get_processed_data(max_len=max_len)
    data["t_0_par"] = jnp.asarray(float(sections["PSPL"]["t_0"]), dtype=jnp.float64)

    init_params = build_init_from_bspl(sections, float(data["t_0_shift"]))
    model = BSPLParallax()
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_loop = build_optimize_loop(
        model.neg_lnprob_fn,
        optimizer,
        n_steps=n_steps,
        min_improvement=min_improvement,
        patience=patience,
    )
    res = opt_loop(init_params, data)
    opt_params = res["params"]
    p_dict = model.to_dict(opt_params, data)
    A = magnification(data["t"], p_dict)
    Fs, Fb, chi2 = get_eval_metrics(A, data["mag"], data["mag_err"])
    dof = int(data["n_valid"]) - len(model.param_names)

    section_lines = ["", "[BSPL+Parallax]"]
    for key in model.param_names:
        section_lines.append(f"{key}: {p_dict[key]}")
    section_lines.append(f"Chi2: {chi2}")
    section_lines.append(f"chi2/dof: {chi2 / dof}")
    section_lines.append(f"Fs: {Fs}")
    section_lines.append(f"Fb: {Fb}")
    section_text = "\n".join(section_lines) + "\n"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        base_text = out_file.read_text(encoding="utf-8")
    else:
        base_text = params_file.read_text(encoding="utf-8")
    if overwrite:
        base_text = remove_section(base_text, "BSPL+Parallax")
    with out_file.open("w", encoding="utf-8") as f:
        f.write(base_text.rstrip() + "\n")
        f.write(section_text)

    print(
        f"Wrote BSPL+Parallax for {event_id}: chi2={float(chi2):.6g}, "
        f"chi2/dof={float(chi2 / dof):.6g} -> {out_file}"
    )
    return True


def resolve_events(args) -> list[str]:
    if args.event:
        return [args.event]
    ids = []
    for p in sorted(Path(args.params_dir).glob("*_params.txt")):
        ids.append(p.name[: -len("_params.txt")])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Append BSPL+Parallax fits to existing *_params.txt files.")
    ap.add_argument("--event", default=None, help="Single event id, e.g. RMDC26_000096. If omitted, process all params files.")
    ap.add_argument("--input-dir", default="data/data_F146")
    ap.add_argument("--params-dir", default="results/optax_results")
    ap.add_argument("--out-dir", default="results/optax_results", help="Where to write updated *_params.txt files.")
    ap.add_argument("--coord-file", default="data/coords.csv")
    ap.add_argument("--max-len", type=int, default=46_208)
    ap.add_argument("--learning-rate", type=float, default=2.0e-3)
    ap.add_argument("--n-steps", type=int, default=10_000)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--min-improvement", type=float, default=1.0e-5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ensure_ephemeris_loaded()

    events = resolve_events(args)
    n_done = 0
    for event_id in events:
        data_file = Path(args.input_dir) / f"{event_id}.csv"
        params_file = Path(args.params_dir) / f"{event_id}_params.txt"
        out_file = Path(args.out_dir) / f"{event_id}_params.txt"
        if not data_file.exists():
            print(f"Skipping {event_id}: missing data file {data_file}")
            continue
        if not params_file.exists():
            print(f"Skipping {event_id}: missing params file {params_file}")
            continue
        try:
            if fit_one(
                event_id,
                data_file,
                params_file,
                out_file,
                args.coord_file,
                args.max_len,
                args.learning_rate,
                args.n_steps,
                args.patience,
                args.min_improvement,
                args.overwrite,
            ):
                n_done += 1
        except Exception as exc:
            print(f"ERROR {event_id}: {exc}")
    print(f"Finished. Added/updated BSPL+Parallax for {n_done} event(s).")


if __name__ == "__main__":
    main()
