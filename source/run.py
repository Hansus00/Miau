from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from data_loader import DataLoader
from models import (
    BSPL,
    BSPLParallax,
    FSBL,
    FSBLParallax,
    FSPL,
    FSPLParallax,
    Parallax,
    PSPL,
)
from pipeline import run_pipeline


MODEL_FACTORY = {
    "PSPL": PSPL,
    "PSPL+Parallax": Parallax,
    "Parallax": Parallax,
    "FSPL": FSPL,
    "FSPL+Parallax": FSPLParallax,
    "BSPL": BSPL,
    "BSPL+Parallax": BSPLParallax,
    "FSBL": FSBL,
    "FSBL+Parallax": FSBLParallax,
}

DEFAULT_MODEL_NAMES = [
    "PSPL",
    "FSPL",
    "PSPL+Parallax",
    "FSPL+Parallax",
    "BSPL",
    "BSPL+Parallax",
    "FSBL",
    "FSBL+Parallax",
]


def _canonical_model_name(name: str) -> str:
    aliases = {
        "parallax": "PSPL+Parallax",
        "pspl_parallax": "PSPL+Parallax",
        "pspl+parallax": "PSPL+Parallax",
        "fspl_parallax": "FSPL+Parallax",
        "fspl+parallax": "FSPL+Parallax",
        "bspl_parallax": "BSPL+Parallax",
        "bspl+parallax": "BSPL+Parallax",
        "fsbl_parallax": "FSBL+Parallax",
        "fsbl+parallax": "FSBL+Parallax",
    }
    return aliases.get(name.strip().lower(), name.strip())


def build_models(model_list: str | None):
    if model_list is None or model_list.lower() == "default":
        names = DEFAULT_MODEL_NAMES
    else:
        names = [_canonical_model_name(x) for x in model_list.split(",") if x.strip()]

    models = []
    for name in names:
        if name not in MODEL_FACTORY:
            raise ValueError(
                f"Unknown model {name!r}. Available: {', '.join(MODEL_FACTORY)}"
            )
        models.append(MODEL_FACTORY[name]())
    return models


def find_files(input_dir: str, event: str | None):
    input_path = Path(input_dir)
    if event is None:
        files = sorted(str(p) for p in input_path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No .csv files found in {input_dir}")
        return files

    event_path = Path(event)
    if event_path.exists():
        return [str(event_path)]

    event_name = event_path.name
    if event_name.endswith(".csv"):
        event_name = event_name[:-4]

    direct = input_path / f"{event_name}.csv"
    if direct.exists():
        return [str(direct)]

    matches = sorted(input_path.glob(f"*{event_name}*.csv"))
    if len(matches) == 1:
        return [str(matches[0])]
    if len(matches) > 1:
        print("Matched multiple files:")
        for m in matches[:50]:
            print(f"  {m}")
        raise RuntimeError("Please provide a more specific --event value.")

    raise FileNotFoundError(f"Could not find event {event!r} in {input_dir}")


def default_out_dir(event: str | None):
    if event is None:
        return "results/optax_results"
    clean = Path(event).name.replace(".csv", "")
    return f"results/single_{clean}"


def main():
    parser = argparse.ArgumentParser(description="Run the Roman microlensing fitting pipeline.")
    parser.add_argument("--input-dir", default="data/data_F146", help="Input directory with event CSV files.")
    parser.add_argument("--coord-file", default="data/coords.csv", help="Coordinate file for parallax models.")
    parser.add_argument("--event", default=None, help="Fit only one event. Give event name, partial name, CSV name, or full CSV path.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: batch results or results/single_<event> for --event.")
    parser.add_argument("--models", default="default", help="Comma-separated models or 'default'. Example: PSPL,FSPL,FSPL+Parallax")
    parser.add_argument("--max-len", type=int, default=46_208, help="Maximum padded light-curve length.")
    args = parser.parse_args()

    files = find_files(args.input_dir, args.event)
    out_dir = args.out_dir or default_out_dir(args.event)
    os.makedirs(out_dir, exist_ok=True)

    data_loader = DataLoader(coord_file=args.coord_file)
    models = build_models(args.models)

    print(f"Input files: {len(files)}")
    if len(files) == 1:
        print(f"Event file: {files[0]}")
    print(f"Output dir: {out_dir}")
    print("Models:", ", ".join(m.name for m in models))

    run_pipeline(files, out_dir, data_loader, models, max_len=args.max_len)


if __name__ == "__main__":
    main()
