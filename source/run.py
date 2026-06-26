import argparse
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from data_loader import DataLoader
from models import BSPL, BSPLParallax, FSBL, FSPL, FSPLParallax, PSPL, Parallax
from pipeline import run_pipeline


MODEL_REGISTRY = {
    "PSPL": PSPL,
    "FSPL": FSPL,
    "PSPL+Parallax": Parallax,
    "Parallax": Parallax,
    "FSPL+Parallax": FSPLParallax,
    "BSPL": BSPL,
    "BSPL+Parallax": BSPLParallax,
    "FSBL": FSBL,
}


def resolve_files(input_dir, event):
    input_dir = Path(input_dir)
    if event:
        p = Path(event)
        if p.exists():
            return [str(p)]
        direct = input_dir / f"{event}.csv"
        if direct.exists():
            return [str(direct)]
        matches = sorted(input_dir.glob(f"*{event}*.csv"))
        if not matches:
            raise FileNotFoundError(f"No event matching {event!r} in {input_dir}")
        return [str(m) for m in matches]
    return [str(input_dir / f) for f in os.listdir(input_dir) if f.endswith(".csv")]


def build_models(text):
    names = [x.strip() for x in text.split(",") if x.strip()]
    out = []
    for name in names:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY)}")
        out.append(MODEL_REGISTRY[name]())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/data_F146")
    parser.add_argument("--event", default=None, help="Event name or CSV path. If omitted, fit all events.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--models",
        default="PSPL,FSPL,PSPL+Parallax,FSPL+Parallax,BSPL,BSPL+Parallax",
        help="Comma-separated model list. Example: PSPL,FSPL,BSPL,BSPL+Parallax,FSBL",
    )
    parser.add_argument("--max-len", type=int, default=46_208)
    parser.add_argument("--coord-file", default="data/coords.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.out_dir is None:
        if args.event:
            safe = Path(args.event).stem
            out_dir = f"results/single_{safe}"
        else:
            out_dir = "results/optax_results"
    else:
        out_dir = args.out_dir

    files = resolve_files(args.input_dir, args.event)
    data_loader = DataLoader(coord_file=args.coord_file)
    models = build_models(args.models)
    run_pipeline(files, out_dir, data_loader, models, max_len=args.max_len)


if __name__ == "__main__":
    main()
