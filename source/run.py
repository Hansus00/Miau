import os

import jax

jax.config.update("jax_enable_x64", True)

from data_loader import DataLoader
from models import BSPL, FSBL, FSPL, PSPL, Parallax
from pipeline import run_pipeline


def main():
    input_dir = "data/data_F146"
    os.makedirs("results", exist_ok=True)
    out_dir = "results/optax_results"

    files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")
    ]

    data_loader = DataLoader(coord_file="data/coords.csv")
    models = [PSPL(), FSPL()]
    run_pipeline(files, out_dir, data_loader, models, max_len=46_208)


if __name__ == "__main__":
    main()