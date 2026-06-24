import os

import jax

jax.config.update("jax_enable_x64", True)

from data_loader import DataLoader
from models import BSPL, BSPLParallax, FSBL, FSBLParallax, Parallax, PSPL
from pipeline import run_pipeline


def main():
    input_dir = "data/data_F146"
    os.makedirs("results", exist_ok=True)
    out_dir = "results/optax_results"

    files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")
    ]

    data_loader = DataLoader(coord_file="data/coords.csv")

    # Model hierarchy:
    #   PSPL             baseline single-lens fit and seed for all later models
    #   PSPL+Parallax    same lens, perturbed observer trajectory
    #   BSPL             binary-source false-positive competitor
    #   BSPL+Parallax    binary source plus the same parallax trajectory correction
    #   FSBL             finite-source binary lens through microJAX
    #   FSBL+Parallax    FSBL plus parallax, seeded from the best FSBL solution
    models = [
        PSPL(),
        Parallax(),
        BSPL(),
        BSPLParallax(),
        FSBL(),
        FSBLParallax(),
    ]
    run_pipeline(files, out_dir, data_loader, models, max_len=46_208)


if __name__ == "__main__":
    main()
