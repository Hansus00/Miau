"""Asynchronous Twinkle GPU grid search + CPU MultiNest refinement.

The parent process uses Twinkle/GPU to produce seed files.  As soon as a seed is
available, a separate CPU-only Python process can start MultiNest/microlux for
that event while the parent continues the next Twinkle grid search.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from twinkle_grid_search import TwinkleGridConfig, run_twinkle_grid_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_flux_event(path: str | Path) -> dict:
    df = pd.read_csv(path, header=None, names=["bjd", "mag", "mag_err"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("bjd")
    t = df["bjd"].to_numpy(float)
    flux = 10.0 ** (-0.4 * (df["mag"].to_numpy(float) - 22.0))
    flux_err = flux * (0.4 * np.log(10.0)) * df["mag_err"].to_numpy(float)
    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    return {"t": t[good], "mag": flux[good], "mag_err": flux_err[good]}


def find_files(input_dir: Path, event: str | None) -> List[Path]:
    if event:
        p = Path(event)
        if p.exists():
            return [p]
        direct = input_dir / f"{event}.csv"
        if direct.exists():
            return [direct]
        matches = sorted(input_dir.glob(f"*{event}*.csv"))
        if not matches:
            raise FileNotFoundError(f"No event matching {event!r} in {input_dir}")
        return matches
    return sorted(input_dir.glob("*.csv"))


def reap_finished(processes):
    still = []
    for event_name, proc in processes:
        ret = proc.poll()
        if ret is None:
            still.append((event_name, proc))
        else:
            print(f"[CPU MultiNest] {event_name} finished with code {ret}", flush=True)
    return still


def wait_for_slot(processes, max_workers):
    while len(processes) >= max_workers:
        processes = reap_finished(processes)
        if len(processes) >= max_workers:
            time.sleep(2.0)
    return processes


def launch_multinest(data_file: Path, seed_file: Path, out_dir: Path, n_live: int, max_points: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("multinest_cpu.py")),
        "--data-file",
        str(data_file),
        "--seed-file",
        str(seed_file),
        "--out-dir",
        str(out_dir),
        "--n-live",
        str(n_live),
    ]
    if max_points:
        cmd.extend(["--max-points", str(max_points)])
    print("[CPU MultiNest] launching:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/data_F146")
    parser.add_argument("--event", default=None, help="Event name or path. Omit to run all CSV files.")
    parser.add_argument("--out-dir", default="results/twinkle_multinest_async")
    parser.add_argument("--launch-multinest", action="store_true", help="Launch CPU MultiNest after each Twinkle grid.")
    parser.add_argument("--cpu-workers", type=int, default=int(os.environ.get("MULTINEST_CPU_WORKERS", "1")))
    parser.add_argument("--n-live", type=int, default=int(os.environ.get("MULTINEST_N_LIVE", "300")))
    parser.add_argument("--multinest-max-points", type=int, default=int(os.environ.get("MULTINEST_MAX_POINTS", "0")))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_files(input_dir, args.event)
    if not files:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    cfg = TwinkleGridConfig.from_env()
    running = []

    print(f"Found {len(files)} event(s). Twinkle grid will run in parent/GPU process.", flush=True)
    if args.launch_multinest:
        print(f"CPU MultiNest workers: {args.cpu_workers}", flush=True)

    for i, file in enumerate(files, start=1):
        event_name = file.stem
        print(f"\n=== [{i}/{len(files)}] Twinkle grid: {event_name} ===", flush=True)
        data = load_flux_event(file)
        seed_file = run_twinkle_grid_search(data, event_name, out_dir, cfg=cfg)

        if args.launch_multinest:
            running = wait_for_slot(running, args.cpu_workers)
            mn_out = out_dir / "multinest" / event_name
            proc = launch_multinest(file, seed_file, mn_out, args.n_live, args.multinest_max_points)
            running.append((event_name, proc))

    if running:
        print("\nWaiting for remaining CPU MultiNest jobs...", flush=True)
    while running:
        running = reap_finished(running)
        if running:
            time.sleep(2.0)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
