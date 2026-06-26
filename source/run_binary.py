"""Run Twinkle MultiNest refinement using existing simple-fit outputs in results/optax_results."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "data_F146"
RESULTS_DIR = REPO_ROOT / "results"
SIMPLE_RESULTS_DIR = RESULTS_DIR / "optax_results"
LOGS_DIR = REPO_ROOT / "logs"
SIMPLE_SUFFIX = "_simple"
MULTINEST_SUFFIX = "_multinest_FSBL_from_FSPL"
CHI2_THRESHOLD = 1.0


def _event_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.csv"))


def _multinest_out_dir(event: str) -> Path:
    return RESULTS_DIR / f"{event}{MULTINEST_SUFFIX}"


def _simple_params_file(event: str) -> Path:
    return SIMPLE_RESULTS_DIR / f"{event}_params.txt"


def _extract_best_chi2_dof(params_file: Path) -> float:
    try:
        text = params_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return float("inf")
    values = [float(x) for x in re.findall(r"chi2/dof:\s*([-+0-9.eE]+)", text)]
    values = [value for value in values if math.isfinite(value)]
    return min(values) if values else 999999.0


def _extract_best_model(params_file: Path) -> str:
    try:
        text = params_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

    current_model = ""
    best_model = ""
    best_chi2_dof = float("inf")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_model = line[1:-1].strip()
            continue
        if current_model and line.startswith("chi2/dof:"):
            match = re.search(r"chi2/dof:\s*([-+0-9.eE]+)", line)
            if match is None:
                continue
            value = float(match.group(1))
            if math.isfinite(value) and value < best_chi2_dof:
                best_chi2_dof = value
                best_model = current_model
    return best_model


def _tee_run(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{Path.home() / 'MultiNest' / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_env = _base_env()

    for event_file in _event_files():
        event = event_file.stem
        print(f"===== EVENT: {event} =====")

        multinest_out = _multinest_out_dir(event)
        if (multinest_out / "best_fit.txt").exists():
            print(f"Skipping {event} because MultiNest best_fit.txt already exists.")
            continue

        params_file = _simple_params_file(event)
        if not params_file.exists():
            print(f"Skipping {event} because {params_file} does not exist.")
            continue

        best_chi2_dof = _extract_best_chi2_dof(params_file)
        if best_chi2_dof == 999999.0:
            print(f"Skipping {event} because no valid chi2/dof found in {params_file}.")
            continue
        best_model = _extract_best_model(params_file)
        print(f"Best simple model for {event} = {best_model or 'unknown'}")
        print(f"Best simple chi2/dof for {event} = {best_chi2_dof}")

        should_run_multinest = (
            (math.isfinite(best_chi2_dof) and best_chi2_dof > CHI2_THRESHOLD)
            or best_model == "BSPL"
        )

        if should_run_multinest:
            if best_model == "BSPL" and not (math.isfinite(best_chi2_dof) and best_chi2_dof > CHI2_THRESHOLD):
                print(f"Running Twinkle-MultiNest for {event} because best model is BSPL")
            else:
                print(f"Running Twinkle-MultiNest for {event} because best simple chi2/dof > {CHI2_THRESHOLD:g}")
            multinest_log = LOGS_DIR / f"{event}_multinest_twinkle.log"
            twinkle_env = base_env.copy()
            twinkle_env.update(
                {
                    "MN_PSPL_T0_WIDTH_TE": "5",
                    "MN_PSPL_TE_HI_FACTOR": "3",
                    "MN_U0_MIN": "0",
                    "MN_U0_MAX": "4",
                    "TWINKLE_PYTHON_DIR": os.environ.get("TWINKLE_PYTHON_DIR", str(Path.home() / "Twinkle" / "python")),
                }
            )
            multinest_cmd = [
                sys.executable,
                "source/multinest_twinkle.py",
                "--data-file",
                str(event_file),
                "--params-file",
                str(params_file),
                "--out-dir",
                str(multinest_out),
                "--prefer-single-lens",
                "FSPL",
                "--n-live",
                "100",
                "--max-points",
                "500",
                "--max-iter",
                "200000",
            ]
            multinest_rc = _tee_run(multinest_cmd, multinest_log, twinkle_env)
            if multinest_rc != 0:
                print(f"Twinkle-MultiNest failed for {event} with exit code {multinest_rc}")
        else:
            print(f"Skipping MultiNest for {event} because best simple chi2/dof <= {CHI2_THRESHOLD:g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())