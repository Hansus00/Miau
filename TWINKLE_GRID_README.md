# Twinkle GPU FSBL grid search

This version adds an optional **Twinkle coarse grid search** before the existing
microJAX/optax FSBL refinement.

The point of this stage is to solve the failure mode where the differentiable
FSBL optimizer starts from a poor binary-lens geometry and converges to a broad,
large-rho local minimum instead of finding the correct caustic solution.

## Install Twinkle

Twinkle is not installed from PyPI. Compile its Python API from the upstream repo:

```bash
git clone https://github.com/AsterLight0626/Twinkle.git
cd Twinkle/python
python setup.py build_ext --inplace
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Then check:

```bash
python -c "import twinkle; print(twinkle)"
```

## Run one binary event with Twinkle grid + FSBL refinement

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TWINKLE_GRID_SEARCH=1 TWINKLE_TOPN=64 TWINKLE_MAX_POINTS=512 TWINKLE_ALPHA_N=16 TWINKLE_MAX_EVALS=0 FSBL_TOPK=16 FSBL_START_CHUNK=2 FSBL_COARSE_MAX_POINTS=512 FSBL_OPT_MAX_POINTS=1200 FSBL_N_STEPS=6000 FSBL_PATIENCE=80 FSBL_FINAL_FULL_EVAL=1 python source/run.py --event RMDC26_000005 --out-dir results/RMDC26_000005_twinkle --models PSPL,FSPL,BSPL,FSBL
```

The grid top candidates are written to:

```text
results/RMDC26_000005_twinkle/twinkle_grid/RMDC26_000005_twinkle_top.csv
```

## Main environment variables

- `TWINKLE_GRID_SEARCH=1` enables Twinkle grid search for `FSBL`.
- `TWINKLE_TOPN=64` keeps this many best Twinkle starts for microJAX refinement.
- `TWINKLE_MAX_POINTS=512` points from the light curve used in Twinkle screening.
- `TWINKLE_MAX_EVALS=0` means no cap. Set e.g. `50000` for a faster test.
- `TWINKLE_PROGRESS_EVERY=20000` controls progress printing.
- `TWINKLE_ALPHA_N=12` or `TWINKLE_ALPHA_GRID=0,15,30,...` controls alpha.
- `TWINKLE_T0_FRAC_GRID=-0.4,0,0.4` offsets t0 by fractions of the PSPL tE.
- `TWINKLE_TE_FACTOR_GRID=0.7,1.0,1.6` explores timescale factors.
- `TWINKLE_U0_GRID`, `TWINKLE_S_GRID`, `TWINKLE_Q_GRID`, `TWINKLE_RHO_GRID` control the physical grid.

## Important

`FSBL+Parallax` is not searched directly with Twinkle. It is seeded from the best
non-parallax `FSBL` result and then refined with microJAX including parallax.
