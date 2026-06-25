## Miau ##

This is a work in progress repo containing the Miaupline of the Microlensing Innovtions Aleje Ujazdowskie team (Miau) for the Roman Microlensing Data Challenge.

## How to run the Miaupline ##

Download the Data Challenge data from https://huggingface.co/datasets/RGES-PIT/Beginner execute `convert_parquet.py` (see issue #1) and run `source/run.py`


## Useful run modes

Fit all events with the default model hierarchy:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python source/run.py
```

Fit one event and write to a separate directory:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python source/run.py --event EVENT_NAME --out-dir results/EVENT_NAME_test
```

Fit one event with only selected models:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python source/run.py --event EVENT_NAME --out-dir results/EVENT_NAME_fspl --models PSPL,FSPL,PSPL+Parallax,FSPL+Parallax
```

Fast finite-source settings used by FSPL/FSBL families:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false FSBL_START_CHUNK=4 FSBL_TOPK=4 FSBL_COARSE_MAX_POINTS=256 FSBL_OPT_MAX_POINTS=768 FSBL_FINAL_FULL_EVAL=0 python source/run.py
```

The FSPL quadrature can be refined for final checks:

```bash
FSPL_N_R=6 FSPL_N_THETA=24 python source/run.py --event EVENT_NAME --models PSPL,FSPL
```
