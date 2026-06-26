#!/usr/bin/env bash
# Run multinest_twinkle_parallax (FSBL+Parallax) only for events where the
# binary-lens (FSBL) model has the best AIC among all fitted models.
set -uo pipefail

mkdir -p logs

for d in results/*_multinest_FSBL_from_FSPL; do
  [ -d "$d" ] || continue
  EVENT=$(basename "$d" | sed "s/_multinest_FSBL_from_FSPL$//")
  echo "===== FSBL+Parallax for $EVENT ====="

  if [ ! -f "$d/best_fit.txt" ]; then
    echo "Skipping $EVENT: no FSBL best_fit.txt"
    continue
  fi
  if [ ! -f "data/data_F146/${EVENT}.csv" ]; then
    echo "Skipping $EVENT: no data file"
    continue
  fi
  if [ -f "results/${EVENT}_multinest_FSBL_Parallax_from_FSBL/best_fit.txt" ]; then
    echo "Skipping $EVENT: FSBL+Parallax already exists"
    continue
  fi

  PARAMS_FILE="results/optax_results/${EVENT}_params.txt"
  python source/check_fsbl_aic.py \
    --params-file "$PARAMS_FILE" \
    --fsbl-best-fit "$d/best_fit.txt"
  AIC_RC=$?
  if [ "$AIC_RC" -ne 0 ]; then
    echo "Skipping $EVENT: FSBL does not have the best AIC"
    continue
  fi

  PLX_T0_WIDTH_TE=0.5 PLX_TE_LO_FACTOR=0.5 PLX_TE_HI_FACTOR=2.0 PLX_U0_WIDTH=1.0 \
  PLX_BINARY_FACTOR=3.0 PLX_RHO_FACTOR=10.0 MN_PI_E_MAX=2.0 MN_PARALLAX_PRIOR_SIGMA=0.15 \
  python source/multinest_twinkle_parallax.py \
    --data-file "data/data_F146/${EVENT}.csv" \
    --fsbl-dir "$d" \
    --out-dir "results/${EVENT}_multinest_FSBL_Parallax_from_FSBL" \
    --coord-file data/coords.csv \
    --ephemeris-file data/Roman_ephemeris_jax.txt \
    --n-live 100 \
    --max-points 500 \
    2>&1 | tee "logs/${EVENT}_multinest_fsbl_parallax.log"
done
