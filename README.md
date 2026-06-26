## Miau ##

This is a work in progress repo containing the Miaupline of the Microlensing Innovtions Aleje Ujazdowskie team (Miau) for the Roman Microlensing Data Challenge.

## How to run the Miaupline ##

Download the Data Challenge data from https://huggingface.co/datasets/RGES-PIT/Beginner execute `convert_parquet.py` (see issue #1) and run `source/run.py`

## Running multinest on event 000005

EVENT=RMDC26_000005 && LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false python source/run.py --event $EVENT --out-dir results/${EVENT}_simple --models PSPL,FSPL,BSPL && MN_PSPL_T0_WIDTH_TE=5 MN_PSPL_TE_HI_FACTOR=3 MN_U0_MIN=0 MN_U0_MAX=4 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH python source/multinest_cpu.py --data-file data/data_F146/${EVENT}.csv --params-file results/${EVENT}_simple/${EVENT}_params.txt --out-dir results/${EVENT}_multinest_FSBL_from_FSPL --prefer-single-lens FSPL --n-live 300 --max-points 1000

## Running multinest on events with chi2/dof >=2
mkdir -p logs && for f in data/data_F146/*.csv; do EVENT=$(basename "$f" .csv); echo "===== EVENT: $EVENT ====="; if [ -f "results/${EVENT}_multinest_FSBL_from_FSPL/best_fit.txt" ]; then echo "Skipping $EVENT because MultiNest best_fit.txt already exists."; continue; fi; LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false python source/run.py --event "$EVENT" --out-dir "results/${EVENT}_simple" --models PSPL,FSPL,BSPL 2>&1 | tee "logs/${EVENT}_simple.log"; BEST_CHI2DOF=$(python -c "import sys,re,math; txt=open(sys.argv[1]).read(); vals=[float(x) for x in re.findall(r'chi2/dof:\s*([-+0-9.eE]+)', txt)]; vals=[v for v in vals if math.isfinite(v)]; print(min(vals) if vals else 999999)" "results/${EVENT}_simple/${EVENT}_params.txt"); echo "Best simple chi2/dof for $EVENT = $BEST_CHI2DOF"; if python -c "import sys,math; x=float(sys.argv[1]); sys.exit(0 if math.isfinite(x) and x > 2.0 else 1)" "$BEST_CHI2DOF"; then echo "Running MultiNest for $EVENT because best simple chi2/dof > 2"; MN_PSPL_T0_WIDTH_TE=5 MN_PSPL_TE_HI_FACTOR=3 MN_U0_MIN=0 MN_U0_MAX=4 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH python source/multinest_cpu.py --data-file "$f" --params-file "results/${EVENT}_simple/${EVENT}_params.txt" --out-dir "results/${EVENT}_multinest_FSBL_from_FSPL" --prefer-single-lens FSPL --n-live 100 --max-points 500 2>&1 | tee "logs/${EVENT}_multinest.log"; else echo "Skipping MultiNest for $EVENT because best simple chi2/dof <= 2"; fi; done