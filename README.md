## Miau ##

This is a work in progress repo containing the Miaupline of the Microlensing Innovtions Aleje Ujazdowskie team (Miau) for the Roman Microlensing Data Challenge.

## How to run the Miaupline ##

Download the Data Challenge data from https://huggingface.co/datasets/RGES-PIT/Beginner execute `convert_parquet.py` (see issue #1) and run `source/run.py`

## Running multinest on event 000005

EVENT=RMDC26_000005 && LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false python source/run.py --event $EVENT --out-dir results/${EVENT}_simple --models PSPL,FSPL,BSPL && MN_PSPL_T0_WIDTH_TE=5 MN_PSPL_TE_HI_FACTOR=3 MN_U0_MIN=0 MN_U0_MAX=4 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH python source/multinest_cpu.py --data-file data/data_F146/${EVENT}.csv --params-file results/${EVENT}_simple/${EVENT}_params.txt --out-dir results/${EVENT}_multinest_FSBL_from_FSPL --prefer-single-lens FSPL --n-live 300 --max-points 1000