# JAMES publication
This repository was used to obtain all results of the paper Pavel Perezhogin, Laure Zanna, Carlos Fernandez-Granda "Data-driven stochastic parameterizations of subgrid mesoscale eddies in an idealized ocean model" submitted to JAMES. 
## Paper Figures
See [notebooks/JAMES_figures.ipynb](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/JAMES_figures.ipynb).

## Generation of data (Expensive and depends on HPC)
```
cd scripts
```
*Check: slurm consistent with your hpc:*
```
python -c "from slurm_helpers import *; create_slurm('','test.py')"
cat launcher.sh
```
Run each script and pay attention to `BASIC_FOLDER`, `SCRIPT_PATH` and so on:
```
python run_reference.py
python run_forcing_datasets.py
python train_parameterizations.py
python run_parameterized.py
python compute_online_metrics.py
```
# Installation of pyqg_generative
## Requirements
```
conda install numpy matplotlib xarray
conda install -c conda-forge pyfftw, gcm_filters, pyqg
```
* Install Torch 
* Optionally, install [pyqg_parameterization_benchmarks](https://github.com/m2lines/pyqg_parameterization_benchmarks)

## Install
```
git clone https://github.com/m2lines/pyqg_generative.git
pip install --editable .
```
