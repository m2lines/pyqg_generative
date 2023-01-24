# JAMES publication
This repository was used to obtain all results of the paper Pavel Perezhogin, Laure Zanna, Carlos Fernandez-Granda "Data-driven stochastic parameterizations of subgrid mesoscale eddies in an idealized ocean model" submitted to JAMES. 
## Paper Figures
See [notebooks/JAMES_figures.ipynb](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/JAMES_figures.ipynb).

## Try it in Google Colab
* [Google-Colab/dataset.ipynb](https://colab.research.google.com/github/m2lines/pyqg_generative/blob/master/Google-Colab/dataset.ipynb) - Description of the dataset containing training data and hires/lores simulations

## Generation of JAMES data (Hard and depends on HPC)
`cd scripts` and *Check that **slurm** is consistent with your HPC:*
```
python -c "from slurm_helpers import *; create_slurm('','test.py')"
cat launcher.sh
```
Run each script and pay attention to `BASIC_FOLDER`, `SCRIPT_PATH` and so on:
* `python run_reference.py`
* Coarsegrain highres simulations with [def coarsegrain_reference_dataset](https://github.com/m2lines/pyqg_generative/blob/master/pyqg_generative/tools/comparison_tools.py#L53)
* `python run_forcing_datasets.py`
* `python train_parameterizations.py`
* `python run_parameterized.py`
* `python compute_online_metrics.py`
# Installation of pyqg_generative
## Requirements
```
pip install numpy matplotlib xarray aiohttp requests zarr pyfftw gcm_filters pyqg
```
* Install [Pytorch](https://pytorch.org/) 
* Optionally, install [pyqg_parameterization_benchmarks](https://github.com/m2lines/pyqg_parameterization_benchmarks)
`pip install git+https://github.com/m2lines/pyqg_parameterization_benchmarks.git`

## Install in editable mode
```
git clone https://github.com/m2lines/pyqg_generative.git
pip install --editable .
```
