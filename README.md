# JAMES publication
This repository was used to obtain all results of the paper Pavel Perezhogin, Laure Zanna, Carlos Fernandez-Granda "Data-driven stochastic parameterizations of subgrid mesoscale eddies in an idealized ocean model" submitted to JAMES. 
## Paper Figures
See [notebooks/JAMES_figures.ipynb](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/JAMES_figures.ipynb).

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
