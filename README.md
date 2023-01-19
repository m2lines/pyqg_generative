# JAMES publication
This repository was used to obtain all results of the paper Pavel Perezhogin, Laure Zanna, Carlos Fernandez-Granda "Data-driven stochastic parameterizations of subgrid mesoscale eddies in an idealized ocean model" submitted to JAMES. 
## Figures
See `notebooks/paper_figures_1.ipynb` and folder `notebooks/paper_figures_1/`.
# Package description
Repository with generative models (GZ2021, cGAN, cVAE) predicting subgrid forcing from pyqg data in probabilistic way.

# Installation
* Make sure packages are istalled:
```
conda install numpy matplotlib xarray
conda install -c conda-forge pyfftw, gcm_filters
```
* Download `pyqg` and install in editable mode:
```
git clone https://github.com/pyqg/pyqg.git
cd pyqg
pip install --editable .
```
* Install Torch 
* There may be need to install packages https://github.com/Zanna-ResearchTeam/pyqg_experiments and https://github.com/m2lines/pyqg_parameterization_benchmarks
* Clone this repository and install it with:
```
pip install --editable .
```
