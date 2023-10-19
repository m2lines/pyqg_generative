# JAMES publication
This repository was used to obtain all results of the paper Pavel Perezhogin, Laure Zanna, Carlos Fernandez-Granda "Generative data-driven approaches for stochastic subgrid parameterizations in an idealized ocean model" published in [JAMES](https://doi.org/10.1029/2023MS003681).

The main idea of the paper is to build stochastic subgrid parameterizations of mesoscale eddies using generative approach of Machine Learning (ML). Subgrid parameterization accounts for the missing physics induced by the eddies which are not resolved on the grid. Efficient parameterization should allow to simulate turbulent flows on a coarse computational grid. Turbulent flow represented on a coarse grid misses the information about the state of the subgrid eddies. It results in an uncertainty in the missing forcing induced by these eddies. Here we aim to produce samples from the distribution of all possible subgrid forcings consistent with resolved flow:
```math
S \sim \rho(S|\overline{q})
```

An example of many possible realizations of the subgrid forcing at fixed resolved flow is shown below:
![](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/eddy.gif)

An animation is produced using GAN model [notebooks/Animation.ipynb](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/Animation.ipynb).

Online simulations with generative models (GAN, VAE) reveal better numerical stability properties compared to the baseline GZ ([Guillaumin Zanna 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002534)):

![](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/solution-animations/velocity_upper.gif)

An animation is produced using [notebooks/Animate-solution.ipynb](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/Animate-solution.ipynb).
## Paper Figures
See [notebooks/JAMES_figures.ipynb](https://github.com/m2lines/pyqg_generative/blob/master/notebooks/JAMES_figures.ipynb).

## Try it in Google Colab
In a case dataset in cloud is not working, download it from [Zenodo](https://doi.org/10.5281/zenodo.7622683)! 
* [Google-Colab/dataset.ipynb](https://colab.research.google.com/github/m2lines/pyqg_generative/blob/master/Google-Colab/dataset.ipynb) - Description of the dataset containing training data and hires/lores simulations
* [Google-Colab/training.ipynb](https://colab.research.google.com/github/m2lines/pyqg_generative/blob/master/Google-Colab/training.ipynb) - An example of training of the generative subgrid models
* [Google-Colab/offline-analysis.ipynb](https://colab.research.google.com/github/m2lines/pyqg_generative/blob/master/Google-Colab/offline-analysis.ipynb) - Prediction and plotting subgrid forcing. Comparing spectral properties of generated fields. Computing offline metrics.
* [Google-Colab/online-simulations.ipynb](https://colab.research.google.com/github/m2lines/pyqg_generative/blob/master/Google-Colab/online-simulations.ipynb) - Run online simulations with pretrained subgrid models on GPUs. Compare Kinetic Energy (KE), spectrum of KE, snapshots. Compute online metrics.

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
pip install numpy matplotlib xarray aiohttp requests zarr pyfftw gcm_filters pyqg cmocean gplearn
```
* Install [Pytorch](https://pytorch.org/) 
* Optionally, install [pyqg_parameterization_benchmarks](https://github.com/m2lines/pyqg_parameterization_benchmarks)
`pip install git+https://github.com/m2lines/pyqg_parameterization_benchmarks.git`

## Install in editable mode
```
git clone https://github.com/m2lines/pyqg_generative.git
pip install --editable .
```
