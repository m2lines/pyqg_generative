import argparse
import os
import xarray as xr

from pyqg_generative.models.cgan_regression import CGANRegression
from pyqg_generative.models.cvae_regression import CVAERegression
from pyqg_generative.models.cvae_bottleneck import CVAEBottleneck
from pyqg_generative.models.mean_var_model import MeanVarModel
from pyqg_generative.models.ols_model import OLSModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='OLSModel')
parser.add_argument('--model_args', type=str, default=str({}))
parser.add_argument('--fit_args', type=str, default=str({}))
parser.add_argument('--train_path', type=str, default='/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/Operator2-64')
parser.add_argument('--transfer_path', type=str, default='/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet/Operator2-64')
args = parser.parse_args()
print(args)

try:
    import torch
    print(torch.__version__)

    # How many GPUs are there?
    print(torch.cuda.device_count())

    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # Is PyTorch using a GPU?
    print(torch.cuda.is_available())
except:
    print('No GPU was detected.')

ds = xr.open_mfdataset(args.train_path, combine='nested', concat_dim='run')
train = ds.isel(run=slice(0,250))
validate = ds.isel(run=slice(250,275))
test = ds.isel(run=slice(275,300))
transfer = xr.open_mfdataset(args.transfer_path, combine='nested', concat_dim='run').isel(run=slice(0,25))

model = eval(args.model)(**eval(args.model_args))
model.fit(train, validate, **eval(args.fit_args))

del train, validate

model.test_offline(test).to_netcdf('offline_test.nc')
model.test_offline(transfer).to_netcdf('offline_transfer.nc')

del ds, test, transfer