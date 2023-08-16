import argparse
import os
import xarray as xr

from pyqg_generative.models.ann_model import ANNModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_args', type=str, default=str({}))
parser.add_argument('--fit_args', type=str, default=str({}))
parser.add_argument('--operator', type=str, default='Operator2-dealias')
args = parser.parse_args()
print(args)

def read_dataset(key=f'eddy/Operator2-dealias-48', idx=range(0,300)):
    base = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{key}'
    files = [f'{base}/{i}.nc' for i in idx]
    return xr.open_mfdataset(files, combine='nested', concat_dim='run')

eddy_48 = read_dataset(f'eddy/{args.operator}-48')
jet_48 = read_dataset(f'jet/{args.operator}-48')
eddy_96 = read_dataset(f'eddy/{args.operator}-96')
jet_96 = read_dataset(f'jet/{args.operator}-96')

ds_list = [eddy_48, jet_48, eddy_96, jet_96]

train = lambda ds: ds.isel(run=slice(0,250))
validate = lambda ds: ds.isel(run=slice(250,275))

model = ANNModel(**eval(args.model_args))
model.fit([train(ds) for ds in ds_list], [validate(ds) for ds in ds_list], **eval(args.fit_args))

del eddy_48, jet_48, eddy_96, jet_96

for conf in ['eddy', 'jet']:
    for res in [32, 48, 64, 96, 128]:
        ds = read_dataset(f'{conf}/{args.operator}-{res}', idx=range(275,300))
        model.test_offline(ds).to_netcdf(f'offline-{conf}-{res}.nc')
        del ds