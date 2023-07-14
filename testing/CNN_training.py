import torch
import xarray as xr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

print(device, device_name)

from pyqg_generative.models.ols_model import OLSModel

ds = xr.open_mfdataset('/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/Operator5-dealias-64/[0-1].nc', concat_dim='run', combine='nested')

model = OLSModel()

model.fit(ds,ds)