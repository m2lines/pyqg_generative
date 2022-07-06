import sys
import os
sys.path.append(os.getcwd()) # https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
import torch

from tools.cnn_tools import read_dataset
from models.mean_var_model import MeanVarModel

if not torch.cuda.is_available():
    print('Warning CUDA is not available:', torch.cuda.is_available())

train = read_dataset("/scratch/zanna/data/pyqg/data/train/*.nc")
test = read_dataset("/scratch/zanna/data/pyqg/data/test/*.nc")

inputs = [('q', 0), ('q', 1)]
targets = [('q_forcing_advection', 0), ('q_forcing_advection', 1)]

model = MeanVarModel(inputs, targets)

model.fit(train, test, num_epochs=100, batch_size=64, learning_rate=0.001)

predict_train = model.predict(train)
predict_test = model.predict(test)

predict_train.to_netcdf('train.nc')
predict_test.to_netcdf('test.nc')
torch.save(model, 'net_state')