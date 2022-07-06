import sys
import os
sys.path.append(os.getcwd()) # https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
import torch
import argparse
import json

from tools.cnn_tools import read_dataset
from models.cgan_model import CGANModel

if not torch.cuda.is_available():
    print('Warning CUDA is not available:', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--n_latent', type=int, default=2)
parser.add_argument('--minibatch_discrimination', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=0)
parser.add_argument('--loss_type', type=str, default="WGAN")
parser.add_argument('--lambda_MSE_mean', type=float, default=0.)
parser.add_argument('--lambda_MSE_sample', type=float, default=0.)
parser.add_argument('--ncritic', type=int, default=5)
parser.add_argument('--training', type=str, default="DCGAN")
parser.add_argument('--generator', type=str, default="Andrew")
parser.add_argument('--discriminator', type=str, default="DCGAN")
parser.add_argument('--bn', type=str, default="None")
parser.add_argument('--GP_shuffle', type=int, default=1)
parser.add_argument('--regression', type=int, default=0)


parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--ensemble_size', type=int, default=100)

args = parser.parse_args()
print(args)
with open('args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

train = read_dataset("/scratch/zanna/data/pyqg/data/train/*.nc")
test = read_dataset("/scratch/zanna/data/pyqg/data/test/*.nc")
transfer = read_dataset("/scratch/zanna/data/pyqg/data/transfer/*.nc")

inputs = [('q',0), ('q',1)]
targets = [('q_forcing_advection', 0), ('q_forcing_advection', 1)]

model = CGANModel(inputs, targets,
        n_latent=args.n_latent, 
        minibatch_discrimination=bool(args.minibatch_discrimination), 
        deterministic=bool(args.deterministic), loss_type=args.loss_type,
        lambda_MSE_mean=args.lambda_MSE_mean, 
        lambda_MSE_sample=args.lambda_MSE_sample,
        ncritic=args.ncritic,
        training = args.training, 
        generator=args.generator, discriminator=args.discriminator,
        bn=args.bn, GP_shuffle=bool(args.GP_shuffle),
        regression = bool(args.regression))

stats = model.fit(train.isel(run=slice(0,225)), train.isel(run=slice(225,None)), num_epochs=args.num_epochs)
stats.attrs = vars(args)
stats.to_netcdf('training_stats.nc')
torch.save(model, 'net_state')

model.predict(test, ensemble_size=args.ensemble_size, stats=stats).to_netcdf('test.nc')
model.predict(transfer, ensemble_size=args.ensemble_size, stats=stats).to_netcdf('transfer.nc')

torch.save(model, 'net_state')