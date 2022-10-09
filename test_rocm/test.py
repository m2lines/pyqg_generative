import xarray as xr
import torch
import torch.optim as optim
from time import time

from pyqg_generative.models.ols_model import OLSModel
from pyqg_generative.tools.cnn_tools import AndrewCNN, ChannelwiseScaler, log_to_xarray, train, \
    apply_function, extract, prepare_PV_data, save_model_args, AverageLoss, evaluate_test, minibatch

ds_train = xr.open_mfdataset('/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/Operator1-96/[0-2].nc', combine='nested', concat_dim='run')
ds_test = xr.open_mfdataset('/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/Operator1-96/[3-5].nc', combine='nested', concat_dim='run')

model = OLSModel()
net = model.net

X_train, Y_train, X_test, Y_test, model.x_scale, model.y_scale = \
            prepare_PV_data(ds_train, ds_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"Training starts on device {device_name}, number of samples {len(X_train)}")

net.train()

num_epochs = 100
learning_rate=0.001
batch_size=64

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)  

try:
    net.log_dict
except:
    net.log_dict = {}

t_s = time()
for epoch in range(0,num_epochs):
    t_e = time()
    logger = AverageLoss(net.log_dict)
    for x, y in minibatch(X_train, Y_train, batch_size=batch_size):
        optimizer.zero_grad()
        losses = net.compute_loss(x.to(device),y.to(device))
        losses['loss'].backward() # optimize over the 'loss' value
        optimizer.step()
        logger.accumulate(net.log_dict, losses, len(x))
    scheduler.step()

    logger.average(net.log_dict)
    evaluate_test(net, X_test, Y_test, batch_size=batch_size)
    t = time()
    print('[%d/%d] [%.2f/%.2f] Loss: [%.3f, %.3f]' 
        % (epoch+1, num_epochs,
        t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1),
        net.log_dict['loss'][-1], net.log_dict['loss_test'][-1]))

del ds_train, ds_test