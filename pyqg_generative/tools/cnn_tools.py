import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import xarray as xr
from time import time

def init_seeds():
    '''
    https://pytorch.org/docs/stable/notes/randomness.html
    Pytorch creates different seeds for each runs
    and for each process, and this seed is used to init numpy!

    Use only if workers are spawned. Otherwise 
    all calls will be identical!!!
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def read_dataset(path):
    return xr.open_mfdataset(path, combine='nested', concat_dim='run')

def weights_init(m):
    '''
    m - nn.Module or nn.Sequential
    Initializes weights randomly with 0 mean and 0.02 std
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def batch_norm(bn, nchannels, ny, nx):
    if bn=='BatchNorm':
        return nn.BatchNorm2d(nchannels)
    elif bn=='LayerNorm':
        return nn.LayerNorm([nchannels, ny, nx])
    elif bn=='InstanceNorm':
        return nn.InstanceNorm2d(nchannels, affine=True)
    elif bn=='None':
        return nn.Identity()
    else:
        print('Wrong bn parameter, bn=', bn)

def make_block(in_channels: int, out_channels: int, kernel_size: int, 
        ReLU = 'ReLU', batch_norm = True) -> list:
    '''
    Packs convolutional layer and optionally ReLU/BatchNorm2d
    layers in a list
    '''
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
        padding='same', padding_mode='circular')
    block = [conv]
    if ReLU == 'ReLU':
        block.append(nn.ReLU())
    elif ReLU == 'LeakyReLU':
        block.append(nn.LeakyReLU(0.2))
    elif ReLU == 'False':
        pass
    else:
        print('Error: wrong ReLU parameter')
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    return block

class AndrewCNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, ReLU = 'ReLU') -> list:
        '''
        Packs sequence of 8 convolutional layers in a list.
        First layer has n_in input channels, and Last layer has n_out
        output channels
        '''
        super().__init__()
        blocks = []
        blocks.extend(make_block(n_in,128,5,ReLU))                #1
        blocks.extend(make_block(128,64,5,ReLU))                  #2
        blocks.extend(make_block(64,32,3,ReLU))                   #3
        blocks.extend(make_block(32,32,3,ReLU))                   #4
        blocks.extend(make_block(32,32,3,ReLU))                   #5
        blocks.extend(make_block(32,32,3,ReLU))                   #6
        blocks.extend(make_block(32,32,3,ReLU))                   #7
        blocks.extend(make_block(32,n_out,3,'False',False))       #8
        self.conv = nn.Sequential(*blocks)
    def forward(self, x):
        return self.conv(x)
    def compute_loss(self, x, ytrue):
        '''
        In case you want to use this block for training 
        as regression model with standard trainer cnn_tools.train()
        '''
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}

def DCGAN_discriminator(in_channels, ndf=64, bn='BatchNorm'):
    '''
    in_channels - number of images to compare
    ndf - some free parameter. Simpler to fix
    Discriminator is supposed to take as input images of 64x64
    Discriminator from tutorial DCGAN:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    Note that sigmoid is removed in favor to better generalizability
    to other architectures (WGAN, and so on), 
    see https://www.researchgate.net/profile/Hao-Wen-Dong/publication/330673057_Towards_a_Deeper_Understanding_of_Adversarial_Losses/links/5c5550cc92851c22a3a28c0e/Towards-a-Deeper-Understanding-of-Adversarial-Losses.pdf
    '''

    model = \
        nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            batch_norm(bn, ndf * 2, 16, 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            batch_norm(bn, ndf * 4, 8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            batch_norm(bn, ndf * 8, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
    return model

def extract_arrays(ds: xr.DataArray, features: list[str], lev=slice(0,2)):
    '''
    Input dataset is expected to have dimensions:
    run, time, lev, lat, lon

    Extracts features from xr.DataArray and stack them
    in feature dimension. Run and time dimensions are
    reshaped as batch dimension
    Nbatch x Nfeatures x Ny x Nx, where
    Nbatch = Nruns * Ntime.

    If you want to extract only one layer, 
    change to lev=slice(0,1) or lev=slice(1,2)
    '''
    if not isinstance(features, list):
        features = [features]
        
    # concatenate features in old "level" dimension
    arr = np.concatenate([
        ds.isel(lev=lev)[feature].values for feature in features
    ], axis=-3)

    return arr.reshape(-1, *arr.shape[-3:]).astype('float32')

def array_to_dataset(ds: xr.DataArray, array: np.array, features: list[str], postfix: str=''):
    '''
    Takes array ds only to read dimensions
    '''
    ds_out = xr.Dataset()
    shape = ds.q.shape
    for i, feature in enumerate(features):
        idx = slice(2*i, 2*i+2) # slice gives two layers
        ds_out[feature+postfix] = 0*ds.q + array[:, idx].reshape(shape) # expand dimensions from ds
    return ds_out

def channelwise_function(X: np.array, fun) -> np.array:
    '''
    For array X of size 
    Nbatch x Nfeatures x Ny x Nx
    applies function "fun" for each channel
    and returns array of size
    1 x Nfeatures x 1 x 1
    '''

    N_features = X.shape[1]
    out = np.zeros((1,N_features,1,1))
    for n_f in range(N_features):
        out[0,n_f,0,0] = fun(X[:,n_f,:,:])

    return out.astype('float32')

def channelwise_std(X: np.array) -> np.array:
    '''
    For array X of size 
    Nbatch x Nfeatures x Ny x Nx
    Computes standard deviation for each channel
    with double precision
    and returns array of size
    1 x Nfeatures x 1 x 1
    '''
    return channelwise_function(X.astype('float64'), np.std)

def channelwise_mean(X: np.array) -> np.array:
    '''
    For array X of size 
    Nbatch x Nfeatures x Ny x Nx
    Computes mean for each channel
    with double precision
    and returns array of size
    1 x Nfeatures x 1 x 1
    '''
    return channelwise_function(X.astype('float64'), np.mean)

class ChannelwiseScaler:
    '''
    Class containing std and mean
    values for each channel
    '''
    def __init__(self, X: np.array):
        ''' 
        Stores std and mean values.
        X is array of size
        Nbatch x Nfeatures x Ny x Nx.
        '''
        self.mean = channelwise_mean(X)
        self.std  = channelwise_std(X)
    def direct(self, X):
        '''
        Remove mean and normalize
        '''
        return (X-self.mean) / self.std
    def inverse(self, X):
        return X * self.std + self.mean
    def normalize(self, X):
        '''
        Divide by std
        '''
        return X / self.std
    def denormalize(self, X):
        return X * self.std
    def normalize_var(self, X):
        '''
        Divide by std squared,
        use for quadratic variables
        '''
        return X / (self.std**2)
    def denormalize_var(self, X):
        '''
        Multiply by std squared,
        use for quadratic variables
        '''
        return X * (self.std**2)

class AverageLoss():
    '''
    Accumulates dictionary of losses over batches
    and computes mean for epoch.
    List of keys to accumulate given by 'losses'
    Usage:
    Init before epoch. 
    Accumulate over batches for given epoch
    Average after epoch
    '''
    def __init__(self, log_dict: dict[str, list]):
        self.init_me = True
        self.count = {}

    def accumulate(self, log_dict: dict[str, list], losses: dict[str, float], n: int):
        '''
        log_dict: dictionary of timeseries
        losses: dictionary of loss on a batch
        n: number of elements in batch
        '''
        keys = losses.keys()
        if (self.init_me):
            new_keys = set(losses.keys())-set(log_dict.keys())
            for key in new_keys:
                log_dict[key] = []
            for key in keys:
                self.count[key] = 0
                log_dict[key].append(0.)
            self.init_me = False

        for key in keys:
            value = losses[key]
            # extract floats from scalar tensors
            if isinstance(value, torch.Tensor):
                try:
                    value = value.item()
                except:
                    value = value.cpu().numpy()
            log_dict[key][-1] += value * n
            self.count[key] += n
    
    def average(self, log_dict: dict[str, list]):
        '''
        Updates last element of dictionary with 
        average value
        '''
        for key in self.count.keys():
            log_dict[key][-1] = log_dict[key][-1] / self.count[key]

def dict_postfix(mydict: dict[float], postfix):
    return {str(key)+postfix: val for key, val in mydict.items()}

def minibatch(*arrays: np.array, batch_size=64, shuffle=True):
    '''
    Recieves arbitrary number of numpy arrays
    of size 
    Nbatch x Nfeatures x Ny x Nx.
    Returns multiple batches of tensors of size 
    batch_size x Nfeatures x Ny x Nx.
    '''
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        yield tuple(torch.as_tensor(array[idx]) for array in arrays)

def evaluate_test(net, *arrays: np.array, batch_size=64, postfix='_test'):
    '''
    Updates logger on test dataset
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # To do not update parameters of 
    # Batchnorm2d layers
    net.eval()

    logger = AverageLoss(net.log_dict)
    for xy in minibatch(*arrays, batch_size=batch_size):
        with torch.no_grad():
            losses = net.compute_loss(*[x.to(device) for x in xy])
        logger.accumulate(net.log_dict, dict_postfix(losses, postfix), len(xy[0]))
    
    logger.average(net.log_dict)
    net.train()

def train(net, X_train: np.array, Y_train:np. array, 
        X_test: np.array, Y_test: np.array, 
        num_epochs, batch_size, learning_rate):
    '''
    X_train, Y_train are arrays of size
    Nbatch x Nfeatures x Ny x Nx.
    For this function to use, class 'net'
    should implement function compute_loss(x,y) returning 
    dictionary, where key 'loss'
    is used for optimization,
    while others are used for logger.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"Training starts on device {device}, number of samples {len(X_train)}")

    # Switch batchnorm2d layer to training mode
    net.train()

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

def apply_function(net, fun, X: np.array) -> np.array:
    '''
    Apply forward function to array X of size
    Nsamples x Nfeatures x Ny x Nx.
    Returns predictions with similar size
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # stack batch predictions in a list
    preds = []
    for x, in minibatch(X, batch_size=64, shuffle=False):
        with torch.no_grad():
            y = fun(x.to(device))
            preds.append(y.cpu().numpy())

    # stack list dimension to batch dimension
    # Result has shape Nsamples x Nfeatures x Ny x Nx
    return np.vstack(preds)