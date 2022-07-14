import gcm_filters
import numpy as np
import xarray as xr
import pyqg
import itertools
from functools import wraps

def xarray_to_model(arr):
    nx = len(arr.x)
    return pyqg.QGModel(nx=nx, log_level=0)

coord = lambda x, name: xr.DataArray(x, attrs={'long_name': name})

def ave_lev(arr: xr.DataArray, delta):
    '''
    Average over depth xarray
    delta = H1/H2
    H = H1+H2
    Weights are:
    Hi[0] = H1/H = H1/(H1+H2)=H1/H2/(H1/H2+1)=delta/(1+delta)
    Hi[1] = H2/H = H2/(H1+H2)=1/(1+delta)
    '''
    if 'lev' in arr.dims:
        Hi = xr.DataArray([delta/(1+delta), 1/(1+delta)], dims=['lev'])
        out  = (arr*Hi).sum(dim='lev')
        out.attrs = arr.attrs
        return out
    else:
        return arr

# Decorator for adjusting input to filters
def array_format(func):
    '''
    If input array is not numpy of Ny x Nx,
    do something
    '''
    @wraps(func)
    def wrapper(X, ratio=None):
        '''
        Possible inputs:
        - numpy Ny x Nx
        - numpy Nlev x Ny n Nx
        - xarray with dimensions x and y
        '''
        if isinstance(X, np.ndarray):
            if len(X.shape) == 2:
                return func(X, ratio)
            elif len(X.shape) == 3:
                return np.stack((func(X[0,:], ratio), func(X[1,:], ratio)))
            else:
                raise ValueError('numpy array should be 2 or 3 dimensional')
        elif isinstance(X, xr.DataArray):
            dims = [dim for dim in X.dims if dim not in ['x', 'y']]
            coords = [X[dim] for dim in dims]
            if func.__name__ in map(lambda x: x.__name__, [coarsegrain, cut_off]):
                Y = 0*X.coarsen(x=ratio, y=ratio).mean()
            else:
                Y = 0*X
            for coords in itertools.product(*coords):
                Y.loc[coords] = func(X.loc[coords].values, ratio)
            return Y
        else:
            raise ValueError('input should be numpy array or xarray')
    return wrapper

# First, we define how filter act on 2D numpy arrays
# ratio is the filter width compared to the grid step

# Following operators do not change resolution
@array_format
def gcm_filter(X, ratio):
    f = gcm_filters.Filter(dx_min=1,
        filter_scale=ratio,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR)

    XX = xr.DataArray(X, dims=['y', 'x'])
    return f.apply(XX, dims=('y','x')).values

@array_format
def gauss_filter(X, ratio):
    Xf = np.fft.rfftn(X)
    # let model construct grid of wavectors
    m = pyqg.QGModel(nx=X.shape[0], log_level=0)
    return np.fft.irfftn(Xf * np.exp(-m.wv**2 * (ratio*m.dx)**2 / 24))

@array_format
def model_filter(X, ratio):
    Xf = np.fft.rfftn(X)
    m = pyqg.QGModel(nx=X.shape[0], log_level=0)
    return np.fft.irfftn(Xf*m.filtr)

# Coarsegraining operators change resolution
@array_format
def coarsegrain(X, ratio):
    if ratio%1 != 0:
        raise ValueError('ratio must be an integer')
    if X.shape[0]%ratio != 0:
        raise ValueError('X should be divisible on ratio')
    
    XX = xr.DataArray(X, dims=['y', 'x'])
    return clean_2h(XX.coarsen(y=ratio, x=ratio).mean().values)

@array_format
def cut_off(X, ratio):
    if X.shape[0]%ratio != 0:
        raise ValueError('X should be divisible on ratio')
    n = X.shape[0] // ratio // 2 # coarse grid size / 2
    Xf = np.fft.rfftn(X)
    trunc = np.vstack((Xf[:n,:n+1],
                       Xf[-n:,:n+1])) / ratio**2

    # Remove 2h harmonics which are not invertible (because do not have phase)
    trunc[n,0] = 0
    trunc[:,n] = 0

    return np.fft.irfftn(trunc)

@array_format
def clean_2h(X, ratio):
    '''
    Remove frequencies which potentially
    can harm reversibility of rfftn
    '''
    Xf = np.fft.rfftn(X)
    n = X.shape[0] // 2
    Xf[n,0] = 0
    Xf[:,n] = 0
    return np.fft.irfftn(Xf)

def Operator1(X, ratio):
    return model_filter(cut_off(X, ratio))

def Operator2(X, ratio):
    return gauss_filter(cut_off(X, ratio), 2)

def Operator3(X, ratio):
    return coarsegrain(gcm_filter(X, ratio), ratio)