import gcm_filters
import numpy as np
import xarray as xr
import pyqg
import itertools
from functools import wraps

FILTER_2h_HARMONICS = True

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
    def wrapper(X, nc=None):
        '''
        Possible inputs:
        - numpy Ny x Nx
        - numpy Nlev x Ny n Nx
        - xarray with dimensions x and y
        '''
        if isinstance(X, np.ndarray):
            if len(X.shape) == 2:
                return func(X, nc)
            elif len(X.shape) == 3:
                return np.stack((func(X[0,:], nc), func(X[1,:], nc)))
            else:
                raise ValueError('numpy array should be 2 or 3 dimensional')
        elif isinstance(X, xr.DataArray):
            dims = [dim for dim in X.dims if dim not in ['x', 'y']]
            coords = [X[dim] for dim in dims]
            if func.__name__ in map(lambda x: x.__name__, [coarsegrain, cut_off]):
                ratio = len(X.x) // nc
                Y = 0*X.coarsen(x=ratio, y=ratio).mean()
            else:
                Y = 0*X
            for coords in itertools.product(*coords):
                Y.loc[coords] = func(X.loc[coords].values, nc)
            return Y
        else:
            raise ValueError('input should be numpy array or xarray')
    return wrapper

# First, we define how filter act on 2D numpy arrays
# ratio is the filter width compared to the grid step

# Following operators do not change resolution
@array_format
def gcm_filter(X, nc):
    ratio = X.shape[0] / nc
    f = gcm_filters.Filter(dx_min=1,
        filter_scale=ratio,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR)

    XX = xr.DataArray(X, dims=['y', 'x'])
    return f.apply(XX, dims=('y','x')).values

@array_format
def gauss_filter(X, nc):
    ratio = X.shape[0] / nc
    Xf = np.fft.rfftn(X)
    # let model construct grid of wavectors
    m = pyqg.QGModel(nx=X.shape[0], log_level=0)
    return np.fft.irfftn(Xf * np.exp(-m.wv**2 * (ratio*m.dx)**2 / 24))

@array_format
def model_filter(X, nc):
    '''
    nc here is fictitious parameter (for decorator to work)
    '''
    Xf = np.fft.rfftn(X)
    m = pyqg.QGModel(nx=X.shape[0], log_level=0)
    return np.fft.irfftn(Xf*m.filtr)

# Coarsegraining operators change resolution
@array_format
def coarsegrain(X, nc):
    if nc%1 != 0:
        raise ValueError('ratio must be an integer')
    if X.shape[0]%nc != 0:
        raise ValueError('X should be divisible on ratio')

    ratio = X.shape[0] // nc
    
    XX = xr.DataArray(X, dims=['y', 'x'])
    Y = XX.coarsen(y=ratio, x=ratio).mean().values
    if FILTER_2h_HARMONICS:
        Y = clean_2h(Y)
    return Y

@array_format
def cut_off(X, nc):
    if nc%2 != 0:
        raise ValueError('nc must be even')
    ratio = X.shape[0] / nc
    n = nc // 2 # coarse grid size / 2
    Xf = np.fft.rfftn(X)
    trunc = np.vstack((Xf[:n,:n+1],
                       Xf[-n:,:n+1])) / ratio**2

    if FILTER_2h_HARMONICS:
        # Remove 2h harmonics which are not invertible (because do not have phase)
        trunc[n,0] = 0
        trunc[:,n] = 0

    return np.fft.irfftn(trunc)

@array_format
def clean_2h(X, nc):
    '''
    Remove frequencies which potentially
    can harm reversibility of rfftn
    '''
    Xf = np.fft.rfftn(X)
    n = X.shape[0] // 2
    Xf[n,0] = 0
    Xf[:,n] = 0
    return np.fft.irfftn(Xf)

def Operator1(X, nc):
    return model_filter(cut_off(X, nc))

def Operator2(X, nc):
    return gauss_filter(cut_off(X, nc), nc//2)

def Operator3(X, nc):
    return coarsegrain(gcm_filter(X, nc), nc)

def apply_operator_to_model(q, nc, operator, pyqg_params):
    '''
    Here q is numpy array of Nlev x Ny x Nx
    operator: Operator1, Operator2, Operator3
    pyqg_params: pyqg model parameters
    '''
    # Coarsegrain main variable
    qf = operator(q.astype('float64'), nc)
    
    # Construct pyqg model
    params = pyqg_params.copy()
    params.update(dict(nx=qf.shape[1], log_level=0))
    m = pyqg.QGModel(**params)
    m.q = qf
    m._invert() # Computes real space velocities
    m._calc_derived_fields() # Computes derived variables

    return m

# Computation of subgrid fluxes. As usual, we assume 
# working with numpy arrays with Nlev x Ny x Nx

def divergence(fx, fy):
    m = pyqg.QGModel(nx=fx.shape[1], log_level=0)
    def ddx(x):
        return m.ifft(m.fft(x) * m.ik)
    def ddy(x):
        return m.ifft(m.fft(x) * m.il)
    return ddx(fx) + ddy(fy)

def advect(var, u, v):
    return divergence(var*u, var*v)

def PV_subgrid_flux(q, nc, operator, pyqg_params):
    '''
    Here q is numpy array of Nlev x Ny x Nx
    operator: Operator1, Operator2, Operator3
    pyqg_params: pyqg model parameters
    '''
    m = apply_operator_to_model(q, 1, lambda x, nc: x, pyqg_params) # Just compute u and v
    mf = apply_operator_to_model(q, nc, operator, pyqg_params)

    uqflux = mf.u * mf.q - operator(m.u*m.q, nc)
    vqflux = mf.v * mf.q - operator(m.v*m.q, nc)
    return uqflux, vqflux

def PV_subgrid_forcing(q, nc, operator, pyqg_params):
    m = apply_operator_to_model(q, 1, lambda x, nc: x, pyqg_params) # Just compute u and v
    mf = apply_operator_to_model(q, nc, operator, pyqg_params)
    forcing = advect(mf.q, mf.u, mf.v) - operator(advect(m.q, m.u, m.v), nc)
    return forcing, mf

def PV_forcing_total(q, nc, operator, pyqg_params):
    params1 = pyqg_params.copy()
    params1.update(nx=q.shape[1], log_level=0)
    m1 = pyqg.QGModel(**params1)
    m1.q = q

    # Coarsegrain main variable
    qf = operator(q.astype('float64'), nc)
    params2 = pyqg_params.copy()
    params2.update(nx=qf.shape[1], log_level=0)
    m2 = pyqg.QGModel(**params2)
    m2.q = qf

    for m in [m1, m2]:
        m._invert()
        m._do_advection()
        m._do_friction()
    
    return operator(m1.ifft(m1.dqhdt), nc) - m2.ifft(m2.dqhdt)

def PV_forcing_true_total(q, nc, operator, pyqg_params):
    params1 = pyqg_params.copy()
    params1.update(nx=q.shape[1], log_level=0)
    m1 = pyqg.QGModel(**params1)
    m1.q = q

    # Coarsegrain main variable
    qf = operator(q.astype('float64'), nc)
    params2 = pyqg_params.copy()
    params2.update(nx=qf.shape[1], log_level=0)
    m2 = pyqg.QGModel(**params2)
    m2.q = qf

    for m in [m1, m2]:
        m._invert()
        m._do_advection()
        m._do_friction()
        m._forward_timestep()
    
    dqhdt_1 = (m1.q - q) / m1.dt
    dqhdt_2 = (m2.q - qf) / m2.dt

    return operator(dqhdt_1, nc) - dqhdt_2