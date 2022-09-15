import numpy as np
import xarray as xr
from pyqg_generative.tools.spectral_tools import spectrum

def PDF_histogram(x, xmin=None, xmax=None, Nbins=30):
    """
    x is 1D numpy array with data
    How to use:
        first apply without arguments
        Then adjust xmin, xmax, Nbins
    """    
    N = x.shape[0]

    mean = x.mean()
    sigma = x.std()
    
    if xmin is None:
        xmin = mean-4*sigma
    if xmax is None:
        xmax = mean+4*sigma

    bandwidth = (xmax - xmin) / Nbins
    
    hist, bin_edges = np.histogram(x, range=(xmin,xmax), bins = Nbins)

    # hist / N is probability to go into bin
    # probability / bandwidth = probability density
    density = hist / N / bandwidth

    # we assign one value to each bin
    points = (bin_edges[0:-1] + bin_edges[1:]) * 0.5

    #print(f"Number of bins = {Nbins}, over the interval ({xmin},{xmax}), with bandwidth = {bandwidth}")
    #print(f"This interval covers {sum(hist)/N} of total probability")
    
    return points, density

def subgrid_scores(true, mean, gen):
    '''
    Compute scalar metrics for three components of subgrid forcing:
    - Mean subgrid forcing      ~ close to true forcing in MSE
    - Generated subgrid forcing ~ close to true forcing in spectrum
    - Genereted residual        ~ close to true residual in spectrum 
    true - xarray with true forcing
    mean - mean prediction
    gen  - generated prediction

    Result is score, i.e. 1-mse/normalization

    Here we assume that dataset has dimensions run x time x lev x Ny x Nx
    '''
    def R2(x, x_true):
        dims = [d for d in x.dims if d != 'lev']
        return float((1 - ((x-x_true)**2).mean(dims) / (x_true).var(dims)).mean())
    def L2(x, x_true):
        dims = [d for d in x.dims if d != 'lev']
        return float( ((((x-x_true)**2).mean(dims) / (x_true**2).mean(dims))**0.5).mean() )
    
    ds = xr.Dataset()
    # first compute R2 for each layer, and after that normalize
    ds['R2_mean'] = R2(mean, true)
    ds['L2_mean'] = L2(mean, true)

    sp = spectrum(time=slice(None,None)) # power spectrum for full time slice

    ds['sp_true'] = sp(true)
    ds['sp_gen'] = sp(gen)
    ds['R2_total'] = R2(ds.sp_gen, ds.sp_true)
    ds['L2_total'] = L2(ds.sp_gen, ds.sp_true)
    
    ds['sp_true_res'] = sp(true-mean)
    ds['sp_gen_res'] = sp(gen-mean)
    ds['R2_residual'] = R2(ds.sp_gen_res, ds.sp_true_res)
    ds['L2_residual'] = L2(ds.sp_gen_res, ds.sp_true_res)

    gen_res = gen - mean
    true_res = true - mean

    dims = [d for d in mean.dims if d != 'lev']
    # xarray(numpyarray()) because suspect on the memory leak
    ds['var_ratio'] = xr.DataArray(np.array((gen_res**2).mean(dims) / (true_res**2).mean(dims)), dims=['lev'])
    del gen_res
    del true_res
    return ds