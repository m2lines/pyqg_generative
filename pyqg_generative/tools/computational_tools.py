import numpy as np
import xarray as xr

def PDF_histogram(x, xmin=None, xmax=None, Nbins=30):
    """
    x is 1D numpy array with data
    How to use:
        first apply without arguments
        Then adjust xmin, xmax, Nbins
    """    
    N = x.shape[0]

    if xmin is None or xmax is None:
        mean = x.mean()
        sigma = x.std()
        xmin = mean-4*sigma; xmax = mean+4*sigma

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