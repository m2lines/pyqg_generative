import numpy as np
import xarray as xr
import pyqg
from pyqg_generative.tools.operators import coord
from pyqg_generative.tools.parameters import AVERAGE_SLICE_ANDREW

class spectrum():
    def __init__(self, type='power', averaging=False, truncate=False, time=AVERAGE_SLICE_ANDREW):
        '''
        Init type of transform here
        Usage to compute power spectrum:
        spectrum()(q)
        '''
        self.type = type
        self.averaging = averaging
        self.truncate = truncate
        self.time = time

    def test(self, sp, *x):
        def sel(_x):
            __x = _x.isel(time=self.time)
            return __x - __x.mean(dim=('x','y'))

        k = sp.k; dk = k[1] - k[0]
        Esp = sp.sum() * dk
        if self.type == 'power':
            E = sel(x[0])**2
        elif self.type == 'energy':
            E = 0.5*sel(x[0])**2
        elif self.type == 'cospectrum':
            E = sel(x[0]) * sel(x[1])
        elif self.type == 'cross_layer':
            _x = sel(x[0])
            E = _x[:,:,0] * _x[:,:,1]

        E = E.mean(dim=('run','time','x','y'))
        try:
            E = E.sum(dim='lev')
        except:
            pass

        rel_error = float(np.abs((Esp-E)/E))
        #print(f'Relative error in spectral sum for {self.type}: {rel_error}')

    def fft2d(self, _xarray):
        M = _xarray.shape[-1] * _xarray.shape[-2]
        x = _xarray.isel(time=self.time).values.astype('float64')
        return np.fft.rfftn(x, axes=(-2,-1)) / M

    def isotropize(self, af2, *x, name, description, units):
        m = pyqg.QGModel(nx=len(x[0].x), log_level=0)
        if self.type != 'cross_layer':
            sp_list = []
            for z in [0,1]:
                k, sp = calc_ispec(m, af2[z,:,:], averaging=self.averaging, 
                    truncate=self.truncate)
                sp_list.append(sp) # as power spectral density

            sp = xr.DataArray(np.stack(sp_list, axis=0), dims=['lev', 'k'], 
                coords=[('lev', [1,2]), coord(k, 'isotropic wavenumber, $m^{-1}$')],
                attrs={'long_name': name, 'description': description, 'units': units})
        elif self.type == 'cross_layer':
            k, sp = calc_ispec(m, af2[:,:], averaging=self.averaging, 
                truncate=self.truncate)
            
            sp = xr.DataArray(sp, dims=['k'], 
                coords=[coord(k, 'isotropic wavenumber, $m^{-1}$')],
                attrs={'long_name': name, 'description': description, 'units': units})
        return sp
    
    def __call__(self, *x, name='', description='', units=''):
        '''
        *x - list of xarray tensors,
        nrun x ntime x nlev x Ny x Nx
        '''
        if self.type == 'power':
            af2 = np.abs(self.fft2d(x[0]))**2
        elif self.type == 'energy':
            af2 = np.abs(self.fft2d(x[0]))**2 / 2
        elif self.type == 'cospectrum':
            af2 = np.real(np.conj(self.fft2d(x[0])) * self.fft2d(x[1]))
        elif self.type == 'cross_layer':
            xf = self.fft2d(x[0])
            af2 = np.real(np.conj(xf[:,:,0]) * xf[:,:,1])
        
        af2 = af2.mean(axis=(0,1))

        sp = self.isotropize(af2, *x, name=name, description=description, units=units)

        #self.test(sp, *x)

        return sp

def calc_ispec(model, _var_dens, averaging = True, truncate=True, nd_wavenumber=False, nfactor = 1):
    """Compute isotropic spectrum `phr` from 2D spectrum of variable signal2d.

    Parameters
    ----------
    model : pyqg.Model instance
        The model object from which `var_dens` originates
    
    var_dens : squared modulus of fourier coefficients like this:
        np.abs(signal2d_fft)**2/m.M**2

    averaging: If True, spectral density is estimated with averaging over circles,
        otherwise summation is used and Parseval identity holds

    truncate: If True, maximum wavenumber corresponds to inner circle in Fourier space,
        otherwise - outer circle
    
    nd_wavenumber: If True, wavenumber is nondimensional: 
        minimum wavenumber is 1 and corresponds to domain length/width,
        otherwise - wavenumber is dimensional [m^-1]

    nfactor: width of the bin in sqrt(dk^2+dl^2) units

    Returns
    -------
    kr : array
        isotropic wavenumber
    phr : array
        isotropic spectrum

    Normalization:
    signal2d.var() = phr.sum() * (kr[1] - kr[0])
    """

    # account for complex conjugate
    var_dens = np.copy(_var_dens)
    var_dens[...,0] /= 2
    var_dens[...,-1] /= 2

    ll_max = np.abs(model.ll).max()
    kk_max = np.abs(model.kk).max()

    if truncate:
        kmax = np.minimum(ll_max, kk_max)
    else:
        kmax = np.sqrt(ll_max**2 + kk_max**2)
    
    kmin = np.minimum(model.dk, model.dl)

    dkr = np.sqrt(model.dk**2 + model.dl**2) * nfactor

    # left border of bins
    kr = np.arange(kmin, kmax-dkr, dkr)
    
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        if averaging:
            fkr =  (model.wv>=kr[i]) & (model.wv<=kr[i]+dkr)    
            if fkr.sum() == 0:
                phr[i] = 0.
            else:
                phr[i] = var_dens[fkr].mean() * (kr[i]+dkr/2) * np.pi / (model.dk * model.dl)
        else:
            fkr =  (model.wv>=kr[i]) & (model.wv<kr[i]+dkr)
            phr[i] = var_dens[fkr].sum() / dkr
        
        phr[i] *= 2
    
    # convert left border of the bin to center
    kr = kr + dkr/2

    # convert to non-dimensional wavenumber 
    # preserving integral over spectrum
    if nd_wavenumber:
        kr = kr / kmin
        phr = phr * kmin

    return kr, phr