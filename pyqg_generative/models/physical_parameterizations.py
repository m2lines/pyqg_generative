import numpy as np
import xarray as xr
import pyqg
from pyqg_generative.models.parameterization import Parameterization
import pyqg.parameterizations as p

class PhysicalParameterization(Parameterization):
    def generate_latent_noise(self, ny, nx):
        return 0
    def predict_snapshot(self, m, noise):
        return self.subgrid_model(m)
    def predict(self, ds, M=1000):
        X = np.array(ds.q)
        Y = 0*X
        for r in range(X.shape[0]):
            for t in range(X.shape[1]):
                q = X[r,t]
                pyqg_params = eval(ds.attrs['pyqg_params']).copy()
                pyqg_params.update({'nx': X.shape[-1], 'log_level': 0})
                m = pyqg.QGModel(**pyqg_params)
                m.q = q.astype('float64')
                m._invert() # Computes real space velocities
                m._calc_derived_fields() # Computes derived variables
                Y[r,t] = self.predict_snapshot(m)
        Y = xr.DataArray(Y, dims=['run', 'time', 'lev', 'y', 'x'])
        return xr.Dataset({'q_forcing_advection': Y, 
            'q_forcing_advection_mean': Y, 'q_forcing_advection_var': Y*0})

class ZannaBolton(PhysicalParameterization):
    def __init__(self):
        self.subgrid_model = p.ZannaBolton2020_q()

class ReynoldsStress(PhysicalParameterization):
    def __init__(self):
        self.subgrid_model = p.Reynolds_stress()

class HybridSymbolic(PhysicalParameterization):
    def __init__(self):
        self.subgrid_model = p.HybridSymbolic()

class ADM(PhysicalParameterization):
    def __init__(self):
        self.subgrid_model = p.ADM()

class BackscatterEddy(PhysicalParameterization):
    def __init__(self):
        self.subgrid_model = p.BackscatterBiharmonic(np.sqrt(0.007), 1.2)

class BackscatterJet(PhysicalParameterization):
    def __init__(self):
        self.subgrid_model = p.BackscatterBiharmonic(np.sqrt(0.005), 0.8)