import numpy as np

class ConfigurationDict(dict):
    def _update(self, d):
        '''
        Copy, modify and return new
        dictionary
        '''
        dd = self.copy()
        dd.update(d)
        return ConfigurationDict(dd)
    def nx(self, _nx):
        '''
        Set nx value and time step
        '''
        dd = self.copy()
        dd['nx'] = _nx
        if _nx == 256:
            dt = 3600
        if _nx == 128 or _nx == 96:
            dt = 7200
        if _nx <= 64:
            dt = 14400

        dd['dt'] = dt
        return ConfigurationDict(dd)

DAY = 86400
YEAR = 360*DAY
EDDY_PARAMS = ConfigurationDict({'nx': 64, 'dt': 3600*4, 'tmax': 10*YEAR, 'tavestart': 5*YEAR})
JET_PARAMS = ConfigurationDict({'nx': 64, 'dt': 3600*4, 'tmax': 10*YEAR, 'tavestart': 5*YEAR, 'rek': 7e-08, 'delta': 0.1, 'beta': 1e-11})

SAMPLE_SLICE = slice(-40, None) # in indices
AVERAGE_SLICE = slice(360*5*DAY,None) # in seconds
AVERAGE_SLICE_ANDREW = slice(44,None) # in indices
ANDREW_1000_STEPS = 3600000