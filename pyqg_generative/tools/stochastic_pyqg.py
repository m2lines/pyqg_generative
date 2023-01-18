import pyqg

class noise_time_sampler():
    '''
    Class which have memory and implements 
    time smoothing of noise

    Usage:
    sampler = AR1_sampler(10)
    if sampler.update(lambda: np.random.randn()):
        compute SGS_force based on sampler.noise
    '''
    def __init__(self, nsteps):
        self.nsteps = nsteps
    def update(self, generate_noise):
        '''
        Noise state is stored in self.noise
        variable, and updated as approriate

        generate_noise - function without arguments
        which can generate white in time noise appropriate
        for latent space of a given CNN model

        Returns bool. 
        If True: need to compute SGS force
        if False: need not
        '''
        raise NotImplementedError

class AR1_sampler(noise_time_sampler):
    def update(self, generate_noise):
        '''
        Sampling from work https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1995.0126
        Predicts noise using AR1 model
        at the next time step.
        nsteps - decorrelation time in time steps
        If nsteps = 1, AR1 is equal to white noise in time sampling
        If nsteps < 0, correlation time is infinite and initial noise is 
        preserved forever
        in case of first call, self.noise is initialized with input noise
        '''
        if hasattr(self, 'noise'):
            if self.nsteps > 0:
                a = 1 - 1/self.nsteps
                b = (1/self.nsteps * (2 - 1/self.nsteps))**0.5
            else:
                a = 1
                b = 0
            self.noise = a * self.noise + b * generate_noise()
        else:
            self.noise = generate_noise()
        # Return True if need to compute SGS force
        compute = True
        return compute

class constant_sampler(noise_time_sampler):
    def update(self, generate_noise):
        '''
        Sampling from work https://www.sciencedirect.com/science/article/pii/S1463500317300100
        '''
        compute = True
        if hasattr(self, 'noise'):
            if self.counter % self.nsteps == 0:
                self.noise = generate_noise()
                self.counter = 1
            else:
                self.counter += 1
                compute = False
        else:
            self.noise = generate_noise()
            self.counter = 1
        return compute

class stochastic_QGModel(pyqg.QGModel):
    '''
    Accumulate noise for correlated SGS closures
    '''
    def __init__(self, pyqg_params, sampling_type='AR1', nsteps=1):
        super().__init__(**pyqg_params)
        self.sampling_type = sampling_type
        if sampling_type == 'AR1':
            self.noise_sampler = AR1_sampler(nsteps)
        elif sampling_type == 'constant':
            self.noise_sampler = constant_sampler(nsteps)
        elif sampling_type == 'deterministic':
            pass
        else:
            raise ValueError('Unknown sampling type')
