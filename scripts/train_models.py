from slurm_helpers import run_experiment, DEFAULT_HPC

def job_name(model, operator, resolution):
    return  dict(Operator1='1', Operator2='2')[operator] + \
      '-' + dict(OLSModel='ls', CGANRegression='gan', CVAERegression='vae', MeanVarModel='gz')[model] + \
      '-' + str(resolution)

for resolution in [64, 96]:
    for operator in ['Operator1', 'Operator2']:
        for model, folder in zip(['CVAERegression'], ['CVAERegression-None-1000']):
            _operator = operator+'-'+str(resolution)
            train_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/' + _operator + '/*.nc'
            transfer_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet/' + _operator + '/*.nc'
            model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/' + _operator + '/' + folder
            script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/train_model.py'
            fit_args = {}
            hours = 20 if resolution==96 else 6
            mem = 64
            if folder == 'CVAERegression':
                model_args = {}
            elif folder == 'CVAERegression-fixed':
                model_args = dict(decoder_var='fixed')
            elif folder == 'CVAERegression-None':
                model_args = dict(regression='None')
                fit_args = dict(num_epochs=200)
                hours = 40 if resolution==96 else 12
            elif folder == 'CVAERegression-None-1000':
                model_args = dict(regression='None')
                fit_args = dict(num_epochs=1000)
                if resolution == 96:
                    hours = 70
                elif resolution == 64:
                    hours = 40
                elif resolution == 48:
                    hours = 20
                elif resolution == 32:
                    hours = 10
                else:
                    raise ValueError('Resolution not supported')
            
            hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': mem, 'gres': 'gpu:v100:1', 'hours': hours, \
                'job-name': job_name(model, operator, resolution), 'output': 'training.txt'})
            args = dict(model=model, train_path=train_path, transfer_path=transfer_path, model_args=model_args, fit_args=fit_args)
            run_experiment(model_folder, script_py, hpc, args, delete=True)