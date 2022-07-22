from slurm_helpers import run_experiment, DEFAULT_HPC

def job_name(model, operator, resolution):
    return  dict(Operator1='1', Operator2='2')[operator] + \
      '-' + dict(OLSModel='ls', CGANRegression='gan', MeanVarModel='gz')[model] + \
      '-' + str(resolution)

for resolution in [32, 48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
        for model in ['OLSModel', 'MeanVarModel', 'CGANRegression']:
            _operator = operator+'-'+str(resolution)
            train_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/' + _operator + '/*.nc'
            transfer_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet/' + _operator + '/*.nc'
            model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/' + _operator + '/' + model
            script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/train_model.py'
            model_args = dict(nx=resolution) if model=='CGANRegression' else {}
            hours = 10 if resolution==96 else 3
            hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 32, 'hours': hours, \
                'job-name': job_name(model, operator, resolution), 'output': 'training.txt'})
            args = dict(model=model, train_path=train_path, transfer_path=transfer_path, model_args=model_args)
            run_experiment(model_folder, script_py, hpc, args, delete=True)