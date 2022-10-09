from slurm_helpers import run_experiment, DEFAULT_HPC

def job_name(model, operator, resolution):
    return  dict(Operator1='1', Operator2='2')[operator] + \
      dict(OLSModel='ls', CGANRegression='gan', CVAERegression='vae', MeanVarModel='gz')[model] + \
      str(resolution)

NUM_REALIZATIONS = 5

def fail_function(model_folder):
    error = model_folder+'/training.err'
    with open(error) as f:
        for line in f.readlines():
            if 'Aborted' in line:
                return True
    return False

for resolution in [48, 64, 96]:
    for operator in ['Operator1', 'Operator2']:
        print(resolution, operator)
        input()
        for model, folder in zip(['OLSModel'], ['OLSModel']):
            for realization in range(0,NUM_REALIZATIONS):
                _operator = operator+'-'+str(resolution)
                train_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/' + _operator + '/*.nc'
                transfer_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet/' + _operator + '/*.nc'
                model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models_retrain/' + _operator + '/' + folder+'-'+str(realization)
                script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/train_model.py'

                if fail_function(model_folder):                
                    # basic parameters
                    fit_args = {}
                    model_args = {}
                    
                    if model == 'OLSModel':
                        hours = {32:1, 48: 1, 64: 2, 96: 4}[resolution]
                    mem=32
                    
                    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': mem, 'gres': 'gpu:mi50:1', 'hours': hours, \
                        'job-name': job_name(model, operator, resolution), 'output': 'training.txt', 'error': 'training.err'})
                    args = dict(model=model, train_path=train_path, transfer_path=transfer_path, model_args=model_args, fit_args=fit_args)
                    run_experiment(model_folder, script_py, hpc, args, delete=True)