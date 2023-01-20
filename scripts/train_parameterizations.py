from slurm_helpers import run_experiment, DEFAULT_HPC

BASIC_FOLDER = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled'
SCRIPT_PATH = '/home/pp2681/pyqg_generative/pyqg_generative/tools/train_model.py'

CONFIGURATION = 'eddy'; TRANSFER = 'jet'
MODELS_FOLDER = 'models'
NUM_REALIZATIONS = 5

def fail_function(model_folder):
    import os
    error = model_folder+'/training.err'
    if not os.path.exists(error):
        return True
    with open(error) as f:
        for line in f.readlines():
            if 'Aborted' in line:
                return True
    return False

def job_name(model, operator, resolution):
    return  dict(Operator1='1', Operator2='2')[operator] + \
      dict(OLSModel='ls', CGANRegression='gan', CVAERegression='vae', MeanVarModel='gz')[model] + \
      str(resolution)

for model, folder in zip(['MeanVarModel', 'CGANRegression', 'CVAERegression'], ['GZ', 'GAN', 'VAE']):
    for resolution in [48, 64, 96]:
        for operator in ['Operator1', 'Operator2']:
            print(resolution, operator, folder, CONFIGURATION)
            input()
            for realization in range(0,NUM_REALIZATIONS):
                _operator = operator+'-'+str(resolution)
                train_path = f'{BASIC_FOLDER}/{CONFIGURATION}/' + _operator + '/*.nc'
                transfer_path = f'{BASIC_FOLDER}/{TRANSFER}/' + _operator + '/*.nc'
                model_folder = f'{BASIC_FOLDER}/{MODELS_FOLDER}/' + _operator + '/' + folder+'-'+str(realization)

                if fail_function(model_folder):                
                    # basic parameters
                    fit_args = {}
                    model_args = {}
                    hours = {32:1, 48: 10, 64: 10, 96: 20}[resolution] 
                    
                    if folder == 'GAN':
                        model_args = dict(nx=resolution)
                    
                    hpc = DEFAULT_HPC._update(
                        {'ntasks': 1, 'mem': 32, 'gres': 'gpu:mi50:1', 'hours': hours, \
                        'job-name': CONFIGURATION[0]+job_name(model, operator, resolution), 
                        'output': 'training.txt', 'error': 'training.err', 
                        'begin': 'now'})
                    args = dict(model=model, train_path=train_path, transfer_path=transfer_path, model_args=model_args, fit_args=fit_args)
                    run_experiment(model_folder, SCRIPT_PATH, hpc, args, delete=False)