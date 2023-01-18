from slurm_helpers import run_experiment, DEFAULT_HPC

def job_name(model, operator, resolution):
    return  dict(Operator1='1', Operator2='2')[operator] + \
      dict(OLSModel='ls', CGANRegression='gan', CVAERegression='vae', MeanVarModel='gz')[model] + \
      str(resolution)

CONFIGURATION = 'eddy'; TRANSFER = 'jet'
MODELS_FOLDER = 'models_retrain'
NUM_REALIZATIONS = 5

#CONFIGURATION = 'jet_300'; TRANSFER = 'eddy'
#MODELS_FOLDER = 'models_jet'
#NUM_REALIZATIONS = 3

# CONFIGURATION = 'jet_40years'; TRANSFER = 'eddy'
# MODELS_FOLDER = 'models_jet_40years'
# NUM_REALIZATIONS = 3

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

def copy_mean_model(target_folder, source_folder):
    import os
    os.system(f'rm -rf {target_folder}')
    if not os.path.exists(target_folder):
        print('Copy mean model from', source_folder, 'to', target_folder)
        os.system(f'mkdir -p {target_folder}/model')
        os.system(f'cp {source_folder}/model/net.pt {target_folder}/model/net_mean.pt')
        os.system(f'cp {source_folder}/model/*scale* {target_folder}/model/')

for model, folder in zip(['CGANRegression'], ['CGANRegression-retrain']):
    for resolution in [48, 64, 96]:
        for operator in ['Operator1', 'Operator2']:
            print(resolution, operator, folder, CONFIGURATION)
            input()
            for realization in range(0,NUM_REALIZATIONS):
                _operator = operator+'-'+str(resolution)
                train_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{CONFIGURATION}/' + _operator + '/*.nc'
                transfer_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{TRANSFER}/' + _operator + '/*.nc'
                model_folder = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/' + _operator + '/' + folder+'-'+str(realization)
                script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/train_model.py'

                if folder in ['MeanVarModel', 'CGANRegression']:
                    copy_mean_model(model_folder, 
                        f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{MODELS_FOLDER}/' + _operator + '/' + 'OLSModel-'+str(realization))

                if fail_function(model_folder):                
                    # basic parameters
                    fit_args = {}
                    model_args = {}
                    hours = {32:1, 48: 10, 64: 10, 96: 20}[resolution] 
                    
                    if folder == 'CGANRegression-retrain':
                        model_args = dict(nx=resolution, regression='None')
                        fit_args = dict(num_epochs=200)
                    elif folder == 'CVAERegression-retrain':
                        model_args = dict(regression='None')
                        fit_args = dict(num_epochs=200)
                        
                    mem=32
                    
                    hpc = DEFAULT_HPC._update(
                        {'ntasks': 1, 'mem': mem, 'gres': 'gpu:mi50:1', 'hours': hours, \
                        'job-name': CONFIGURATION[0]+job_name(model, operator, resolution), 
                        'output': 'training.txt', 'error': 'training.err', 
                        'begin': 'now'})
                    args = dict(model=model, train_path=train_path, transfer_path=transfer_path, model_args=model_args, fit_args=fit_args)
                    run_experiment(model_folder, script_py, hpc, args, delete=False)