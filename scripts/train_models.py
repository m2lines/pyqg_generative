from slurm_helpers import run_experiment, DEFAULT_HPC

for operator in ['Operator1-64', 'Operator2-64']:
    for model in ['OLSModel', 'MeanVarModel', 'CGANRegression']:
        train_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/' + operator + '/*.nc'
        transfer_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/jet/' + operator + '/*.nc'
        model_folder = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/' + operator + '/' + model
        script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/train_model.py'
        hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 32, 'hours': 3, 'job-name': model+'_'+operator, 'output': 'training.txt'})
        args = dict(model=model, train_path=train_path, transfer_path=transfer_path)
        run_experiment(model_folder, script_py, hpc, args)