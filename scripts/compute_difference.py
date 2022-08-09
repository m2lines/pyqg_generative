import sys
import os
import glob

from slurm_helpers import run_experiment, DEFAULT_HPC

# Copypaste from comparison_tools.py
def folder_iterator(
    return_blowup=False, return_reference=False,
    base_folder='/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/', 
    Resolution = [32, 48, 64, 96],
    Operator = ['Operator1', 'Operator2'],
    Model = ['OLSModel', 'MeanVarModel', 'CGANRegression', 'CGANRegression-residual', 'CGANRegressionxy-full'],
    Sampling = ['AR1', 'constant'],
    Decorrelation = [0, 12, 24, 36, 48],
    Configuration = ['eddy']
    ):

    for resolution in Resolution:
        for operator in Operator:
            for model in Model:
                _operator = operator+'-'+str(resolution)
                for sampling in Sampling:
                    for decorrelation in Decorrelation: # in hours; 0 means tau=dt
                        if model=='OLSModel' and sampling=='AR1':
                            continue
                        if sampling=='AR1' and decorrelation==0:
                            continue
                        if decorrelation>0 and sampling=='constant':
                            continue
                        for configuration in Configuration:
                            folder = base_folder + _operator + '/' + model
                            subfolder = configuration + '-' + sampling + '-' + str(decorrelation)
                            folder = folder + '/' + subfolder
                            if not os.path.exists(folder):
                                continue
                            nfiles = len(glob.glob(os.path.join(folder, '*.nc')))
                            if not return_blowup:
                                if nfiles != 10:
                                    continue
                            
                            if return_reference:
                                reference = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_256/'+operator+'-'+str(resolution)+'.nc'
                                key = _operator + '/' + model + '/' + subfolder
                                baseline = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_'+str(resolution) + '/*.nc'
                                yield folder, reference, baseline, key
                            else:
                                yield folder

nfile = 0
for model_folder, target_path, _, key in folder_iterator(return_reference=True):
    nfile +=1
    script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
    model_path = os.path.join(model_folder, '*.nc')
    save_file = f'/home/pp2681/pyqg_generative/notebooks/difference/{str(nfile)}.json'
    save_folder = '/home/pp2681/pyqg_generative/notebooks/'

    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 16, 'hours': 1, 
                        'job-name': nfile, 'gres': 'NONE', 
                        'output': f'difference/{str(nfile)}.out',
                        'error' : f'difference/{str(nfile)}.err'})

    args = dict(model_path=model_path, target_path=target_path, key=key, save_file=save_file)
    run_experiment(save_folder, script_py, hpc, args, delete=False)
    print(nfile)

for operator in ['Operator1', 'Operator2']:
    for res in ['32', '48', '64', '96']:
        nfile +=1
        script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
        model_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_'+res+'/*.nc'
        target_path = '/scratch/pp2681/pyqg_generative/Reference-Default-scaled/eddy/reference_256/'+operator+'-'+res+'.nc'
        
        save_file = f'/home/pp2681/pyqg_generative/notebooks/difference/{str(nfile)}.json'
        save_folder = '/home/pp2681/pyqg_generative/notebooks/'
        key = operator+'-'+res+'/Reference'

        hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 16, 'hours': 1, 
                            'job-name': nfile, 'gres': 'NONE'})

        args = dict(model_path=model_path, target_path=target_path, key=key, save_file=save_file)
        run_experiment(save_folder, script_py, hpc, args, delete=False)
        print(nfile)