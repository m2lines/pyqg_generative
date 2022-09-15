import sys
import os
import glob

from slurm_helpers import run_experiment, DEFAULT_HPC

# Copypaste from comparison_tools.py
def folder_iterator(
    return_blowup=False,
    base_folder='/scratch/pp2681/pyqg_generative/Reference-Default-scaled/models/', 
    Resolution = [32, 48, 64, 96],
    Operator = ['Operator1', 'Operator2'],
    Model = ['OLSModel', 'MeanVarModel', 'CGANRegression', 'CGANRegression-recompute', 'CGANRegression-None-recompute', 'CGANRegression-Unet', 'OLSModel-div', 'CGANRegression-div', 'CVAERegression-None'],
    Sampling = ['AR1', 'constant'],
    Decorrelation = [0, 12, 24, 36, 48],
    Configuration = ['eddy', 'eddy-3600', 'jet-3600', 'jet', 'eddy-recompute'],
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
                            
                            conf = 'eddy' if configuration.find('eddy') > -1 else 'jet'
                            reference = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{conf}/reference_256/'+operator+'-'+str(resolution)+'.nc'
                            key = _operator + '/' + model + '/' + subfolder
                            yield folder, reference, key

os.system('rm /home/pp2681/pyqg_generative/notebooks/difference/*.sh')
os.system('rm /home/pp2681/pyqg_generative/notebooks/difference/output/*')

nfile = 0
for model_folder, target_path, key in folder_iterator():
    nfile +=1
    script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
    model_path = os.path.join(model_folder, '*.nc')
    save_file = f'output/{str(nfile)}.json'
    save_folder = '/home/pp2681/pyqg_generative/notebooks/difference'

    hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
                        'hours': 0, 'minutes': 4,
                        'job-name': nfile, 'gres': 'NONE', 
                        'output': f'output/{str(nfile)}.out',
                        'error' : f'output/{str(nfile)}.err',
                        'launcher': f'{str(nfile)}.sh',
                        'partition': 'cs'})

    args = dict(model_path=model_path, target_path=target_path, key=key, save_file=save_file)
    run_experiment(save_folder, script_py, hpc, args, delete=False)
    print(nfile)

for operator in ['Operator1', 'Operator2']:
    for res in ['32', '48', '64', '96']:
        for timestep in ['', '_3600']:
            for conf in ['eddy', 'jet']:
                nfile +=1
                script_py = '/home/pp2681/pyqg_generative/pyqg_generative/tools/comparison_tools.py'
                model_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{conf}/reference{timestep}_'+res+'/*.nc'
                target_path = f'/scratch/pp2681/pyqg_generative/Reference-Default-scaled/{conf}/reference_256/'+operator+'-'+res+'.nc'
                
                save_file = f'output/{str(nfile)}.json'
                save_folder = '/home/pp2681/pyqg_generative/notebooks/difference'
                postfix = '' if timestep=='' else f'-{timestep[1:]}'
                key = operator+'-'+res+f'/Reference/{conf}'+postfix

                hpc = DEFAULT_HPC._update({'ntasks': 1, 'mem': 8, 
                                'hours': 0, 'minutes': 4,
                                'job-name': nfile, 'gres': 'NONE', 
                                'output': f'output/{str(nfile)}.out',
                                'error' : f'output/{str(nfile)}.err',
                                'launcher': f'{str(nfile)}.sh',
                                'partition': 'cs'})

                args = dict(model_path=model_path, target_path=target_path, key=key, save_file=save_file)
                run_experiment(save_folder, script_py, hpc, args, delete=False)
                print(nfile)