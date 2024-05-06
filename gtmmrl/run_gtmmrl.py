import subprocess
import json
import os
import time
import itertools
import datetime
from Run_configs import Configuration

# GPU setting
# GPU setting
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# gpus = '1'
gpus = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["CUDA_LAUNCH_BLOCKING"] = '0'

# TPU setting
TPU = False

"""
# 'knowmix':'summary','adm','init,linearize'
+init: replace the graph imbedding with the mean of the textual description token embedddings.
"""

TASK_POOL = {0: 'Pre',
             1: 'Re',
             2: 'Gen',
             4: 'ErrDetect',
             5: 'ReAdm',
             7: 'Death30',
             8: 'Death180',
             9: 'Death365'}

print("**** gpus = " + gpus)



for preset in [
    {'db': 'dx,prx', 'model': 'single', 'architecture': 'lm', 'knowmix': '', 'scratch': True, 'unimodal': 'text',
     'note': 'reproduce'},
]:
    config = {

        'db': preset['db'],
        'model': preset['model'],
        # unimodal : graph / text / ""(multimodal)
        'unimodal': preset['unimodal'],
        # architecture : both / kg / lm / rand
        'architecture': preset['architecture'],
        # label domain : graph / text
        'label_domain': 'text',
        'P': True,
        'A': not preset['scratch'],  # aligh
        # 'R': False if preset['db'] == 'px' else True,
        'R': False,
        'KnowMix': preset['knowmix'],  # layer, init, adm
        'scratch': preset['scratch'],
        'evaluation': False,
        'top_k': 10,
        'note': preset['note'],
        'dropout': 0.1,  # try 0.5?
        'n_negatives': 1,
        'use_tpu': TPU,
        'queue_size': 10400,
        'momentum': 0.995,
        'alpha': 0.4,
        'tmp': 0.07,
        'GTA': True,
        'GTM': True,
        'RC': True,
        'MLP': True,
        'MLM': True,
        'MoD': True,
        'MoDDownstream': True,
        'hard': True,
        'pre_epochs': 200
    }

    for _task in [4]:  # 0, 1, 2, 4, 5, 7
        if (_task == 3) and (preset['db'] == 'px'):
            continue
        for _SEED in [1234]:  # , 123, 12, 1, 42]: # , 1, 42]:
            current_time = datetime.datetime.now()
            print('')
            print('')
            print("================== subprocess task " + TASK_POOL[_task] + " start time: " + str(
                current_time) + ' --------------------')

            if (_task == 0) and (_SEED != 1234):
                continue

            config['task_number'] = _task
            config['seed'] = _SEED

            # Training configs
            if _task == 0:
                config['train_bsize'] = 16 if preset['db'] == 'px' else 32
                config['eval_bsize'] = 4 if preset['db'] == 'px' else 8
                config['lr'] = 5e-4
                config['num_epochs'] = config['pre_epochs']
            elif _task == 2:
                config['train_bsize'] = 16 if preset['db'] == 'px' else 32
                config['eval_bsize'] = 4 if preset['db'] == 'px' else 8
                config['lr'] = 6e-5
                config['num_epochs'] = 300
            elif _task in [1, 3]:
                config['train_bsize'] = 16 if preset['db'] == 'px' else 32
                config['eval_bsize'] = 4 if preset['db'] == 'px' else 8
                config['lr'] = 5e-5
                config['num_epochs'] = 200
            elif _task in [4]:
                config['train_bsize'] = 32 if preset['db'] == 'px' else 32
                config['eval_bsize'] = 8 if preset['db'] == 'px' else 8
                config['lr'] = 2e-4
                config['num_epochs'] = 200
            elif _task in [5, 7]:
                config['train_bsize'] = 16 if preset['db'] == 'px' else 32
                config['eval_bsize'] = 4 if preset['db'] == 'px' else 16
                config['lr'] = 1e-5
                config['num_epochs'] = 3
            elif _task in [6, 8]:
                config['train_bsize'] = 16 if preset['db'] == 'px' else 32
                config['eval_bsize'] = 4 if preset['db'] == 'px' else 16
                config['lr'] = 1e-5
                config['num_epochs'] = 3

            # Run script
            exp_config = Configuration(config)
            SRC_PATH, TRAINING_CONFIG_LIST = exp_config.get_configuration()

            # Sanity check
            RUN_FLAG, error_log = exp_config.assertion()
            if not RUN_FLAG:
                print(error_log)
                continue

            # Bash run
            subprocess.run(['/home/caoyu/anaconda3/envs/torch_kg_txt/bin/python', SRC_PATH] + TRAINING_CONFIG_LIST)

            # subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
            # 此处，如果直接调用'python',SRC_PATH 则调用的是默认的python环境，而不是torch_kg_txt，
            current_time = datetime.datetime.now()
            print("================== subprocess task " + TASK_POOL[_task] + " end time: " + str(
                current_time) + ' --------------------')
            print('')
            print('')
