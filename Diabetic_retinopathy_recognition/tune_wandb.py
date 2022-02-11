from input_pipeline.file_extraction import load_file_names
from input_pipeline.tfrecord import build_dataset_from_tfrecord
import wandb
import gin
import math
from util import utils_params
import logging
import traceback
from choose_mode import Train


def HParameterOptimization_wandb():
    sweep_config = {
        'command': [
            '${env}',
            'python3',
            '${program}',
            '${args}'],
        'program': 'tune_wandb.py',
        'name': 'HP Optimization',
        'method': 'random',
        'metric': {
            'name': 'best_Validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'Trainer.opt': {
                'distribution': 'categorical',
                'values': [1, 2]
            },
            'Trainer.wd': {
                'distribution': 'log_uniform',
                'min': math.log(1e-6),
                'max': math.log(1e-2)
            },
            'Trainer.lr': {
                'distribution': 'log_uniform',
                'min': math.log(1e-4),
                'max': math.log(4e-3)
            },
            'choose_model.dpr': {
                'distribution': 'q_uniform',
                'q': 0.1,
                'min': 0.,
                'max': 0.6
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train_func, count=50)


def train_func(model_id="16"):
    wandb.init()
    gin.clear_config()
    # Hyperparameters
    bindings = []
    for key, value in wandb.config.items():
        bindings.append(f'{key}={value}')

    run_paths = utils_params.gen_run_folder('Tune', None, ','.join(bindings))

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config_Wandb.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    Train(run_paths, model_id)



