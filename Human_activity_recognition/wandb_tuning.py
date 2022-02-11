import wandb
import pprint
import gin
from util import utils_params
import wandb_hyper_optimization as whp
from absl import flags
import sys
import os
from models.model_utils import Model_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tensorflow, info+warning msgs are not printed
FLAGS = flags.FLAGS
flags.DEFINE_string('source_data', 'HAPT', 'Original Dataset-Source HAPT/HAR')
flags.DEFINE_string('model_index', '1', 'Choose your model from the Model index.')
flags.DEFINE_string('Mode', 'Train', 'Specify whether to Train or Evaluate a model.')
flags.DEFINE_string('Sensor_pos', '6', 'Select which sensor will be used.')
positions = {'1': 'chest', '2': 'head', '3': 'shin', '4': 'thigh', '5': 'upperarm', '6': 'waist', '7': 'forearm'}
flags.DEFINE_string('wandb_login', '8284b8969d481810475eec586a359a2accdd1fc6', 'Insert personal Login-Code')

FLAGS(sys.argv)


def config_wandb_HAPT_hp_tuning():
    wandb.login(key=FLAGS.wandb_login)

    config_HAPT_RNN_single_LSTM_layer = {
        'name': 'HP Optimization',
        'method': 'random',
        'metric': {
            'name': 'best_Validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                "values": [0.01, 0.001, 0.0001]
            },
            'dropout_rate': {
                "values": [0.0, 0.1, 0.3, 0.5]
            },
            'lstm_units': {
                'distribution': 'int_uniform',
                'min': 6,
                'max': 15
            },
            'window_size': {
                "values": [250]
            },
            'window_shift_ratio': {
                'distribution': 'q_uniform',
                'min': 0.1,
                'max': 0.5,
                'q': 0.1
            },
            'window_labeling_threshold': {
                'distribution': 'q_uniform',
                'min': 0.7,
                'max': 0.9,
                'q': 0.1
            }
        }
    }
    config_HAPT_RNN_double_LSTM_layer = {
        'name': 'HP Optimization',
        'method': 'random',
        'metric': {
            'name': 'best_Validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                "values": [0.01, 0.001, 0.0001]
            },
            'dropout_rate': {
                "values": [0.0, 0.1, 0.3, 0.5]
            },
            'lstm_units_1': {
                'distribution': 'int_uniform',
                'min': 7,
                'max': 15
            },
            'lstm_units_2': {
                'distribution': 'int_uniform',
                'min': 7,
                'max': 15
            },
            'window_size': {
                "values": [250]
            },
            'window_shift_ratio': {
                'distribution': 'q_uniform',
                'min': 0.1,
                'max': 0.5,
                'q': 0.1
            },
            'window_labeling_threshold': {
                'distribution': 'q_uniform',
                'min': 0.7,
                'max': 0.9,
                'q': 0.1
            }
        }
    }
    if FLAGS.model_index == "1":
        config = config_HAPT_RNN_single_LSTM_layer
    elif FLAGS.model_index == "2":
        config = config_HAPT_RNN_double_LSTM_layer
    pprint.pprint(config)
    sweep_id = wandb.sweep(config, project='HP_HAPT_LSTM')  # NEW Sweep-ID
    wandb.agent(sweep_id, function=train_func, count=50)


def config_wandb_HAR_hp_tuning():
    wandb.login(key=FLAGS.wandb_login)

    config_HAR_RNN_single_LSTM_layer = {
        'name': 'HP Optimization',
        'method': 'random',
        'metric': {
            'name': 'best_Validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                "values": [1, 0.1, 0.01, 0.001]
            },
            'dropout_rate': {
                "values": [0.2, 0.6]
            },
            'lstm_units': {
                'distribution': 'int_uniform',
                'min': 6,
                'max': 20
            },
            'window_size': {
                "values": [300, 500, 1000]
            },
            'window_shift_ratio': {
                'distribution': 'q_uniform',
                'min': 0.1,
                'max': 0.5,
                'q': 0.1
            }
        }
    }
    config_HAR_RNN_double_LSTM_layer = {
        'name': 'HP Optimization',
        'method': 'random',
        'metric': {
            'name': 'best_Validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                "values": [1, 0.1, 0.01, 0.001]
            },
            'dropout_rate': {
                "values": [0.0, 0.1]
            },
            'lstm_units_1': {
                'distribution': 'int_uniform',
                'min': 6,
                'max': 15
            },
            'lstm_units_2': {
                'distribution': 'int_uniform',
                'min': 6,
                'max': 15
            },
            'window_size': {
                "values": [250, 300]
            },
            'window_shift_ratio': {
                'distribution': 'q_uniform',
                'min': 0.2,
                'max': 0.4,
                'q': 0.1
            }
        }
    }
    if FLAGS.model_index == "1":
        config = config_HAR_RNN_single_LSTM_layer
    elif FLAGS.model_index == "2":
        config = config_HAR_RNN_double_LSTM_layer
    pprint.pprint(config)
    sweep_id = wandb.sweep(config, project='HP_HAR_LSTM')  # NEW Sweep-ID
    wandb.agent(sweep_id, function=train_func, count=50)


@gin.configurable()
def train_func(config=None):
    with wandb.init(config=config):

        config = wandb.config
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in config.items():
            bindings.append(f'{key}={value}')
        # gin-config
        if FLAGS.source_data == "HAPT":
            gin.parse_config_files_and_bindings(['configs/config_HAPT_hp_tune.gin'], bindings)
            sensor_pos = '6'
        elif FLAGS.source_data == "HAR":
            gin.parse_config_files_and_bindings(['configs/config_HAR_hp_tune.gin'], bindings)
            sensor_pos = FLAGS.Sensor_pos

        # generate folder structures
        if FLAGS.source_data == "HAPT":
            run_paths = utils_params.gen_run_folder(Mode=FLAGS.Mode,
                                                    path_model_id=Model_index.get(FLAGS.model_index),
                                                    source_data=FLAGS.source_data + '_' + positions[sensor_pos],
                                                    window_size=config.window_size,
                                                    window_shift_ratio=config.window_shift_ratio,
                                                    labeling_threshold=config.window_labeling_threshold
                                                    )
        elif FLAGS.source_data == "HAR":
            run_paths = utils_params.gen_run_folder(Mode=FLAGS.Mode,
                                                    path_model_id=Model_index.get(FLAGS.model_index),
                                                    source_data=FLAGS.source_data + '_' + positions[sensor_pos],
                                                    window_size=config.window_size,
                                                    window_shift_ratio=config.window_shift_ratio
                                                    )
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        whp.start_Hyper(FLAGS=FLAGS, run_paths=run_paths, config=config)


if FLAGS.source_data == "HAPT":
    config_wandb_HAPT_hp_tuning()
elif FLAGS.source_data == "HAR":
    config_wandb_HAR_hp_tuning()
