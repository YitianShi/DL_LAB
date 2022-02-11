import os
import gin
import train
from util import utils_params, utils_misc
from absl import flags
import sys
from models.model_utils import chooseModel, Model_index
import wandb
import logging
from eval.evaluation import evaluate, soft_voting
from data_handling import input_pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tensorflow, info+warning msgs are not printed
FLAGS = flags.FLAGS
flags.DEFINE_string('source_data', 'HAR', 'Original Dataset-Source HAPT/HAR')
flags.DEFINE_string('model_index', '2', 'Choose your model from the Model index.')
flags.DEFINE_string('Mode', 'Train', 'Specify whether to "Train", "Evaluate" or "Ensemble" a model(s).')
flags.DEFINE_string('Sensor_pos', '6', 'Select which sensor will be used.')
positions = {'1': 'chest', '2': 'head', '3': 'shin', '4': 'thigh', '5': 'upperarm', '6': 'waist', '7': 'forearm'}
FLAGS(sys.argv)


def main(argv):
    if FLAGS.source_data == "HAPT":
        gin.parse_config_files_and_bindings(['configs/config_HAPT.gin'], [])
        sensor_pos = '6'
    elif FLAGS.source_data == "HAR":
        gin.parse_config_files_and_bindings(['configs/config_HAR.gin'], [])
        sensor_pos = FLAGS.Sensor_pos
    else:
        raise ValueError()

    # generate folder structures
    run_paths = utils_params.gen_run_folder(Mode=FLAGS.Mode, path_model_id=Model_index.get(FLAGS.model_index),
                                            source_data=FLAGS.source_data + '_' + positions[sensor_pos])
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    logging.info(positions[sensor_pos])
    wandb.init(FLAGS.Mode, name=run_paths['path_model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    train_ds, val_ds, test_ds = input_pipeline.get_datasets(FLAGS, run_paths, sensor_pos)

    model = chooseModel(model_type=FLAGS.model_index, run_paths=run_paths, Mode=FLAGS.Mode) if FLAGS.Mode in (
        'Train', 'Evaluate') else None

    if FLAGS.Mode == 'Train':
        train.training(model=model,
                       train_ds=train_ds,
                       val_ds=val_ds,
                       test_ds=test_ds,
                       run_paths=run_paths)

    elif FLAGS.Mode == 'Evaluate':
        evaluate(model, test_ds, run_paths)

    elif FLAGS.Mode == "Ensemble":
        models = [chooseModel(model_type="2", run_paths=run_paths,
                              units=(11, 11),
                              dropout_rate=0.1,
                              ckpt_fn='HAPT_ws250_sr0.5_th0.9//HAPT_Stacked_RNN_11_11_dpr0.1', Mode=FLAGS.Mode),
                  chooseModel(model_type="2", run_paths=run_paths,
                              units=(8, 11),
                              dropout_rate=0.5,
                              ckpt_fn='HAPT_ws250_sr0.5_th0.9//HAPT_Stacked_RNN_8_11_dpr0.5', Mode=FLAGS.Mode),
                  chooseModel(model_type="2", run_paths=run_paths,
                              units=(15, 12),
                              dropout_rate=0.3,
                              ckpt_fn='HAPT_ws250_sr0.5_th0.9//HAPT_Stacked_RNN_15_12_dpr0.3', Mode=FLAGS.Mode),
                  chooseModel(model_type="1", run_paths=run_paths,
                              units=13,
                              dropout_rate=0.4,
                              ckpt_fn='HAPT_ws250_sr0.5_th0.9//HAPT_RNN_13_dpr0.4', Mode=FLAGS.Mode)]

        soft_voting(models, train_ds, run_paths, 12)

        soft_voting(models, test_ds, run_paths, 12)
        soft_voting(models, val_ds, run_paths, 12)

    else:
        raise KeyError(f'No model calls {FLAGS.Mode}')


main(sys.argv)
