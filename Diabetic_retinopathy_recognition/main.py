import gin
import wandb
from input_pipeline.k_fold import k_fold_Training
from util import utils_params, utils_misc
from absl import app, flags
import logging
import traceback
from choose_mode import Train, Evaluate, Kaggle, Tune, Ensemble
from tune_wandb import HParameterOptimization_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('model_id', '16', 'Choose your model from the Model index.')
Model_index = {'18': 'ResNet18', '34': 'ResNet34', '50': 'ResNet50', '101': 'ResNet101', '152': 'ResNet152',
               '121': 'DenseNet121', '169': 'DenseNet169', '201': 'DenseNet201', '264': 'DenseNet264',
               'IRV2': 'InceptionResnetV2', 'IV3': 'InceptionV3',
               'B2': 'EfficientNetB2', 'B1': 'EfficientNetB1', 'B0': 'EfficientNetB0',
               'M2': 'MobilenetV2',
               '16': 'vision_transformer_16'}

flags.DEFINE_string('Mode', 'Train', 'Specify mode from: Train, Evaluate, Tune/Wandb, Ensemble or Kaggle.')
KFOLD = True


def main(argv):

    if FLAGS.Mode == 'Wandb':

        # k-fold training process
        HParameterOptimization_wandb()
        return None

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin' if not FLAGS.Mode == 'Kaggle'
                                         else 'configs/config_Kaggle.gin'], [])
    # generate folder structures
    run_paths = utils_params.gen_run_folder(Mode=FLAGS.Mode, path_model_id=Model_index.get(FLAGS.model_id), KFOLD=KFOLD)
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # initialize wandb
    wandb.init(FLAGS.Mode, name=run_paths['path_model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    if FLAGS.Mode == 'Train':
        if KFOLD:
            # k-fold training process
            k_fold_Training(run_paths, FLAGS.model_id)

        else:
            # normal training process
            Train(run_paths, FLAGS.model_id)

    elif FLAGS.Mode == 'Evaluate':
        # Evaluation of model using metrics and deep visualization
        Evaluate(run_paths, FLAGS.model_id)

    elif FLAGS.Mode == 'Ensemble':

        Ensemble(run_paths)

    elif FLAGS.Mode == 'Tune':

        # Tuning with normal tuning method using hparams in Tensorboard
        Tune(run_paths, FLAGS.model_id)

    elif FLAGS.Mode == 'Kaggle':

        # training with Kaggle dataset
        Kaggle(run_paths, FLAGS.model_id)

    else:
        raise ValueError('No Mode called {} chosen, please try again!'.format(FLAGS.Mode))


if __name__ == "__main__":

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S %p')
    try:
        app.run(main)

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
