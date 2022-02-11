import os.path
import numpy as np
import tensorflow as tf
import logging
from tensorboard.plugins.hparams import api as hp
from models.model_utils import choose_model
from train import Trainer
from util.utils_misc import marked_info


def HParameterOptimization(model_id, train_ds, validation_ds, test_ds, run_paths, model_info):
    paths = run_paths['path_model_Tensorboard']
    input_shape = model_info['input_shape']
    HP_lr = hp.HParam('learning_rate', hp.RealInterval(4e-4, 1.6e-3))
    HP_dropout = hp.HParam('dropout', hp.RealInterval(0., 0.5))
    HP_opt = hp.HParam('optimizer', hp.Discrete([0, 1]))

    with tf.summary.create_file_writer(paths).as_default():
        hp.hparams_config(hparams=[HP_opt, HP_lr, HP_dropout],
                          metrics=[hp.Metric('accuracy', display_name='test accuracy')])

    def train(hparams, session_num):
        model, Transfer_learning = choose_model(model_id, input_shape=input_shape,
                                                run_paths=run_paths, Mode='Train', dpr=hparams[HP_dropout])
        model._name = model.name + '_' + str(session_num)
        Train = Trainer(model, train_ds, validation_ds, test_ds, run_paths=run_paths, lr=hparams[HP_lr],
                        opt_name=hparams[HP_opt], Transfer_learning=Transfer_learning, ckpts=False)
        _, test_acc = Train.train()
        return test_acc

    def run(run_dir, hparams, session_num):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)
            test_acc = train(hparams, session_num)
            tf.summary.scalar('test_accuracy', test_acc, step=1)
        return test_acc

    marked_info('Start Tuning')
    logging.info('Following combinations will be trained:')

    for opt in HP_opt.domain.values:
        for dpr in (HP_dropout.domain.min_value, HP_dropout.domain.max_value, 3):
            for lr in np.linspace(HP_lr.domain.min_value, HP_lr.domain.max_value, 4):
                hparams = {HP_lr: lr, HP_opt: opt, HP_dropout: dpr}
                logging.info({h.name: hparams[h] for h in hparams})

    session_num = 0
    for opt in HP_opt.domain.values:
        for dpr in np.linspace(HP_dropout.domain.min_value, HP_dropout.domain.max_value, 3):
            for lr in np.linspace(HP_lr.domain.min_value, HP_lr.domain.max_value, 4):
                hparams = {HP_lr: lr, HP_opt: opt, HP_dropout: dpr}
                run_name = f'run-{session_num + 1}'
                marked_info(f'Start trail {run_name}')
                logging.info({h.name: hparams[h] for h in hparams})
                test_acc = run(os.path.join(paths, run_name), hparams, session_num)
                session_num += 1
                logging.info(f'final test accuracy of this trial: {test_acc * 100}%')
