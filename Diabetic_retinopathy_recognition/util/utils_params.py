import os
import datetime
import gin
import logging
from util.utils_misc import marked_info

@gin.configurable
def gen_run_folder(Mode, previous_ckpt_fn, path_model_id, fold_number=10, KFOLD=False, Graham=False):
    run_paths = dict()

    if not os.path.isdir(path_model_id):

        '''
        generate folder structures for root and data records, which saves preprocessed dataset,
        tfrecords, pretrained weights etc.
        '''

        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                       os.pardir, os.pardir,
                                                       'experiments'))
        path_data_tfrecord = os.path.abspath(os.path.join(path_model_root,
                                                          'records', 'tfrecord'))
        path_model_save = os.path.abspath(os.path.join(path_model_root,
                                                       'records', 'saved_model'))
        path_model_others = os.path.abspath(os.path.join(path_model_root,
                                                         'records', 'others'))
        path_data_preprocess = os.path.abspath(os.path.join(path_model_root,
                                                            'records', 'data_after_preprocessing'))
        date_creation = datetime.datetime.now().strftime('%Y.%m.%d_T_%H-%M-%S')

        # generate folder structures for experiment runs according to the run Mode.

        if Mode in ['Tune', 'Kaggle', 'Train']:
            run_id = 'run_' + date_creation + path_model_id
            run_id += '(G)' if Graham else ''
            if Mode in ['Tune', 'Kaggle']:
                run_id += Mode
            elif KFOLD:
                run_id += f'_{fold_number}-Fold'
        elif Mode == 'Evaluate':
            run_id = previous_ckpt_fn + '_Evaluation'
        elif Mode == 'Ensemble':
            run_id = 'run_' + date_creation + '_Ensemble_Learning'
        else:
            raise ValueError('No such Mode')
        run_paths['path_root'] = path_model_root
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
        run_paths['path_data_tfrecord'] = path_data_tfrecord
        run_paths['path_model_save'] = path_model_save
        run_paths['path_model_others'] = path_model_others
        run_paths['path_data_preprocess'] = path_data_preprocess
    else:
        run_paths['path_model_id'] = path_model_id

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'run.log')

    if Mode in ('Train', 'Kaggle', 'Ensemble', 'Tune'):
        run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
        run_paths['path_model_Tensorboard'] = os.path.join(run_paths['path_model_id'], 'Tensorboard')

    elif Mode == 'Evaluate':
        run_paths['path_eval'] = os.path.join(run_paths['path_model_id'], 'eval')

    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    marked_info('building directories')
    # Create folders
    path_needed = ['path_model', 'path_ckpts', 'path_data'] if Mode in ('Train', 'Kaggle', 'Ensemble', 'Tune') \
        else ['path_eval', 'path_data']
    for k, v in run_paths.items():
        if any([x in k for x in path_needed]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)
                logging.info(f'file path : {v} built')
            else:
                logging.info(f'file path : {v} already exists')

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient
    marked_info('finish building directories')
    return run_paths


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data["/".join([name, k])] = v

    return data
