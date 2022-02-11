import os
import datetime
import gin
import logging


@gin.configurable
def gen_run_folder(Mode, previous_ckpt_fn, path_model_id, source_data, selected_classes, window_size,
                   labeling_threshold, window_shift_ratio):
    run_paths = dict()

    tfr_template = "TFR_" + source_data + "/ws{}_th{:.2f}_sr{:.2f}_sc{}/"
    sc_string = str(selected_classes).replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
    tfr_filepath = tfr_template.format(window_size, labeling_threshold, window_shift_ratio, sc_string)

    if not os.path.isdir(path_model_id):

        if os.path.dirname(__file__).startswith('/'):
            path_home = "/home/data"
        else:
            path_home = "C:/Users/jonasV/dl-lab-21w-team18/Data"

        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                       os.pardir, os.pardir,
                                                       'experiments'))
        path_data_tfrecord = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                          os.pardir, os.pardir,
                                                          'Data',
                                                          tfr_filepath))
        path_model_save = os.path.abspath(os.path.join(path_model_root,
                                                       'records', 'saved_model'))
        path_model_others = os.path.abspath(os.path.join(path_model_root,
                                                         'records', 'others'))
        path_data_preprocess = os.path.abspath(os.path.join(path_model_root,
                                                            'records',
                                                            'data_after_preprocessing'))

        date_creation = datetime.datetime.now().strftime('%Y.%m.%d_T_%H-%M-%S')

        if Mode == 'Train':
            run_id = 'run_' + date_creation
            if path_model_id:
                run_id += '_' + path_model_id
        elif Mode == 'Evaluate':
            run_id = previous_ckpt_fn + '_Evaluation'
        elif Mode == 'Ensemble':
            run_id = 'Ensemble_voting'

        run_paths['home'] = path_home
        run_paths['path_root'] = path_model_root
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
        run_paths['path_data_tfrecord'] = path_data_tfrecord
        run_paths['path_model_save'] = path_model_save
        run_paths['path_model_others'] = path_model_others
        run_paths['path_data_preprocess'] = path_data_preprocess
        run_paths['path_data'] = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              os.pardir, os.pardir,'Data'))

        run_paths['path_data_realworld2016'] = os.path.join(run_paths['path_data'], "realworld2016_dataset_after_preprocessing")
    else:
        run_paths['path_model_id'] = path_model_id

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'run.log')

    if Mode == 'Train':
        run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
        run_paths['path_model_Tensorboard'] = os.path.join(run_paths['path_model_id'], 'Tensorboard')

    elif Mode in ('Evaluate', 'Ensemble'):
        run_paths['path_eval'] = os.path.join(run_paths['path_model_id'], 'eval')

    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    logging.info('################################# building directories #################################')
    # Create folders
    path_needed = ['path_model', 'path_ckpts', 'path_data'] if Mode == 'Train' else ['path_eval', 'path_data']
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
    logging.info('############################## finish building directories ##############################')
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
