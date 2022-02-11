import os
import gin
import pandas as pd
from input_pipeline.file_extraction import load_file_names
from input_pipeline.tfrecord import build_dataset_from_tfrecord
from train import Trainer
import logging
from models.model_utils import choose_model
from util.utils_misc import marked_info


@gin.configurable
def k_fold_Training(run_paths, model_id, fold_number=10):

    """ k-fold training process of the module """

    # define a dataframe to record the result for each training fold
    record = pd.DataFrame(columns=['val_acc', 'test_acc'],
                          index=[f'Fold_{n + 1}' for n in range(fold_number)])

    path_ckpts_train = run_paths['path_ckpts_train']
    path_model_Tensorboard = run_paths['path_model_Tensorboard']
    test_ds = None

    _, model_info = choose_model(model_id, run_paths=run_paths, Mode='Train')

    # get data of each fold from data generator and start training procedure
    for n, (train_files, valid_files, test_files, train_labels, valid_labels, test_labels) \
            in enumerate(load_file_names(run_paths=run_paths, model_info=model_info)):

        # define the location to save checkpoints and tensorboard for each training fold
        run_paths['path_ckpts_train'] = path_ckpts_train + f'/fold_{n + 1}'
        run_paths['path_model_Tensorboard'] = path_model_Tensorboard + f'/fold_{n + 1}'
        os.mkdir(run_paths['path_ckpts_train'])
        os.mkdir(run_paths['path_model_Tensorboard'])

        marked_info(f'Fold {n + 1} start training ...')

        # build datasets

        model, model_info = choose_model(model_id, run_paths=run_paths, Mode='Train')
        train_ds = build_dataset_from_tfrecord(train_files, train_labels, run_paths,
                                               model_info, name="Train" + str(n))
        validation_ds = build_dataset_from_tfrecord(valid_files, valid_labels, run_paths,
                                                    model_info, name="Validation" + str(n), test_set=True)
        test_ds = build_dataset_from_tfrecord(test_files, test_labels, run_paths, model_info, name="Test",
                                              test_set=True) if n == 0 else test_ds

        # training process
        Train = Trainer(model, train_ds, validation_ds, test_ds,
                        model_info=model_info, run_paths=run_paths)

        val_acc, test_acc = Train.train()

        # logging results of one fold
        record.loc[f'Fold_{n + 1}', 'val_acc'] = val_acc.numpy()*100
        record.loc[f'Fold_{n + 1}', 'test_acc'] = test_acc.numpy()*100
        record.to_csv(os.path.join(run_paths['path_model_id'], f'{fold_number}_fold_record.csv'))
        # logging.info(f'final test accuracy: {test_acc * 100}%')
        marked_info(f'Fold {n + 1} finish')

    # compute final average results and save the record to csv
    record.loc['avg'] = record.mean(0)
    record.to_csv(os.path.join(run_paths['path_model_id'], f'{fold_number}_fold_record.csv'))
