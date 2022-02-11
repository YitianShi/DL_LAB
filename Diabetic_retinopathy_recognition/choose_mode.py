import gin
import wandb
from absl import app, flags
from input_pipeline.file_extraction import load_file_names
from eval.evaluation import evaluate
from input_pipeline.tfrecord import build_dataset_from_tfrecord
from input_pipeline.tfrecord_kaggle import load
from train import Trainer
import logging
import traceback
from tuning import HParameterOptimization
from models.model_utils import choose_model
from ensemble import Voting, Stacking


def Train(run_paths, model_id):
    model, model_info = choose_model(model_id, run_paths=run_paths, Mode='Train')

    train_ds, validation_ds, test_ds = Dataset(run_paths, model_info)

    training_process = Trainer(model, train_ds, validation_ds, test_ds,
                               model_info=model_info, run_paths=run_paths)
    _, test_acc = training_process.train()

    logging.info(f'final test accuracy: {test_acc * 100}%')


@gin.configurable()
def Evaluate(run_paths, model_id):
    model, model_info = choose_model(model_id, run_paths=run_paths, Mode='Evaluate')

    train_ds, validation_ds, test_ds = Dataset(run_paths, model_info)

    evaluate(ds=test_ds, model=model, model_info=model_info, run_paths=run_paths)


def Ensemble(run_paths):
    train_ds, validation_ds, test_ds = Dataset(run_paths)
    train_ds_224, validation_ds_224, test_ds_224 = Dataset(run_paths, offset=224)
    # Stacking(train_ds, validation_ds, test_ds, run_paths)
    Voting((test_ds, test_ds_224), run_paths=run_paths)


def Tune(run_paths, model_id):
    _, model_info = choose_model(model_id, run_paths=run_paths, Mode='Tune')
    train_ds, validation_ds, test_ds = Dataset(run_paths, model_info)
    HParameterOptimization(model_id, train_ds, validation_ds, test_ds,
                           run_paths=run_paths, model_info=model_info)


def Kaggle(run_paths, model_id):
    model, model_info = choose_model(model_id, run_paths=run_paths, Mode='Kaggle')

    path = 'C:/Users/67064/experiments/records/kaggle/'
    train_ds, validation_ds, test_ds = load(path)

    training_process = Trainer(model, train_ds, validation_ds, test_ds,
                               model_info=model_info, run_paths=run_paths, num_classes=5,
                               Profiler=False, is_Kaggle=True)

    _, test_acc = training_process.train()
    logging.info(f'final test accuracy: {test_acc * 100}%')


def Dataset(run_paths, model_info=None, offset=256):
    if model_info is None:
        model_info = {'input_shape': (offset, offset, 3), 'transfer': False}
        offset = str(offset)
    else:
        offset = ''

    train_files, valid_files, test_files, train_labels, valid_labels, test_labels \
        = next(load_file_names(run_paths=run_paths, model_info=model_info))

    train_ds = build_dataset_from_tfrecord(train_files, train_labels, run_paths, model_info, "Train" + offset)
    validation_ds = build_dataset_from_tfrecord(valid_files, valid_labels, run_paths, model_info,
                                                "Validation" + offset, test_set=True)
    test_ds = build_dataset_from_tfrecord(test_files, test_labels, run_paths, model_info, "Test" + offset,
                                          test_set=True)

    return train_ds, validation_ds, test_ds
