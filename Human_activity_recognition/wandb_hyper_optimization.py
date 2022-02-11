import os.path
import gin
from data_handling import input_pipeline, hapt_data
import data_handling.tfr as tfr
import train
from models.model_utils import Model_index, chooseModel
import logging
import wandb
import numpy as np


@gin.configurable
def train_model_and_find_Hyperparameters(run_paths, model, train_ds, val_ds, test_ds, num_classes=12):
    train_cm, val_cm = train.training(model=model,
                                      train_ds=train_ds,
                                      val_ds=val_ds,
                                      test_ds=test_ds,
                                      run_paths=run_paths,
                                      num_classes=num_classes
                                      )


@gin.configurable()
def start_Hyper(FLAGS, run_paths, num_classes=12, sensor_channels=6, batch_size=20,
                buffer_size=200, config=None):
    tfr_filepath = run_paths["path_data_tfrecord"]

    if not os.path.exists(tfr_filepath + "/test_ds.tfrecord"):
        print("creating new TFR-Files")
        # os.mkdir(tfr_filepath)
        hapt_data.import_data_from_raw_files(run_paths=run_paths,
                                             window_size=config.window_size,
                                             labeling_threshold=config.window_labeling_threshold,
                                             window_shift_ratio=config.window_shift_ratio,
                                             tfr_filepath=tfr_filepath)
    else:
        print("reading from existing TFR-Files")

    train_dataset = tfr.read_dataset_and_rearrange("train_ds", tfr_filepath, window_size=config.window_size)
    val_dataset = tfr.read_dataset_and_rearrange("val_ds", tfr_filepath, window_size=config.window_size)
    test_dataset = tfr.read_dataset_and_rearrange("test_ds", tfr_filepath, window_size=config.window_size)

    # final preparation of datasets:
    train_ds = input_pipeline.shuffle_batch_prefetch_dataset(train_dataset, batch_size=batch_size,
                                                             buffer_size=buffer_size)
    val_ds = input_pipeline.shuffle_batch_prefetch_dataset(val_dataset, batch_size=batch_size, buffer_size=buffer_size)
    test_ds = input_pipeline.shuffle_batch_prefetch_dataset(test_dataset, batch_size=batch_size,
                                                            buffer_size=buffer_size)

    # read Dataset-Distribution and log to wandb-workspace
    f = open(tfr_filepath + "/info.txt", "r")
    dataset_distribution = wandb.Table(data=np.array(f.readlines()).reshape(24, 1), columns=["info"])
    wandb.log({"Class-Distribution": dataset_distribution}, step=0)
    f.close()

    model_name = Model_index[FLAGS.model_index]  # "RNN","Stacked_RNN"
    input_shape = (config.window_size, sensor_channels)
    try:
        units = (config.lstm_units_1, config.lstm_units_2)
    except:
        units = config.lstm_units

    model = chooseModel(model_type=FLAGS.model_index, input_shape=input_shape, units=units,
                        num_classes=num_classes, batch_size=batch_size, run_paths=run_paths,
                        dropout_rate=config.dropout_rate,
                        ckpt_fn=(model_name + "_" + str(units)))

    train_model_and_find_Hyperparameters(run_paths=run_paths, model=model, train_ds=train_ds, val_ds=val_ds,
                                         test_ds=test_ds)


logging.basicConfig(filename='log.txt', level=logging.DEBUG, filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p')
