import numpy as np
import logging
import tensorflow as tf
import gin
from data_handling import tfr, hapt_data, har_data
import wandb

import os
import matplotlib.pyplot as plt

activity_HAR = {1: 'WALKING', 2: 'RUNNING', 3: 'SITTING', 4: 'STANDING', 5: 'LYING', 6: 'CLIMBING_UP',
                7: 'CLIMBING_DOWN', 8: 'JUMPING', }
activity_HAPT = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING',
                 6: 'LAYING', 7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
                 11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND'}


def get_datasets(FLAGS, run_paths, sensor_pos):
    if not os.path.exists(os.path.join(run_paths["path_data_tfrecord"], "test_ds.tfrecord")):
        if FLAGS.source_data == "HAPT":
            logging.info("creating new TFR-Files")
            hapt_data.import_data_from_raw_files(run_paths, tfr_filepath=run_paths["path_data_tfrecord"])
        elif FLAGS.source_data == "HAR":
            logging.info("creating new TFR-Files")
            har_data.import_data_from_raw_files(run_paths, tfr_filepath=run_paths["path_data_tfrecord"],
                                                sensor_position=sensor_pos)
    else:
        logging.info("Corresponding tfrecord files exist, reading from existing TFR-Files")

    # READ from TFRecord-file:
    train_dataset = tfr.read_dataset_and_rearrange("train_ds", run_paths["path_data_tfrecord"])
    val_dataset = tfr.read_dataset_and_rearrange("val_ds", run_paths["path_data_tfrecord"])
    test_dataset = tfr.read_dataset_and_rearrange("test_ds", run_paths["path_data_tfrecord"])

    plot_random_window(run_paths, train_dataset, save_path=run_paths['path_data_tfrecord'])

    # final preparation of datasets:
    train_ds = shuffle_batch_prefetch_dataset(train_dataset)
    val_ds = shuffle_batch_prefetch_dataset(val_dataset)
    test_ds = shuffle_batch_prefetch_dataset(test_dataset)

    return train_ds, val_ds, test_ds


@gin.configurable
def shuffle_batch_prefetch_dataset(dataset, batch_size=100, buffer_size=100):
    dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    return dataset


def plot_random_window(run_paths, dataset, save_path, specific_class=None):
    # plot single window

    # get labels out of tf.map.dataset:
    data, labels = tuple(zip(*dataset.shuffle(10000)))
    r = np.random.randint(0, len(labels))
    if specific_class is not None:
        one_hot_target = np.eye(len(labels[0]))[specific_class]
        while not np.array(labels[r] == one_hot_target).all():
            r += 1
            if r >= len(labels):
                print("reset")
                r = 0

    label = reverse_one_hot_coding(labels[r]).numpy() + 1
    activity = activity_HAR[label] if 'HAR' in run_paths["path_data_tfrecord"] else activity_HAPT[label]

    acc, gyr = np.array_split(data[r], 2, axis=1)

    plt.subplot(211)
    plt.title("Label: " + activity + " random_sample:" + str(r))
    plt.plot(range(len(data[r])), acc)
    plt.legend("xyz")
    plt.ylabel("accelerometer")
    plt.subplot(212)
    plt.plot(range(len(data[r])), gyr)
    plt.legend("xyz")
    plt.ylabel("gyroscope")
    plt.show()
    plt.savefig(save_path + '/random_window.jpg')
    plt.close()


def label_stats(labels1, labels2, labels3, title="Statistics of dataset labels"):
    percentages1 = np.array(labels1)/sum(labels1)*100
    percentages2 = np.array(labels2) / sum(labels2)*100
    percentages3 = np.array(labels3) / sum(labels3)*100

    fig, ax = plt.subplots(figsize=(16,9))
    x=np.arange(len(labels1))
    width = 0.3  # the width of the bars

    rects1 = ax.bar(x - width, percentages1, width, label='train_dataset')
    rects2 = ax.bar(x, percentages2, width, label='val_dataset')
    rects3 = ax.bar(x + width, percentages3, width, label='test_dataset')


    plt.title(title)
    for a, b, c in zip(x, labels1, percentages1):
        ax.text(x=a - width, y=0.2+c, s='{} ({:.1f}%)'.format(b, c), rotation=90, ha='center')
    for a, b, c in zip(x, labels2, percentages2):
        ax.text(x=a, y=0.2+c, s='{} ({:.1f}%)'.format(b, c), rotation=90, ha='center')
    for a, b, c in zip(x, labels3, percentages3):
        ax.text(x=a + width, y=0.2+c, s='{} ({:.1f}%)'.format(b, c), rotation=90, ha='center')

    plt.xticks(x)
    ax.legend()
    plt.ylim(0,max(percentages1)+5)
    plt.show()


def reverse_one_hot_coding(tensor):
    ndim = tf.rank(tensor)
    tensor = tf.argmax(tensor, axis=ndim - 1)
    return tensor
