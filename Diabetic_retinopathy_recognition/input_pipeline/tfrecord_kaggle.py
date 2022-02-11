import os.path
import glob
import gin
import tensorflow as tf
from input_pipeline.preprocess import augment
import tensorflow_datasets as tfds


def load(file_path=None):
    file_path = '/home/data/tensorflow_datasets/' \
        if not os.path.isdir(file_path) else file_path
    if not os.path.isdir(file_path):
        raise ValueError('No such file path for tfrecords.')
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'diabetic_retinopathy_detection/btgraham-300',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        with_info=True,
        data_dir=file_path)
    ds_train = build_dataset_k(ds_train)
    ds_val = build_dataset_k(ds_val, is_test_set=True)
    ds_test = build_dataset_k(ds_test, is_test_set=True)
    return ds_train, ds_val, ds_test


@gin.configurable
def build_dataset_k(dataset, batch_size, is_test_set=False):
    dataset = dataset.map(lambda x: parse_example(x, is_test_set), tf.data.experimental.AUTOTUNE)
    if not is_test_set:
        dataset = dataset.shuffle(batch_size * 100, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def parse_example(dict, test_set=False):
    dict['image'] = tf.image.resize(tf.cast(dict['image'], tf.float32), (300, 300))
    if not test_set:
        dict['image'] = augment(dict['image'], True)
    dict['image'] = dict['image'] / 255.
    return dict['image'], dict['label']

