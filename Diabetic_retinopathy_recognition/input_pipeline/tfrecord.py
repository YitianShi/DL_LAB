import logging
import os.path
import glob
import gin
import tensorflow as tf
from input_pipeline.preprocess import augment

N_parallel_iterations = 30

feature_description = {
    # define a structure to tell the decoder about the information of each feature
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    # 'fn': tf.io.FixedLenFeature([], tf.string)
}


def Write_TFRecord_File(filenames, labels, path):
    logging.info(f"creating TFRecords {path} now...")
    with tf.io.TFRecordWriter(path) as writer:
        for filename, label in zip(filenames, labels):
            image = open(filename, 'rb').read()  # read image to RAM
            feature = {  # build Feature dictionary
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                # 'fn': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # build Example
            writer.write(example.SerializeToString())  # serialize and write into TFRecord files
    logging.info("Finish.")


def build_dataset_from_tfrecord(files, labels, run_paths, model_info, name, test_set=False):
    input_shape = model_info['input_shape']
    path = os.path.join(run_paths['path_data_tfrecord'], name + ".tfrecords")
    Write_TFRecord_File(files, labels, path)
    dataset = tf.data.TFRecordDataset(path)
    ds = build_dataset(dataset, input_shape, test_set=test_set)
    return ds


@gin.configurable
def build_dataset(dataset, input_shape, batch_size, Graham, test_set=False):
    dataset = dataset.map(lambda x: parse_example(x, input_shape, Graham, test_set),
                          N_parallel_iterations)
    if not test_set:
        dataset = dataset.shuffle(batch_size * 30, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size if not test_set else 103)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def parse_example(example, input_shape, Graham=False, test_set=False):
    feature_dict = tf.io.parse_single_example(example, feature_description)  # decode the serialized TFRecord files
    feature_dict['image'] = tf.cast(tf.io.decode_jpeg(feature_dict['image'], channels=3), tf.float32)  # decode images
    feature_dict['image'] = tf.reshape(feature_dict['image'], input_shape)
    if not test_set:
        feature_dict['image'] = augment(feature_dict['image'], Graham)
    feature_dict['label'] = tf.expand_dims(feature_dict['label'], axis=-1)
    feature_dict['image'] = feature_dict['image'] / 255.
    return feature_dict['image'], feature_dict['label']  # , feature_dict['fn']
