import logging
import gin
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# write TFRecords:
def write_to_TFRecords_file(inputs, labels, filepath):
    logging.info(f'Creating {filepath} ...')
    labels = labels.astype(int)
    with tf.io.TFRecordWriter(filepath) as tfrecord:
        for idx0 in range(len(inputs)):  # 5782
            label = labels[idx0]
            byteString = []
            for idx1 in range(len(inputs[0])):  # 250
                feature = inputs[idx0][idx1]

                tensor = tf.convert_to_tensor(feature)
                result = tf.io.serialize_tensor(tensor)
                byteString.append((result.numpy()[10:]))

            features = {
                "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=byteString)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            tfrecord.write(example.SerializeToString())
    logging.info('finish')


@gin.configurable
def read_dataset_and_rearrange(dataset_category, tfr_filepath, window_size=100):
    dataset = tf.data.TFRecordDataset(tfr_filepath + "/" + dataset_category + '.tfrecord')

    dataset = dataset.map(lambda x: map_fn(x, windows_size=window_size))
    dataset = dataset.map(lambda x: bytestring_to_sensor_values(x, window_size=window_size))
    return dataset


@gin.configurable
def map_fn(serialized_example, windows_size, N_classes=5):
    # creates sample-arrays of strings
    # out eg.: ((250-str), (12,str))
    features = {
        "feature": tf.io.FixedLenFeature([windows_size], tf.string),
        "label": tf.io.FixedLenFeature([N_classes], tf.int64)
    }
    example = tf.io.parse_single_example(serialized_example, features)
    return example


@tf.function
@gin.configurable
def bytestring_to_sensor_values(parsed_example, window_size=200, N_classes=12, sensor_channels=6):
    # turns sample-arrays into correct datatype and subarray-size
    # out eg.: ((250,6 float), (12,int))
    byte_string = parsed_example["feature"]
    window = tf.io.decode_raw(byte_string, tf.float64)

    sensor_values = tf.reshape(window, [window_size, sensor_channels])
    labels = parsed_example["label"]
    labels = tf.reshape(labels, [N_classes])

    return sensor_values, labels
