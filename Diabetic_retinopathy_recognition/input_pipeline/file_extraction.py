import logging
import zipfile
import os
from sklearn.model_selection import StratifiedKFold
import gin
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from input_pipeline.preprocess import preprocess_pictures
from util.utils_misc import marked_info


@gin.configurable
def load_file_names(num_sample, model_info, Graham, run_paths, num_classes,
                    fold_number=10, Mode='Train', ckpt_fn=None):
    """
    Iterable data generator for k-fold,
    which can also be used for normal training process by using next() function
    """

    # Check if data has been extracted, if not then extract it
    num_per_sample = num_sample // num_classes
    marked_info('start building datasets')
    data_path = '/home/data' if os.path.isdir('/home/data') \
        else os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')
    if os.path.isdir(data_path + '/IDRID_dataset'):
        logging.info("Data set already extracted...")
    else:
        logging.info("Extracting data set...")
        try:
            zip_file = zipfile.ZipFile(data_path + "/idrid.zip")
        except:
            raise ValueError('no dataset found')
        zip_list = zip_file.namelist()
        for f in zip_list:
            zip_file.extract(f, "dataset")
        zip_file.close()

    if Mode == "Evaluate":
        Graham = True if '(G)' in ckpt_fn else False

    input_shape = model_info['input_shape']

    # preprocess pictures and save them to specific location records
    file_position = preprocess_pictures(Graham, run_paths=run_paths, input_shape=input_shape, data_path=data_path)

    # get dataset information
    files = pd.read_csv(data_path + "/IDRID_dataset/labels/train.csv")
    files.dropna(inplace=True, axis='columns')
    files_with_labels = files[['Image name', 'Retinopathy grade']].drop_duplicates()
    files_with_labels['Image name'] = files_with_labels['Image name'].map(
        lambda x: file_position + "/train/" + x + ".jpg")

    # data transition if 2-class-classification
    if num_classes == 2:
        files_with_labels['Retinopathy grade'] = files_with_labels['Retinopathy grade'].apply(data_exchange)

    # build a list of k-fold indexes of each fold. The data is stratified according to the label of each class.
    raw_files = files_with_labels.to_numpy()
    labels = (raw_files[:, 1]).astype(int)
    kf = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=2021)
    KFold_List = [split for split in kf.split(raw_files, labels)]

    # generate dataset according to the index of each fold
    n = 0
    while n < fold_number:

        train_ds_raw = raw_files[KFold_List[n][0]]
        train_labels_raw = train_ds_raw[:, 1].astype(int)
        valid_ds = raw_files[KFold_List[n][1]]
        valid_labels = valid_ds[:, 1].astype(int)
        valid_files = valid_ds[:, 0]

        # resample the train dataset
        train_ds_raw = pd.DataFrame(data=train_ds_raw, columns=['files', 'labels'])

        train_ds_raw = train_ds_raw.groupby(['labels'])
        train_ds = train_ds_raw.apply(lambda x: x.sample(num_per_sample, replace=True)).reset_index(drop=True)
        train_ds = np.array(train_ds.sample(frac=1))

        train_labels = train_ds[:, 1].astype(np.int32)
        train_files = train_ds[:, 0]

        # plot the data statics
        plot_statistics_labels(labels, f"raw data")
        plot_statistics_labels(train_labels_raw, f"training data of Fold_{n + 1}")
        plot_statistics_labels(train_labels, f"training data after resampling of Fold_{n + 1}")
        plot_statistics_labels(valid_labels, f"validation data of Fold_{n + 1}")

        # test set only generated in the first loop
        if n == 0:
            test_files = glob.glob(file_position + "/test/*.jpg")
            test_files.sort()
            test_labels = pd.read_csv(data_path + "/IDRID_dataset/labels/test.csv").to_numpy()[:, 1].astype(np.int32)
            test_labels = list(map(data_exchange, test_labels)) if num_classes == 2 else test_labels
            plot_statistics_labels(test_labels, f"test data")
        else:
            test_files, test_labels = None, None

        marked_info('finish building datasets')
        n += 1

        # print(train_files, valid_files)
        yield train_files, valid_files, test_files, train_labels, valid_labels, test_labels


def plot_statistics_labels(label, name):
    dict = {}
    for key in label:
        dict[key] = dict.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(dict.keys())
    y = list(dict.values())
    plt.bar(x, y)
    plt.title('Statistics of {} labels'.format(name))
    for a, b in zip(x, y):
        ax.text(a, b, '{}({:.2f}%)'.format(b, (100 * b / len(label)), 2), ha='center', va='bottom')
    plt.show()


def data_exchange(label):
    label = 0 if label < 1.5 else 1
    return label
