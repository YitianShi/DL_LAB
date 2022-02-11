import zipfile
import os
from scipy import interpolate
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import random
import gin
import data_handling.tfr as tfr
import logging
import pandas as pd
from data_handling.hapt_Experiments import normalization

activity_labels = {1: 'walking', 2: 'running', 3: 'sitting', 4: 'standing', 5: 'lying', 6: 'climbingup',
                   7: 'climbingdown', 8: 'jumping', }
positions = {'1': 'chest', '2': 'head', '3': 'shin', '4': 'thigh', '5': 'upperarm', '6': 'waist', '7': 'forearm'}


@gin.configurable()
def import_data_from_raw_files(run_paths, window_size=250, window_shift_ratio=0.5, labeling_threshold=0.5,
                               SELECTED_CLASSES=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], tfr_filepath="",
                               sensor_position=6):
    har_files = file_extractions(run_paths)

    # Extract Data
    train_dataset, val_dataset, test_dataset = data_extraction(har_files, sensor_position, window_size,
                                                               window_shift_ratio,
                                                               labeling_threshold)

    # balancing
    # random oversampling:

    balanced_train_dataset = oversample_data(train_dataset, f"Train_Dataset_{positions[sensor_position]}")
    write_sample_info_to_txt(train_dataset, balanced_train_dataset, f"Train_Dataset_{positions[sensor_position]}",
                             tfr_filepath, window_size,
                             window_shift_ratio, labeling_threshold, SELECTED_CLASSES)
    balanced_test_dataset = oversample_data(test_dataset, f"Test_Dataset_{positions[sensor_position]}")
    write_sample_info_to_txt(test_dataset, balanced_test_dataset, f"Test_Dataset_{positions[sensor_position]}",
                             tfr_filepath, window_size,
                             window_shift_ratio, labeling_threshold, SELECTED_CLASSES)
    balanced_val_dataset = oversample_data(val_dataset, f"Validation_Dataset_{positions[sensor_position]}")
    write_sample_info_to_txt(val_dataset, balanced_val_dataset, f"Validation_Dataset_{positions[sensor_position]}",
                             tfr_filepath, window_size,
                             window_shift_ratio, labeling_threshold, SELECTED_CLASSES)

    # Save Data to TFRecord
    tfr.write_to_TFRecords_file(balanced_train_dataset[0], balanced_train_dataset[1],
                                tfr_filepath + "/train_ds.tfrecord")
    tfr.write_to_TFRecords_file(balanced_test_dataset[0], balanced_test_dataset[1], tfr_filepath + "/test_ds.tfrecord")
    tfr.write_to_TFRecords_file(balanced_val_dataset[0], balanced_val_dataset[1], tfr_filepath + "/val_ds.tfrecord")


# return 2 lists with all filenames
def file_extractions(run_paths):
    if not len(glob.glob(run_paths['path_data_realworld2016'] + "/*.csv")) == 0:
        logging.info("HAR Data set already extracted...")
    else:
        HAR_dataset(run_paths['path_data_realworld2016'])
    files = sorted(glob.glob(run_paths['path_data_realworld2016'] + "/*.csv"))
    return files


def HAR_dataset(save_path, path='/home/data', Plot=False):
    data_path = os.path.join(path, 'realworld2016_dataset/proband{}/data')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    subjects = {'train': [1, 2, 5, 8, 11, 12, 13, 15], 'validation': [3], 'test': [9, 10]}
    activities = ['climbingdown', 'climbingup', 'jumping', 'lying',
                  'running', 'sitting', 'standing', 'walking']
    devices = ['Gyroscope', 'acc']

    for dataset in subjects.keys():
        for subject in subjects.get(dataset):
            files = glob.glob(data_path.format(subject) + '/*.csv')
            files.sort()
            for activity in activities:
                time_min = 0
                time_max = float('inf')
                data_dict = {}
                feature_name = []
                path = os.path.join(save_path, dataset + str(subject) + '_' + activity + '.csv')
                if os.path.isfile(path):
                    print(path + ' already exists')
                    continue
                else:
                    pass
                for csv in files:
                    if any(x in csv for x in devices) and activity in csv:
                        index = str(subject) + '_' + os.path.basename(csv).split('.')[0]
                        data = pd.read_csv(csv)
                        position = index.split('_')[-1]
                        device = index.split('_')[1]
                        title = [position + '_' + device + '_' + a for a in data.columns.tolist()[2:]]

                        data = data.to_numpy()[1:, 1:]
                        fun = interpolate.interp1d(data[:, 0], data, axis=0, kind='nearest')
                        data_dict[index] = fun

                        time_min = int(data[0, 0]) if int(data[0, 0]) > time_min else time_min
                        time_max = int(data[-1, 0]) if int(data[-1, 0]) < time_max else time_max
                        feature_name += title
                time_range = np.arange(time_min + 5e3, time_max - 5e3, 20)
                data_total = []

                for index, fun in data_dict.items():
                    data_after_interp = fun(time_range)
                    data_total.append(data_after_interp[:, 1:])
                data_total = np.concatenate(data_total, axis=-1)

                data_total -= np.mean(data_total, axis=0, keepdims=True)
                data_total /= np.std(data_total, axis=0)

                if Plot:
                    plt.figure(figsize=(40, 5))
                    plt.plot(time_range, data_total, label=feature_name)
                    plt.title(str(subject) + '_' + activity)
                    plt.show()

                data_total = pd.DataFrame(data_total, index=time_range, columns=feature_name)
                print('creating ' + path + ' ...')
                data_total.to_csv(path, index=True)
    logging.info('finish')


# Extract Data from files
def data_extraction(files_list, sensor_position, window_size, window_shift_ratio, record_plot=True):
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    for num, f in enumerate(files_list):
        logging.info(f'Preprocessing {f} ...')

        exp_num, activity = f.split('/' if '/' in f else '\\')[-1].split("_")
        activity = activity[:-4]

        for key, value in activity_labels.items():
            if value == activity:
                label = key
                break

        data = pd.read_csv(f)
        if len(data.columns) < 43:
            if all([positions[sensor_position] not in c for c in data.columns]):
                logging.info('No corresponding sensor data in this file.')
                continue

        data = np.array(data[[a for a in data.columns if positions[sensor_position] in a]])
        for i in range(3):
            data[:, [i, i + 3]] = data[:, [i + 3, i]]

        if record_plot:
            logging.info('Plot pictures ... ')
            plot_path = f'Documentation/Pictures/HAR_Rawdata/{positions[sensor_position]}/'
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plt.figure(figsize=(40, 10))
            plt.subplot(211)
            plt.title('Proband:' + str(exp_num) + " Label: " + activity + " Position:" + positions[sensor_position])
            plt.plot(np.arange(data.shape[0]) / 3000, data[:, :3])
            plt.ylabel('Accelerometer')
            plt.legend("xyz")
            plt.subplot(212)
            plt.plot(np.arange(data.shape[0]) / 3000, data[:, 3:])
            plt.ylabel('Gyroscope')
            plt.legend("xyz")
            plt.savefig(plot_path + f'{str(exp_num)}_{activity}.jpg')
            plt.close()

        windowed_data = slide_window(data, window_size, window_shift_ratio)
        window_labels = len(windowed_data) * [label]

        if "train" in f:
            train_data += windowed_data
            train_labels.append(window_labels)
        elif "validation" in f:
            validation_data += windowed_data
            validation_labels.append(window_labels)
        elif "test" in f:
            test_data += windowed_data
            test_labels.append(window_labels)
        logging.info(f'finish')

    train_data = np.array(train_data)
    train_labels = one_hot(np.concatenate(train_labels, 0))
    test_data = np.array(test_data)
    test_labels = one_hot(np.concatenate(test_labels, 0))
    validation_data = np.array(validation_data)
    validation_labels = one_hot(np.concatenate(validation_labels, 0))

    return (train_data, train_labels), (validation_data, validation_labels), (test_data, test_labels)


def slide_window(data, window_size, shift_ratio=0.5):
    windowed_data = []
    start = (window_size * shift_ratio)
    for i in range(int(len(data) / (window_size * shift_ratio))):
        single_window_values = data[int(i * start):int(i * start) + window_size]
        # only continue with full length windows - drop last (smaller) window:
        if len(single_window_values) == window_size:
            windowed_data.append(single_window_values)

    return windowed_data


def oversample_data(dataset, dataset_name):
    logging.info(f'Oversampling {dataset_name} ...')
    # input as tuple[images, labels]
    x, y = dataset
    sum_all = np.sum(y, axis=0)
    target_sample_amount = max(sum_all)

    for i in range(len(sum_all)):
        x_values = np.zeros((sum_all[i], x.shape[1], x.shape[2]))
        y_values = np.zeros((sum_all[i], y.shape[1]))
        l = 0
        for k in range(len(x)):
            if y[k][i] == 1:
                x_values[l] = x[k]
                y_values[l] = y[k]
                l += 1

        added_samples = 0
        if i == 0:
            x_values_out = x_values
            y_values_out = y_values
        else:
            x_values_out = np.concatenate((x_values_out, x_values))
            y_values_out = np.concatenate((y_values_out, y_values))
        while (sum_all[i] + added_samples) < target_sample_amount:
            # -> samples needed
            x_values_out = np.concatenate((x_values_out, np.array(random.sample(list(x_values), 1))))
            y_values_out = np.concatenate((y_values_out, y_values[0:1]))
            added_samples += 1
    logging.info('finish')

    return np.array(x_values_out), np.array(y_values_out)


def one_hot(labels):
    encoder = OneHotEncoder(dtype=int)
    labels = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    return labels


def write_sample_info_to_txt(unbalanced_dataset, balanced_dataset, dataset_type, tfr_filepath, window_size,
                             window_shift_ratio, LABELING_THRESHOLD,
                             SELECTED_CLASSES):
    x1, y1 = unbalanced_dataset
    x2, y2 = balanced_dataset

    try:
        f = open(tfr_filepath + "/info.txt", "x")
        f.write("window_size = " + str(window_size) + "\n")
        f.write("window_shift_ratio = " + str(window_shift_ratio) + "\n")
        f.write("LABELING_THRESHOLD = " + str(LABELING_THRESHOLD) + "\n")
        f.write("SELECTED_CLASSES = " + str(SELECTED_CLASSES) + "\n")
        f.write("\n")
        f.close()
    except:
        pass

    f = open(tfr_filepath + "/info.txt", "a")
    f.write("\n\n--- " + dataset_type + "")
    f.write("\nclass-samples before oversampling:\n")
    f.write(str(sum(y1)))
    f.write("\nclass-samples after oversampling:\n")
    f.write(str(sum(y2)))
    f.close()


def label_stats(labels1, labels2, labels3, title="Statistics of dataset labels"):
    percentages1 = np.array(labels1) / sum(labels1) * 100
    percentages2 = np.array(labels2) / sum(labels2) * 100
    percentages3 = np.array(labels3) / sum(labels3) * 100

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(labels1)).reshape(len(labels1), 1)
    width = 0.3  # the width of the bars

    rects1 = ax.bar(x=(float(x) - width),
                    height=percentages1,
                    width=width,
                    label='train_dataset')
    rects2 = ax.bar(x, percentages2, width, label='val_dataset')
    rects3 = ax.bar(x + width, percentages3, width, label='test_dataset')

    plt.title(title)
    for a, b, c in zip(x, labels1, percentages1):
        ax.text(x=a - width, y=0.2 + c, s='{} ({:.1f}%)'.format(b, c), rotation=90, ha='center')
    for a, b, c in zip(x, labels2, percentages2):
        ax.text(x=a, y=0.2 + c, s='{} ({:.1f}%)'.format(b, c), rotation=90, ha='center')
    for a, b, c in zip(x, labels3, percentages3):
        ax.text(x=a + width, y=0.2 + c, s='{} ({:.1f}%)'.format(b, c), rotation=90, ha='center')

    plt.xticks(x)
    ax.legend()
    plt.ylim(0, max(percentages1) + 5)
    plt.show()
