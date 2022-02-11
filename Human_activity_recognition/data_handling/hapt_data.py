import zipfile
import os
import logging
import glob
import matplotlib.pyplot as plt
import numpy as np
from data_handling import hapt_Experiments
import random
import gin
import pickle
import data_handling.tfr as tfr


@gin.configurable()
def import_data_from_raw_files(run_paths, window_size=250, window_shift_ratio=0.5, labeling_threshold=0.5,
                               SELECTED_CLASSES=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], tfr_filepath=""):
    files_acc, files_gyro = file_extractions(run_paths)

    # Extract Data
    # Listing entries linked to Experiments:
    if not os.path.exists(run_paths['path_data'] + '/data_after_extraction.txt'):
        logging.info('no extracted data found, extracting data ...')
        experiments_list = data_extraction(run_paths, files_acc, window_size, window_shift_ratio, labeling_threshold)
        # Listing entries linked to user
        logging.info('Saving extracted data by pickle ...')
        with open(run_paths['path_data'] + '/data_after_extraction.txt', 'wb') as f:
            pickle.dump(experiments_list, f)
    else:
        logging.info('data already extracted, restoring data by pickle ...')
        with open(run_paths['path_data'] + '/data_after_extraction.txt', 'rb') as f:
            experiments_list = pickle.load(f)

    user_dataset_input, user_dataset_labels = combine_datasets_of_same_users(experiments_list)

    # divide data:
    # ca: 70/10/20%
    train_dataset = combine_dataset_of_specified_users(user_dataset_input, user_dataset_labels, 1, 21)
    test_dataset = combine_dataset_of_specified_users(user_dataset_input, user_dataset_labels, 22, 27)
    val_dataset = combine_dataset_of_specified_users(user_dataset_input, user_dataset_labels, 28, 30)

    # select/remove activities:
    train_dataset = select_classes_to_keep_in_dataset(train_dataset, SELECTED_CLASSES)
    test_dataset = select_classes_to_keep_in_dataset(test_dataset, SELECTED_CLASSES)
    val_dataset = select_classes_to_keep_in_dataset(val_dataset, SELECTED_CLASSES)

    # balancing
    # random oversampling:
    balanced_train_dataset = oversample_data(train_dataset)
    write_sample_info_to_txt(train_dataset, balanced_train_dataset, "Train_Dataset", tfr_filepath, window_size,
                             window_shift_ratio, labeling_threshold, SELECTED_CLASSES)
    balanced_test_dataset = oversample_data(test_dataset)
    write_sample_info_to_txt(test_dataset, balanced_test_dataset, "Test_Dataset", tfr_filepath, window_size,
                             window_shift_ratio, labeling_threshold, SELECTED_CLASSES)
    balanced_val_dataset = oversample_data(val_dataset)
    write_sample_info_to_txt(val_dataset, balanced_val_dataset, "Validation_Dataset", tfr_filepath, window_size,
                             window_shift_ratio, labeling_threshold, SELECTED_CLASSES)

    # Save Data to TFRecord
    tfr.write_to_TFRecords_file(balanced_train_dataset[0], balanced_train_dataset[1],
                                tfr_filepath + "/train_ds.tfrecord")
    tfr.write_to_TFRecords_file(balanced_test_dataset[0], balanced_test_dataset[1], tfr_filepath + "/test_ds.tfrecord")
    tfr.write_to_TFRecords_file(balanced_val_dataset[0], balanced_val_dataset[1], tfr_filepath + "/val_ds.tfrecord")


# return 2 lists with all filenames
def file_extractions(run_paths):
    datapath = os.path.join(run_paths['home'], 'HAPT_dataset/RawData/')

    datapath = datapath if os.path.exists(datapath) else ''
    if os.path.isdir(datapath):
        logging.info("Data set already extracted...")
    else:
        logging.info("Extracting data set...")
        zip_file = zipfile.ZipFile("/home/Data/HAPT_Data_Set.zip")
        zip_list = zip_file.namelist()
        for f in zip_list:
            zip_file.extract(f, "./data")
        zip_file.close()
    files_acc = sorted(glob.glob(datapath + "acc_exp*.txt"))
    files_gyro = sorted(glob.glob(datapath + "gyro_exp*.txt"))
    return files_acc, files_gyro


# Extract Data from files
def data_extraction(run_paths, files_list, window_size, window_shift_ratio, labeling_threshold):
    experiments_list = []
    folder = os.path.join(run_paths['home'], 'HAPT_dataset/RawData/')
    labels = None
    for elist_counter, f in enumerate(files_list):
        if all(a in f for a in ['exp', 'user', 'acc']):
            # find experiment and user number
            experiment = f.split('exp')[1].split('_user')[0]
            user = f.split('_user')[1].split('.txt')[0]
            logging.info(f'Preprocessing experiment {experiment} ...')

        # Continue by instancing experimentClass and saving Data from each Experiment in windowed datasets
        experiments_list.append(hapt_Experiments.Experiment(experiment, user))
        if elist_counter == 0:
            hapt_Experiments.Experiment.find_labels(folder=folder)
        experiments_list[elist_counter].label_frames = labels
        experiments_list[elist_counter].find_matching_files(folder)
        experiments_list[elist_counter].combine_sensor_values()
        experiments_list[elist_counter].add_labels_to_combined_data()
        experiments_list[elist_counter].slide_window(window_size, window_shift_ratio, labeling_threshold)
        logging.info(f'Experiment {experiment} finish')
    return experiments_list


def combine_datasets_of_same_users(experiments_list):
    # combine all datasets that belong to the same user
    user_dataset_input = []
    user_dataset_label = []
    i = 0
    temporary_exp_list = experiments_list.copy()

    for i1 in range(temporary_exp_list.__len__()):
        if temporary_exp_list[i1].participant is None:
            continue
        user_dataset_input.append(temporary_exp_list[i1].windowed_combined_values)
        user_dataset_label.append(temporary_exp_list[i1].windowed_labels)

        for i2 in range(temporary_exp_list.__len__()):
            if temporary_exp_list[i1] == temporary_exp_list[i2]:
                continue
            if temporary_exp_list[i1].participant == temporary_exp_list[i2].participant:
                user_dataset_input[i].extend(temporary_exp_list[i2].windowed_combined_values)
                user_dataset_label[i].extend(temporary_exp_list[i2].windowed_labels)
                temporary_exp_list[i2].participant = None  # prevents double reading
        i += 1

    return user_dataset_input, user_dataset_label


def combine_dataset_of_specified_users(user_dataset_input, user_dataset_label, user_start, user_end):
    # combine all Data of range(user_start, user_end)
    user_start -= 1
    user_end -= 1

    user_dataset_input = np.array(user_dataset_input, dtype=object)
    user_dataset_label = np.array(user_dataset_label, dtype=object)

    for i in range(user_start, user_end):
        if i == user_start:
            combined_dataset_input = user_dataset_input[i]
            combined_dataset_labels = user_dataset_label[i]
        else:
            combined_dataset_input = np.concatenate((combined_dataset_input, user_dataset_input[i]), axis=0)
            combined_dataset_labels = np.concatenate((combined_dataset_labels, user_dataset_label[i]), axis=0)

    return combined_dataset_input, combined_dataset_labels


def plot_windows(sensorvalues, labels):

    plt.figure(figsize=(25, 5))
    plt.plot(range(sensorvalues), sensorvalues[:, 0:3])
    # plt.title("Accelerometer of experiment num.{} user.{}, State: {}".format())
    plt.show()
    plt.figure(figsize=(25, 5))
    plt.plot(range(sensorvalues), sensorvalues[:, 3:6])
    # plt.title("Accelerometer of experiment num.{} user.{}, State: {}".format())
    plt.show()
    plt.figure(figsize=(25, 5))
    plt.plot(range(labels), labels[:, 0])
    # plt.title("Accelerometer of experiment num.{} user.{}, State: {}".format())
    plt.show()


# data plotting !!!only usable for training dataset
def plot_data(data, exp_num):
    labels_all = np.loadtxt("RawData/labels.txt").astype(int)
    exp_num -= 1
    activity = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING',
                6: 'LAYING', 7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
                11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND'}
    plt.figure(figsize=(25, 5))
    plt.plot(range((data[exp_num].shape[0])), (data[exp_num][:, 0:3]))
    plt.title("Accelerometer of experiment num.{} user.{}, State: {}".format(
        labels_all[exp_num, 0],
        labels_all[exp_num, 1],
        activity[labels_all[exp_num, 2]]))
    plt.show()
    plt.figure(figsize=(25, 5))
    plt.plot(range((data[exp_num].shape[0])), (data[exp_num][:, 3:6]))
    plt.title("Gyroscope of experiment num.{} user.{}, State: {}".format(
        labels_all[exp_num, 0],
        labels_all[exp_num, 1],
        activity[labels_all[exp_num, 2]]))
    plt.show()


def select_classes_to_keep_in_dataset(dataset, selected_classes):
    # selected_classes = [0,1,1,1,1,1,1,1,1,1,1,1,1]  #remove class '0'

    mask = np.ones(len(dataset[1]), dtype=bool)
    mask_list = []
    arr_collumns = []
    for i in range(len(selected_classes)):
        if selected_classes[i] == 0:
            arr_collumns.append(i)

    for i in range(len(dataset[1])):
        if not (dataset[1][i] & selected_classes).any():
            mask_list.append(i)
    
    # remove columns
    labels = np.delete(dataset[1], arr_collumns, 1)
    mask[mask_list] = False
    data = dataset[0][mask]
    labels = labels[mask]
    return data, labels


def oversample_data(dataset):
    # input as tuple[images, labels]
    x, y = dataset
    sum_all = sum(y)
    target_sample_amount = sum_all.max()

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

    return np.array(x_values_out), np.array(y_values_out)


def write_sample_info_to_txt(unbalanced_dataset, balanced_dataset, dataset_type, tfr_filepath, WINDOW_SIZE,
                             WINDOW_SHIFT_RATIO, LABELING_THRESHOLD,
                             SELECTED_CLASSES):
    x1, y1 = unbalanced_dataset
    x2, y2 = balanced_dataset

    try:
        f = open(tfr_filepath + "/info.txt", "x")
        f.write("WINDOW_SIZE = " + str(WINDOW_SIZE) + "\n")
        f.write("WINDOW_SHIFT_RATIO = " + str(WINDOW_SHIFT_RATIO) + "\n")
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

