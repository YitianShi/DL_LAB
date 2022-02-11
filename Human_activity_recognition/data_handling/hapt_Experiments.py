import os
import numpy as np


class Experiment:
    def __init__(self, experiment_number, participant):
        # public
        self.experiment_number = experiment_number
        self.participant = participant
        self.windowed_combined_values = []
        self.windowed_labels = []

        # private
        self._acceleration_values = []
        self._gyroscope_values = []
        self._combined_sensor_values = []
        self._labels = []
    label_frames = []  # static Variable

    def find_labels(folder):
        # iterate over files in folder and choose label-file
        # to save all labels in class(static) Array
        for filename in os.listdir(folder):
            if filename.__contains__("labels"):
                f = os.path.join(folder, filename)
                values = np.loadtxt(f, dtype=int)
                Experiment.label_frames = values

    def find_matching_files(self, folder):
        # iterate over files in folder and choose corresponding Acc+Gyr-Files to Experiment
        for filename in os.listdir(folder):
            if filename.__contains__("exp" + self.experiment_number):
                f = os.path.join(folder, filename)
                values = np.loadtxt(f, dtype=float)
                if filename.__contains__("acc"):
                    self._acceleration_values = values
                else:
                    self._gyroscope_values = values

    def combine_sensor_values(self):
        # Array-Structure:
        # 3xAccel, 3xGyro, Label
        # normalize values from each sensor:
        for i in range(self._acceleration_values[0].__len__()):
            self._acceleration_values[:, i] = normalization(self._acceleration_values[:, i])

        for i in range(self._gyroscope_values[0].__len__()):
            self._gyroscope_values[:, i] = normalization(self._gyroscope_values[:, i])

        self._combined_sensor_values = np.concatenate((self._acceleration_values, self._gyroscope_values), axis=1)

    def add_labels_to_combined_data(self):
        # go through label-file-entries and adjust labels of combinedSensorData
        self._labels = [0] * self._combined_sensor_values.__len__()
        for i in range(Experiment.label_frames.__len__()):
            if(Experiment.label_frames[i][0] == int(self.experiment_number)):
                temp_start = Experiment.label_frames[i][3]
                temp_end = Experiment.label_frames[i][4]
                temp_lbl = Experiment.label_frames[i][2]
                for i2 in range(temp_start, temp_end):
                    self._labels[i2] = temp_lbl

    def slide_window(self, window_size, shift_ratio=0.5, labeling_threshold=0.5):
        start = (window_size * shift_ratio)
        # end = (window_size)

        for i in range(int(len(self._combined_sensor_values) / (window_size * shift_ratio))):
            single_window_values = self._combined_sensor_values[int(i * start):int(i * start) + window_size]
            single_window_labels = self._labels[int(i * start):int(i * start) + window_size]
            # only continue with full length windows - drop last (smaller) window:
            if len(single_window_values) == window_size:
                self.windowed_combined_values.append(single_window_values)
                self.windowed_labels.append(assign_one_label_to_window(single_window_labels, labeling_threshold))


# normalization of data
def normalization(data):
    data_afterNormalization = []
    for i in range(len(data)):
        data_afterNormalization.append((data[i] - np.mean(data)) / np.sqrt(np.var(data)))
    return data_afterNormalization
    
    
def assign_one_label_to_window(labels_window, threshold):
    # eg. 60%: threshold = 0.6
    # count appearance of all labels, 0..13:
    global predominant_label
    y = np.bincount(labels_window)
    max_appearance = max(y)
    if max_appearance > threshold * len(labels_window):
        for i in range(len(y)):
            if y[i] == max_appearance:
                predominant_label = i
    else:
        predominant_label = 0

    # look for full-transition-activity inside window and prioritize
    # some are very short compared to continuous actions
    if (len(y) > 7) and (predominant_label < 7):
        max_appearance_transitions = max(y[7:])
        if max_appearance_transitions > (threshold / 2) * len(labels_window):
            # full transition-activity inside window OR big portion of window consists of transition-activity
            if ((labels_window[0] < 7) and (labels_window[-1] < 7)) \
                    or (max_appearance_transitions > (threshold - 0.2) * len(labels_window)):
                for i in range(len(y)):
                    if y[i] == max_appearance_transitions:
                        predominant_label = i

    predominant_label = np.eye(13, dtype=int)[predominant_label]
    return predominant_label
