import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import logging
import os

class ConfusionMatrix(k.metrics.Metric):
    """ self-build confusion matrix """

    def __init__(self, num_classes, Regression=False, name="confusion_matrix", balanced_acc_output=False, **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.Regression = Regression
        self.balanced_acc_output = balanced_acc_output
        self.cm = self.add_weight(name='confusion_matrix',
                                  shape=(self.num_classes, self.num_classes),
                                  initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.cm.assign_add(self.metrics(y_true, y_pred))

    def metrics(self, y_true, y_pred):

        if not self.Regression:
            y_pred = tf.argmax(y_pred, -1)

        else:
            # turn the regression problem to classification problem
            y_pred = tf.clip_by_value(tf.round(y_pred), 0, self.num_classes - 1)
        cm = tf.math.confusion_matrix(tf.squeeze(y_true),
                                      tf.squeeze(y_pred),
                                      num_classes=self.num_classes,
                                      dtype=tf.float32)
        cm = tf.transpose(cm)
        return cm

    def reset_state(self):
        for i in self.variables:
            i.assign(tf.zeros(shape=i.shape, dtype=tf.float32))

    def result(self):
        return self.cm

    def one_against_others(self, class_number):
        tp = self.cm[class_number, class_number]
        fn = tf.reduce_sum(self.cm[:, class_number]) - tp
        fp = tf.reduce_sum(self.cm[class_number, :]) - tp
        tn = tf.reduce_sum(self.cm) - tp - fp - fn
        return tp, fn, fp, tn

    def binary_eval_result(self, class_number=None):
        if self.num_classes == 2:
            tn, fn = self.cm[0]
            fp, tp = self.cm[1]
        else:
            tp, fn, fp, tn = self.one_against_others(class_number)

        sensitivity = tp / (tp + fn + tf.constant(1e-15))
        specificity = tn / (tn + fp + tf.constant(1e-15))
        precision = tp / (fp + tp + tf.constant(1e-15))
        error_rate = (fn + fp) / (tp + fn + fp + tn)
        balanced_error_rate = 0.5 * (2 - sensitivity - specificity)
        f1 = 2 / (1 / precision + 1 / sensitivity + tf.constant(1e-15))
        return sensitivity, specificity, precision, 1 - error_rate, 1 - balanced_error_rate, f1

    def accuracy(self):
        cm_joint_prob = self.cm / tf.reduce_sum(self.cm)
        accuracy = tf.linalg.trace(cm_joint_prob)
        return accuracy

    def balanced_accuracy(self):
        likelihood = self.cm / (tf.reduce_sum(self.cm, axis=0) + tf.constant(1e-15))
        balanced_accuracy = tf.linalg.trace(likelihood) / self.num_classes
        return balanced_accuracy

    def acc(self):
        if self.balanced_acc_output:
            return self.balanced_accuracy()
        else:
            return self.accuracy()

    def summary(self, save_path):

        """ summary of evaluation results """

        results = []
        template_binary = 'sensitivity: {:.2f}%, specificity: {:.2f}%, precision: {:.2f}%, ' \
                          'accuracy: {:.2f}%, balanced_accuracy: {:.2f}%, F1-score:{:.2f}%'
        template_multi = '\nTotal evaluation: \naccuracy: {:.2f}%, balanced accuracy: {:.2f}%'
        logging.info(
            f'Evaluation result for the {self.num_classes}-class classification task'
            if self.num_classes > 1 else 'Regression task')

        # evaluation for binary classification
        if self.num_classes == 2:
            sensitivity, specificity, precision, accuracy, balanced_accuracy, f1 \
                = self.binary_eval_result()
            result = template_binary.format(sensitivity * 100, specificity * 100, precision * 100,
                                            accuracy * 100, balanced_accuracy * 100, f1 * 100)
            logging.info(result)
            results.append(result)

        # evaluation for multiclass classification by results of one-against-others and total
        else:
            for i in range(self.num_classes):
                sensitivity, specificity, precision, accuracy, balanced_accuracy, f1 = \
                    self.binary_eval_result(class_number=i)
                result = (f'\nClass {i} against all:\n ' +
                          template_binary.format(sensitivity * 100, specificity * 100, precision * 100,
                                                 accuracy * 100, balanced_accuracy * 100, f1 * 100))
                logging.info(result)
                results.append(result)
            accuracy = self.accuracy()
            balanced_accuracy = self.balanced_accuracy()
            result = template_multi.format(accuracy * 100, balanced_accuracy * 100)
            logging.info(result)
            results.append(result)

        file = open(os.path.join(save_path, 'evaluation result.txt'), 'w')
        for result in results:
            file.write(result)
        file.close()
        print(self.cm.numpy())

    def plot(self, save_path):
        cm = self.cm.numpy()
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:0.1f}'.format(z) if np.int(z) > 0 else '', ha='center', va='center')
        plt.show()
        plt.savefig(save_path)
