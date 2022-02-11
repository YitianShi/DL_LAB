import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import logging
import io
import wandb
import os


class ConfusionMatrix(k.metrics.Metric):
    def __init__(self, num_classes, Regression=False, name="confusion_matrix", save_path=None, **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.Regression = Regression
        self.save_path = save_path
        self.cm = self.add_weight(name='confusion_matrix',
                                  shape=(self.num_classes, self.num_classes),
                                  initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.cm.assign_add(self.metrics(y_true, y_pred))

    def metrics(self, y_true, y_pred):
        if not self.Regression:
            y_pred = tf.argmax(y_pred, -1)
        else:
            y_pred = tf.clip_by_value(tf.round(y_pred), 0, self.num_classes - 1)
        cm = tf.math.confusion_matrix(tf.squeeze(y_true),
                                      tf.squeeze(y_pred),
                                      num_classes=self.num_classes,
                                      dtype=tf.float32)

        # cm = tf.transpose(cm)
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
        acc = 1 - (fn + fp) / (tp + fn + fp + tn)
        balanced_acc = 0.5 * (sensitivity + specificity)
        f1 = 2 / (1 / precision + 1 / sensitivity + tf.constant(1e-15))
        return sensitivity, specificity, precision, acc, balanced_acc, f1

    def acc(self):
        cm_joint_prob = self.cm / tf.reduce_sum(self.cm)
        acc = tf.linalg.trace(cm_joint_prob)
        return acc

    def balanced_acc(self):
        likelihood = self.cm / (tf.reduce_sum(self.cm, axis=1) + tf.constant(1e-15))
        balanced_acc = tf.linalg.trace(likelihood) / self.num_classes
        return balanced_acc

    def summary(self):
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
            accuracy = self.acc()
            balanced_accuracy = self.balanced_acc()
            result = template_multi.format(accuracy * 100, balanced_accuracy * 100)
            logging.info(result)
            results.append(result)

        file = open(os.path.join(self.save_path, 'evaluation result.txt'), 'w')
        for result in results:
            file.write(result)
        file.close()
        print(self.cm.numpy())

    def plot(self, show_plot=False, log_wandb=False, plottitle="val_cm"):
        # plt.ioff() #interactive-mode off: dont display plots before .show()
        cm = self.cm.numpy()
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:0.1f}'.format(z / np.sum(cm[i]) * 100) if np.int(z) > 0 else '', ha='center', va='center')

        if log_wandb:
            wandb.log({plottitle: fig})
        if show_plot:
            plt.savefig(self.save_path + '/evaluation_result.jpg')
        plt.close(fig)
