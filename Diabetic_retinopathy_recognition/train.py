from eval.metrics import ConfusionMatrix
import gin
import wandb
import tensorflow as tf
import tensorflow.keras as k
import datetime
import math
import os
from util.utils_misc import marked_info
import logging


def schedule(lr_max, epoch, N_epoch):
    """ self-build cosine annealing """

    epoch_max_lr = int(0.1 * N_epoch)
    epoch_max_keep = int(0.05 * N_epoch)
    epoch_cos = epoch_max_lr + epoch_max_keep
    lr_start = lr_max * 0.01
    lr_end = lr_max * 0.01

    if epoch <= epoch_max_lr:
        lr = (lr_max - lr_start) / epoch_max_lr * epoch + lr_start
    elif epoch < epoch_cos:
        lr = lr_max
    else:
        epoch = epoch - epoch_cos
        lr = ((1 + math.cos(epoch * math.pi / (N_epoch - epoch_cos))) / 2) * (
                lr_max - lr_end) + lr_end  # cosine annealing
    return lr


class FocalLoss(tf.keras.losses.Loss):
    """ focal loss """

    def __init__(self, alpha: tf.Tensor, gamma=2.0):
        self.gamma = gamma
        self.alpha = alpha / tf.reduce_sum(alpha, -1) * alpha.shape[-1]
        super(FocalLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_true = tf.one_hot(y_true, y_pred.shape[-1])
        epsilon = tf.keras.backend.epsilon()  # 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        y_true = tf.cast(y_true, tf.float32)
        loss = -  y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred) * self.alpha
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss


@gin.configurable
class Trainer(object):
    """ training process """

    @gin.configurable
    def __init__(self, model, train_ds, validation_ds, test_ds, run_paths,
                 num_classes=2, N_epochs=1000,
                 lr=2e-4, wd=0., opt=0,
                 model_info=None, balanced_acc=True,
                 Ckpts=True, Wandb=True, Wandb_Foto=True, Profiler=True, is_Kaggle=False):

        # model for training
        opt_name = 'sgd' if opt == 1 else 'adam'
        self.model = model
        # datasets
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.test_ds = test_ds
        # Parameters
        self.run_paths = run_paths
        self.num_classes = num_classes
        self.N_epochs = N_epochs
        self.lr = lr  # learning rate
        self.wd = wd  # weight decay
        self.opt_name = opt_name  # optimizer
        self.Transfer_learning = model_info['transfer']
        self.Ckpts = Ckpts  # choose whether to open checkpoint manager
        self.Wandb = Wandb  # choose whether to upload data to wandb
        self.Wandb_Foto = Wandb_Foto  # choose whether to upload photos to wandb, it's recommended to be False
        self.Profiler = Profiler  # choose whether to record profiler, highly recommended to be False.
        self.alpha = tf.constant([7, 70, 30, 210, 210], tf.float32)  # class weights for focal loss computation

        self.stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.classification = True if num_classes in [2, 5] else False
        self.opt = k.optimizers.SGD(momentum=0.9) if 'sgd' in self.opt_name else k.optimizers.Adam()
        k.backend.set_value(self.opt.learning_rate, self.lr * 10 if 'sgd' in self.opt_name else self.lr)
        wandb.log({'opt_name': self.opt_name}) if self.Wandb else None

        # define metrics, loss function and optimizer
        # since the self-built confusion matrix already includes calculation of categorical and mean average accuracy,
        # no need to use metrics in keras to evaluate the accuracy
        if self.classification:
            self.lossFun = FocalLoss(self.alpha) if is_Kaggle \
                else k.losses.SparseCategoricalCrossentropy(from_logits=True)
            # train_acc = k.metrics.SparseCategoricalAccuracy(name='train_acc')
            # test_acc = k.metrics.SparseCategoricalAccuracy(name='test_acc')

            self.train_loss = k.metrics.Mean(name='train_loss')
            self.test_loss = k.metrics.Mean(name='test_loss')
            self.validation_loss = k.metrics.Mean(name='validation_loss')

            self.train_cm = ConfusionMatrix(num_classes=num_classes, balanced_acc_output=balanced_acc)
            self.test_cm = ConfusionMatrix(num_classes=num_classes, balanced_acc_output=balanced_acc)
            self.validation_cm = ConfusionMatrix(num_classes=num_classes, balanced_acc_output=balanced_acc)

        else:
            self.lossFun = k.losses.Huber(delta=0.35)
            # train_acc = k.metrics.Accuracy(name='train_acc')
            # test_acc = k.metrics.Accuracy(name='test_acc')

            self.train_loss = k.metrics.Mean(name='train_loss')
            self.test_loss = k.metrics.Mean(name='test_loss')
            self.validation_loss = k.metrics.Mean(name='validation_loss')

            self.train_cm = ConfusionMatrix(num_classes=5, Regression=True, balanced_acc_output=balanced_acc)
            self.test_cm = ConfusionMatrix(num_classes=5, Regression=True, balanced_acc_output=balanced_acc)
            self.validation_cm = ConfusionMatrix(num_classes=5, Regression=True, balanced_acc_output=balanced_acc)

    def l2Loss(self):

        """ l2 loss with weight decay """

        l2_kernel = []
        for i, weight in enumerate(self.model.trainable_variables):
            if 'kernel' in weight.name:
                l2_kernel.append(tf.nn.l2_loss(weight))
        l2_loss = self.wd * tf.add_n(l2_kernel)
        # tf.print(f'{i} regularization terms')
        return l2_loss

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            if 'Efficient' in self.model.name:
                images *= 255.
            predictions = self.model(images, training=True)
            loss = self.lossFun(labels, predictions)
            loss = loss + self.l2Loss()  # \
            # if 'transformer' in self.model.name else loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        # train_acc(labels, predictions)
        self.train_loss(loss)
        self.train_cm.update_state(labels, predictions)

    @tf.function
    def validation_step(self, images, labels):
        if 'Efficient' in self.model.name:
            images *= 255.
        predictions = self.model(images)
        loss = self.lossFun(labels, predictions)
        # test_acc(labels, predictions)
        self.validation_loss(loss)
        self.validation_cm.update_state(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        if 'Efficient' in self.model.name:
            images *= 255.
        predictions = self.model(images)
        loss = self.lossFun(labels, predictions)
        # test_acc(labels, predictions)
        self.test_loss(loss)
        self.test_cm.update_state(labels, predictions)

    def train(self):
        # initialize records and parameters
        logging.info(f'Checkpoint manager: {self.Ckpts}, Tensorboard Profiler: {self.Profiler}, '
                     f'Wandb data upload: {self.Wandb}, Wandb photo upload: {self.Wandb_Foto}.')
        if self.Ckpts:
            # wandb.init(sync_tensorboard=True)
            marked_info('initializing Tensorboard...')
            summary_writer = tf.summary.create_file_writer(self.run_paths['path_model_Tensorboard'])
            tf.summary.trace_on(graph=True, profiler=self.Profiler)
            checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
            manager = tf.train.CheckpointManager(checkpoint, self.run_paths["path_ckpts_train"], max_to_keep=3)
        else:
            manager = None
            marked_info('no Tensorboard initialization')
        train_acc_max = 0
        validation_acc_max = 0
        validation_loss_min = 1
        no_progress_epochs = 0
        train_acc_all = []
        unfreezer = 0
        epoch_schedule = 0
        final_test_acc = tf.constant(0, dtype=tf.float32)
        lr_init = self.lr * 10
        # start training
        for epoch in range(self.N_epochs):
            logging.info('Start of epoch %d' % (epoch + 1))

            # reset metrics
            self.train_loss.reset_states()
            self.train_cm.reset_state()
            self.validation_cm.reset_state()
            self.validation_loss.reset_state()

            # update learning rate according to learning schedule
            if 'sgd' in self.opt_name:
                self.opt.learning_rate = schedule(lr_max=lr_init, epoch=epoch_schedule, N_epoch=140)
                epoch_schedule += 1

            # training step
            for image, labels in self.train_ds:
                self.train_step(image, labels)
                # record the first image of each batch
                wandb.log({'Train data': wandb.Image(image[0])}) if self.Wandb_Foto else None

            train_acc_all.append(self.train_cm.acc().numpy())

            if self.Ckpts:
                with summary_writer.as_default():
                    tf.summary.image("Training data", image, step=epoch)

            # validation step
            for image, labels in self.validation_ds:
                self.validation_step(image, labels)
                # record the first image of each batch
                wandb.log({'validation data': wandb.Image(image[0])}) if self.Wandb_Foto else None

            tf.print(self.train_cm.result())
            tf.print(self.validation_cm.result())

            # record evaluation results
            if self.Ckpts:
                with summary_writer.as_default():
                    tf.summary.scalar('Train loss', self.train_loss.result(), step=epoch)
                    # tf.summary.scalar('Train accuracy', train_acc.result(), step=epoch)
                    tf.summary.scalar('Train accuracy', self.train_cm.acc(), step=epoch)

                    tf.summary.scalar('Validation loss', self.validation_loss.result(), step=epoch)
                    # tf.summary.scalar('Validation accuracy', test_acc.result(), step=epoch)
                    tf.summary.scalar('Validation accuracy', self.validation_cm.acc(), step=epoch)

            wandb.log({'Train loss': self.train_loss.result(),
                       'Train accuracy': self.train_cm.acc(),
                       'Validation loss': self.validation_loss.result(),
                       'Validation accuracy': self.validation_cm.acc(),
                       'epoch': epoch,
                       'learning_rate': self.opt.learning_rate.numpy(),
                       'best_Validation_accuracy': validation_acc_max}) if self.Wandb else None

            template = 'Epoch {}, Train Accuracy: {:4f}, Validation Accuracy: {:4f}, ' \
                       'Train Loss: {:4f}, Validation Loss: {:4f}, ' \
                       'learning rate:{:4f}'

            logging.info(template.format(epoch + 1,
                                         self.train_cm.acc() * 100,
                                         self.validation_cm.acc() * 100,
                                         self.train_loss.result(),
                                         self.validation_loss.result(),
                                         self.opt.learning_rate.numpy()))

            # save the checkpoint if the validation accuracy better than before
            # for the same validation accuracy, the checkpoint will be saved if it has lower loss
            if validation_acc_max < self.validation_cm.acc() \
                    or (validation_acc_max == self.validation_cm.acc()
                        and self.validation_loss.result() < validation_loss_min):
                validation_acc_max = self.validation_cm.acc()
                validation_loss_min = self.validation_loss.result()
                if self.Ckpts:
                    try:
                        marked_info()
                        logging.info("Validation_acc well, saving model weights/checkpoint ...")
                        manager.save()
                        logging.info("model weights/checkpoint saved successfully.")

                    except:
                        logging.info("can't save model weights/checkpoint.")
                    marked_info()
                else:
                    marked_info()
                    logging.info(f"{validation_acc_max * 100}% is the best test accuracy till now")
                    marked_info()

            # After each five epochs the average training accuracy of them will be calculated
            # to determine whether to stop training.
            if len(train_acc_all) > 5:
                train_acc_all.pop(0)
                train_acc_avg = sum(train_acc_all) / len(train_acc_all)

                # if the increase of average train_acc lower than the threshold, it will be recognized as no progress
                if train_acc_max < train_acc_avg:
                    no_progress_epochs = 0
                    train_acc_max = train_acc_avg
                else:
                    no_progress_epochs += 1
                logging.info(f'no progress epoch: {no_progress_epochs}')

                # if the number of epochs reaches 30 or maximum epochs run out,
                # stop training if it's not transfer learning.
                # Otherwise part of the backbone will be unfrozen and continue learning.
                if no_progress_epochs > 15 or epoch is self.N_epochs:
                    if self.Transfer_learning:
                        marked_info()
                        unfreezer += 20
                        logging.info(f"Seems no obvious progress, unfreeze {unfreezer}"
                                     f"% of the frozen layers and reduce the learning rate...")
                        if self.opt_name == 'sgd':
                            lr_init = 0.75 * lr_init
                        else:
                            k.backend.set_value(self.opt.learning_rate, 0.5 * self.lr)

                        untrainable_till = len(self.model.layers) - int(
                            (len(self.model.layers) - 2) * unfreezer * 0.01) - 2

                        for layer in self.model.layers[untrainable_till:]:
                            layer.trainable = True

                        num_trainable = 0
                        for layer in self.model.layers:
                            if layer.trainable:
                                logging.info(f'trainable layer:{layer.name}')
                                num_trainable += 1

                        logging.info(f"{num_trainable} "
                                     f"layers trainable now, continue learning progress.")
                        no_progress_epochs = 0
                        epoch_schedule = 0

                        if unfreezer > 39:
                            self.Transfer_learning = False
                        marked_info()
                    else:
                        marked_info()
                        logging.info("Seems no obvious progress in train_acc, stop training now.")
                        break

        # After the training process the checkpoint with best validation_acc will be restored
        if self.Ckpts:
            logging.info('restoring parameters with best validation_acc...')
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(tf.train.latest_checkpoint(self.run_paths['path_ckpts_train']))
            logging.info("Successfully loaded checkpoint")

            self.test_loss.reset_states()
            self.test_cm.reset_state()

            # test step with the restored model
            for n, (image, labels) in enumerate(self.test_ds):
                self.test_step(image, labels)
            with summary_writer.as_default():
                tf.summary.trace_export(name="model_trace", step=0,
                                        profiler_outdir=self.run_paths['path_model_Tensorboard'])

            # evaluation of the test result
            self.test_cm.summary(self.run_paths['path_ckpts_train'])
            final_test_acc = self.test_cm.acc()
            wandb.log({'Test accuracy': final_test_acc}) if self.Wandb else None

        return validation_acc_max, final_test_acc
