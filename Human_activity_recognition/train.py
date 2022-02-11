from eval.metrics import ConfusionMatrix
import gin
import tensorflow as tf
import tensorflow.keras as k
import datetime
import logging
import wandb


def reverse_one_hot_coding(tensor):
    ndim = tf.rank(tensor)
    tensor = tf.argmax(tensor, axis=ndim - 1)
    return tensor


class GCAdam(tf.keras.optimizers.Adam):
    def get_gradients(self):
        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        super(FocalLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()  # 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        y_true = tf.cast(y_true, tf.float32)
        loss = -  y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss


@gin.configurable
def training(model, train_ds, val_ds, test_ds, run_paths, num_classes, N_epochs, l_rate):
    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    opt = k.optimizers.Adam(learning_rate=l_rate)

    lossFun = k.losses.CategoricalCrossentropy()
    train_acc = k.metrics.CategoricalAccuracy(name='train_acc')
    val_acc = k.metrics.CategoricalAccuracy(name='val_acc')
    test_acc = k.metrics.CategoricalAccuracy(name='test_acc')

    train_loss = k.metrics.Mean(name='train_loss')
    val_loss = k.metrics.Mean(name='val_loss')
    test_loss = k.metrics.Mean(name='test_loss')

    train_cm = ConfusionMatrix(num_classes=num_classes, save_path=run_paths['path_model_id'])
    val_cm = ConfusionMatrix(num_classes=num_classes, save_path=run_paths['path_model_id'])
    test_cm = ConfusionMatrix(num_classes=num_classes, save_path=run_paths['path_model_id'])

    @tf.function
    def train_step(data, labels, opt):
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            loss = lossFun(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc(labels, predictions)
        train_loss(loss)
        labels_one_hot_decoded = reverse_one_hot_coding(labels)
        train_cm(labels_one_hot_decoded, predictions)

    @tf.function
    def val_step(data, labels):
        predictions = model(data)
        loss = lossFun(labels, predictions)
        val_acc(labels, predictions)
        val_loss(loss)

        labels_one_hot_decoded = reverse_one_hot_coding(labels)
        val_cm(labels_one_hot_decoded, predictions)

    @tf.function
    def test_step(data, labels):
        predictions = model(data)
        loss = lossFun(labels, predictions)
        test_acc(labels, predictions)
        test_loss(loss)

        labels_one_hot_decoded = reverse_one_hot_coding(labels)
        test_cm.update_state(labels_one_hot_decoded, predictions)

    ckpt = tf.train.Checkpoint(optimizer=opt, model=model)
    manager = tf.train.CheckpointManager(ckpt, run_paths["path_ckpts_train"], max_to_keep=3)

    train_acc_max = 0
    val_acc_max = 0
    val_loss_min = 1
    no_progress_epochs = 0
    train_acc_all = []
    val_acc_all = []

    for epoch in range(N_epochs):
        logging.info('Start of epoch %d' % (epoch + 1))
        train_acc.reset_states()
        val_acc.reset_states()
        train_loss.reset_states()
        val_loss.reset_states()
        train_cm.reset_state()
        val_cm.reset_state()

        for data, labels in train_ds:
            train_step(data, labels, opt=opt)
            train_acc_all.append(train_cm.balanced_acc().numpy())

        for data, labels in val_ds:
            val_step(data, labels)
            val_acc_all.append(val_cm.balanced_acc().numpy())

        template = 'Epoch {}, Train Accuracy: {:.2f}, Val Accuracy: {:.2f}, Train Balanced_Acc: {:.2f},' \
                   ' Val Balanced Acc: {:.2f}, Train Loss: {:.2f}, Val Loss: {:.2f}'
        logging.info(template.format(epoch + 1,
                                     train_cm.acc() * 100,
                                     val_cm.acc() * 100,
                                     train_cm.balanced_acc() * 100,
                                     val_cm.balanced_acc() * 100,
                                     train_loss.result(),
                                     val_loss.result()
                                     ))

        wandb.log({'Train loss': train_loss.result(),
                   # 'Train accuracy': train_cm.acc()*100,
                   'Train Balanced Acc': train_cm.balanced_acc() * 100,
                   'Validation loss': val_loss.result(),
                   # 'Validation accuracy': val_cm.acc()*100,
                   'Val Balanced Acc': val_cm.balanced_acc() * 100,
                   'epoch': epoch,
                   'learning_rate': opt.learning_rate.numpy(),
                   'best_Validation_accuracy': val_acc_max * 100
                   }, step=epoch)
        val_cm.plot(show_plot=True, log_wandb=True)

        if val_acc_max < val_cm.balanced_acc() or \
                (val_acc_max == val_cm.balanced_acc() and val_loss.result() < val_loss_min):

            val_acc_max = val_cm.balanced_acc()
            val_loss_min = val_loss.result()
            try:
                logging.info("="*30)
                logging.info("Validation_acc well, saving model weights/checkpoint ...")
                manager.save()
                logging.info("model weights/checkpoint saved successfully.")
            except:
                logging.info("can't save model weights/checkpoint.")
                logging.info("="*30)

        if (epoch + 1) % 5 == 0:
            train_acc_avg = sum(train_acc_all) / len(train_acc_all)
            if train_acc_max < train_acc_avg:
                no_progress_epochs = 0
                train_acc_max = train_acc_avg
            else:
                no_progress_epochs += 5
            train_acc_all = []

            if no_progress_epochs > 100:
                logging.info("=" * 30)
                logging.info("Seems no obvious progress in train_acc, stop training now.")
                break

    logging.info('restoring parameters with best validation_acc...')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train']))
    logging.info("Successfully loaded checkpoint")

    test_loss.reset_states()
    test_cm.reset_state()

    # test step with the restored model
    for n, (image, labels) in enumerate(test_ds):
        test_step(image, labels)

    # evaluation of the test result
    final_test_acc = test_cm.balanced_acc()
    wandb.log({'Test accuracy': final_test_acc})
    test_cm.plot(show_plot=True, log_wandb=True, plottitle="test_cm")
    test_cm.summary()

    return train_cm, val_cm
