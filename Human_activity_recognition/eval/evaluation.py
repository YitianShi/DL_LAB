from eval.metrics import ConfusionMatrix
from data_handling.input_pipeline import reverse_one_hot_coding
import tensorflow as tf


def evaluate(model, ds, run_paths):
    model.trainable = False

    num_output = model.output.shape[-1]

    test_cm = ConfusionMatrix(num_classes=num_output, save_path=run_paths['path_eval'])
    test_cm.reset_state()

    @tf.function
    def test_step(data, labels):
        predictions = model(data)
        labels_one_hot_decoded = reverse_one_hot_coding(labels)
        test_cm.update_state(labels_one_hot_decoded, predictions)

    # test step with the restored model
    for n, (image, labels) in enumerate(ds):
        test_step(image, labels)

    # evaluation of the test result
    test_cm.plot(show_plot=True, log_wandb=True, plottitle="test_cm")
    test_cm.summary()


# combining predictions of multiple models:
def soft_voting(models, ds, run_paths, num_classes=12):
    cm = ConfusionMatrix(num_classes=num_classes, save_path=run_paths['path_eval'])
    cm.reset_state()
    for images, labels in ds:
        predictions = tf.cast(labels * 0, dtype=tf.float32)
        for model in models:
            predictions = predictions.__add__(model(images))
        labels_one_hot_decoded = reverse_one_hot_coding(labels)
        cm.update_state(labels_one_hot_decoded, predictions)

    cm.summary()
    cm.plot(show_plot=True)


