import gin
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from cv2 import cv2
from tensorflow.python.ops.numpy_ops import np_config
import os


@gin.configurable
class Visualisation:

    """ deep visualisation using grad-CAM, guided back propagation and guided grad-CAM """

    def __init__(self, image_id, run_paths, model, input_shape, Graham):

        self.h, self.w, self.c = input_shape
        self.save_path = run_paths['path_eval']
        Mode = 'Graham method' if Graham else 'normal resize'
        Mode += '224' if self.h == 224 else ''
        stamp = os.path.abspath(os.path.join(run_paths['path_data_preprocess'], Mode, 'train', image_id))
        print(stamp)

        image = cv2.imread(stamp)
        self.image = tf.io.decode_jpeg(tf.io.read_file(stamp), channels=3)
        self.image = tf.expand_dims(tf.cast(self.image, tf.float32), 0)
        if 'EfficientNet' not in model.name:
            self.image /= 255.0
        self.grey_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        self.model = model
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if layer.name == 'last_conv_layer' or 'coder' in layer.name:
                last_conv_layer = layer
                break

        self.grad_model = keras.Model(model.inputs, [last_conv_layer.output, model.output])

    def GradCAM(self):

        """ define grad-CAM process"""

        with tf.GradientTape() as tape:
            feature_map, output = self.grad_model(self.image)
            index = tf.argmax(output, axis=-1)[0]
            output = output[:, index]
        grad = tape.gradient(output, feature_map)
        grad_average = tf.reduce_mean(grad, axis=(0, 1, 2))

        feature_map = feature_map.numpy()[0]
        for i in range(feature_map.shape[-1]):
            feature_map[:, :, i] *= grad_average.numpy()[i]
        heatmap = np.mean(feature_map, axis=-1)

        # heatmap generation
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (self.h, self.w), cv2.INTER_LINEAR)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        map_show(self.grey_image, heatmap, self.save_path, 'Grad-CAM')

        return heatmap / 255.0

    def guidedBackProp(self):

        """ define guided back propagation process"""

        for layer in self.model.layers:
            if hasattr(layer, "activation"):
                if layer.activation == k.activations.relu:
                    layer.activation = guided_ReLU

        with tf.GradientTape() as tape:
            tape.watch(self.image)
            output = self.model(self.image)
            index = tf.argmax(output, axis=-1)[0]
            output = output[:, index]
        gbp_map = tape.gradient(output, self.image)[0].numpy()

        gbp_map = cv2.resize(gbp_map, (self.h, self.w), cv2.INTER_LINEAR)

        np_config.enable_numpy_behavior()
        gbp_map = (gbp_map - gbp_map.mean()) * 0.25 / (gbp_map.std() + k.backend.epsilon())
        gbp_map = (np.clip(gbp_map + 0.5, 0.25, 1) * 255).astype('uint8')
        gbp_map = cv2.cvtColor(gbp_map, cv2.COLOR_BGR2RGB)

        map_show(self.grey_image, gbp_map, self.save_path, 'Guided Back-Propagation')

        return gbp_map / 255.0

    def guidedGradCAM(self):

        """ define guided grad-CAM process """

        ggcam_map = self.guidedBackProp() * self.GradCAM()
        ggcam_map = np.uint8(255 * ggcam_map)
        map_show(self.grey_image, ggcam_map, self.save_path, 'Guided Grad_CAM')


# overlap heatmap with image
def map_show(image, Map, save_path, title):
    np_config.enable_numpy_behavior()
    Map = 0.6 * Map + image
    Map = (Map / np.max(Map) * 255).astype('uint8')
    plt.matshow(Map)
    plt.title('{}'.format(title))
    plt.savefig(f'{save_path}/{title}.png')


@tf.custom_gradient
def guided_ReLU(x):
    y = tf.nn.relu(x)

    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy

    return y, grad
