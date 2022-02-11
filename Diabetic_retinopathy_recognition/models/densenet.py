import tensorflow.keras as k
import tensorflow as tf
import numpy as np


class Dense_unit(k.layers.Layer):
    def __init__(self, num_channel=64, name="Dense_unit",
                 initializer='he_normal',
                 regularizer=k.regularizers.l2(5e-5),
                ):
        super(Dense_unit, self).__init__(name=name)
        self.conv1 = k.layers.Conv2D(filters=2 * num_channel,
                                     kernel_size=1,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer
                                     )

        self.conv2 = k.layers.Conv2D(filters=num_channel / 2,
                                     kernel_size=3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer
                                     )
        self.bn1 = k.layers.BatchNormalization()
        self.bn2 = k.layers.BatchNormalization()

    def call(self, inputs):
        outputs = self.conv1(tf.nn.relu(self.bn1(inputs)))
        outputs = self.conv2(tf.nn.relu(self.bn2(outputs)))
        return outputs


class Dense_block(k.layers.Layer):
    def __init__(self, num_units, block_num, num_channel,
                 name="Dense_block",
                 initializer='he_normal',
                 regularizer=k.regularizers.l2(5e-5),
                 ):
        super(Dense_block, self).__init__(name=name)
        self.block = [Dense_unit(name="Dense_unit_{}".format(i + 1),
                                 initializer=initializer,
                                 regularizer=regularizer,
                                 ) for i in range(num_units)]
        self.bn = k.layers.BatchNormalization()
        self.block_num = block_num
        if block_num == 3:
            self.pl = k.layers.GlobalAvgPool2D()
        else:
            self.conv = k.layers.Conv2D(num_channel, 1,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizer
                                        )
            self.pl = k.layers.AveragePooling2D(pool_size=2, strides=2)

    def call(self, inputs):
        for layer in self.block:
            x = layer.call(inputs)
            inputs = tf.concat((x, inputs), axis=-1)
        if not self.block_num == 3:
            inputs = self.conv(inputs)
        return self.pl(self.bn(inputs))


class Densenet_121(k.Model):
    def __init__(self, num_classes=5, initializer='he_normal', Densenet=121,
                 regularizer=k.regularizers.l2(5e-5)):
        super(Densenet_121, self).__init__(name="Densenet_121")
        self.conv1 = k.layers.Conv2D(filters=64,
                                     kernel_size=7,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)  # 128
        self.bn = k.layers.BatchNormalization()
        self.pl = k.layers.AveragePooling2D(pool_size=3, strides=2, padding="same")  # 65
        if Densenet == 121:
            configuration = [6, 12, 24, 16]
        elif Densenet == 169:
            configuration = [6, 12, 32, 32]
        elif Densenet == 201:
            configuration = [6, 12, 48, 32]
        elif Densenet == 264:
            configuration = [6, 12, 64, 48]
        num_channel = [64]
        for i in range(4):
            num_channel.append((32 * configuration[i] + num_channel[i]) // 2)
        self.Dense_blocks = [
            Dense_block(num_units=units,
                        num_channel=num_channel[block_num + 1],
                        block_num=block_num,
                        name="Dense_block{}".format(block_num + 1))
            for block_num, units in enumerate(configuration)]

        self.FC = k.layers.Dense(num_classes, kernel_initializer=initializer,
                                 kernel_regularizer=regularizer, name='head')

    def call(self, inputs):
        outputs = self.pl(tf.nn.relu(self.bn(self.conv1(inputs))))
        for block in self.Dense_blocks:
            outputs = block.call(outputs)
        return self.FC(outputs)
