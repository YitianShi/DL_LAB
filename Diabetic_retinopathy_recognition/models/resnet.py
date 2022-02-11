import tensorflow.keras as k
import tensorflow as tf
import numpy as np


class Resblock(k.layers.Layer):
    def __init__(self,
                 num_channel,
                 name='Resblock',
                 stride=1,
                 Head=False,
                 initializer='he_normal',
                 regularizer=k.regularizers.l2(5e-5),
                 ):
        super(Resblock, self).__init__(name=name)
        self.Head = Head
        if Head:
            stride = 2
        self.conv1 = k.layers.Conv2D(filters=num_channel,
                                     kernel_size=3,
                                     padding='same',
                                     strides=stride,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)

        self.conv2 = k.layers.Conv2D(filters=num_channel,
                                     kernel_size=3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)
        if Head:
            self.conv_bypass = k.layers.Conv2D(filters=num_channel,
                                               kernel_size=1,
                                               padding='same',
                                               strides=2,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=regularizer)
        self.bn1 = k.layers.BatchNormalization()
        self.bn2 = k.layers.BatchNormalization()

    def call(self, inputs):
        outputs = tf.nn.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.Head:
            inputs = self.conv_bypass(inputs)
        outputs += inputs
        outputs = tf.nn.relu(outputs)
        return outputs


class Resblock_neo(k.layers.Layer):
    def __init__(self,
                 num_channel,
                 first_block=False,
                 Head=False,
                 initializer='he_normal',
                 regularizer=k.regularizers.l2(5e-5),
                 name="Resblock_neo"
                 ):
        super(Resblock_neo, self).__init__(name=name)
        self.Head = Head
        stride = 1
        if Head and not first_block:
            stride = 2
        self.conv1 = k.layers.Conv2D(filters=num_channel,
                                     kernel_size=1,
                                     padding='same',
                                     strides=stride,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)

        self.conv2 = k.layers.Conv2D(filters=num_channel,
                                     kernel_size=3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)

        self.conv3 = k.layers.Conv2D(filters=4 * num_channel,
                                     kernel_size=1,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)
        if Head or first_block:
            self.conv_bypass = k.layers.Conv2D(filters=4 * num_channel,
                                               kernel_size=1,
                                               padding='same',
                                               strides=stride,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=regularizer)
        self.bn1 = k.layers.BatchNormalization()
        self.bn2 = k.layers.BatchNormalization()
        self.bn3 = k.layers.BatchNormalization()

    def call(self, inputs):
        outputs = tf.nn.relu(self.bn1(self.conv1(inputs)))
        outputs = tf.nn.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        if self.Head:
            inputs = self.conv_bypass(inputs)
        outputs += inputs
        outputs = tf.nn.relu(outputs)
        return outputs



class Resnet(k.Model):
    def __init__(self,
                 num_classes=5,
                 resnet=18,
                 initializer='he_normal',
                 regularizer=k.regularizers.l2(5e-5)):

        super(Resnet, self).__init__(name="Resnet{}".format(resnet))
        self.conv1 = k.layers.Conv2D(filters=64,
                                     kernel_size=7,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)
        self.bn = k.layers.BatchNormalization()
        self.pl = k.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.Res_layer_list = []

        num_channel = 64
        if resnet == 34 or resnet == 50:
            configuration = [3, 4, 6, 3]
        elif resnet == 101:
            configuration = [3, 4, 23, 3]
        elif resnet == 152:
            configuration = [3, 8, 36, 3]
        else:
            configuration = [2, 2, 2, 2]
        if resnet == 18 or resnet == 34:
            for index, num_of_blocks_layer in enumerate(configuration):
                if num_channel == 64:
                    self.Res_layer_list.append(Resblock(num_channel=num_channel,
                                                        initializer=initializer,
                                                        regularizer=regularizer, ))
                else:
                    self.Res_layer_list.append(Resblock(num_channel=num_channel,
                                                        Head=True,
                                                        initializer=initializer,
                                                        regularizer=regularizer))
                for i in range(num_of_blocks_layer - 1):
                    self.Res_layer_list.append(Resblock(num_channel=num_channel,
                                                        initializer=initializer,
                                                        regularizer=regularizer))
                num_channel *= 2
        elif resnet >= 50:
            for index, num_of_blocks_layer in enumerate(configuration):
                if num_channel == 64:
                    self.Res_layer_list.append(Resblock_neo(num_channel=num_channel,
                                                            first_block=True,
                                                            Head=True,
                                                            initializer=initializer,
                                                            regularizer=regularizer))
                else:
                    self.Res_layer_list.append(Resblock_neo(num_channel=num_channel,
                                                            Head=True,
                                                            initializer=initializer,
                                                            regularizer=regularizer))
                for i in range(num_of_blocks_layer - 1):
                    self.Res_layer_list.append(Resblock_neo(num_channel=num_channel,
                                                            initializer=initializer,
                                                            regularizer=regularizer))
                num_channel *= 2

        for index, layer in enumerate(self.Res_layer_list):
            layer._name = layer.name + str(index)

        self.GAP = k.layers.GlobalAvgPool2D()
        self.FC2 = k.layers.Dense(units=num_classes,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer, name='head')

    def call(self, inputs, **kwargs):
        outputs = self.pl(tf.nn.relu(self.bn(self.conv1(inputs))))
        for block in self.Res_layer_list:
            outputs = block.call(outputs)
        outputs = self.FC2(self.GAP(outputs))
        return outputs
