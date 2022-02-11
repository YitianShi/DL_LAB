import tensorflow as tf
import tensorflow.keras as k
import numpy as np


class Embedding(k.layers.Layer):
    def __init__(self, embedded_dim=768, h=196, grid_size=16, name=None):
        super(Embedding, self).__init__(name=name)
        self.conv1 = k.layers.Conv2D(embedded_dim,
                                     kernel_size=grid_size,
                                     strides=grid_size,
                                     padding='same',
                                     kernel_initializer=k.initializers.LecunNormal(),
                                     name='conv2d'
                                     )
        self.embedded_dim = embedded_dim
        self.grid = (h // grid_size, h // grid_size)
        self.num_grid = self.grid[0] * self.grid[1]

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = tf.reshape(x, [x.shape[0], self.num_grid, self.embedded_dim])
        return x


class class_token_and_position_embedding(k.layers.Layer):
    def __init__(self, embedded_dim=768, num_grid=196, name=None):
        super(class_token_and_position_embedding, self).__init__(name=name)
        self.embedded_dim = embedded_dim
        self.num_grid = num_grid

    def build(self, input_shape):
        self.class_token = self.add_weight(name="class_token",
                                           shape=(1, 1, self.embedded_dim),
                                           initializer=k.initializers.Zeros(),
                                           trainable=True,
                                           dtype=tf.float32)
        self.position_embedding = self.add_weight(name="position_embedd",
                                                  shape=(1, 1 + self.num_grid, self.embedded_dim),
                                                  initializer=k.initializers.Zeros(),
                                                  trainable=True,
                                                  dtype=tf.float32)

    def call(self, inputs):
        class_token = tf.broadcast_to(self.class_token, shape=(inputs.shape[0], 1, self.embedded_dim))
        x = tf.concat((class_token, inputs), axis=1) + self.position_embedding
        return x


class attention(k.layers.Layer):
    def __init__(self, embedded_dim, num_heads=8, QKV_bias=False, dropoutrate=0., name=None
                 ):
        super(attention, self).__init__(name=name)
        kernel_init = k.initializers.GlorotUniform()
        bias_init = k.initializers.RandomNormal(stddev=1e-6)
        self.embedded_dim = embedded_dim
        self.num_heads = num_heads
        self.head_dim = embedded_dim // num_heads
        self.QK_scale = embedded_dim ** -0.5
        self.dense_QKV = k.layers.Dense(3 * embedded_dim, use_bias=QKV_bias, name="QKV",
                                        kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dropout1 = k.layers.Dropout(dropoutrate)
        self.dense_projection = k.layers.Dense(embedded_dim, kernel_initializer=kernel_init,
                                               name="out", bias_initializer=bias_init)
        self.dropout2 = k.layers.Dropout(dropoutrate)

    def call(self, inputs, training=None):
        QKV = self.dense_QKV(inputs)
        QKV = tf.reshape(QKV, (inputs.shape[0], inputs.shape[1], 3, self.num_heads, self.head_dim))
        QKV = tf.transpose(QKV, [2, 0, 3, 1, 4])
        Q, K, V = QKV[0], QKV[1], QKV[2]

        Attention = tf.nn.softmax(tf.matmul(a=Q, b=K, transpose_b=True) * self.QK_scale, axis=-1)
        Attention = self.dropout1(Attention, training=training)

        weighted_V = tf.matmul(Attention, V)
        weighted_V = tf.transpose(weighted_V, [0, 2, 1, 3])
        outputs = tf.reshape(weighted_V, (inputs.shape[0], inputs.shape[1], self.embedded_dim))

        outputs = self.dense_projection(outputs)
        outputs = self.dropout2(outputs)
        return outputs


class MLP(k.layers.Layer):
    k_ini = k.initializers.GlorotUniform()
    b_ini = k.initializers.RandomNormal(stddev=1e-6)

    def __init__(self, embedded_dim, ratio=4, dropoutrate=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = k.layers.Dense(int(embedded_dim * ratio),
                                  kernel_initializer=self.k_ini,
                                  bias_initializer=self.b_ini,
                                  name="Dense_0")
        self.act = k.layers.Activation("gelu")
        self.fc2 = k.layers.Dense(embedded_dim,
                                  kernel_initializer=self.k_ini,
                                  bias_initializer=self.b_ini,
                                  name="Dense_1")
        self.drop = k.layers.Dropout(dropoutrate)

    def call(self, inputs, training=None):
        x = self.drop(self.act(self.fc1(inputs)), training=training)
        x = self.drop(self.fc2(x), training=training)
        return x


class Transformer_encoder(k.layers.Layer):
    def __init__(self,
                 embedded_dim,
                 drop_path_rate=0.,
                 dropoutrate=0.,
                 num_heads=8,
                 QKV_bias=False, name=None):
        super(Transformer_encoder, self).__init__(name=name)
        self.norm1 = k.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.Attention = attention(embedded_dim, num_heads=num_heads,
                                   QKV_bias=QKV_bias,
                                   dropoutrate=0., name="MultiHeadAttention")
        if dropoutrate > 0:
            self.drop_path = k.layers.Dropout(rate=drop_path_rate, noise_shape=(None, 1, 1))
        else:
            self.drop_path = k.layers.Activation("linear")

        self.norm2 = k.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        self.mlp = MLP(embedded_dim, dropoutrate=dropoutrate, name="MlpBlock")

    def call(self, inputs, training=None):
        x = inputs + self.drop_path(self.Attention(self.norm1(inputs)), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


class VisionTransformer(k.Model):

    def __init__(self,
                 num_classes=5,
                 h=256,
                 grid_size=16,
                 embedded_dim=768,
                 num_blocks=12,
                 num_heads=12,
                 QKV_bias=None,
                 representation_size=None,
                 dropout=0.,
                 name="ViTB16"):
        super(VisionTransformer, self).__init__(name=name)
        self.embedded_dim = embedded_dim
        self.num_blocks = num_blocks
        self.QKV_bias = QKV_bias
        self.num_classes = num_classes

        self.Embedding = Embedding(h=h, grid_size=grid_size, embedded_dim=embedded_dim, name='embedding')
        num_grid = self.Embedding.num_grid
        self.class_token_and_position_embedding = class_token_and_position_embedding(
            embedded_dim=embedded_dim, num_grid=num_grid, name="class_position_embedd")
        self.dropout = k.layers.Dropout(dropout)
        drop_path_rate = np.linspace(0, dropout, num_blocks)
        self.transformer_encoder_block = [Transformer_encoder(embedded_dim=embedded_dim,
                                                              dropoutrate=0., drop_path_rate=drop_path_rate[i],
                                                              num_heads=num_heads,
                                                              QKV_bias=QKV_bias,
                                                              name="Encoderblock_{}".format(i)) for i in
                                          range(num_blocks)]
        self.norm = k.layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")
        if representation_size:
            self.layer_logits = k.layers.Dense(representation_size, activation="tanh", name='logits')
        else:
            self.layer_logits = k.layers.Activation("linear")

        self.head = k.layers.Dense(num_classes, kernel_initializer=k.initializers.he_normal, name='head')

    def call(self, inputs, training=None):
        x = self.Embedding(inputs)
        x = self.class_token_and_position_embedding(x)
        x = self.dropout(x, training=training)
        for block in self.transformer_encoder_block:
            x = block(x, training=training)
        x = self.norm(x)
        x = self.layer_logits(x[:, 0])
        x = self.head(x)
        return x
