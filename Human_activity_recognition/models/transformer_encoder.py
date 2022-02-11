import tensorflow as tf
import tensorflow.keras as k
import numpy as np

'''
Since the task emphasizes on getting features, 
there's no need to build the Transformer-decoder here any more.
However this model still support S2S by time-distributed dense layers
at the top of the encoder.
'''


class attention(k.layers.Layer):
    def __init__(self, embedded_dim, num_heads=8, name=None):
        super(attention, self).__init__(name=name)
        kernel_init = k.initializers.GlorotUniform()
        bias_init = k.initializers.RandomNormal(stddev=1e-6)
        self.embedded_dim = embedded_dim
        self.num_heads = num_heads
        self.head_dim = embedded_dim // num_heads
        self.QK_scale = self.head_dim ** -0.5
        self.dense_QKV = k.layers.Dense(3 * embedded_dim,
                                        kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_projection = k.layers.Dense(embedded_dim, kernel_initializer=kernel_init, bias_initializer=bias_init)

    def call(self, inputs, training=None):
        QKV = self.dense_QKV(inputs)
        QKV = tf.reshape(QKV, (inputs.shape[0], inputs.shape[1], 3, self.num_heads, self.head_dim))
        QKV = tf.transpose(QKV, [2, 0, 3, 1, 4])
        Q, K, V = QKV[0], QKV[1], QKV[2]

        weights = tf.nn.softmax(tf.matmul(a=Q, b=K, transpose_b=True) * self.QK_scale, axis=-1)
        weighted_V = tf.matmul(weights, V)
        weighted_V = tf.transpose(weighted_V, [0, 2, 1, 3])
        outputs = tf.reshape(weighted_V, (inputs.shape[0], inputs.shape[1], self.embedded_dim))
        outputs = self.dense_projection(outputs)
        return outputs


class FeedForward(k.layers.Layer):
    k_ini = k.initializers.GlorotUniform()
    b_ini = k.initializers.RandomNormal(stddev=1e-6)

    def __init__(self, embedded_dim, dff=2048, dropoutrate=0., name=None):
        super(FeedForward, self).__init__(name=name)
        self.fc1 = k.layers.Dense(int(dff),
                                  kernel_initializer=self.k_ini,
                                  bias_initializer=self.b_ini)
        self.act = k.layers.Activation("relu")
        self.fc2 = k.layers.Dense(embedded_dim,
                                  kernel_initializer=self.k_ini,
                                  bias_initializer=self.b_ini)
        self.drop = k.layers.Dropout(dropoutrate)

    def call(self, inputs, training=None):
        x = self.act(self.fc1(inputs))
        x = self.drop(self.fc2(x), training=training)
        return x


class Encoder_block(k.layers.Layer):
    def __init__(self,
                 embedded_dim,
                 dff,
                 dpr=0.1,
                 num_heads=8,
                 QKV_bias=False, name=None):
        super(Encoder_block, self).__init__(name=name)
        self.norm1 = k.layers.LayerNormalization(epsilon=1e-6)
        self.Attention = attention(embedded_dim, num_heads=num_heads,
                                   dropoutrate=0.)

        self.drop1 = k.layers.Dropout(rate=dpr)
        self.drop2 = k.layers.Dropout(rate=dpr)

        self.norm2 = k.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(embedded_dim, dff=dff, dropoutrate=dpr)

    def call(self, inputs, training=None):
        x = self.norm1(inputs + self.drop1(self.Attention.call(inputs)[0], training=training))
        x = self.norm2(x + self.drop2(self.ffn.call(x), training=training))
        return x


class Encoder(k.layers.Layer):
    def __init__(self, kernel_size, dff, num_layers, num_heads, embedded_dim, dpr, num_channels=6):
        super(Encoder, self).__init__()
        self.embedded_dim = embedded_dim
        self.num_layers = num_layers
        self.Encoder_blocks = [Encoder_block(embedded_dim, dff, dpr=dpr, num_heads=num_heads, name=f'Encoder_block{i}')
                               for i in range(num_layers)]
        self.dropout = k.layers.Dropout(dpr)
        self.embedding = k.layers.Conv1D(embedded_dim, kernel_size, 1, padding='same',)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.lrelu = tf.keras.layers.LeakyReLU()

    def positional_Embedding(self, window_size):
        self.Embedding = np.zeros((window_size, self.embedded_dim))
        for pos in range(window_size):
            for i in range(self.embedded_dim):
                if i % 2 == 0:
                    self.Embedding[pos][i] = np.sin(pos / pow(10000.0, i / self.embedded_dim))
                else:
                    self.Embedding[pos][i] = np.cos(pos / pow(10000.0, (i - 1) / self.embedded_dim))
        self.Embedding = tf.expand_dims(tf.convert_to_tensor(self.Embedding, dtype=tf.float32), 0)

    def call(self, inputs, training):
        self.positional_Embedding(inputs.shape[-2])
        seq_len = tf.shape(inputs)[1]
        inputs *= tf.math.sqrt(tf.cast(self.embedded_dim, tf.float32))
        x = self.embedding(inputs)
        x = self.batch_norm(x)
        x = self.lrelu(x)
        x += self.Embedding
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.Encoder_blocks[i].call(x, training)
        return x


class Transformer_Encoder(tf.keras.Model):
    def __init__(self,
                 kernel_size=6,
                 num_layers=12,
                 num_classes=12,
                 embedded_dim=512,
                 dff=2048,
                 num_heads=8,
                 S2S=False,
                 name="TransformerS2S",
                 training=True,
                 dpr=0.1):
        super(Transformer_Encoder, self).__init__(name=name)
        self.training = training
        self.encoder = Encoder(kernel_size, dff, num_layers, num_heads, embedded_dim, dpr)
        self.num_classes = num_classes
        self.S2S = S2S
        if S2S:
            self.dpr = k.layers.TimeDistributed(k.layers.Dropout(dpr))
            self.dense = k.layers.TimeDistributed(k.layers.Dense(self.num_classes, "softmax"))
        else:
            self.pooling = k.layers.GlobalAveragePooling1D()
            self.dpr = k.layers.Dropout(dpr)
            self.dense = k.layers.Dense(self.num_classes, "softmax")

    def call(self, inputs):
        x = self.encoder.call(inputs, training=self.training)
        final_output = self.dense(self.dpr(x if self.S2S else self.pooling(x)))
        return final_output
