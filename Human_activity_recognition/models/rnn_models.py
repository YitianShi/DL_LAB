import keras.layers
from tensorflow import keras as k


class RNN(k.Model):
    def __init__(self, num_of_classes,
                 cell_type="rnn",
                 unit=20,
                 return_sequences=False,
                 Bidirection=True,
                 activation="tanh",
                 dropout_rate=1):
        super(RNN, self).__init__()
        self.return_sequences = return_sequences
        self.activation = activation
        self.Bidirection = Bidirection
        self.num_of_classes = num_of_classes
        self.cell_type = cell_type
        self.unit = unit
        self.rnn_architecture()
        self.last_layers_architecture(self.return_sequences)
        self.dropout_rate = dropout_rate
        self.dropout_layer = keras.layers.Dropout(self.dropout_rate)

    def rnn_architecture(self):

        if self.cell_type == 'lstm':
            self.rnn_layer = k.layers.LSTM(units=self.unit,
                                           return_sequences=self.return_sequences,
                                           activation=self.activation)
        elif self.cell_type == 'gru':
            self.rnn_layer = k.layers.GRU(units=self.unit,
                                          return_sequences=self.return_sequences,
                                          activation=self.activation)
        else:
            self.rnn_layer = k.layers.SimpleRNN(units=self.unit,
                                                return_sequences=self.return_sequences,
                                                activation=self.activation)
        if self.Bidirection:
            self.rnn_layer = k.layers.Bidirectional(self.rnn_layer)

    def last_layers_architecture(self, return_sequences):
        if not return_sequences:
            self.dense = k.layers.Dense(self.num_of_classes)
            self.softmax = k.layers.Softmax()
        else:
            self.dense = k.layers.TimeDistributed(k.layers.Dense(self.num_of_classes))
            self.softmax = k.layers.TimeDistributed(k.layers.Softmax())

    def call(self, input):
        output = self.dropout_layer(input)
        output = self.rnn_layer(output)
        output = self.dense(output)
        output = self.softmax(output)
        return output


class stack_RNN(RNN):
    def __init__(self, stacking_list,
                 num_of_classes,
                 return_sequences=False,
                 cell_type='rnn',
                 activation="tanh",
                 dropout_rate=1):
        self.return_sequences = return_sequences
        self.activation = activation
        self.stacking_list = stacking_list

        super(stack_RNN, self).__init__(
            num_of_classes, unit=20,
            return_sequences=self.return_sequences,
            activation=self.activation
        )
        self.rnn_layer = None
        self.rnn_layer_list = []
        self.dropout_rate = dropout_rate
        self.dropout_layer = keras.layers.Dropout(self.dropout_rate)

        if cell_type == 'lstm':
            for i in self.stacking_list[:-1]:
                self.rnn_layer_list.append(k.layers.LSTM(units=i,
                                                         return_sequences=True,
                                                         activation=self.activation))
            self.rnn_layer_list.append(k.layers.LSTM(units=stacking_list[-1],
                                                     return_sequences=self.return_sequences,
                                                     activation=self.activation))

        elif cell_type == 'gru':
            for i in self.stacking_list[:-1]:
                self.rnn_layer_list.append(k.layers.GRU(units=i,
                                                        return_sequences=True,
                                                        activation=self.activation))

            self.rnn_layer_list.append(k.layers.GRU(units=self.stacking_list[-1],
                                                    return_sequences=self.return_sequences,
                                                    activation=self.activation))

        else:
            for i in self.stacking_list[:-1]:
                self.rnn_layer_list.append(k.layers.SimpleRNN(units=i,
                                                              return_sequences=True,
                                                              activation=self.activation))

            self.rnn_layer_list.append(k.layers.SimpleRNN(units=self.stacking_list[-1],
                                                          return_sequences=self.return_sequences,
                                                          activation=self.activation))

        super(stack_RNN, self).last_layers_architecture(return_sequences=self.return_sequences)

    def call(self, input):

        output = self.dropout_layer(input)
        for layer in self.rnn_layer_list:
            output = layer(output)
        output = self.dense(output)
        output = self.softmax(output)

        return output
