import keras
from tensorflow import keras as k
import tensorflow as tf
import logging
import os
import gin

from models.rnn_models import RNN, stack_RNN
from models.transformer_encoder import Transformer_Encoder

Model_index = {'1': 'RNN',
               '2': 'Stacked_RNN',
               '3': 'CNN_LSTM',
               '4': 'Transformer_Encoder'
               }


@gin.configurable
def chooseModel(model_type, input_shape, units, num_classes, batch_size, run_paths, dropout_rate, kernel_size=2,
                ckpt_fn='', Mode='Train'):
    for key, value in Model_index.items():
        if key == model_type:
            model_num = key
            logging.info(f"############################## {Mode} of "
                         f"{value} start ##############################")
            break

    if model_num == '1':
        model = RNN(
            unit=units,
            dropout_rate=dropout_rate,
            num_of_classes=num_classes,
            Bidirection=False,
            activation="tanh",
            return_sequences=False,
            cell_type='lstm')

    elif model_num == '2':
        model = stack_RNN(
            stacking_list=units,  # here tuple, eg.: (8,6)
            dropout_rate=dropout_rate,
            num_of_classes=num_classes,
            activation="tanh",
            return_sequences=False,
            cell_type='lstm')

    elif model_num == '3':
        model = k.models.Sequential()
        model.add(k.layers.ConvLSTM1D(units, kernel_size=kernel_size, padding='same', return_sequences=False))

        model.add(k.layers.Flatten(batch_size=batch_size))
        model.add(k.layers.Dense(12, activation="relu", batch_size=batch_size))

        input_shape = (None, input_shape[0], input_shape[1])

    elif model_num == '4':
        model = Transformer_Encoder(
            kernel_size=7,
            num_classes=num_classes,
            embedded_dim=32,
            num_layers=1,
            dff=128,
            num_heads=4,
            S2S=False,
            name="TransformerS2S",
            training=True,
            dpr=0.3)


    else:
        raise ValueError(model_num + ' is not in the index')

    Input = k.layers.Input(shape=input_shape, batch_size=batch_size)
    Output = model.call(Input)
    model = k.Model(Input, Output, name=model_type)

    model.summary()
    save_path = run_paths['path_model_save'] if Mode == 'Ensemble' else run_paths['path_root']
    print(Model_index.get(model_num), ckpt_fn)
    if Model_index.get(model_num) in ckpt_fn:
        logging.info('checkpoint found, restoring parameters ...')
        ckpt_path = os.path.join(save_path, ckpt_fn, 'ckpts')
        ckpt = tf.train.Checkpoint(model=model)
        try:
            logging.info(f"loading checkpoint from latest ckpt of {ckpt_fn}")
            logging.info(ckpt_path)
            ckpt.restore(tf.train.latest_checkpoint(ckpt_path))
            # model.load_weights(tf.train.latest_checkpoint(chekpoint_path))
            logging.info("Successfully loaded checkpoint")

        except Exception as e:
            logging.error(e)
            logging.info(f"fail to load checkpoit in {ckpt_path}, start training with raw initialized model")
    else:
        logging.info('no checkpoint found, start training with raw initialized model')

    return model
