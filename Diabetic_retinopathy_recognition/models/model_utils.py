from models.vision_transformer import VisionTransformer
from models.transfer_learning import freezer_Vit, transfer_Vit, transferred_model
from models.densenet import Densenet_121
from models.resnet import Resnet
from tensorflow import keras as k
import tensorflow as tf
import logging
import os
import gin
from util.utils_misc import marked_info

Model_index = {'18': 'ResNet18', '34': 'ResNet34', '50': 'ResNet50', '101': 'ResNet101', '152': 'ResNet152',
               '121': 'DenseNet121', '169': 'DenseNet169', '201': 'DenseNet201', '264': 'DenseNet264',
               'IRV2': 'InceptionResnetV2', 'IV3': 'InceptionV3',
               'B2': 'EfficientNetB2', 'B1': 'EfficientNetB1', 'B0': 'EfficientNetB0',
               'M2': 'MobilenetV2',
               '16': 'vision_transformer_16'
               }


@gin.configurable
def choose_model(model_id, num_classes, batch_size, run_paths, Mode, dpr=0., ckpt_fn='', num_units=None):

    """choose model according to the index form the dictionary"""

    # if the Mode is evaluation, the name of the model in checkpoint file name will have more priority
    # than the one from model_id
    if Mode == 'Evaluate':
        for key, value in Model_index.items():
            if value in ckpt_fn:
                logging.info(f'model {value} found in checkpoint filename')
                model_id = key
                marked_info(f"Evaluation of {value} start")
                break
        else:
            logging.info('no model corresponds to the checkpoint filename ')
    else:
        learning_form = 'Transfer Learning ' \
            if model_id in ['169', 'IV3', 'M2', 'IRV2', '16', 'B2', 'B1', 'B0'] else 'Learning '

        marked_info(learning_form + f"of {Model_index.get(model_id)} start")

    input_shape = (224, 224, 3) if model_id in ['16', 'B2', 'B1', 'B0'] else (256, 256, 3)
    input_shape = (300, 300, 3) if Mode == 'Kaggle' else input_shape
    h, w, c = input_shape

    if model_id in ['18', '34', '50', '101']:
        Transfer_learning = False
        model = Resnet(resnet=int(model_id), num_classes=num_classes)
        after_call = False

    elif model_id == '121':
        Transfer_learning = False
        model = Densenet_121(num_classes=num_classes)
        after_call = False

    elif model_id in ['169', 'IV3', 'M2', 'IRV2', 'B2', 'B1', 'B0']:
        Transfer_learning = True
        backbone_name = Model_index.get(model_id)
        model = transferred_model(backbone_name,
                                  input_shape=input_shape,
                                  num_classes=num_classes,
                                  num_units=num_units,
                                  dpr=dpr)
        after_call = True

    elif model_id == '16':
        Transfer_learning = True
        model = VisionTransformer(h=h,
                                  grid_size=16,
                                  embedded_dim=768,
                                  num_blocks=12,
                                  num_heads=12,
                                  QKV_bias=None,
                                  representation_size=None,
                                  num_classes=num_classes,
                                  dropout=dpr,
                                  name="ViTB16")
        model.build((batch_size, h, w, c))

        if Model_index.get(model_id) not in ckpt_fn:
            transfer_Vit(model_name="ViTB16", model=model, run_paths=run_paths)
            pre_weights_path = os.path.join(run_paths['path_model_others'], 'ViTB16_after_transfer.h5')
            model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

        freezer_Vit(model)
        after_call = False

    else:
        raise ValueError(model_id + ' is not in the index')

    if not after_call:
        Input = k.layers.Input(shape=input_shape, batch_size=batch_size)
        Output = model.call(Input)
        model = k.Model(Input, Output, name=Model_index.get(model_id))

    for layer in reversed(model.layers):
        # print(layer.name, len(layer.get_output_shape_at(0)))
        if len(layer.get_output_shape_at(0)) == 4 and not hasattr(model_id, '16'):
            layer._name = 'last_conv_layer'
            break

    model.summary()

    if Model_index.get(model_id) in ckpt_fn:
        logging.info('checkpoint found, restoring parameters ...')
        ckpt_path = os.path.join(run_paths['path_root'], ckpt_fn, 'ckpts')
        ckpt = tf.train.Checkpoint(model=model)
        try:
            logging.info(f"loading checkpoint from latest ckpt of {ckpt_fn}")
            ckpt.restore(tf.train.latest_checkpoint(ckpt_path))
            # model.load_weights(tf.train.latest_checkpoint(chekpoint_path))
            logging.info("Successfully loaded checkpoint")

        except Exception as e:
            logging.error(e)
            logging.info(f"fail to load checkpoit in {ckpt_path}, start training with raw initialized model")
    else:
        logging.info('no checkpoint found, start training with raw initialized model')

    model_info = {'name': model.name,
                  'transfer': Transfer_learning,
                  'input_shape': input_shape}

    return model, model_info
