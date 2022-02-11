from models.vision_transformer import *
import logging
import tensorflow_hub as hub
import os
from tensorflow import keras as k
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, MobileNetV2, DenseNet169, \
    EfficientNetB2, EfficientNetB1, EfficientNetB0


def backboneModel(backbone_name, input_shape):
    if backbone_name == "InceptionV3":
        backbone = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    elif backbone_name == "MobilenetV2":
        backbone = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    elif backbone_name == "InceptionResnetV2":
        backbone = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    elif backbone_name == "DenseNet169":
        backbone = DenseNet169(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    elif backbone_name == "EfficientNetB2":
        backbone = EfficientNetB2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    elif backbone_name == "EfficientNetB1":
        backbone = EfficientNetB1(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    elif backbone_name == "EfficientNetB0":
        backbone = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
    else:
        raise ValueError('no such model in the directory')
    backbone.trainable = False
    return backbone


def transferred_model(backbone_name, input_shape, num_classes, dpr, num_units=None):
    backbone = backboneModel(backbone_name, input_shape)
    x = k.layers.GlobalAvgPool2D()(backbone.output)
    if num_units is not None:
        x = k.layers.Dense(num_units)(x)
    x = k.layers.Dropout(dpr)(x)
    output = k.layers.Dense(num_classes, name='head')(x)
    mdl = k.Model(backbone.input, output, name=backbone_name)
    return mdl


def transfer_Vit(model_name: str, run_paths, model: tf.keras.Model):
    url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
    weights_path = os.path.join(run_paths['path_model_others'], "ViT-B_16.npz")
    if not os.path.exists(weights_path):
        logging.info('No pretrained model weights found. Downloading pretrained model of ViT-16 from ' + url)
        try:
            tf.keras.utils.get_file(weights_path, url)
            logging.info('Successfully downloaded pretrained model. Start parsing weights to the self-built ViT '
                         'model...')
        except Exception as e:
            raise KeyError(e)
    else:
        logging.info('Model weights already found. Start parsing weights to the self-built ViT-16 model...')

    var_dict = {v.name.split(':')[0]: v for v in model.weights}
    ckpt_dict = np.load(weights_path, allow_pickle=False)
    # keys, values = zip(*list(ckpt_dict.items()))
    w_dict = {}
    for key, value in ckpt_dict.items():

        key_ = key.replace("Transformer/", ""). \
            replace("MultiHeadDotProductAttention_1", "MultiHeadAttention"). \
            replace("MlpBlock_3", "MlpBlock"). \
            replace("posembed_input/pos_embedding", "class_position_embedd/position_embedd"). \
            replace("encoder_norm/bias", "encoder_norm/beta"). \
            replace("encoder_norm/scale", "encoder_norm/gamma"). \
            replace("LayerNorm_0/bias", "LayerNorm_1/beta"). \
            replace("LayerNorm_0/scale", "LayerNorm_1/gamma"). \
            replace("LayerNorm_2/bias", "LayerNorm_2/beta"). \
            replace("LayerNorm_2/scale", "LayerNorm_2/gamma"). \
            replace("embedding", "embedding/conv2d"). \
            replace("cls", "class_position_embedd/class_token"). \
            replace("encoderblock_", "Encoderblock_")
        if key_ != "head/kernel" or key_ != "head/bias":
            w_dict[key_] = value

    for i in range(model.num_blocks):

        q_kernel = w_dict.pop("Encoderblock_{}/MultiHeadAttention/query/kernel".format(i))
        k_kernel = w_dict.pop("Encoderblock_{}/MultiHeadAttention/key/kernel".format(i))
        v_kernel = w_dict.pop("Encoderblock_{}/MultiHeadAttention/value/kernel".format(i))
        q_kernel = np.reshape(q_kernel, [q_kernel.shape[0], -1])
        k_kernel = np.reshape(k_kernel, [k_kernel.shape[0], -1])
        v_kernel = np.reshape(v_kernel, [v_kernel.shape[0], -1])
        qkv_kernel = np.concatenate([q_kernel, k_kernel, v_kernel], axis=1)
        w_dict["Encoderblock_{}/MultiHeadAttention/QKV/kernel".format(i)] = qkv_kernel

        if model.QKV_bias:
            q_bias = w_dict.pop("Encoderblock_{}/MultiHeadAttention/query/bias".format(i))
            k_bias = w_dict.pop("Encoderblock_{}/MultiHeadAttention/key/bias".format(i))
            v_bias = w_dict.pop("Encoderblock_{}/MultiHeadAttention/value/bias".format(i))
            q_bias = np.reshape(q_bias, [-1])
            k_bias = np.reshape(k_bias, [-1])
            v_bias = np.reshape(v_bias, [-1])
            qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
            w_dict["Encoderblock_{}/MultiHeadAttention/QKV/bias".format(i)] = qkv_bias

        out_kernel = w_dict["Encoderblock_{}/MultiHeadAttention/out/kernel".format(i)]
        out_kernel = np.reshape(out_kernel, [-1, out_kernel.shape[-1]])
        w_dict["Encoderblock_{}/MultiHeadAttention/out/kernel".format(i)] = out_kernel

    for key, var in var_dict.items():
        if key in w_dict:
            if w_dict[key].shape != var.shape:
                msg = "shape mismatch: {}, because {} didn't match {}".format(key, w_dict[key].shape, var.shape)
                logging.info(msg)
            else:
                var.assign(w_dict[key], read_value=False)
                print("just match: {}".format(key))
        else:
            msg = "Not found {} in {}".format(key, weights_path)
            print(msg)
            # pass
    # print(var_dict["head/kernel"],var_dict["head/bias"])
    model.save_weights(run_paths['path_model_others'] + "/{}_after_transfer.h5".format(model_name))
    # https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz


def freezer_Vit(model):
    for layer in model.layers:
        if "Encoderblock" in layer.name \
                or "encoder_norm" in layer.name \
                or "embedd" in layer.name \
                or "dropout" in layer.name:
            layer.trainable = False

    for layer in model.layers:
        if layer.trainable:
            logging.info("trainable layers now:{}".format(layer.name))
        else:
            pass
