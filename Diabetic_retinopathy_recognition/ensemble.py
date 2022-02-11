import tensorflow as tf
from tensorflow import keras as k
import gin
import logging
from models.model_utils import choose_model
from train import Trainer
from eval.metrics import ConfusionMatrix
import os
import glob

Model_index = {'18': 'ResNet18', '34': 'ResNet34', '50': 'ResNet50', '101': 'ResNet101', '152': 'ResNet152',
               '121': 'DenseNet121', '169': 'DenseNet169', '201': 'DenseNet201', '264': 'DenseNet264',
               'IRV2': 'InceptionResnetV2', 'IV3': 'InceptionV3',
               'B2': 'EfficientNetB2', 'B3': 'EfficientNetB3', 'B4': 'EfficientNetB4',
               'M2': 'MobilenetV2',
               '16': 'vision_transformer_16'}


def load_models(ckpt_fns, run_paths):
    """ load models for ensemble learning """

    model_stack = []
    for ckpt_fn in ckpt_fns:
        for model_id, model_name in Model_index.items():
            if model_name in ckpt_fn:
                model, model_info = choose_model(model_id=model_id, ckpt_fn=ckpt_fn, Mode='Evaluate',
                                                 run_paths=run_paths)
                model.trainable = False
                model_stack.append((model, model_info))
    return model_stack


def load_models_k(ckpt_fn, run_paths):
    model_stack = []
    for model_id, model_name in Model_index.items():
        if model_name in ckpt_fn:
            for i in range(len(glob.glob(os.path.join(run_paths['path_root'], ckpt_fn, 'ckpts', '*')))):
                model, model_info = choose_model(model_id=model_id, Mode='Train', run_paths=run_paths)
                ckpt_path = os.path.join(run_paths['path_root'], ckpt_fn, 'ckpts', f'fold_{i}')
                ckpt = tf.train.Checkpoint(model=model)
                ckpt.restore(tf.train.latest_checkpoint(ckpt_path))
                model.trainable = False
                model_stack.append((model, model_info))
    return model_stack


@gin.configurable
def Voting(test_ds_all, num_classes, run_paths, ckpt_fns):
    """ voting of models """

    def vote(preds):
        votes = []
        preds = tf.transpose(preds)
        for pred in preds:
            y, _, count = tf.unique_with_counts(pred)
            votes.append(y[tf.argmax(count).numpy()])
        return tf.concat(votes, axis=0)

    models = load_models(ckpt_fns, run_paths) if isinstance(ckpt_fns, list) \
        else load_models_k(ckpt_fns, run_paths)
    preds = []
    for model_with_info in models:
        model = model_with_info[0]
        model_info = model_with_info[1]
        test_ds = test_ds_all[1] if model_info['input_shape'][0] == 224 else test_ds_all[0]
        for images, labels in test_ds:
            if 'EfficientNet' in model.name:
                images *= 255.0
            pred = tf.argmax(model.call(images), axis=-1)
            preds.append(tf.expand_dims(pred, axis=0))
            tf.print(tf.squeeze(labels))
    preds = tf.concat(preds, axis=0)
    result_onehot = tf.one_hot(vote(preds), num_classes)
    print(tf.squeeze(pred).tolist() for pred in preds)

    cm = ConfusionMatrix(num_classes)
    cm.reset_state()
    for _, labels in test_ds_all[0]:
        cm.update_state(labels, result_onehot)
    logging.info('result of Voting:')
    cm.summary(run_paths['path_model_id'])


@gin.configurable()
class StackingModel(k.Model):
    def __init__(self, num_classes, models: list, name=None):
        super(StackingModel, self).__init__(name=name)
        self.models = models
        self.dense = k.layers.Dense(num_classes, name='ensemble_dense', )

    def call(self, inputs):
        results = [tf.expand_dims(tf.argmax(model.call(inputs), -1), -1) for model in self.models]
        results = tf.concat(results, axis=-1) * 2 - 1
        tf.print(results)
        outputs = self.dense(results)
        return outputs


@gin.configurable()
def Stacking(train_ds, validation_ds, test_ds, run_paths, ckpt_fns):
    models, model_info = load_models(ckpt_fns, run_paths) if isinstance(ckpt_fns, list) \
        else load_models_k(ckpt_fns, run_paths)
    model = StackingModel(models=models)
    model.build((None, 256, 256, 3))
    Train = Trainer(model, train_ds, validation_ds, test_ds, run_paths, model_info=model_info)
    _, test_acc = Train.train()
    logging.info(f'Last test accuracy:{test_acc}')
