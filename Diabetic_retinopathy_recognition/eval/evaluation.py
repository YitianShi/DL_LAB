from eval.deep_visualisation import Visualisation
import tensorflow as tf
from eval.metrics import ConfusionMatrix
import os
import logging


def evaluate(model, ds, run_paths, model_info):
    input_shape = model_info['input_shape']
    model.trainable = False
    if 'bedding' not in model.layers[1].name:
        Map = Visualisation(model=model, run_paths=run_paths, input_shape=input_shape)
        Map.guidedGradCAM()
        Map.GradCAM()
    else:
        logging.info('GradCAM not support Vit-16')

    num_output = model.output.shape[-1]

    cm = ConfusionMatrix(num_classes=num_output) if not num_output == 1 \
        else ConfusionMatrix(num_classes=5, Regression=True)

    cm.reset_state()
    for images, labels in ds:
        if 'EfficientNet' in model.name:
            images *= 255.
        predictions = model(images)
        cm.update_state(labels, predictions)

    save_path = run_paths['path_eval']
    cm.summary(save_path)
    cm.plot(save_path)
