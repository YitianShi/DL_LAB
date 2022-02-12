import tensorflow as tf
from models.model_utils import chooseModel, Model_index
from util import utils_misc
from util import utils_params
import logging
import gin


@gin.configurable
def tflite_generator(model_type):
    gin.parse_config_files_and_bindings(['configs/config_tffile.gin'], [])
    run_paths = utils_params.gen_run_folder(Mode='Evaluate', path_model_id=Model_index.get(model_type),
                                            source_data='HAPT_waist')
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    model = chooseModel(model_type, run_paths=run_paths, Mode='Evaluate')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    open("../Android_App_Human_activity_recognition/MyApplication/app/src/main/ml/tflite_model.tflite", "wb").write(tflite_model)


tflite_generator('2')
