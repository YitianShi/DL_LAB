import logging
import os
import cv2.cv2
import gin
from cv2 import cv2
import glob
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def preprocess_pictures(Graham, run_paths, input_shape, data_path):

    """ preprocess of images with normal resize or self-build Graham preprocess """

    Mode = 'Graham method' if Graham else 'normal resize'
    logging.info(f"preprocessing data with {Mode} preprocess ...")

    path_data_root = os.path.abspath(os.path.join(run_paths['path_data_preprocess'],
                                                  Mode))
    path_data_root = path_data_root + '224' if input_shape[0] == 224 else path_data_root
    path_train_data = os.path.join(path_data_root, "train")
    path_test_data = os.path.join(path_data_root, "test")
    try:
        os.makedirs(path_train_data)
        os.makedirs(path_test_data)
    except:
        logging.info("Directories for preprocessed data already exist.")
    raw_fn = glob.glob(data_path + "/IDRID_dataset/images/train/*.jpg")
    test_fn = glob.glob(data_path + "/IDRID_dataset/images/test/*.jpg")
    if not len(glob.glob(os.path.join(path_train_data, "*.jpg"))) == 413 \
            or not len(glob.glob(os.path.join(path_test_data, "*.jpg"))) == 103:
        logging.info("Preprocessing data ...")
        for i in raw_fn + test_fn:
            stamp = path_data_root + i.split("images")[-1]
            image = preprocess(i, Graham, input_shape)
            cv2.imwrite(stamp, image)
            logging.info("image preprocessed successfully in new filename {} .".format(stamp))
        logging.info("Over")
    else:
        logging.info("Data already preprocessed...")
    return path_data_root


def preprocess(fn, Graham, input_shape):
    h, w, c = input_shape
    image = cv2.imread(fn)
    image = crop_image(image)
    image = cv2.resize(image, (300, 300 * image.shape[0] // image.shape[1]))
    if Graham:
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), image.shape[0] / 18), -4, 128)
    image = cv2.copyMakeBorder(image,
                               int((image.shape[1] - image.shape[0]) / 2),
                               int((image.shape[1] - image.shape[0]) / 2), 0, 0,
                               borderType=cv2.BORDER_CONSTANT,
                               value=(0, 0, 0))
    if Graham:
        mask = np.zeros(image.shape)
        mask = cv2.circle(mask, (mask.shape[0] // 2, mask.shape[1] // 2),
                          int(min(mask.shape[0], mask.shape[1]) * 0.5 * 0.95), (1, 1, 1), -1, 8, 0)
        image = image * mask #+ 128 * (1 - mask) if Graham else image * mask
    image = cv2.copyMakeBorder(image, 1, 0, 1, 0,
                               borderType=cv2.BORDER_CONSTANT,
                               value=(0, 0, 0) if not Graham else (128, 128, 128))
    image = cv2.resize(image, (h, w))
    return image


def crop_image(img, tol=12):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    img = np.stack([img1, img2, img3], axis=-1)
    return img


seed = np.random.seed(525)


def augment(image, Graham=False):
    """Data augmentation"""
    # Randomly rotate
    random_angles = tf.random.uniform(shape=(), minval=-np.pi / 8, maxval=np.pi / 8, )
    image = tfa.image.rotate(image, random_angles)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    scale = tf.random.uniform([2], 0.8, 1)
    shear = tf.random.uniform([2], minval=-0.1, maxval=0.1, dtype=tf.float32)
    image_crop = tf.image.random_crop(image,
                                      size=(scale[0] * image.shape[0],
                                            scale[1] * image.shape[1], 3), seed=seed)
    image = tf.image.resize(image_crop, (image.shape[0], image.shape[1]))
    image = tfa.image.transform(image, [1.0, shear[0], 0, shear[1], 1.0, 0.0, 0.0, 0.0])

    # if not Graham:
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.01)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # tensor_to_image(tf.cast(image*255, 'uint8'), 'example01')
    return image


def tensor_to_image(image, label):
    image = tf.image.encode_jpeg(image)
    name = tf.constant('{}.jpg'.format(label))
    tf.io.write_file(name, image)
