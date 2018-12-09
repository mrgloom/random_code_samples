import sys
import random as rn

import setGPU
import tensorflow as tf
import keras
gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.2)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_config))
keras.backend.tensorflow_backend.set_session(session)
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adadelta
from keras import backend as K
import numpy as np
import cv2

from keras.callbacks import TensorBoard, Callback

from custom_model import get_unet
import config

rn.seed(2018)
np.random.seed(2018)

#MULTIPLIER = 1.0
MULTIPLIER = config.IMAGE_H*config.IMAGE_W - 1
OUTPUT_ACTIVATION = 'linear'
#OUTPUT_ACTIVATION = 'relu'
#OUTPUT_ACTIVATION = 'sigmoid'
IS_USE_GAUSSIAN = False
IS_BN_ON_OUTPUT = True
EXPERIMENT_NAME = 'unet_mse_loss'

def batch_generator():
    def gen_random_sample():
        img = np.zeros((config.IMAGE_H, config.IMAGE_W, config.N_CHANNELS), dtype=np.uint8)
        mask = np.zeros((config.IMAGE_H, config.IMAGE_W, config.N_CLASSES), dtype=np.uint8)

        colors = np.random.permutation(256)

        # Background
        img[:, :, 0] = colors[0]
        img[:, :, 1] = colors[1]
        img[:, :, 2] = colors[2]

        # Object
        obj1_color0 = colors[3]
        obj1_color1 = colors[4]
        obj1_color2 = colors[5]
        while True:
            center_x = rn.randint(0, config.IMAGE_W)
            center_y = rn.randint(0, config.IMAGE_H)
            r_x = rn.randint(10, 50)
            r_y = rn.randint(10, 50)
            if (center_x + r_x < config.IMAGE_W and center_x - r_x > 0 and
                    center_y + r_y < config.IMAGE_H and center_y - r_y > 0):
                cv2.ellipse(img, (int(center_x), int(center_y)), (int(r_x), int(r_y)), int(0), int(0), int(360),
                            (int(obj1_color0), int(obj1_color1), int(obj1_color2)), int(-1))
                mask[center_y, center_x] = 1
                break

        # White noise
        density = rn.uniform(0, 0.1)
        for i in range(config.IMAGE_H):
            for j in range(config.IMAGE_W):
                if rn.random() < density:
                    img[i, j, 0] = rn.randint(0, 255)
                    img[i, j, 1] = rn.randint(0, 255)
                    img[i, j, 2] = rn.randint(0, 255)


        # Gradient image
        if rn.random() > 0.5:
            if rn.random() > 0.5:
                # Horizontal gradient
                row = np.arange(config.IMAGE_W)
                if rn.random() > 0.5:
                    row = row[::-1]
                grad_img = np.tile(row, (config.IMAGE_H, 1))
                grad_img = np.clip(grad_img, 0, 255)
                grad_img = grad_img.astype(np.uint8)
                grad_img = cv2.cvtColor(grad_img, cv2.COLOR_GRAY2BGR)
            else:
                # Vertical gradient
                col = np.arange(config.IMAGE_H).transpose()
                if rn.random() > 0.5:
                    col = col[::-1]
                col = col[:, np.newaxis]
                grad_img = np.tile(col, (1, config.IMAGE_W))
                grad_img = np.clip(grad_img, 0, 255)
                grad_img = grad_img.astype(np.uint8)
                grad_img = cv2.cvtColor(grad_img, cv2.COLOR_GRAY2BGR)

            grad_img = grad_img / 255.0
            img = img * grad_img
            img = img.astype(np.uint8)

        # cv2.imshow('image', img)
        # k = cv2.waitKey(0)
        # if k == 27 or k == ord('q'):
        #     cv2.destroyAllWindows()
        #     sys.exit()

        return img, mask

    while True:
        # Generate one batch of data
        img_arr = np.zeros((config.BATCH_SIZE, config.IMAGE_H, config.IMAGE_W, config.N_CHANNELS), dtype=np.float32)
        mask_arr = np.zeros((config.BATCH_SIZE, config.IMAGE_H, config.IMAGE_W, config.N_CLASSES), dtype=np.float32)
        for i in range(config.BATCH_SIZE):
            img, mask = gen_random_sample()
            img_arr[i] = img
            mask_arr[i] = mask

        # Preprocessing
        img_arr = img_arr / 255.0
        mask_arr = mask_arr * MULTIPLIER

        yield img_arr, mask_arr


class CustomCallback(Callback):
    def __init__(self, train_generator, test_generator):
        self.train_generator = train_generator
        self.test_generator = test_generator

    def summary_image(self, tensor):
        import io
        from PIL import Image

        tensor = tensor.astype(np.uint8)

        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                             width=width,
                             colorspace=channel,
                             encoded_image_string=image_string)

    def probability_map_to_color_map(self, probability_map):
        # From probability map to normalized color map
        probability_map = probability_map / np.max(probability_map)
        probability_map = probability_map * 255.0
        probability_map = np.clip(probability_map, 0, 255)
        probability_map = probability_map.astype(np.uint8)
        color_map = cv2.applyColorMap(probability_map, cv2.COLORMAP_JET)
        return color_map

    def on_epoch_end(self, epoch, logs={}):
        def get_visualization(generator):
            img_arr, mask_arr = next(generator)

            idx = 0
            res = np.zeros((config.IMAGE_H, config.IMAGE_W * 4, 3))

            y_pred = self.model.predict(img_arr[0:1, ...])

            # Original image
            res[:, :config.IMAGE_W, :] = img_arr[idx,:] * 255

            # Ground truth mask
            mask = mask_arr[idx,:]
            mask = (mask / MULTIPLIER) * 255
            mask = mask.astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            res[:, config.IMAGE_W: 2 * config.IMAGE_W, :] = mask_bgr

            # Probability as normalized color map
            probability_map = y_pred[0]
            color_map = self.probability_map_to_color_map(probability_map)
            res[:, 2 * config.IMAGE_W: 3 * config.IMAGE_W, :] = color_map

            # Predicted mask
            probability_map = y_pred[0]
            mask = np.zeros_like(mask)
            probability_map = probability_map / MULTIPLIER
            mask[probability_map > 0.5] = 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            res[:, 3 * config.IMAGE_W: 4 * config.IMAGE_W, :] = mask_bgr

            image = self.summary_image(res)

            return image

        train_sample_visualization = get_visualization(self.train_generator)
        test_sample_visualization = get_visualization(self.test_generator)

        summary_train = tf.Summary(value=[tf.Summary.Value(tag='train', image=train_sample_visualization)])
        summary_test = tf.Summary(value=[tf.Summary.Value(tag='test', image=test_sample_visualization)])

        writer = tf.summary.FileWriter(EXPERIMENT_NAME + '_logs')
        writer.add_summary(summary_train, epoch)
        writer.add_summary(summary_test, epoch)
        writer.close()

        return


def mse_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss


def y_true_sum(y_true, y_pred):
    return K.sum(y_true)


def y_pred_sum(y_true, y_pred):
    return K.sum(y_pred)


def pred_min(y_true, y_pred):
    return K.min(y_pred)


def pred_max(y_true, y_pred):
    return K.max(y_pred)


def all_zeros_baseline(y_true, y_pred):
    y_pred_zeros = tf.zeros_like(y_true)
    loss = tf.reduce_mean(tf.square(y_true-y_pred_zeros))
    return loss


def get_model():
    inputs = Input((config.IMAGE_H, config.IMAGE_W, config.N_CHANNELS))

    base = get_unet(inputs, config.N_CLASSES)

    if IS_BN_ON_OUTPUT:
        x = BatchNormalization()(base)
    else:
        x = base
    x = Activation(OUTPUT_ACTIVATION)(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adadelta(), loss=mse_loss,
                  metrics=[y_true_sum, y_pred_sum, pred_min, pred_max, all_zeros_baseline])

    # print(model.summary())
    # import sys
    # sys.exit()

    return model


if __name__ == '__main__':
    model = get_model()

    callbacks = [
        TensorBoard(log_dir=EXPERIMENT_NAME+'_logs', histogram_freq=0, write_graph=True, write_images=True),
        CustomCallback(batch_generator(), batch_generator())
    ]

    history = model.fit_generator(
        generator=batch_generator(),
        epochs=config.N_EPOCHS,
        steps_per_epoch=100,
        validation_data=batch_generator(),
        validation_steps=10,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

    model.save(EXPERIMENT_NAME+'.h5')

    print('Done!')