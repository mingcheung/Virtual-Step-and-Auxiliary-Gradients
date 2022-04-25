import os
import numpy as np
from absl import app
import tensorflow as tf
from imageio import imsave
from utils import load_data

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet152V2


def save_images(images, filenames, save_dir):
    for i, filename in enumerate(filenames):
        print("Saving file {}...".format(filename))
        save_path = os.path.join(save_dir, filename)
        imsave(save_path, images[i, :, :, :], format="png")


def input_diversity(input_tensor, opened=False):
    if opened:
        image_width = 299
        image_resize = 330
        prob = 0.5
        rnd = tf.random.uniform((), image_width, image_resize, dtype=tf.int32)
        rescaled = tf.image.resize(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = tf.random.uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        pad_left = tf.random.uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
        padded.set_shape((input_tensor.shape[0], image_resize, image_resize, 3))
        return tf.cond(tf.random.uniform(shape=[1])[0] < tf.constant(prob), lambda: padded, lambda: input_tensor)
    else:
        return input_tensor


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()

    return kernel


kernel = gkern(15, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

using_translation_invariant = False    # the on-off flag for translation invariant attack.


def fgsm_attacks_with_push_and_pull(model_li, eps, num_iters, loss_fn, x,
                                    alpha=0.006, aux_num=7, clip_min=0., clip_max=1.):
    x = tf.cast(x, tf.float32)
    x_clean = x
    num_classes = 1000

    predicted_labels = tf.argmax(model_li[0](x), axis=1)
    y = tf.one_hot(predicted_labels, num_classes)

    for _ in range(num_iters):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            predicted_logits = 0
            for model in model_li:
                predicted_logits += model(input_diversity(x, opened=False))  # the on-off flag for diverse input attack.
            predicted_logits = predicted_logits / len(model_li)

            losses = []

            # Calculating the main loss.
            loss = loss_fn(labels=y, logits=predicted_logits)
            losses.append(loss)

            # Calculating the auxiliary loss.
            labels_woy = np.delete(np.arange(num_classes), np.argmax(y, axis=1))
            aux_labels = np.random.choice(labels_woy, aux_num, replace=False)
            for aux_label in aux_labels:
                aux_loss = -loss_fn(labels=tf.one_hot(aux_label, num_classes), logits=predicted_logits)
                losses.append(aux_loss)

        grads = []
        for loss in losses:
            grads.append(tape.gradient(loss, x))

        for grad in grads:
            if using_translation_invariant:
                grad = tf.nn.depthwise_conv2d(grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
            x = x + alpha * tf.sign(grad)
            x = tf.clip_by_value(x, clip_min, clip_max)

    perturbations = x - x_clean
    x = x_clean + tf.clip_by_value(perturbations, -eps, eps)

    return x


def main(_):
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # InceptionV3
    inc_v3 = InceptionV3(weights="imagenet", include_top=True,
                         input_shape=(299, 299, 3),
                         pooling="avg",
                         classifier_activation=None)
    inc_v3.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=["accuracy"])

    # Xception
    xcep = Xception(weights="imagenet", include_top=True,
                    input_shape=(299, 299, 3),
                    pooling="avg",
                    classifier_activation=None)
    xcep.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"])

    # InceptionResNetV2
    incres_v2 = InceptionResNetV2(weights="imagenet", include_top=True,
                                  input_shape=(299, 299, 3),
                                  pooling="avg",
                                  classifier_activation=None)
    incres_v2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

    # ResNet152V2
    res152_v2 = ResNet152V2(weights="imagenet", include_top=True,
                            input_shape=(299, 299, 3),
                            pooling="avg",
                            classifier_activation=None)
    res152_v2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

    # model_li = [inc_v3, xcep, incres_v2]
    # model_li = [inc_v3, xcep, res152_v2]
    # model_li = [inc_v3, incres_v2, res152_v2]
    # model_li = [xcep, incres_v2, res152_v2]
    model_li = [inc_v3, xcep, incres_v2, res152_v2]

    images, true_labels, target_labels, filenames = load_data("data/images")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits

    for i in range(images.shape[0]):
        adv_image = fgsm_attacks_with_push_and_pull(model_li, eps=16/255., num_iters=20,
                                                    loss_fn=cross_entropy, x=images[i:i+1, :, :, :],
                                                    alpha=0.007, aux_num=0)


if __name__ == '__main__':
    app.run(main)
