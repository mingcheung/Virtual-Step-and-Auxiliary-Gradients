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


def fgsm_attacks(model_fn, eps, num_iters, loss_fn, x, clip_min=0., clip_max=1.):
    alpha = eps / num_iters
    x = tf.cast(x, tf.float32)
    num_classes = 1000

    predicted_labels = tf.argmax(model_fn(x), axis=1)
    y = tf.one_hot(predicted_labels, num_classes)

    for _ in range(num_iters):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            predicted_logits = model_fn(input_diversity(x, opened=False))   # the on-off flag for diverse input attack.

            # Calculating the loss.
            loss = loss_fn(labels=y, logits=predicted_logits)

        grad = tape.gradient(loss, x)

        if using_translation_invariant:
            grad = tf.nn.depthwise_conv2d(grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
        x = x + alpha * tf.sign(grad)
        x = tf.clip_by_value(x, clip_min, clip_max)

    return x


def main(_):
    # InceptionV3, Xception, InceptionResNetV2, ResNet152V2
    model = ResNet152V2(weights="imagenet", include_top=True,
                        input_shape=(299, 299, 3),
                        pooling="avg",
                        classifier_activation=None)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    images, true_labels, target_labels, filenames = load_data("data/images")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits

    for i in range(images.shape[0]):
        adv_image = fgsm_attacks(model, eps=16/255., num_iters=20,
                                 loss_fn=cross_entropy, x=images[i:i+1, :, :, :])


if __name__ == '__main__':
    app.run(main)
