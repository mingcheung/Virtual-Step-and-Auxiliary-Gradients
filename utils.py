import os
import math
import numpy as np
import pandas as pd
from imageio import imread
from tensorflow.keras.utils import to_categorical


def load_data(input_dir, raw=False, suffix=".png"):
    bulletin = pd.read_csv("data/dev_dataset.csv")
    image_ids = bulletin["ImageId"] + suffix
    true_labels = bulletin["TrueLabel"] - 1
    target_classes = bulletin["TargetClass"] - 1

    images = []
    for i in range(len(image_ids)):
        img = imread(os.path.join(input_dir, image_ids[i])).astype(np.int32)
        images.append(np.expand_dims(img, 0))

    images = np.vstack(images)
    labels = np.expand_dims(np.array(true_labels), 1).astype(np.int32)
    targets = np.expand_dims(np.array(target_classes), 1).astype(np.int32)
    filenames = np.array(image_ids)

    num_classes = 1000
    if not raw:
        images = images / 255.0
        labels = to_categorical(labels, num_classes)
        targets = to_categorical(targets, num_classes)

    return images, labels, targets, filenames


def compute_accuracy(preds, labels, offset=0):
    num_correct = np.sum(np.argmax(preds, axis=1)-offset == np.argmax(labels, axis=1))
    acc = num_correct / preds.shape[0]

    return acc


def to_batches(inputs, batch_size):
    batches = []
    n = inputs.shape[0]
    num_batch = math.ceil(n / batch_size)
    for i in range(num_batch):
        batches.append(inputs[i*batch_size: (i+1)*batch_size])

    return batches
