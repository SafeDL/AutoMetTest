import cv2
import random
import numpy as np
import os

xs = []
ys = []

train_batch_pointer = 0
val_batch_pointer = 0

# (1) Train on CARLA simulation dataset
# with open("../driving_dataset/carla_train.txt") as f:
#     for index, line in enumerate(f):
#         # The first 36799 samples are original CARLA data, the rest are augmented data
#         # if index > 36799:
#         #     break
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/carla_train', image_path))
#         ys.append(angle * np.pi / 180)

# (2) Train on domain-adapted Udacity dataset
with open("../driving_dataset/udacity_train.txt") as f:
    for index, line in enumerate(f):
        line_values = line.split(",")[0].split()
        image_path = line_values[0]
        angle = float(line_values[1])
        xs.append(os.path.join('../driving_dataset/udacity_train', image_path))
        ys.append(angle * np.pi / 180)

# (3) Train on domain-adapted California dataset
# with open("../driving_dataset/california_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/california_train', image_path))
#         ys.append(angle * np.pi / 180)

# (4) Train on domain-adapted comma.ai dataset
# with open("../driving_dataset/comma_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/comma_train', image_path))
#         ys.append(angle * np.pi / 180)

# Shuffle the dataset
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[-int(len(xs) * 0.3):]
val_ys = ys[-int(len(xs) * 0.3):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(
            cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-200:], (128, 128)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-200:], (128, 128)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
