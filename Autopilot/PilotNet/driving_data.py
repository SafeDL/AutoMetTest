import cv2
import random
import os
import numpy as np

xs = []
ys = []

# Pointers to the end of the last batch for training and validation
train_batch_pointer = 0
val_batch_pointer = 0

# (1) Training on CARLA simulation dataset
# with open("../driving_dataset/carla_train.txt") as f:
#     for index, line in enumerate(f):
#         # The first 36799 samples are original CARLA data, the rest are augmented data
#         # If you want to use only the original data, uncomment the following lines:
#         # if index > 36799:
#         #     break
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/carla_train/" + image_path)
#         # Convert angle from degrees to radians
#         ys.append(angle * np.pi / 180)

# (2) Training on domain-adapted Udacity dataset
with open("../driving_dataset/udacity_train.txt") as f:
    for index, line in enumerate(f):
        line_values = line.split(",")[0].split()
        image_path = line_values[0]
        angle = float(line_values[1])
        xs.append(os.path.join('../driving_dataset/udacity_train', image_path))
        ys.append(angle * np.pi / 180)

# (3) Training on domain-adapted California dataset
# with open("../driving_dataset/california_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/california_train', image_path))
#         ys.append(angle * np.pi / 180)

# (4) Training on domain-adapted comma.ai dataset
# with open("../driving_dataset/comma_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/comma_train', image_path))
#         ys.append(angle * np.pi / 180)

num_images = len(xs)

# Shuffle the list of images and corresponding steering angles
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# Split the dataset into training (70%) and validation (30%) sets
train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[-int(len(xs) * 0.3):]
val_ys = ys[-int(len(xs) * 0.3):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    """
    Load a batch of training data.

    Args:
        batch_size (int): Number of samples in the batch.

    Returns:
        tuple: (x_out, y_out)
            x_out: List of preprocessed images (shape: [batch_size, 66, 200, 3])
            y_out: List of steering angles (shape: [batch_size, 1])
    """
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Read and preprocess image: crop, resize, normalize
        x_out.append(
            cv2.resize(
                cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-200:], 
                (200, 66)
            ) / 255.0
        )
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    """
    Load a batch of validation data.

    Args:
        batch_size (int): Number of samples in the batch.

    Returns:
        tuple: (x_out, y_out)
            x_out: List of preprocessed images (shape: [batch_size, 66, 200, 3])
            y_out: List of steering angles (shape: [batch_size, 1])
    """
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Read and preprocess image: crop, resize, normalize
        x_out.append(
            cv2.resize(
                cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-200:], 
                (200, 66)
            ) / 255.0
        )
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
