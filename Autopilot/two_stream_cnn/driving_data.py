"""
Load RGB + optical flow data for model training.
"""

import numpy as np
import random

# Initialize dataset
xs = []
ys = []

train_batch_pointer = 0
val_batch_pointer = 0

# Train on CARLA simulation data
# with open("../driving_dataset/carla_flow_train.txt") as f:
#     for index, line in enumerate(f):
#         # if index > 36615:  # The last 36615 samples are augmented by mutation
#         #     break
#         line_values = line.split(",")[0].split()
#         file_name = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/carla_flow_train/" + file_name)
#         ys.append(angle * np.pi / 180)

# Train on domain-adapted Udacity dataset
with open("../driving_dataset/udacity_flow_train.txt") as f:
    for index, line in enumerate(f):
        line_values = line.split(",")[0].split()
        image_path = line_values[0]
        angle = float(line_values[1])
        xs.append("../driving_dataset/udacity_flow_train/" + image_path)
        ys.append(angle * np.pi / 180)

# Train on domain-adapted California dataset
# with open("../driving_dataset/california_flow_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/california_flow_train/" + image_path)
#         ys.append(angle * np.pi / 180)

# Train on domain-adapted comma.ai dataset
# with open("../driving_dataset/comma_flow_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/comma_flow_train/" + image_path)
#         ys.append(angle * np.pi / 180)


# Shuffle the dataset
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# Split into training and validation sets
train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[-int(len(xs) * 0.3):]
val_ys = ys[-int(len(xs) * 0.3):]

# Calculate the size of training and validation sets
num_train_images = len(train_xs)
num_val_images = len(val_xs)


# Load a batch of training data
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    rgb_out = []
    flow_out = []
    y_out = []
    for i in range(0, batch_size):
        # Load npy file from the specified path
        file_path = train_xs[(train_batch_pointer + i) % num_train_images]
        rgb_image, flow_image = np.load(file_path)['arr_0'][0], np.load(file_path)['arr_0'][1]
        rgb_out.append(rgb_image)
        flow_out.append(flow_image)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    train_batch_pointer += batch_size
    return rgb_out, flow_out, y_out


# Load a batch of validation data
def LoadValBatch(batch_size):
    global val_batch_pointer
    rgb_out = []
    flow_out = []
    y_out = []
    for i in range(0, batch_size):
        file_path = val_xs[(val_batch_pointer + i) % num_val_images]
        rgb_image, flow_image = np.load(file_path)['arr_0'][0], np.load(file_path)['arr_0'][1]
        rgb_out.append(rgb_image)
        flow_out.append(flow_image)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])

    val_batch_pointer += batch_size
    return rgb_out, flow_out, y_out
