# For stacked convolutional networks, we need to prepare consecutive image frames in advance
import numpy as np
import random
import os

# Initialize dataset lists
xs = []
ys = []

# Pointers to the end of the last batch for training and validation
train_batch_pointer = 0
val_batch_pointer = 0

# Training on simulated data
# with open("../driving_dataset/carla_seq_train.txt") as f:
#     for index, line in enumerate(f):
#         # Only read the first 36431 lines if needed, which represent the original training set
#         # if index > 36431:
#         #     break
#         line_values = line.split(",")[0].split()
#         file_name = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/carla_seq_train/" + file_name)
#         ys.append(angle * np.pi / 180)

# Training on domain-adapted Udacity dataset
with open("../driving_dataset/udacity_seq_train.txt") as f:
    for index, line in enumerate(f):
        line_values = line.split(",")[0].split()
        image_path = line_values[0]
        angle = float(line_values[1])
        xs.append(os.path.join(r'../driving_dataset/udacity_seq_train', image_path))
        ys.append(angle * np.pi / 180)

# Training on domain-adapted California dataset
# with open("../driving_dataset/california_seq_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/california_seq_train', image_path))
#         ys.append(angle * np.pi / 180)

# Training on domain-adapted comma.ai dataset
# with open("../driving_dataset/comma_seq_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join('../driving_dataset/comma_seq_train', image_path))
#         ys.append(angle * np.pi / 180)

# Shuffle the dataset
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# Split the dataset into training and validation sets (70% train, 30% val)
train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[-int(len(xs) * 0.3):]
val_ys = ys[-int(len(xs) * 0.3):]

# Calculate the number of images in training and validation sets
num_train_images = len(train_xs)
num_val_images = len(val_xs)

# Load a batch of training data
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Read the npy file from the specified position
        file_path = train_xs[(train_batch_pointer + i) % num_train_images]
        stacked_image = np.load(file_path)['arr_0']
        # Stack consecutive time_steps frames together
        x_out.append(stacked_image)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    train_batch_pointer += batch_size
    return x_out, y_out

# Load a batch of validation data
def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        file_path = val_xs[(val_batch_pointer + i) % num_val_images]
        stacked_image = np.load(file_path)['arr_0']
        x_out.append(stacked_image) 
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])

    val_batch_pointer += batch_size
    return x_out, y_out
