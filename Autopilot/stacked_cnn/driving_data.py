"""
For stacked convolutional networks, the order of images must not be shuffled,
because we need consecutive image frames to predict the steering angle.
"""
import numpy as np
import random

# Initialize dataset lists for image paths and steering angles
xs = []
ys = []

# Pointers to keep track of the current position in the training and validation batches
train_batch_pointer = 0
val_batch_pointer = 0


# Load from Carla simulation dataset
# with open("../driving_dataset/carla_stacked_train.txt") as f:
#     for index, line in enumerate(f):
#         # If you want to exclude metamorphic test data, uncomment the next two lines
#         # if index > 36431:
#         #     break
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/carla_stacked_train/" + image_path)
#         ys.append(angle * np.pi / 180)

# Load from Udacity domain adaptation dataset
with open("../driving_dataset/udacity_stacked_train.txt") as f:
    for index, line in enumerate(f):
        line_values = line.split(",")[0].split()
        image_path = line_values[0]
        angle = float(line_values[1])
        xs.append("../driving_dataset/udacity_stacked_train/" + image_path)
        ys.append(angle * np.pi / 180)

# Load from California domain adaptation dataset
# with open("../driving_dataset/california_stacked_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/california_stacked_train/" + image_path)
#         ys.append(angle * np.pi / 180)

# Load from comma.ai domain adaptation dataset
# with open("../driving_dataset/comma_stacked_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append("../driving_dataset/comma_stacked_train/" + image_path)
#         ys.append(angle * np.pi / 180)

# Shuffle the dataset (image path and angle pairs)
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# Split the dataset into training and validation sets (70% train, 30% validation)
train_xs = xs[:int(len(xs) * 0.7)]
train_ys = ys[:int(len(xs) * 0.7)]

val_xs = xs[-int(len(xs) * 0.3):]
val_ys = ys[-int(len(xs) * 0.3):]

# Calculate the number of images in training and validation sets
num_train_images = len(train_xs)
num_val_images = len(val_xs)

# Function to load a batch of training data
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Use modulo to wrap around if pointer exceeds dataset size
        file_path = train_xs[(train_batch_pointer + i) % num_train_images]
        # Load stacked image from .npz file (expects key 'arr_0')
        stacked_image = np.load(file_path)['arr_0']
        x_out.append(stacked_image)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

# Function to load a batch of validation data
def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Use modulo to wrap around if pointer exceeds dataset size
        file_path = val_xs[(val_batch_pointer + i) % num_val_images]
        # Load stacked image from .npz file (expects key 'arr_0')
        stacked_image = np.load(file_path)['arr_0']
        x_out.append(stacked_image)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
