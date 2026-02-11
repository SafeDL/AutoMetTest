import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from model import build_stacked_cnn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def stack_images(index, images, image_path):
    # param index: index of the current frame
    # param images: list of image filenames
    # param image_path: path to the test images

    stacked_images = []
    full_image = None

    if index < 2:
        full_image = cv2.imread(os.path.join(image_path, images[index]))
        image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
        # If less than three frames are available, repeat the current frame three times
        for _ in range(3):
            stacked_images.append(image)
    else:
        for jdx in range(3):
            full_image = cv2.imread(os.path.join(image_path, images[index + jdx - 2]))
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            stacked_images.append(image)

    return np.dstack(stacked_images), full_image


def run_dataset(testing_data_path):
    # Given the path to the test set, read images and test the model
    img = cv2.imread('../steering_wheel_image.jpg', 0)
    rows, cols = img.shape
    smoothed_angle = 0

    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0])) 

    if os.path.exists('../predicted_steers.txt'):
        os.remove('../predicted_steers.txt')

    # Write the current prediction results to predicted_steers.txt
    with open('../predicted_steers.txt', 'a') as f:
        for idx, filename in enumerate(tqdm(images)):
            image, full_image = stack_images(index=idx, images=images, image_path=testing_data_path)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0] * 180.0 / np.pi
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            f.write(f'{filename} {smoothed_angle}\n')


def plot_comparative_curves(testing_data_path, truth_angles):
    predicted_steers = []
    actual_steers = []
    image_paths = []

    # Extract ground truth steering angles
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(line_values[0])
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0])) 

    # Predict model outputs
    smoothed_angle = 0
    for idx, filename in enumerate(tqdm(image_paths)):
        image, _ = stack_images(index=idx, images=images, image_path=testing_data_path)
        degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
        # Smooth the steering angle output
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        predicted_steers.append(smoothed_angle)

    # Plot predicted and actual trajectories in the same figure
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_steers, 'r.-', label='predict')
    plt.plot(actual_steers, 'b.-', label='truth')
    plt.legend(loc='best')
    plt.title("Predicted vs Truth")
    plt.show()

    # Print RMSE between predictions and ground truth
    rmse = np.sqrt(np.mean((np.array(predicted_steers) - np.array(actual_steers)) ** 2))
    print(f'RMSE between predictions and ground truth: {rmse}')

    # Print median absolute error between predictions and ground truth
    mae = np.median(np.abs(np.array(predicted_steers) - np.array(actual_steers)))
    print(f'Median absolute error between predictions and ground truth: {mae}')


if __name__ == '__main__':
    # Restore the model
    model = build_stacked_cnn()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/udacity.ckpt")
    # Path to the images to be tested
    image_path = '../driving_dataset/udacity_test'

    # Run the test set
    run_dataset(testing_data_path=image_path)

    # Plot comparison between predicted and ground truth curves
    ground_truth_file = '../driving_dataset/udacity_test.txt'
    plot_comparative_curves(testing_data_path=image_path, truth_angles=ground_truth_file)
