import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from model import build_conv_gru
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def stack_images(index, images, image_path):
    """
    Stack three consecutive images for temporal input to the model.

    Args:
        index (int): Index of the current frame.
        images (list): List of image filenames.
        image_path (str): Path to the directory containing test images.

    Returns:
        tuple: (stacked_images, full_image)
            stacked_images: numpy array of stacked images (shape: [3, 66, 200, 3])
            full_image: the last full image read (for visualization if needed)
    """
    stacked_images = []
    full_image = None

    if index < 2:
        # If there are not enough previous frames, repeat the current frame three times
        full_image = cv2.imread(os.path.join(image_path, images[index]))
        image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
        for _ in range(3):
            stacked_images.append(image)
    else:
        # Stack the current frame and the previous two frames
        for jdx in range(3):
            full_image = cv2.imread(os.path.join(image_path, images[index + jdx - 2]))
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            stacked_images.append(image)

    return np.array(stacked_images), full_image

def run_dataset(testing_data_path):
    """
    Run the model on the test dataset and write predictions to a file.

    Args:
        testing_data_path (str): Path to the test images directory.
    """
    smoothed_angle = 0

    # Get all image filenames in the test directory
    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort by frame index

    # Remove previous prediction file if exists
    if os.path.exists('../predicted_steers.txt'):
        os.remove('../predicted_steers.txt')

    # Write predictions to file
    with open('../predicted_steers.txt', 'a') as f:
        for idx, filename in enumerate(tqdm(images)):
            # Prepare stacked images for the current frame
            image, full_image = stack_images(index=idx, images=images, image_path=testing_data_path)
            # Predict steering angle (in degrees)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0] * 180.0 / np.pi
            # Apply smoothing to the predicted angle
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            f.write(f'{filename} {smoothed_angle}\n')

    cv2.destroyAllWindows()

def plot_comparative_curves(testing_data_path, truth_angles):
    """
    Plot predicted steering angles vs. ground truth and compute RMSE.

    Args:
        testing_data_path (str): Path to the test images directory.
        truth_angles (str): Path to the file containing ground truth steering angles.
    """
    predicted_steers = []
    actual_steers = []
    image_paths = []

    # Read ground truth steering angles
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(line_values[0])
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    # Get all image filenames in the test directory
    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0])) 

    # Predict steering angles for each frame
    smoothed_angle = 0
    for idx, filename in enumerate(tqdm(image_paths)):
        image, _ = stack_images(index=idx, images=images, image_path=testing_data_path)
        degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
        # Apply smoothing to the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        predicted_steers.append(smoothed_angle)

    # Plot predicted and actual steering angles on the same figure
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_steers, 'r.-', label='predict')
    plt.plot(actual_steers, 'b.-', label='truth')
    plt.legend(loc='best')
    plt.title("Predicted vs Truth")
    plt.show()

    # Print RMSE between predictions and ground truth
    rmse = np.sqrt(np.mean((np.array(predicted_steers) - np.array(actual_steers)) ** 2))
    print(f'RMSE between predictions and ground truth: {rmse}')

if __name__ == '__main__':
    # Restore the trained model from checkpoint
    model = build_conv_gru()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/carla_original.ckpt")
    image_path = '../driving_dataset/udacity_test'  

    # Run the test set and write predictions
    run_dataset(testing_data_path=image_path)

    # Plot comparison between predicted and ground truth steering angles
    ground_truth_file = '../driving_dataset/udacity_test.txt'
    plot_comparative_curves(testing_data_path=image_path, truth_angles=ground_truth_file)
