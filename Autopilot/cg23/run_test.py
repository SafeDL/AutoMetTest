import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from model import build_cnn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_dataset(testing_data_path):
    """
    Given the path to the test set, read images and evaluate the model.
    The predicted steering angles are written to 'predicted_steers.txt'.
    """
    smoothed_angle = 0
    # Get all image files in the test directory
    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    # Sort images by filename (assumes filenames are numeric)
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  

    # Remove previous prediction file if exists
    if os.path.exists('../predicted_steers.txt'):
        os.remove('../predicted_steers.txt')

    # Write current predictions to 'predicted_steers.txt'
    with open('../predicted_steers.txt', 'a') as f:
        for _, filename in tqdm(enumerate(images)):
            full_image = cv2.imread(os.path.join(testing_data_path, filename))
            # Crop and resize the image as required by the model
            image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
            # Predict steering angle (in degrees)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0] * 180.0 / np.pi
            # Smooth the steering angle output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            f.write(f'{filename} {smoothed_angle}\n')

def plot_comparative_curves(testing_data_path, truth_angles):
    """
    Plot the predicted steering angles and ground truth angles for comparison.
    Also computes and prints the RMSE between prediction and ground truth.
    """
    predicted_steers = []
    actual_steers = []
    image_paths = []

    # Extract ground truth steering angles from the label file
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(line_values[0])
            # Convert ground truth angle from degrees to radians
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    # Predict steering angles for each image in the test set
    smoothed_angle = 0
    for _, filename in enumerate(tqdm(image_paths)):
        full_image = cv2.imread(os.path.join(testing_data_path, filename))
        image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
        degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
        # Smooth the steering angle output
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

    # Print RMSE between predicted and actual steering angles
    rmse = np.sqrt(np.mean((np.array(predicted_steers) - np.array(actual_steers)) ** 2))
    print(f'RMSE between predicted and ground truth: {rmse}')

if __name__ == '__main__':
    # Restore the trained model
    model = build_cnn()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "./save/udacity.ckpt")
    # Path to the test images
    image_path = '../driving_dataset/udacity_test'

    # Run the test set and write predictions
    run_dataset(testing_data_path=image_path)

    # Plot comparison between predicted and ground truth steering angles
    ground_truth_file = '../driving_dataset/udacity_test.txt'
    plot_comparative_curves(testing_data_path=image_path, truth_angles=ground_truth_file)
