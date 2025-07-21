import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from model import build_pilotnet_cnn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_dataset(testing_data_path):
    """
    Run inference on the test dataset and write predicted steering angles to a file.
    Args:
        testing_data_path (str): Path to the directory containing test images.
    """
    # Read a sample steering wheel image for shape reference (not used in prediction)
    img = cv2.imread('../steering_wheel_image.jpg', 0)
    rows, cols = img.shape
    smoothed_angle = 0

    # Collect all image filenames with supported extensions
    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    # Sort images numerically by filename (assumes filenames are numbers)
    images.sort(key=lambda x: int(os.path.splitext(x)[0])) 

    # Remove previous prediction file if exists
    if os.path.exists('../predicted_steers.txt'):
        os.remove(r'../predicted_steers.txt')

    # Write current predictions to predicted_steers.txt
    with open('../predicted_steers.txt', 'a') as f:
        for _, filename in tqdm(enumerate(images)):
            # Read and preprocess image
            full_image = cv2.imread(os.path.join(testing_data_path, filename))
            # Crop and resize image to model input size, normalize to [0, 1]
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            # Predict steering angle (in radians), convert to degrees
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0] * 180.0 / np.pi
            # Smooth the predicted angle for visualization
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            # Write filename and smoothed angle to file
            f.write(f'{filename} {smoothed_angle}\n')

    cv2.destroyAllWindows()

def plot_comparative_curves(testing_data_path, truth_angles):
    """
    Plot predicted steering angles vs. ground truth and compute error metrics.
    Args:
        testing_data_path (str): Path to the directory containing test images.
        truth_angles (str): Path to the file containing ground truth steering angles.
    """
    predicted_steers = []
    actual_steers = []
    image_paths = []

    # Read ground truth steering angles and corresponding image filenames
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(line_values[0])
            # Convert ground truth angle from degrees to radians
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    smoothed_angle = 0
    differences = []

    # Run model prediction for each image and compare with ground truth
    for index, filename in enumerate(tqdm(image_paths)):
        full_image = cv2.imread(os.path.join(testing_data_path, filename))
        image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
        degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
        # Smooth the predicted angle for visualization
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        predicted_steers.append(smoothed_angle)
        # Record absolute difference between prediction and ground truth
        differences.append(abs(smoothed_angle - actual_steers[index]))

    # Plot predicted and actual steering angles on the same figure
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_steers, 'r.-', label='predict')
    plt.plot(actual_steers, 'b.-', label='truth')
    plt.legend(loc='best')
    plt.title("Predicted vs Truth")
    plt.show()

    # Calculate and print RMSE between predictions and ground truth
    rmse = np.sqrt(np.mean((np.array(predicted_steers) - np.array(actual_steers)) ** 2))
    print(f'RMSE between predictions and ground truth: {rmse}')

    # Calculate and print median absolute error
    mae = np.median(np.abs(np.array(predicted_steers) - np.array(actual_steers)))
    print(f'Median absolute error between predictions and ground truth: {mae}')

if __name__ == '__main__':
    # Restore the trained model from checkpoint
    model = build_pilotnet_cnn()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/udacity.ckpt")
    tf.keras.backend.set_learning_phase(0)
    image_path = '../driving_dataset/udacity_test'  
    ground_truth_file = '../driving_dataset/udacity_test.txt'

    # Run inference on the test set
    run_dataset(testing_data_path=image_path)

    # Plot comparison between predicted and ground truth steering angles
    plot_comparative_curves(testing_data_path=image_path, truth_angles=ground_truth_file)
