"""
Analyze the prediction performance of various ADS on large-scale observational datasets
"""
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import os
import numpy as np
import argparse
import openpyxl
from utils.make_flow_images import calculate_optical_flow
import warnings
warnings.filterwarnings('ignore')


def stack_images_in_depth(index, image_paths):
    """
    Stack images along the depth direction for use by the stacked_CNN model
    param index: index of the current frame
    param image_paths: test image paths
    """
    stacked_images = []
    if index < 2:
        full_image = cv2.imread(image_paths[index])
        image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
        for _ in range(3):
            stacked_images.append(image)
    else:
        for jdx in range(3):
            full_image = cv2.imread(image_paths[index + jdx - 2])
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            stacked_images.append(image)

    return np.dstack(stacked_images)


def stack_images(index, image_paths):
    """
    Stack every three images together for use by the CNN_GRU model
    param index: index of the current frame
    param image_paths: test image paths
    """
    stacked_images = []
    if index < 2:
        full_image = cv2.imread(image_paths[index])
        image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
        for _ in range(3):
            stacked_images.append(image)
    else:
        for jdx in range(3):
            full_image = cv2.imread(image_paths[index + jdx - 2])
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            stacked_images.append(image)

    return np.array(stacked_images)


def make_flow_images(index, image_paths):
    """
    Prepare original RGB images and optical flow images for training the two_stream_CNN model
    """
    pre_image_path = image_paths[index - 1]
    current_image_path = image_paths[index]
    current_image, flow_image = calculate_optical_flow(prev_img_path=pre_image_path,
                                                       current_img_path=current_image_path)

    return np.array(current_image), np.array(flow_image)


def cal_difference(testing_data_path, truth_angles, type):
    # Calculate the difference between the predicted steering angle and the ground truth under a certain condition
    predicted_steers = []
    actual_steers = []
    image_paths = []
    smoothed_angle = 0

    # Extract the ground truth steering angles
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(os.path.join(testing_data_path, line_values[0]))
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    # Model prediction output
    if type == 'PilotNet':
        for index in range(0, len(image_paths)):
            full_image = cv2.imread(image_paths[index])
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering angle output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'cg23':
        for index in range(0, len(image_paths)):
            full_image = cv2.imread(image_paths[index])
            image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering angle output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'cnn_gru':
        for idx in range(0, len(image_paths)):
            image = stack_images(index=idx, image_paths=image_paths)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering angle output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'stacked_cnn':
        for idx in range(0, len(image_paths)):
            image = stack_images_in_depth(index=idx, image_paths=image_paths)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'two_stream_cnn':
        # The first image has no history to refer to, so predict 0 directly
        predicted_steers.append(0)
        for idx in range(1, len(image_paths)):
            current_image, flow_image = make_flow_images(index=idx, image_paths=image_paths)
            current_image_expanded = np.expand_dims(current_image, axis=0)
            flow_image_expanded = np.expand_dims(flow_image, axis=0)
            degrees = model.output.eval(feed_dict={model.inputs[0]: current_image_expanded, model.inputs[1]: flow_image_expanded})[0][0]
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)

    # Plot the predicted and actual trajectories in the same figure
    # plt.figure(figsize=(10, 6))
    # plt.plot(predicted_steers, 'r.-', label='predict')
    # plt.plot(actual_steers, 'b.-', label='truth')
    # plt.legend(loc='best')
    # plt.title("Predicted vs Truth")
    # plt.show()

    # Print the RMSE between predicted and ground truth values
    rmse = np.sqrt(np.mean((np.array(predicted_steers) - np.array(actual_steers)) ** 2))
    # print(f'RMSE between predicted and ground truth: {rmse}')
    return rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='import ADS model and test on observational dataset')
    parser.add_argument('--model_type', type=str, default='cg23',
                        choices=['PilotNet', 'cg23', 'cnn_gru', 'stacked_cnn', 'two_stream_cnn'],
                        help='Select the model type to import')

    # Parse arguments
    args = parser.parse_args()

    # Select the model path to restore according to the flag
    if args.model_type == 'PilotNet':
        from PilotNet.model import build_pilotnet_cnn
        model = build_pilotnet_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./PilotNet/save/carla_original.ckpt")
        my_column = 1  # Column index to store in the excel file
    elif args.model_type == 'cg23':
        from cg23.model import build_cnn
        model = build_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./cg23/save/carla_original.ckpt")
        my_column = 2
    elif args.model_type == 'cnn_gru':
        from cnn_gru.model import build_conv_gru
        model = build_conv_gru()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn_gru/save/carla_original.ckpt")
        my_column = 3
    elif args.model_type == 'stacked_cnn':
        from stacked_cnn.model import build_stacked_cnn
        model = build_stacked_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./stacked_cnn/save/carla_original.ckpt")
        my_column = 4
    elif args.model_type == 'two_stream_cnn':
        from two_stream_cnn.model import build_two_stream_cnn
        model = build_two_stream_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./two_stream_cnn/save/carla_original.ckpt")
        my_column = 5
    else:
        raise ValueError('Unknown model type')

    # Create an excel file to store all observational conditions
    excel_path = os.path.join('./detection_results.xlsx')
    workbook = openpyxl.load_workbook(excel_path)
    worksheet = workbook.worksheets[0]

    # Start traversing all observational conditions
    for index in range(0, len(os.listdir('./driving_dataset/carla_observational'))):
        name = 'condition_%d' % index
        print(f"we are now testing {name}")
        images_path = os.path.join('./driving_dataset/carla_observational', name, 'images')
        # Calculate the difference between prediction results and ground truth steering values
        ground_truth_file = os.path.join('./driving_dataset/carla_observational', name, 'ground_truth_steer.txt')
        rmse = cal_difference(testing_data_path=images_path, truth_angles=ground_truth_file, type=args.model_type)
        # Write the statistics to the excel file
        worksheet.cell(index + 2, column=my_column, value=rmse)
        # Save the file
        workbook.save(excel_path)

    print("we have done all the test")
