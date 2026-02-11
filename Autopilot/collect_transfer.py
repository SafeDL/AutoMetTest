"""
Test on real datasets (udacity, california, comma.ai) according to whether the model has undergone domain adaptation with real data,
and collect their performance metrics.
"""
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import os
import numpy as np
import argparse
import openpyxl
import matplotlib.pyplot as plt
from utils.make_flow_images import calculate_optical_flow
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def stack_images_in_depth(index, image_paths):
    """
    Stack images along the depth direction for the stacked_CNN model.
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
    Stack every three images together for the CNN_GRU model.
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
    Prepare original RGB images and optical flow images for two_stream_CNN model training.
    """
    pre_image_path = image_paths[index - 1]
    current_image_path = image_paths[index]
    current_image, flow_image = calculate_optical_flow(prev_img_path=pre_image_path,
                                                       current_img_path=current_image_path)

    return np.array(current_image), np.array(flow_image)


def cal_difference(testing_data_path, truth_angles, type):
    # Evaluate the model's performance on the real test set after domain adaptation training.
    predicted_steers = []
    actual_steers = []
    image_paths = []
    smoothed_angle = 0

    # Extract ground truth steering angles
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(os.path.join(testing_data_path, line_values[0]))
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    # Predict model outputs
    if type == 'PilotNet':
        for index in tqdm(range(0, len(image_paths))):
            full_image = cv2.imread(image_paths[index])
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            degrees = fine_tune_model.output.eval(feed_dict={fine_tune_model.input: [image]})[0][0]
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'cg23':
        for index in tqdm(range(0, len(image_paths))):
            # Do not visualize the test process
            full_image = cv2.imread(image_paths[index])
            image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
            degrees = fine_tune_model.output.eval(feed_dict={fine_tune_model.input: [image]})[0][0]
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
                    degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'cnn_gru':
        for idx in tqdm(range(0, len(image_paths))):
            image = stack_images(index=idx, image_paths=image_paths)
            degrees = fine_tune_model.output.eval(feed_dict={fine_tune_model.input: [image]})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'stacked_cnn':
        for idx in tqdm(range(0, len(image_paths))):
            image = stack_images_in_depth(index=idx, image_paths=image_paths)
            degrees = fine_tune_model.output.eval(feed_dict={fine_tune_model.input: [image]})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'two_stream_cnn':
        # The first image has no historical reference, so predict 0 directly
        predicted_steers.append(0)
        for idx in tqdm(range(1, len(image_paths))):
            current_image, flow_image = make_flow_images(index=idx, image_paths=image_paths)
            current_image_expanded = np.expand_dims(current_image, axis=0)
            flow_image_expanded = np.expand_dims(flow_image, axis=0)
            degrees = fine_tune_model.output.eval(
                feed_dict={fine_tune_model.inputs[0]: current_image_expanded,
                           fine_tune_model.inputs[1]: flow_image_expanded})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)

    # Plot predicted and actual paths in the same figure
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

    return predicted_steers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import TensorFlow model from different paths according to the flag')
    parser.add_argument('--model_type', type=str, default='cg23',
                        choices=['PilotNet', 'cg23', 'cnn_gru', 'stacked_cnn', 'two_stream_cnn'],
                        help='Select the model type to import')

    args = parser.parse_args()

    # Select the model path to restore according to the flag
    if args.model_type == 'PilotNet':
        from PilotNet.model import build_pilotnet_cnn, build_finetune_pilotnet

        pretrained_model = build_pilotnet_cnn(weights_path="./PilotNet/save/carla_original.ckpt")
        fine_tune_model = build_finetune_pilotnet(pretrained_model)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./PilotNet/save/carla_original2udacity.ckpt")
        my_column = 1  
    elif args.model_type == 'cg23':
        from cg23.model import build_cnn, build_finetune_model

        pretrained_model = build_cnn(weights_path="./cg23/save/carla_original.ckpt")
        fine_tune_model = build_finetune_model(pretrained_model)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./cg23/save/carla_original2udacity.ckpt")
        my_column = 2
    elif args.model_type == 'cnn_gru':
        from cnn_gru.model import build_conv_gru, build_finetune_model

        pretrained_model = build_conv_gru(weights_path="./cnn_gru/save/carla_original.ckpt")
        fine_tune_model = build_finetune_model(pretrained_model)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./cnn_gru/save/carla_original2udacity.ckpt")
        my_column = 3
    elif args.model_type == 'stacked_cnn':
        from stacked_cnn.model import build_stacked_cnn, build_finetune_stacked_cnn

        pretrained_model = build_stacked_cnn(weights_path="./stacked_cnn/save/carla_original.ckpt")
        fine_tune_model = build_finetune_stacked_cnn(pretrained_model)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./stacked_cnn/save/carla_original2udacity.ckpt")
        my_column = 4
    elif args.model_type == 'two_stream_cnn':
        from two_stream_cnn.model import build_two_stream_cnn, build_fine_tuned_two_stream_cnn

        pretrained_model = build_two_stream_cnn(weights_path="./two_stream_cnn/save/carla_original.ckpt")
        fine_tune_model = build_fine_tuned_two_stream_cnn(pretrained_model)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "./two_stream_cnn/save/carla_original2udacity.ckpt")
        my_column = 5
    else:
        raise ValueError('Unknown model type')

    # Create an Excel file to store all observed conditions
    # NOTE: When evaluating, pay attention to modify the output activation range in the corresponding ADS model
    excel_path = os.path.join('./transfer_results.xlsx')
    workbook = openpyxl.load_workbook(excel_path)
    worksheet = workbook.worksheets[0]

    # Read ground truth steering angles and test images
    ground_truth_file = os.path.join('./driving_dataset', 'udacity_test.txt')
    images_path = './driving_dataset/udacity_test'
    predicted_steers = cal_difference(testing_data_path=images_path, truth_angles=ground_truth_file,
                                      type=args.model_type)

    # Write predicted_steers to the Excel file
    for idx, steer in enumerate(predicted_steers):
        worksheet.cell(row=idx + 2, column=my_column, value=steer)
    workbook.save(excel_path)

    print("we have done all the test")
