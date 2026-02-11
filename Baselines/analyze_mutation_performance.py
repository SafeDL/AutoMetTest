"""
Analyze the impact of Metamorphic Testing baselines on autonomous driving models.
1. Count the number of Inconsistent Behaviors.
2. Calculate steering smoothness.
"""
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import os
import numpy as np
import openpyxl
from Autopilot.utils.make_flow_images import calculate_optical_flow
import argparse
import warnings

warnings.filterwarnings('ignore')


def stack_images_in_depth(index, image_paths):
    """
    Stack images along the depth dimension for the stacked_CNN model.
    param index: Index of the current frame.
    param image_paths: Paths to the test images.
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
    param index: Index of the current frame.
    param image_paths: Paths to the test images.
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
    Prepare original RGB images and optical flow images for training the two_stream_CNN model.
    """
    pre_image_path = image_paths[index - 1]
    current_image_path = image_paths[index]
    current_image, flow_image = calculate_optical_flow(prev_img_path=pre_image_path,
                                                       current_img_path=current_image_path)

    return np.array(current_image), np.array(flow_image)


def cal_difference(testing_data_path, truth_angles, type):
    # Calculate the difference between predicted and actual steering angles under a certain condition.
    predicted_steers = []
    actual_steers = []
    image_paths = [] 
    smoothed_angle = 0

    # Extract actual steering angles.
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(os.path.join(testing_data_path, line_values[0]))
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    # Predict model outputs.
    if type == 'PilotNet':
        for index in range(0, len(image_paths)):
            full_image = cv2.imread(image_paths[index])
            image = cv2.resize(full_image[-200:], (200, 66)) / 255.0
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'cg23':
        for index in range(0, len(image_paths)):
            full_image = cv2.imread(image_paths[index])
            image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'cnn_gru':
        for idx in range(0, len(image_paths)):
            image = stack_images(index=idx, image_paths=image_paths)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'stacked_cnn':
        for idx in range(0, len(image_paths)):
            image = stack_images_in_depth(index=idx, image_paths=image_paths)
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)
    elif type == 'two_stream_cnn':
        # The first image has no history to refer to, so predict 0 directly.
        predicted_steers.append(0)
        for idx in range(1, len(image_paths)):
            current_image, flow_image = make_flow_images(index=idx, image_paths=image_paths)
            current_image_expanded = np.expand_dims(current_image, axis=0)
            flow_image_expanded = np.expand_dims(flow_image, axis=0)
            degrees = model.output.eval(
                feed_dict={model.inputs[0]: current_image_expanded, model.inputs[1]: flow_image_expanded})[0][0]
            # Smooth steering output
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            predicted_steers.append(smoothed_angle)

    return actual_steers, predicted_steers


if __name__ == '__main__':
    # NOTE: Please modify the activation function range of the output layer in the corresponding model.py file.
    parser = argparse.ArgumentParser(description='Select the search method: such as TACTIC, MUNIT, DeepRoad')
    parser.add_argument('--search_type', type=str, default='CycleGAN', help='such as: MUNIT/TACTIC/DeepRoad/DeepTest/CycleGAN')
    parser.add_argument('--model_type', type=str, default='cg23',
                        choices=['PilotNet', 'cg23', 'cnn_gru', 'stacked_cnn', 'two_stream_cnn'],
                        help='Select the model type to import')

    args = parser.parse_args()

    if args.model_type == 'PilotNet':
        from Autopilot.PilotNet.model import build_pilotnet_cnn

        model = build_pilotnet_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "../Autopilot/PilotNet/save/carla_original.ckpt")
    elif args.model_type == 'cg23':
        from Autopilot.cg23.model import build_cnn

        model = build_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "../Autopilot/cg23/save/carla_original.ckpt")
    elif args.model_type == 'cnn_gru':
        from Autopilot.cnn_gru.model import build_conv_gru

        model = build_conv_gru()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "../Autopilot/cnn_gru/save/carla_original.ckpt")
    elif args.model_type == 'stacked_cnn':
        from Autopilot.stacked_cnn.model import build_stacked_cnn

        model = build_stacked_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "../Autopilot/stacked_cnn/save/carla_original.ckpt")
    elif args.model_type == 'two_stream_cnn':
        from Autopilot.two_stream_cnn.model import build_two_stream_cnn

        model = build_two_stream_cnn()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "../Autopilot/two_stream_cnn/save/carla_original.ckpt")
    else:
        raise ValueError('Unknown model type')

    # Find the original environmental conditions and mutated environmental types to be tested.
    original_conditions = [1, 3, 45, 69, 70, 71, 72, 75, 76, 77, 80, 81, 82, 83, 85, 86, 87, 88,
                           89, 90, 95, 98, 115, 120, 123, 127, 130, 131, 133, 134, 136, 165]

    # Just for TACTIC\MUNIT\DeepRoad\CycleGAN
    mutate_conditions = ['cloudy', 'dawndusk', 'overcast', 'foggy', 'rainy', 'night', 'night_rainy', 'night_fog']

    # Just for DeepTest
    # mutate_conditions = ['image_translation', 'image_shear', 'image_scale', 'image_rotation',
    #                      'image_contrast', 'image_brightness', 'image_blur', 'image_add_noise']

    # Specify the path to store results
    # excel_path = os.path.join('../Metamorphic/mutation_results.xlsx')
    # workbook = openpyxl.load_workbook(excel_path)
    # if args.search_type == 'TACTIC':
    #     worksheet = workbook.worksheets[8]
    # elif args.search_type == 'MUNIT':
    #     worksheet = workbook.worksheets[9]
    # elif args.search_type == 'DeepRoad':
    #     worksheet = workbook.worksheets[10]
    # elif args.search_type == 'DeepTest':
    #     worksheet = workbook.worksheets[11]
    # elif args.search_type == 'CycleGAN':
    #     worksheet = workbook.worksheets[12]
    # else:
    #     raise ValueError("search type cannot be found, please check the path")

    # Initialize statistics variables
    num_of_inconsistency_1 = 0  # Count of Inconsistent Behaviors in 0°~5°
    num_of_inconsistency_2 = 0  # Count of Inconsistent Behaviors in 5°~10°
    num_of_inconsistency_3 = 0  # Count of Inconsistent Behaviors in 10°~15°
    num_of_inconsistency_4 = 0  # Count of Inconsistent Behaviors >15°
    dP_dT_list = []

    for idx, mutate_condition in enumerate(mutate_conditions):  
        for jdx, condition in enumerate(original_conditions):
            name = 'condition_%d' % condition
            print(f"we are now testing mutated {name}")

            # First: Call the original model to record predictions on the original images.
            ground_truth_file = os.path.join('../Autopilot/driving_dataset/carla_collect', name, 'ground_truth_steer.txt')
            original_images_path = os.path.join(f'../Autopilot/driving_dataset/carla_collect/{name}/images')
            _, original_predicted_steers = cal_difference(testing_data_path=original_images_path, type=args.model_type, truth_angles=ground_truth_file)

            if args.search_type == 'TACTIC':
                images_path = os.path.join('./TACTIC/test_outputs/%s_tactic/condition_%d' % (mutate_condition, condition))
            elif args.search_type == 'MUNIT':
                images_path = os.path.join('./TACTIC/test_outputs/%s_munit/condition_%d' % (mutate_condition, condition))
            elif args.search_type == 'DeepRoad':
                images_path = os.path.join('./TACTIC/test_outputs/%s_deeproad/condition_%d' % (mutate_condition, condition))
            elif args.search_type == 'DeepTest':
                images_path = os.path.join('./DeepTest/test_outputs/%s/condition_%d' % (mutate_condition, condition))
            elif args.search_type == 'CycleGAN':
                images_path = os.path.join('./CycleGAN/%s/condition_%d' % (mutate_condition, condition))
            else:
                raise ValueError("search type must can not be found, please check the path")

            # Read the original prediction outputs and compare with them.
            _, predicted_steers = cal_difference(testing_data_path=images_path, type=args.model_type, truth_angles=ground_truth_file)

            # Metric 1: Count and record the RMSE between predicted and actual values.
            if mutate_condition == 'night':
                my_column = 3
            elif mutate_condition == 'overcast':
                my_column = 9
            elif mutate_condition == 'rainy':
                my_column = 15
            elif mutate_condition == 'foggy':
                my_column = 21
            elif mutate_condition == 'dawndusk':
                my_column = 27
            elif mutate_condition == 'cloudy':
                my_column = 33
            elif mutate_condition == 'night_rainy':
                my_column = 39
            elif mutate_condition == 'night_fog':
                my_column = 45
            else:
                raise ValueError("mutate condition can not be found, please check the path")

            # Metric 1: Count RMSE between predicted and actual steering angles (just for DeepTest)
            # if mutate_condition == 'image_translation':
            #     my_column = 3
            # elif mutate_condition == 'image_scale':
            #     my_column = 9
            # elif mutate_condition == 'image_shear':
            #     my_column = 15
            # elif mutate_condition == 'image_rotation':
            #     my_column = 21
            # elif mutate_condition == 'image_contrast':
            #     my_column = 27
            # elif mutate_condition == 'image_brightness':
            #     my_column = 33
            # elif mutate_condition == 'image_blur':
            #     my_column = 39
            # elif mutate_condition == 'image_add_noise':
            #     my_column = 45
            # else:
            #     raise ValueError("mutate condition can not be found, please check the path")

            # Metric 1: Count the number of Inconsistent Behaviors
            for i in range(0, len(original_predicted_steers)):
                if abs(predicted_steers[i] * 180 / np.pi - original_predicted_steers[i] * 180 / np.pi) < 5:
                    num_of_inconsistency_1 += 1
                elif 5 <= abs(predicted_steers[i] * 180 / np.pi - original_predicted_steers[i] * 180 / np.pi) < 10:
                    num_of_inconsistency_2 += 1
                elif 10 <= abs(predicted_steers[i] * 180 / np.pi - original_predicted_steers[i] * 180 / np.pi) < 15:
                    num_of_inconsistency_3 += 1
                elif 15 <= abs(predicted_steers[i] * 180 / np.pi - original_predicted_steers[i] * 180 / np.pi):
                    num_of_inconsistency_4 += 1
                else:
                    continue

            # Metric 2: Calculate steering smoothness
            dP_dT = np.zeros(len(predicted_steers))
            for i in range(1, len(predicted_steers) - 1):
                dP_dT[i] = (predicted_steers[i + 1] - predicted_steers[i - 1]) / 2 

            dP_dT[0] = predicted_steers[1] - predicted_steers[0] 
            dP_dT[-1] = predicted_steers[-1] - predicted_steers[-2] 

            W_hitness = np.sqrt(np.sum(dP_dT ** 2) / len(predicted_steers))
            dP_dT_list.append(W_hitness)

    # Print the number of Inconsistent Behaviors for each mutation size for the corresponding model
    print(f"model {args.model_type} has {num_of_inconsistency_1} Inconsistent Behavior times in 0°~5°")
    print(f"model {args.model_type} has {num_of_inconsistency_2} Inconsistent Behavior times in 5°~10°")
    print(f"model {args.model_type} has {num_of_inconsistency_3} Inconsistent Behavior times in 10°~15°")
    print(f"model {args.model_type} has {num_of_inconsistency_4} Inconsistent Behavior times in >15°")

    # Print the average steering smoothness for the corresponding model
    print(f"model {args.model_type} has average Smoothness {np.mean(dP_dT_list)}")
    print("we have done all the test")
