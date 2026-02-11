"""
将数据集分成多个小批次,每个小批次内保持时间序列的顺序,但在训练过程中对这些小批次进行打乱
准备这些数据用于stacked-cnn或者cnn-gru模型训练
"""
import os
import numpy as np
import cv2


def make_sequences(condition_path, label_txt):
    # xs, ys分别为按照时间序列读取的图像和对应的方向盘角度
    xs = []
    ys = []
    with open(label_txt) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            filename = line_values[0]
            angle = float(line_values[1])
            xs.append(os.path.join(condition_path, filename))
            ys.append(angle)

    return len(xs), xs, ys


def convert_to_sequence_samples(num_total_images, xs, ys):
    # 依次堆叠加载图像和对应的方向盘角度
    time_steps = 3
    if not os.path.exists("../driving_dataset/carla_seq_train"):
        os.makedirs("../driving_dataset/carla_seq_train")
    existed_images = len(os.listdir("../driving_dataset/carla_seq_train"))

    with open('../driving_dataset/carla_seq_train.txt', 'a') as f:
        for idx in range(num_total_images):
            if idx + time_steps > num_total_images:
                break
            stacked_images = []
            for jdx in range(time_steps):
                image_path = xs[idx + jdx]
                image = cv2.imread(image_path)[-200:]  # 读取图像的最后200行
                stacked_images.append(cv2.resize(image, (200, 66)) / 255.0)  # 调整大小并且归一化
            # 将连续的三帧堆叠在一起,作为一个样本,并且以npz保留
            np.savez_compressed(f"../driving_dataset/carla_seq_train/{existed_images}.npz", np.array(stacked_images).astype(np.float32))
            # 保存对应的方向盘角度
            angle = ys[idx + time_steps - 1]
            f.write(f'{existed_images}.npz {angle},2025-06-21\n')
            # 更新已经保存的图像数量
            existed_images = len(os.listdir("../driving_dataset/carla_seq_train"))

    print(f"Processed {existed_images} images")


def convert_to_stacked_samples(num_total_images, xs, ys):
    time_steps = 3
    if not os.path.exists("../driving_dataset/carla_stacked_train"):
        os.makedirs("../driving_dataset/carla_stacked_train")
    existed_images = len(os.listdir("../driving_dataset/carla_stacked_train"))

    with open('../driving_dataset/carla_stacked_train.txt', 'a') as f:
        for idx in range(num_total_images):
            if idx + time_steps > num_total_images:
                break
            stacked_images = []
            for jdx in range(time_steps):
                image_path = xs[idx + jdx]
                image = cv2.imread(image_path)[-200:]  # 读取图像的最后200行
                stacked_images.append(cv2.resize(image, (200, 66)) / 255.0)  # 调整大小并且归一化
            # 将连续的三帧堆叠在一起,作为一个样本,并且以npz保留
            np.savez_compressed(f"../driving_dataset/carla_stacked_train/{existed_images}.npz", np.dstack(stacked_images).astype(np.float32))
            # 保存对应的方向盘角度
            angle = ys[idx + time_steps - 1]
            f.write(f'{existed_images}.npz {angle},2025-06-21\n')
            # 更新已经保存的图像数量
            existed_images = len(os.listdir("../driving_dataset/carla_stacked_train"))

    print(f"Processed {existed_images} images")


if __name__ == '__main__':
    # 将carla的仿真数据转变为序列数据
    # carla_collect_path = "../driving_dataset/carla_collect"
    # for index in range(0, 184):
    #     dir = f"condition_{index}"
    #     img_path = f'{carla_collect_path}/{dir}/images'
    #     label_txt_path = f'{carla_collect_path}/{dir}/ground_truth_steer.txt'
    #     num_total_images, total_images, total_angles = make_sequences(condition_path=img_path, label_txt=label_txt_path)
    #
    #     # 按照图像通道顺序加载图像和对应的方向盘角度
    #     convert_to_sequence_samples(num_total_images=num_total_images, xs=total_images, ys=total_angles)
    #     convert_to_stacked_samples(num_total_images=num_total_images, xs=total_images, ys=total_angles)

    # 将udacity和california等真实数据集转换为序列数据
    img_path = os.path.join('../driving_dataset/comma_train')
    label_txt_path = os.path.join('../driving_dataset/comma_train.txt')
    num_total_images, total_images, total_angles = make_sequences(condition_path=img_path, label_txt=label_txt_path)
    convert_to_sequence_samples(num_total_images=num_total_images, xs=total_images, ys=total_angles)
    convert_to_stacked_samples(num_total_images=num_total_images, xs=total_images, ys=total_angles)





