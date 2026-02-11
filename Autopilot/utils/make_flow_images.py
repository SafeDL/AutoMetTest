# 构造RGB+光流训练数据,供双流CNN(two stream cnn)模型训练
import cv2
import numpy as np
import os


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


def calculate_optical_flow(prev_img_path, current_img_path):
    # load images
    prev_img = cv2.imread(prev_img_path)
    current_img = cv2.imread(current_img_path)
    # 统一resize到640x384
    prev_img = cv2.resize(prev_img, (640, 384))
    current_img = cv2.resize(current_img, (640, 384))

    # 在屏幕上显示原始图像
    # cv2.imshow('Original Image', current_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # convert to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # convert flow to HSV
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_img)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # display the optical flow images
    # cv2.imshow('Optical Flow', flow_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('optical_flow.png', flow_bgr)

    # 对读取的原始图像和光流图像进行裁剪
    current_image = cv2.resize(current_img[-200:], (200, 66)) / 255.0  # 调整大小并且归一化
    flow_image = cv2.resize(flow_bgr[-200:], (200, 66)) / 255.0  # 调整大小并且归一化

    return current_image, flow_image


def convert_to_flow_samples(num_total_images, xs, ys):
    # 依次堆叠加载图像和对应的方向盘角度
    if not os.path.exists("../driving_dataset/carla_flow_train"):
        os.makedirs("../driving_dataset/carla_flow_train")
    existed_images = len(os.listdir("../driving_dataset/carla_flow_train"))
    with open('../driving_dataset/carla_flow_train.txt', 'a') as f:
        for idx in range(1, num_total_images):
            # 获取历史图像和当前图像的地址
            pre_image_path = xs[idx - 1]
            current_image_path = xs[idx]
            # 计算光流
            current_image, flow_image = calculate_optical_flow(prev_img_path=pre_image_path, current_img_path=current_image_path)
            # 将current_image和flow_image以列表的形式保存为一个npz文件
            np.savez_compressed(f"../driving_dataset/carla_flow_train/{existed_images}.npz", np.array([current_image, flow_image]).astype(np.float32))
            # 保存对应的方向盘角度
            angle = ys[idx]
            f.write(f'{existed_images}.npz {angle},2025-06-21\n')
            # 更新已经保存的图像数量
            existed_images = len(os.listdir("../driving_dataset/carla_flow_train"))

    print(f"Processed {existed_images} images")


if __name__ == '__main__':
    # 将仿真数据转换为光流图像数据
    # carla_collect_path = "../driving_dataset/carla_collect"
    # for index in range(0, 184):
    #     dir = f"condition_{index}"
    #     img_path = f'{carla_collect_path}/{dir}/images/'
    #     label_txt_path = f'{carla_collect_path}/{dir}/ground_truth_steer.txt'
    #     num_total_images, total_images, total_angles = make_sequences(condition_path=img_path, label_txt=label_txt_path)
    #     # 将原始图像数据转换为光流图像数据
    #     convert_to_flow_samples(num_total_images=num_total_images, xs=total_images, ys=total_angles)

    # 将真实数据转换为光流图像数据
    img_path = os.path.join("../driving_dataset/comma_train")
    label_txt_path = os.path.join("../driving_dataset/comma_train.txt")
    num_total_images, total_images, total_angles = make_sequences(condition_path=img_path, label_txt=label_txt_path)
    convert_to_flow_samples(num_total_images=num_total_images, xs=total_images, ys=total_angles)

