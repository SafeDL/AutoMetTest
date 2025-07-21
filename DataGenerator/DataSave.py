import os
import numpy as np
import cv2
from PIL import Image
import carla


class DataSave:
    def __init__(self, index):
        self.index = index  
        self.rgb_folder = None
        self.seg_folder = None
        self.labelIds_folder = None
        # Note: Change the data save path as needed
        self._generate_path('../Baselines/Random/mutation_0')
        self.captured_frame_no = self._current_captured_frame_num()

    def _generate_path(self, root_path):
        # Generate the path for data storage
        self.rgb_folder = os.path.join(root_path, 'condition_{}'.format(self.index), 'images')
        # self.seg_folder = os.path.join(root_path, 'condition_{}'.format(self.index), 'seg_images')
        # self.labelIds_folder = os.path.join(root_path, 'condition_{}'.format(self.index), 'labelIds')
        if not os.path.exists(self.rgb_folder):
            os.makedirs(self.rgb_folder)
        # if not os.path.exists(self.seg_folder):
        #     os.makedirs(self.seg_folder)
        # if not os.path.exists(self.labelIds_folder):
        #     os.makedirs(self.labelIds_folder)

    def _current_captured_frame_num(self):
        # Get the number of data files already saved
        num_existing_data_files = len([name for name in os.listdir(self.rgb_folder) if name.endswith('.jpg')])

        return num_existing_data_files

    def existing_data_files(self):
        # Get the number of data files in the folder
        existing_files_num = len([name for name in os.listdir(self.rgb_folder) if name.endswith('.jpg')])

        return existing_files_num

    def save_training_files(self, data):
        # Image name format: self.OUTPUT_FOLDER\0000.jpg
        img_name = '{0:05}.jpg'.format(self.captured_frame_no)
        img_path = os.path.join(self.rgb_folder, img_name)
        for agent, dt in data["agents_data"].items():
            steer = dt["control"]

            # Save RGB image locally
            img_bgr = self.carla_img_to_array(dt["sensor_data"][0])
            cv2.imwrite(img_path, img_bgr)
            # Get CARLA semantic segmentation image
            # seg_img_bgr_carla = self.process_semantic(dt["sensor_data"][1])

            # Convert color image to single-channel class ID image
            # seg_label_ids = self.process_pixels(seg_img_bgr_carla[:, :, ::-1])
            # Image.fromarray(seg_label_ids.astype(np.uint8)).save(os.path.join(self.labelIds_folder, img_name))

            # Convert CARLA semantic segmentation image to Cityscapes semantic segmentation image using Cityscapes color mapping
            # seg_img_rgb_city = self.process_semantic_city(seg_label_ids)
            # cv2.imwrite(os.path.join(self.seg_folder, img_name), cv2.cvtColor(seg_img_rgb_city, cv2.COLOR_RGB2BGR))

            # Create a txt file in self.OUTPUT_FOLDER to record the image path and corresponding steering angle
            with open(os.path.join('../Baselines/Random/mutation_0',
                                   'condition_{}'.format(self.index), 'ground_truth_steer.txt'), 'a') as f:
                f.write(f'{img_name} {steer},2025-06-23\n')

        self.captured_frame_no += 1

    def carla_img_to_array(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        img_bgr = array[:, :, :3]
        return img_bgr

    def process_semantic(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        return array

    def process_semantic_city(self, seg_label_ids):
        # Convert Cityscapes class ID image to Cityscapes defined color RGB image
        TAG_TO_COLOR = {
            0: (0, 0, 0),
            1: (0, 0, 0),
            2: (0, 0, 0),
            3: (0, 0, 0),
            4: (0, 0, 0),
            5: (111, 74, 0),
            6: (81, 0, 81),
            7: (128, 64, 128),
            8: (244, 35, 232),
            9: (250, 170, 160),
            10: (230, 150, 140),
            11: (70, 70, 70),
            12: (102, 102, 156),
            13: (190, 153, 153),
            14: (180, 165, 180),
            15: (150, 100, 100),
            16: (150, 120, 90),
            17: (153, 153, 153),
            18: (153, 153, 153),
            19: (250, 170, 30),
            20: (220, 220, 0),
            21: (107, 142, 35),
            22: (152, 251, 152),
            23: (70, 130, 180),
            24: (220, 20, 60),
            25: (255, 0, 0),
            26: (0, 0, 142),
            27: (0, 0, 70),
            28: (0, 60, 100),
            29: (0, 0, 90),
            30: (0, 0, 110),
            31: (0, 80, 100),
            32: (0, 0, 230),
            33: (119, 11, 32),
            -1: (0, 0, 142),
        }

        # Create an empty color image (height, width, 3)
        height, width = seg_label_ids.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Map each tag to its color
        for tag, color in TAG_TO_COLOR.items():
            mask = (seg_label_ids == tag)  # Find all pixels with the current tag
            color_image[mask] = color  # Assign color

        return color_image

    def process_pixels(self, semantic_image):
        # Define the mapping between CARLA tags and CARLA colors
        LABELS = {
            (0, 0, 0): 0,
            (70, 70, 70): 1,
            (100, 40, 40): 2,
            (55, 90, 80): 3,
            (220, 20, 60): 4,
            (153, 153, 153): 5,
            (157, 234, 50): 6,
            (128, 64, 128): 7,
            (244, 35, 232): 8,
            (107, 142, 35): 9,
            (0, 0, 142): 10,
            (102, 102, 156): 11,
            (220, 220, 0): 12,
            (70, 130, 180): 13,
            (81, 0, 81): 14,
            (150, 100, 100): 15,
            (230, 150, 140): 16,
            (180, 165, 180): 17,
            (250, 170, 30): 18,
            (110, 190, 160): 19,
            (170, 120, 50): 20,
            (45, 60, 150): 21,
            (145, 170, 100): 22,
        }
        # Define the mapping from CARLA tag to Cityscapes tag
        carla_to_cityscapes = {
            0: 0,  # Unlabeled
            1: 11,  # Building
            2: 13,  # Fence
            3: 0,  # Other
            4: 24,  # Pedestrian
            5: 17,  # Pole
            6: 0,  # RoadLine
            7: 7,  # Road
            8: 8,  # SideWalk
            9: 21,  # Vegetation
            10: 26,  # Vehicles
            11: 12,  # Wall
            12: 20,  # TrafficSign
            13: 23,  # Sky
            14: 6,  # Ground
            15: 15,  # Bridge
            16: 10,  # RailTrack
            17: 14,  # GuardRail
            18: 19,  # TrafficLight
            19: 4,  # Static
            20: 5,  # Dynamic
            21: 0,  # Water
            22: 22,  # Terrain
        }
        # In CARLA view, convert RGB image to single-channel label image
        height, width, _ = semantic_image.shape
        # Create a single-channel image of the same size
        label_ids = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Get the RGB value of the current pixel
                pixel = tuple(semantic_image[y, x])

                # Map to label (if mapping exists)
                tag = LABELS.get(pixel, 'Unknown')
                # If mapping does not exist, immediately stop and return error
                if tag == 'Unknown':
                    tag = 0

                # Print pixel label info
                # print(f"Pixel at ({x}, {y}) is tagged as: {tag}")
                label_ids[y, x] = tag

        # In Cityscapes view, convert CARLA class ID to Cityscapes class ID
        seg_label_ids = np.vectorize(carla_to_cityscapes.get)(label_ids)

        return seg_label_ids
