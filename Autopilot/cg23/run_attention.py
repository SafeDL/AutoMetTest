"""
Randomly select images from the test set and generate Grad-CAM attention heatmaps for the model
"""
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from model import build_cnn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_grad_cam(model, image, layer_name="conv2d_2", sess=None):
    """
    Generate Grad-CAM heatmap for a given image and model (TensorFlow 1.x static graph).
    :param model: Trained Keras model
    :param image: Input image (128x128, normalized)
    :param layer_name: Name of the target convolutional layer
    :param sess: TensorFlow session
    :return: Grad-CAM heatmap (same size as cropped region)
    """
    if sess is None:
        raise ValueError("A valid TensorFlow session `sess` must be provided.")

    # Add batch dimension and convert image to tensor
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

    # Get the output of the target conv layer and model prediction
    conv_layer = model.get_layer(layer_name).output
    preds = model.output

    # Compute gradients of the output with respect to the conv layer
    grads = tf.gradients(preds, conv_layer)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

    # Weight the feature maps by the pooled gradients
    conv_outputs = conv_layer[0]  # Remove batch dimension
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # Run the computation graph to get the values
    heatmap, pooled_grads_val, conv_outputs_val = sess.run(
        [heatmap, pooled_grads, conv_outputs],
        feed_dict={model.input: image[np.newaxis, ...]}
    )

    # Apply ReLU and normalize the heatmap to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def visualize_grad_cam(testing_data_path, model, sess):
    """
    Visualize Grad-CAM heatmaps for randomly selected images from the test set.
    :param testing_data_path: Path to the test dataset
    :param model: Trained Keras model
    :param sess: TensorFlow session
    """
    # List all image files in the test directory
    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    # Randomly select 100 images
    selected_images = random.sample(images, 100)

    # Name of the last convolutional layer (can be changed if needed)
    layer_name = "conv2d_2"

    for img_name in selected_images:
        # Load the original image
        img_path = os.path.join(testing_data_path, img_name)
        full_image = cv2.imread(img_path)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Crop and resize the image as required by the model
        cropped_image = cv2.resize(full_image[-200:], (128, 128)) / 255.0

        # Generate Grad-CAM heatmap
        heatmap = generate_grad_cam(model, cropped_image, layer_name=layer_name, sess=sess)

        # Resize heatmap to match the cropped region size
        heatmap_resized = cv2.resize(heatmap, (full_image.shape[1], 200))

        # Create a mask for the attention area
        heatmap_threshold = 0.2  # Threshold for attention region
        mask = (heatmap_resized > heatmap_threshold).astype(np.uint8)
        heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Overlay the heatmap on the original image (only on the cropped region)
        overlay_image = full_image.copy()
        overlay_cropped = overlay_image[-200:]
        for c in range(3):
            overlay_cropped[:, :, c] = (
                overlay_cropped[:, :, c] * (1 - mask) + heatmap_color[:, :, c] * mask
            )
        overlay_image[-200:] = overlay_cropped

        # Display the result
        plt.imshow(overlay_image.astype(np.uint8))
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    # Restore the trained model
    model = build_cnn()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "./save/udacity.ckpt")

    # Path to the test dataset
    image_path = '../driving_dataset/udacity_test'

    # Visualize Grad-CAM heatmaps
    visualize_grad_cam(testing_data_path=image_path, model=model, sess=sess)
