"""
The stacked convolutional network model is similar to the basic convolutional network structure, except that the number of input channels is increased to accommodate multiple video frames.
In stacked convolutional networks, temporal information is fused at the input layer, and the temporal relationships between image features of different frames can be propagated to the next layer along the depth channels of the convolutional neural network.
"""
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def build_stacked_cnn(image_size=(66, 200), weights_path=None):
    input_shape = image_size + (9,)

    img_input = Input(shape=input_shape)

    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    y = Dense(1)(x)

    # Activation function range for udacity and carla datasets
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Activation function range for california dataset
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Activation function range for comma.ai dataset
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    model = Model(inputs=img_input, outputs=y)

    if weights_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(weights_path).expect_partial()

    return model


def build_finetune_stacked_cnn(pretrained_model, trainable_layers=3):
    """
    Build a Stacked CNN model for fine-tuning, freezing part of the layers based on the pretrained model and adding new fully connected layers.

    Args:
    - pretrained_model: Stacked CNN pretrained model with loaded weights
    - trainable_layers: int, number of trainable layers for fine-tuning (unfreeze from the last several layers)
    """
    # Freeze all layers except for the last trainable_layers layers
    for layer in pretrained_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Start from the output of the Flatten layer
    x = pretrained_model.layers[-6].output

    # Add new fully connected layers
    x = Dense(256, activation='relu')(x)  
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)  
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)  
    x = Dense(32, activation='relu')(x) 
    x = Dense(16, activation='relu')(x) 

    # Activation function range for udacity and carla datasets
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Activation function range for california dataset
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Activation function range for comma.ai dataset
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    # Build the fine-tuning model
    fine_tune_model = Model(inputs=pretrained_model.input, outputs=y)

    return fine_tune_model
