import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

def build_cnn(image_size=None, weights_path=None):
    """
    Build a simple CNN for regression (steering angle prediction).
    :param image_size: tuple, input image size, default (128, 128)
    :param weights_path: str, path to pretrained weights (optional)
    :return: Keras Model
    """
    image_size = image_size or (128, 128)
    input_shape = image_size + (3,)

    img_input = Input(shape=input_shape)

    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Third convolutional block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    # Flatten and fully connected layers
    y = Flatten()(x)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1)(y)

    # Output activation for carla/udacity dataset: restrict output range
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # For california dataset, use the following instead:
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # For comma.ai dataset, use the following instead:
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    model = Model(inputs=img_input, outputs=y)

    # Load pretrained weights if provided
    if weights_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(weights_path).expect_partial()

    return model

def build_finetune_model(pretrained_model, trainable_layers=5):
    """
    Build a fine-tuning model by freezing most layers and adding new dense layers.
    :param pretrained_model: Keras Model, the base model with loaded weights
    :param trainable_layers: int, number of trainable layers from the end
    :return: Keras Model
    """
    # Freeze all layers except the last `trainable_layers`
    for layer in pretrained_model.layers[:-trainable_layers]:
        layer.trainable = False  # Freeze most weights

    # Add new dense layers on top of the existing model
    x = pretrained_model.layers[-3].output  # Get output before the last two layers
    x = Dense(512, activation='relu')(x)    # Add a dense layer
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)    # Add another dense layer
    x = Dropout(0.25)(x)
    y = Dense(1)(x)

    # Output activation for carla/udacity dataset: restrict output range
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # For california dataset, use the following instead:
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # For comma.ai dataset, use the following instead:
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    fine_tune_model = Model(inputs=pretrained_model.input, outputs=y)
    return fine_tune_model
