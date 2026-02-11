import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GRU, TimeDistributed, Lambda

def build_conv_gru(image_size=(66, 200), time_steps=3, weights_path=None):
    """
    Build a Conv-GRU model for sequential image input.

    Args:
        image_size (tuple): The size of each input image (height, width).
        time_steps (int): Number of sequential frames to stack as input.
        weights_path (str, optional): Path to pretrained weights to restore.

    Returns:
        model (tf.keras.Model): The constructed Conv-GRU model.
    """
    input_shape = (time_steps,) + image_size + (3,)

    img_input = Input(shape=input_shape)

    # Use TimeDistributed to apply Conv2D layers to each time step independently
    x = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))(img_input)
    x = TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TimeDistributed(Conv2D(48, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)

    # GRU layer to process temporal features
    x = GRU(100, return_sequences=False)(x)
    x = Dropout(0.25)(x)

    # Fully connected layers and custom activation for output
    y = Dense(50, activation='relu')(x)
    y = Dense(10, activation='relu')(y)
    y = Dense(1)(y)

    # Output activation for Udacity and Carla datasets: restricts output range
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Output activation for California dataset (uncomment if needed)
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Output activation for comma.ai dataset (uncomment if needed)
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    model = Model(inputs=img_input, outputs=y)

    # Restore weights if a path is provided
    if weights_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(weights_path).expect_partial()

    return model

def build_finetune_model(pretrained_model, trainable_layers=1):
    """
    Build a fine-tuning model by freezing part of the pretrained model and adding extra dense layers.

    Args:
        pretrained_model (tf.keras.Model): The pretrained model with loaded weights.
        trainable_layers (int): Number of trainable layers from the end of the model (unfreeze from the last N layers).

    Returns:
        fine_tune_model (tf.keras.Model): The constructed fine-tuning model.
    """
    # Freeze all layers except the last `trainable_layers` layers
    for layer in pretrained_model.layers[:-trainable_layers]:
        layer.trainable = False  # Freeze layer

    # Add fine-tuning layers on top of the pretrained model
    x = pretrained_model.layers[-4].output  # Get output from the GRU layer
    x = Dense(512, activation='relu')(x)    # Add a fully connected layer
    x = Dropout(0.25)(x)                    # Add Dropout to prevent overfitting
    x = Dense(256, activation='relu')(x)    # Add another fully connected layer
    x = Dropout(0.25)(x)                    # Add Dropout again
    x = Dense(128, activation='relu')(x)    # Add another fully connected layer
    x = Dropout(0.25)(x)                    # Add Dropout again
    y = Dense(1)(x)                         # Output layer

    # Output activation for Udacity and Carla datasets: restricts output range
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Output activation for California dataset (uncomment if needed)
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Output activation for comma.ai dataset (uncomment if needed)
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    # Build the fine-tuning model
    fine_tune_model = Model(inputs=pretrained_model.input, outputs=y)

    return fine_tune_model
