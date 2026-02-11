import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Lambda

# Build custom convolutional neural network (PilotNet architecture)
def build_pilotnet_cnn(weights_path=None):
    # Define input layer for images of shape (66, 200, 3)
    img_input = Input(shape=(66, 200, 3))
    h_conv1 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(img_input)
    h_conv2 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(h_conv1)
    h_conv3 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(h_conv2)
    h_conv4 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(h_conv3)
    h_conv5 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(h_conv4)
    h_conv5_flat = Flatten()(h_conv5)
    h_fc1 = Dense(1164, activation='relu')(h_conv5_flat)
    h_fc1_drop = Dropout(0.3)(h_fc1)
    h_fc2 = Dense(100, activation='relu')(h_fc1_drop)
    h_fc2_drop = Dropout(0.3)(h_fc2)
    h_fc3 = Dense(50, activation='relu')(h_fc2_drop)
    h_fc3_drop = Dropout(0.3)(h_fc3)
    h_fc4 = Dense(10, activation='relu')(h_fc3_drop)
    h_fc4_drop = Dropout(0.3)(h_fc4)
    # Output layer: single neuron (regression output)
    y = Dense(1)(h_fc4_drop)

    # Output activation for Udacity and Carla datasets: restrict range using arctan
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Output activation for California dataset: restrict range using tanh
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Output activation for comma.ai dataset: restrict range using tanh
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    # Create and return the Keras model
    model = Model(inputs=img_input, outputs=y)

    # Optionally load weights from a checkpoint if provided
    if weights_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(weights_path).expect_partial()

    return model


def build_finetune_pilotnet(pretrained_model, trainable_layers=1):
    """
    Build a fine-tuning PilotNet model by freezing all but the last N layers of a pretrained model,
    and adding extra fully connected layers for further training.

    Args:
        pretrained_model: The PilotNet model with loaded weights to be fine-tuned.
        trainable_layers: int, number of trainable layers from the end of the model (default: 1).
    """
    # Freeze all layers except the last 'trainable_layers' layers
    for layer in pretrained_model.layers[:-trainable_layers]:
        layer.trainable = False  # Freeze layer

    # Get the output of the last dense layer before the new fine-tuning layers
    x = pretrained_model.layers[-8].output

    # Add new fully connected layers for fine-tuning
    x = Dense(256, activation='relu')(x)      # Add a dense layer with 256 units
    x = Dropout(0.3)(x)                       # Add dropout to prevent overfitting
    x = Dense(128, activation='relu')(x)      # Add another dense layer with 128 units
    x = Dropout(0.3)(x)                       # Add another dropout layer
    x = Dense(64, activation='relu')(x)       # Add a third dense layer with 64 units
    y = Dense(1)(x)                           # Output layer

    # Output activation for Udacity and Carla datasets: restrict range using arctan
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Output activation for California dataset: restrict range using tanh
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Output activation for comma.ai dataset: restrict range using tanh
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    # Build and return the fine-tuned model
    fine_tune_model = Model(inputs=pretrained_model.input, outputs=y)

    return fine_tune_model
