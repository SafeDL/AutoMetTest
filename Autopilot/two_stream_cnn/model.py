import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate


def build_two_stream_cnn(image_size=(66, 200), weights_path=None):
    input_rgb = Input(shape=image_size + (3,))
    input_flow = Input(shape=image_size + (3,))

    # RGB image branch
    x_rgb = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(input_rgb)
    x_rgb = Dropout(0.25)(x_rgb)

    x_rgb = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x_rgb)
    x_rgb = Dropout(0.25)(x_rgb)

    x_rgb = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x_rgb)
    x_rgb = Dropout(0.25)(x_rgb)

    x_rgb = Conv2D(64, (3, 3), activation='relu')(x_rgb)
    x_rgb = Dropout(0.25)(x_rgb)

    x_rgb = Conv2D(64, (3, 3), activation='relu')(x_rgb)
    x_rgb = Dropout(0.25)(x_rgb)
    x_rgb = Flatten()(x_rgb)

    # Optical flow image branch
    x_flow = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(input_flow)
    x_flow = Dropout(0.25)(x_flow)

    x_flow = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x_flow)
    x_flow = Dropout(0.25)(x_flow)

    x_flow = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x_flow)
    x_flow = Dropout(0.25)(x_flow)

    x_flow = Conv2D(64, (3, 3), activation='relu')(x_flow)
    x_flow = Dropout(0.25)(x_flow)

    x_flow = Conv2D(64, (3, 3), activation='relu')(x_flow)
    x_flow = Dropout(0.25)(x_flow)
    x_flow = Flatten()(x_flow)

    # Feature fusion
    x = concatenate([x_rgb, x_flow])
    x = Dense(1164, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)

    y = Dense(1)(x)

    # Activation range for udacity and carla datasets
    y = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Activation range for california dataset
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 6))(y)
    # Activation range for comma.ai dataset
    # y = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    model = Model(inputs=[input_rgb, input_flow], outputs=y)

    if weights_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(weights_path).expect_partial()

    return model


def build_fine_tuned_two_stream_cnn(base_model):
    """
    Args:
    - base_model: original pre-trained model
    - image_size: input image size
    - fine_tune_at: layer index to start unfreezing
    """
    # Freeze all layers before the specified layer
    for layer in base_model.layers[:-7]:
        layer.trainable = False

    # Get original model inputs
    input_rgb = base_model.input[0]
    input_flow = base_model.input[1]

    # Get the output of the feature extraction part of the original model
    feature_output = base_model.layers[-7].output  # Output of feature fusion layer (Dense 1164)

    # Add new fully connected layers after the feature fusion layer
    x = Dense(512, activation='relu', name="fine_tune_dense1")(feature_output)
    x = Dropout(0.25, name="fine_tune_dropout1")(x)
    x = Dense(256, activation='relu', name="fine_tune_dense2")(x)
    x = Dropout(0.25, name="fine_tune_dropout2")(x)
    x = Dense(128, activation='relu', name="fine_tune_dense3")(x)
    x = Dropout(0.25, name="fine_tune_dropout3")(x)
    x = Dense(64, activation='relu', name="fine_tune_dense4")(x)

    # Output layer: strictly keep the original regression logic, output 1 value
    y = Dense(1, name="fine_tune_output")(x)

    # Activation range for udacity and carla datasets
    new_outputs = Lambda(lambda x: tf.multiply(tf.atan(x), 2))(y)
    # Activation range for california dataset
    # new_outputs = Lambda(lambda x: tf.multiply(tf.tanh(x), 6), name="fine_tune_custom_activation")(y)
    # Activation range for comma.ai dataset
    # new_outputs = Lambda(lambda x: tf.multiply(tf.tanh(x), 8))(y)

    # Define fine-tuned model
    fine_tuned_model = Model(inputs=[input_rgb, input_flow], outputs=new_outputs)

    return fine_tuned_model
