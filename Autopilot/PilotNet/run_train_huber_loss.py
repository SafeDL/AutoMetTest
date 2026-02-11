"""
Training script for PilotNet using Huber Loss
"""
import os
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
from model import build_pilotnet_cnn

# Build the PilotNet model
model = build_pilotnet_cnn()
LOGDIR = './save'
sess = tf.InteractiveSession()
L2NormConst = 1e-4  # L2 regularization coefficient
train_vars = tf.trainable_variables()

# Define placeholder for ground truth steering angle
y_true = tf.placeholder(tf.float32, shape=[None, 1])

# Define Huber Loss (with L2 regularization)
delta = 1.0  
huber_loss = tf.where(
    tf.abs(y_true - model.output) < delta,
    0.5 * tf.square(y_true - model.output),
    delta * (tf.abs(y_true - model.output) - 0.5 * delta)
)
# The total loss is the mean Huber loss plus L2 regularization
loss = tf.reduce_mean(huber_loss) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# Define dynamic learning rate with exponential decay
initial_learning_rate = 1e-4
decay_steps = 10000 
decay_rate = 0.95  
learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    global_step=tf.train.get_or_create_global_step(),
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True  
)

# Optimizer: Adam with dynamic learning rate
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())

# Create TensorBoard summaries for training and validation loss
train_loss_summary = tf.summary.scalar("training_loss", loss)
avg_val_loss_placeholder = tf.placeholder(tf.float32, shape=(), name='avg_val_loss')
avg_val_loss_summary = tf.summary.scalar("avg_val_loss", avg_val_loss_placeholder)
saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# Training hyperparameters
epochs = 100
batch_size = 64
min_val_loss = float('inf')  # Track the best validation loss

# Start training loop
for epoch in range(epochs):
    # Training phase
    for step in range(int(driving_data.num_train_images / batch_size)):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        train_step.run(feed_dict={model.input: xs, y_true: ys})
        # Write training loss to TensorBoard at every iteration
        summary = train_loss_summary.eval(feed_dict={model.input: xs, y_true: ys})
        summary_writer.add_summary(summary, epoch * driving_data.num_train_images / batch_size + step)

    # Validation phase: compute average validation loss at the end of each epoch
    val_loss_values = []
    for step in range(int(driving_data.num_val_images / batch_size)):
        xs, ys = driving_data.LoadValBatch(batch_size)
        val_loss_value = loss.eval(feed_dict={model.input: xs, y_true: ys})
        val_loss_values.append(val_loss_value)
    avg_val_loss = sum(val_loss_values) / len(val_loss_values)
    print("Epoch: %d, Validation Loss: %g" % (epoch, avg_val_loss))

    # Write validation loss to TensorBoard
    avg_val_loss_summary_str = avg_val_loss_summary.eval(feed_dict={avg_val_loss_placeholder: avg_val_loss})
    summary_writer.add_summary(avg_val_loss_summary_str, epoch)

    # Save the model if validation loss improves
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        checkpoint_path = os.path.join(LOGDIR, "carla_original.ckpt")
        filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)
