import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os

def model(x, y, lr_size, scale, batch, lr, dsm):
    """
    Implementation of FSRCNN: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html.
    """
    # Unpack the tuple inside the function
    d, s, m = dsm  
    channels = 1
    PS = channels * (scale * scale)  # for sub-pixel, PS = Phase Shift
    bias_initializer = tf.constant_initializer(value=0.0)

    # -- Filters and Biases
    filters = [
    tf.Variable(tf.random.normal([5, 5, 1, d], stddev=0.1), name="f1"),
    tf.Variable(tf.random.normal([1, 1, d, s], stddev=0.1), name="f2"),
    tf.Variable(tf.random.normal([1, 1, s, d], stddev=0.1), name="f%d" % (3 + m)),
    tf.Variable(tf.random.normal([1, 1, d, PS], stddev=0.1), name="f%d" % (4 + m))
]

    bias = [
        tf.get_variable(shape=[d], initializer=bias_initializer, name="b1"),
        tf.get_variable(shape=[s], initializer=bias_initializer, name="b2"),
        tf.get_variable(shape=[d], initializer=bias_initializer, name="b%d" % (3 + m)),
        tf.get_variable(shape=[PS], initializer=bias_initializer, name="b%d" % (4 + m))
    ]

    # Add filters and biases for 'non-linear mapping' layers (depending on m)
    for i in range(0, m):
        filters.insert(i + 2, tf.Variable(tf.random_normal([3, 3, s, s], stddev=0.1), name="f%d" % (3 + i)))
        bias.insert(i + 2, tf.get_variable(shape=[s], initializer=bias_initializer, name="b%d" % (3 + i)))

    # -- Model architecture --
    # Feature extraction
    x = tf.nn.conv2d(x, filters[0], [1, 1, 1, 1], padding='SAME', name="conv1")
    x = x + bias[0]
    x = prelu(x, "alpha1")

    # Shrinking
    x = tf.nn.conv2d(x, filters[1], [1, 1, 1, 1], padding='SAME', name="conv2")
    x = x + bias[1]
    x = prelu(x, "alpha2")

    # Non-linear mapping (amount of layers depends on m)
    for i in range(0, m):
        x = tf.nn.conv2d(x, filters[2 + i], [1, 1, 1, 1], padding='SAME', name="conv%d" % (3 + i))
        x = x + bias[2 + i]
        x = prelu(x, "alpha{}".format(3 + i))

    # Expanding
    x = tf.nn.conv2d(x, filters[3 + (m - 1)], [1, 1, 1, 1], padding='SAME', name="conv%d" % (3 + m))
    x = x + bias[3 + (m - 1)]
    x = prelu(x, "alpha{}".format(3 + m))

    # Final convolution (sub-pixel layer)
    x = tf.nn.conv2d(x, filters[4 + (m - 1)], [1, 1, 1, 1], padding='SAME', name="conv%d" % (4 + m))

    # Sub-pixel (depth-to-space)
    x = tf.nn.depth_to_space(x, scale, data_format='NHWC')
    out = tf.nn.bias_add(x, bias[4 + (m - 1)], name="NHWC_output")

    # -- Outputs --
    out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")  # Transpose to NCHW format if needed
    psnr = tf.image.psnr(out, y, max_val=1.0)  # Compute PSNR
    loss = tf.losses.mean_squared_error(out, y)  # Compute MSE loss
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # Optimizer

    return out, loss, train_op, psnr


def prelu(_x, name):
    """
    Parametric ReLU activation function.
    """
    alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32, trainable=True)
    pos = tf.nn.relu(_x)  # Positive part
    neg = alphas * (_x - abs(_x)) * 0.5  # Negative part
    return pos + neg


def load_dataset(lr_dir, hr_dir):
    """
    Load low-resolution and high-resolution image pairs from directories.
    """
    lr_images = []
    hr_images = []

    for lr_file in os.listdir(lr_dir):
        lr_path = os.path.join(lr_dir, lr_file)
        hr_path = os.path.join(hr_dir, lr_file)  # Assuming filenames match

        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Normalize to [0, 1]
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE) / 255.0

        lr_images.append(np.expand_dims(lr_img, axis=-1))  # Add channel dimension
        hr_images.append(np.expand_dims(hr_img, axis=-1))

    return np.array(lr_images), np.array(hr_images)


def main():
    """
    Main function for training/testing the FSRCNN model.
    """
    # Hyperparameters
    batch_size = 16
    scale = 3  # Upscaling factor
    lr_size = 0.0001  # Learning rate
    dsm = (64, 12, 5)  # Example values for d, s, m
    epochs = 10

    # Dataset paths
    lr_dir = "./data/lr_images"  # Directory containing low-resolution images
    hr_dir = "./data/hr_images"  # Directory containing high-resolution images

    # Load dataset
    lr_images, hr_images = load_dataset(lr_dir, hr_dir)

    # Placeholders for input and output images
    x = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="input_image")  # LR input image
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="target_image")  # HR target image

    # Build the model
    out, loss, train_op, psnr = model(x, y, lr_size, scale, batch_size, lr_size, dsm)

    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):  # Number of epochs
            print(f"Epoch {epoch + 1}")
            
            # Shuffle dataset
            indices = np.arange(len(lr_images))
            np.random.shuffle(indices)
            lr_images = lr_images[indices]
            hr_images = hr_images[indices]

            for i in range(0, len(lr_images), batch_size):
                batch_lr = lr_images[i:i + batch_size]
                batch_hr = hr_images[i:i + batch_size]

                # Train step
                _, current_loss, current_psnr = sess.run([train_op, loss, psnr], feed_dict={x: batch_lr, y: batch_hr})
                print(f"Batch {i // batch_size}: Loss: {current_loss:.4f}, PSNR: {np.mean(current_psnr):.4f}")

        # Save the trained model
        saver = tf.train.Saver()
        saver.save(sess, "./fsrcnn_model.ckpt")
        print("Model saved.")


if __name__ == "__main__":
    main()
