from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os

def model(x, y, lr_size, scale, batch, lr, dsm):
    """
    Implementation of FSRCNN: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html.
    """
    d, s, m = dsm  # Unpack the tuple inside the function

    channels = 1
    PS = channels * (scale * scale)  # for sub-pixel, PS = Phase Shift
    bias_initializer = tf.constant_initializer(value=0.0)

    # -- Filters and Biases
    filters = [
        tf.Variable(tf.random_normal([5, 5, 1, d], stddev=0.1), name="f1"),  # (f1, n1, c1) = (5, 64, 1)
        tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.1), name="f2"),  # (f2, n2, c2) = (3, 12, 56)
        tf.Variable(tf.random_normal([1, 1, s, d], stddev=0.1), name="f%d" % (3 + m)),  # (f3, n3, c3) = (1, 56, 12)
        tf.Variable(tf.random_normal([1, 1, d, PS], stddev=0.1), name="f%d" % (4 + m))  # (f4, n4, c4) = (9, 1, 56)
    ]
    
    bias = [
        tf.get_variable(shape=[d], initializer=bias_initializer, name="b1"),
        tf.get_variable(shape=[s], initializer=bias_initializer, name="b2"),
        tf.get_variable(shape=[d], initializer=bias_initializer, name="b%d" % (3 + m)),
        tf.get_variable(shape=[1], initializer=bias_initializer, name="b%d" % (4 + m))
    ]
    
    # Add filters and biases for 'non-linear mapping' layers (depending on m), and name them in order
    for i in range(0, m):
        filters.insert(i + 2, tf.Variable(tf.random_normal([3, 3, s, s], stddev=0.1), name="f%d" % (3 + i)))  # (f5, n5, c5) = (3, 12, 12)
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

    x = tf.nn.conv2d(x, filters[4 + (m - 1)], [1, 1, 1, 1], padding='SAME', name="conv%d" % (4 + m))

    # Sub-pixel (depth-to-space)
    x = tf.nn.depth_to_space(x, scale, data_format='NHWC')
    out = tf.nn.bias_add(x, bias[4 + (m - 1)], name="NHWC_output")

    # -- --

    # Some outputs
    out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")
    psnr = tf.image.psnr(out, y, max_val=1.0)
    loss = tf.losses.mean_squared_error(out, y)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return out, loss, train_op, psnr

def prelu(_x, name):
    """
    Parametric ReLU.
    """
    alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def main():
    # Example main function for training/testing model
    batch_size = 16
    scale = 3
    lr_size = 0.0001
    dsm = (64, 12, 5)  # Example values for d, s, m
    x = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # Placeholder for input image
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # Placeholder for output image
    
    out, loss, train_op, psnr = model(x, y, lr_size, scale, batch_size, lr_size, dsm)
    
    # Further training/test code goes here

if __name__ == "__main__":
    main()
