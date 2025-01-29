import tensorflow as tf

def model(input_shape, scale, dsm):
    """
    Implementation of FSRCNN using TensorFlow 2.x Keras API.
    """
    d, s, m = dsm  # Unpack the tuple here
    channels = 1
    PS = channels * (scale * scale)  # for sub-pixel, PS = Phase Shift

    # Input layer
    x = tf.keras.Input(shape=input_shape, name="input_image")

    # Feature extraction
    x = tf.keras.layers.Conv2D(d, kernel_size=5, strides=1, padding='same', activation=None, name="conv1")(x)
    x = parametric_relu(x, "alpha1")

    # Shrinking
    x = tf.keras.layers.Conv2D(s, kernel_size=1, strides=1, padding='same', activation=None, name="conv2")(x)
    x = parametric_relu(x, "alpha2")

    # Non-linear mapping (m layers)
    for i in range(m):
        x = tf.keras.layers.Conv2D(s, kernel_size=3, strides=1, padding='same', activation=None, name=f"conv{i + 3}")(x)
        x = parametric_relu(x, f"alpha{i + 3}")

    # Expanding
    x = tf.keras.layers.Conv2D(d, kernel_size=1, strides=1, padding='same', activation=None, name=f"conv{m + 3}")(x)
    x = parametric_relu(x, f"alpha{m + 3}")

    # Final convolution (sub-pixel layer)
    x = tf.keras.layers.Conv2D(PS, kernel_size=1, strides=1, padding='same', activation=None, name=f"conv{m + 4}")(x)

    # Sub-pixel (depth-to-space)
    x = tf.nn.depth_to_space(x, scale, data_format='NHWC')

    # Output layer
    out = tf.keras.layers.Activation('linear', name="output")(x)

    # Define the model
    return tf.keras.Model(inputs=x, outputs=out)

def parametric_relu(_x, name):
    """
    Parametric ReLU activation function.
    """
    alphas = tf.Variable(tf.constant(0.1, shape=[_x.shape[-1]]), name=name, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - tf.abs(_x)) * 0.5
    return pos + neg
