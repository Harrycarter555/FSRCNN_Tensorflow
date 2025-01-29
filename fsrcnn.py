import tensorflow as tf

def model(input_shape, scale, dsm):
    """
    Implementation of FSRCNN using TensorFlow 2.x Keras API.
    """
    d, s, m = dsm  # Unpack the tuple
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
