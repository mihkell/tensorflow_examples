import tensorflow as tf


def fully_connected_layers():
    inputs = tf.placeholder(tf.float32, shape=(None, 2))
    labels = tf.placeholder(tf.float32, shape=(None, 1))

    # make forward pass of layer
    layer_sigmoid = tf.contrib.layers.fully_connected(inputs, 1,
                                                      activation_fn=tf.sigmoid,
                                                      weights_initializer=tf.zeros_initializer())
    loss = tf.losses.mean_squared_error(layer_sigmoid, labels)

    # make backward pass of layer (e.g. backpropagation)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2):  # 2 rounds of training
            sess.run(train, {inputs: [[4., 0], [1., 1], [5., 1], [2., 0], [6., 0], [12., 0], [13., 1]],
                             labels: [[1], [0], [0], [1], [1], [1], [0]]})

        sigmoid_layer, ran_loss = sess.run([layer_sigmoid, loss], {inputs: [[5., 1], [1., 1]], labels: [[0], [0]]})
        print('sigmoid_layer:\n', sigmoid_layer)
        print('ran_loss:\n', ran_loss)


if __name__ == "__main__":
    fully_connected_layers()
