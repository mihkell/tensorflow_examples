import tensorflow as tf


def fully_connected_layers():
    """ Method creates 3 layers and runs them - no training is made, no connection between layers. """

    inputs = [[-3.], [4.], [5.]]  # inputs need to be list of lists (rank 2) with type float or complex

    layer_relu_default = tf.contrib.layers.fully_connected(inputs, 1, weights_initializer=tf.ones_initializer())  #

    layer_sigmoid = tf.contrib.layers.fully_connected(inputs, 1, activation_fn=tf.sigmoid,
                                                      weights_initializer=tf.ones_initializer())

    layer_identity = tf.contrib.layers.fully_connected(inputs, 1, activation_fn=None,
                                                       weights_initializer=tf.ones_initializer())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sigmoid_layer, relu_layer, identity_Layer = sess.run([layer_sigmoid, layer_relu_default, layer_identity])
        print('relu_layer:\n', relu_layer)
        print('sigmoid_layer:\n', sigmoid_layer)
        print('identity_Layer:\n', identity_Layer)


if __name__ == "__main__":
    fully_connected_layers()
