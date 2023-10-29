import tensorflow as tf

from third_party.deeplabv3plus import Deeplabv3


def get_network(input_shape, n_output_channels):

    deeplab = Deeplabv3(input_shape=input_shape,
                            classes=n_output_channels,
                            backbone="mobilenetv2",
                            weights=None,
                            activation="softmax")

    i = tf.keras.layers.Input(input_shape)
    o = deeplab(i)

    return tf.keras.models.Model(i, o)
