'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''

import numpy as np
import scipy.stats as st
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:

    def __init__(self, vgg19_npy_path=None):
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')

    def build_conv4_4(self, rgb, include_fc=False):
        rgb_scaled = (rgb + 1) * 127.5

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                        green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = tf.nn.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = tf.nn.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        return self.conv4_4, self.conv2_1

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            # relu = tf.nn.relu(bias)
            return bias

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def vggloss_4_4(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_conv4_4(image_a)
    vgg_b = vgg_model.build_conv4_4(image_b)
    VGG_loss = tf.losses.absolute_difference(vgg_a, vgg_b)
    # VGG_loss = tf.nn.l2_loss(vgg_a - vgg_b)
    h, w, c = vgg_a.get_shape().as_list()[1:]
    VGG_loss = tf.reduce_mean(VGG_loss) / (h * w * c)
    return VGG_loss

def main():
    a, b = Vgg19("./pretrain_model/vgg19.npy")
    print(a, b)

if __name__ == "__main__":
    main()