import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


class Featrue_Extract(tf.keras.Model):

    def __init__(self, padding="same"):
        super(Featrue_Extract, self).__init__()
        # 先试用
        self.vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(256, 256, 3))

        self.conv_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding=padding)
        self.conv_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding=padding)
        self.conv_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding=padding)

    def call(self, inputs, training=None, mask=None):
        x = self.vgg19(inputs)
        # x = x.layers[15].output
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        list_ = [x_1, x_2, x_3]
        return x_3, list_

class Conv_base(tf.keras.Model):

    def __init__(self, filter, k_size, stride, padding="same", activation="relu"):
        super(Conv_base, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filter, kernel_size=k_size, strides=stride, padding=padding,
                                           activation=activation)

    def call(self, inputs, training=None, mask=None):
        return self.conv(inputs)

class ResBlock(tf.keras.Model):
    def __init__(self, out_channels, is_training=True):
        super(ResBlock, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(trainable=is_training),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(trainable=is_training)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.main(inputs) + inputs


class Concat_S_A(tf.keras.Model):

    def __init__(self):
        super(Concat_S_A, self).__init__()

        self.conv1 = Conv_base(filter=512, k_size=3, stride=1, padding="same")
        self.conv2 = Conv_base(filter=512, k_size=3, stride=1, activation="sigmoid", padding="same")
        self.res = ResBlock(out_channels=512)

    def call(self, inputs, training=None, mask=None):
        feature, inputs_ = inputs[0], inputs[1]
        res_block = self.res(inputs_)
        out_CSA = tf.concat([feature, res_block], axis=3)
        out_CSA = self.conv1(out_CSA)
        out_CSA = self.conv2(out_CSA)
        mul = tf.multiply(res_block, out_CSA)
        return mul + inputs_

def crop_and_out(inp1, inp2):
    out = inp2 + inp1
    return out


class Anime_Generator(tf.keras.Model):

    def __init__(self):
        super(Anime_Generator, self).__init__()

        self.down_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")
        self.down_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")
        self.down_conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")
        self.down_conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")
        self.down_conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")

        self.csa1 = Concat_S_A()
        self.csa2 = Concat_S_A()
        self.csa3 = Concat_S_A()
        self.csa4 = Concat_S_A()
        self.csa5 = Concat_S_A()
        self.csa6 = Concat_S_A()
        self.csa7 = Concat_S_A()
        self.csa8 = Concat_S_A() # 8, 8, 512

        self.feature = Featrue_Extract()

        self.up_d_conv1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same") # 16, 16, 256
        self.up_d_conv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation="relu", padding="same") # 32, 32, 128
        self.up_d_conv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same") # 64, 64, 64
        self.up_d_conv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same") # 128, 128, 32
        self.up_d_conv5 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="relu", padding="same") # 256, 256, 3


    def call(self, inputs, training=None, mask=None):
        sketch, hint = inputs[0], inputs[1]
        cat = tf.concat([sketch, hint], axis=3)
        down_1 = self.down_conv1(cat)
        down_2 = self.down_conv2(down_1)
        down_3 = self.down_conv3(down_2)
        down_4 = self.down_conv4(down_3)
        down_5 = self.down_conv5(down_4)

        feature_, _ = self.feature(hint)
        x = self.csa1([feature_, down_5])
        x = self.csa2([feature_, x])
        x = self.csa3([feature_, x])
        x = self.csa4([feature_, x])
        x = self.csa5([feature_, x])
        x = self.csa6([feature_, x])
        x = self.csa7([feature_, x])
        x = self.csa8([feature_, x])

        de_conv1 = self.up_d_conv1(x)
        de_conv1 = crop_and_out(down_4, de_conv1)
        de_conv2 = self.up_d_conv2(de_conv1)
        de_conv2 = crop_and_out(down_3, de_conv2)
        de_conv3 = self.up_d_conv3(de_conv2)
        de_conv3 = crop_and_out(down_2, de_conv3)
        de_conv4 = self.up_d_conv4(de_conv3)
        de_conv4 = crop_and_out(down_1, de_conv4)

        out = self.up_d_conv5(de_conv4)
        return out


class ConvBlock_Norm(keras.Model):
    def __init__(self, dim_out, spec_norm=False, LR=0.01, stride=1, is_training=True):
        super(ConvBlock_Norm, self).__init__()
        if spec_norm:
            self.main = keras.Sequential([
                tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim_out, kernel_size=3, strides=stride,
                                    padding="same", use_bias=False)),
                keras.layers.BatchNormalization(trainable=is_training),
                keras.layers.LeakyReLU(LR)
            ])
        else:
            self.main = keras.Sequential([
                keras.layers.Conv2D(dim_out, kernel_size=3, strides=stride,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization(trainable=is_training),
                keras.layers.LeakyReLU(LR)
            ])

    def call(self, inputs, training=None, mask=None):
        return self.main(inputs)

class Discriminator_64(tf.keras.Model):
    def __init__(self, spec_norm=True, LR=0.2):
        super(Discriminator_64, self).__init__()
        self.main = tf.keras.Sequential([
            ConvBlock_Norm(16, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(32, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(64, spec_norm=spec_norm, stride=1, LR=LR),
            ConvBlock_Norm(128, spec_norm=spec_norm, stride=1, LR=LR),
            keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
        ])

    def call(self, inputs, training=None, mask=None):
        return self.main(inputs)


class Discriminator_32(tf.keras.Model):
    def __init__(self, spec_norm=True, LR=0.2):
        super(Discriminator_32, self).__init__()
        self.main = tf.keras.Sequential([
            ConvBlock_Norm(16, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(32, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(64, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(128, spec_norm=spec_norm, stride=1, LR=LR),
            keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
        ])

    def call(self, inputs, training=None, mask=None):
        return self.main(inputs)

class Discriminator_16(tf.keras.Model):
    def __init__(self, spec_norm=True, LR=0.2):
        super(Discriminator_16, self).__init__()
        self.main = tf.keras.Sequential([
            ConvBlock_Norm(16, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(32, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(64, spec_norm=spec_norm, stride=2, LR=LR),
            ConvBlock_Norm(128, spec_norm=spec_norm, stride=2, LR=LR),
            keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
        ])

    def call(self, inputs, training=None, mask=None):
        return self.main(inputs)



# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # a = tf.cast(tf.image.decode_png(tf.io.read_file("./image/68.png")), dtype=tf.float32)
# a = tf.random.normal([1, 8, 8, 512])
# b = tf.random.normal([1, 8, 8, 512])
# mo = Discriminator_72()
# mo.build((None, 256, 256, 3))
# mo.summary()