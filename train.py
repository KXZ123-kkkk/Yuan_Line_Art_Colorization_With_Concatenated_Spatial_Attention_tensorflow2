import tensorflow as tf
from tensorflow import keras

import numpy as np
from PIL import Image

from model import Anime_Generator, Discriminator_16, Discriminator_32, Discriminator_64, Featrue_Extract
import glob
import time

from hint import hint_getter
# physical_devices = tf.config.experimental.list_physical_devices('CPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def hint_(image):
    return hint_getter(image.numpy())


def extract(model, source, num=-1):
    # precess = tf.cast(source, dtype=tf.float32) / 127.5 - 1.
    inputs = model.input
    output = model.layers[num].output
    model_ = tf.keras.Model(inputs, output)
    return model_(source)


def loss_FM(model_D, model_G, line, hint, real):
    layer_len = len(model_D.layers[0].layers)
    model_d_layers = tf.concat([line, real], axis=3)
    fake = model_G([line, hint])
    model_d_2_layers = tf.concat([line, fake], axis=3)
    loss = 0
    for index in range(layer_len):
        inputs_1 = model_D.layers[0].input

        output_1 = model_D.layers[0].layers[index].output
        model_ = tf.keras.Model(inputs_1, output_1)
        out_1 = model_(model_d_layers)
        out_2 = model_(model_d_2_layers)
        loss_ = tf.reduce_mean(tf.losses.mae(out_1, out_2))
        loss += loss_
    return loss


def loss_cGan(model_D, model_G, line, hint, real):
    L__1 = model_D(tf.concat([line, real], axis=3))
    L__2 = 1 - model_D(tf.concat([line, model_G([line, hint])], axis=3))
    loss = tf.exp(L__1) + tf.exp(L__2)
    return loss


def loss_perc(model_G, model_vgg, line, hint, real):
    fake = model_G([line, hint])
    model_vgg_1, ls_1 = model_vgg(real)
    model_vgg_2, ls_2 = model_vgg(fake)
    loss = 0
    layers_len = len(model_vgg.layers)
    # + len(model_vgg.layers[0].layers) - 1
    for i in range(layers_len):
        if i == 0:
            # target layers
            for k in [3, 8, 15]:
                inputs_v = model_vgg.layers[0].input
                output_v = model_vgg.layers[0].layers[k].output
                model_v = tf.keras.Model(inputs_v, output_v)
                vgg_1 = model_v(real)
                vgg_2 = model_v(fake)
                loss_ = tf.reduce_mean(tf.losses.mae(vgg_1, vgg_2))
                loss += loss_
        else:
            loss_ = tf.reduce_mean(tf.losses.mae(ls_1[i - 1], ls_2[i - 1]))
            loss += loss_
    return loss


def loss_G_ALL(G, F, D_16, D_32, D_64, line, hint, real):
    L_C_D16 = loss_cGan(D_16, G, line, hint, real)
    L_C_D32 = loss_cGan(D_32, G, line, hint, real)
    L_C_D64 = loss_cGan(D_64, G, line, hint, real)

    L_C_D16 = tf.reduce_mean(tf.reduce_mean(tf.squeeze(L_C_D16, axis=3), axis=2), 1)
    L_C_D32 = tf.reduce_mean(tf.reduce_mean(tf.squeeze(L_C_D32, axis=3), axis=2), 1)
    L_C_D64 = tf.reduce_mean(tf.reduce_mean(tf.squeeze(L_C_D64, axis=3), axis=2), 1)

    ahead_loss = L_C_D16 + L_C_D32 + L_C_D64

    L_FM_D16 = loss_FM(D_16, G, line, hint, real)
    L_FM_D32 = loss_FM(D_32, G, line, hint, real)
    L_FM_D64 = loss_FM(D_64, G, line, hint, real)

    L_PERC = loss_perc(G, F, line, hint, real)

    langda = 1
    after_loss = langda * (L_FM_D16 + L_PERC + L_FM_D32 + L_PERC + L_FM_D64 + L_PERC)
    loss = after_loss + ahead_loss
    return loss


# D_loss
def loss_D(D, fake, real, sketch):
    real_score = D(tf.concat([real, sketch], axis=3))
    fake_score = D(tf.concat([fake, sketch], axis=3))
    d_loss_real = tf.losses.mse(tf.ones_like(real_score), real_score)
    d_loss_fake = tf.losses.mse(tf.zeros_like(fake_score), fake_score)
    loss = 1 * d_loss_real + 1 * d_loss_fake
    return loss


def norm(sketch_keras, real):
    sketch_keras = sketch_keras / 127.5 - 1.
    # hint = hint / 127.5 - 1.
    real = real / 127.5 - 1.
    return sketch_keras, real


def pro_train(sketch, real):
    sketch = tf.io.read_file(sketch)
    # hint = tf.io.read_file(hint)
    real = tf.io.read_file(real)
    sketch = tf.image.decode_png(sketch, channels=1)
    # hint = tf.image.decode_png(hint)
    real = tf.image.decode_png(real)

    sketch_keras = tf.image.resize(sketch, [256, 256])
    # hint = tf.image.resize(hint, [256, 256])
    real = tf.image.resize(real, [256, 256])
    sketch_keras, real = norm(sketch_keras, real)
    sketch_keras = tf.cast(sketch_keras, dtype=tf.float32)
    real = tf.cast(real, dtype=tf.float32)
    hint = tf.py_function(func=hint_, inp=[real], Tout=tf.float32)

    return sketch_keras, hint, real


def pro_test(sketch, real):
    sketch = tf.io.read_file(sketch)
    # hint = tf.io.read_file(hint)
    real = tf.io.read_file(real)
    sketch = tf.image.decode_png(sketch)
    # hint = tf.image.decode_png(hint)
    sketch_keras = tf.image.resize(sketch, [256, 256])
    # hint = tf.image.resize(hint, [256, 256])
    real = tf.image.resize(real, [256, 256])
    # hint = tf.cast(hint, dtype=tf.float32)
    real = tf.cast(real, dtype=tf.float32)
    sketch_keras = tf.cast(sketch_keras, dtype=tf.float32)
    sketch_keras, real = norm(sketch_keras, real)
    hint = tf.py_function(func=hint_, inp=[real], Tout=tf.float32)

    return sketch_keras, hint, real


def data_load_path_train():
    # color = glob.glob(r"image\color\*.png")
    # sketch = glob.glob(r"image\sketch\*.png")
    color = glob.glob(r"D:\TensorFlow\实战集合\翻译2020\Kaggle\anime-sketch-colorization-pair\data\color\*.png")[:500]
    sketch = glob.glob(r"D:\TensorFlow\实战集合\翻译2020\Kaggle\anime-sketch-colorization-pair\data\sketch\*.png")[:500]
    # print(color)
    # hint = glob.glob(r"image\hint\*.png")
    return (sketch, color)


def data_load_path_test():
    return ()


def save_img(matrix, path, mode="RGB"):
    img_uint = np.asarray((matrix.numpy() * 0.5 + 0.5) * 255., dtype=np.uint8)
    im = Image.fromarray(img_uint, mode)
    im.save(path)


def save_model(G, D_16, D_32, D_64):
    base_path = "./model_ckpt"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    G.save_weights(os.path.join(base_path, "G.ckpt"))
    D_16.save_weights(os.path.join(base_path, "D_16.ckpt"))
    D_32.save_weights(os.path.join(base_path, "D_32.ckpt"))
    D_64.save_weights(os.path.join(base_path, "D_64.ckpt"))
    print("save model ok!")


def restore_model(G, D_16, D_32, D_64):
    base_path = "./model_ckpt"
    G.load_weights(os.path.join(base_path, "G.ckpt"))
    D_16.load_weights(os.path.join(base_path, "D_16.ckpt"))
    D_32.load_weights(os.path.join(base_path, "D_32.ckpt"))
    D_64.load_weights(os.path.join(base_path, "D_64.ckpt"))
    print("restore model ok!")


def create_logdir_tensorboard(log_name: list, log_dir="log"):
    ls = []
    for name in log_name:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_dir_e = os.path.join(log_dir, name)
        if not os.path.exists(log_dir_e):
            os.mkdir(log_dir_e)
        name = tf.summary.create_file_writer(log_dir_e)
        ls.append(name)
    return ls


def log(writer, steps, data, log_name="cat"):
    with writer.as_default():
        tf.summary.scalar(log_name, data, step=steps)
        writer.flush()
    print(log_name, " ok!")


def loss_D_2_scalar(loss):
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(loss)))


def main():
    train_data = tf.data.Dataset.from_tensor_slices(data_load_path_train())
    train_data = train_data.map(pro_train).shuffle(60000).batch(1)
    # test_data = tf.data.Dataset.from_tensor_slices(data_load_path_test())
    # test_data = test_data.map(pro_test).shuffle(60000).batch(1)

    A_G = Anime_Generator()

    D_16 = Discriminator_16()
    D_32 = Discriminator_32()
    D_64 = Discriminator_64()
    F_E = Featrue_Extract()
    A_G.build(input_shape=[(None, 256, 256, 1), (None, 256, 256, 3)])
    D_16.build(input_shape=(None, 256, 256, 4))
    D_32.build(input_shape=(None, 256, 256, 4))
    D_64.build(input_shape=(None, 256, 256, 4))
    F_E.build(input_shape=(None, 256, 256, 3))
    A_G.summary()
    D_16.summary()
    D_32.summary()
    D_64.summary()
    F_E.summary()

    opt_A_G = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    opt_D_16 = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    opt_D_32 = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    opt_D_64 = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

    if os.path.exists("./model_ckpt/checkpoint"):
        restore_model(A_G, D_16, D_32, D_64)
    Epoch = 100

    for i in range(Epoch):
        writer_list = create_logdir_tensorboard(
            ["{}_G_loss".format(i), "{}_D_16_loss".format(i), "{}_D_32_loss".format(i), "{}_D_64_loss".format(i)])
        # print(writer_list)
        start = time.perf_counter()
        for index, (sketch, hint, real) in enumerate(train_data):
            fake = A_G([sketch, hint])
            save_img(fake[0], r"results/fake{}.png".format(index))
            save_img(tf.squeeze(sketch[0], axis=2), r"results/sketch{}.png".format(index), mode="L")
            save_img(hint[0], r"results/hint{}.png".format(index))
            save_img(real[0], r"results/real{}.png".format(index))
            with tf.GradientTape() as tp:
                loss = loss_G_ALL(A_G, F_E, D_16, D_32, D_64, sketch, hint, real)
                print("G_loss:", tf.reduce_mean(loss).numpy())
                log(writer_list[0], index, tf.reduce_mean(loss).numpy(), "G_loss")
            grad = tp.gradient(loss, A_G.trainable_variables)
            opt_A_G.apply_gradients(zip(grad, A_G.trainable_variables))

            with tf.GradientTape() as tp16:
                loss_16 = loss_D(D_16, fake, real, sketch)
                log(writer_list[1], index, loss_D_2_scalar(loss_16).numpy(), "D_16_loss")

            grad1 = tp16.gradient(loss_16, D_16.trainable_variables)
            opt_D_16.apply_gradients(zip(grad1, D_16.trainable_variables))

            with tf.GradientTape() as tp32:
                loss_32 = loss_D(D_32, fake, real, sketch)
                log(writer_list[2], index, loss_D_2_scalar(loss_32).numpy(), "D_32_loss")
            grad2 = tp32.gradient(loss_32, D_32.trainable_variables)
            opt_D_32.apply_gradients(zip(grad2, D_32.trainable_variables))

            with tf.GradientTape() as tp64:
                loss_64 = loss_D(D_64, fake, real, sketch)
                log(writer_list[3], index, loss_D_2_scalar(loss_64).numpy(), "D_64_loss")
            grad3 = tp64.gradient(loss_64, D_64.trainable_variables)
            opt_D_64.apply_gradients(zip(grad3, D_64.trainable_variables))
            end = time.perf_counter()
            print("time: ", end - start / 60, " min")
            if index % 5 == 0:
                save_model(A_G, D_16, D_32, D_64)


if __name__ == '__main__':
    main()
