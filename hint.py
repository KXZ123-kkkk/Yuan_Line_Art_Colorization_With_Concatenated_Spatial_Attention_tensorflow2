import tensorflow as tf
from PIL import Image
import numpy as np


def save_img(matrix, path, mode="RGB", data_mode="tf", norm_mode="gan"):
    if data_mode == "tf":
        matrix = np.asarray(matrix, dtype=np.float32)
    else:
        matrix = matrix
    if norm_mode == "gan":
        img_uint = np.asarray((matrix * 0.5 + 0.5) * 255., dtype=np.uint8)
    elif norm_mode == "0-1":
        img_uint = np.asarray(matrix * 255., dtype=np.uint8)
    else:
        img_uint = np.asarray(matrix, dtype=np.uint8)
    im = Image.fromarray(img_uint, mode)
    im.save(path)


def color_change(hint_map, color, x, y):
    hint_map[x][y] = color
    return hint_map


def paintCel(hint_map, image, random):
    x = np.random.randint(1, hint_map.shape[0] - 1)
    y = np.random.randint(1, hint_map.shape[1] - 1)
    color = image[x][y]
    hint_map_ = color_change(hint_map, color, x, y)

    if random() > 0.5:
        hint_map_ = color_change(hint_map_, color, x, y + 1)
        hint_map_ = color_change(hint_map_, color, x, y - 1)
    if random() > 0.5:
        hint_map_ = color_change(hint_map_, color, x + 1, y)
        hint_map_ = color_change(hint_map_, color, x - 1, y)
    return hint_map_


def hint_getter(img):
    random = np.random.rand
    image = np.asarray(img, dtype=np.float32)

    # shape = image.shape
    hint_map = np.ones_like(image)
    # np.random.randint(15, 256)

    for i in range(int(image.shape[1]/2) * int(image.shape[1]/2)):
        hint_map = paintCel(hint_map, image, random)
    return hint_map
    # save_img(hint_map, r"C:\Users\ASUS\Desktop\99.png", data_mode="numpy")
    # return hint


if __name__ == "__main__":
    im = tf.io.read_file(r"C:\Users\ASUS\Desktop\62.png")
    im_ma = tf.image.decode_png(im, dtype=tf.float32)  / 127.5 - 1
    hint_getter(im_ma)
