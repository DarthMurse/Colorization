import numpy as np
from skimage import color, io
import os
import shutil


def convert_back_to_rgb(L_image, ab_image):
    L = np.transpose(100.0 * L_image.detach().numpy(), (1, 2, 0))
    ab = np.transpose(256.0 * ab_image.detach().numpy() - 128.0, (1, 2, 0))

    Lab = np.dstack((L, ab)).astype(np.float64)
    img = color.lab2rgb(Lab)

    return img


def PILToNumpyRGB(image):
    image = np.asarray(image)
    if len(image.shape) < 3:
        image = color.gray2rgb(image)
    return image


def RGBToLAB(image):
    image = color.rgb2lab(image)
    return image


def NormalizeValues(image):
    image[:, :, :1] /= 100.0
    image[:, :, 1:] += 128.0
    image[:, :, 1:] /= 256.0
    return image.astype(np.float32)


def format_folder(path):
    classes = os.listdir(path)
    for name in classes:
        current_dir = path + "/" + name
        if os.path.exists(current_dir + "/" + name + "_boxes.txt"):
            os.remove(current_dir + "/" + name + "_boxes.txt")
        image_dir = current_dir + "/images"
        images = os.listdir(image_dir)
        for image in images:
            shutil.move(image_dir + "/" + image, current_dir + "/")
        os.rmdir(image_dir)


if __name__ == "__main__":
    format_folder("../tiny-imagenet-200/train")
