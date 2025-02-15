from large_model import LTBC
from data import places365DataModule
import numpy as np
from skimage import color, io
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from utils import *

##Reload a checkpoint if needed
from_checkpoint = True
checkpoint = "./logs/tensorboard_logs/lightning_logs/version_18/checkpoints/last.ckpt"

if from_checkpoint:
    ltbc = LTBC.load_from_checkpoint(checkpoint, strict=False)
    print("loaded model")

## Select new images
torch.manual_seed(42)
data_folder = "../imagenet/val"
dm = places365DataModule(data_folder, batch_size=100)
dm.setup(stage="test")
print("set up datamodule")

## Retrieve a visualizable RGB Image using a LAB image (from the dataset or network output)


## Prediction test
test_dataloader = dm.test_dataloader()
batch = next(iter(test_dataloader))

images, labels = batch

images = images.cuda()
L_image = images[:, :1, :, :]
ab_image = images[:, 1:, :, :]
pred_ab = ltbc(L_image)
print("MSE =", nn.MSELoss(reduction="mean")(pred_ab, ab_image))
print("predicted batch")

## Comparison (Ground truth vs. Prediction vs. Grayscale Image)

for img_idx in range(images.shape[0]):

    # Compare colored images
    img = images[img_idx].detach().cpu()
    img = convert_back_to_rgb(img[:1, :, :], img[1:, :, :])

    pred_Lab_image = torch.cat([L_image, pred_ab], dim=1).cpu()
    pred_rgb_image = convert_back_to_rgb(
        pred_Lab_image.detach()[img_idx, :1, :, :],
        pred_Lab_image.detach()[img_idx, 1:, :, :],
    )
    im = Image.fromarray(np.uint8(255 * pred_rgb_image))
    im.save("../figures/" + str(img_idx) + ".png")

    """
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Ground truth")
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(pred_rgb_image)
    plt.title("Prediction")
    plt.subplot(1, 3, 3)
    plt.imshow(color.rgb2gray(img), cmap="gray")
    plt.title("Grayscale")
    plt.savefig("../figures/" + str(img_idx) + ".png")
    """

    # Compare label prediction
    # print(labels[img_idx])
    # print(np.sort(pred_label[img_idx].detach().numpy())[-10:-1])
    # print(np.argsort(pred_label[img_idx].detach().numpy())[-10:-1])
