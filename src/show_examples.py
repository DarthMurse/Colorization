from large_model import LTBC
from data import ImageNetDataModule
import numpy as np
from skimage import color, io
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from utils import *

##Reload a checkpoint if needed
from_checkpoint = True
# checkpoint1 = "./logs/tensorboard_logs/lightning_logs/version_16/checkpoints/epoch=83-val_loss=4.35-train_loss=0.00.ckpt"
checkpoint2 = "./logs/tensorboard_logs/lightning_logs/version_18/checkpoints/epoch=83-val_loss=4.44-train_loss=0.00.ckpt"

if from_checkpoint:
    # ltbc1 = LTBC.load_from_checkpoint(checkpoint1, strict=False)
    ltbc2 = LTBC.load_from_checkpoint(checkpoint2, strict=False)
    print("loaded model")

## Select new images
torch.manual_seed(42)
data_folder = "../imagenet/val"
dm = ImageNetDataModule(data_folder, batch_size=50)
dm.setup(stage="test")
print("set up datamodule")

## Retrieve a visualizable RGB Image using a LAB image (from the dataset or network output)


## Prediction test
test_dataloader = dm.test_dataloader()
iter_b = iter(test_dataloader)
next(iter_b)
batch = next(iter_b)

images, labels = batch

images = images.cuda()
L_image = images[:, :1, :, :]
ab_image = images[:, 1:, :, :]
# pred_ab1 = ltbc1(L_image)
pred_ab2 = ltbc2(L_image)
# print("MSE =", nn.MSELoss(reduction="mean")(pred_ab, ab_image))
print("predicted batch")

## Comparison (Ground truth vs. Prediction vs. Grayscale Image)
"""
fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    gridspec_kw={"wspace": 0, "hspace": 0},
    squeeze=True,
    figsize=(9, 6),
)  """
num = [19, 20, 25]
for i, img_idx in enumerate(range(images.shape[0])):

    # Compare colored images
    img = images[img_idx].detach().cpu()
    img = convert_back_to_rgb(img[:1, :, :], img[1:, :, :])

    # pred_Lab_image1 = torch.cat([L_image, pred_ab1], dim=1).cpu()
    # pred_rgb_image1 = convert_back_to_rgb(
    #    pred_Lab_image1.detach()[img_idx, :1, :, :],
    #    pred_Lab_image1.detach()[img_idx, 1:, :, :],
    # )
    pred_Lab_image2 = torch.cat([L_image, pred_ab2], dim=1).cpu()
    pred_rgb_image2 = convert_back_to_rgb(
        pred_Lab_image2.detach()[img_idx, :1, :, :],
        pred_Lab_image2.detach()[img_idx, 1:, :, :],
    )
    # im = Image.fromarray(np.uint8(255 * pred_rgb_image))
    # im.save("../figures/" + str(img_idx) + ".png")

    # axs[0, i].axis("off")
    # axs[0, i].imshow(pred_rgb_image1, aspect="auto")
    # axs[1, i].axis("off")
    # axs[1, i].imshow(pred_rgb_image2, aspect="auto")
    # """
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Ground truth")
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(pred_rgb_image2)
    plt.title("Prediction")
    plt.subplot(1, 3, 3)
    plt.imshow(color.rgb2gray(img), cmap="gray")
    plt.title("Grayscale")
    plt.savefig("../figures/" + str(img_idx) + ".png")
    # """

    # Compare label prediction
    # print(labels[img_idx])
    # print(np.sort(pred_label[img_idx].detach().numpy())[-10:-1])
    # print(np.argsort(pred_label[img_idx].detach().numpy())[-10:-1])
# fig.savefig("result.png")
