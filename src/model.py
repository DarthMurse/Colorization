import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from utils import *
from random import random


class ConvModule(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        stride,
        kernel_size=3,
        padding=1,
        nonlinearity=torch.relu,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_features)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.nonlinearity(h)
        return h


class UnetBlock(nn.Module):
    def __init__(
        self,
        nf,
        ni,
        submodule=None,
        input_c=None,
        dropout=False,
        innermost=False,
        outermost=False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_c is None:
            input_c = nf
        downconv = nn.Conv2d(
            input_c, ni, kernel_size=4, stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                ni, nf, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(
                num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True
            )
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(
            output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True
        )

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [
            self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)
        ]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(
        self, ni, nf, k=4, s=2, p=1, norm=True, act=True
    ):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)
        ]  # it's always helpful to make a separate method for that purpose
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, init="norm", gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and "Conv" in classname:
            if init == "norm":
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class LTBC(pl.LightningModule):
    def __init__(self, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.lambda_L1 = lambda_L1
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.beta1 = beta1
        self.beta2 = beta2

        self.net_G = Unet()
        self.net_G = init_weights(self.net_G)
        self.net_D = PatchDiscriminator(3)
        self.net_D = init_weights(self.net_D)

        self.GANcriterion = GANLoss(gan_mode="vanilla")
        self.L1criterion = nn.L1Loss()

    def forward(self, inputs):
        return self.net_G(inputs)

    def configure_optimizers(self):
        # No further info about optimizer was provided in the paper, we only know it was adadelta
        # optim = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        opt_G = torch.optim.Adam(
            self.net_G.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2)
        )
        opt_D = torch.optim.Adam(
            self.net_D.parameters(), lr=self.lr_D, betas=(self.beta1, self.beta2)
        )

        return opt_G, opt_D

    def training_step(self, batch, batch_idx):
        self.net_G.train()
        self.net_D.train()
        image, label = batch
        opt_G, opt_D = self.optimizers()

        L = image[:, :1, :, :]
        ab = image[:, 1:, :, :]

        fake_color = self(L)
        fake_image = torch.cat([L, fake_color], dim=1)

        opt_D.zero_grad()
        fake_preds = self.net_D(fake_image.detach())
        loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([L, ab], dim=1)
        real_preds = self.net_D(real_image)
        loss_D_real = self.GANcriterion(real_preds, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()
        fake_preds = self.net_D(fake_image)
        loss_G_GAN = self.GANcriterion(fake_preds, True)
        loss_G_L1 = self.L1criterion(fake_color, ab) * self.lambda_L1
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        opt_G.step()

        self.log("loss_D_fake", loss_D_fake, prog_bar=True)
        self.log("loss_D_real", loss_D_real, prog_bar=True)
        self.log("loss_G_GAN", loss_G_GAN, prog_bar=True)
        self.log("loss_G_L1", loss_G_L1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.net_G.eval()
        self.net_D.eval()

        with torch.no_grad():
            image, label = batch
            L = image[:, :1, :, :]
            ab = image[:, 1:, :, :]
            fake_color = self(L)
            fake_image = torch.cat([L, fake_color], dim=1)
            real_image = torch.cat([L, ab], dim=1)

            fake_preds = self.net_D(fake_image)
            loss_D_fake = self.GANcriterion(fake_preds, False)
            real_preds = self.net_D(real_image)
            loss_D_real = self.GANcriterion(real_preds, True)
            loss_G_L1 = self.L1criterion(fake_color, ab) * self.lambda_L1
            loss_G_GAN = self.GANcriterion(fake_preds, True)

            self.log("val_loss_D_fake", loss_D_fake, prog_bar=True)
            self.log("val_loss_D_real", loss_D_real, prog_bar=True)
            self.log("val_loss_G_GAN", loss_G_GAN, prog_bar=True)
            self.log("val_loss_G_L1", loss_G_L1, prog_bar=True)
            self.log("val_loss", loss_G_L1)

    def log_images(self, L_image, ab_image, ab_pred):
        for img_idx in range(L_image.shape[0]):
            # Log one image out of 10
            if random() > 0.9:
                L_image = L_image.detach().cpu()
                ab_image = ab_image.detach().cpu()
                ab_pred = ab_pred.detach().cpu()
                gt_Lab_image = torch.cat([L_image, ab_image], dim=1)
                gt_rgb_image = convert_back_to_rgb(
                    gt_Lab_image.detach()[img_idx, :1, :, :],
                    gt_Lab_image.detach()[img_idx, 1:, :, :],
                )

                pred_Lab_image = torch.cat([L_image, ab_pred], dim=1)
                pred_rgb_image = convert_back_to_rgb(
                    pred_Lab_image.detach()[img_idx, :1, :, :],
                    pred_Lab_image.detach()[img_idx, 1:, :, :],
                )

                three_images = torch.stack(
                    [
                        torch.tensor(gt_rgb_image),
                        torch.tensor(pred_rgb_image),
                        torch.tensor(color.rgb2gray(pred_rgb_image))
                        .unsqueeze(2)
                        .expand(-1, -1, 3),
                    ]
                )
                side_by_side = torchvision.utils.make_grid(
                    three_images.permute(0, 3, 1, 2)
                )
                self.logger.experiment.add_image(
                    f"comparison: image {img_idx}", side_by_side, 0
                )
