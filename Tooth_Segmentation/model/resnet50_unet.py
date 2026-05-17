import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet_backbone import resnet50


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, decoder_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = ConvBNReLU(decoder_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, skip_feat, decoder_feat):
        x = torch.cat([skip_feat, self.up(decoder_feat)], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet50UNet(nn.Module):
    def __init__(self, num_classes=21, return_aux=True):
        super().__init__()
        self.return_aux = return_aux
        self.resnet = resnet50()

        decoder_channels = [64, 128, 256, 512]

        self.up_concat4 = DecoderBlock(2048, 1024, decoder_channels[3])
        self.up_concat3 = DecoderBlock(decoder_channels[3], 512, decoder_channels[2])
        self.up_concat2 = DecoderBlock(decoder_channels[2], 256, decoder_channels[1])
        self.up_concat1 = DecoderBlock(decoder_channels[1], 64, decoder_channels[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNReLU(decoder_channels[0], decoder_channels[0]),
            ConvBNReLU(decoder_channels[0], decoder_channels[0]),
        )

        self.final = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)

    def forward(self, inputs):
        feat1, feat2, feat3, feat4, feat5 = self.resnet(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.up_conv(up1)

        seg_logits = self.final(up1)

        if self.return_aux:
            return {"seg": seg_logits}
        return seg_logits


Unet = ResNet50UNet
