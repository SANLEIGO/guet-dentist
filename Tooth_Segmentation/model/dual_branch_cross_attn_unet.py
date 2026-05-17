import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List

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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x



class CNNTransformerFusion(nn.Module):
    def __init__(self, channels, mlp_ratio=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        mlp_hidden = channels * mlp_ratio

        # MLP for interaction learning
        self.mlp_interact = nn.Sequential(
            nn.Linear(channels * 2, channels ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels , channels),
            nn.Dropout(0.1),
        )

        # 特征增强 MLP
        self.mlp_cnn = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, channels),
        )
        self.mlp_trans = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, channels),
        )
        self.cnn_fuse = nn.Sequential(
            ConvBNReLU(channels,channels,3,1),
        )



    def forward(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape

        a = feat_a.flatten(2).transpose(1, 2)  # (B, HW, C)
        b = feat_b.flatten(2).transpose(1, 2)
        interaction = self.mlp_interact(torch.cat([a, b], dim=-1))
        fused = interaction + self.mlp_cnn(self.norm1(a)) + self.mlp_trans(self.norm2(b))
        out = fused.transpose(1, 2).reshape(B, C, H, W)

        return self.cnn_fuse(out)





class PretrainedSwinEncoder(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, img_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBNReLU(32, 32),
        )
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )
        self.out_channels = [32, 96, 192, 384, 768]

    def _to_nchw(self, feat, channels):
        if feat.dim() != 4:
            raise ValueError("Swin feature must be a 4D tensor")
        if feat.shape[1] == channels:
            return feat.contiguous()
        if feat.shape[-1] == channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unexpected Swin feature shape {tuple(feat.shape)} for {channels} channels")

    def forward(self, x):
        feat1 = self.stem(x)
        swin_feats = self.backbone(x)
        feat2 = self._to_nchw(swin_feats[0], 96)
        feat3 = self._to_nchw(swin_feats[1], 192)
        feat4 = self._to_nchw(swin_feats[2], 384)
        feat5 = self._to_nchw(swin_feats[3], 768)
        return [feat1, feat2, feat3, feat4, feat5]


LightweightTransformerEncoder = PretrainedSwinEncoder


class DecoderGuidedFusion(nn.Module):
    def __init__(self, cnn_channels, trans_channels, decoder_channels, out_channels):
        super().__init__()

        self.decoder_proj = ConvBNReLU(decoder_channels, out_channels, 1,0)

        self.cnn_trans_fuse = CNNTransformerFusion(out_channels)

        # self.weight_gate = nn.Sequential(
        #     ConvBNReLU(out_channels * 2, out_channels, 1,0),
        #     nn.Conv2d(out_channels, 2, 1)
        # )

        # self.refine = DoubleConv(out_channels, out_channels)
        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * 2, out_channels, 1,0),
        )

        self.up = UpBlock(out_channels,out_channels)

    def forward(self, cnn_feat, trans_feat, decoder_feat):

        dec_p = self.decoder_proj(decoder_feat)

        ct_fused = self.cnn_trans_fuse(cnn_feat, trans_feat)

        feat_cat = torch.cat([ct_fused, dec_p], dim=1)
        # weights = torch.softmax(self.weight_gate(feat_cat), dim=1)
        # w1, w2 = torch.chunk(weights, 2, dim=1)
        # fused = w1 * ct_fused + w2 * dec_p

        # final_feat = self.refine(fused)
        feat = self.fuse(feat_cat)
        final_feat = self.up(feat)
        return final_feat

class DualBranchCrossAttnUNet(nn.Module):
    def __init__(self, num_classes=21, return_aux=True):
        super().__init__()
        self.return_aux = return_aux
        self.cnn_encoder = resnet50()
        self.trans_encoder = LightweightTransformerEncoder()

        self.cnn4_proj = ConvBNReLU(2048, 1024, 1, 0)
        self.cnn3_proj = ConvBNReLU(1024, 512, 1, 0)
        self.cnn2_proj = ConvBNReLU(512,  256, 1, 0)
        self.cnn1_proj = ConvBNReLU(256,  128, 1, 0)


        self.trans4_proj = ConvBNReLU(768, 1024, 1, 0)
        self.trans3_proj = ConvBNReLU(384, 512, 1, 0)
        self.trans2_proj = ConvBNReLU(192, 256, 1, 0)
        self.trans1_proj = ConvBNReLU(96,  128, 1, 0)


        decoder_channels = [64, 128, 256, 512, 1024]

        self.bottleneck_fusion = ConvBNReLU(2048, 1024,1,0)



        self.skip_fuse4 = DecoderGuidedFusion(1024, 1024, 1024, decoder_channels[4])
        self.skip_fuse3 = DecoderGuidedFusion(512, 512, decoder_channels[4], decoder_channels[3])
        self.skip_fuse2 = DecoderGuidedFusion(256, 256, decoder_channels[3], decoder_channels[2])
        self.skip_fuse1 = DecoderGuidedFusion(128, 128, decoder_channels[2], decoder_channels[1])

        # self.boundary_refine = SimplifiedBoundaryHead(num_classes)

        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.boundary_head = nn.Sequential(
            ConvBNReLU(128, 64),
            ConvBNReLU(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=1),
        )



    def forward(self, inputs):
        _, cnn_feat1, cnn_feat2, cnn_feat3, cnn_feat4 = self.cnn_encoder(inputs)
        _, trans_feat1, trans_feat2, trans_feat3, trans_feat4 = self.trans_encoder(inputs)

        cnn_feat4 = self.cnn4_proj(cnn_feat4)
        cnn_feat3 = self.cnn3_proj(cnn_feat3)
        cnn_feat2 = self.cnn2_proj(cnn_feat2)
        cnn_feat1 = self.cnn1_proj(cnn_feat1)


        trans_feat4 = self.trans4_proj(trans_feat4)
        trans_feat3 = self.trans3_proj(trans_feat3)
        trans_feat2 = self.trans2_proj(trans_feat2)
        trans_feat1 = self.trans1_proj(trans_feat1)


        bottleneck = self.bottleneck_fusion(torch.cat([cnn_feat4, trans_feat4], dim=1))
        up4 = self.skip_fuse4(cnn_feat4, trans_feat4, bottleneck)
        up3 = self.skip_fuse3(cnn_feat3, trans_feat3, up4)
        up2 = self.skip_fuse2(cnn_feat2, trans_feat2, up3)
        up1 = self.skip_fuse1(cnn_feat1, trans_feat1, up2)
        seg_logits = self.seg_head(up1)
        boundary_logits = self.boundary_head(up1)
        # seg_logits, boundary_logits = self.boundary_refine(cnn_feat1, trans_feat1,up1)

        return {"seg": seg_logits, "boundary": boundary_logits}
        # return {"seg": seg_logits}



Unet = DualBranchCrossAttnUNet
