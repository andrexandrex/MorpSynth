import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from decoder_models import DeepLabV3Plus,UNetDecoder,DeepLabV3, ASPPBlock,TransUNetDecoder
from encoder_models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    efficientnet_v2_m,
    resnet34_unet_encoder,TransUNetTransformer,SwinTinyEncoder,  # <- NUEVO
)

class ResNet34UNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet34UNet, self).__init__()

        self.encoder = resnet34_unet_encoder(pretrained=pretrained)

        encoder_channels = [512, 256, 128, 64, 64]
        decoder_channels = [256, 128, 64, 32]

        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]

        # Encoder forward pass
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        x0 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x0)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        # Collect features for decoder
        encoder_features = [x4, x3, x2, x1, x0]

        # Decoder forward pass
        x = self.decoder(encoder_features)

        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet34UNetWithASPP(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet34UNetWithASPP, self).__init__()

        self.encoder = resnet34(pretrained=pretrained)

        self.aspp = ASPPBlock(
            in_channels=512, atrous_rates=[12, 24, 36], aspp_out_channels=256
        )

        encoder_channels = [256, 256, 128, 64, 64]  # Adjusted because ASPP output is 256
        decoder_channels = [256, 128, 64, 32]

        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]

        # Encoder forward pass
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        x0 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x0)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        # Apply ASPP module to the bottleneck features
        x4 = self.aspp(x4)

        # Collect features for decoder
        encoder_features = [x4, x3, x2, x1, x0]

        # Decoder forward pass
        x = self.decoder(encoder_features)

        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet18DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet18(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=512, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet34DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet34(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=512, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet50DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet50(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=2048, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet101DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet101(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=2048, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class EfficientNetSDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = efficientnet_v2_s(pretrained=pretrained)
        self.segmenter = DeepLabV3(
            in_channels=self.encoder.num_channels_final_block,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class EfficientNetMDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = efficientnet_v2_m(pretrained=pretrained)
        self.segmenter = DeepLabV3(
            in_channels=self.encoder.num_channels_final_block,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class EfficientNetLDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = efficientnet_v2_l(pretrained=pretrained)
        self.segmenter = DeepLabV3(
            in_channels=self.encoder.num_channels_final_block,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x

class TransUNet(nn.Module):
    def __init__(self, config, img_size, num_classes, vis=False):
        super(TransUNet, self).__init__()
        self.transformer = TransUNetTransformer(config, img_size, vis)
        self.decoder = TransUNetDecoder(config)
        self.segmentation_head = nn.Conv2d(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=1
        )
        self.config = config

    def forward(self, x):
        input_shape = x.shape[2:]
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        x, attn_weights, features = self.transformer(x)
        #print('haaaa',x)
        x = x[:, 1:, :]  # Exclude cls_token
        #print(x)
        x = self.decoder(x, features)
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    def load_from(self, weights):
        self.transformer.load_from(weights)
        #self.decoder.load_from(weights)

class SwinUNetTiny(nn.Module):
    """
    Encoder Swin-Tiny (timm, ImageNet) + UNetDecoder de la casa.
    Entrega logits del tamaño de entrada (H,W).
    """
    def __init__(self, num_classes, in_channels=2, pretrained=True, image_size=512):
        super().__init__()

        # Encoder Swin (devuelve [f0,f1,f2,f3] con canales [96,192,384,768])
        self.encoder = SwinTinyEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            img_size=image_size
        )

        # Tu UNetDecoder espera 5 entradas: [bottleneck, skip3, skip2, skip1, skip0]
        # Con Swin: f3(768), f2(384), f1(192), f0(96). Duplicamos f0 como último skip.
        encoder_channels = [768, 384, 192, 96, 96]   # f3, f2, f1, f0, f0
        decoder_channels = [384, 192, 96, 96]        # ← AJUSTE CLAVE
        
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,   # ← tu decoder ya saca logits; NO hace falta head extra
        )

        self.image_size = image_size
        self.in_channels = in_channels

    def forward(self, x):
        H, W = x.shape[-2:]

        # Tolerancia por si llega 1 o 3 canales
        if x.size(1) == 1 and self.in_channels == 2:
            x = torch.cat([x, x], dim=1)          # 1→2
        elif x.size(1) == 3 and self.in_channels == 2:
            x = x[:, :2, :, :]                    # 3→2 (usa VV,VH)

        f0, f1, f2, f3 = self.encoder(x)          # [H/4,H/8,H/16,H/32]
        enc_feats = [f3, f2, f1, f0, f0]          # [x4,x3,x2,x1,x0]
        y = self.decoder(enc_feats)               # logits a baja resolución interna

        # Asegura salida del tamaño de entrada
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y

