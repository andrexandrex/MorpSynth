import copy
import math
import torch
import numpy as np
import torch.nn as nn
import timm  # <- NUEVO
from torch import Tensor
from functools import partial
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.efficientnet import (
    _MBConvConfig,
    MBConvConfig,
    FusedMBConvConfig,
    _efficientnet_conf,
) 
from torchvision.models.efficientnet import (
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)

from collections import OrderedDict
from os.path import join as pjoin

class CustomResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        block=BasicBlock,
        zero_init_residual=False,
        groups=1,
        num_classes=1000,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        """
        CustomResNet class to build the CustomResNet encoder model

        ----------
        Attributes
        ----------
        layers : list
            list of number of layers in each residual block
        block : object of block type
            type of the residual block (options = [BasicBlock, Bottleneck])
        zero_init_residual : bool
            to indicate whether to use zero weights for BN
        groups : int
            indicates the number of groups (default: 1)
        num_classes : int
            indicates the number of classes (default: 1000)
        width_per_group : int
            indicates the width per group (default: 64)
        replace_stride_with_dilation : list
            a list indicating whether to replace stride with dilation (default: None)
        norm_layer : object
            object of type batch norm (default: None)
        """

        super(CustomResNet, self).__init__()

        self.dict_encoder_features = {}

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ---------
        Arguments
        ---------
        x : torch tensor
            a tensor of input features

        -------
        Returns
        -------
        x : torch tensor
            output of the CustomResNet
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        self.dict_encoder_features["block_1"] = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class CustomResNetForUNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        block=BasicBlock,
        zero_init_residual=False,
        groups=1,
        num_classes=1000,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(CustomResNetForUNet, self).__init__()

        self.dict_encoder_features = {}

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        # Initialization remains the same
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size, 3, H, W)
        x0 = self.conv1(x)    # Shape: (batch_size, 64, H/2, W/2)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0) # Shape: (batch_size, 64, H/4, W/4)
        self.dict_encoder_features["x0"] = x0

        x1 = self.layer1(x0)  # Shape: (batch_size, 64, H/4, W/4)
        self.dict_encoder_features["x1"] = x1

        x2 = self.layer2(x1)  # Shape: (batch_size, 128, H/8, W/8)
        self.dict_encoder_features["x2"] = x2

        x3 = self.layer3(x2)  # Shape: (batch_size, 256, H/16, W/16)
        self.dict_encoder_features["x3"] = x3

        x4 = self.layer4(x3)  # Shape: (batch_size, 512, H/32, W/32)
        self.dict_encoder_features["x4"] = x4

        return x4  # Bottleneck features

def resnet34_unet_encoder(pretrained=True):
    if pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(
        block_type=BasicBlock,
        layers=[3, 4, 6, 3],
        weights=weights,
        model_class=CustomResNetForUNet,
        exclude_fc=True  # Exclude 'fc' layer weights
    )


def _resnet(block_type, layers, weights=None, progress=True, model_class=CustomResNet, exclude_fc=False):
    model = model_class(layers, block=block_type)
    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress)
        if exclude_fc:
            # Filter out 'fc.weight' and 'fc.bias'
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=True):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    ---------
    Arguments
    ---------
    pretrained : bool
        if True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(BasicBlock, [2, 2, 2, 2], weights=weights)


def resnet34(pretrained=True):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    ---------
    Arguments
    ---------
    pretrained : bool
        if True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(BasicBlock, [3, 4, 6, 3], weights=weights)


def resnet50(pretrained=True):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    ---------
    Arguments
    ---------
    pretrained : bool
        if True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(Bottleneck, [3, 4, 6, 3], weights=weights)


def resnet101(pretrained=True):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    ---------
    Arguments
    ---------
    pretrained : bool
        if True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(Bottleneck, [3, 4, 23, 3], weights=weights)


class CustomEfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        EfficientNet V1 and V2 main class

        ----------
        Attributes
        ----------
            inverted_residual_setting : Sequence
                network structure
            dropout : float
                the droupout probability
            stochastic_depth_prob : float
                the stochastic depth probability
            num_classes : int
                number of classes
            norm_layer : object
                object of type Module specifying the normalization layer to use
            last_channel : int
                the number of channels on the penultimate layer
        """
        super().__init__()
        self.dict_encoder_features = {}

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.num_channels_final_block = lastconv_output_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return x


def _efficientnet(
    inverted_residual_setting,
    dropout: float,
    last_channel,
    weights=None,
    norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
    progress=True,
    **kwargs: Any,
):
    """
    ---------
    Arguments
    ---------
        inverted_residual_setting : Sequence
            network structure
        dropout : float
            the droupout probability
        last_channel : last_channel
            the last channel
        weights : object
            object of type efficient_net weights
        norm_layer : object
            object of type Module specifying the normalization layer to use
        progress : bool
            indicates whether to show progress or not
    """
    model = CustomEfficientNet(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        norm_layer=norm_layer,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model




def efficientnet_v2_m(pretrained=True, **kwargs: Any):
    """
    ---------
    Arguments
    ---------
    pretrained : bool
        if True, returns a model pre-trained on ImageNet
    **kwargs :
        additional arguments
    """
    which_efficientnet = "efficientnet_v2_m"
    inverted_residual_setting, last_channel = _efficientnet_conf(which_efficientnet)
    if pretrained:
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        weights=weights,
        **kwargs,
    )



class TransUNetTransformer(nn.Module):
    def __init__(self, config, img_size, vis=False):
        super(TransUNetTransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)
        
    def forward(self, x):
        x, features = self.embeddings(x)
        x, attn_weights = self.encoder(x)
        return x, attn_weights, features
    def load_from(self, weights):
        # Load embeddings
        self.embeddings.load_from(weights)
        # Load encoder layers
        for i, block in enumerate(self.encoder.layer):
            block.load_from(weights, n_block=i)
        # Load encoder norm
        with torch.no_grad():
            self.encoder.encoder_norm.weight.copy_(torch.from_numpy(weights['Transformer/encoder_norm/scale']))
            self.encoder.encoder_norm.bias.copy_(torch.from_numpy(weights['Transformer/encoder_norm/bias']))
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        for _ in range(config['transformer']["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
from torch.nn.modules.utils import _pair
class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        if config["patches"].get("grid") is not None:
            grid_size = config["patches"]["grid"]
            self.hybrid = True
            # The total downsampling factor of ResNetV2 is 32
            downsampling_factor = 16
            output_size = (
                img_size[0] // downsampling_factor,
                img_size[1] // downsampling_factor,
            )
            # Compute patch size based on output size and grid size
            patch_size = (
                output_size[0] // grid_size[0],
                output_size[1] // grid_size[1],
            )
            n_patches = grid_size[0] * grid_size[1]
            if patch_size[0] == 0 or patch_size[1] == 0:
                patch_size = (1, 1)
        else:
            self.hybrid = False
            patch_size = _pair(config["patches"]["size"])
            n_patches = (
                img_size[0] // patch_size[0]
            ) * (img_size[1] // patch_size[1])

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config["resnet"]["num_layers"],
                width_factor=config["resnet"]["width_factor"],
            )
            in_channels = self.hybrid_model.out_channels  # Should be 1024

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config["hidden_size"],
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches + 1, config["hidden_size"])
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["transformer"]["dropout_rate"])



    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.hybrid:
            x, features = self.hybrid_model(x)
            #print(f"After backbone: x.shape = {x.shape}")
        else:
            features = None
        x = self.patch_embeddings(x)
        #print(f"After patch_embeddings: x.shape = {x.shape}")
        x = x.flatten(2).transpose(-1, -2)
        #print(f"After flatten and transpose: x.shape = {x.shape}")
        x = torch.cat((cls_tokens, x), dim=1)
        #print(f"After adding cls_tokens: x.shape = {x.shape}")
        #print(f"Position embeddings shape: {self.position_embeddings.shape}")
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features
    def load_from(self, weights):
        with torch.no_grad():
            # Load patch embeddings
            pretrained_patch_embed = np2th(weights["embedding/kernel"], conv=False)
            # Permute to match PyTorch's [out_channels, in_channels, kernel_height, kernel_width]
            pretrained_patch_embed = pretrained_patch_embed.permute(3, 2, 0, 1)
            self.patch_embeddings.weight.copy_(pretrained_patch_embed)
            self.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            #print(f"Model's patch_embeddings.weight.shape: {self.patch_embeddings.weight.shape}")

            #print(f"Pre-trained patch embeddings shape after permutation: {pretrained_patch_embed.shape}")
            # Load CLS token
            self.cls_token.copy_(np2th(weights["cls"]))

            # Load and resize positional embeddings (unchanged)
            pretrained_pos_emb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            cls_pos_emb = pretrained_pos_emb[:, :1, :]  # [1, 1, hidden_size]
            pos_emb = pretrained_pos_emb[:, 1:, :]      # [1, n_patches_pretrained, hidden_size]

            n_patches_pretrained = pos_emb.shape[1]
            n_patches_new = self.position_embeddings.shape[1] - 1

            if n_patches_pretrained != n_patches_new:
                #print(f"Resizing positional embeddings from {n_patches_pretrained} to {n_patches_new}")
                # Reshape and interpolate
                gs_old = int(np.sqrt(n_patches_pretrained))
                gs_new = int(np.sqrt(n_patches_new))

                pos_emb = pos_emb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                pos_emb = F.interpolate(pos_emb, size=(gs_new, gs_new), mode='bilinear', align_corners=False)
                pos_emb = pos_emb.permute(0, 2, 3, 1).reshape(1, n_patches_new, -1)

            new_pos_emb = torch.cat((cls_pos_emb, pos_emb), dim=1)
            self.position_embeddings.copy_(new_pos_emb)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": lambda x: x * torch.sigmoid(x)
}
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config['hidden_size']
        self.attention_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config['transformer']["num_heads"]
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config['hidden_size'], self.all_head_size)
        self.key = Linear(config['hidden_size'], self.all_head_size)
        self.value = Linear(config['hidden_size'], self.all_head_size)

        self.out = Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = Dropout(config['transformer']["attention_dropout_rate"])
        self.proj_dropout = Dropout(config['transformer']["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


def swish(x):
    return x * torch.sigmoid(x)

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config['hidden_size'], config['transformer']["mlp_dim"])
        self.fc2 = Linear(config['transformer']["mlp_dim"], config['hidden_size'])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config['transformer']["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


#####################RESNET_V2CODE################################################################################################

from collections import OrderedDict
from os.path import join as pjoin



def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet model."""

    def __init__(self, block_units, width_factor):
        super(ResNetV2, self).__init__()
        width = int(64 * width_factor)
        self.width = width

        # Root (conv1)
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Body (conv2_x to conv4_x) - Only three blocks
        self.body = nn.ModuleList()
        in_channels = [width, width * 4, width * 8]  # [64, 256, 512]
        out_channels = [width * 4, width * 8, width * 16]  # [256, 512, 1024]
        strides = [1, 2, 2]  # Stride 1 for conv2_x, stride 2 for others

        for idx, num_blocks in enumerate(block_units):
            blocks = []
            cin = in_channels[idx]
            cout = out_channels[idx]
            cmid = width * (2 ** idx)  # cmid doubles with each block
            stride = strides[idx]

            # First block in each layer may have different input channels and stride
            blocks.append(('unit1', PreActBottleneck(cin=cin, cout=cout, cmid=cmid, stride=stride)))
            for i in range(1, num_blocks):
                blocks.append((f'unit{i + 1}', PreActBottleneck(cin=cout, cout=cout, cmid=cmid)))
            self.body.append(nn.Sequential(OrderedDict(blocks)))

        # Store output channels for use in Embeddings
        self.out_channels = out_channels[-1]  # This will be 1024

    # In the ResNetV2 forward method
    def forward(self, x):
        features = []
        x = self.root(x)
        #print(f"After root: x.shape = {x.shape}")
        features.append(x)
        for idx, block in enumerate(self.body):
            x = block(x)
            #print(f"After block {idx}: x.shape = {x.shape}")
            features.append(x)
        return x, features
    

class SwinTinyEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        pretrained=True,
        backbone='swin_tiny_patch4_window7_224',
        target_channels=(96,192,384,768),
        img_size=512,                      # <-- añade este parámetro
    ):
        super().__init__()
        self.sar_to_rgb = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,1,2,3),
            img_size=img_size,             # <-- pásalo aquí
        )
        feat_ch = self.backbone.feature_info.channels()
        self.adapters = nn.ModuleList([
            nn.Conv2d(cin, cout, 1, bias=False) for cin, cout in zip(feat_ch, target_channels)
        ])

    def forward(self, x):
        x = self.sar_to_rgb(x)
        feats = self.backbone(x)  # lista de 4 tensores

        fixed = []
        for f in feats:
            # Si el canal viene al final (NHWC), permutamos a NCHW
            # Detectamos esto viendo si el 2º dim no parece "C" pero el último sí.
            if f.ndim == 4 and f.shape[1] not in (96, 192, 384, 768) and f.shape[-1] in (96, 192, 384, 768):
                f = f.permute(0, 3, 1, 2).contiguous()
            fixed.append(f)

        feats = [ad(f) for ad, f in zip(self.adapters, fixed)]
        return feats  # [f0,f1,f2,f3] en NCHW
