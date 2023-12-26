from collections import OrderedDict
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .backbone_HECAmobilenetV3 import mobilenet_v3_large
from .Unet_decode import Up, Up_B, OutConv, _PSPModule
from torch.nn import functional as F


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}


        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class MFFLnet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(MFFLnet, self).__init__()

        norm_layer = nn.BatchNorm2d

        backbone = mobilenet_v3_large()

        if pretrain_backbone:

            model_path = 'mobilenet_v3_large.pth'
            print('Load weights {}.'.format(model_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict = backbone.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)

            # for k in list(pretrained_dict.keys()):
            #     if "aux_classifier" in k:
            #         del pretrained_dict[k]
            #
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and (v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_dict)
            backbone.load_state_dict(model_dict)


        backbone = backbone.features

        self.stage_out_channels = [16, 24, 40, 80, 160, 160]
        return_layers = {'0': 'out0', '2': 'out1', '4': 'out2', '7': 'out3', '13': 'out4', '15': 'out5'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.master_branch = nn.Sequential(
            _PSPModule(self.stage_out_channels[5], pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)
        )


        c = self.stage_out_channels[5] + self.stage_out_channels[4]
        self.up0 = Up_B(c, self.stage_out_channels[4])                                                                  # (160+160, 160)
        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])                                                                    # (160+80, 80)
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])                                                                    # (80+40, 40)
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])                                                                    # (40+24, 24)
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])                                                                    # (24+16, 16)

        self.conv = OutConv(16, num_classes=num_classes)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        result = OrderedDict()
        backbone_out = self.backbone(x)
        x = self.master_branch(backbone_out['out5'])
        x = self.up0(x, backbone_out['out4'])
        x = self.up1(x, backbone_out['out3'])
        x = self.up2(x, backbone_out['out2'])
        x = self.up3(x, backbone_out['out1'])
        x = self.up4(x, backbone_out['out0'])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = self.conv(x)

        result["out"] = x
        return result
