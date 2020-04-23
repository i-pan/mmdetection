# EfficientNet feature extractor from:
# https://github.com/rwightman/pytorch-image-models
import torch.nn as nn

from timm.models.efficientnet import *
from ..registry import BACKBONES


@BACKBONES.register_module
class EfficientNet(nn.Module):


    def __init__(self, name):
        super().__init__()
        self.net = eval(name)(features_only=True, out_indices=(2,3,4))
        self._name = name


    def init_weights(self, pretrained=None):
        if pretrained:
            self.net = eval(self._name)(features_only=True, out_indices=(2,3,4), pretrained=True)


    def forward(self, x):
        return self.net(x)






