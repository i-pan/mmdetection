import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from typing import List, Optional
from timm.models.efficientnet import *
from timm import create_model
from timm.models.layers import create_conv2d, drop_path, create_pool2d, Swish
from ..registry import NECKS

_DEBUG = False

_ACT_LAYER = Swish


"""EfficientDet Configurations

Adapted from official impl at https://github.com/google/automl/tree/master/efficientdet

TODO use a different config system, separate model from train specific hparams
"""

import ast
import copy
import json
import six


def eval_str_fn(val):
    if val in {'true', 'false'}:
        return val == 'true'
    try:
        return ast.literal_eval(val)
    except ValueError:
        return val


# pylint: disable=protected-access
class Config(object):
    """A config utility class."""

    def __init__(self, config_dict=None):
        self.update(config_dict)

    def __setattr__(self, k, v):
        self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)

    def __getattr__(self, k):
        return self.__dict__[k]

    def __repr__(self):
        return repr(self.as_dict())

    def __str__(self):
        try:
            return json.dumps(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())

    def _update(self, config_dict, allow_new_keys=True):
        """Recursively update internal members."""
        if not config_dict:
            return

        for k, v in six.iteritems(config_dict):
            if k not in self.__dict__.keys():
                if allow_new_keys:
                    self.__setattr__(k, v)
                else:
                    raise KeyError('Key `{}` does not exist for overriding. '.format(k))
            else:
                if isinstance(v, dict):
                    self.__dict__[k]._update(v, allow_new_keys)
                else:
                    self.__dict__[k] = copy.deepcopy(v)

    def get(self, k, default_value=None):
        return self.__dict__.get(k, default_value)

    def update(self, config_dict):
        """Update members while allowing new keys."""
        self._update(config_dict, allow_new_keys=True)

    def override(self, config_dict_or_str):
        """Update members while disallowing new keys."""
        if isinstance(config_dict_or_str, str):
            config_dict = self.parse_from_str(config_dict_or_str)
        elif isinstance(config_dict_or_str, dict):
            config_dict = config_dict_or_str
        else:
            raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

        self._update(config_dict, allow_new_keys=False)

    def parse_from_str(self, config_str):
        """parse from a string in format 'x=a,y=2' and return the dict."""
        if not config_str:
            return {}
        config_dict = {}
        try:
            for kv_pair in config_str.split(','):
                if not kv_pair:  # skip empty string
                    continue
                k, v = kv_pair.split('=')
                config_dict[k.strip()] = eval_str_fn(v.strip())
            return config_dict
        except ValueError:
            raise ValueError('Invalid config_str: {}'.format(config_str))

    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            else:
                config_dict[k] = copy.deepcopy(v)
        return config_dict


def default_detection_configs():
    """Returns a default detection configs."""
    h = Config()

    # model name.
    h.name = 'tf_efficientdet_d1'

    # input preprocessing parameters
    h.image_size = 640
    h.input_rand_hflip = True
    h.train_scale_min = 0.1
    h.train_scale_max = 2.0
    h.autoaugment_policy = None

    # dataset specific parameters
    h.num_classes = 90
    h.skip_crowd_during_training = True

    # model architecture
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    h.anchor_scale = 4.0
    h.pad_type = 'same'

    # is batchnorm training mode
    h.is_training_bn = True

    # optimization
    h.momentum = 0.9
    h.learning_rate = 0.08
    h.lr_warmup_init = 0.008
    h.lr_warmup_epoch = 1.0
    h.first_lr_drop_epoch = 200.0
    h.second_lr_drop_epoch = 250.0
    h.clip_gradients_norm = 10.0
    h.num_epochs = 300

    # classification loss
    h.alpha = 0.25
    h.gamma = 1.5

    # localization loss
    h.delta = 0.1
    h.box_loss_weight = 50.0

    # regularization l2 loss.
    h.weight_decay = 4e-5

    # For detection.
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_relu_pattern = False
    h.use_native_resize_op = False
    h.pooling_type = None

    # version.
    h.fpn_name = None
    h.fpn_config = None

    # No stochastic depth in default.
    h.survival_prob = None  # FIXME remove
    h.drop_path_rate = 0.

    h.lr_decay_method = 'cosine'
    h.moving_average_decay = 0.9998
    h.ckpt_var_scope = None
    h.backbone_name = 'tf_efficientnet_b1'
    h.backbone_config = None

    # RetinaNet.
    h.resnet_depth = 50
    return h


efficientdet_model_param_dict = {
    'tf_efficientdet_d0':
        dict(
            name='efficientdet_d0',
            backbone_name='tf_efficientnet_b0',
            image_size=512,
            fpn_channels=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    'tf_efficientdet_d1':
        dict(
            name='efficientdet_d1',
            backbone_name='tf_efficientnet_b1',
            image_size=640,
            fpn_channels=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    'tf_efficientdet_d2':
        dict(
            name='efficientdet_d2',
            backbone_name='tf_efficientnet_b2',
            image_size=768,
            fpn_channels=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
        ),
    'tf_efficientdet_d3':
        dict(
            name='efficientdet_d3',
            backbone_name='tf_efficientnet_b3',
            image_size=896,
            fpn_channels=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    'tf_efficientdet_d4':
        dict(
            name='efficientdet_d4',
            backbone_name='tf_efficientnet_b4',
            image_size=1024,
            fpn_channels=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'tf_efficientdet_d5':
        dict(
            name='efficientdet_d5',
            backbone_name='tf_efficientnet_b5',
            image_size=1280,
            fpn_channels=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'tf_efficientdet_d6':
        dict(
            name='efficientdet_d6',
            backbone_name='tf_efficientnet_b6',
            image_size=1280,
            fpn_channels=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        ),
    'tf_efficientdet_d7':
        dict(
            name='efficientdet_d7',
            backbone_name='tf_efficientnet_b6',
            image_size=1536,
            fpn_channels=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        ),
}


def get_efficientdet_config(model_name='efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_configs()
    h.override(efficientdet_model_param_dict[model_name])
    return h


class SequentialAppend(nn.Sequential):
    def __init__(self, *args):
        super(SequentialAppend, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x))
        return x


class SequentialAppendLast(nn.Sequential):
    def __init__(self, *args):
        super(SequentialAppendLast, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x[-1]))
        return x


class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels, **norm_kwargs)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, act_layer=_ACT_LAYER,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(SeparableConv2d, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels, **norm_kwargs)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResampleFeatureMap(nn.Sequential):

    def __init__(self, in_channels, out_channels, reduction_ratio=1., pad_type='', pooling_type='max',
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, conv_after_downsample=False, apply_bn=False):
        super(ResampleFeatureMap, self).__init__()
        pooling_type = pooling_type or 'max'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None, norm_kwargs=norm_kwargs, bias=True, act_layer=None)

        if reduction_ratio > 1:
            stride_size = int(reduction_ratio)
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            self.add_module(
                'downsample',
                create_pool2d(
                    pooling_type, kernel_size=stride_size + 1, stride=stride_size, padding=pad_type))
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=scale))

    # def forward(self, x):
    #     #  here for debugging only
    #     assert x.shape[1] == self.in_channels
    #     if self.reduction_ratio > 1:
    #         if hasattr(self, 'conv') and not self.conv_after_downsample:
    #             x = self.conv(x)
    #         x = self.downsample(x)
    #         if hasattr(self, 'conv') and self.conv_after_downsample:
    #             x = self.conv(x)
    #     else:
    #         if hasattr(self, 'conv'):
    #             x = self.conv(x)
    #         if self.reduction_ratio < 1:
    #             x = self.upsample(x)
    #     return x


class FPNCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='',
                 pooling_type='max', norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 apply_bn_for_resampling=False, conv_after_downsample=False, weight_method='attn'):
        super(FPNCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels, fpn_channels, reduction_ratio=reduction_ratio, pad_type=pad_type,
                pooling_type=pooling_type, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                apply_bn=apply_bn_for_resampling, conv_after_downsample=conv_after_downsample)

        if weight_method == 'attn' or weight_method == 'fastattn':
            # WSM
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        for offset in self.inputs_offsets:
            input_node = x[offset]
            input_node = self.resample[str(offset)](input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.type(dtype), dim=0)
            x = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.type(dtype))
            weights_sum = torch.sum(edge_weights)
            x = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            x = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        x = torch.sum(x, dim=-1)
        return x


class BiFPNLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 pooling_type='max', norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER,
                 apply_bn_for_resampling=False, conv_after_downsample=True, conv_bn_relu_pattern=False,
                 separable_conv=True):
        super(BiFPNLayer, self).__init__()
        self.fpn_config = fpn_config
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = []
        self.fnode = SequentialAppend()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            fnode_layers = OrderedDict()

            # combine features
            reduction = fnode_cfg['reduction']
            fnode_layers['combine'] = FPNCombine(
                feature_info, fpn_config, fpn_channels, fnode_cfg['inputs_offsets'], target_reduction=reduction,
                pad_type=pad_type, pooling_type=pooling_type, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                apply_bn_for_resampling=apply_bn_for_resampling, conv_after_downsample=conv_after_downsample,
                weight_method=fpn_config.weight_method)
            self.feature_info.append(dict(num_chs=fpn_channels, reduction=reduction))

            # after combine ops
            after_combine = OrderedDict()
            if not conv_bn_relu_pattern:
                after_combine['act'] = act_layer(inplace=True)
                conv_bias = True
                conv_act = None
            else:
                conv_bias = False
                conv_act = act_layer
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=conv_bias, norm_layer=norm_layer, norm_kwargs=norm_kwargs, act_layer=conv_act)
            after_combine['conv'] = SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs)
            fnode_layers['after_combine'] = nn.Sequential(after_combine)

            self.fnode.add_module(str(i), nn.Sequential(fnode_layers))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x):
        x = self.fnode(x)
        return x[-self.num_levels::]


def bifpn_sum_config(base_reduction=8):
    """BiFPN config with sum."""
    p = Config()
    p.nodes = [
        {'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]},
        {'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]},
        {'reduction': base_reduction, 'inputs_offsets': [0, 7]},
        {'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]},
        {'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]},
        {'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]},
    ]
    p.weight_method = 'sum'
    return p


def bifpn_attn_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'attn'
    return p


def bifpn_fa_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'fastattn'
    return p


def get_fpn_config(fpn_name):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_sum_config(),
        'bifpn_attn': bifpn_attn_config(),
        'bifpn_fa': bifpn_fa_config(),
    }
    return name_to_config[fpn_name]


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self, name, norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER):
        super(BiFPN, self).__init__()
        config = get_efficientdet_config(name)
        backbone_name = name.replace('det_d', 'net_b')
        self.config = config
        fpn_config = config.fpn_config or get_fpn_config(config.fpn_name)
        backbone = eval(backbone_name)(features_only=True, out_indices=(2,3,4))
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction'])
                        for i, f in enumerate(backbone.feature_info())]
        del backbone
        self.resample = SequentialAppendLast()
        for level in range(config.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample.add_module(str(level), ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    pad_type=config.pad_type,
                    pooling_type=config.pooling_type,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    reduction_ratio=reduction_ratio,
                    apply_bn=config.apply_bn_for_resampling,
                    conv_after_downsample=config.conv_after_downsample,
                ))
                in_chs = config.fpn_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = nn.Sequential()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFPNLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                pooling_type=config.pooling_type,
                norm_layer=norm_layer,
                norm_kwargs=norm_kwargs,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_bn_for_resampling=config.apply_bn_for_resampling,
                conv_after_downsample=config.conv_after_downsample,
                conv_bn_relu_pattern=config.conv_bn_relu_pattern
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

        # FIXME init weights for training

    def forward(self, x):
        assert len(self.resample) == self.config.num_levels - len(x)
        x = self.resample(x)
        x = self.cell(x)
        return x


    def init_weights(self, pretrained=None):
        pass