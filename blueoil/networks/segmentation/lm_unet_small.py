# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import functools

import tensorflow as tf

from blueoil.blocks import conv_bn_act
from blueoil.networks.segmentation.base import SegnetBase


class LmUnetSmall(SegnetBase):
    """LM small U-net semantic segmentation network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = tf.nn.relu
        self.custom_getter = None

    def _get_lmnet_block(self, is_training, channels_data_format):
        data_format = 'NHWC' if channels_data_format == 'channels_last' else 'NCHW'
        return functools.partial(conv_bn_act,
                                 weight_decay_rate=self.weight_decay_rate,
				 activation=self.activation,
                                 is_training=is_training,
                                 data_format=data_format)

    def _space_to_depth(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])
        output = tf.space_to_depth(inputs, block_size=block_size, name=name)
        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def _depth_to_space(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])
        output = tf.depth_to_space(inputs, block_size=block_size, name=name)
        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def base(self, images, is_training, *args, **kwargs):
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images

        down1 = block('conv1', images, 32, 3)
        down2 = self._space_to_depth(name='space2depth1', inputs=down1)
        down2 = block('conv2', down2, 64, 3)
        down2 = block('conv3', down2, 64, 3)
        down3 = self._space_to_depth(name='space2depth2', inputs=down2)
        down3 = block('conv4', down3, 128, 3)
        bottom = self._space_to_depth(name='space2depth3', inputs=down3)
        bottom = block('conv5', bottom, 256, 3)
        up1 = self._depth_to_space(name='depth2space1', inputs=bottom)
        up1 = tf.concat([down3, up1], axis=-1)
        up1 = block('conv6', up1, 128, 3)
        up2 = self._depth_to_space(name='depth2space2', inputs=up1)
        up2 = tf.concat([down2, up2], axis=-1)
        up2 = block('conv7', up2, 64, 3)
        up2 = block('conv8', up2, 64, 3)
        up3 = self._depth_to_space(name='depth2space3', inputs=up2)
        up3 = tf.concat([down1, up3], axis=-1)
        up3 = block('conv9', up3, 32, 3)
        x = block('conv10', up3, self.num_classes, 3)
        self._heatmap_layer = x

        return x


class LmUnetSmallQuantize(LmUnetSmall):
    """LM original quantize semantic segmentation network.

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater.
            See more at `blueoil.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater.
            See more at `blueoil.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.

    """

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs={},
            weight_quantizer=None,
            weight_quantizer_kwargs={},
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.

        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.compat.v1.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
