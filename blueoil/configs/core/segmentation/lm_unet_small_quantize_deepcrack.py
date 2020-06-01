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
from easydict import EasyDict
import tensorflow as tf

from blueoil.common import Tasks
from blueoil.networks.segmentation.lm_unet_small import LmUnetSmallQuantize
from blueoil.datasets.crackdetection import DeepCrack
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    DivideBy255,
)
from blueoil.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
)
from blueoil.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LmUnetSmallQuantize
DATASET_CLASS = DeepCrack

# Hyperparameter to tune: Input size
IMAGE_SIZE = [320, 480]
# Hyperparameter to tune: Batch size
BATCH_SIZE = 4
DATA_FORMAT = "NHWC"
TASK = Tasks.SEMANTIC_SEGMENTATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 150000
SAVE_CHECKPOINT_STEPS = 3000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 1000


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# for debug
# BATCH_SIZE = 2
# SUMMARISE_STEPS = 1
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255(),
])
POST_PROCESSOR = None

NETWORK = EasyDict()
# Hyperparameters to tune: Optim and LR / LR schedule
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.01}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.00005
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
# Lowest priority: Change augmentation args
DATASET.AUGMENTOR = Sequence([
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    FlipLeftRight(),
    Hue((-10, 10)),
])
# DATASET.ENABLE_PREFETCH = True
