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
import glob
import os.path

import numpy as np
import pandas as pd
from PIL import Image

from blueoil.datasets.base import SegmentationBase
from blueoil.utils.random import shuffle


def get_image(filename, convert_rgb=True, ignore_class_idx=None):
    """Returns numpy array of an image"""
    image = Image.open(filename)
    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
        image = np.array(image)
    else:
        image = image.convert("L")
        image_bw = image.point(lambda x: 0 if x < 128 else 255, '1')
        image = np.array(image_bw) 
        if ignore_class_idx is not None:
            # Replace ignore labelled class with enough large value
            image = np.where(image == ignore_class_idx, 255, image)
            image = np.where((image > ignore_class_idx) & (image != 255), image - 1, image)

    return image


class CrackBase(SegmentationBase):
    """Base class for CrackDetection datasets"""
    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    extend_dir = None
    ignore_class_idx = None

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations[0])

    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            text = "train.txt"

        if self.subset == "validation":
            text = "val.txt"

        filename = os.path.join(self.data_dir, text)
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['image_files', 'label_files'],
        )

        image_files = df.image_files.tolist()
        label_files = df.label_files.tolist()

        image_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in image_files]
        label_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in label_files]

        return image_files, label_files

    def __getitem__(self, i):
        image_files, label_files = self.files_and_annotations

        image = get_image(image_files[i])
        label = get_image(label_files[i], convert_rgb=False, ignore_class_idx=self.ignore_class_idx).copy()

        return (image, label)

    def __len__(self):
        return self.num_per_epoch


class DeepCrack(CrackBase):
    """DeepCrack dataset
    https://github.com/yhlleo/DeepCrack/blob/master/dataset/DeepCrack.zip
    """

    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 300
    NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 237

    classes = [
        "background",
        "crack"
    ]
    num_classes = len(classes)

    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @property
    def label_colors(self):
        crack = [128, 128, 128]
        background = [192, 192, 128]

        label_colors = np.array([background, crack])

        return label_colors

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            imgs = "train_img"
            labs = "train_lab"

        if self.subset == "validation":
            imgs = "test_img"
            labs = "test_lab"

        img_dir = os.path.join(self.data_dir, imgs)
        lab_dir = os.path.join(self.data_dir, labs)
        image_files = sorted(glob.glob(img_dir + '/*'))
        label_files = sorted(glob.glob(lab_dir + '/*'))

        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files
