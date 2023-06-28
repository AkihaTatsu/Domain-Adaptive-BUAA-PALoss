# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Collection of available datasets."""

import os

from tasks.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/instances_val2017.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/Annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOCdevkit/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOCdevkit/VOC2007/Annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOCdevkit'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOCdevkit/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOCdevkit/VOC2007/Annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOCdevkit'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOCdevkit/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOCdevkit/VOC2012/Annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOCdevkit'
    },
    'voc_2012_test': {
        IM_DIR:
            _DATA_DIR + '/VOCdevkit/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOCdevkit/VOC2012/Annotations/voc_2012_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOCdevkit'
    },
    'clipart_train': {
        IM_DIR:
            _DATA_DIR + '/other/clipart/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/other/clipart/Annotations/clipart_train.json',
        DEVKIT_DIR:
            _DATA_DIR + '/other'
    },
    'clipart_test': {
        IM_DIR:
            _DATA_DIR + '/other/clipart/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/other/clipart/Annotations/clipart_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/other'
    },
    'comic_train': {
        IM_DIR:
            _DATA_DIR + '/other/comic/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/other/comic/Annotations/comic_train.json',
        DEVKIT_DIR:
            _DATA_DIR + '/other'
    },
    'comic_test': {
        IM_DIR:
            _DATA_DIR + '/other/comic/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/other/comic/Annotations/comic_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/other'
    },
    'watercolor_train': {
        IM_DIR:
            _DATA_DIR + '/other/watercolor/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/other/watercolor/Annotations/watercolor_train.json',
        DEVKIT_DIR:
            _DATA_DIR + '/other'
    },
    'watercolor_test': {
        IM_DIR:
            _DATA_DIR + '/other/watercolor/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/other/watercolor/Annotations/watercolor_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/other'
    },
}
