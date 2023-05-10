# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# TensorflowUNetALLTrainer.py
# 2023/05/05 to-arai

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import traceback

from ConfigParser import ConfigParser
from BrainTumorDataset import BrainTumorDataset
#from EpochChangeCallback import EpochChangeCallback

from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    config   = ConfigParser(config_file)

    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")
    channels = config.get(MODEL, "image_channels")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # 1 Create train dataset
    resized_image    = (height, width, channels)
    dataset          = BrainTumorDataset(resized_image)

    original_data_path  = config.get(TRAIN, "image_datapath")
    segmented_data_path = config.get(TRAIN, "mask_datapath")
    x_train, y_train = dataset.create(original_data_path, segmented_data_path)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # 2 Create a UNetMolde and compile
    model          = TensorflowUNet(config_file)

    # 3 Start training
    model.train(x_train, y_train)

  except:
    traceback.print_exc()
    
