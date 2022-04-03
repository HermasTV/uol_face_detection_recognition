#!/usr/bin/env python3

"""detectors.py: contains face detectors modules."""

__author__ = "Ahmed Hermas"
__copyright__ = "Copyright 2022, © UOL "
__license__ = "MIT"
__version__ = "0.0.1"
__email__ = "a7medhermas@gmail.com"
import os
import torch
from cv2 import cv2
import utils
from vgg18 import myResNet

class Encoder ():
    def __init__(self, encoder_name):
        self.encoder_name = encoder_name
        # self.model = 
    def _load_vgg(self):
        weights_link = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        weights_path = "recognition/assets/resnet18.pth"
        if not os.path.exists(weights_path):
            print("path not found,Downloading model weights")
            utils.download_url(weights_link,weights_path)
        model = myResNet()
        model.load_state_dict(torch.load(weights_path))


class Distance ():
    def __init__(self, model_name):
        self.model_name = model_name

def circule_loss():
    pass
def arc_face():
    pass