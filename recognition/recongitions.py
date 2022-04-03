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
import numpy as np
from Siamese_resnet18 import myResNet

class Encoder ():
    def __init__(self, encoder_name):
        self.encoder_name = encoder_name
        
    def _load_vgg_encoder(self):
        weights_link = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        weights_path = "recognition/assets/resnet18.pth"
        if not os.path.exists(weights_path):
            print("path not found,Downloading model weights")
            utils.download_url(weights_link,weights_path)
        model = myResNet()
        model.load_state_dict(torch.load(weights_path))
    
    def euclidean_distance(a,b):

        a/= np.sqrt(np.maximum(np.sum(np.square(a)),1e-10))
        b/= np.sqrt(np.maximum(np.sum(np.square(b)),1e-10))

        dist = np.sqrt(np.sum(np.square(a-b)))

        return dist

