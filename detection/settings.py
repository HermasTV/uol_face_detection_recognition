#!/usr/bin/env python3

"""settings.py: detection module settings file """

__author__ = "Ahmed Hermas"
__copyright__ = "Copyright 2022, © UOL "
__license__ = "MIT"
__version__ = "0.1.0"
__email__ = "a7medhermas@gmail.com"

MODELS_CONFIG = {
    'hog' : 
        {'loader':'_hog_loader',
         'detector':'_getface_hog'},
    'mmod':
        {'loader':'_mmod_loader',
         'detector':'_getface__mmod'},
    'retina':
        {'loader':'_retina_loader',
         'detector':'_getface_retina'},
    'cascade':
        {'loader':'_cascade_loader',
         'detector':'_getface_cascade'},
    'mediapipe':
        {'loader':'_mediapipe_loader',
         'detector':'_getface_mediapipe'}
}