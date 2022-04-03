#!/usr/bin/env python3

"""detectors.py: contains face detectors modules."""

__author__ = "Ahmed Hermas"
__copyright__ = "Copyright 2022, © UOL "
__license__ = "MIT"
__version__ = "0.1.0"
__email__ = "a7medhermas@gmail.com"

from cv2 import cv2
import dlib
import numpy as np
import math
import os
import logging
from .settings import MODELS_CONFIG


class Detector ():
    def __init__(self, model_name):

        self.model_name = model_name
        # calling the laoder function of the spicified model name
        self.model = getattr(self, MODELS_CONFIG[self.model_name]['loader'])()

    # Models Loaders
    def _hog_loader(self):
        """ hog and cnn detector model
        Returns:
            None
        """
        model = dlib.get_frontal_face_detector()
        return model

    def _cascade_loader(self):
        """ Cascade detector model

        Returns:
            [class]: detector class object
        """
        cascadePath = os.path.join(os.path.dirname(__file__),
                                   'assets/haarcascade_frontalface_default.xml')
        detector = cv2.CascadeClassifier(cascadePath)

        return detector

    def _retina_loader(self):
        """ retina detector model

        Returns:
            [class]: detector class object
        """
        global RetinaFace
        from retinaface import RetinaFace
        
        model = RetinaFace.build_model()
        return model

    def _mediapipe_loader(self):
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        model = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        return model

    # Models Detectors
    def _getface_hog(self, img):
        """ the detection function for cnn and hog models

        Args:
            img (np array): Input Image
            mode (int): 1 (xywh) , 2 (top,right,bottom,left)

        Returns:
            [tuple]: the output faces boundries
        """
        dets, scores, idx = self.model.run(img, 0, 0.2)
        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.left())
            y1 = int(d.top())
            x2 = int(d.right())
            y2 = int(d.bottom())

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)

    def _getface_cascade(self, img):
        """the detection function for the cascade model

        Args:
            img (np array): Input Image
            mode (int): 1 (xywh) , 2(top,right,bottom,left)

        Returns:
            [tuple]: the output faces boundries
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_data = self.model.detectMultiScale(gray, 1.3, 5)
        faces = []
        for face in faces_data :
            x1 = face[0]
            y1 = face[1]
            x2 = face[0]+face[2]
            y2 = face[1]+face[3]
            faces.append([x1,y1,x2,y2])
        return faces

    def _getface_retina(self, img):
        """ This is the retina face detection model implementation

        Args:
            img (np array): Input Image
            mode (int): 1 (xywh) , 2(top,right,bottom,left)

        Returns:
            [tuple]: the output faces boundries
        """
        faces_data = RetinaFace.detect_faces(img,0.9,self.model)
        faces = []
        if type(faces_data) == tuple:
            return faces
        else :
            # print(faces_data)
            for face in faces_data.values():
                faces.append(face["facial_area"])
        return faces

    def _getface_mediapipe(self, img):
        
        results = self.model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detections = results.detections
        faces =[]
        if not detections :
            return faces
        else :
            for det in detections:
                locs = det.location_data.relative_bounding_box
                height,width,_ = img.shape
                x1 = int(locs.xmin * width)
                y1 = int(locs.ymin * height)
                x2 = int(x1 + (locs.width * width))
                y2 = int(y1 + (locs.height * height))
                faces.append([x1,y1,x2,y2])
        return faces
    def detect(self, image):
        """ 
        detects faces in images and return the boundries
        Args:
            image (numpy array): the input image

        Raises:
            ValueError: [4000]

        Returns:
            [tuple]: the output faces boundries
        """
        try:
            out = getattr(
                self, MODELS_CONFIG[self.model_name]['detector'])(image)

        except UnboundLocalError as e:
            logging.exception("You Have entered a wrong mode number ")

        return out

