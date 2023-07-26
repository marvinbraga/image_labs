from abc import abstractmethod, ABCMeta

import cv2
import dlib
import numpy as np

from modules.exceptions import TooManyFaces, NoFaces


class LandmarkFinder(metaclass=ABCMeta):
    @abstractmethod
    def find_landmarks(self, image):
        pass


class DlibLandmarkFinder(LandmarkFinder):
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def find_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return np.matrix([[p.x, p.y] for p in self.predictor(image, rects[0]).parts()])
