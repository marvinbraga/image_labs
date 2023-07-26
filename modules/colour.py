from abc import abstractmethod, ABCMeta

import cv2
import numpy as np
from modules.utils import LEFT_EYE_POINTS, RIGHT_EYE_POINTS, COLOUR_CORRECT_BLUR_FRAC


class ColorCorrector(metaclass=ABCMeta):
    @abstractmethod
    def correct_colours(self, im1, im2, landmarks1):
        pass


class DefaultColorCorrector(ColorCorrector):
    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
        )
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
        return im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
