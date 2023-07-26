from abc import abstractmethod, ABCMeta

import cv2
import numpy as np


class ImageTransformer(metaclass=ABCMeta):
    @abstractmethod
    def transform_image(self, image, M, dshape):
        pass


class AffineImageTransformer(ImageTransformer):
    def transform_image(self, image, M, dshape):
        output_image = np.zeros(dshape, dtype=image.dtype)
        cv2.warpAffine(
            image,
            M[:2],
            (dshape[1], dshape[0]),
            dst=output_image,
            borderMode=cv2.BORDER_TRANSPARENT,
            flags=cv2.WARP_INVERSE_MAP
        )
        return output_image
