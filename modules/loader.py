from abc import abstractmethod, ABCMeta

import cv2


class ImageLoader(metaclass=ABCMeta):
    @abstractmethod
    def load_image(self, path):
        pass


class DefaultImageLoader(ImageLoader):
    def load_image(self, path):
        return cv2.imread(path)
