import cv2
import numpy as np

from utils import transformation_from_points, ALIGN_POINTS, get_face_mask


class ImageMorpher:
    def __init__(self, image_loader, landmark_finder, image_transformer, color_corrector):
        self.image_loader = image_loader
        self.landmark_finder = landmark_finder
        self.image_transformer = image_transformer
        self.color_corrector = color_corrector

    def morph_images(self, image1_path, image2_path, output_path):
        image1 = self.image_loader.load_image(image1_path)
        image2 = self.image_loader.load_image(image2_path)

        landmarks1 = self.landmark_finder.find_landmarks(image1)
        landmarks2 = self.landmark_finder.find_landmarks(image2)

        M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

        mask = get_face_mask(image2, landmarks2)
        warped_mask = self.image_transformer.transform_image(mask, M, image1.shape)
        combined_mask = np.max([get_face_mask(image1, landmarks1), warped_mask], axis=0)

        warped_image2 = self.image_transformer.transform_image(image2, M, image1.shape)
        warped_corrected_image2 = self.color_corrector.correct_colours(image1, warped_image2, landmarks1)

        output_image = image1 * (1.0 - combined_mask) + warped_corrected_image2 * combined_mask
        cv2.imwrite(output_path, output_image)
