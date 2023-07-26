import cv2
import numpy as np
from scipy.spatial import Delaunay
from skimage.draw import polygon
from skimage.transform import AffineTransform, warp


class ImageMorphingAnimator:
    def __init__(self, image_transformer, num_frames=20):
        self.image_transformer = image_transformer
        self.num_frames = num_frames

    def animate(self, image1, image2, landmarks1, landmarks2, output_path):
        # Compute Delaunay triangulation.
        tri = Delaunay(landmarks1)

        # Create array to hold morph frames.
        morph_frames = []

        # Generate intermediate morph frames.
        for t in np.linspace(0, 1, self.num_frames):
            morphed_image = np.zeros_like(image1)
            for triangle in tri.simplices:
                self._morph_triangle(image1, image2, landmarks1, landmarks2, morphed_image, triangle, t)
            morph_frames.append((morphed_image * 255).astype(np.uint8))

        # Save frames to video.
        height, width, _ = morph_frames[0].shape
        # video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

        for frame in morph_frames:
            video_writer.write(frame)
        video_writer.release()

    def _morph_triangle(self, image1, image2, landmarks1, landmarks2, morphed_image, triangle, t):
        # Retrieve the triangles from each image and the morphed image.
        tr1 = landmarks1[triangle]
        tr2 = landmarks2[triangle]
        tr = tr1 * (1.0 - t) + tr2 * t

        # Compute the affine transformation for this triangle.
        trans1 = AffineTransform()
        trans1.estimate(tr, tr1)
        trans2 = AffineTransform()
        trans2.estimate(tr, tr2)

        # Apply the transformation to the region bounded by this triangle in the images.
        im1_warped = warp(image1, trans1.inverse, output_shape=image1.shape)
        im2_warped = warp(image2, trans2.inverse, output_shape=image2.shape)

        # Form the intermediate image for this triangle.
        warped_image = (1.0 - t) * im1_warped + t * im2_warped

        # Create a mask for this triangle.
        mask = np.zeros_like(image1, dtype=bool)
        tr_x, tr_y = np.ravel(tr[:, 0]), np.ravel(tr[:, 1])
        rr, cc = polygon(tr_y, tr_x)  # Troque o tr[:, 1] por tr_y e tr[:, 0] por tr_x
        mask[rr, cc, :] = True

        # Apply the mask to the morphed image for this triangle.
        morphed_image[mask] = warped_image[mask]
        return self
