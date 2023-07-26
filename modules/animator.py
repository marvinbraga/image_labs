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
        codecs = ["H264", "mp4v", "XVID"]
        codec = codecs[1]
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), 20, (width, height), isColor=True)

        for frame in morph_frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

        # Create a mask for this triangle using OpenCV fillPoly.
        mask = np.zeros_like(image1, dtype=np.uint8)
        pts = np.array(tr, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (1, 1, 1))

        # Apply the mask to the morphed image for this triangle.
        morphed_image = (1.0 - mask) * morphed_image + mask * warped_image
        return self
