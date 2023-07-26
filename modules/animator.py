import cv2
import numpy as np
import skimage
from scipy.spatial import Delaunay
from skimage.transform import AffineTransform, warp


class ImageMorphingAnimator:
    def __init__(self, image_transformer, num_frames=20, show_images=False, verbose=False):
        self._verbose = verbose
        self._show_images = show_images
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
                if self._verbose:
                    # Print the landmarks for the current triangle before calling _morph_triangle
                    print("Triangle Landmarks:", landmarks1[triangle], landmarks2[triangle])
                self._morph_triangle(image1, image2, landmarks1, landmarks2, morphed_image, triangle, t)

            # Append the current morphed_image to morph_frames.
            content = (morphed_image * 255).astype(np.uint8)
            morph_frames.append(content)

            if self._show_images:
                # Save the morphed image to a file for visualization
                image_filename = f'outputs/morph_frame_{t}.png'
                cv2.imwrite(image_filename, content)
                if self._verbose:
                    print(f"{image_filename}: {morphed_image * 255}")

        # Save frames to video.
        height, width, _ = morph_frames[0].shape
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in morph_frames:
            video_writer.write(frame)
        video_writer.release()

    def _morph_triangle(self, image1, image2, landmarks1, landmarks2, morphed_image, triangle, t):
        # Retrieve the triangles from each image and the morphed image.
        tr1 = landmarks1[triangle].reshape(-1, 2)
        tr2 = landmarks2[triangle].reshape(-1, 2)
        tr = tr1 * (1.0 - t) + tr2 * t

        # Compute the affine transformation for this triangle.
        trans1 = AffineTransform()
        trans1.estimate(tr, tr1)
        trans2 = AffineTransform()
        trans2.estimate(tr, tr2)

        rr, cc = skimage.draw.polygon(np.ravel(tr[:, 0]), np.ravel(tr[:, 1]))

        mask = np.zeros_like(image1, dtype=np.float32)
        mask[cc, rr, :] = 1

        warped_triangle1 = warp(image1, trans1.inverse, output_shape=image1.shape)
        warped_triangle2 = warp(image2, trans2.inverse, output_shape=image2.shape)

        mask_y, mask_x, _ = np.where(mask == 1)

        # Then, we use these coordinates to access the respective values in the triangle images
        # warped_image = np.zeros_like(image1, dtype=np.float32)
        # warped_image[mask_y, mask_x] = (1.0 - t) * warped_triangle1[mask_y, mask_x] + t * warped_triangle2[
        #     mask_y, mask_x]
        warped_image = (1.0 - t) * warped_triangle1 + t * warped_triangle2

        # Apply the mask to the morphed image for this triangle.
        # morphed_image[cc, rr, :] = (
        #         morphed_image[cc, rr, :] * (1 - mask[cc, rr, :]) + warped_image[cc, rr, :] * mask[cc, rr, :]
        # )
        # TODO: A transformação não está funcionando corretamente.
        morphed_image[rr, cc, :] = (
            morphed_image[rr, cc, :] * (1 - mask[rr, cc, :]) + warped_image[rr, cc, :] * mask[rr, cc, :]
        )
