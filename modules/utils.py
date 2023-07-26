import cv2
import numpy as np

# Pontos de referência para o olho esquerdo, olho direito e boca
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 61))

# Pontos de referência usados para alinhar as imagens.
ALIGN_POINTS = LEFT_EYE_POINTS + RIGHT_EYE_POINTS + MOUTH_POINTS

# Pontos de referência que são usados para sobrepor um rosto em cima do outro.
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + MOUTH_POINTS]

# Quantidade de borrão para usar durante a correção de cores
COLOUR_CORRECT_BLUR_FRAC = 0.6


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def get_face_mask(image, landmarks):
    im = np.zeros(image.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        points = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(im, points, color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    return im


def point_in_triangle(p, triangle):
    # Check if the point `p` is inside the triangle defined by its vertices `triangle`.
    v0, v1, v2 = triangle
    d00 = np.dot(v0 - v1, v0 - v1)
    d01 = np.dot(v0 - v1, v0 - v2)
    d11 = np.dot(v0 - v2, v0 - v2)
    d20 = np.dot(p - v1, v0 - v1)
    d21 = np.dot(p - v1, v0 - v2)
    inv_denom = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d20 - d01 * d21) * inv_denom
    v = (d00 * d21 - d01 * d20) * inv_denom
    return u >= 0 and v >= 0 and u + v <= 1


def barycentric_coordinates(p, triangle):
    # Compute the barycentric coordinates of the point `p` with respect to the triangle `triangle`.
    v0, v1, v2 = triangle
    d00 = np.dot(v0 - v1, v0 - v1)
    d01 = np.dot(v0 - v1, v0 - v2)
    d11 = np.dot(v0 - v2, v0 - v2)
    d20 = np.dot(p - v1, v0 - v1)
    d21 = np.dot(p - v1, v0 - v2)
    inv_denom = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d20 - d01 * d21) * inv_denom
    v = (d00 * d21 - d01 * d20) * inv_denom
    w = 1 - u - v
    return u, v, w
