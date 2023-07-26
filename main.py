import cv2

from modules.animator import ImageMorphingAnimator
from modules.colour import DefaultColorCorrector
from modules.landmark import DlibLandmarkFinder
from modules.loader import DefaultImageLoader
from modules.transformer import AffineImageTransformer

if __name__ == '__main__':
    # Cria instâncias dos objetos necessários
    image_loader = DefaultImageLoader()
    landmark_finder = DlibLandmarkFinder('res/shape_predictor_68_face_landmarks.dat')
    image_transformer = AffineImageTransformer()
    color_corrector = DefaultColorCorrector()

    # Carrega as duas imagens e encontra os pontos de referência
    image1 = image_loader.load_image('inputs/image1.png')
    image2 = image_loader.load_image('inputs/image2.png')

    # Redimensiona as imagens para ter o mesmo tamanho
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    landmarks1 = landmark_finder.find_landmarks(image1)
    landmarks2 = landmark_finder.find_landmarks(image2)

    # Cria uma instância de ImageMorphingAnimator
    animator = ImageMorphingAnimator(image_transformer, num_frames=60)

    # Usa o ImageMorphingAnimator para transformar a imagem morphed_image em um vídeo
    animator.animate(image1, image2, landmarks1, landmarks2, 'outputs/output.mp4')
