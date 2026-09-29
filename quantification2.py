import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('screen/peppers-512-original.png', img)

cartes_de_couleurs = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'ocean': cv2.COLORMAP_OCEAN,
    'pink': cv2.COLORMAP_PINK,
}

for nom, carte in cartes_de_couleurs.items():
    image_coloree = cv2.applyColorMap(img, carte)
    cv2.imwrite(f'screen/peppers-512-lut-{nom}.png', image_coloree)
