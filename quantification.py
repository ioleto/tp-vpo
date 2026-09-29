import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)

niveaux = [64, 16, 4, 2]
images_quantifiees = {}

for nombre_niveaux in niveaux:
    pas = 256 // nombre_niveaux
    images_quantifiees[nombre_niveaux] = (img // pas) * pas

for nombre_niveaux in niveaux:
    image_quantifiee = images_quantifiees[nombre_niveaux]
    histogramme = cv2.calcHist([image_quantifiee], [0], None, [256], [0, 256])

    cv2.imwrite(f'screen/peppers-512-quantifiee-{nombre_niveaux}.png', image_quantifiee)

    plt.figure(figsize=(8, 4))
    plt.plot(histogramme, color='black')
    plt.title(f'Histogramme - {nombre_niveaux} niveaux')
    plt.xlim([0, 256])
    plt.xlabel('Niveau de gris')
    plt.ylabel('Nombre de pixels')
    plt.tight_layout()
    plt.show()
