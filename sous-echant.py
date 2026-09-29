import cv2
import numpy as np
from matplotlib import pyplot as plt

img_gray = cv2.imread('imagesDeTest/peppers-256.png', cv2.IMREAD_GRAYSCALE)

def sous_echantillonner(image, facteur):
    h, w = image.shape
    new_h = h // facteur
    new_w = w // facteur

    nouvelle = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            nouvelle[i, j] = image[i * facteur, j * facteur]

    return nouvelle

img_echantillonage = sous_echantillonner(img_gray, 4)

print(f"Forme au début : {img_gray.shape}, ensuite : {img_echantillonage.shape}")

img_echantillonage2 = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_NEAREST)
img_echantillonage3 = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_AREA)

cv2.imshow("Boucle", img_echantillonage)
cv2.imshow("Inter_Nearest", img_echantillonage2)
cv2.imshow("Inter_Area", img_echantillonage3)