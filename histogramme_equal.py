import cv2
import numpy as np
from matplotlib import pyplot as plt
# Chargement en niveaux de gris
img_gray = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")

img_equal = cv2.equalizeHist(img_gray)

cv2.imshow("Gris", img_gray)
cv2.imshow("Egalisee", img_equal)

histo_img = cv2.calcHist([img_gray],[0],None,[256],[0,256])
histo_equal_img = cv2.calcHist([img_equal],[0],None,[256],[0,256])

plt.plot(histo_img)
plt.xlim([0,256])
plt.show()
plt.plot(histo_equal_img)
plt.xlim([0,256])
plt.show()

