import cv2
import numpy as np
from matplotlib import pyplot as plt


img_color = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_COLOR)
print(f"Forme (couleur) : {img_color.shape}, Type : {img_color.dtype}")

resultat = cv2.convertScaleAbs(img_color, alpha=1, beta=40)

cv2.imshow("Gris", img_color)
cv2.imshow("Egalisee", resultat)

histo_img = cv2.calcHist([img_color],[0],None,[256],[0,256])
histo_equal_img = cv2.calcHist([resultat],[0],None,[256],[0,256])

plt.plot(histo_img, label="Original")
plt.plot(histo_equal_img, label="Contrastée")
plt.xlim([0, 256])
plt.legend()
plt.show()
