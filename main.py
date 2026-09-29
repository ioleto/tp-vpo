import cv2
import numpy as np

# Chargement en niveaux de gris
img_gray = cv2.imread('imagesDeTest/monarch.png', cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")
# Forme typique : (H, W)

# Chargement en couleur (BGR)
img_color = cv2.imread('imagesDeTest/monarch.png', cv2.IMREAD_COLOR)
print(f"Forme (couleur) : {img_color.shape}, Type : {img_color.dtype}")
# Forme typique : (H, W, 3)

# Chargement d'une image avec canal alpha (par exemple PNG avec transparence)
img_rgba = cv2.imread('imagesDeTest/jaguar_rgba.png', cv2.IMREAD_UNCHANGED)

print(f"Forme : {img_rgba.shape}") # (H, W, 4)

print(f"Type : {img_rgba.dtype}")

# Accès à un pixel (100, 150) :
b, g, r, a = img_rgba[100, 150]
print(f"B: {b}, G: {g}, R: {r}, A: {a}")