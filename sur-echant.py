import cv2
import numpy as np
from matplotlib import pyplot as plt

img_gray = cv2.imread('imagesDeTest/peppers-128.png', cv2.IMREAD_GRAYSCALE)

res = cv2.resize(img_gray,None,fx=4, fy=4, interpolation = cv2.INTER_NEAREST)
res2 = cv2.resize(img_gray,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)

print(f"Forme au début : {img_gray.shape}, ensuite : {res.shape}")

cv2.imwrite('screen/sur-echant_1.png', res)
cv2.imwrite('screen/sur-echant_2.png', res2)


cv2.imshow("Boucle", res)
cv2.imshow("Inter_Nearest", res2)
cv2.imshow("Inter_Area", img_gray)