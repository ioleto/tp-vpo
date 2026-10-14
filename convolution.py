import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_kernel(img, kernel):
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel)

img_gray = cv2.imread('imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE)

kernel1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
kernel2 = np.ones((3,3), np.float32) / 9
kernel3 = np.array([[ 1, -3,  1], [-3,  9, -3], [ 1, -3,  1]], np.float32)
kernel4 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
kernel5 = np.array([[0, -1, -1], [1,  0, -1], [1,  1,  0]], np.float32)
kernel6 = np.array([[0,  1, 0], [1, -4, 1], [0,  1, 0]], np.float32)

kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
results = [apply_kernel(img_gray, k) for k in kernels]

cv2.imwrite('screen/conv_base.png', img_gray)
cv2.imwrite('screen/conv_k1.png', np.clip(results[0], 0, 255).astype(np.uint8))
cv2.imwrite('screen/conv_k2.png', np.clip(results[1], 0, 255).astype(np.uint8))
cv2.imwrite('screen/conv_k3.png', np.clip(results[2], 0, 255).astype(np.uint8))
cv2.imwrite('screen/conv_k4.png', cv2.normalize(np.abs(results[3]), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imwrite('screen/conv_k5.png', cv2.normalize(np.abs(results[4]), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imwrite('screen/conv_k6.png', cv2.normalize(np.abs(results[5]), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))


plt.subplot(2, 4, 1), plt.imshow(img_gray, cmap='gray'), plt.title('Original')
plt.subplot(2, 4, 2), plt.imshow(results[0], cmap='gray'), plt.title('K1')
plt.subplot(2, 4, 3), plt.imshow(results[1], cmap='gray'), plt.title('K2')
plt.subplot(2, 4, 4), plt.imshow(results[2], cmap='gray'), plt.title('K3')
plt.subplot(2, 4, 5), plt.imshow(np.abs(results[3]), cmap='gray'), plt.title('K4')
plt.subplot(2, 4, 6), plt.imshow(np.abs(results[4]), cmap='gray'), plt.title('K5')
plt.subplot(2, 4, 7), plt.imshow(np.abs(results[5]), cmap='gray'), plt.title('K6')
plt.show()
