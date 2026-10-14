import cv2

image = cv2.imread('imagesDeTest/peppers-512-RGB.png', cv2.IMREAD_COLOR)

b, g, r = cv2.split(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('screen/peppers-512-RGB-original.png', image)
cv2.imwrite('screen/peppers-512-RGB-canal-bleu.png', b)
cv2.imwrite('screen/peppers-512-RGB-canal-vert.png', g)
cv2.imwrite('screen/peppers-512-RGB-canal-rouge.png', r)
cv2.imwrite('screen/peppers-512-RGB-gris.png', gray)

