# OpenCV practice

import numpy
import cv2

img = cv2.imread('burger.jpg', 1)

# understanding layers of an image
def modifyRGB(layer, value):
    img[:, :, layer] = value

modifyRGB(0, 1)
modifyRGB(1, 1)
modifyRGB(2, 1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
