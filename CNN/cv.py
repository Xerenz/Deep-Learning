# OpenCV practice

import numpy as np

import cv2

img = cv2.imread('burger.jpg', 1)

# understanding layers of an image
def modifyRGB(layer, value):
    img[:, :, layer] = value

''' arithematic operation '''

# addition using cv2
img2 = cv2.imread('maxresdefault.jpg', 1)
img2 = img2[100:600, 100:600] # getting them to same shape
img3 = cv2.add(img, img2)

# addition using numpy
img4 = cv2.imread('1XBYPmr.jpg', 1)
img4 = img4[500:1000, 1000:1500] # reshaping
img5 = img + img4 # results in noisy images

''' blended transition '''

# g(x) = (1 - \alpha)f0(x) + \alpha f1(x)
# alpha varies 0 ---> 1
img6 = cv2.addWeighted(img, 0.2, img4, 0.8, 0)

# smooth transition
for theta in range(10000):
    alpha = abs(np.sin(theta)) # val between 0&1
    img7 = cv2.addWeighted(img, alpha, img4, 1 - alpha, 0)
    cv2.imshow('transition', img7)

'''
cv2.imshow('image', img6)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
